"""Parent-process adapter for native HQ-SAM2 temporal video propagation.

The subprocess owns PyTorch memory. Cancelling terminates the child, then its
temporary masks are removed. Existing image ONNX sessions and GUI stay intact.

A Track covers the whole clip, or one span between two keyframes ("Track
tiến ▶", "◀ Track lùi"): the worker gets only that span's frames, in tracking
order.

The app tracks with HQ3 (app_track_model) at video.track_image_size. With
video.reuse_held_frames, frames whose whole picture repeats the frame before
(anime holds a drawing 2-3 frames) are not sent to SAM2: they take that frame's
mask (video_holds.picture_holds). A Track keeps its keyframes and final masks
in its diagnostics folder: the video page imports them back (load_track_log)
instead of tracking again;
tools/compare_track_models.py replays it with other SAM2 models (run_tracker),
tools/clean_plate_preview.py reads its masks.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from . import video_objects as vo
from . import video_holds as vh
from .video_pipeline import VideoCanceled, VideoFormatError, _check_cancel
from .video_diagnostics import REPORT_VERSION, dump_json, dump_csv, create_run
from .video_sidecar import run_worker, selected_video_device
from .video_track_worker import HQ_CONFIG, PROGRESS_PREFIX as PREFIX

# Label maps a Track keeps in its diagnostics folder, <clip frame:05d>.png:
# the user's keyframes (to replay the Track) and the final masks (Export's input).
KEYFRAME_DIR = 'keyframes'
MASK_DIR = 'masks'


class TrackModel(NamedTuple):
    """SAM2 model of a Track, as video_track_worker.load_predictor builds it."""
    base: Path
    decoder: Path | None             # fine-tuned HQ decoder; None = released checkpoint as is
    config: str = HQ_CONFIG
    image_size: int | None = None    # None = the config's 1024


class TrackResult(NamedTuple):
    frames: tuple[int, ...]          # clip frames tracked, in time order
    masks: tuple[np.ndarray, ...]    # label map of each of those frames (read only)
    held: dict[int, vh.Hold]         # frame -> the earlier frame it holds (video_holds)
    agreement: dict | None           # vs the previous Track of the same keyframes
    sam_frames: int                  # frames SAM2 tracked; the others repeat an earlier picture


def save_label_maps(folder: Path, maps: dict) -> None:
    """clip frame -> label map, as <frame:05d>.png."""
    folder.mkdir(parents=True, exist_ok=True)
    for frame, labels in maps.items():
        if not cv2.imwrite(str(folder / f'{frame:05d}.png'), np.asarray(labels, np.uint8)):
            raise OSError(f'Không lưu được mask frame {frame} vào {folder}')


def read_label_maps(folder: Path) -> dict:
    """<frame:05d>.png files -> {clip frame: label map}; empty when there is no folder."""
    maps = {}
    for path in sorted(folder.glob('*.png')) if folder.is_dir() else ():
        labels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if labels is None:
            raise VideoFormatError('File mask hỏng: ' + str(path))
        maps[int(path.stem)] = labels
    return maps


class TrackLog(NamedTuple):
    """An earlier Track of a clip, read back from its diagnostics folder."""
    keyframes: dict            # clip frame -> label map: what that Track was given
    masks: tuple | None        # every frame's final mask; None unless a finished whole-clip Track
    held: dict                 # frame -> vh.Hold, with the masks


def load_track_log(run: Path, clip) -> TrackLog:
    """The keyframes, and the masks of a finished whole-clip Track, that a
    Track of this same clip (same video, same part, same proxy) kept in ``run``."""
    report_path = Path(run) / 'track_report.json'
    if not report_path.is_file():
        raise VideoFormatError('Thư mục này không phải log Track (thiếu track_report.json).')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    logged = report['clip']
    if 'source_frames' not in logged:
        raise VideoFormatError('Log Track này của bản app cũ, chưa lưu keyframe.')
    this = {'name': Path(clip.source).name, 'frames': clip.count,
            'source_frames': [clip.source_frame_indices[0], clip.source_frame_indices[-1]],
            'proxy_hw': [clip.proxy_height, clip.proxy_width]}
    if any(logged[key] != value for key, value in this.items()):
        raise VideoFormatError(
            f'Log Track này là của {logged["name"]}, frame nguồn {logged["source_frames"]}; '
            f'video đang mở là {this["name"]}, frame nguồn {this["source_frames"]}. '
            'Hãy mở đúng video và cắt đúng đoạn đó.')
    shape = (clip.proxy_height, clip.proxy_width)
    keyframes = read_label_maps(Path(run) / KEYFRAME_DIR)
    if not keyframes or sorted(keyframes) != report['keyframes'] \
            or any(labels.shape != shape for labels in keyframes.values()):
        raise VideoFormatError('Log Track này thiếu keyframe.')
    masks, held = None, {}
    if report['status'] == 'OK' and report['tracked_frames']['count'] == clip.count:
        saved = read_label_maps(Path(run) / MASK_DIR)  # empty in logs of App_01
        if sorted(saved) == list(range(clip.count)) and all(m.shape == shape for m in saved.values()):
            masks = tuple(saved[frame] for frame in range(clip.count))
            held = {int(frame): vh.Hold(*hold) for frame, hold in report['held_frames'].items()}
    return TrackLog(keyframes, masks, held)


def app_track_model() -> TrackModel:
    """HQ3: HQ-SAM2 Large base + the notebook's fine-tuned decoder."""
    from ..model_registry import model_path
    return TrackModel(model_path('hqsam_base_checkpoint'), model_path('hqsam_decoder_checkpoint'))


def run_tracker(model: TrackModel, clip, frames, keyframes, diag, *, device, precision,
                keyframes_in_memory, progress=None, cancel=None, samples=None,
                label='HQ-SAM2 Video Predictor'):
    """SAM2 label maps of ``frames``, in time order, as the worker returns them.

    ``frames``: clip frames in tracking order; ``keyframes``: clip frame ->
    label map, every one inside ``frames``. The worker writes
    track_worker_report.json and track_frames.csv to ``diag``. Returns
    (masks, the worker's memory samples).
    """
    position = {frame: pos for pos, frame in enumerate(frames)}
    hw = (clip.proxy_height, clip.proxy_width)
    with tempfile.TemporaryDirectory(prefix='anibgrmv_video_track_') as tmp:
        staging = Path(tmp)
        in_dir = staging / 'keyframes'
        out_dir = staging / 'masks'
        in_dir.mkdir()
        out_dir.mkdir()
        paths = {}
        for idx, mask in keyframes.items():  # the worker counts frames by position
            destination = in_dir / f'{position[idx]:05d}.png'
            if not cv2.imwrite(str(destination), np.asarray(mask, np.uint8)):
                raise OSError('Không lưu được keyframe mask ' + str(idx))
            paths[str(position[idx])] = str(destination)
        job = dict(base_checkpoint=str(model.base.resolve()),
                   decoder_checkpoint=str(model.decoder.resolve()) if model.decoder else None,
                   model_config=model.config, image_size=model.image_size,
                   frame_paths=[str(Path(clip.frame_paths[i]).resolve()) for i in frames],
                   hw=list(hw), keyframe_masks=paths,
                   output_dir=str(out_dir), worker_report=str(Path(diag) / 'track_worker_report.json'),
                   sample_every=8, device=device, precision=precision,
                   keyframes_in_memory=keyframes_in_memory)
        job_file = staging / 'job.json'
        job_file.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        memory = run_worker('video_track_worker.py', job_file, PREFIX, label=label,
                            progress=progress, cancel=cancel, samples=samples,
                            sample_gpu=device != 'cpu')
        masks = []
        for i in sorted(frames):
            mask = cv2.imread(str(out_dir / f'{position[i]:05d}.png'), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != hw:
                raise RuntimeError('Thiếu mask frame ' + str(i))
            masks.append(mask)
        return tuple(masks), memory


def track_video(clip, keyframes, progress=None, cancel=None, report_dir=None, previous=None,
                frames=None):
    """HQ-SAM2 Track in its own process; held frames reuse their source's mask.

    Frames repeating the whole picture before them are not sent to SAM2 (see
    the module notes); the others are checked around the object after Track.

    ``frames``: clip frames to track, in tracking order (a span starting at a
    keyframe; backwards = descending); default the whole clip. Only the
    keyframes inside it are prompts. ``previous``: label maps of an earlier
    Track of the same keyframes over the same frames; the report then says how
    far the new masks are from it (vo.track_agreement), e.g. after changing
    video.track_keyframes_in_memory.
    """
    from ..app_config import APP_CONFIG
    _check_cancel(cancel)
    if APP_CONFIG['model_settings']['video_backend'] != 'hqsam2':
        raise VideoFormatError('Video backend chưa được hỗ trợ bởi bản app này.')
    device = selected_video_device()
    frames = tuple(range(clip.count)) if frames is None else tuple(map(int, frames))
    if not frames or len(set(frames)) != len(frames) or not all(0 <= i < clip.count for i in frames):
        raise VideoFormatError('Đoạn frame cần Track không hợp lệ.')
    keyframes = {idx: mask for idx, mask in keyframes.items() if idx in frames}
    started = time.perf_counter()
    diag = Path(report_dir) if report_dir else create_run(Path(clip.source).parent, 'track')
    diag.mkdir(parents=True, exist_ok=True)
    # source_frames: the part of the video the clip was cut from (replay tools re-extract it).
    report = {'version': REPORT_VERSION, 'status': 'RUNNING', 'action': 'track',
              'clip': {'name': Path(clip.source).name, 'frames': clip.count,
                       'source_frames': [clip.source_frame_indices[0], clip.source_frame_indices[-1]],
                       'proxy_hw': [clip.proxy_height, clip.proxy_width],
                       'output_fps': clip.output_fps},
              'tracked_frames': {'first': frames[0], 'last': frames[-1], 'count': len(frames)},
              'keyframes': sorted(keyframes), 'device': device, 'samples': [], 'errors': []}
    dump_json(diag/'track_report.json', report)
    if not keyframes or not any(np.any(m > 0) for m in keyframes.values()):
        raise VideoFormatError('Chưa bôi mask hợp lệ. Hãy Restore trước khi Track.')
    if not all(mask.shape == (clip.proxy_height, clip.proxy_width) for mask in keyframes.values()):
        raise VideoFormatError('Keyframe mask không khớp video.')
    model = app_track_model()._replace(image_size=APP_CONFIG['video']['track_image_size'])
    report['track_image_size'] = model.image_size
    missing = [str(p) for p in (model.base, model.decoder) if not p.is_file()]
    if missing:
        raise VideoFormatError(
            'SAM2 Video Predictor cần checkpoint PyTorch đầy đủ: '\
            + ', '.join(missing) + '. Hai graph HQ-SAM ONNX ảnh tĩnh không thay thế được bộ nhớ video.'
        )
    save_label_maps(diag / KEYFRAME_DIR, keyframes)
    try:
        order = tuple(sorted(frames))  # time order
        paths = [clip.frame_paths[i] for i in order]
        # A span starts and ends at a keyframe (or the clip's end), and keyframes
        # stand alone: no hold crosses its bounds.
        keep = frozenset(pos for pos, i in enumerate(order) if i in keyframes)
        reuse = APP_CONFIG['video']['reuse_held_frames']
        repeats = {order[pos]: hold._replace(source=order[hold.source])
                   for pos, hold in enumerate(vh.picture_holds(paths, keep) if reuse else ())
                   if hold.source != pos}
        sam_frames = tuple(i for i in frames if i not in repeats)  # tracking order kept
        report['sam_frames'] = len(sam_frames)
        masks, report['worker_memory_samples'] = run_tracker(
            model, clip, sam_frames, keyframes, diag, device=device,
            precision=APP_CONFIG['video']['track_precision'],
            keyframes_in_memory=APP_CONFIG['video']['track_keyframes_in_memory'],
            progress=progress, cancel=cancel, samples=report['samples'])
        # On keyframes: the user's mask, not SAM2's copy of it.
        by_frame = {i: np.asarray(keyframes[i], np.uint8) if i in keyframes else mask
                    for i, mask in zip(sorted(sam_frames), masks)}
        for frame, hold in repeats.items():  # a source is never a repeat itself
            by_frame[frame] = vh.shifted(by_frame[hold.source], hold.dx, hold.dy)
        result = [by_frame[i] for i in order]
        held = {}
        if reuse:
            holds = vh.held_frames(paths, result, keep=keep)
            for pos, hold in enumerate(holds):
                if hold.source != pos:
                    result[pos] = vh.shifted(result[hold.source], hold.dx, hold.dy)
                    held[order[pos]] = hold._replace(source=order[hold.source])
        report['held_frames'] = {str(i): list(hold) for i, hold in held.items()}
        save_label_maps(diag / MASK_DIR, dict(zip(order, result)))
        agreement = None if previous is None else vo.track_agreement(previous, result)
        report['agreement_with_previous_track'] = agreement
        worker_report_path = diag / 'track_worker_report.json'
        if worker_report_path.is_file():
            child = json.loads(worker_report_path.read_text(encoding='utf-8'))
            report['stages_sec'] = child.get('stages_sec', {})
            report['gpu_device'] = child.get('gpu')
            report['compute_device'] = child.get('device', device)
            report['worker_total_sec'] = child.get('total_sec')
            report['track_frames_perf'] = child.get('track_frames_perf', {})
        report['status'] = 'OK'
        report['total_sec'] = round(time.perf_counter() - started, 3)
        report['avg_total_sec_per_frame'] = round(report['total_sec'] / len(frames), 3)
        dump_json(diag / 'track_report.json', report)
        dump_csv(diag / 'process_samples.csv', report['samples'])
        return TrackResult(order, tuple(result), held, agreement, len(sam_frames))
    except BaseException as exc:
        report.update(status='CANCELLED' if isinstance(exc, VideoCanceled) else 'ERROR',
                      error=f'{type(exc).__name__}: {exc}',
                      total_sec=round(time.perf_counter() - started, 3))
        dump_json(diag / 'track_report.json', report)
        raise
