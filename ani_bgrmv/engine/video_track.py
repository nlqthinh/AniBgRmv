"""Parent-process adapter for native HQ-SAM2 temporal video propagation.

The subprocess owns PyTorch memory. Cancelling terminates the child, then its
temporary masks are removed. Existing image ONNX sessions and GUI stay intact.

A Track covers the whole clip, or one span between two keyframes ("Track
tiến ▶", "◀ Track lùi"): the worker gets only that span's frames, in tracking
order.
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

PREFIX = '__ANIBGRMV_VIDEO_TRACK__'


class TrackResult(NamedTuple):
    frames: tuple[int, ...]          # clip frames tracked, in time order
    masks: tuple[np.ndarray, ...]    # label map of each of those frames (read only)
    held: dict[int, vh.Hold]         # frame -> the earlier frame it holds (video_holds)
    agreement: dict | None           # vs the previous Track of the same keyframes


def track_video(clip, keyframes, progress=None, cancel=None, report_dir=None, previous=None,
                frames=None):
    """HQ-SAM2 Track in its own process, then held frames reuse their source's mask.

    ``frames``: clip frames to track, in tracking order (a span starting at a
    keyframe; backwards = descending); default the whole clip. Only the
    keyframes inside it are prompts. ``previous``: label maps of an earlier
    Track of the same keyframes over the same frames; the report then says how
    far the new masks are from it (vo.track_agreement), e.g. after changing
    video.track_keyframes_in_memory.
    """
    from ..app_config import APP_CONFIG
    from ..model_registry import model_path
    _check_cancel(cancel)
    if APP_CONFIG['model_settings']['video_backend'] != 'hqsam2':
        raise VideoFormatError('Video backend chưa được hỗ trợ bởi bản app này.')
    device = selected_video_device()
    frames = tuple(range(clip.count)) if frames is None else tuple(map(int, frames))
    if not frames or len(set(frames)) != len(frames) or not all(0 <= i < clip.count for i in frames):
        raise VideoFormatError('Đoạn frame cần Track không hợp lệ.')
    position = {frame: pos for pos, frame in enumerate(frames)}
    keyframes = {idx: mask for idx, mask in keyframes.items() if idx in position}
    started = time.perf_counter()
    diag = Path(report_dir) if report_dir else create_run(Path(clip.source).parent, 'track')
    diag.mkdir(parents=True, exist_ok=True)
    report = {'version': REPORT_VERSION, 'status': 'RUNNING', 'action': 'track',
              'clip': {'name': Path(clip.source).name, 'frames': clip.count,
                       'proxy_hw': [clip.proxy_height, clip.proxy_width],
                       'output_fps': clip.output_fps},
              'tracked_frames': {'first': frames[0], 'last': frames[-1], 'count': len(frames)},
              'keyframes': sorted(keyframes), 'device': device, 'samples': [], 'errors': []}
    dump_json(diag/'track_report.json', report)
    if not keyframes or not any(np.any(m > 0) for m in keyframes.values()):
        raise VideoFormatError('Chưa bôi mask hợp lệ. Hãy Restore trước khi Track.')
    if not all(mask.shape == (clip.proxy_height, clip.proxy_width) for mask in keyframes.values()):
        raise VideoFormatError('Keyframe mask không khớp video.')
    base, decoder = model_path('hqsam_base_checkpoint'), model_path('hqsam_decoder_checkpoint')
    missing = [str(p) for p in (base, decoder) if not p.is_file()]
    if missing:
        raise VideoFormatError(
            'SAM2 Video Predictor cần checkpoint PyTorch đầy đủ: '\
            + ', '.join(missing) + '. Hai graph HQ-SAM ONNX ảnh tĩnh không thay thế được bộ nhớ video.'
        )
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
        job = dict(base_checkpoint=str(base.resolve()),
                   decoder_checkpoint=str(decoder.resolve()),
                   frame_paths=[str(Path(clip.frame_paths[i]).resolve()) for i in frames],
                   hw=[clip.proxy_height, clip.proxy_width],
                   keyframe_masks=paths,
                   output_dir=str(out_dir), worker_report=str(diag / 'track_worker_report.json'),
                   sample_every=8, device=device,
                   precision=APP_CONFIG['video']['track_precision'],
                   keyframes_in_memory=APP_CONFIG['video']['track_keyframes_in_memory'])
        job_file = staging / 'job.json'
        job_file.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        try:
            report['worker_memory_samples'] = run_worker(
                'video_track_worker.py', job_file, PREFIX, label='HQ-SAM2 Video Predictor',
                progress=progress, cancel=cancel, samples=report['samples'],
                sample_gpu=device != 'cpu')
            order = tuple(sorted(frames))  # time order
            result = []
            for i in order:
                mask = cv2.imread(str(out_dir / f'{position[i]:05d}.png'), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (clip.proxy_height, clip.proxy_width):
                    raise RuntimeError('Thiếu mask frame ' + str(i))
                # The user's keyframe, not SAM2's copy of it.
                result.append(np.asarray(keyframes[i], np.uint8) if i in keyframes else mask)
            held = {}
            if APP_CONFIG['video']['reuse_held_frames']:
                # A span starts and ends at a keyframe (or the clip's end), and
                # keyframes stand alone: no hold crosses its bounds.
                holds = vh.held_frames([clip.frame_paths[i] for i in order], result,
                                       keep=frozenset(pos for pos, i in enumerate(order) if i in keyframes))
                for pos, hold in enumerate(holds):
                    if hold.source != pos:
                        result[pos] = vh.shifted(result[hold.source], hold.dx, hold.dy)
                        held[order[pos]] = hold._replace(source=order[hold.source])
            report['held_frames'] = {str(i): list(hold) for i, hold in held.items()}
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
            return TrackResult(order, tuple(result), held, agreement)
        except BaseException as exc:
            report.update(status='CANCELLED' if isinstance(exc, VideoCanceled) else 'ERROR',
                          error=f'{type(exc).__name__}: {exc}',
                          total_sec=round(time.perf_counter() - started, 3))
            dump_json(diag / 'track_report.json', report)
            raise
