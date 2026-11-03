"""Short video decoding, manual masks, native HQ-SAM2 temporal Track and MP4 export."""
from __future__ import annotations

import math
import hashlib
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from .video_diagnostics import REPORT_VERSION, dump_json, create_run, ram_mib

import cv2
import numpy as np

MAX_SECONDS = 10.0
MAX_TRACK_FPS = 24.0
PROXY_LONG_EDGE = 1280
SUPPORTED_EXTENSIONS = frozenset({'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'})


class VideoFormatError(ValueError):
    pass


class VideoCanceled(Exception):
    pass


@dataclass(frozen=True)
class VideoClip:
    source: str
    frame_paths: tuple[str, ...]
    source_fps: float
    output_fps: float
    source_width: int
    source_height: int
    proxy_width: int
    proxy_height: int
    duration: float
    source_frame_indices: tuple[int, ...]
    temp_dir: str

    @property
    def count(self) -> int:
        return len(self.frame_paths)


def _check_cancel(cancel: Callable[[], bool] | None):
    if cancel and cancel():
        raise VideoCanceled('Đã dừng thao tác video.')


def _reader(path: str) -> cv2.VideoCapture:
    if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoFormatError('Sai định dạng video. Hỗ trợ MP4, MOV, MKV, WEBM, AVI, M4V.')
    if not Path(path).is_file():
        raise VideoFormatError('Không tìm thấy file video.')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoFormatError('Không thể giải mã video. File có thể bị hỏng hoặc codec chưa hỗ trợ.')
    return cap


def extract_clip(path: str, progress=None, cancel=None) -> VideoClip:
    """Decode only a <=10s clip to capped-size JPEG proxies, at <=24 FPS.

    Verify declared *and* observed duration; a missing/wrong container frame
    count must never allow a longer clip.  Cancel/exception deletes temp data.
    """
    cap = _reader(path)
    tmp = tempfile.TemporaryDirectory(prefix='anibgrmv_video_')
    # Detached lifetime: VideoClip owns pathname; caller removes it explicitly.
    # Keep the actual TemporaryDirectory object alive via global registry.
    temp_dir = Path(tmp.name)
    _ACTIVE_TEMPS[str(temp_dir)] = tmp
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or not (0.5 <= fps <= 240) or width < 2 or height < 2:
            raise VideoFormatError('Video thiếu FPS/kích thước hợp lệ hoặc codec không đọc được.')
        if declared_frames > 0 and declared_frames / fps > MAX_SECONDS + 1.0 / fps:
            raise VideoFormatError('Video dài quá 10 giây. Hãy cắt ngắn video rồi upload lại.')
        max_frames = int(math.floor(MAX_SECONDS * fps + 0.0001))
        out_fps = min(fps, MAX_TRACK_FPS)
        scale = min(1.0, PROXY_LONG_EDGE / max(width, height))
        pw, ph = max(2, round(width * scale)), max(2, round(height * scale))
        frames = []
        indices = []
        read_count = 0
        while True:
            _check_cancel(cancel)
            ok, bgr = cap.read()
            if not ok:
                break
            if read_count >= max_frames:
                raise VideoFormatError('Video dài quá 10 giây. Giới hạn tính theo frame thực tế.')
            # Deterministic resampling by source frame timestamp.
            sample_id = int(math.floor(read_count * out_fps / fps + 1e-7))
            if sample_id >= len(frames):
                if (pw, ph) != (width, height):
                    bgr = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
                frame_path = temp_dir / f'{len(frames):05d}.jpg'
                if not cv2.imwrite(str(frame_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise OSError('Không thể lưu frame video tạm.')
                frames.append(str(frame_path))
                indices.append(read_count)
                if progress:
                    progress(min(99, int(100 * (read_count + 1) / max(1, declared_frames or max_frames))), 'Đang đọc video')
            read_count += 1
        if not frames or read_count == 0:
            raise VideoFormatError('Video không chứa frame hợp lệ.')
        # Some broken/variable-frame-rate files have inconsistent metadata.
        reported_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if reported_time > MAX_SECONDS + 0.1:
            raise VideoFormatError('Video dài quá 10 giây theo thời gian phát thực tế.')
        duration = read_count / fps
        if duration > MAX_SECONDS + 1e-5:
            raise VideoFormatError('Video dài quá 10 giây.')
        if progress:
            progress(100, 'Đã đọc video')
        return VideoClip(str(path), tuple(frames), fps, out_fps, width, height, pw, ph,
                         duration, tuple(indices), str(temp_dir))
    except Exception:
        release_clip_temp(str(temp_dir))
        raise
    finally:
        cap.release()


_ACTIVE_TEMPS: dict[str, tempfile.TemporaryDirectory] = {}


def release_clip_temp(temp_dir: str):
    tmp = _ACTIVE_TEMPS.pop(str(temp_dir), None)
    if tmp is not None:
        tmp.cleanup()


def blank_alpha(clip: VideoClip) -> np.ndarray:
    return np.zeros((clip.proxy_height, clip.proxy_width), np.uint8)


def paint_alpha(alpha: np.ndarray, points, brush_size: int, operation: str, box=None) -> np.ndarray:
    """Manual Paint/Box matches Edit semantics. Returns a new uint8 mask."""
    result = np.asarray(alpha, dtype=np.uint8).copy()
    color = 255 if operation == 'restore' else 0
    h, w = result.shape
    if box is not None:
        x0, y0, x1, y1 = map(int, box)
        x0, x1 = sorted((max(0, min(w-1, x0)), max(0, min(w-1, x1))))
        y0, y1 = sorted((max(0, min(h-1, y0)), max(0, min(h-1, y1))))
        if x1 > x0 and y1 > y0:
            result[y0:y1+1, x0:x1+1] = color
    elif points:
        pts = [(max(0, min(w-1, round(x))), max(0, min(h-1, round(y)))) for x, y in points]
        thickness = max(1, int(brush_size))
        cv2.circle(result, pts[0], max(1, thickness//2), color, -1)
        for p, q in zip(pts, pts[1:]):
            cv2.line(result, p, q, color, thickness, lineType=cv2.LINE_8)
    return result


def track_masks(clip: VideoClip, keyframes: dict[int, np.ndarray],
                progress=None, cancel=None, report_dir=None) -> tuple[np.ndarray, ...]:
    """Track with REAL HQ-SAM2 temporal memory; no optical-flow fallback.

    The user mask is a conditioning prompt via add_new_mask. The GPU child uses
    the HQ-SAM2 Large base + fine-tuned HQ-SAM decoder. This adapter raises an
    explicit error if PyTorch/native checkpoints are not available.
    """
    if not keyframes:
        raise VideoFormatError('Chưa bôi mask. Hãy Restore hoặc Erase trên ít nhất một frame trước khi Track.')
    if not any(np.any(mask > 0) for mask in keyframes.values()):
        raise VideoFormatError('Mask đang trống. Hãy chọn vùng cần giữ bằng Restore trước khi Track.')
    _check_cancel(cancel)
    from .video_track import track_video
    extra = {"report_dir": report_dir} if report_dir is not None else {}
    return track_video(clip, keyframes, progress=progress, cancel=cancel, **extra)


def stabilize_identical_frames(clip: VideoClip, masks, keyframes):
    """Exact RGB duplicate => exact mask, except conflicting user keyframes.

    Only identical decoded JPEG proxy bytes are grouped. Does not smooth
    similar-looking frames, and never overwrites an explicit keyframe.
    """
    if len(masks) != clip.count:
        raise ValueError('Frame count mismatch when stabilizing mask.')
    groups = {}
    for fi, path in enumerate(clip.frame_paths):
        digest = hashlib.sha256(Path(path).read_bytes()).digest()
        groups.setdefault(digest, []).append(fi)
    result = [np.asarray(m, np.uint8).copy() for m in masks]
    reused = conflicts = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        annotated = [idx for idx in group if idx in keyframes]
        if len(annotated)>1 and any(not np.array_equal(keyframes[annotated[0]], keyframes[idx])
                                     for idx in annotated[1:]):
            conflicts += 1
            continue
        canonical = annotated[0] if annotated else group[0]
        base = np.asarray(keyframes[canonical], np.uint8) if annotated else result[canonical]
        for idx in group:
            if idx not in keyframes and idx!=canonical:
                result[idx] = base.copy()
                reused += 1
    for idx,mask in keyframes.items():
        result[idx] = np.asarray(mask,np.uint8).copy()
    return tuple(result), reused, conflicts


# UI backgrounds encoded in RGB.  Green is the default for chroma-key MP4.
EXPORT_BACKGROUNDS = {'green': (0, 255, 0), 'black': (0, 0, 0),
                      'white': (255, 255, 255), 'blue': (0, 0, 255)}


def export_video(clip: VideoClip, masks: tuple[np.ndarray, ...], dest: str,
                 background='green', progress=None, cancel=None,
                 matting='binary', report_dir=None) -> str:
    """Optional ViTMatte-SAM 16-bit soft alpha on CUDA, then MP4 chroma export.

    ViTMatte-SAM runs in a child process *after* SAM2 tracking has exited.
    Binary mode stays usable for the CPU edition and codec self-tests. MP4 has
    no alpha channel: export is chroma-key, without audio.
    """
    if len(masks) != clip.count:
        raise VideoFormatError('Cần Track lại sau khi thay đổi mask trước khi Export.')
    matting = str(matting).lower()
    if matting not in ('vitmatte_sam', 'binary'):
        raise VideoFormatError('video.matting_export phải là vitmatte_sam hoặc binary.')
    diag = Path(report_dir) if report_dir else create_run(Path(dest).parent, 'export')
    diag.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dest_path = Path(dest)
    report = {'version': REPORT_VERSION, 'status': 'RUNNING', 'action': 'export',
              'video': str(Path(clip.source).name), 'frames': clip.count,
              'matting': matting, 'output_fps': clip.output_fps,
              'source_resolution': [clip.source_width, clip.source_height],
              'stages_sec': {}, 'parent_rss_start_mib': ram_mib(),
              'note': 'MP4 chroma key, no alpha channel, no audio'}
    dump_json(diag/'export_report.json', report)
    temp_path = dest_path.with_name(dest_path.stem + '.partial.mp4')
    try:
        _check_cancel(cancel)
        alpha_files = None
        if matting == 'vitmatte_sam':
            # No silent fallback: fail loudly if CUDA/ViTMatte-SAM is unavailable.
            from .video_matting import run_vitmatte_sam
            t = time.perf_counter()
            def matting_progress(value, message):
                if progress:
                    progress(min(78, round(value*.78)), message)
            alpha_files = run_vitmatte_sam(clip, masks, diag, progress=matting_progress, cancel=cancel)
            report['stages_sec']['matting_alpha'] = round(time.perf_counter()-t,3)
        else:
            report['stages_sec']['matting_alpha'] = 0

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cap = _reader(clip.source)
        writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 clip.output_fps, (clip.source_width, clip.source_height))
        if not writer.isOpened():
            cap.release();writer.release()
            raise RuntimeError('Không mở được MP4 encoder OpenCV mp4v.')
        bg_rgb = EXPORT_BACKGROUNDS.get(background, EXPORT_BACKGROUNDS['green'])
        bg_bgr = np.asarray(bg_rgb[::-1], dtype=np.float32)
        source_idx = -1
        compose_t0 = time.perf_counter()
        try:
            for i, desired_index in enumerate(clip.source_frame_indices):
                _check_cancel(cancel)
                while source_idx < desired_index:
                    ok, frame = cap.read()
                    if not ok:
                        raise VideoFormatError(f'Không đọc được frame {desired_index} khi export.')
                    source_idx += 1
                if alpha_files is not None:
                    a16 = cv2.imread(str(alpha_files[i]), cv2.IMREAD_UNCHANGED)
                    if a16 is None or a16.dtype != np.uint16:
                        raise RuntimeError(f'Alpha ViTMatte-SAM frame {i} không phải PNG 16-bit.')
                    alpha = a16.astype(np.float32) / 65535.0
                else:
                    alpha = masks[i].astype(np.float32) / 255.0
                if alpha.shape != (clip.source_height, clip.source_width):
                    alpha = cv2.resize(alpha, (clip.source_width, clip.source_height),
                                       interpolation=cv2.INTER_LINEAR)
                inv_a = 1.0 - alpha
                composed = np.empty_like(frame)
                for c in range(3):
                    composed[...,c] = np.clip(frame[...,c]*alpha + bg_bgr[c]*inv_a,
                                               0,255).astype(np.uint8)
                writer.write(composed)
                if progress:
                    progress((78 if alpha_files else 0)+round((22 if alpha_files else 100)*(i+1)/clip.count),
                             f'Đang xuất MP4 {i+1}/{clip.count}')
        finally:
            writer.release()
            cap.release()
        _check_cancel(cancel)
        report['stages_sec']['compose_mp4'] = round(time.perf_counter()-compose_t0,3)
        temp_path.replace(dest_path)
        report.update(status='OK', total_sec=round(time.perf_counter()-started,3),
                      output=str(dest_path), output_bytes=dest_path.stat().st_size,
                      parent_rss_finish_mib=ram_mib())
        dump_json(diag/'export_report.json', report)
        return str(dest_path)
    except BaseException as exc:
        temp_path.unlink(missing_ok=True)
        report.update(status='CANCELLED' if isinstance(exc,VideoCanceled) else 'ERROR',
                      error=f'{type(exc).__name__}: {exc}',
                      total_sec=round(time.perf_counter()-started,3))
        dump_json(diag/'export_report.json', report)
        raise
