"""Short video decoding, manual masks, native HQ-SAM2 temporal Track and MP4 export."""
from __future__ import annotations

import math
import os
import shutil
import time
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from .foreground import estimate_foreground
from .video_diagnostics import REPORT_VERSION, dump_json, create_run, ram_mib

import cv2
import numpy as np

MAX_SECONDS = 10.0
MAX_TRACK_FPS = 24.0
PROXY_LONG_EDGE = 1280
COMPOSE_THREADS = 4  # Export frames composited at once (foreground colours: ~70% of the work)
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


@dataclass(frozen=True)
class VideoInfo:
    """What the container declares (frame_count may be 0 when it does not know)."""
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    @property
    def too_long(self) -> bool:
        """Longer than a clip may be (one frame of slack for rounding)."""
        return self.duration > MAX_SECONDS + 1.0 / self.fps

    @property
    def max_clip_frames(self) -> int:
        """Most source frames a clip may keep (MAX_SECONDS)."""
        return int(math.floor(MAX_SECONDS * self.fps + 0.0001))


def _stream_info(cap: cv2.VideoCapture, path: str) -> VideoInfo:
    info = VideoInfo(str(path), float(cap.get(cv2.CAP_PROP_FPS)),
                     max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if (not math.isfinite(info.fps) or not (0.5 <= info.fps <= 240)
            or info.width < 2 or info.height < 2):
        raise VideoFormatError('Video thiếu FPS/kích thước hợp lệ hoặc codec không đọc được.')
    return info


def probe_video(path: str) -> VideoInfo:
    """Format, FPS, size and declared length, without decoding frames."""
    cap = _reader(path)
    try:
        return _stream_info(cap, path)
    finally:
        cap.release()


class FramePreview:
    """Frames of a long video for choosing the part to keep. Seeking may land
    a frame or two off on some files: fine for a preview, never used for the
    clip itself (extract_clip decodes in order)."""

    def __init__(self, path: str):
        self._cap = _reader(path)

    def frame(self, index: int) -> np.ndarray | None:
        """RGB picture at source frame ``index`` (None if it cannot be decoded)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, bgr = self._cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def close(self):
        self._cap.release()


def _skip_frames(cap: cv2.VideoCapture, count: int):
    """Decode past ``count`` frames without converting them (grab, not read)."""
    for _ in range(count):
        if not cap.grab():
            raise VideoFormatError('Video kết thúc trước frame cần đọc.')


def extract_clip(path: str, progress=None, cancel=None, start: int = 0, end: int | None = None) -> VideoClip:
    """Decode a <=10s clip to capped-size JPEG proxies, at <=24 FPS.

    Whole video (``end`` None): it must be <=10 s; verify declared *and*
    observed duration, a missing/wrong container frame count must never allow
    a longer clip. A part of a longer video: source frames [start, end), at
    most MAX_SECONDS long; the frames before it are decoded in order, not
    seeked, so the clip's source_frame_indices always match the video.
    Cancel/exception deletes temp data.
    """
    cap = _reader(path)
    tmp = tempfile.TemporaryDirectory(prefix='anibgrmv_video_')
    # Detached lifetime: VideoClip owns pathname; caller removes it explicitly.
    # Keep the actual TemporaryDirectory object alive via global registry.
    temp_dir = Path(tmp.name)
    _ACTIVE_TEMPS[str(temp_dir)] = tmp
    try:
        info = _stream_info(cap, path)
        fps, width, height = info.fps, info.width, info.height
        whole = end is None
        if whole and info.too_long:
            raise VideoFormatError('Video dài quá 10 giây. Hãy cắt ngắn video rồi upload lại.')
        end = start + info.max_clip_frames if whole else int(end)
        if not 0 <= start < end or end - start > info.max_clip_frames:
            raise VideoFormatError('Đoạn video cần giữ phải dài từ 1 frame tới 10 giây.')
        out_fps = min(fps, MAX_TRACK_FPS)
        scale = min(1.0, PROXY_LONG_EDGE / max(width, height))
        pw, ph = max(2, round(width * scale)), max(2, round(height * scale))
        frames = []
        indices = []
        total = info.frame_count if whole and info.frame_count else end  # for progress
        source_index = start
        second = max(1, round(fps))
        for skipped in range(0, start, second):  # ~1 s at a time, so Dừng stays responsive
            _check_cancel(cancel)
            _skip_frames(cap, min(second, start - skipped))
            if progress:
                progress(min(99, int(100 * (skipped + 1) / total)), 'Đang tua tới đoạn đã chọn')
        while True:
            _check_cancel(cancel)
            ok, bgr = cap.read()
            if not ok:
                break
            if source_index >= end:
                if whole:
                    raise VideoFormatError('Video dài quá 10 giây. Giới hạn tính theo frame thực tế.')
                break
            # Deterministic resampling by source frame timestamp.
            sample_id = int(math.floor((source_index - start) * out_fps / fps + 1e-7))
            if sample_id >= len(frames):
                if (pw, ph) != (width, height):
                    bgr = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
                frame_path = temp_dir / f'{len(frames):05d}.jpg'
                if not cv2.imwrite(str(frame_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise OSError('Không thể lưu frame video tạm.')
                frames.append(str(frame_path))
                indices.append(source_index)
                if progress:
                    progress(min(99, int(100 * (source_index + 1) / total)), 'Đang đọc video')
            source_index += 1
        read_count = source_index - start
        if not frames or read_count == 0:
            raise VideoFormatError('Video không chứa frame hợp lệ.')
        # Some broken/variable-frame-rate files have inconsistent metadata.
        if whole and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 > MAX_SECONDS + 0.1:
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


def cleanup_legacy_matting_cache(output_dir: str) -> None:
    """Remove only the obsolete per-output-dir cache used before App_03."""
    cache = Path(output_dir) / '_matting_cache'
    if cache.is_symlink():
        cache.unlink()
    elif cache.is_dir():
        shutil.rmtree(cache)


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
                progress=None, cancel=None, report_dir=None, previous=None, frames=None):
    """Track with REAL HQ-SAM2 temporal memory; no optical-flow fallback.

    ``keyframes`` and the result masks are label maps (0 = no object,
    k = Object k, see video_objects.py); each object is a conditioning prompt
    via add_new_mask with its own memory. The GPU child uses the HQ-SAM2 Large
    base + fine-tuned HQ-SAM decoder. This adapter raises an explicit error
    if PyTorch/native checkpoints are not available. Returns
    video_track.TrackResult (``previous``, ``frames``: see track_video).
    """
    if not keyframes:
        raise VideoFormatError('Chưa bôi mask. Hãy chọn vùng mask trên ít nhất một frame trước khi Track.')
    if not any(np.any(mask > 0) for mask in keyframes.values()):
        raise VideoFormatError('Mask đang trống. Hãy Restore để tô object cần giữ trước khi Track.')
    _check_cancel(cancel)
    from .video_track import track_video
    extra = {"report_dir": report_dir} if report_dir is not None else {}
    return track_video(clip, keyframes, progress=progress, cancel=cancel, previous=previous,
                       frames=frames, **extra)


# Settings "Tùy chọn xuất video": a chroma-key MP4 (colors in RGB), or a
# transparent RGBA PNG sequence that keeps the soft alpha for video editors.
CHROMA_BACKGROUNDS = {'green': (0, 255, 0), 'blue': (0, 0, 255)}
TRANSPARENT = 'transparent'


def export_destination(folder, source: str, background: str) -> Path:
    """Unused name '<video>_cut.mp4' (chroma) or folder '<video>_cut_png' (transparent)."""
    folder = Path(folder)
    stem = Path(source).stem + '_cut'
    pattern = '{}_png' if background == TRANSPARENT else '{}.mp4'
    path, suffix = folder / pattern.format(stem), 2
    while path.exists():
        path, suffix = folder / pattern.format(f'{stem}_{suffix}'), suffix + 1
    return path


class _Mp4Output:
    """Chroma-key MP4: frame over a solid color. No alpha channel, no audio."""

    def __init__(self, dest: Path, clip: VideoClip, background: str):
        self.dest = dest
        self.temp = dest.with_name(dest.stem + '.partial.mp4')
        self.writer = cv2.VideoWriter(str(self.temp), cv2.VideoWriter_fourcc(*'mp4v'),
                                      clip.output_fps, (clip.source_width, clip.source_height))
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError('Không mở được MP4 encoder OpenCV mp4v.')
        self.color_bgr = np.asarray(CHROMA_BACKGROUNDS[background][::-1], dtype=np.float32)

    def render(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Thread-safe part: the composited frame."""
        a = alpha[..., None].astype(np.float32) / 255.0
        return np.clip(frame * a + self.color_bgr * (1.0 - a), 0, 255).astype(np.uint8)

    def save(self, index: int, rendered: np.ndarray):
        """In frame order."""
        self.writer.write(rendered)

    def commit(self):
        self.writer.release()
        self.temp.replace(self.dest)

    def discard(self):
        self.writer.release()
        self.temp.unlink(missing_ok=True)


class _PngOutput:
    """Folder of RGBA PNG frames 00000.png, 00001.png ... (straight alpha, 8-bit)."""

    def __init__(self, dest: Path):
        self.dest = dest
        self.temp = dest.with_name(dest.name + '.partial')
        shutil.rmtree(self.temp, ignore_errors=True)
        self.temp.mkdir()

    def render(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Thread-safe part: the encoded PNG."""
        ok, png = cv2.imencode('.png', np.dstack([frame, alpha]))
        if not ok:
            raise OSError('Không mã hóa được PNG frame.')
        return png

    def save(self, index: int, rendered: np.ndarray):
        (self.temp / f'{index:05d}.png').write_bytes(rendered.tobytes())

    def commit(self):
        self.temp.rename(self.dest)

    def discard(self):
        shutil.rmtree(self.temp, ignore_errors=True)


def _alpha_reader(clip: VideoClip, masks, alpha_files, holds):
    """frame -> its 8-bit alpha at source size (what the 8-bit output shows).

    ViTMatte-SAM PNGs (``alpha_files``: frame -> path); a held frame gets its
    source's alpha moved by the shake, scaled from proxy to source pixels.
    Without them, the tracked binary mask.
    """
    size = (clip.source_width, clip.source_height)
    if alpha_files is None:
        return lambda i: (masks[i] if masks[i].shape == size[::-1]
                          else cv2.resize(masks[i], size, interpolation=cv2.INTER_LINEAR))
    from .video_holds import shifted  # video_holds imports this module, so not at the top
    to_source = clip.source_width / clip.proxy_width, clip.source_height / clip.proxy_height

    def read(i):
        hold = holds.get(i)
        alpha = cv2.imread(str(alpha_files[hold.source if hold else i]), cv2.IMREAD_UNCHANGED)
        if alpha is None or alpha.dtype != np.uint8 or alpha.shape != size[::-1]:
            raise RuntimeError(f'Alpha ViTMatte-SAM frame {i} không phải PNG 8-bit cỡ video.')
        if hold:
            alpha = shifted(alpha, hold.dx * to_source[0], hold.dy * to_source[1], cv2.INTER_LINEAR)
        return alpha
    return read


def _source_frames(clip: VideoClip):
    """The full-size source frame of every clip frame, decoding the video once."""
    cap = _reader(clip.source)
    try:
        next_index = 0
        for desired_index in clip.source_frame_indices:
            _skip_frames(cap, desired_index - next_index)
            ok, frame = cap.read()
            if not ok:
                raise VideoFormatError(f'Không đọc được frame {desired_index} khi export.')
            next_index = desired_index + 1
            yield frame
    finally:
        cap.release()


def _compose(clip: VideoClip, output, alpha_of, foreground: bool, report_progress, cancel):
    """Every frame: alpha, foreground colours, composite, on COMPOSE_THREADS
    threads (OpenCV and NumPy release the GIL); written in frame order."""
    def render(i, frame):
        alpha = alpha_of(i)
        if foreground:
            frame = estimate_foreground(frame, alpha.astype(np.float32) / 255.0)
        return output.render(frame, alpha)

    threads = max(1, min(COMPOSE_THREADS, os.cpu_count() or 1))
    pending = deque()  # (frame, its render job), in frame order

    def write_oldest():
        index, job = pending.popleft()
        output.save(index, job.result())
        report_progress(index)

    frames = _source_frames(clip)
    with ThreadPoolExecutor(threads) as pool:
        try:
            for i, frame in enumerate(frames):
                _check_cancel(cancel)
                pending.append((i, pool.submit(render, i, frame)))
                if len(pending) > 2 * threads:  # bounded memory: ~2 frames per thread
                    write_oldest()
            while pending:
                write_oldest()
        finally:
            frames.close()
            for _, job in pending:
                job.cancel()


def export_video(clip: VideoClip, masks: tuple[np.ndarray, ...], dest: str,
                 background='green', progress=None, cancel=None,
                 matting='binary', report_dir=None, foreground=True, holds=None) -> str:
    """Optional ViTMatte-SAM soft alpha on CUDA, then MP4 or PNG sequence.

    ViTMatte-SAM runs in a child process *after* SAM2 tracking has exited.
    ``holds`` (frame -> video_holds.Hold, from Track, checked by the caller
    against the masks exported now): a held frame is not matted, it reuses its
    source's alpha moved by the shake, so held hair does not flicker. Binary
    mode stays usable for the CPU edition and codec self-tests. Frames keep
    the Track fps (clip.output_fps); there is no audio. ``foreground``
    recolours soft edges of every frame (foreground.py) before compositing.
    """
    holds = dict(holds or {})
    if len(masks) != clip.count:
        raise VideoFormatError('Cần Track lại sau khi thay đổi mask trước khi Export.')
    matting = str(matting).lower()
    if matting not in ('vitmatte_sam', 'binary'):
        raise VideoFormatError('video.matting_export phải là vitmatte_sam hoặc binary.')
    if background != TRANSPARENT and background not in CHROMA_BACKGROUNDS:
        raise VideoFormatError('Nền xuất video không hợp lệ.')
    kind = 'PNG' if background == TRANSPARENT else 'MP4'
    diag = Path(report_dir) if report_dir else create_run(Path(dest).parent, 'export')
    diag.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dest_path = Path(dest)
    report = {'version': REPORT_VERSION, 'status': 'RUNNING', 'action': 'export',
              'video': str(Path(clip.source).name), 'frames': clip.count,
              'matting': matting, 'output_fps': clip.output_fps, 'background': background,
              'foreground_estimation': bool(foreground),
              'source_resolution': [clip.source_width, clip.source_height],
              'stages_sec': {}, 'parent_rss_start_mib': ram_mib(),
              'note': ('PNG RGBA sequence, no audio' if kind == 'PNG'
                       else 'MP4 chroma key, no alpha channel, no audio')}
    dump_json(diag/'export_report.json', report)
    output = None
    try:
        _check_cancel(cancel)
        alpha_files = None
        if matting == 'vitmatte_sam':
            # No silent fallback: fail loudly if CUDA/ViTMatte-SAM is unavailable.
            from .video_matting import run_vitmatte_sam
            t = time.perf_counter()
            computed = [i for i in range(clip.count) if i not in holds]
            report['held_frames'] = {str(i): list(hold) for i, hold in sorted(holds.items())}
            def matting_progress(value, message):
                if progress:
                    progress(min(70, round(value*.70)), message)
            files = run_vitmatte_sam(clip, masks, diag, computed,
                                     progress=matting_progress, cancel=cancel)
            alpha_files = dict(zip(computed, files))
            report['stages_sec']['matting_alpha'] = round(time.perf_counter()-t,3)
        else:
            report['stages_sec']['matting_alpha'] = 0

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        output = (_PngOutput(dest_path) if kind == 'PNG'
                  else _Mp4Output(dest_path, clip, background))
        start, span = (70, 30) if alpha_files else (0, 100)

        def compose_progress(index):
            if progress:
                progress(start + round(span*(index+1)/clip.count),
                         f'Đang xuất {kind} {index+1}/{clip.count}')
        compose_t0 = time.perf_counter()
        _compose(clip, output, _alpha_reader(clip, masks, alpha_files, holds),
                 foreground, compose_progress, cancel)
        _check_cancel(cancel)
        output.commit()
        report['stages_sec']['compose_' + kind.lower()] = round(time.perf_counter()-compose_t0,3)
        size = (sum(f.stat().st_size for f in dest_path.iterdir()) if dest_path.is_dir()
                else dest_path.stat().st_size)
        report.update(status='OK', total_sec=round(time.perf_counter()-started,3),
                      output=str(dest_path), output_bytes=size,
                      parent_rss_finish_mib=ram_mib())
        dump_json(diag/'export_report.json', report)
        return str(dest_path)
    except BaseException as exc:
        if output is not None:
            output.discard()
        report.update(status='CANCELLED' if isinstance(exc,VideoCanceled) else 'ERROR',
                      error=f'{type(exc).__name__}: {exc}',
                      total_sec=round(time.perf_counter()-started,3))
        dump_json(diag/'export_report.json', report)
        raise
