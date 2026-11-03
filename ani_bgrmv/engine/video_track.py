"""Parent-process adapter for native HQ-SAM2 temporal video propagation.

The subprocess owns PyTorch memory. Cancelling terminates the child, then its
temporary masks are removed. Existing image ONNX sessions and GUI stay intact.
"""
from __future__ import annotations

import json
import time
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

from .video_pipeline import VideoCanceled, VideoFormatError, _check_cancel
from .video_diagnostics import (REPORT_VERSION, dump_json, dump_csv, ram_mib, create_run,
                                gpu_snapshot, process_cpu_seconds)

PREFIX = '__ANIBGRMV_VIDEO_TRACK__'


def _resolve_model(raw: str) -> Path:
    from ..app_config import APP_BASE_DIR
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value
    bundled = Path(getattr(sys, '_MEIPASS', APP_BASE_DIR))
    # Development models live under ani_bgrmv/models, frozen under _internal/models.
    for root in (Path(APP_BASE_DIR), bundled, Path(sys.executable).resolve().parent):
        path = root / value
        if path.is_file():
            return path
    return Path(APP_BASE_DIR) / value


def _worker_script() -> Path:
    # In a frozen EXE, package this pure .py file as data (not a PyInstaller
    # bytecode module): external Python must be able to execute it directly.
    script = Path(__file__).resolve().with_name('video_track_worker.py')
    if script.is_file():
        return script
    script = (Path(getattr(sys, '_MEIPASS', '')) / 'ani_bgrmv' / 'engine'
              / 'video_track_worker.py')
    if script.is_file():
        return script
    raise VideoFormatError('Thiếu file video_track_worker.py của HQ-SAM2 Video Predictor.')


def _python_runner() -> Path:
    from ..app_config import APP_CONFIG
    selected = APP_CONFIG.get('video', {}).get('tracker_python_path', '').strip()
    if selected:
        py = Path(selected).expanduser()
        if not py.is_file():
            raise VideoFormatError('Không tìm thấy Python cho video tracker: ' + str(py))
        return py
    if getattr(sys, 'frozen', False):
        raise VideoFormatError(
            'GPU EXE không đóng gói PyTorch Video Predictor. '
            'Hãy thiết lập video.tracker_python_path trong app_config.json '
            'đến .venv\\Scripts\\python.exe có torch và HQ-SAM2, '
            'hoặc chạy python app.py từ project. Không tự fallback Optical Flow.'
        )
    return Path(sys.executable)


def track_video(clip, keyframes, progress=None, cancel=None, report_dir=None):
    from ..app_config import APP_CONFIG
    _check_cancel(cancel)
    started = time.perf_counter()
    diag = Path(report_dir) if report_dir else create_run(Path(clip.source).parent, 'track')
    diag.mkdir(parents=True, exist_ok=True)
    report = {'version': REPORT_VERSION, 'status': 'RUNNING', 'action': 'track',
              'clip': {'name': Path(clip.source).name, 'frames': clip.count,
                       'proxy_hw': [clip.proxy_height, clip.proxy_width],
                       'output_fps': clip.output_fps},
              'keyframes': sorted(keyframes), 'samples': [], 'errors': []}
    dump_json(diag/'track_report.json', report)
    if not keyframes or not any(np.any(m > 0) for m in keyframes.values()):
        raise VideoFormatError('Chưa bôi mask hợp lệ. Hãy Restore trước khi Track.')
    if not all(0 <= idx < clip.count and
               mask.shape == (clip.proxy_height, clip.proxy_width)
               for idx, mask in keyframes.items()):
        raise VideoFormatError('Keyframe mask không khớp video.')
    base = _resolve_model(APP_CONFIG['sam']['hqsam_base_checkpoint_path'])
    decoder = _resolve_model(APP_CONFIG['sam']['hqsam_decoder_checkpoint_path'])
    missing = [str(p) for p in (base, decoder) if not p.is_file()]
    if missing:
        raise VideoFormatError(
            'SAM2 Video Predictor cần checkpoint PyTorch đầy đủ: '\
            + ', '.join(missing) + '. Hai graph HQ-SAM ONNX ảnh tĩnh không thay thế được bộ nhớ video.'
        )
    script, python = _worker_script(), _python_runner()
    with tempfile.TemporaryDirectory(prefix='anibgrmv_video_track_') as tmp:
        staging = Path(tmp)
        in_dir = staging / 'keyframes'
        out_dir = staging / 'masks'
        in_dir.mkdir()
        out_dir.mkdir()
        paths = {}
        for idx, mask in keyframes.items():
            destination = in_dir / f'{idx:05d}.png'
            if not cv2.imwrite(str(destination), np.asarray(mask, np.uint8)):
                raise OSError('Không lưu được keyframe mask ' + str(idx))
            paths[str(idx)] = str(destination)
        job = dict(base_checkpoint=str(base.resolve()),
                   decoder_checkpoint=str(decoder.resolve()),
                   frames_dir=str(Path(clip.temp_dir).resolve()),
                   frame_count=clip.count,
                   hw=[clip.proxy_height, clip.proxy_width],
                   keyframe_masks=paths,
                   output_dir=str(out_dir), worker_report=str(diag / 'track_worker_report.json'),
                   sample_every=8)
        job_file = staging / 'job.json'
        job_file.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        # Windows may default redirected child stdout to cp1252/cp1258. The
        # The progress protocol includes Vietnamese text, so force UTF-8 in the child
        # rather than relying on the parent pipe's decoding configuration.
        child_env = os.environ.copy()
        child_env['PYTHONIOENCODING'] = 'utf-8:backslashreplace'
        # Windows: do not spawn a console window for each tracking operation.
        creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
        proc = subprocess.Popen(
            [str(python), '-u', str(script), str(job_file)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=creationflags, env=child_env,
        )
        events: queue.Queue[str | None] = queue.Queue()
        def read_output():
            try:
                for line in proc.stdout:
                    events.put(line.rstrip('\n'))
            finally:
                events.put(None)
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        log_tail = []
        sample_last = time.perf_counter()
        child_mem_samples = []
        reported_error = None
        completed = False
        try:
            while True:
                _check_cancel(cancel)
                if time.perf_counter() - sample_last > 3.0:
                    report['samples'].append({'elapsed_sec': round(time.perf_counter()-started,3),
                                              'parent_rss_mib': ram_mib(),
                                              'worker_rss_mib': ram_mib(proc.pid),
                                              'worker_cpu_total_sec': process_cpu_seconds(proc.pid),
                                              **gpu_snapshot()})
                    sample_last = time.perf_counter()
                try:
                    line = events.get(timeout=0.15)
                except queue.Empty:
                    if proc.poll() is not None and not reader.is_alive():
                        break
                    continue
                if line is None:
                    break
                if line.startswith(PREFIX):
                    try:
                        obj = json.loads(line[len(PREFIX):])
                    except json.JSONDecodeError:
                        continue
                    if obj.get('event') == 'metric':
                        child_mem_samples.append(obj.get('sample', {}))
                    if obj.get('event') == 'error':
                        reported_error = str(obj.get('message', 'Video Predictor thất bại'))
                    elif obj.get('event') == 'complete':
                        completed = True
                    if progress and 'percent' in obj:
                        progress(int(obj['percent']), str(obj.get('message', 'Video Track')))
                elif line.strip():
                    log_tail.append(line[:240])
                    log_tail = log_tail[-12:]
            code = proc.wait(timeout=5)
            _check_cancel(cancel)
            if code or not completed:
                details = reported_error or '\n'.join(log_tail[-5:]) or 'Không có báo cáo từ worker.'
                raise RuntimeError(f'HQ-SAM2 Video Predictor thất bại (exit {code}): {details}')
            result = []
            for i in range(clip.count):
                file = out_dir / f'{i:05d}.png'
                mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (clip.proxy_height, clip.proxy_width):
                    raise RuntimeError('Thiếu mask frame ' + str(i))
                result.append(mask)
            # Correct exact duplicates using the most recent user keyframe
            # as canonical. Never replace conflicting manual corrections.
            from .video_pipeline import stabilize_identical_frames
            result, reused, conflicts = stabilize_identical_frames(clip, result, keyframes)
            report['duplicate_rgb_reuses'] = reused
            report['conflicting_duplicate_keyframes'] = conflicts
            report['worker_memory_samples'] = child_mem_samples
            worker_report_path=diag/'track_worker_report.json'
            if worker_report_path.is_file():
                child=json.loads(worker_report_path.read_text(encoding='utf-8'))
                report['stages_sec']=child.get('stages_sec', {})
                report['gpu_device']=child.get('gpu')
                report['worker_total_sec']=child.get('total_sec')
                report['track_frames_perf']=child.get('track_frames_perf',{})
            report['status'] = 'OK'
            report['total_sec'] = round(time.perf_counter()-started,3)
            report['avg_total_sec_per_frame'] = round(report['total_sec']/clip.count,3)
            dump_json(diag/'track_report.json', report)
            dump_csv(diag/'process_samples.csv', report['samples'])
            return tuple(result)
        except BaseException as exc:
            report.update(status='CANCELLED' if isinstance(exc, VideoCanceled) else 'ERROR',
                          error=f'{type(exc).__name__}: {exc}',
                          total_sec=round(time.perf_counter()-started,3))
            dump_json(diag/'track_report.json', report)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
            reader.join(timeout=1)
