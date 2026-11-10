"""Launch and supervise the video workers that run in an external Python.

Track (HQ-SAM2, PyTorch) and Export matting (ViTMatte-SAM, ONNX CUDA) run as
child processes, so their GPU memory and crashes never reach the GUI. Each
worker prints '<PREFIX>{json}' events on stdout: progress (percent, message),
metric, error and complete. This module is the only place that starts them.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .video_diagnostics import gpu_snapshot, process_cpu_seconds, ram_mib
from .video_pipeline import VideoFormatError, _check_cancel

SAMPLE_EVERY_SEC = 3.0


def python_runner() -> Path:
    """Python with torch/HQ-SAM2 (video.tracker_python_path), or this one when run from source."""
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


def worker_script(name: str) -> Path:
    """Worker .py file. The EXE bundles it as plain data (AniBGRmv.spec VIDEO_SIDECAR_FILES)
    because the external Python must be able to run it directly."""
    for path in (Path(__file__).resolve().with_name(name),
                 Path(getattr(sys, '_MEIPASS', '')) / 'ani_bgrmv' / 'engine' / name):
        if path.is_file():
            return path
    raise VideoFormatError(f'Thiếu file {name} trong bản build.')


def selected_video_device() -> str:
    """'cpu' or 'cuda:N' from the saved CPU/GPU choice (compute_selection).

    The GUI and the workers run in separate interpreters, so the device must
    travel in each job rather than be inferred from the worker's CUDA access.
    """
    from ..compute_selection import check_visible_devices, cuda_index, current_selection
    selection = current_selection()
    if selection['mode'] == 'cpu':
        return 'cpu'
    try:
        index = cuda_index(selection)
        check_visible_devices(index)
    except RuntimeError as exc:
        raise VideoFormatError(str(exc)) from exc
    return f'cuda:{index}'


def run_worker(script_name: str, job_file: Path, prefix: str, *, label: str,
               progress=None, cancel=None, samples: list | None = None,
               sample_gpu: bool = True) -> list[dict]:
    """Run one worker until it exits; return its 'metric' events.

    Raises VideoCanceled on cancel, RuntimeError if the worker fails or exits
    without 'complete'. The child is always stopped before returning. Every
    SAMPLE_EVERY_SEC a RAM/CPU(/GPU) sample is appended to `samples`, so a
    caller still has them when the run fails.
    """
    env = os.environ.copy()
    # Events carry Vietnamese text: force UTF-8 whatever the console code page.
    env['PYTHONIOENCODING'] = 'utf-8:backslashreplace'
    # Windows: do not open a console window for each job.
    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
    proc = subprocess.Popen(
        [str(python_runner()), '-u', str(worker_script(script_name)), str(job_file)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
        errors='replace', bufsize=1, creationflags=creationflags, env=env)
    lines: queue.Queue[str | None] = queue.Queue()

    def pump():
        try:
            for line in proc.stdout:
                lines.put(line.rstrip('\r\n'))
        finally:
            lines.put(None)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    started = last_sample = time.perf_counter()
    metrics, log_tail, error, completed = [], [], None, False
    try:
        while True:
            _check_cancel(cancel)
            if samples is not None and time.perf_counter() - last_sample > SAMPLE_EVERY_SEC:
                samples.append({'elapsed_sec': round(time.perf_counter() - started, 3),
                                'parent_rss_mib': ram_mib(), 'worker_rss_mib': ram_mib(proc.pid),
                                'worker_cpu_total_sec': process_cpu_seconds(proc.pid),
                                **(gpu_snapshot() if sample_gpu else {})})
                last_sample = time.perf_counter()
            try:
                line = lines.get(timeout=0.15)
            except queue.Empty:
                if proc.poll() is not None and not reader.is_alive():
                    break
                continue
            if line is None:
                break
            if not line.startswith(prefix):
                if line.strip():
                    log_tail = (log_tail + [line[:400]])[-15:]
                continue
            try:
                event = json.loads(line[len(prefix):])
            except ValueError:
                continue
            kind = event.get('event')
            if kind == 'metric':
                metrics.append(event.get('sample', {}))
            elif kind == 'error':
                error = str(event.get('message', 'không rõ lỗi'))
            elif kind == 'complete':
                completed = True
            if progress and 'percent' in event:
                progress(int(event['percent']), str(event.get('message', label)))
        code = proc.wait(timeout=5)
        _check_cancel(cancel)
        if code or not completed:
            details = error or '\n'.join(log_tail[-6:]) or 'Không có log từ worker.'
            raise RuntimeError(f'{label} thất bại (exit {code}): {details}')
        return metrics
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        reader.join(timeout=1)
