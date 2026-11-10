"""Persistent per-job video logs; never put checkpoints or image pixels in logs."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

REPORT_VERSION = 'video-report-v1'


def create_run(output_dir, kind):
    folder = Path(output_dir) / '_diagnostics' / (
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + kind + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def dump_json(path, data):
    """Persist partial diagnostics after a failure, without corrupting an older report."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, p)


def dump_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _processes(pid):
    """This process, or a worker *with its children*: on Windows the .venv
    python.exe is only a launcher that runs the real Python as its child, so
    the worker pid alone stays at ~4 MiB (App_13 Track report)."""
    import psutil
    if not pid:
        return [psutil.Process()]
    worker = psutil.Process(pid)
    return [worker, *worker.children(recursive=True)]


def _sum_over(pid, read):
    """Sum ``read(process)``; None without psutil or once the worker has exited."""
    try:
        import psutil
    except ImportError:
        return None
    total = 0.0
    try:
        for process in _processes(pid):
            try:
                total += read(process)
            except psutil.NoSuchProcess:  # a child that just ended
                pass
    except (psutil.Error, OSError, RuntimeError):
        return None
    return total


def ram_mib(pid=None):
    total = _sum_over(pid, lambda p: p.memory_info().rss)
    return None if total is None else round(total / 1048576, 1)


def qseconds(value):
    return round(float(value), 4)


def now():
    return time.perf_counter()


# nvidia-smi field -> report key. Clocks, power and P-state show whether a
# slow run is a GPU held at low clock (power limit, laptop power saving).
GPU_FIELDS = (
    ('utilization.gpu', 'gpu_util_percent'),
    ('memory.used', 'gpu_dedicated_used_mib'),
    ('memory.total', 'gpu_total_mib'),
    ('temperature.gpu', 'gpu_temp_c'),
    ('clocks.sm', 'gpu_sm_clock_mhz'),
    ('clocks.max.sm', 'gpu_sm_clock_max_mhz'),
    ('power.draw', 'gpu_power_w'),
    ('power.limit', 'gpu_power_limit_w'),
    ('pstate', 'gpu_pstate'),
)


def _gpu_value(text):
    text = text.strip()
    if text.startswith('['):  # [N/A], [Not Supported] (common on laptops)
        return None
    try:
        return float(text)
    except ValueError:
        return text  # pstate, e.g. 'P0'


def gpu_snapshot():
    """Best-effort discrete GPU metrics (WDDM shared memory is not observable).

    Never blocks UI: only called in background processes and timeout 2 seconds.
    Failure is non-fatal, because not every Windows machine has nvidia-smi.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=' + ','.join(field for field, _ in GPU_FIELDS),
             '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=2, check=True)
        parts = result.stdout.strip().splitlines()[0].split(',')
    except (OSError, IndexError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    if len(parts) != len(GPU_FIELDS):
        return {}
    return {key: _gpu_value(text) for (_, key), text in zip(GPU_FIELDS, parts)}


def process_cpu_seconds(pid):
    """Cumulative CPU seconds of a worker and its children; divide by elapsed time and cores."""
    def busy(process):
        times = process.cpu_times()
        return times.user + times.system
    total = _sum_over(pid, busy)
    return None if total is None else round(total, 4)
