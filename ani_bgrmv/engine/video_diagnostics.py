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


def ram_mib(pid=None):
    try:
        import psutil
        p = psutil.Process(pid) if pid else psutil.Process()
        return round(p.memory_info().rss / 1048576, 1)
    except (ImportError, OSError, RuntimeError):
        return None


def qseconds(value):
    return round(float(value), 4)


def now():
    return time.perf_counter()


def gpu_snapshot():
    """Best-effort discrete GPU metrics (WDDM shared memory is not observable).

    Never blocks UI: only called in background processes and timeout 2 seconds.
    Failure is non-fatal, because not every Windows machine has nvidia-smi.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits', '-i', '0'],
            capture_output=True, text=True, timeout=2, check=True)
        parts = result.stdout.strip().splitlines()[0].split(',')
        if len(parts) != 4:
            return {}
        return dict(zip(('gpu_util_percent','gpu_dedicated_used_mib',
                         'gpu_total_mib','gpu_temp_c'),
                        (float(x.strip()) for x in parts)))
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}


def process_cpu_seconds(pid):
    """Cumulative CPU seconds for one process; divide by elapsed time and cores."""
    try:
        import psutil
        t = psutil.Process(pid).cpu_times()
        return round(float(t.user + t.system), 4)
    except (ImportError,OSError,RuntimeError):
        return None
