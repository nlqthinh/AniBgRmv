"""Run ViTMatte-SAM ONNX in a separate Python process after HQ-SAM2 Track exits.

The worker can be canceled without leaking ORT CUDA state into the GUI process.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .video_diagnostics import REPORT_VERSION, dump_json, dump_csv
from .video_pipeline import VideoCanceled, _check_cancel
from .video_sidecar import run_worker, selected_video_device

PREFIX = '__ANIBGRMV_VIDEO_MATTING__'


def run_vitmatte_sam(clip, masks, report_dir, frames, progress=None, cancel=None):
    """ViTMatte-SAM 8-bit alpha PNGs for ``frames`` (sorted clip indices), in that order."""
    from ..app_config import APP_CONFIG
    from ..model_registry import model_path
    _check_cancel(cancel)
    device = selected_video_device()
    if device == 'cpu':
        raise RuntimeError('Đã chọn CPU: không thể chạy ViTMatte-SAM CUDA. Chọn Auto hoặc OFF cho video.')
    from ..gpu_memory_options import cuda_provider_options
    provider_options = cuda_provider_options(int(device.split(':')[1]),
                                             settings=APP_CONFIG.get('compute', {}))
    model_file = model_path('video_vitmatte_sam')
    if not model_file.is_file():
        raise FileNotFoundError(f'Thiếu ViTMatte-SAM ONNX cho video: {model_file}')
    diag = Path(report_dir)
    diag.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run_report = {'version': REPORT_VERSION, 'status': 'RUNNING', 'frames': len(frames),
                  'action': 'matting_export', 'samples': []}
    dump_json(diag / 'matting_adapter_report.json', run_report)
    with tempfile.TemporaryDirectory(prefix='anibgrmv_matting_') as tmp:
        staging = Path(tmp)
        mask_paths = []
        for idx in frames:
            _check_cancel(cancel)
            dest = staging / f'{idx:05d}.png'
            if not cv2.imwrite(str(dest), np.asarray(masks[idx], np.uint8)):
                raise OSError('Không thể lưu mask ' + str(idx))
            mask_paths.append(str(dest))
        job = {'model_onnx': str(model_file), 'source_video': clip.source,
               'source_frame_indices': [clip.source_frame_indices[idx] for idx in frames],
               'mask_paths': mask_paths, 'frame_count': len(frames),
               'report_dir': str(diag), 'cache_dir': str(Path(clip.temp_dir) / '_matting_cache'),
               'cuda_provider_options': provider_options}
        job_path = staging / 'job.json'
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        try:
            run_worker('video_matting_worker.py', job_path, PREFIX, label='ViTMatte-SAM Video Export',
                       progress=progress, cancel=cancel, samples=run_report['samples'])
            alpha_files = json.loads((diag / 'matting_alpha_index.json').read_text(encoding='utf-8'))
            if len(alpha_files) != len(frames) or not all(Path(f).is_file() for f in alpha_files):
                raise RuntimeError('ViTMatte-SAM trả thiếu alpha cache.')
            run_report.update(status='OK', total_sec=round(time.perf_counter() - started, 3))
            dump_json(diag / 'matting_adapter_report.json', run_report)
            dump_csv(diag / 'matting_process_samples.csv', run_report['samples'])
            return alpha_files
        except BaseException as exc:
            run_report.update(status='CANCELLED' if isinstance(exc, VideoCanceled) else 'ERROR',
                              error=str(exc), total_sec=round(time.perf_counter() - started, 3))
            dump_json(diag / 'matting_adapter_report.json', run_report)
            raise
