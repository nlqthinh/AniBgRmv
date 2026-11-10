"""Run UNet inside the *built EXE*, matching the GUI CUDA session settings.

Called as ``AniBGRmv.exe --cuda-selftest`` after PyInstaller COLLECT.
Writes JSON even if the runw.exe build has no visible console. No PySide6.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


def _report_path() -> Path:
    return Path(sys.executable).resolve().parent / 'gpu_selftest.json'


def _test_impl() -> dict:
    from .cuda_bundle_runtime import preload_runtime, loaded_cuda_modules, cudnn_version
    preload_runtime()
    import numpy as np
    import onnxruntime as ort
    from .engine import matting_ops as ops
    from .gpu_memory_options import cuda_provider_options

    root = Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
    cfg = json.loads((root / 'app_config.json').read_text(encoding='utf-8'))
    model = root / cfg['model']['unet_onnx_path']
    if not model.is_file():
        raise RuntimeError(f'Bundled UNet model missing: {model}')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError(f'CUDAExecutionProvider unavailable: {ort.get_available_providers()}')
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.enable_profiling = True
    so.profile_file_prefix = str(_report_path().with_suffix(''))
    so.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    opts = cuda_provider_options(0, settings=cfg.get('compute', {}))
    sess = ort.InferenceSession(str(model), sess_options=so,
                                providers=[('CUDAExecutionProvider', opts)])
    sess.disable_fallback()
    if 'CUDAExecutionProvider' not in sess.get_providers():
        raise RuntimeError(f'CUDAExecutionProvider not initialized: {sess.get_providers()}')
    input_name = sess.get_inputs()[0].name
    # Same shape as the app's UNet letterbox image. Two values exercise both
    # first-run cuDNN graph build and subsequent execution, with real data.
    shape = ops.model_input_shape(ops.UNET_CHANNELS)
    samples = [np.zeros(shape, np.float32),
               np.random.default_rng(20261031).uniform(0.0, 1.0, shape).astype(np.float32)]
    outputs = []
    for x in samples:
        y = sess.run(None, {input_name:x})
        if not y or not np.isfinite(y[0]).all():
            raise RuntimeError('UNet CUDA output missing or nonfinite')
        outputs.append(list(y[0].shape))
    trace = Path(sess.end_profiling())
    try:
        events = json.loads(trace.read_text(encoding='utf-8'))
    finally:
        trace.unlink(missing_ok=True)
    convs = [e for e in events if e.get('cat') == 'Node' and
             'conv' in str(e.get('name', '')).lower()]
    gpu_convs = [e for e in convs if e.get('args', {}).get('provider') == 'CUDAExecutionProvider']
    cpu_convs = [e for e in convs if e.get('args', {}).get('provider') != 'CUDAExecutionProvider']
    if not gpu_convs or cpu_convs:
        raise RuntimeError(f'UNet Conv GPU strict: CUDA={len(gpu_convs)} CPU/other={len(cpu_convs)}')
    return {
        'status': 'PASS', 'ort_version': ort.__version__,
        'cudnn_version': cudnn_version(),
        'cuda_modules': loaded_cuda_modules(),
        'gpu_conv_events': len(gpu_convs), 'output_shapes': outputs,
        'model': model.name,
    }


def main() -> int:
    try:
        result = _test_impl()
        code = 0
    except BaseException as exc:
        # Runtime hook failures are separately written to gpu_preload_error.txt.
        from .cuda_bundle_runtime import loaded_cuda_modules, cudnn_version
        try:
            version = cudnn_version()
            modules = loaded_cuda_modules()
        except Exception:
            version, modules = None, {}
        result = {
            'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc(),
            'cudnn_version': version, 'cuda_modules': modules,
        }
        code = 1
    try:
        _report_path().write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    except OSError as exc:
        print(f'Could not write selftest report: {exc}', file=sys.stderr)
        return 1
    print(f"[{result['status']}] Frozen UNet CUDA selftest. See {_report_path()}", flush=True)
    return code
