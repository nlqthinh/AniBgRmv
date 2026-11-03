"""CPU edition post-build smoke test: real UNet and ViTMatte-UNet ONNX in the frozen EXE."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


def _report_path() -> Path:
    return Path(sys.executable).resolve().parent / 'cpu_selftest.json'


def _test_impl() -> dict:
    import numpy as np
    import onnxruntime as ort
    root = Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
    cfg = json.loads((root/'app_config.json').read_text(encoding='utf-8'))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    outputs = {}
    for label, key, channels in (('UNet','unet_onnx_path',3),
                                  ('ViTMatte-UNet','vitmatte_unet_onnx_path',4)):
        model = root/cfg['model'][key]
        if not model.is_file():
            raise FileNotFoundError(f'{label}: bundled model missing: {model}')
        session = ort.InferenceSession(str(model), sess_options=opts,
                                       providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError(f'{label}: CPU execution provider mismatch')
        item = session.get_inputs()[0]
        if len(item.shape) != 4:
            raise RuntimeError(f'{label}: invalid input shape: {item.shape}')
        data = np.zeros((1,channels,768,768),np.float32)
        if channels == 4:
            data[:,3,:,:] = .5
        value = session.run(None,{item.name:data})
        if not value or not np.isfinite(value[0]).all():
            raise RuntimeError(f'{label}: invalid/nonfinite ONNX output')
        outputs[label] = list(value[0].shape)
    return {'status':'PASS','ort_version':ort.__version__, 'output_shapes':outputs}


def main() -> int:
    try:
        record = _test_impl()
        status = 0
    except BaseException as exc:
        record = {'status':'FAIL','error':f'{type(exc).__name__}: {exc}',
                  'traceback':traceback.format_exc()}
        status = 1
    try:
        _report_path().write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    except OSError as exc:
        print(f'[FAIL] Cannot write CPU smoke result: {exc}',file=sys.stderr)
        return 1
    print(f"[{record['status']}] Frozen UNet + ViTMatte-UNet CPU selftest", flush=True)
    return status
