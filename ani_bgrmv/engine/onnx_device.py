"""The app's ONNX Runtime policy for one device ('cpu' or 'cuda:N'), shared by every model.

* GPU never falls back to CPU: the session must start on CUDAExecutionProvider
  and fallback is disabled; a CPU session must use CPUExecutionProvider only.
* A model file that cannot be read or a CUDA start failure gives a message
  that says what to do, instead of the raw ONNX Runtime error.
* GPU arena shrinkage after a run (4 GB cards) comes from one place.
Callers that need their own SessionOptions (the image engine) pass them; the
default options are those of HQ-SAM, whose first GPU run is also profiled to
prove its layers executed on CUDA.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import onnxruntime as ort

from .onnx_cuda_audit import audit_cuda_profile


class OnnxDevice:
    def __init__(self, device: str, *, cpu_threads: int = 2, shrink_gpu_arena: bool = True,
                 log_timing: bool = False, cuda_options: dict | None = None):
        self.device = device
        self.uses_gpu = device.startswith('cuda:')
        if self.uses_gpu:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('ONNX Runtime không có CUDAExecutionProvider; không fallback CPU.')
            self._providers = [('CUDAExecutionProvider', dict(cuda_options or {}))]
        elif device == 'cpu':
            self._providers = ['CPUExecutionProvider']
        else:
            raise ValueError(f'Thiết bị ONNX không hợp lệ: {device}')
        self._cpu_threads = max(1, min(4, int(cpu_threads)))
        self._shrink_gpu_arena = bool(shrink_gpu_arena)
        self._log_timing = bool(log_timing)
        self._audited: set[str] = set()

    def _default_options(self, part: str) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        # Keep the ORT threadpool small so the GUI thread stays responsive.
        opts.intra_op_num_threads = self._cpu_threads
        opts.inter_op_num_threads = 1
        if self.uses_gpu and part not in self._audited:
            opts.enable_profiling = True  # first run only; see _audit_first_run
        return opts

    def session(self, path: Path, part: str, inputs=(), outputs=(), *,
                options: ort.SessionOptions | None = None) -> ort.InferenceSession:
        """Open ``path`` on this device; ``inputs``/``outputs`` are required graph names."""
        try:
            session = ort.InferenceSession(str(path), sess_options=options or self._default_options(part),
                                           providers=self._providers)
        except Exception as exc:
            reason = str(exc)
            if 'INVALID_PROTOBUF' in reason or 'Protobuf parsing failed' in reason:
                raise RuntimeError(
                    f'{part} ONNX không đọc được (INVALID_PROTOBUF): {path}. '
                    'File model bị hỏng hoặc copy chưa xong; hãy export/copy lại. '
                    f'Chi tiết: {exc}') from exc
            if self.uses_gpu:
                raise RuntimeError(
                    f'{part} không khởi tạo được CUDAExecutionProvider: {exc}. '
                    'Kiểm tra DLL CUDA/cuDNN và VRAM; app không chuyển sang CPU.') from exc
            raise
        if self.uses_gpu:
            if session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError(f'{part} không chạy CUDAExecutionProvider; không fallback CPU.')
            session.disable_fallback()
        elif session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError(f'{part} ở chế độ CPU phải dùng CPUExecutionProvider.')
        ins = {x.name for x in session.get_inputs()}
        outs = {x.name for x in session.get_outputs()}
        if not set(inputs).issubset(ins) or not set(outputs).issubset(outs):
            raise RuntimeError(f'{part} ONNX sai graph: inputs={ins}; outputs={outs}.')
        return session

    def run_options(self) -> ort.RunOptions | None:
        """Return idle CUDA arena memory after a run (4 GB GPUs); weights stay."""
        if not (self.uses_gpu and self._shrink_gpu_arena):
            return None
        options = ort.RunOptions()
        options.add_run_config_entry('memory.enable_memory_arena_shrinkage',
                                     'gpu:' + self.device.split(':', 1)[1])
        return options

    def _audit_first_run(self, session, part: str) -> None:
        """Strict GPU: prove the first run executed its layers on CUDA."""
        if not self.uses_gpu or part in self._audited:
            return
        path = Path(session.end_profiling())
        try:
            audit_cuda_profile(json.loads(path.read_text(encoding='utf-8')), part)
        finally:
            path.unlink(missing_ok=True)
        self._audited.add(part)

    def run(self, session, part: str, outputs, feed):
        """Run a session opened with the default options (first GPU run audited)."""
        started = time.perf_counter()
        result = session.run(list(outputs), feed, self.run_options())
        self._audit_first_run(session, part)
        if self._log_timing:
            print(f'[AniBgRmv HQ-SAM] {part}_ms={(time.perf_counter() - started) * 1000:.1f}',
                  flush=True)
        return result
