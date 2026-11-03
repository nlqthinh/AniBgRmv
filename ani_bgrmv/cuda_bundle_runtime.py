"""Deterministic CUDA/cuDNN loading in the PyInstaller GPU edition.

No PySide6 or PyTorch import. On Windows, validate actual loaded module paths,
not merely CUDAExecutionProvider availability. Development mode keeps the
existing ONNX Runtime preload behavior.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

_HANDLES = []
_LOADED = False
_DLL_NAMES = (
    'cudnn_engines_runtime_compiled64_9.dll',
    'cudnn_engines_precompiled64_9.dll',
    'cudnn_heuristic64_9.dll',
    'cudnn_ops64_9.dll',
    'cudnn_adv64_9.dll',
    'cudnn_graph64_9.dll',
    'cudnn64_9.dll',
    'cudnn_engines_tensor_ir64_9.dll',
)


def loaded_module_path(name: str) -> str | None:
    """Return the location Windows *actually* loaded, not the selected source."""
    if os.name != 'nt':
        return None
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint
    handle = kernel.GetModuleHandleW(name)
    if not handle:
        return None
    buf = ctypes.create_unicode_buffer(32768)
    count = kernel.GetModuleFileNameW(handle, buf, len(buf))
    return buf.value if count else None


def loaded_cuda_modules() -> dict[str, str]:
    found = {}
    for name in _DLL_NAMES + ('cublas64_12.dll', 'cublasLt64_12.dll',
                              'cudart64_12.dll', 'nvrtc64_120_0.dll'):
        path = loaded_module_path(name)
        if path:
            found[name] = path
    return found


def _require_bundled_cudnn(bundled: Path) -> None:
    for name in ('cudnn64_9.dll', 'cudnn_graph64_9.dll', 'cudnn_ops64_9.dll'):
        if not (bundled / name).is_file():
            raise RuntimeError(f'CUDA bundle missing {name}: {bundled}')


def _check_module_paths(bundled: Path) -> None:
    expected = bundled.resolve()
    for name, actual in loaded_cuda_modules().items():
        if not (bundled / name).is_file():
            continue
        if Path(actual).resolve().parent != expected:
            raise RuntimeError(
                f'CUDA DLL loaded from wrong directory: {name} = {actual}; '
                f'expected {bundled / name}. Another CUDA distribution is shadowing the EXE.'
            )


def preload_runtime():
    global _LOADED
    if _LOADED:
        return
    # Work with source-mode Python without changing established behavior.
    if os.name != 'nt' or not getattr(sys, 'frozen', False):
        import onnxruntime as ort
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        _LOADED = True
        return

    import ctypes
    base = Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
    bundled = base / 'gpu_runtime'
    if not bundled.is_dir():
        raise RuntimeError(f'bundled gpu_runtime missing: {bundled}')
    _require_bundled_cudnn(bundled)

    # Register the directory *before* importing ORT. Dynamic cuDNN backend
    # modules are resolved at execution time, not always at session creation.
    _HANDLES.append(os.add_dll_directory(str(bundled)))
    prefix = str(bundled) + os.pathsep
    if not os.environ.get('PATH', '').lower().startswith(prefix.lower()):
        os.environ['PATH'] = prefix + os.environ.get('PATH', '')

    # ORT's preload helper uses an explicit cuDNN9 order. Use explicit paths
    # here so we can verify actual modules even if ORT imports implicitly.
    for name in _DLL_NAMES:
        candidate = bundled / name
        if candidate.is_file():
            try:
                _HANDLES.append(ctypes.WinDLL(str(candidate)))
            except OSError as exc:
                if name in ('cudnn64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_graph64_9.dll'):
                    raise RuntimeError(f'could not preload required {candidate}: {exc}') from exc
                # Some optional cuDNN engines require NVRTC at load time.
                # They may be unused by UNet; the frozen UNet selftest is the gate.
                print(f'[WARN] optional cuDNN plugin unavailable: {name}: {exc}', flush=True)

    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls(directory=str(bundled))
    _check_module_paths(bundled)
    _LOADED = True


def cudnn_version() -> str | None:
    if os.name != 'nt':
        return None
    import ctypes
    path = loaded_module_path('cudnn64_9.dll')
    if not path:
        return None
    lib = ctypes.WinDLL(path)
    func = lib.cudnnGetVersion
    func.restype = ctypes.c_size_t
    raw = int(func())
    # cuDNN version encoded as MAJOR*10000 + MINOR*100 + PATCH.
    return f'{raw // 10000}.{(raw // 100) % 100}.{raw % 100}'
