"""Persisted CPU/GPU selection; F3, M7 and both SAM graphs run on ONNX Runtime.

Strict GPU (A) means NVIDIA CUDA and never fallback to CPU; AMD/Intel are
visible in GPU Check but not supported by this release.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .app_config import APP_CONFIG, get_external_config_path
from .portable_device_settings import (build_variant, load_machine_selection, save_machine_selection)


def normalized_selection(raw=None):
    data = raw if isinstance(raw, dict) else {}
    mode = str(data.get("mode", "cpu")).lower()
    if mode not in ("cpu", "gpu"):
        mode = "cpu"
    gpu_id = data.get("gpu_id")
    gpu_id = gpu_id if isinstance(gpu_id, str) and gpu_id else None
    gpu_name = data.get("gpu_name")
    gpu_name = gpu_name if isinstance(gpu_name, str) else None
    return {"mode": mode, "gpu_id": gpu_id, "gpu_name": gpu_name}


def current_selection():
    if getattr(sys, "frozen", False):
        return normalized_selection(load_machine_selection())
    return normalized_selection(APP_CONFIG.get("compute", {}))


def save_selection(selection):
    """Update only 'compute' in external JSON; preserve user paths/settings."""
    selection = normalized_selection(selection)
    if getattr(sys, "frozen", False):
        return normalized_selection(save_machine_selection(selection))
    path = get_external_config_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f"Cấu hình không hợp lệ: {path}")
    data["compute"] = selection
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    APP_CONFIG["compute"] = selection
    return selection


def cuda_index(selection):
    if selection["mode"] != "gpu":
        raise ValueError("Không ở chế độ GPU")
    if build_variant() == "cpu":
        raise RuntimeError("Đây là CPU Edition, không chứa CUDA. Cần GPU Edition để dùng NVIDIA.")
    match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", selection.get("gpu_id") or "")
    if match is None:
        raise RuntimeError("GPU đã chọn chưa hỗ trợ CUDA ONNX cho F3/M7/SAM. Chọn GPU NVIDIA bằng GPU Check.")
    return int(match.group(1))


# Keep directory handles alive: Windows searches those for dependent CUDA DLLs.
_CUDA_DLL_HANDLES = []


def preload_cuda_runtime():
    """Preload CUDA/cuDNN without importing torch in the frozen app."""
    import onnxruntime as ort
    # PyInstaller's gpu_runtime folder contains only selected CUDA DLLs, not torch.
    root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    bundled = root / "gpu_runtime"
    if os.name == 'nt' and bundled.is_dir():
        _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(bundled)))
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls(directory=str(bundled))
    elif hasattr(ort, 'preload_dlls'):
        # Dev environment: use torch/lib DLLs if present, but NEVER import torch.
        ort.preload_dlls()


def verify_gpu_for_sam(selection):
    """Check strict ONNX CUDA prerequisites, without importing PyTorch.

    Model/session execution is checked separately by ONNX Runtime when loaded.
    """
    import onnxruntime as ort
    idx = cuda_index(selection)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            f"onnxruntime {ort.__version__} không có CUDAExecutionProvider. "
            "Cần onnxruntime-gpu CUDA phù hợp; không fallback CPU."
        )
    from .app_config import SAM_ENCODER_ONNX_PATH, SAM_DECODER_ONNX_PATH
    for label, path in (("SAM encoder", SAM_ENCODER_ONNX_PATH),
                        ("SAM decoder", SAM_DECODER_ONNX_PATH)):
        if not Path(path).is_file():
            raise RuntimeError(f"Không tìm thấy {label}: {path}. Cần hai file SAM ONNX.")

    # Respect strict same-GPU semantics: CUDA_VISIBLE_DEVICES can remap indices.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible and visible != ','.join(str(i) for i in range(idx + 1)):
        raise RuntimeError(
            f'CUDA_VISIBLE_DEVICES={visible!r} có thể đổi thứ tự GPU. '
            'Hãy bỏ biến này và GPU Check lại để đảm bảo đúng GPU đã chọn.'
        )
    # GPU Check uses nvidia-smi; cross-check the selection's name and index.
    exe = shutil.which('nvidia-smi')
    if not exe and os.name == 'nt':
        candidate = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'NVIDIA Corporation/NVSMI/nvidia-smi.exe'
        if candidate.exists():
            exe = str(candidate)
    if not exe:
        raise RuntimeError('Không tìm thấy nvidia-smi để đối chiếu GPU. Không dùng CPU thay thế.')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    try:
        data = subprocess.check_output(
            [exe, '--query-gpu=index,name', '--format=csv,noheader,nounits'],
            timeout=15, creationflags=flags, text=True, errors='replace',
        )
        names = {}
        for line in data.splitlines():
            if ',' in line:
                gpu_idx, name = line.split(',', 1)
                names[int(gpu_idx.strip())] = name.strip()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'Không xác minh được NVIDIA CUDA bằng nvidia-smi: {exc}') from exc
    if idx not in names:
        raise RuntimeError(f'Không tìm thấy NVIDIA GPU {idx} trong nvidia-smi; hãy GPU Check lại.')
    name = names[idx]
    expected = selection.get('gpu_name') or ''
    def core(value):
        return re.sub(r'[^a-z0-9]', '', value.lower().replace('nvidia','').replace('geforce',''))
    if expected and core(expected) not in core(name) and core(name) not in core(expected):
        raise RuntimeError(f'GPU đã lưu ({expected}) không khớp NVIDIA GPU {idx} ({name}).')
    preload_cuda_runtime()
    return idx, name


class GpuScanThread(QThread):
    scanned = Signal(list)
    scanFailed = Signal(str)
    scanStopped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
        self._process_lock = threading.Lock()
        self._process = None

    def stop(self):
        self._stop.set()
        with self._process_lock:
            proc = self._process
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _command(self, cmd, timeout=35):
        if self._stop.is_set():
            return ""
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=flags)
        with self._process_lock:
            self._process = process
        start = time.monotonic()
        try:
            while process.poll() is None:
                if self._stop.is_set() or time.monotonic() - start > timeout:
                    process.kill()
                    process.communicate()
                    if self._stop.is_set():
                        return ""
                    raise TimeoutError("Quá thời gian dò GPU")
                time.sleep(0.08)
            out, err = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(err.decode("utf-8", errors="replace").strip() or f"Exit code {process.returncode}")
            return out.decode("utf-8-sig", errors="replace").strip()
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None

    @staticmethod
    def _core(name):
        return re.sub(r"[^a-z0-9]", "", name.lower().replace("nvidia", "").replace("geforce", ""))

    def _windows_adapters(self):
        if os.name != "nt":
            return []
        ps = shutil.which("powershell") or "powershell"
        script = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID | ConvertTo-Json -Compress"
        raw = self._command([ps, "-NoProfile", "-NonInteractive", "-Command", script])
        if not raw:
            return []
        result = json.loads(raw)
        data = result if isinstance(result, list) else [result]
        return [{"name": str(v.get("Name") or "GPU"), "pnp": str(v.get("PNPDeviceID") or "")} for v in data if isinstance(v, dict) and v.get("Name")]

    def _cuda_adapters(self):
        # nvidia-smi uses the CUDA/driver GPU index. Validate against PyTorch
        # again when loading the model (important if CUDA_VISIBLE_DEVICES differs).
        exe = shutil.which("nvidia-smi")
        if not exe and os.name == "nt":
            candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
            if candidate.exists():
                exe = str(candidate)
        if exe:
            try:
                raw = self._command([exe, "--query-gpu=index,name", "--format=csv,noheader,nounits"], timeout=15)
                devices = []
                for line in raw.splitlines():
                    index, name = line.split(",", 1)
                    devices.append({"id": f"cuda:{int(index.strip())}", "name": name.strip(), "compatible": True})
                return devices
            except (ValueError, RuntimeError, TimeoutError):
                pass
        # nvidia-smi is the authoritative CUDA index source here; avoid
        # importing PyTorch just to enumerate devices (not shipped in EXE).
        return []

    def run(self):
        try:
            adapters = []
            errors = []
            try:
                adapters = self._windows_adapters()
            except Exception as exc:
                errors.append(f"Windows GPU: {exc}")
            if self._stop.is_set():
                self.scanStopped.emit()
                return
            cuda = self._cuda_adapters()
            if self._stop.is_set():
                self.scanStopped.emit()
                return
            devices = list(cuda)
            matched = {v["pnp"] for v in adapters for c in cuda
                       if self._core(c["name"]) and self._core(c["name"]) in self._core(v["name"])}
            for adapter in adapters:
                if adapter["pnp"] in matched:
                    continue
                name = adapter["name"]
                if name.lower().startswith("microsoft basic"):
                    continue
                devices.append({"id": "adapter:" + (adapter["pnp"] or name), "name": name, "compatible": False})
            if not devices:
                self.scanFailed.emit("Tìm thất bại" + ((" — " + errors[0]) if errors else " — Không phát hiện GPU"))
            else:
                self.scanned.emit(devices)
        except Exception as exc:
            if self._stop.is_set():
                self.scanStopped.emit()
            else:
                self.scanFailed.emit(f"Tìm thất bại — {exc}")
