"""Persisted compute selection and conservative GPU detection for strict mode.

GPU mode intentionally supports CUDA only. Standard PyTorch SAM cannot use a
DirectML ONNX adapter; allowing DirectML here would silently split the models
across devices, contradicting the strict single-GPU choice.
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
    return normalized_selection(APP_CONFIG.get("compute", {}))


def save_selection(selection):
    """Update only 'compute' in external JSON; preserve user paths/settings."""
    selection = normalized_selection(selection)
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
    match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", selection.get("gpu_id") or "")
    if match is None:
        raise RuntimeError("GPU đã chọn chưa hỗ trợ chạy đồng thời ONNX CUDA và PyTorch SAM. Chọn GPU NVIDIA CUDA bằng GPU Check.")
    return int(match.group(1))


def verify_gpu_for_sam(selection):
    """Fail closed before creating any CUDA ONNX session."""
    idx = cuda_index(selection)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Thiếu PyTorch có hỗ trợ CUDA; không thể chạy SAM trên GPU đã chọn.") from exc
    # nvidia-smi/Windows only prove that a display adapter exists. They do
    # NOT prove that this Python interpreter has CUDA-capable PyTorch installed.
    compiled_cuda = getattr(torch.version, "cuda", None)
    if not compiled_cuda:
        raise RuntimeError(
            "Đã tìm thấy GPU NVIDIA qua Windows/nvidia-smi, nhưng PyTorch "
            f"{torch.__version__} hiện là bản CPU-only (torch.version.cuda=None). "
            "Hãy cài bản PyTorch CUDA vào đúng môi trường Python chạy app. "
            "Chạy 'python tools/diagnose_gpu.py' để xem chi tiết. "
            "Không chuyển sang CPU."
        )
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
    except Exception as exc:
        raise RuntimeError(
            f"PyTorch {torch.__version__} (CUDA build {compiled_cuda}) "
            f"không khởi tạo được CUDA: {exc}. Kiểm tra NVIDIA driver, "
            "CUDA_VISIBLE_DEVICES và chạy tools/diagnose_gpu.py. "
            "Không chuyển sang CPU."
        ) from exc
    if not available or idx >= count:
        raise RuntimeError(
            f"GPU cuda:{idx} có trong nvidia-smi nhưng PyTorch "
            f"{torch.__version__} (CUDA build {compiled_cuda}) "
            f"chỉ thấy {count} CUDA device(s); is_available={available}. "
            "Kiểm tra NVIDIA driver, CUDA_VISIBLE_DEVICES, khả năng hỗ trợ "
            "GTX 1650 của PyTorch và đúng môi trường Python. "
            "Chạy 'python tools/diagnose_gpu.py'. Không chuyển sang CPU."
        )
    try:
        device_name = torch.cuda.get_device_name(idx)
    except Exception as exc:
        raise RuntimeError(f"Không đọc được cuda:{idx} từ PyTorch: {exc}") from exc
    expected = selection.get("gpu_name") or ""
    # Names from WMI, nvidia-smi and PyTorch may have different vendor prefixes.
    def core(s):
        return re.sub(r"[^a-z0-9]", "", s.lower().replace("nvidia", "").replace("geforce", ""))
    if expected and core(expected) not in core(device_name) and core(device_name) not in core(expected):
        raise RuntimeError(f"GPU đã lưu ({expected}) khác GPU cuda:{idx} hiện tại ({device_name}). Hãy GPU Check lại.")
    # Validate SAM dependencies *before enabling any GPU image-processing tasks*.
    # The SAM weights themselves stay lazy-loaded (on first Guided Edit),
    # avoiding a large, needless startup VRAM reservation.
    from .app_config import SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE
    if not Path(SAM_CHECKPOINT_PATH).is_file():
        raise RuntimeError(f"Chế độ GPU bắt buộc cần SAM checkpoint: {SAM_CHECKPOINT_PATH}")
    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise RuntimeError("Thiếu segment_anything cho SAM; chế độ GPU không thể bật tất cả model.") from exc
    if SAM_MODEL_TYPE not in sam_model_registry:
        raise RuntimeError(f"SAM model_type không hợp lệ: {SAM_MODEL_TYPE}")
    # This also tests a real CUDA allocation; it is not merely checking a driver name.
    try:
        # Synchronize to catch asynchronous kernel/init errors before accepting.
        probe = torch.ones(16, device=f"cuda:{idx}")
        torch.cuda.synchronize(idx)
        del probe
    except Exception as exc:
        raise RuntimeError(f"Không thể thực thi phép toán CUDA trên GPU cuda:{idx}: {exc}") from exc
    return idx, device_name


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
        # For development installs without nvidia-smi, use torch in a child
        # interpreter so Stop can actually terminate the probe.
        if not getattr(sys, "frozen", False):
            script = ("import json,torch; print(json.dumps([{'id':f'cuda:{i}',"
                      "'name':torch.cuda.get_device_name(i),'compatible':True} "
                      "for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []))")
            try:
                raw = self._command([sys.executable, "-c", script], timeout=30)
                return json.loads(raw.splitlines()[-1]) if raw else []
            except (ValueError, RuntimeError, TimeoutError, IndexError):
                pass
        # A frozen EXE cannot launch sys.executable -c; probe torch in this
        # background thread instead. Stop is observed immediately afterward.
        if getattr(sys, "frozen", False) and not self._stop.is_set():
            try:
                import torch
                if torch.cuda.is_available():
                    return [{"id": f"cuda:{i}", "name": torch.cuda.get_device_name(i), "compatible": True}
                            for i in range(torch.cuda.device_count())]
            except (ImportError, RuntimeError, OSError):
                pass
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
