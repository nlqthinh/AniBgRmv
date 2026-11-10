"""The saved CPU/GPU choice and the CUDA device it means: one rule for the whole app.

Image engine (ONNX Runtime), video workers (another process), annotation tools
and the GPU Check page all read the choice here:
  frozen build  per-PC file in LOCALAPPDATA (portable_device_settings), so a
                release never carries the build machine's GPU;
  source mode   'compute' in app_config.json.
Strict GPU: an invalid GPU choice is an error, never a silent switch to CPU.
No Qt here (GpuScanThread lives in device_management).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .app_config import APP_CONFIG, get_external_config_path
from .portable_device_settings import build_variant, load_machine_selection, save_machine_selection


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
        raise RuntimeError("GPU chưa được chọn hoặc không phải NVIDIA CUDA. Chọn GPU NVIDIA bằng GPU Check.")
    return int(match.group(1))


def check_visible_devices(index: int) -> None:
    """CUDA_VISIBLE_DEVICES may renumber GPUs: allow only an unchanged order."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return
    parts = visible.replace(" ", "").split(",")
    if index >= len(parts) or any(not p.isdecimal() or int(p) != i for i, p in enumerate(parts)):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} có thể đổi thứ tự GPU. "
                           "Hãy bỏ biến này và GPU Check lại để đảm bảo đúng GPU đã chọn.")


def nvidia_smi_path() -> str | None:
    """nvidia-smi gives the CUDA/driver GPU index; no PyTorch import needed."""
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        candidate = (Path(os.environ.get("ProgramFiles", "C:/Program Files"))
                     / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe")
        if candidate.exists():
            exe = str(candidate)
    return exe


def gpu_name_core(name: str) -> str:
    """'NVIDIA GeForce GTX 1650' -> 'gtx1650', to compare names from different tools."""
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("nvidia", "").replace("geforce", ""))


def verify_gpu(selection):
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
    # GPU mode always offers HQ-SAM Guided Edit: its files must exist.
    from .model_registry import guided_model_assets
    for label, path in guided_model_assets():
        if not path.is_file():
            raise RuntimeError(f"Không tìm thấy {label}: {path}.")
    check_visible_devices(idx)
    # GPU Check uses nvidia-smi; cross-check the selection's name and index.
    exe = nvidia_smi_path()
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
    expected = gpu_name_core(selection.get('gpu_name') or '')
    actual = gpu_name_core(name)
    if expected and expected not in actual and actual not in expected:
        raise RuntimeError(f'GPU đã lưu ({selection.get("gpu_name")}) không khớp NVIDIA GPU {idx} ({name}).')
    # Same CUDA/cuDNN loader as the frozen self-test and the PyInstaller hook.
    from .cuda_bundle_runtime import preload_runtime
    preload_runtime()
    return idx, name
