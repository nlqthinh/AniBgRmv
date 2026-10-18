"""Check the Python environment used to run AniBgRmv (no models required).

Run with the SAME interpreter as app.py:
    python tools/diagnose_gpu.py
Or inside an active venv:
    .venv\\Scripts\\python.exe tools\\diagnose_gpu.py

This tool never installs software, edits config, or enables a CPU fallback.
"""

import os
import subprocess
import sys
import traceback


def line(label, value):
    print(f"{label}: {value}", flush=True)


def main():
    print("=== AniBgRmv GPU diagnosis (M35) ===", flush=True)
    line("Python executable", sys.executable)
    line("Python version", sys.version.split()[0])
    line("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        line("nvidia-smi", proc.stdout.strip() or proc.stderr.strip() or f"exit {proc.returncode}")
    except Exception as exc:
        line("nvidia-smi", f"unavailable ({type(exc).__name__}: {exc})")

    try:
        import torch
        line("torch version", torch.__version__)
        line("torch CUDA build", torch.version.cuda)
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        line("torch.cuda.is_available()", available)
        line("torch.cuda.device_count()", count)
        if torch.version.cuda is None:
            print("RESULT/TORCH: CPU-only PyTorch; install a CUDA build for strict GPU mode.")
        elif not available:
            print("RESULT/TORCH: CUDA PyTorch installed, but CUDA is not usable. Check driver/DLL/visibility.")
        for i in range(count if available else 0):
            try:
                name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                line(f"cuda:{i}", f"{name}, compute capability={capability}")
                x = torch.ones(16, device=f"cuda:{i}")
                torch.cuda.synchronize(i)
                line(f"cuda:{i} allocation + kernel", f"OK, sum={float(x.sum().item())}")
                del x
            except Exception as exc:
                line(f"cuda:{i} allocation + kernel", f"FAILED: {exc}")
    except Exception as exc:
        line("torch import/check", f"FAILED: {type(exc).__name__}: {exc}")

    try:
        import onnxruntime as ort
        line("onnxruntime version", ort.__version__)
        providers = ort.get_available_providers()
        line("ONNX available providers", providers)
        if "CUDAExecutionProvider" not in providers:
            print("RESULT/ONNX: CUDAExecutionProvider missing; install compatible onnxruntime-gpu (not CPU onnxruntime).")
        elif hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
                print("RESULT/ONNX: CUDA/cuDNN preload OK (actual model load still required).")
            except Exception as exc:
                line("ONNX CUDA/cuDNN DLL preload", f"FAILED: {type(exc).__name__}: {exc}")
        else:
            print("RESULT/ONNX: CUDA EP is listed; DLL loading will be verified by actual ONNX model initialization.")
    except Exception as exc:
        line("onnxruntime import/check", f"FAILED: {type(exc).__name__}: {exc}")

    try:
        from segment_anything import sam_model_registry
        line("SAM module", f"OK, registry={sorted(sam_model_registry.keys())}")
    except Exception as exc:
        line("SAM module", f"unavailable: {type(exc).__name__}: {exc}")
    print("=== End. Copy the full output when asking for help. ===")


if __name__ == "__main__":
    main()
