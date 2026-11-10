"""GPU Check: list the PC's GPUs in a background Qt thread.

The saved CPU/GPU choice and the strict CUDA checks are in compute_selection.
AMD/Intel GPUs are shown but marked not supported by this release.
"""
import json
import os
import shutil
import subprocess
import threading
import time

from PySide6.QtCore import QThread, Signal

from .compute_selection import gpu_name_core, nvidia_smi_path


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
        # nvidia-smi uses the CUDA/driver GPU index (verify_gpu re-checks it).
        exe = nvidia_smi_path()
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
        # nvidia-smi is the authoritative CUDA index source; no PyTorch import.
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
                       if gpu_name_core(c["name"]) and gpu_name_core(c["name"]) in gpu_name_core(v["name"])}
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
