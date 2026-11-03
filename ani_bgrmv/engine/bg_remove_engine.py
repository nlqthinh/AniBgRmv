"""ONNX inference engine: automatic cut, Guided Edit and on-demand matting.

Automatic cut (Single/Batch):
  image -> 768 letterbox -> UNet alpha -> loose trimap -> ViTMatte-UNet
        -> crop padding -> resize to original size -> RGBA cutout

Guided Edit: HQ-SAM (+ ViTMatte-SAM on GPU) or GrabCut, see guided_edit.py.
All sessions run on one device: strict CUDA (no CPU fallback) or CPU.
"""
from pathlib import Path
import gc
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from . import matting_ops as ops
from .guided_edit import GuidedEditMixin, GuidedPolicy
from .save_prediction import SavePredictionMixin


class BgRemoveEngine(GuidedEditMixin, SavePredictionMixin):

    def __init__(self, unet_model_path: str, vitmatte_unet_model_path: str,
                 size: int = ops.LETTERBOX_SIZE, compute_selection: dict | None = None):
        from ..app_config import APP_CONFIG, guided_model_assets
        from ..device_management import normalized_selection, verify_gpu

        # Automatic models and Guided models share one engine. Serialize
        # inference so a CUDA session is never freed while another job uses it.
        self._inference_lock = threading.RLock()
        self.size = int(size)
        self.unet_model_path = self._resolve_asset_path(unet_model_path)
        self.vitmatte_unet_model_path = self._resolve_asset_path(vitmatte_unet_model_path)
        for label, path in (("UNet", self.unet_model_path),
                            ("ViTMatte-UNet", self.vitmatte_unet_model_path)):
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy {label} ONNX: {path}")

        self.compute_selection = normalized_selection(compute_selection)
        self.compute_mode = self.compute_selection["mode"]
        self.strict_gpu = self.compute_mode == "gpu"
        sam_config = APP_CONFIG["sam"]
        self.cpu_matting_model = str(sam_config["cpu_matting_model"]).lower()
        self.guided_policy = GuidedPolicy.for_device(
            self.strict_gpu, str(sam_config["cpu_guided_backend"]).lower(), sam_config)

        if self.strict_gpu:
            index, gpu_name = verify_gpu(self.compute_selection)
            self.device = f"cuda:{index}"
            from ..gpu_memory_options import cuda_provider_options
            self._cuda_options = cuda_provider_options(index, settings=APP_CONFIG["compute"])
            self.providers = [("CUDAExecutionProvider", self._cuda_options)]
        else:
            self.device = "cpu"
            self._cuda_options = None
            self.providers = ["CPUExecutionProvider"]
            if "CPUExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime không có CPUExecutionProvider.")

        if self.guided_policy.selector == "hqsam":
            for label, path in guided_model_assets():
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Không tìm thấy {label}: {path}")

        self._unet_sess = self._vitmatte_unet_sess = None
        self._vitmatte_sam_sess = None   # lazy: Guided Edit / CPU Matting only
        self._hqsam = None               # lazy: first HQ-SAM Guided stroke
        self._guided_image_key = None
        self._ensure_auto_sessions()

        if self.strict_gpu:
            self.runtime_info = (f"GPU {gpu_name} ({self.device}) — UNet/ViTMatte-UNet: CUDA, "
                                 "HQ-SAM + ViTMatte-SAM: CUDA (nạp khi Edit)")
        elif self.guided_policy.selector == "grabcut":
            self.runtime_info = (f"CPU — UNet/ViTMatte-UNet: ONNX, Guided: GrabCut; "
                                 f"Matting thủ công: {self.cpu_matting_model}")
        else:
            self.runtime_info = (f"CPU — UNet/ViTMatte-UNet: ONNX, Guided: HQ-SAM CPU; "
                                 f"Matting thủ công: {self.cpu_matting_model}")

    # ------------------------------------------------------------ sessions
    @staticmethod
    def _resolve_asset_path(relative_or_abs_path: str) -> Path:
        p = Path(relative_or_abs_path)
        if p.is_absolute() and p.exists():
            return p
        here = Path(__file__).resolve().parent
        for candidate in (Path.cwd() / p, here / p,
                          Path(getattr(sys, "_MEIPASS", here)) / p,
                          Path(sys.executable).resolve().parent / p):
            if candidate.exists():
                return candidate
        return Path.cwd() / p

    def _session_options(self) -> ort.SessionOptions:
        if not self.strict_gpu:
            from ..cpu_optimizations import cpu_session_options
            return cpu_session_options()
        options = ort.SessionOptions()
        # Model computation stays on CUDA: no host-side arena or memory pattern,
        # and ORT must fail instead of silently assigning nodes to CPU.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        return options

    def _new_session(self, label: str, path: Path) -> ort.InferenceSession:
        try:
            session = ort.InferenceSession(str(path), sess_options=self._session_options(),
                                           providers=self.providers)
        except Exception as exc:
            reason = str(exc)
            if "INVALID_PROTOBUF" in reason or "Protobuf parsing failed" in reason:
                raise RuntimeError(
                    f"{label} ONNX không đọc được (INVALID_PROTOBUF): {path}. "
                    "File model bị hỏng hoặc copy chưa xong; hãy export/copy lại. "
                    f"Chi tiết: {exc}") from exc
            if self.strict_gpu:
                raise RuntimeError(
                    f"{label} không khởi tạo được CUDAExecutionProvider: {exc}. "
                    "Kiểm tra DLL CUDA/cuDNN và VRAM; app không chuyển sang CPU.") from exc
            raise
        if self.strict_gpu:
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"{label} không chạy CUDAExecutionProvider; không fallback CPU.")
            session.disable_fallback()
        elif session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"{label} ở chế độ CPU phải dùng CPUExecutionProvider.")
        return session

    def _ensure_auto_sessions(self) -> None:
        """(Re)create UNet + ViTMatte-UNet together; never keep a half pair."""
        with self._inference_lock:
            if self._unet_sess is not None and self._vitmatte_unet_sess is not None:
                return
            self._unet_sess = self._vitmatte_unet_sess = None
            unet = self._new_session("UNet", self.unet_model_path)
            matte = self._new_session("ViTMatte-UNet", self.vitmatte_unet_model_path)
            self._unet_input = unet.get_inputs()[0].name
            self._vitmatte_unet_input = matte.get_inputs()[0].name
            self._unet_sess, self._vitmatte_unet_sess = unet, matte

    def release_auto_sessions(self) -> None:
        """GPU 4 GB: UNet/ViTMatte-UNet are idle while Guided Edit runs.

        They are recreated lazily on the next Single/Batch cut.
        """
        from ..app_config import APP_CONFIG
        if not self.strict_gpu or not APP_CONFIG["sam"]["unload_auto_models_during_guided"]:
            return
        with self._inference_lock:
            if self._unet_sess is None and self._vitmatte_unet_sess is None:
                return
            self._unet_sess = self._vitmatte_unet_sess = None
            gc.collect()

    def _hqsam_predictor(self):
        if self._hqsam is None:
            from ..app_config import (APP_CONFIG, HQSAM_DECODER_ONNX_PATH,
                                      HQSAM_ENCODER_ONNX_PATH)
            from ..disk_cache import cached_sha256
            from .hqsam_onnx import HqSamOnnxPredictor
            sam_config = APP_CONFIG["sam"]
            self._hqsam = HqSamOnnxPredictor(
                self._resolve_asset_path(HQSAM_ENCODER_ONNX_PATH),
                self._resolve_asset_path(HQSAM_DECODER_ONNX_PATH),
                self.device, digest=cached_sha256,
                cpu_threads=APP_CONFIG["compute"].get("cpu_threads", 2),
                shrink_gpu_arena=sam_config["shrink_gpu_arena_after_run"],
                log_timing=sam_config["log_timing"], cuda_options=self._cuda_options)
            self._guided_image_key = None
        return self._hqsam

    def _vitmatte_sam_session(self):
        if self._vitmatte_sam_sess is None:
            from ..app_config import VITMATTE_SAM_ONNX_PATH
            path = self._resolve_asset_path(VITMATTE_SAM_ONNX_PATH)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Thiếu ViTMatte-SAM ONNX: {path}. Đặt checkpoint vào ani_bgrmv/models "
                    "rồi chạy bat/prepare_vitmatte_sam_onnx.bat.")
            session = self._new_session("ViTMatte-SAM", path)
            model_input, outputs = session.get_inputs(), session.get_outputs()
            if (len(model_input) != 1 or len(outputs) != 1 or model_input[0].type != "tensor(float)"
                    or model_input[0].shape != [1, 4, self.size, self.size]):
                raise RuntimeError("ViTMatte-SAM ONNX sai contract; cần input float32 [1,4,768,768].")
            self._vitmatte_sam_input = model_input[0].name
            self._vitmatte_sam_sess = session
        return self._vitmatte_sam_sess

    # ---------------------------------------------------------- inference
    def _predict_unet_raw_letterbox(self, image_letterbox_pil: Image.Image) -> np.ndarray:
        self._ensure_auto_sessions()
        y = np.asarray(self._unet_sess.run(None, {self._unet_input: ops.unet_input(image_letterbox_pil)})[0])
        if y.ndim == 4:
            return y[0, 0].astype(np.float32)
        if y.ndim == 3:
            return y[0].astype(np.float32)
        raise ValueError(f"UNet ONNX output shape không hợp lệ: {y.shape}")

    def _predict_vitmatte_unet_letterbox(self, image_letterbox_pil, trimap) -> np.ndarray:
        self._ensure_auto_sessions()
        x = ops.vitmatte_input(image_letterbox_pil, trimap, self.size)
        y = np.asarray(self._vitmatte_unet_sess.run(None, {self._vitmatte_unet_input: x})[0])
        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            raise ValueError(f"ViTMatte-UNet ONNX output shape không hợp lệ: {y.shape}")
        return np.clip(y.astype(np.float32), 0.0, 1.0)

    def _predict_vitmatte_sam_letterbox(self, image_letterbox_pil, trimap) -> np.ndarray:
        """Same 4-channel 768x768 contract as ViTMatte-UNet, trained on HQ-SAM masks."""
        from ..app_config import APP_CONFIG
        session = self._vitmatte_sam_session()
        x = ops.vitmatte_input(image_letterbox_pil, trimap, self.size)
        run_options = None
        if self.strict_gpu and APP_CONFIG["sam"]["shrink_gpu_arena_after_run"]:
            run_options = ort.RunOptions()
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage",
                                             "gpu:" + self.device.split(":", 1)[1])
        y = np.asarray(session.run(None, {self._vitmatte_sam_input: x}, run_options)[0])
        if y.shape != (1, 1, self.size, self.size):
            raise RuntimeError(f"ViTMatte-SAM ONNX output sai shape: {y.shape}, cần [1,1,768,768].")
        return np.clip(y[0, 0].astype(np.float32), 0.0, 1.0)

    # Guided Edit helpers expected by GuidedEditMixin.
    def _letterbox_rgb(self, image_pil):
        return ops.letterbox_rgb(image_pil, self.size)

    @staticmethod
    def _restore_alpha_from_letterbox(alpha_square, meta):
        return ops.restore_from_letterbox(alpha_square, meta)

    def apply_guided_edit(self, *args, **kwargs):
        with self._inference_lock:
            return super().apply_guided_edit(*args, **kwargs)

    def matte_current_alpha(self, image_path: str, current_alpha: np.ndarray,
                            progress_callback=None) -> dict:
        """CPU Matting button: one ViTMatte pass on the whole edited alpha.

        CPU Guided strokes are binary (fast). After several strokes the user
        runs this once. The current alpha is the trimap source (UNet is not
        rerun); definite interior background/foreground stays protected.
        """
        if self.strict_gpu:
            raise RuntimeError("Matting thủ công chỉ dùng ở chế độ CPU.")
        model = self.cpu_matting_model
        if model == "off":
            raise RuntimeError("Matting đang tắt; đặt sam.cpu_matting_model trong app_config.json.")

        def report(pct, msg):
            if progress_callback is not None:
                progress_callback(pct, msg)

        started = time.perf_counter()
        with self._inference_lock:
            report(5, "Đang chuẩn bị alpha hiện tại")
            alpha = np.asarray(current_alpha, dtype=np.uint8).copy()
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            if alpha.shape != (image.height, image.width):
                raise ValueError("Alpha hiện tại không khớp kích thước ảnh gốc.")

            refined = alpha
            if np.any(alpha == 0) and np.any(alpha > 0):  # there is an edge to matte
                report(20, "Đang tạo trimap từ toàn bộ mask")
                letterbox, valid, meta = self._letterbox_rgb(image)
                rw, rh = meta["resized_width"], meta["resized_height"]
                px, py = meta["pad_left"], meta["pad_top"]
                square = np.zeros((self.size, self.size), dtype=np.uint8)
                square[py:py + rh, px:px + rw] = cv2.resize(alpha, (rw, rh), interpolation=cv2.INTER_LINEAR)
                trimap = ops.loose_trimap(square.astype(np.float32) / 255.0, valid)
                report(45, f"Đang chạy {model} trên CPU")
                predict = (self._predict_vitmatte_sam_letterbox if model == "vitmatte_sam"
                           else self._predict_vitmatte_unet_letterbox)
                soft = np.rint(predict(letterbox, trimap) * 255.0).astype(np.uint8)
                # Protect the definite interior of finished Erase/Restore edits;
                # a small kernel keeps intentionally drawn small marks/holes.
                core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                soft[cv2.erode((square == 0).astype(np.uint8), core) > 0] = 0
                soft[cv2.erode((square >= 254).astype(np.uint8), core) > 0] = 255
                report(85, "Đang trả alpha về kích thước ảnh gốc")
                refined = self._restore_alpha_from_letterbox(soft, meta)
                refined[cv2.erode((alpha == 0).astype(np.uint8), core) > 0] = 0
                refined[cv2.erode((alpha >= 254).astype(np.uint8), core) > 0] = 255

            report(95, "Đang cập nhật preview")
            elapsed = time.perf_counter() - started
            prediction = self.build_prediction_from_alpha(
                image_path=image_path, final_alpha=refined, elapsed_sec=elapsed,
                source=f"cpu_matting_{model}", include_debug=False)
            prediction["matting_info"] = {
                "model": model, "elapsed_sec": elapsed,
                "changed_pixels": int(np.count_nonzero(refined != alpha)),
            }
        return prediction

    def predict_alpha(self, image_path: str, include_debug: bool = True) -> dict:
        with self._inference_lock:
            return self._predict_alpha_locked(image_path, include_debug=include_debug)

    def _predict_alpha_locked(self, image_path: str, include_debug: bool) -> dict:
        image_pil = Image.open(image_path).convert("RGB")
        letterbox, valid_768, meta = self._letterbox_rgb(image_pil)
        valid = valid_768 > 0

        unet = ops.unet_alpha_for_trimap(self._predict_unet_raw_letterbox(letterbox), valid)
        trimap = unet["trimap"]
        matte_raw = self._predict_vitmatte_unet_letterbox(letterbox, trimap)

        # Guard: calibrate only a visibly low-contrast ViTMatte output, so
        # normal soft edges are never over-sharpened.
        matte_fixed, matte_fix_stats = ops.stretch_if_low_contrast(matte_raw, valid, "vitmatte_output")
        matte_stats = ops.array_stats(matte_raw, valid)
        use_matte_fix = bool(matte_fix_stats.get("applied")) and (
            (matte_stats["p99"] is not None and matte_stats["p99"] < 0.88)
            or (matte_stats["p01"] is not None and matte_stats["p01"] > 0.10))
        matte = matte_fixed if use_matte_fix else matte_raw

        result = {
            "final_alpha": self._restore_alpha_from_letterbox(matte, meta),
            "letterbox_meta": meta,
            "trimap_stats": ops.trimap_ratios(trimap),
            "trimap_stats_valid": unet["trimap_ratios_valid"],
            "alpha_fix_stats": {"unet_alpha_mode": unet["mode"], "unet_fix": unet["fix_stats"],
                                "matte_fix_used_for_final": use_matte_fix},
        }
        if not include_debug:
            return result  # no extra full-size debug arrays in Release/Batch

        def to_uint8(x: np.ndarray) -> np.ndarray:
            return (np.clip(x, 0.0, 1.0) * 255.0).round().clip(0, 255).astype(np.uint8)

        result["alpha_fix_stats"].update(unet_raw_stats=unet["raw_stats"],
                                         matte_raw_stats=matte_stats, matte_fix=matte_fix_stats)
        result.update({
            "unet_raw_output_768": to_uint8(np.clip(unet["raw_output"], 0.0, 1.0)),
            "unet_alpha_before_fix_768": to_uint8(unet["alpha_before_fix"]),
            "unet_alpha_768": to_uint8(unet["alpha_used"]),
            "unet_alpha_original": self._restore_alpha_from_letterbox(unet["alpha_used"], meta),
            "image_letterbox_pil": letterbox,
            "valid_mask_768": valid_768,
            "trimap_768": trimap,
            "matte_raw_alpha_768": to_uint8(matte_raw),
            "matte_alpha_768": to_uint8(matte),
        })
        return result

    # ------------------------------------------------------- presentation
    def make_cutout(self, image_path: str, alpha: np.ndarray) -> Image.Image:
        # putalpha avoids allocating another whole-image RGBA numpy copy.
        with Image.open(image_path) as source:
            rgba = source.convert("RGBA")
        alpha = np.asarray(alpha, dtype=np.uint8)
        if alpha.shape[:2] != (rgba.height, rgba.width):
            alpha = cv2.resize(alpha, rgba.size, interpolation=cv2.INTER_LINEAR)
        rgba.putalpha(Image.fromarray(alpha, "L"))
        return rgba

    @staticmethod
    def make_preview_white(cutout_pil: Image.Image) -> Image.Image:
        cutout_pil = cutout_pil.convert("RGBA")
        background = Image.new("RGBA", cutout_pil.size, (255, 255, 255, 255))
        background.alpha_composite(cutout_pil)
        return background.convert("RGB")

    @staticmethod
    def make_preview_checker(cutout_pil: Image.Image, tile: int = 24) -> Image.Image:
        cutout_pil = cutout_pil.convert("RGBA")
        w, h = cutout_pil.size
        ys, xs = np.mgrid[0:h, 0:w]
        shade = np.where(((xs // tile) + (ys // tile)) % 2 == 0, 220, 255).astype(np.uint8)
        checker = np.dstack([shade, shade, shade, np.full_like(shade, 255)])
        image = Image.fromarray(checker, mode="RGBA")
        image.alpha_composite(cutout_pil)
        return image.convert("RGB")

    def build_prediction_from_alpha(self, image_path: str, final_alpha: np.ndarray,
                                    elapsed_sec: float = 0.0, source: str = "edited_alpha",
                                    include_debug: bool = True) -> dict:
        """Prediction dict for an externally edited alpha (Manual/Guided/Matting).

        Debug previews are added only on request; genuine model intermediates
        stay on the original prediction instead of being invented here.
        """
        image_path = str(image_path)
        with Image.open(image_path) as source_image:
            size = source_image.size
        alpha = np.asarray(final_alpha).astype(np.uint8)
        if alpha.shape[:2] != (size[1], size[0]):
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)
        cutout = self.make_cutout(image_path, alpha)
        prediction = {"input_path": image_path, "final_alpha": alpha, "cutout_pil": cutout,
                      "elapsed_sec": elapsed_sec, "safety_notes": [source]}
        if include_debug:
            prediction["preview_checker_pil"] = self.make_preview_checker(cutout)
            prediction["preview_white_pil"] = self.make_preview_white(cutout)
        return prediction

    def process_for_preview(self, image_path: str, include_debug: bool = True) -> dict:
        started = time.time()
        image_path = str(image_path)
        prediction = self.predict_alpha(image_path, include_debug=include_debug)
        cutout = self.make_cutout(image_path, prediction["final_alpha"])
        prediction.update(input_path=image_path, cutout_pil=cutout,
                          safety_notes=["letterbox_768_unet_vitmatte"])
        if include_debug:
            prediction["preview_checker_pil"] = self.make_preview_checker(cutout)
            prediction["preview_white_pil"] = self.make_preview_white(cutout)
        prediction["elapsed_sec"] = time.time() - started
        return prediction
