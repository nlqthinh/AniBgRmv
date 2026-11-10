"""ONNX inference engine: automatic cut, Guided Edit and on-demand matting.

Automatic cut (Single/Batch):
  image -> 768 letterbox -> UNet alpha -> loose trimap -> ViTMatte-UNet
        -> crop padding -> resize to original size -> RGBA cutout
           (soft edges recoloured by foreground.py, output.foreground_estimation)

Guided Edit: HQ-SAM on selected device (+ ViTMatte-SAM on GPU), see guided_edit.py.
All sessions run on one device: strict CUDA (no CPU fallback) or CPU (onnx_device.py).
Settings and model files are read once, at creation (engine_settings.py).
"""
from pathlib import Path
import gc
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from . import matting_ops as ops
from .engine_settings import EngineSettings
from .foreground import estimate_foreground
from .guided_edit import GuidedEditMixin, GuidedPolicy
from .onnx_device import OnnxDevice
from .save_prediction import SavePredictionMixin


class BgRemoveEngine(GuidedEditMixin, SavePredictionMixin):

    def __init__(self, compute_selection: dict | None = None,
                 settings: EngineSettings | None = None):
        from ..app_config import APP_CONFIG
        from ..compute_selection import normalized_selection, verify_gpu
        from ..model_registry import guided_model_assets

        self.settings = settings or EngineSettings.from_config(APP_CONFIG)
        models = self.settings.models
        # Automatic models and Guided models share one engine. Serialize
        # inference so a CUDA session is never freed while another job uses it.
        self._inference_lock = threading.RLock()
        # (UNet, ViTMatte-UNet, ViTMatte on) per Settings mode.
        self._auto_profiles = {
            'single': (models['unet'], models['vitmatte_unet'], self.settings.image_vit),
            'batch': (models['batch_unet'], models['batch_vitmatte_unet'], self.settings.batch_vit),
        }
        self.unet_model_path, self.vitmatte_unet_model_path, self.use_image_vit = self._auto_profiles['single']
        self._active_auto_profile = 'single'
        assets = []
        for profile, (unet_path, matte_path, enabled) in self._auto_profiles.items():
            assets.append((f'{profile} UNet', unet_path))
            if enabled:
                assets.append((f'{profile} ViTMatte-UNet', matte_path))
        for label, path in assets:
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy {label} ONNX: {path}")

        self.compute_selection = normalized_selection(compute_selection)
        self.compute_mode = self.compute_selection["mode"]
        self.strict_gpu = self.compute_mode == "gpu"
        self.cpu_matting_model = self.settings.cpu_matting_model
        self.foreground_estimation = self.settings.foreground_estimation
        self.guided_policy = GuidedPolicy.for_device(
            self.strict_gpu, self.settings.small_object_crop,
            self.settings.refine_each_stroke and self.settings.image_sam_vit)

        if self.strict_gpu:
            index, gpu_name = verify_gpu(self.compute_selection)
            self.device = f"cuda:{index}"
            from ..gpu_memory_options import cuda_provider_options
            self._cuda_options = cuda_provider_options(index, settings=self.settings.compute)
        else:
            self.device = "cpu"
            self._cuda_options = None
            if "CPUExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime không có CPUExecutionProvider.")
        self._onnx = OnnxDevice(self.device, shrink_gpu_arena=self.settings.shrink_gpu_arena_after_run,
                                cuda_options=self._cuda_options)

        # CPU auto-cut must start even if optional Guided HQ-SAM assets are absent.
        # Guided Edit validates its assets lazily in _hqsam_predictor().
        if self.strict_gpu:
            for label, path in guided_model_assets():
                if not path.is_file():
                    raise FileNotFoundError(f"Không tìm thấy {label}: {path}")

        self._unet_sess = self._vitmatte_unet_sess = None
        self._vitmatte_sam_sess = None   # lazy: Guided Edit / CPU Matting only
        self._hqsam = None               # lazy: first HQ-SAM Guided stroke
        self._guided_image_key = None
        self._edit_source = None         # (file key, decoded RGB) of the image in Edit
        self._ensure_auto_sessions()

        if self.strict_gpu:
            self.runtime_info = (f"GPU {gpu_name} ({self.device}) — UNet/ViTMatte-UNet: CUDA, "
                                 "HQ-SAM + ViTMatte-SAM: CUDA (nạp khi Edit)")
        else:
            self.runtime_info = (f"CPU — UNet/ViTMatte-UNet: ONNX, Guided: HQ-SAM CPU; "
                                 f"Matting thủ công: {self.cpu_matting_model}")

    # ------------------------------------------------------------ sessions
    def _session_options(self) -> ort.SessionOptions:
        if not self.strict_gpu:
            from ..cpu_optimizations import cpu_session_options
            return cpu_session_options(self.settings.compute)
        options = ort.SessionOptions()
        # Model computation stays on CUDA: no host-side arena or memory pattern,
        # and ORT must fail instead of silently assigning nodes to CPU.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        return options

    def _new_session(self, label: str, path: Path) -> ort.InferenceSession:
        return self._onnx.session(path, label, options=self._session_options())

    def _select_auto_profile(self, profile: str) -> None:
        # Called only while the shared inference lock is held. Swap sessions
        # rather than loading four ONNX models simultaneously on a 4GB GPU.
        if profile not in self._auto_profiles:
            raise ValueError(f'Unknown automatic image profile: {profile}')
        if profile == self._active_auto_profile:
            return
        self._unet_sess = self._vitmatte_unet_sess = None
        (self.unet_model_path, self.vitmatte_unet_model_path,
         self.use_image_vit) = self._auto_profiles[profile]
        self._active_auto_profile = profile
        gc.collect()

    def _ensure_auto_sessions(self) -> None:
        """(Re)create UNet + ViTMatte-UNet together; never keep a half pair."""
        with self._inference_lock:
            if self._unet_sess is not None and (not self.use_image_vit or self._vitmatte_unet_sess is not None):
                return
            self._unet_sess = self._vitmatte_unet_sess = None
            unet = self._new_session("UNet", self.unet_model_path)
            matte = (self._new_session("ViTMatte-UNet", self.vitmatte_unet_model_path)
                     if self.use_image_vit else None)
            self._unet_input = unet.get_inputs()[0].name
            self._vitmatte_unet_input = matte.get_inputs()[0].name if matte else None
            self._unet_sess, self._vitmatte_unet_sess = unet, matte

    def release_auto_sessions(self) -> None:
        """GPU 4 GB: UNet/ViTMatte-UNet are idle while Guided Edit runs.

        They are recreated lazily on the next Single/Batch cut.
        """
        if not self.strict_gpu or not self.settings.unload_auto_models_during_guided:
            return
        with self._inference_lock:
            if self._unet_sess is None and self._vitmatte_unet_sess is None:
                return
            self._unet_sess = self._vitmatte_unet_sess = None
            gc.collect()

    def release_for_video_worker(self) -> None:
        """Before Track / GPU Export: the video worker is another process, so the
        image sessions here must not keep GPU memory (4 GB cards)."""
        self.release_guided_resources()
        self.release_auto_sessions()

    def _hqsam_predictor(self):
        if self._hqsam is None:
            from ..disk_cache import cached_sha256
            from .hqsam_onnx import HqSamOnnxPredictor
            self._hqsam = HqSamOnnxPredictor(
                self.settings.models['hqsam_encoder'], self.settings.models['hqsam_decoder'],
                self.device, digest=cached_sha256,
                cpu_threads=self.settings.compute.get("cpu_threads", 2),
                shrink_gpu_arena=self.settings.shrink_gpu_arena_after_run,
                log_timing=self.settings.log_timing, cuda_options=self._cuda_options)
            self._guided_image_key = None
        return self._hqsam

    def _vitmatte_sam_session(self):
        if self._vitmatte_sam_sess is None:
            path = self.settings.models['vitmatte_sam']
            if not path.is_file():
                raise FileNotFoundError(
                    f"Thiếu ViTMatte-SAM ONNX: {path}. Đặt checkpoint vào ani_bgrmv/models "
                    "rồi chạy bat/prepare_vitmatte_sam_onnx.bat.")
            session = self._new_session("ViTMatte-SAM", path)
            model_input, outputs = session.get_inputs(), session.get_outputs()
            expected = ops.model_input_shape(ops.MATTING_CHANNELS)
            if (len(model_input) != 1 or len(outputs) != 1 or model_input[0].type != "tensor(float)"
                    or tuple(model_input[0].shape) != expected):
                raise RuntimeError(f"ViTMatte-SAM ONNX sai contract; cần input float32 {list(expected)}.")
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
        # CPU's manual Matting button may still need M7 while the automatic
        # image pipeline has ViT = OFF. Load M7 only when requested.
        if self._vitmatte_unet_sess is None:
            self._vitmatte_unet_sess = self._new_session('ViTMatte-UNet', self.vitmatte_unet_model_path)
            self._vitmatte_unet_input = self._vitmatte_unet_sess.get_inputs()[0].name
        x = ops.vitmatte_input(image_letterbox_pil, trimap)
        y = np.asarray(self._vitmatte_unet_sess.run(None, {self._vitmatte_unet_input: x})[0])
        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            raise ValueError(f"ViTMatte-UNet ONNX output shape không hợp lệ: {y.shape}")
        return np.clip(y.astype(np.float32), 0.0, 1.0)

    def _predict_vitmatte_sam_letterbox(self, image_letterbox_pil, trimap) -> np.ndarray:
        """Same contract as ViTMatte-UNet (matting_ops), trained on HQ-SAM masks."""
        session = self._vitmatte_sam_session()
        x = ops.vitmatte_input(image_letterbox_pil, trimap)
        y = np.asarray(session.run(None, {self._vitmatte_sam_input: x}, self._onnx.run_options())[0])
        if y.shape != ops.ALPHA_OUTPUT_SHAPE:
            raise RuntimeError(f"ViTMatte-SAM ONNX output sai shape: {y.shape}, "
                               f"cần {list(ops.ALPHA_OUTPUT_SHAPE)}.")
        return np.clip(y[0, 0].astype(np.float32), 0.0, 1.0)

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
            self._select_auto_profile('single')
            report(5, "Đang chuẩn bị alpha hiện tại")
            alpha = np.asarray(current_alpha, dtype=np.uint8).copy()
            image = self.edit_source(image_path)
            if alpha.shape != (image.height, image.width):
                raise ValueError("Alpha hiện tại không khớp kích thước ảnh gốc.")

            refined = alpha
            if np.any(alpha == 0) and np.any(alpha > 0):  # there is an edge to matte
                report(20, "Đang tạo trimap từ toàn bộ mask")
                # Each model gets the letterbox and trimap of its own notebook.
                sam = model == "vitmatte_sam"
                letterbox, valid, meta = ops.letterbox_rgb(
                    image, interpolation=ops.SAM_MATTING_INTERPOLATION if sam else None)
                square = ops.to_letterbox(alpha, meta, cv2.INTER_LINEAR)
                if sam:
                    trimap = ops.mask_trimap(ops.to_letterbox(alpha >= 128, meta, cv2.INTER_NEAREST))
                    predict = self._predict_vitmatte_sam_letterbox
                else:
                    trimap = ops.loose_trimap(square.astype(np.float32) / 255.0, valid)
                    predict = self._predict_vitmatte_unet_letterbox
                report(45, f"Đang chạy {model} trên CPU")
                soft = np.rint(predict(letterbox, trimap) * 255.0).astype(np.uint8)
                # Protect the definite interior of finished Erase/Restore edits;
                # a small kernel keeps intentionally drawn small marks/holes.
                core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                soft[cv2.erode((square == 0).astype(np.uint8), core) > 0] = 0
                soft[cv2.erode((square >= 254).astype(np.uint8), core) > 0] = 255
                report(85, "Đang trả alpha về kích thước ảnh gốc")
                refined = ops.restore_from_letterbox(soft, meta)
                refined[cv2.erode((alpha == 0).astype(np.uint8), core) > 0] = 0
                refined[cv2.erode((alpha >= 254).astype(np.uint8), core) > 0] = 255

            report(95, "Đang cập nhật preview")
            elapsed = time.perf_counter() - started
            prediction = self.build_prediction_from_alpha(
                image_path=image_path, final_alpha=refined, elapsed_sec=elapsed,
                source=f"cpu_matting_{model}")
            prediction["matting_info"] = {
                "model": model, "elapsed_sec": elapsed,
                "changed_pixels": int(np.count_nonzero(refined != alpha)),
            }
        return prediction

    def predict_alpha(self, image_rgb: Image.Image, include_debug: bool = True,
                      profile: str = 'single') -> dict:
        """``image_rgb`` comes from ops.load_rgb(); the caller reuses it for the cutout."""
        with self._inference_lock:
            self._select_auto_profile(profile)
            return self._predict_alpha_locked(image_rgb, include_debug=include_debug)

    def _predict_alpha_locked(self, image_rgb: Image.Image, include_debug: bool) -> dict:
        """F3 -> M7 exactly as the notebooks: UNet alpha -> loose trimap -> ViTMatte-UNet."""
        letterbox, valid_768, meta = ops.letterbox_rgb(image_rgb)
        unet_alpha = ops.unet_alpha(self._predict_unet_raw_letterbox(letterbox))
        trimap = ops.loose_trimap(unet_alpha, valid_768)
        matte = (self._predict_vitmatte_unet_letterbox(letterbox, trimap)
                 if self.use_image_vit else unet_alpha)

        result = {
            "final_alpha": ops.restore_from_letterbox(matte, meta),
            "letterbox_meta": meta,
            "trimap_stats": ops.trimap_ratios(trimap),
            "trimap_stats_valid": ops.trimap_ratios(trimap, valid_768),
        }
        if not include_debug:
            return result  # no extra full-size debug arrays in Release/Batch

        def to_uint8(x: np.ndarray) -> np.ndarray:
            return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

        result.update({
            "unet_alpha_768": to_uint8(unet_alpha),
            "unet_alpha_original": ops.restore_from_letterbox(unet_alpha, meta),
            "image_letterbox_pil": letterbox,
            "valid_mask_768": valid_768,
            "trimap_768": trimap,
            "matte_alpha_768": to_uint8(matte),
        })
        return result

    # ------------------------------------------------------- presentation
    def make_cutout(self, image_rgb: Image.Image, alpha: np.ndarray) -> Image.Image:
        """RGBA cutout from the already decoded source (no file read here).

        With output.foreground_estimation, soft-edge pixels get their estimated
        foreground colour (foreground.py); every other pixel keeps the source.
        """
        alpha = np.asarray(alpha, dtype=np.uint8)
        if alpha.shape[:2] != (image_rgb.height, image_rgb.width):
            alpha = cv2.resize(alpha, image_rgb.size, interpolation=cv2.INTER_LINEAR)
        if self.foreground_estimation:
            image_rgb = Image.fromarray(estimate_foreground(
                np.asarray(image_rgb), alpha.astype(np.float32) / 255.0), "RGB")
        # putalpha avoids allocating another whole-image RGBA numpy copy.
        rgba = image_rgb.convert("RGBA")
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
        # Broadcast two 1-D parities: an 8-bit H x W plane, no full-size int64 grids.
        odd = ((np.arange(h) // tile) % 2).astype(bool)[:, None] ^ ((np.arange(w) // tile) % 2).astype(bool)[None, :]
        shade = np.where(odd, 255, 220).astype(np.uint8)
        image = Image.fromarray(shade, mode="L").convert("RGBA")
        image.alpha_composite(cutout_pil)
        return image.convert("RGB")

    def build_prediction_from_alpha(self, image_path: str, final_alpha: np.ndarray,
                                    elapsed_sec: float = 0.0, source: str = "edited_alpha") -> dict:
        """Prediction dict for an externally edited alpha (Manual/Guided/Matting).

        Genuine model intermediates stay on the original prediction instead of
        being invented here. The source comes from the Edit cache (edit_source).
        """
        image_path = str(image_path)
        image = self.edit_source(image_path)
        alpha = np.asarray(final_alpha).astype(np.uint8)
        if alpha.shape[:2] != (image.height, image.width):
            alpha = cv2.resize(alpha, image.size, interpolation=cv2.INTER_LINEAR)
        return {"input_path": image_path, "final_alpha": alpha,
                "cutout_pil": self.make_cutout(image, alpha),
                "elapsed_sec": elapsed_sec, "safety_notes": [source]}

    def process_for_preview(self, image_path: str, include_debug: bool = True,
                            profile: str = 'single') -> dict:
        """Automatic cut; the source is decoded once for both model and cutout."""
        started = time.time()
        image_path = str(image_path)
        image = ops.load_rgb(image_path)
        prediction = self.predict_alpha(image, include_debug=include_debug, profile=profile)
        prediction.update(input_path=image_path,
                          cutout_pil=self.make_cutout(image, prediction["final_alpha"]),
                          safety_notes=["letterbox_768_unet_vitmatte"])
        prediction["elapsed_sec"] = time.time() - started
        return prediction
