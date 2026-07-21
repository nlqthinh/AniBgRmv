from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .sam_guided_edit import SamGuidedEditMixin
from .save_prediction import SavePredictionMixin


class BgRemoveEngine(SamGuidedEditMixin, SavePredictionMixin):
    """
    ONNX inference engine for AniBGRmv.

    M32 / M7 pipeline:
    image
    -> letterbox constant 768x768, RGB pad 114
    -> F3 U-Net ONNX auto alpha on letterbox image
    -> loose trimap from auto alpha, with padding forced to background
    -> M7 ViTMatte ONNX alpha on letterbox image + trimap
    -> crop padding away from ViTMatte alpha
    -> resize alpha to original image size
    -> RGBA cutout

    This follows the deploy helper in M7_ViTMatte_letterbox_constant Cell 27.
    """

    def __init__(
        self,
        f2a1_model_path: str,
        m6_model_path: str,
        size: int = 768,
        providers: list[str] | None = None,
        enforce_trimap_known_regions: bool = False,
    ):
        self.size = int(size)
        self.enforce_trimap_known_regions = enforce_trimap_known_regions

        self.f2a1_model_path = self._resolve_asset_path(f2a1_model_path)
        self.m6_model_path = self._resolve_asset_path(m6_model_path)

        if not self.f2a1_model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy U-Net/F3 ONNX: {self.f2a1_model_path}")
        if not self.m6_model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ViTMatte/M7 ONNX: {self.m6_model_path}")

        self.providers = providers or self._default_providers()

        self.f2_sess = ort.InferenceSession(str(self.f2a1_model_path), providers=self.providers)
        self.m6_sess = ort.InferenceSession(str(self.m6_model_path), providers=self.providers)

        self.f2_input_name = self.f2_sess.get_inputs()[0].name
        self.m6_input_name = self.m6_sess.get_inputs()[0].name

        self.runtime_info = " / ".join(self.f2_sess.get_providers())

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.letterbox_image_pad_mode = "constant"
        self.letterbox_image_pad_value = 114
        self.letterbox_alpha_pad_value = 0
        self.unet_preprocess_name = "m7_letterbox_constant"
        self.vitmatte_preprocess_name = "m7_letterbox_constant"

    def _resolve_asset_path(self, relative_or_abs_path: str) -> Path:
        p = Path(relative_or_abs_path)

        if p.is_absolute() and p.exists():
            return p

        candidates = [
            Path.cwd() / p,
            Path(__file__).resolve().parent / p,
            Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / p,
            Path(sys.executable).resolve().parent / p,
        ]

        for c in candidates:
            if c.exists():
                return c

        return Path.cwd() / p

    def _default_providers(self) -> list[str]:
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        selected = [p for p in preferred if p in available]
        return selected or available

    def _array_stats(self, x: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
        values = np.asarray(x, dtype=np.float32)
        if valid_mask is not None:
            valid = np.asarray(valid_mask).astype(bool)
            if valid.shape == values.shape and valid.any():
                values = values[valid]

        finite = np.isfinite(values)
        if not finite.any():
            return {
                "min": None,
                "max": None,
                "mean": None,
                "p01": None,
                "p05": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "dynamic_p01_p99": None,
            }

        values = values[finite]
        p01, p05, p50, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "p01": float(p01),
            "p05": float(p05),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "dynamic_p01_p99": float(p99 - p01),
        }

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return 1.0 / (1.0 + np.exp(-x))

    def _output_to_alpha_candidates(self, x: np.ndarray) -> dict:
        """
        Notebook M7 Cell 27 does torch.sigmoid(f3_logits). In app/ONNX,
        different exports may return raw logits or already-sigmoided alpha.
        Keep both candidates so the production path can choose the one that
        produces a sane M7 trimap instead of blindly trusting a low-contrast mask.
        """
        x = np.asarray(x, dtype=np.float32)

        finite = np.isfinite(x)
        if not finite.any():
            zero = np.zeros_like(x, dtype=np.float32)
            return {
                "raw": zero,
                "direct": zero,
                "sigmoid": zero,
                "raw_stats": self._array_stats(zero),
            }

        raw = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        direct = np.clip(raw, 0.0, 1.0)
        sigmoid = np.clip(self._sigmoid(raw), 0.0, 1.0)

        return {
            "raw": raw,
            "direct": direct,
            "sigmoid": sigmoid,
            "raw_stats": self._array_stats(raw),
        }

    def _trimap_stats_on_valid(self, trimap: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
        arr = np.asarray(trimap)
        if valid_mask is not None:
            valid = np.asarray(valid_mask).astype(bool)
            if valid.shape == arr.shape and valid.any():
                arr = arr[valid]

        total = max(1, int(arr.size))
        return {
            "trimap_bg_ratio_valid": float((arr == 0).sum() / total),
            "trimap_unknown_ratio_valid": float((arr == 128).sum() / total),
            "trimap_fg_ratio_valid": float((arr == 255).sum() / total),
        }

    def _stretch_alpha_for_trimap_if_needed(
        self,
        alpha_01: np.ndarray,
        valid_mask: np.ndarray | None,
        reason_prefix: str,
    ) -> tuple[np.ndarray, dict]:
        """
        Safety guard for ONNX deployment.

        If F3 alpha is low-contrast, M7 loose trimap can mark nearly the whole
        valid image as unknown. That matches the user's observed 'gray mask'
        failure and makes ViTMatte return a translucent/foggy cutout.

        This function only rescales the F3 alpha used to build the trimap.
        It does not change image letterbox geometry, ViTMatte input image, or the
        final crop/resize logic from M7.
        """
        alpha = np.clip(alpha_01.astype(np.float32), 0.0, 1.0)
        valid = None
        values = alpha.reshape(-1)

        if valid_mask is not None:
            valid = np.asarray(valid_mask).astype(bool)
            if valid.shape == alpha.shape and valid.any():
                values = alpha[valid]
            else:
                valid = None

        stats_before = self._array_stats(alpha, valid)
        p01 = stats_before.get("p01")
        p99 = stats_before.get("p99")
        dynamic = stats_before.get("dynamic_p01_p99")

        if p01 is None or p99 is None or dynamic is None or dynamic < 1e-6:
            return alpha, {
                "applied": False,
                "reason": f"{reason_prefix}_not_enough_dynamic_range",
                "stats_before": stats_before,
                "stats_after": stats_before,
            }

        trimap_before = self.make_loose_trimap(alpha, valid_mask=valid)
        trimap_stats_before = self._trimap_stats_on_valid(trimap_before, valid)

        low_contrast = (p99 < 0.90) or (p01 > 0.08) or (dynamic < 0.55)
        trimap_too_open = (
            trimap_stats_before["trimap_bg_ratio_valid"] < 0.08
            or trimap_stats_before["trimap_unknown_ratio_valid"] > 0.70
        )

        if not (low_contrast or trimap_too_open):
            return alpha, {
                "applied": False,
                "reason": f"{reason_prefix}_already_sane",
                "stats_before": stats_before,
                "stats_after": stats_before,
                "trimap_stats_before": trimap_stats_before,
            }

        # Percentile stretch within the real image area. Padding remains 0.
        lo = float(p01)
        hi = float(p99)
        stretched = (alpha - lo) / max(hi - lo, 1e-6)
        stretched = np.clip(stretched, 0.0, 1.0).astype(np.float32)

        if valid is not None:
            stretched[~valid] = 0.0

        # Keep soft edges, but make confident bg/fg actually confident.
        stretched[stretched < 0.015] = 0.0
        stretched[stretched > 0.985] = 1.0

        stats_after = self._array_stats(stretched, valid)
        trimap_after = self.make_loose_trimap(stretched, valid_mask=valid)
        trimap_stats_after = self._trimap_stats_on_valid(trimap_after, valid)

        return stretched, {
            "applied": True,
            "reason": f"{reason_prefix}_low_contrast_or_open_trimap",
            "stats_before": stats_before,
            "stats_after": stats_after,
            "trimap_stats_before": trimap_stats_before,
            "trimap_stats_after": trimap_stats_after,
            "stretch_p01": lo,
            "stretch_p99": hi,
        }

    def _choose_f3_alpha_for_m7_trimap(
        self,
        raw_output_2d: np.ndarray,
        valid_mask: np.ndarray,
    ) -> dict:
        """
        Choose the F3 alpha used for the M7 trimap.

        Priority:
        1. Match notebook when the ONNX output is logits: sigmoid(raw).
        2. Avoid double-sigmoid when the ONNX output is already alpha.
        3. If the chosen alpha is low-contrast and creates an all-unknown trimap,
           percentile-stretch it for trimap generation.
        """
        candidates = self._output_to_alpha_candidates(raw_output_2d)
        raw = candidates["raw"]
        raw_stats = self._array_stats(raw, valid_mask)

        raw_min = raw_stats.get("min")
        raw_max = raw_stats.get("max")

        if raw_min is not None and raw_max is not None and (raw_min < 0.0 or raw_max > 1.0):
            selected_name = "sigmoid_from_logits_like_notebook"
            selected = candidates["sigmoid"]
        else:
            selected_name = "direct_alpha_from_onnx"
            selected = candidates["direct"]

        selected_before_fix = np.clip(selected, 0.0, 1.0).astype(np.float32)
        selected_fixed, fix_stats = self._stretch_alpha_for_trimap_if_needed(
            selected_before_fix,
            valid_mask=valid_mask,
            reason_prefix=selected_name,
        )

        trimap = self.make_loose_trimap(selected_fixed, valid_mask=valid_mask)
        trimap_stats = self._trimap_stats_on_valid(trimap, valid_mask)

        return {
            "raw_output": raw,
            "direct_alpha": candidates["direct"],
            "sigmoid_alpha": candidates["sigmoid"],
            "alpha_before_fix": selected_before_fix,
            "alpha_used": selected_fixed,
            "selected_mode": selected_name,
            "raw_stats": raw_stats,
            "alpha_before_fix_stats": self._array_stats(selected_before_fix, valid_mask),
            "alpha_used_stats": self._array_stats(selected_fixed, valid_mask),
            "alpha_fix_stats": fix_stats,
            "trimap": trimap,
            "trimap_stats_valid": trimap_stats,
        }

    def _clip_vitmatte_alpha(self, x: np.ndarray) -> np.ndarray:
        """
        ViTMatte notebook uses out.alphas directly. No sigmoid and no trimap clamp.
        """
        return np.clip(x.astype(np.float32), 0.0, 1.0)

    def _letterbox_rgb_for_m7(self, image_pil: Image.Image) -> tuple[Image.Image, np.ndarray, dict]:
        """
        Match M7 Cell 07 / F3 letterbox constant:
        RGB -> keep aspect ratio -> constant gray padding -> 768x768.
        valid_mask is 255 in the real image area and 0 in padding.
        """
        image_np = np.array(image_pil.convert("RGB"))
        orig_h, orig_w = image_np.shape[:2]

        scale = min(self.size / orig_w, self.size / orig_h)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_img = cv2.resize(image_np, (resized_w, resized_h), interpolation=interpolation)

        pad_left = (self.size - resized_w) // 2
        pad_right = self.size - resized_w - pad_left
        pad_top = (self.size - resized_h) // 2
        pad_bottom = self.size - resized_h - pad_top

        if self.letterbox_image_pad_mode != "constant":
            raise ValueError(f"M7 expects constant letterbox padding, got {self.letterbox_image_pad_mode!r}.")

        pad_value = int(self.letterbox_image_pad_value)
        padded_img = cv2.copyMakeBorder(
            resized_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )

        valid_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        valid_mask[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = 255

        meta = {
            "original_width": int(orig_w),
            "original_height": int(orig_h),
            "resized_width": int(resized_w),
            "resized_height": int(resized_h),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "pad_right": int(pad_right),
            "pad_bottom": int(pad_bottom),
            "scale": float(scale),
            "target_size": int(self.size),
            "image_pad_mode": self.letterbox_image_pad_mode,
            "image_pad_value": int(self.letterbox_image_pad_value),
            "alpha_pad_value": int(self.letterbox_alpha_pad_value),
        }

        return Image.fromarray(padded_img.astype(np.uint8)), valid_mask, meta

    def _restore_alpha_from_letterbox(self, alpha_square_np: np.ndarray, meta: dict) -> np.ndarray:
        """
        Match M7 Cell 27 restore_alpha_from_letterbox:
        alpha square -> crop padding -> resize to original image size.
        """
        alpha = np.asarray(alpha_square_np)

        if alpha.dtype != np.uint8:
            alpha_uint8 = (np.clip(alpha, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            alpha_uint8 = alpha.astype(np.uint8)

        x1 = int(meta["pad_left"])
        y1 = int(meta["pad_top"])
        x2 = x1 + int(meta["resized_width"])
        y2 = y1 + int(meta["resized_height"])

        cropped = alpha_uint8[y1:y2, x1:x2]

        restored = cv2.resize(
            cropped,
            (int(meta["original_width"]), int(meta["original_height"])),
            interpolation=cv2.INTER_LINEAR,
        )

        return restored.astype(np.uint8)

    def _preprocess_unet_letterbox(self, image_letterbox_pil: Image.Image) -> np.ndarray:
        arr = np.asarray(image_letterbox_pil.convert("RGB")).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        return arr[None, :, :, :].astype(np.float32)

    def _predict_f3_raw_letterbox(self, image_letterbox_pil: Image.Image) -> np.ndarray:
        x = self._preprocess_unet_letterbox(image_letterbox_pil)
        y = self.f2_sess.run(None, {self.f2_input_name: x})[0]
        y = np.asarray(y)

        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            raise ValueError(f"U-Net/F3 ONNX output shape không hợp lệ: {y.shape}")

        return y.astype(np.float32)

    def _predict_f3_alpha_letterbox(self, image_letterbox_pil: Image.Image) -> np.ndarray:
        """Backward-compatible helper used by older code paths."""
        raw = self._predict_f3_raw_letterbox(image_letterbox_pil)
        candidates = self._output_to_alpha_candidates(raw)
        raw_stats = self._array_stats(raw)
        raw_min = raw_stats.get("min")
        raw_max = raw_stats.get("max")

        if raw_min is not None and raw_max is not None and (raw_min < 0.0 or raw_max > 1.0):
            return candidates["sigmoid"]
        return candidates["direct"]

    def make_loose_trimap(
        self,
        alpha_01: np.ndarray,
        fg_thresh: float = 0.995,
        bg_thresh: float = 0.01,
        rough_thresh: float = 0.03,
        erode_k: int = 21,
        dilate_k: int = 35,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        M7/M6 loose trimap from F3 auto alpha.

        Padding area must be background when valid_mask is provided.
        bg_thresh is kept for config compatibility; M7 loose trimap does not use it.
        """
        _ = bg_thresh

        auto_alpha_float = np.clip(alpha_01, 0.0, 1.0).astype(np.float32)

        strong_fg = (auto_alpha_float >= fg_thresh).astype(np.uint8) * 255
        rough_fg = (auto_alpha_float >= rough_thresh).astype(np.uint8) * 255

        erode_kernel = np.ones((int(erode_k), int(erode_k)), np.uint8)
        dilate_kernel = np.ones((int(dilate_k), int(dilate_k)), np.uint8)

        sure_fg = cv2.erode(strong_fg, erode_kernel, iterations=1)
        possible_fg = cv2.dilate(rough_fg, dilate_kernel, iterations=1)

        trimap = np.full(auto_alpha_float.shape, 128, dtype=np.uint8)
        trimap[possible_fg == 0] = 0
        trimap[sure_fg == 255] = 255

        if valid_mask is not None:
            valid = np.asarray(valid_mask).astype(bool)
            if valid.shape != trimap.shape:
                valid = cv2.resize(
                    valid.astype(np.uint8) * 255,
                    (trimap.shape[1], trimap.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            trimap[~valid] = 0

        return trimap

    def trimap_stats(self, trimap: np.ndarray) -> dict:
        total = max(1, trimap.size)
        return {
            "trimap_bg_ratio": float((trimap == 0).sum() / total),
            "trimap_unknown_ratio": float((trimap == 128).sum() / total),
            "trimap_fg_ratio": float((trimap == 255).sum() / total),
        }

    def _preprocess_vitmatte_letterbox(self, image_letterbox_pil: Image.Image, trimap_768: np.ndarray) -> np.ndarray:
        """
        Build ONNX pixel_values equivalent for M7 ViTMatte:
        normalized RGB letterbox image + trimap channel in 0..1.
        """
        if image_letterbox_pil.size != (self.size, self.size):
            image_letterbox_pil = image_letterbox_pil.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        else:
            image_letterbox_pil = image_letterbox_pil.convert("RGB")

        img = np.asarray(image_letterbox_pil).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        if trimap_768.shape != (self.size, self.size):
            trimap_768 = cv2.resize(trimap_768, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        tri = trimap_768.astype(np.float32) / 255.0
        tri = tri[None, :, :]

        x = np.concatenate([img, tri], axis=0)
        return x[None, :, :, :].astype(np.float32)

    def _predict_m7_alpha_letterbox(self, image_letterbox_pil: Image.Image, trimap_768: np.ndarray) -> np.ndarray:
        x = self._preprocess_vitmatte_letterbox(image_letterbox_pil, trimap_768)
        y = self.m6_sess.run(None, {self.m6_input_name: x})[0]
        y = np.asarray(y)

        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            raise ValueError(f"ViTMatte/M7 ONNX output shape không hợp lệ: {y.shape}")

        return self._clip_vitmatte_alpha(y)

    def predict_alpha(self, image_path: str) -> dict:
        image_pil = Image.open(image_path).convert("RGB")

        image_letterbox_pil, valid_mask_768, meta = self._letterbox_rgb_for_m7(image_pil)
        valid_bool = valid_mask_768 > 0

        f3_raw_letterbox = self._predict_f3_raw_letterbox(image_letterbox_pil)
        f3_selection = self._choose_f3_alpha_for_m7_trimap(f3_raw_letterbox, valid_bool)

        f3_alpha_before_fix = f3_selection["alpha_before_fix"]
        f3_alpha_letterbox = f3_selection["alpha_used"]
        trimap_768 = f3_selection["trimap"]

        vit_alpha_letterbox = self._predict_m7_alpha_letterbox(image_letterbox_pil, trimap_768)

        # Deployment guard: if the exported ViTMatte ONNX returns a low-contrast
        # alpha for a hard case, the final PNG looks like a translucent image.
        # Calibrate only in that failure mode; otherwise keep notebook out.alphas.
        vit_alpha_fixed, vit_alpha_fix_stats = self._stretch_alpha_for_trimap_if_needed(
            vit_alpha_letterbox,
            valid_mask=valid_bool,
            reason_prefix="vitmatte_output",
        )

        # Only use the ViTMatte calibration when the raw ViTMatte alpha itself is
        # visibly low-contrast. This avoids over-sharpening normal soft edges.
        vit_stats = self._array_stats(vit_alpha_letterbox, valid_bool)
        use_vit_fix = bool(vit_alpha_fix_stats.get("applied")) and (
            (vit_stats.get("p99") is not None and vit_stats["p99"] < 0.88)
            or (vit_stats.get("p01") is not None and vit_stats["p01"] > 0.10)
        )
        vit_alpha_for_final = vit_alpha_fixed if use_vit_fix else vit_alpha_letterbox

        final_alpha_original = self._restore_alpha_from_letterbox(vit_alpha_for_final, meta)
        unet_alpha_original = self._restore_alpha_from_letterbox(f3_alpha_letterbox, meta)

        def alpha_float_to_uint8(x: np.ndarray) -> np.ndarray:
            return (np.clip(x, 0.0, 1.0) * 255.0).round().clip(0, 255).astype(np.uint8)

        f3_raw_debug = alpha_float_to_uint8(np.clip(f3_raw_letterbox, 0.0, 1.0))
        f3_before_fix_uint8 = alpha_float_to_uint8(f3_alpha_before_fix)
        f3_alpha_letterbox_uint8 = alpha_float_to_uint8(f3_alpha_letterbox)
        vit_alpha_raw_uint8 = alpha_float_to_uint8(vit_alpha_letterbox)
        vit_alpha_final_uint8 = alpha_float_to_uint8(vit_alpha_for_final)

        return {
            "final_alpha": final_alpha_original,
            "f2_alpha_768": f3_alpha_letterbox_uint8,
            "f3_raw_output_letterbox_debug": f3_raw_debug,
            "f3_auto_alpha_before_fix_768": f3_before_fix_uint8,
            "unet_alpha_original": unet_alpha_original,
            "unet_alpha_final_768": vit_alpha_final_uint8.copy(),
            "f3_letterbox_meta": meta,
            "image_letterbox_pil": image_letterbox_pil,
            "valid_mask_768": valid_mask_768,
            "trimap_768": trimap_768,
            "m6_alpha_768": vit_alpha_final_uint8,
            "m6_raw_alpha_768": vit_alpha_raw_uint8,
            "trimap_stats": self.trimap_stats(trimap_768),
            "trimap_stats_valid": f3_selection.get("trimap_stats_valid", {}),
            "alpha_fix_stats": {
                "f3_selected_mode": f3_selection["selected_mode"],
                "f3_raw_stats": f3_selection["raw_stats"],
                "f3_before_fix_stats": f3_selection["alpha_before_fix_stats"],
                "f3_used_stats": f3_selection["alpha_used_stats"],
                "f3_fix": f3_selection["alpha_fix_stats"],
                "vit_raw_stats": vit_stats,
                "vit_fix": vit_alpha_fix_stats,
                "vit_fix_used_for_final": use_vit_fix,
            },
            "vitmatte_edge_stats": {"edge_refine_used": False, "mode": "m7_letterbox_full_vitmatte"},
            "safety_notes": [
                "M33_m7_letterbox_constant_full_vitmatte",
                "rgb_pad_114_alpha_pad_0",
                "valid_mask_padding_forced_background",
                "f3_alpha_calibrated_only_when_low_contrast_or_trimap_too_open",
                "vitmatte_alpha_calibrated_only_when_low_contrast",
                "crop_padding_then_resize_alpha_to_original",
            ],
        }

    def make_cutout(self, image_path: str, alpha: np.ndarray) -> Image.Image:
        image_pil = Image.open(image_path).convert("RGBA")
        rgba = np.asarray(image_pil).copy()

        if alpha.shape[:2] != rgba.shape[:2]:
            alpha = cv2.resize(alpha, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LINEAR)

        rgba[:, :, 3] = alpha.astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")

    def make_preview_white(self, cutout_pil: Image.Image) -> Image.Image:
        cutout_pil = cutout_pil.convert("RGBA")
        bg = Image.new("RGBA", cutout_pil.size, (255, 255, 255, 255))
        bg.alpha_composite(cutout_pil)
        return bg.convert("RGB")

    def make_preview_checker(self, cutout_pil: Image.Image, tile: int = 24) -> Image.Image:
        cutout_pil = cutout_pil.convert("RGBA")
        w, h = cutout_pil.size

        checker = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        arr = np.asarray(checker).copy()

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                if ((x // tile) + (y // tile)) % 2 == 0:
                    arr[y:y + tile, x:x + tile, :3] = 220
                else:
                    arr[y:y + tile, x:x + tile, :3] = 255

        checker = Image.fromarray(arr, mode="RGBA")
        checker.alpha_composite(cutout_pil)
        return checker.convert("RGB")

    def build_prediction_from_alpha(
        self,
        image_path: str,
        final_alpha: np.ndarray,
        elapsed_sec: float = 0.0,
        source: str = "edited_alpha",
    ) -> dict:
        """
        Build a prediction dictionary from an externally edited alpha matte.
        Used by Manual Erase/Restore and SAM Guided edits.
        """
        image_path = str(image_path)
        image_pil = Image.open(image_path).convert("RGB")

        alpha = np.asarray(final_alpha).astype(np.uint8)
        if alpha.shape[:2] != (image_pil.height, image_pil.width):
            alpha = cv2.resize(alpha, (image_pil.width, image_pil.height), interpolation=cv2.INTER_LINEAR)

        alpha_768 = cv2.resize(alpha, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        alpha_768_float = alpha_768.astype(np.float32) / 255.0
        trimap_768 = self.make_loose_trimap(alpha_768_float)

        cutout = self.make_cutout(image_path, alpha)
        preview_checker = self.make_preview_checker(cutout)
        preview_white = self.make_preview_white(cutout)

        return {
            "input_path": image_path,
            "final_alpha": alpha,
            "f2_alpha_768": alpha_768,
            "unet_alpha_original": alpha,
            "unet_alpha_final_768": alpha_768,
            "f3_letterbox_meta": None,
            "image_letterbox_pil": None,
            "valid_mask_768": None,
            "alpha_fix_stats": {},
            "vitmatte_edge_stats": {},
            "trimap_768": trimap_768,
            "m6_alpha_768": alpha_768,
            "m6_raw_alpha_768": alpha_768.copy(),
            "trimap_stats": self.trimap_stats(trimap_768),
            "safety_notes": [source],
            "cutout_pil": cutout,
            "preview_checker_pil": preview_checker,
            "preview_white_pil": preview_white,
            "elapsed_sec": elapsed_sec,
        }

    def process_for_preview(self, image_path: str) -> dict:
        start = time.time()
        image_path = str(image_path)

        pred = self.predict_alpha(image_path)
        cutout = self.make_cutout(image_path, pred["final_alpha"])
        preview_checker = self.make_preview_checker(cutout)
        preview_white = self.make_preview_white(cutout)

        elapsed = time.time() - start

        return {
            "input_path": image_path,
            "final_alpha": pred["final_alpha"],
            "f2_alpha_768": pred["f2_alpha_768"],
            "unet_alpha_original": pred.get("unet_alpha_original"),
            "unet_alpha_final_768": pred.get("unet_alpha_final_768"),
            "f3_letterbox_meta": pred.get("f3_letterbox_meta"),
            "image_letterbox_pil": pred.get("image_letterbox_pil"),
            "valid_mask_768": pred.get("valid_mask_768"),
            "alpha_fix_stats": pred.get("alpha_fix_stats", {}),
            "vitmatte_edge_stats": pred.get("vitmatte_edge_stats", {}),
            "trimap_768": pred["trimap_768"],
            "m6_alpha_768": pred["m6_alpha_768"],
            "m6_raw_alpha_768": pred["m6_raw_alpha_768"],
            "trimap_stats": pred["trimap_stats"],
            "safety_notes": pred["safety_notes"],
            "cutout_pil": cutout,
            "preview_checker_pil": preview_checker,
            "preview_white_pil": preview_white,
            "elapsed_sec": elapsed,
        }
