from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class BgRemoveEngine:
    """
    ONNX inference engine.

    M18 goal:
    - Match M6_5k / M6b notebook logic as closely as possible.
    - Remove previous ad-hoc trimap safety / alpha repair logic.
    - ViTMatte output is treated like notebook `out.alphas`.

    Pipeline:
    image
    -> U-Net/C7 ONNX auto alpha 768
    -> M6 loose trimap from auto alpha 768
    -> ViTMatte ONNX alpha 768
    -> resize alpha to original size
    -> RGBA cutout

    Important:
    Notebook M6 does NOT clamp final alpha by trimap after ViTMatte.
    Notebook M6 uses pred_alpha = out.alphas directly.
    """

    def __init__(
        self,
        f2a1_model_path: str,
        m6_model_path: str,
        size: int = 768,
        providers: list[str] | None = None,
        enforce_trimap_known_regions: bool = False,
    ):
        self.size = size

        # Kept for backward compatibility with older app.py, but M18 does not use it.
        self.enforce_trimap_known_regions = enforce_trimap_known_regions

        self.f2a1_model_path = self._resolve_asset_path(f2a1_model_path)
        self.m6_model_path = self._resolve_asset_path(m6_model_path)

        if not self.f2a1_model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy U-Net/C7 ONNX: {self.f2a1_model_path}")
        if not self.m6_model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ViTMatte ONNX: {self.m6_model_path}")

        self.providers = providers or self._default_providers()

        self.f2_sess = ort.InferenceSession(str(self.f2a1_model_path), providers=self.providers)
        self.m6_sess = ort.InferenceSession(str(self.m6_model_path), providers=self.providers)

        self.f2_input_name = self.f2_sess.get_inputs()[0].name
        self.m6_input_name = self.m6_sess.get_inputs()[0].name

        self.runtime_info = " / ".join(self.f2_sess.get_providers())

        # Same ImageNet normalize used by C7/F3 and by ViTMatte processor image channels.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
        preferred = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]

        selected = [p for p in preferred if p in available]
        return selected or available

    def _sigmoid_if_needed_for_unet(self, x: np.ndarray) -> np.ndarray:
        """
        U-Net/C7 export may return logits or alpha.
        If output is outside 0..1, apply sigmoid. Otherwise use as alpha.
        """
        x = x.astype(np.float32)

        if float(np.nanmin(x)) < 0.0 or float(np.nanmax(x)) > 1.0:
            x = 1.0 / (1.0 + np.exp(-x))

        return np.clip(x, 0.0, 1.0)

    def _clip_vitmatte_alpha(self, x: np.ndarray) -> np.ndarray:
        """
        M6 notebook uses:
            pred_alpha = out.alphas

        No sigmoid.
        No trimap clamp.
        Only clamp to valid alpha range.
        """
        return np.clip(x.astype(np.float32), 0.0, 1.0)

    def _preprocess_rgb_square(self, image_pil: Image.Image) -> np.ndarray:
        """
        C7/F3-style preprocess:
        RGB -> resize 768 -> scale 0..1 -> ImageNet normalize -> NCHW.
        """
        image_pil = image_pil.convert("RGB")
        resized = image_pil.resize((self.size, self.size), Image.BILINEAR)

        arr = np.asarray(resized).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))

        return arr[None, :, :, :].astype(np.float32)

    def _predict_f2_alpha_768(self, image_pil: Image.Image) -> np.ndarray:
        x = self._preprocess_rgb_square(image_pil)
        y = self.f2_sess.run(None, {self.f2_input_name: x})[0]

        y = np.asarray(y)

        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            raise ValueError(f"U-Net/C7 ONNX output shape không hợp lệ: {y.shape}")

        return self._sigmoid_if_needed_for_unet(y)

    def make_loose_trimap(
        self,
        alpha_01: np.ndarray,
        fg_thresh: float = 0.995,
        bg_thresh: float = 0.01,
        rough_thresh: float = 0.03,
        erode_k: int = 21,
        dilate_k: int = 35,
    ) -> np.ndarray:
        """
        Exact M6/M6b loose trimap logic.

        Copied conceptually from notebook:

            strong_fg = auto_alpha >= fg_thresh
            rough_fg  = auto_alpha >= rough_thresh
            sure_fg = erode(strong_fg, erode_k)
            possible_fg = dilate(rough_fg, dilate_k)
            trimap = 128
            trimap[possible_fg == 0] = 0
            trimap[sure_fg == 255] = 255

        Note:
        - bg_thresh is kept in signature because M6 config stores it,
          but original create_loose_trimap_from_auto_alpha does not use it.
        - Do not add app-only filtering/fallback here.
        """
        _ = bg_thresh

        auto_alpha_float = np.clip(alpha_01, 0.0, 1.0).astype(np.float32)

        strong_fg = (auto_alpha_float >= fg_thresh).astype(np.uint8) * 255
        rough_fg = (auto_alpha_float >= rough_thresh).astype(np.uint8) * 255

        erode_kernel = np.ones((erode_k, erode_k), np.uint8)
        dilate_kernel = np.ones((dilate_k, dilate_k), np.uint8)

        sure_fg = cv2.erode(strong_fg, erode_kernel, iterations=1)
        possible_fg = cv2.dilate(rough_fg, dilate_kernel, iterations=1)

        trimap = np.full(auto_alpha_float.shape, 128, dtype=np.uint8)
        trimap[possible_fg == 0] = 0
        trimap[sure_fg == 255] = 255

        return trimap

    def trimap_stats(self, trimap: np.ndarray) -> dict:
        total = max(1, trimap.size)
        return {
            "trimap_bg_ratio": float((trimap == 0).sum() / total),
            "trimap_unknown_ratio": float((trimap == 128).sum() / total),
            "trimap_fg_ratio": float((trimap == 255).sum() / total),
        }

    def _preprocess_vitmatte_768(self, image_pil: Image.Image, trimap_768: np.ndarray) -> np.ndarray:
        """
        Approximate Hugging Face VitMatteImageProcessor output for ONNX.

        Notebook does:
            vit_processor(images=images_pil, trimaps=trimaps_pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

        ONNX model exported for app expects pixel_values.
        This function builds pixel_values directly:
            3 normalized RGB channels + 1 trimap channel in 0..1
        """
        image_pil = image_pil.convert("RGB")
        resized = image_pil.resize((self.size, self.size), Image.BILINEAR)

        img = np.asarray(resized).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        if trimap_768.shape != (self.size, self.size):
            trimap_768 = cv2.resize(
                trimap_768,
                (self.size, self.size),
                interpolation=cv2.INTER_NEAREST,
            )

        tri = trimap_768.astype(np.float32) / 255.0
        tri = tri[None, :, :]

        x = np.concatenate([img, tri], axis=0)
        return x[None, :, :, :].astype(np.float32)

    def _predict_m6_alpha_768(self, image_pil: Image.Image, trimap_768: np.ndarray) -> np.ndarray:
        x = self._preprocess_vitmatte_768(image_pil, trimap_768)
        y = self.m6_sess.run(None, {self.m6_input_name: x})[0]

        y = np.asarray(y)

        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            raise ValueError(f"ViTMatte ONNX output shape không hợp lệ: {y.shape}")

        return self._clip_vitmatte_alpha(y)

    def predict_alpha(self, image_path: str) -> dict:
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        f2_alpha_768 = self._predict_f2_alpha_768(image_pil)
        trimap_768 = self.make_loose_trimap(f2_alpha_768)

        # M6 logic: final 768 alpha is ViTMatte out.alphas.
        # No trimap clamp. No app repair.
        m6_alpha_768 = self._predict_m6_alpha_768(image_pil, trimap_768)

        final_alpha_255 = (np.clip(m6_alpha_768, 0.0, 1.0) * 255.0).clip(0, 255).astype(np.uint8)
        final_alpha_original = cv2.resize(
            final_alpha_255,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return {
            "final_alpha": final_alpha_original,
            "f2_alpha_768": (np.clip(f2_alpha_768, 0.0, 1.0) * 255.0).clip(0, 255).astype(np.uint8),
            "trimap_768": trimap_768,
            "m6_alpha_768": final_alpha_255,
            "m6_raw_alpha_768": final_alpha_255.copy(),
            "trimap_stats": self.trimap_stats(trimap_768),
            "safety_notes": ["M18_exact_M6_logic_no_clamp_no_repair"],
        }

    def make_cutout(self, image_path: str, alpha: np.ndarray) -> Image.Image:
        image_pil = Image.open(image_path).convert("RGBA")
        rgba = np.asarray(image_pil).copy()

        if alpha.shape[:2] != rgba.shape[:2]:
            alpha = cv2.resize(
                alpha,
                (rgba.shape[1], rgba.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        rgba[:, :, 3] = alpha
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

    def process_for_preview(self, image_path: str) -> dict:
        """
        Run AI but do not save files yet.
        """
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

    def _unique_path(self, folder: Path, filename: str) -> Path:
        """
        Avoid overwriting when batch has duplicate image names.
        """
        folder.mkdir(parents=True, exist_ok=True)

        candidate = folder / filename
        if not candidate.exists():
            return candidate

        stem = candidate.stem
        suffix = candidate.suffix

        idx = 2
        while True:
            p = folder / f"{stem}_{idx}{suffix}"
            if not p.exists():
                return p
            idx += 1

    def optimize_cutout_png(self, cutout_pil: Image.Image) -> Image.Image:
        """
        Optimize cutout PNG without changing visible pixels.

        If alpha == 0, RGB is invisible, so set invisible RGB to 0 to make PNG
        compression easier. Soft alpha edges are untouched.
        """
        rgba = cutout_pil.convert("RGBA")
        arr = np.asarray(rgba).copy()

        alpha = arr[:, :, 3]
        transparent = alpha == 0

        arr[transparent, 0] = 0
        arr[transparent, 1] = 0
        arr[transparent, 2] = 0

        return Image.fromarray(arr, mode="RGBA")

    def save_prediction(self, prediction: dict, output_dir: str, save_debug: bool = True) -> dict:
        """
        Only call this when the user clicks Save.
        """
        input_path = Path(prediction["input_path"])
        output_dir = Path(output_dir)
        stem = input_path.stem

        cutout_dir = output_dir / "01_cutouts"
        preview_checker_dir = output_dir / "02_preview_checker"
        preview_white_dir = output_dir / "03_preview_white"

        final_alpha_dir = output_dir / "99_debug" / "01_final_alpha"
        f2_alpha_dir = output_dir / "99_debug" / "02_unet_c7_auto_alpha_768"
        trimap_dir = output_dir / "99_debug" / "03_m6_loose_trimap_768"
        m6_alpha_dir = output_dir / "99_debug" / "04_vitmatte_m6_alpha_768"
        m6_raw_alpha_dir = output_dir / "99_debug" / "05_vitmatte_m6_raw_alpha_768"

        cutout_path = self._unique_path(cutout_dir, f"{stem}_cutout.png")
        preview_checker_path = self._unique_path(preview_checker_dir, f"{stem}_preview_checker.jpg")
        preview_white_path = self._unique_path(preview_white_dir, f"{stem}_preview_white.jpg")

        cutout_optimized = self.optimize_cutout_png(prediction["cutout_pil"])
        cutout_optimized.save(
            cutout_path,
            format="PNG",
            optimize=True,
            compress_level=9,
        )
        prediction["preview_checker_pil"].save(preview_checker_path, quality=95, optimize=True)
        prediction["preview_white_pil"].save(preview_white_path, quality=95, optimize=True)

        saved = {
            "input_path": str(input_path),
            "cutout_path": str(cutout_path),
            "preview_checker_path": str(preview_checker_path),
            "preview_white_path": str(preview_white_path),
            "elapsed_sec": prediction.get("elapsed_sec", None),
            "trimap_stats": prediction.get("trimap_stats", {}),
            "safety_notes": ";".join(prediction.get("safety_notes", [])),
        }

        if save_debug:
            final_alpha_path = self._unique_path(final_alpha_dir, f"{stem}_final_alpha.png")
            f2_alpha_path = self._unique_path(f2_alpha_dir, f"{stem}_unet_c7_auto_alpha_768.png")
            trimap_path = self._unique_path(trimap_dir, f"{stem}_m6_loose_trimap_768.png")
            m6_alpha_path = self._unique_path(m6_alpha_dir, f"{stem}_vitmatte_m6_alpha_768.png")
            m6_raw_alpha_path = self._unique_path(m6_raw_alpha_dir, f"{stem}_vitmatte_m6_raw_alpha_768.png")

            Image.fromarray(prediction["final_alpha"]).save(final_alpha_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["f2_alpha_768"]).save(f2_alpha_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["trimap_768"]).save(trimap_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["m6_alpha_768"]).save(m6_alpha_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["m6_raw_alpha_768"]).save(m6_raw_alpha_path, optimize=True, compress_level=9)

            saved.update({
                "final_alpha_path": str(final_alpha_path),
                "f2_alpha_path": str(f2_alpha_path),
                "trimap_path": str(trimap_path),
                "m6_alpha_path": str(m6_alpha_path),
                "m6_raw_alpha_path": str(m6_raw_alpha_path),
            })

        return saved
