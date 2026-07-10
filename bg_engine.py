from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class BgRemoveEngine:
    """
    M10.1 engine.

    Pipeline:
    image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX
    -> enforce known trimap regions -> final alpha -> cutout PNG

    Important:
    In trimap matting, M6 should mainly refine the unknown region.
    Known background and known foreground from trimap must be enforced.
    """

    def __init__(
        self,
        f2a1_model_path: str,
        m6_model_path: str,
        size: int = 768,
        providers: list[str] | None = None,
        enforce_trimap_known_regions: bool = True,
    ):
        self.size = size
        self.enforce_trimap_known_regions = enforce_trimap_known_regions

        self.f2a1_model_path = self._resolve_asset_path(f2a1_model_path)
        self.m6_model_path = self._resolve_asset_path(m6_model_path)

        if not self.f2a1_model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy F2_A1 ONNX: {self.f2a1_model_path}")
        if not self.m6_model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy M6 ONNX: {self.m6_model_path}")

        self.providers = providers or self._default_providers()

        self.f2_sess = ort.InferenceSession(str(self.f2a1_model_path), providers=self.providers)
        self.m6_sess = ort.InferenceSession(str(self.m6_model_path), providers=self.providers)

        self.f2_input_name = self.f2_sess.get_inputs()[0].name
        self.m6_input_name = self.m6_sess.get_inputs()[0].name

        self.runtime_info = " / ".join(self.f2_sess.get_providers())

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
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        selected = [p for p in preferred if p in available]
        return selected or available

    def _sigmoid_if_needed(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if float(np.nanmin(x)) < 0.0 or float(np.nanmax(x)) > 1.0:
            x = 1.0 / (1.0 + np.exp(-x))
        return np.clip(x, 0.0, 1.0)

    def _preprocess_rgb_square(self, image_pil: Image.Image) -> np.ndarray:
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
            raise ValueError(f"F2_A1 ONNX output shape không hợp lệ: {y.shape}")

        return self._sigmoid_if_needed(y)

    def make_loose_trimap(
        self,
        alpha_01: np.ndarray,
        kernel_size: int = 21,
        fg_threshold: int = 250,
        bg_threshold: int = 5,
    ) -> np.ndarray:
        alpha_u8 = (np.clip(alpha_01, 0.0, 1.0) * 255.0).astype(np.uint8)

        binary = (alpha_u8 >= 128).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        sure_fg = cv2.erode(binary, k, iterations=1) > 0
        sure_bg = cv2.dilate(binary, k, iterations=1) == 0

        trimap = np.full(alpha_u8.shape, 128, dtype=np.uint8)
        trimap[sure_bg] = 0
        trimap[sure_fg] = 255

        trimap[alpha_u8 <= bg_threshold] = 0
        trimap[alpha_u8 >= fg_threshold] = 255

        return trimap

    def enforce_known_regions_from_trimap(self, alpha_01: np.ndarray, trimap_u8: np.ndarray) -> np.ndarray:
        """
        Trimap meaning:
        - 0   = sure background  -> alpha must be 0
        - 128 = unknown          -> trust M6 prediction
        - 255 = sure foreground  -> alpha must be 1
        """
        alpha = np.clip(alpha_01.astype(np.float32), 0.0, 1.0).copy()
        alpha[trimap_u8 == 0] = 0.0
        alpha[trimap_u8 == 255] = 1.0
        return alpha

    def _preprocess_vitmatte_768(self, image_pil: Image.Image, trimap_768: np.ndarray) -> np.ndarray:
        image_pil = image_pil.convert("RGB")
        resized = image_pil.resize((self.size, self.size), Image.BILINEAR)

        img = np.asarray(resized).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        if trimap_768.shape != (self.size, self.size):
            trimap_768 = cv2.resize(trimap_768, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

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
            raise ValueError(f"M6 ONNX output shape không hợp lệ: {y.shape}")

        return self._sigmoid_if_needed(y)

    def predict_alpha(self, image_path: str) -> dict:
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        f2_alpha_768 = self._predict_f2_alpha_768(image_pil)
        trimap_768 = self.make_loose_trimap(f2_alpha_768)
        m6_raw_alpha_768 = self._predict_m6_alpha_768(image_pil, trimap_768)

        if self.enforce_trimap_known_regions:
            m6_final_alpha_768 = self.enforce_known_regions_from_trimap(m6_raw_alpha_768, trimap_768)
        else:
            m6_final_alpha_768 = m6_raw_alpha_768

        final_alpha_255 = (m6_final_alpha_768 * 255.0).clip(0, 255).astype(np.uint8)
        final_alpha_original = cv2.resize(final_alpha_255, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return {
            "final_alpha": final_alpha_original,
            "f2_alpha_768": (f2_alpha_768 * 255.0).clip(0, 255).astype(np.uint8),
            "trimap_768": trimap_768,
            "m6_alpha_768": final_alpha_255,
            "m6_raw_alpha_768": (m6_raw_alpha_768 * 255.0).clip(0, 255).astype(np.uint8),
        }

    def make_cutout(self, image_path: str, alpha: np.ndarray) -> Image.Image:
        image_pil = Image.open(image_path).convert("RGBA")
        rgba = np.asarray(image_pil).copy()

        if alpha.shape[:2] != rgba.shape[:2]:
            alpha = cv2.resize(alpha, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LINEAR)

        rgba[:, :, 3] = alpha
        return Image.fromarray(rgba)

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

        checker = Image.fromarray(arr)
        checker.alpha_composite(cutout_pil)
        return checker.convert("RGB")

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
            "trimap_768": pred["trimap_768"],
            "m6_alpha_768": pred["m6_alpha_768"],
            "m6_raw_alpha_768": pred["m6_raw_alpha_768"],
            "cutout_pil": cutout,
            "preview_checker_pil": preview_checker,
            "preview_white_pil": preview_white,
            "elapsed_sec": elapsed,
        }

    def _unique_path(self, folder: Path, filename: str) -> Path:
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

    def save_prediction(self, prediction: dict, output_dir: str, save_debug: bool = True) -> dict:
        input_path = Path(prediction["input_path"])
        output_dir = Path(output_dir)
        stem = input_path.stem

        cutout_dir = output_dir / "01_cutouts"
        preview_checker_dir = output_dir / "02_preview_checker"
        preview_white_dir = output_dir / "03_preview_white"

        final_alpha_dir = output_dir / "99_debug" / "01_final_alpha"
        f2_alpha_dir = output_dir / "99_debug" / "02_f2_auto_alpha_768"
        trimap_dir = output_dir / "99_debug" / "03_loose_trimap_768"
        m6_alpha_dir = output_dir / "99_debug" / "04_m6_alpha_768_clamped"
        m6_raw_alpha_dir = output_dir / "99_debug" / "05_m6_raw_alpha_768_before_clamp"

        cutout_path = self._unique_path(cutout_dir, f"{stem}_cutout.png")
        preview_checker_path = self._unique_path(preview_checker_dir, f"{stem}_preview_checker.jpg")
        preview_white_path = self._unique_path(preview_white_dir, f"{stem}_preview_white.jpg")

        prediction["cutout_pil"].save(cutout_path)
        prediction["preview_checker_pil"].save(preview_checker_path, quality=95)
        prediction["preview_white_pil"].save(preview_white_path, quality=95)

        saved = {
            "input_path": str(input_path),
            "cutout_path": str(cutout_path),
            "preview_checker_path": str(preview_checker_path),
            "preview_white_path": str(preview_white_path),
            "elapsed_sec": prediction.get("elapsed_sec", None),
        }

        if save_debug:
            final_alpha_path = self._unique_path(final_alpha_dir, f"{stem}_final_alpha.png")
            f2_alpha_path = self._unique_path(f2_alpha_dir, f"{stem}_f2_auto_alpha_768.png")
            trimap_path = self._unique_path(trimap_dir, f"{stem}_loose_trimap_768.png")
            m6_alpha_path = self._unique_path(m6_alpha_dir, f"{stem}_m6_alpha_768_clamped.png")
            m6_raw_alpha_path = self._unique_path(m6_raw_alpha_dir, f"{stem}_m6_raw_alpha_768_before_clamp.png")

            Image.fromarray(prediction["final_alpha"]).save(final_alpha_path)
            Image.fromarray(prediction["f2_alpha_768"]).save(f2_alpha_path)
            Image.fromarray(prediction["trimap_768"]).save(trimap_path)
            Image.fromarray(prediction["m6_alpha_768"]).save(m6_alpha_path)
            Image.fromarray(prediction["m6_raw_alpha_768"]).save(m6_raw_alpha_path)

            saved.update({
                "final_alpha_path": str(final_alpha_path),
                "f2_alpha_path": str(f2_alpha_path),
                "trimap_path": str(trimap_path),
                "m6_alpha_path": str(m6_alpha_path),
                "m6_raw_alpha_path": str(m6_raw_alpha_path),
            })

        return saved
