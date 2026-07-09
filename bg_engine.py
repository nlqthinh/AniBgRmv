from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class BgRemoveEngine:
    """
    M9 ONNX runtime engine.

    Pipeline:
    image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX -> final alpha -> cutout PNG

    Runtime goal:
    - App side does not need torch / segmentation_models_pytorch / transformers.
    - Export scripts still need those packages, but end users should not.
    """

    def __init__(
        self,
        f2a1_model_path: str,
        m6_model_path: str,
        size: int = 768,
        providers: list[str] | None = None,
    ):
        self.size = size

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

        # F2_A1 and ViTMatte image branch both use ImageNet normalization.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resolve_asset_path(self, relative_or_abs_path: str) -> Path:
        """
        Resolve files both in source mode and PyInstaller mode.

        Source mode:
            project/models/xxx.onnx

        PyInstaller onedir/onefile mode:
            dist/AniBGRmv/models/xxx.onnx
            or temp _MEIPASS/models/xxx.onnx
        """
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

        # Return the most natural path for a useful error message.
        return Path.cwd() / p

    def _default_providers(self) -> list[str]:
        available = ort.get_available_providers()

        # CPU is the safest default for "run exe immediately".
        # If you deliberately install onnxruntime-gpu or onnxruntime-directml,
        # the engine can use those providers without changing app.py.
        preferred = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]

        selected = [p for p in preferred if p in available]
        if not selected:
            selected = available

        return selected

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
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return arr[None, :, :, :].astype(np.float32)

    def _predict_f2_alpha_768(self, image_pil: Image.Image) -> np.ndarray:
        x = self._preprocess_rgb_square(image_pil)
        y = self.f2_sess.run(None, {self.f2_input_name: x})[0]

        # Expected: [1, 1, H, W], but handle common variants.
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
        """
        Loose trimap from F2_A1 alpha.

        Concept:
        - Vùng chắc chắn nền: alpha rất thấp hoặc nằm ngoài vùng dilate.
        - Vùng chắc chắn foreground: phần bên trong sau erode.
        - Vùng không chắc: biên tóc/viền/chi tiết mảnh, đưa cho ViTMatte xử lý.

        Output:
        - 0   = background
        - 128 = unknown
        - 255 = foreground
        """
        alpha_u8 = (np.clip(alpha_01, 0.0, 1.0) * 255.0).astype(np.uint8)

        binary = (alpha_u8 >= 128).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        sure_fg = cv2.erode(binary, k, iterations=1) > 0
        sure_bg = cv2.dilate(binary, k, iterations=1) == 0

        trimap = np.full(alpha_u8.shape, 128, dtype=np.uint8)
        trimap[sure_bg] = 0
        trimap[sure_fg] = 255

        # Extremely confident zones from F2_A1 can stay known.
        trimap[alpha_u8 <= bg_threshold] = 0
        trimap[alpha_u8 >= fg_threshold] = 255

        return trimap

    def _preprocess_vitmatte_768(self, image_pil: Image.Image, trimap_768: np.ndarray) -> np.ndarray:
        """
        Reproduce ViTMatte processor logic for fixed 768x768:
        - RGB image: rescale 0..1 then ImageNet normalize.
        - trimap: rescale 0..1, no ImageNet normalize.
        - concat into 4 channels: [R, G, B, trimap].
        """
        image_pil = image_pil.convert("RGB")
        resized = image_pil.resize((self.size, self.size), Image.BILINEAR)

        img = np.asarray(resized).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # 3,H,W

        if trimap_768.shape != (self.size, self.size):
            trimap_768 = cv2.resize(
                trimap_768,
                (self.size, self.size),
                interpolation=cv2.INTER_NEAREST,
            )

        tri = trimap_768.astype(np.float32) / 255.0
        tri = tri[None, :, :]  # 1,H,W

        x = np.concatenate([img, tri], axis=0)
        return x[None, :, :, :].astype(np.float32)

    def _predict_m6_alpha_768(self, image_pil: Image.Image, trimap_768: np.ndarray) -> np.ndarray:
        x = self._preprocess_vitmatte_768(image_pil, trimap_768)
        y = self.m6_sess.run(None, {self.m6_input_name: x})[0]

        # Expected: [1, 1, H, W], but handle common variants.
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
        m6_alpha_768 = self._predict_m6_alpha_768(image_pil, trimap_768)

        final_alpha_255 = (m6_alpha_768 * 255.0).clip(0, 255).astype(np.uint8)
        final_alpha_original = cv2.resize(
            final_alpha_255,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        f2_alpha_255 = (f2_alpha_768 * 255.0).clip(0, 255).astype(np.uint8)

        return {
            "final_alpha": final_alpha_original,
            "f2_alpha_768": f2_alpha_255,
            "trimap_768": trimap_768,
            "m6_alpha_768": final_alpha_255,
        }

    def make_cutout(self, image_path: str, alpha: np.ndarray):
        image_pil = Image.open(image_path).convert("RGBA")
        rgba = np.asarray(image_pil).copy()

        if alpha.shape[:2] != rgba.shape[:2]:
            alpha = cv2.resize(
                alpha,
                (rgba.shape[1], rgba.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        rgba[:, :, 3] = alpha
        return Image.fromarray(rgba)

    def make_preview_white(self, cutout_pil: Image.Image):
        cutout_pil = cutout_pil.convert("RGBA")
        bg = Image.new("RGBA", cutout_pil.size, (255, 255, 255, 255))
        bg.alpha_composite(cutout_pil)
        return bg.convert("RGB")

    def make_preview_checker(self, cutout_pil: Image.Image, tile: int = 24):
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

    def process_one(self, image_path: str, output_dir: str):
        start = time.time()

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem

        pred = self.predict_alpha(str(image_path))
        alpha = pred["final_alpha"]

        cutout = self.make_cutout(str(image_path), alpha)
        preview_white = self.make_preview_white(cutout)
        preview_checker = self.make_preview_checker(cutout)

        # Main outputs
        mask_path = output_dir / f"{stem}_final_alpha.png"
        cutout_path = output_dir / f"{stem}_cutout.png"
        preview_white_path = output_dir / f"{stem}_preview_white.jpg"
        preview_checker_path = output_dir / f"{stem}_preview_checker.jpg"

        # Debug outputs. These help verify the M9 pipeline.
        f2_alpha_path = output_dir / f"{stem}_debug_f2_auto_alpha_768.png"
        trimap_path = output_dir / f"{stem}_debug_loose_trimap_768.png"
        m6_alpha_path = output_dir / f"{stem}_debug_m6_alpha_768.png"

        Image.fromarray(alpha).save(mask_path)
        Image.fromarray(pred["f2_alpha_768"]).save(f2_alpha_path)
        Image.fromarray(pred["trimap_768"]).save(trimap_path)
        Image.fromarray(pred["m6_alpha_768"]).save(m6_alpha_path)

        cutout.save(cutout_path)
        preview_white.save(preview_white_path, quality=95)
        preview_checker.save(preview_checker_path, quality=95)

        elapsed = time.time() - start

        return {
            "input_path": str(image_path),
            "mask_path": str(mask_path),
            "cutout_path": str(cutout_path),
            "preview_white_path": str(preview_white_path),
            "preview_checker_path": str(preview_checker_path),
            "f2_alpha_path": str(f2_alpha_path),
            "trimap_path": str(trimap_path),
            "m6_alpha_path": str(m6_alpha_path),
            "elapsed_sec": elapsed,
        }
