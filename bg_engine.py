from pathlib import Path
import time

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image


class BgRemoveEngine:
    def __init__(
        self,
        checkpoint_path: str,
        size: int = 768,
        device: str | None = None,
    ):
        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = self._build_model()
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Normalize giống ImageNet. Nếu F2_A1 của bạn dùng mean/std khác,
        # sửa lại đúng theo notebook train.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _build_model(self):
        return smp.Unet(
            encoder_name="mit_b1",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint_path = str(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            raise ValueError("Checkpoint format không hợp lệ.")

        # Xử lý checkpoint train bằng DataParallel nếu có prefix module.
        clean_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            clean_state[k] = v

        self.model.load_state_dict(clean_state, strict=True)

    def _preprocess(self, image_pil: Image.Image):
        image_pil = image_pil.convert("RGB")
        orig_w, orig_h = image_pil.size

        resized = image_pil.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(resized).astype(np.float32) / 255.0

        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        return tensor, (orig_w, orig_h)

    @torch.inference_mode()
    def predict_alpha(self, image_path: str):
        image_path = str(image_path)
        image_pil = Image.open(image_path).convert("RGB")
        tensor, (orig_w, orig_h) = self._preprocess(image_pil)

        tensor = tensor.to(self.device, non_blocking=True)

        logits = self.model(tensor)
        alpha = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        alpha_255 = (alpha * 255.0).clip(0, 255).astype(np.uint8)
        alpha_original = cv2.resize(
            alpha_255,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return alpha_original

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

        alpha = self.predict_alpha(str(image_path))
        cutout = self.make_cutout(str(image_path), alpha)
        preview_white = self.make_preview_white(cutout)
        preview_checker = self.make_preview_checker(cutout)

        mask_path = output_dir / f"{stem}_mask.png"
        cutout_path = output_dir / f"{stem}_cutout.png"
        preview_white_path = output_dir / f"{stem}_preview_white.jpg"
        preview_checker_path = output_dir / f"{stem}_preview_checker.jpg"

        Image.fromarray(alpha).save(mask_path)
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
            "elapsed_sec": elapsed,
        }