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


    def build_prediction_from_alpha(
        self,
        image_path: str,
        final_alpha: np.ndarray,
        elapsed_sec: float = 0.0,
        source: str = "edited_alpha",
    ) -> dict:
        """
        Build the same prediction dictionary shape as process_for_preview(),
        but from an externally edited alpha matte.

        Used by the UI edit mode:
        - Manual Erase / Restore edits alpha directly.
        - Guided SAM edit returns an edited alpha.
        """
        image_path = str(image_path)
        image_pil = Image.open(image_path).convert("RGB")

        alpha = np.asarray(final_alpha).astype(np.uint8)
        if alpha.shape[:2] != (image_pil.height, image_pil.width):
            alpha = cv2.resize(
                alpha,
                (image_pil.width, image_pil.height),
                interpolation=cv2.INTER_LINEAR,
            )

        alpha_768 = cv2.resize(
            alpha,
            (self.size, self.size),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha_768_float = alpha_768.astype(np.float32) / 255.0
        trimap_768 = self.make_loose_trimap(alpha_768_float)

        cutout = self.make_cutout(image_path, alpha)
        preview_checker = self.make_preview_checker(cutout)
        preview_white = self.make_preview_white(cutout)

        return {
            "input_path": image_path,
            "final_alpha": alpha,
            "f2_alpha_768": alpha_768,
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

    def _load_sam_predictor(
        self,
        sam_checkpoint_path: str,
        model_type: str = "vit_b",
        device: str = "auto",
    ):
        """
        Lazy-load SAM only when Guided Edit is used.
        This keeps normal auto-cut startup unchanged.
        """
        if not hasattr(self, "_sam_cache"):
            self._sam_cache = {}

        checkpoint = self._resolve_asset_path(sam_checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Không tìm thấy SAM checkpoint: {checkpoint}\n"
                "Hãy tải sam_vit_b_01ec64.pth hoặc sửa app_config.json mục sam.checkpoint_path."
            )

        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as e:
            raise ImportError(
                "Thiếu package SAM. Cài bằng:\n"
                "pip install torch torchvision\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        key = (str(checkpoint.resolve()), model_type, device)
        if key in self._sam_cache:
            return self._sam_cache[key]

        if model_type not in sam_model_registry:
            raise ValueError("model_type SAM không hợp lệ. Hãy dùng vit_b, vit_l hoặc vit_h.")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        sam.eval()
        predictor = SamPredictor(sam)

        self._sam_cache[key] = predictor
        return predictor

    def _stroke_points_to_prompt(
        self,
        stroke_points: list[tuple[float, float]],
        image_size: tuple[int, int],
        brush_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a UI brush stroke to SAM point + box prompt.
        Box makes Guided Edit much more stable than point-only prompt.
        """
        if not stroke_points:
            raise ValueError("Guided edit cần ít nhất một điểm/stroke.")

        w, h = image_size
        pts = np.array(stroke_points, dtype=np.float32)

        # Limit point count. Too many points can make SAM unstable and slower.
        if len(pts) > 12:
            idx = np.linspace(0, len(pts) - 1, 12).astype(np.int32)
            pts = pts[idx]

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        labels = np.ones((len(pts),), dtype=np.int32)

        pad = max(24, int(brush_size * 3))
        x1 = max(0.0, float(np.min(pts[:, 0]) - pad))
        y1 = max(0.0, float(np.min(pts[:, 1]) - pad))
        x2 = min(float(w - 1), float(np.max(pts[:, 0]) + pad))
        y2 = min(float(h - 1), float(np.max(pts[:, 1]) + pad))
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        return pts.astype(np.float32), labels, box

    def _stroke_mask(
        self,
        shape_hw: tuple[int, int],
        stroke_points: list[tuple[float, float]],
        brush_size: int,
    ) -> np.ndarray:
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        if not stroke_points:
            return mask.astype(bool)

        radius = max(1, int(brush_size // 2))
        pts = [(int(round(x)), int(round(y))) for x, y in stroke_points]

        if len(pts) == 1:
            cv2.circle(mask, pts[0], radius, 255, -1, lineType=cv2.LINE_AA)
        else:
            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(mask, p1, p2, 255, thickness=max(1, brush_size), lineType=cv2.LINE_AA)
            for p in pts:
                cv2.circle(mask, p, radius, 255, -1, lineType=cv2.LINE_AA)

        return mask > 0

    def _refine_alpha_from_user_edit(
        self,
        image_path: str,
        edited_alpha: np.ndarray,
        force_zero_mask: np.ndarray | None = None,
        force_one_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Refine edited alpha with ViTMatte.

        Important for edit mode:
        - The user edit is a hard constraint.
        - ViTMatte refines edges, but it must not undo the user's erase/restore action.
        """
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        alpha = np.asarray(edited_alpha).astype(np.uint8)
        if alpha.shape[:2] != (orig_h, orig_w):
            alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        alpha_768 = cv2.resize(alpha, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        alpha_768_01 = alpha_768.astype(np.float32) / 255.0
        trimap_768 = self.make_loose_trimap(alpha_768_01)

        m6_alpha_768 = self._predict_m6_alpha_768(image_pil, trimap_768)
        final = (np.clip(m6_alpha_768, 0.0, 1.0) * 255.0).astype(np.uint8)
        final = cv2.resize(final, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        if force_zero_mask is not None:
            force_zero = np.asarray(force_zero_mask).astype(bool)
            if force_zero.shape[:2] != final.shape[:2]:
                force_zero = cv2.resize(
                    force_zero.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            final[force_zero] = 0

        if force_one_mask is not None:
            force_one = np.asarray(force_one_mask).astype(bool)
            if force_one.shape[:2] != final.shape[:2]:
                force_one = cv2.resize(
                    force_one.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            final[force_one] = 255

        return final

    def apply_sam_guided_edit(
        self,
        image_path: str,
        current_alpha: np.ndarray,
        stroke_points: list[tuple[float, float]],
        brush_size: int,
        operation: str,
        sam_checkpoint_path: str,
        sam_model_type: str = "vit_b",
        sam_device: str = "auto",
        refine_with_vitmatte: bool = True,
    ) -> dict:
        """
        Guided Erase / Restore:
        1. Use SAM to select the local region indicated by the user's click/stroke.
        2. Apply that region to the current alpha.
        3. Optionally run ViTMatte once to clean the edge.

        M24 speed fix:
        - Do not call predictor.set_image() for every stroke.
        - Cache SAM image embedding per image. The first Guided edit on an image is still
          the expensive step; subsequent strokes reuse the embedding and are much faster.
        - Restrict the chosen SAM mask to a local box around the user's stroke, so SAM does
          not restore/erase a distant unrelated object.
        """
        start = time.time()

        image_path = str(image_path)
        image_pil = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(image_pil)
        h, w = image_rgb.shape[:2]

        alpha = np.asarray(current_alpha).astype(np.uint8).copy()
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

        predictor = self._load_sam_predictor(
            sam_checkpoint_path=sam_checkpoint_path,
            model_type=sam_model_type,
            device=sam_device,
        )

        brush_mask = self._stroke_mask((h, w), stroke_points, brush_size)
        pts_full = np.array(stroke_points, dtype=np.float32)
        if pts_full.size == 0:
            raise ValueError("Guided edit cần ít nhất một điểm/stroke.")
        pts_full[:, 0] = np.clip(pts_full[:, 0], 0, w - 1)
        pts_full[:, 1] = np.clip(pts_full[:, 1], 0, h - 1)

        # Local prompt box around stroke. This does NOT crop/re-encode the image; it only
        # tells SAM which local region the user wants to edit.
        pad = max(48, int(brush_size * 4))
        x1 = int(max(0, np.floor(float(np.min(pts_full[:, 0])) - pad)))
        y1 = int(max(0, np.floor(float(np.min(pts_full[:, 1])) - pad)))
        x2 = int(min(w - 1, np.ceil(float(np.max(pts_full[:, 0])) + pad)))
        y2 = int(min(h - 1, np.ceil(float(np.max(pts_full[:, 1])) + pad)))

        # Prevent very long strokes from creating a huge local box.
        # A guided edit should be local; large areas are better handled with Manual brush.
        max_box_side = 512
        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1
        if box_w > max_box_side or box_h > max_box_side:
            cx = float(np.mean(pts_full[:, 0]))
            cy = float(np.mean(pts_full[:, 1]))
            half = max_box_side // 2
            x1 = int(max(0, min(w - max_box_side, round(cx - half)))) if w > max_box_side else 0
            y1 = int(max(0, min(h - max_box_side, round(cy - half)))) if h > max_box_side else 0
            x2 = int(min(w - 1, x1 + max_box_side - 1))
            y2 = int(min(h - 1, y1 + max_box_side - 1))

        local_box_mask = np.zeros((h, w), dtype=bool)
        local_box_mask[y1:y2 + 1, x1:x2 + 1] = True

        # Keep a small number of prompt points inside the local box.
        pts_prompt = pts_full[
            (pts_full[:, 0] >= x1) & (pts_full[:, 0] <= x2) &
            (pts_full[:, 1] >= y1) & (pts_full[:, 1] <= y2)
        ]
        if len(pts_prompt) == 0:
            pts_prompt = np.array([[float((x1 + x2) / 2), float((y1 + y2) / 2)]], dtype=np.float32)
        if len(pts_prompt) > 8:
            keep_idx = np.linspace(0, len(pts_prompt) - 1, 8).astype(np.int32)
            pts_prompt = pts_prompt[keep_idx]

        point_coords = pts_prompt.astype(np.float32)
        point_labels = np.ones((len(point_coords),), dtype=np.int32)
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Cache SAM image embedding for the current image. This is the real speed win.
        try:
            image_stat = Path(image_path).stat()
            image_key = (str(Path(image_path).resolve()), int(image_stat.st_mtime_ns), int(w), int(h))
        except Exception:
            image_key = (str(image_path), int(w), int(h))

        if getattr(self, "_sam_current_image_key", None) != image_key:
            predictor.set_image(image_rgb)
            self._sam_current_image_key = image_key

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        if masks.ndim != 3 or masks.shape[0] == 0:
            raise RuntimeError(f"SAM trả mask không hợp lệ: {masks.shape}")

        best_idx = 0
        best_rank = None
        for cand_idx in range(masks.shape[0]):
            cand = masks[cand_idx].astype(bool) & local_box_mask
            overlap = int((cand & brush_mask).sum())
            area = int(cand.sum())
            rank = (overlap, float(scores[cand_idx]), -area)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_idx = cand_idx

        idx = int(best_idx)
        sam_mask = masks[idx].astype(bool) & local_box_mask

        # Keep only SAM connected components that touch the user's stroke.
        if sam_mask.any() and brush_mask.any():
            num_labels, labels = cv2.connectedComponents(sam_mask.astype(np.uint8), connectivity=8)
            keep = np.zeros_like(sam_mask, dtype=bool)
            for label_id in range(1, num_labels):
                comp = labels == label_id
                if np.any(comp & brush_mask):
                    keep |= comp
            if keep.any():
                sam_mask = keep

        # Safety: always include the exact user stroke.
        edit_mask = sam_mask | brush_mask

        edited = alpha.copy()
        op = operation.lower().strip()
        if op == "erase":
            edited[edit_mask] = 0
            final = self._refine_alpha_from_user_edit(
                image_path,
                edited,
                force_zero_mask=edit_mask,
                force_one_mask=None,
            ) if refine_with_vitmatte else edited
            source = f"guided_erase_sam_mask_score={float(scores[idx]):.4f}"
        elif op == "restore":
            edited[edit_mask] = 255
            final = self._refine_alpha_from_user_edit(
                image_path,
                edited,
                force_zero_mask=None,
                force_one_mask=edit_mask,
            ) if refine_with_vitmatte else edited
            source = f"guided_restore_sam_mask_score={float(scores[idx]):.4f}"
        else:
            raise ValueError("operation phải là 'erase' hoặc 'restore'.")

        prediction = self.build_prediction_from_alpha(
            image_path=image_path,
            final_alpha=final,
            elapsed_sec=time.time() - start,
            source=source,
        )
        prediction["guided_edit_info"] = {
            "operation": op,
            "sam_score": float(scores[idx]),
            "sam_selected_index": idx,
            "sam_all_scores": [float(x) for x in scores],
            "brush_size": int(brush_size),
            "stroke_points_count": len(stroke_points),
            "sam_prompt_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "sam_prompt_size": [int(x2 - x1 + 1), int(y2 - y1 + 1)],
            "sam_embedding_cached": True,
        }
        return prediction

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
