"""Guided Erase/Restore: a Paint stroke or Box prompt selects an object.

Pipeline for one stroke:
  1. prompt   : stroke -> points + local window, or user box
  2. selection: HQ-SAM instance mask (CPU or CUDA)
  3. alpha    : erase with feathered edge / restore, optionally refined by
                ViTMatte-SAM (GPU only, see GuidedPolicy)

Which steps run is decided once per engine by GuidedPolicy, not by
comparing backend names throughout the code.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from . import matting_ops as ops
from .guided_alpha import (blend_soft_selection, erase_with_feather, hard_erase_mask,
                           selection_trimap_kernels, stroke_mask)
from .guided_selection import instance_support_box, select_prompt_instance, small_object_crop_box
from .box_prompt import normalized_box

# HQ-SAM Paint prompts use a window around the stroke, at most this size.
PAINT_WINDOW_MAX_SIDE = 512
MAX_PROMPT_POINTS = 8


@dataclass(frozen=True)
class GuidedPolicy:
    """What one Guided stroke is allowed to cost on this device.

    crop_small_objects extra HQ-SAM encoder pass on a magnified crop for
                       clicks/short strokes. Cheap on GPU, ~1-2 min on old CPUs.
    refine_each_stroke ViTMatte-SAM matting after every stroke. On CPU the
                       user runs it once via the Matting button instead.
    """
    crop_small_objects: bool
    refine_each_stroke: bool

    @classmethod
    def for_device(cls, uses_gpu: bool, small_object_crop: bool,
                   refine_each_stroke: bool) -> "GuidedPolicy":
        return cls(small_object_crop, refine_each_stroke) if uses_gpu else cls(False, False)


@dataclass
class _Prompt:
    box_mode: bool
    points: np.ndarray          # (N, 2) positive points; empty in Box mode
    window: tuple[int, int, int, int]  # inclusive x1, y1, x2, y2
    brush_mask: np.ndarray


@dataclass
class _Selection:
    mask: np.ndarray            # bool (H, W); empty when nothing was found
    score: float | None
    source: str                 # 'crop' | 'full' | 'box' | 'none'
    raw_mask: np.ndarray | None = None
    support_window: tuple[int, int, int, int] | None = None


def _file_key(image_path) -> tuple:
    """Identifies one version of a file: a re-saved image is decoded/encoded again."""
    path = Path(image_path)
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


class GuidedEditMixin:
    """Requires the engine to provide: settings, guided_policy, strict_gpu,
    _hqsam_predictor(), release_auto_sessions(), _predict_vitmatte_sam_letterbox(),
    build_prediction_from_alpha(), _guided_image_key, _edit_source.
    """

    # ---------------------------------------------------------------- prompt
    @staticmethod
    def _build_prompt(shape_hw, stroke_points, brush_size, prompt_box, windowed: bool) -> _Prompt:
        h, w = shape_hw
        brush = stroke_mask((h, w), stroke_points, brush_size)
        if prompt_box is not None:
            if len(prompt_box) != 4:
                raise ValueError("Guided Box yêu cầu x0,y0,x1,y1.")
            valid_box = normalized_box(prompt_box, w, h)
            if valid_box is None:
                raise ValueError("Guided Box nằm ngoài ảnh hoặc không có diện tích hợp lệ.")
            # The two corners are the real SAM box prompt; no synthetic click.
            return _Prompt(True, np.empty((0, 2), np.float32), valid_box, brush)

        pts = np.array(stroke_points, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            raise ValueError("Guided Paint cần ít nhất một điểm/stroke.")
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pad = max(48, int(brush_size * 4))
        x1 = int(max(0, np.floor(float(np.min(pts[:, 0])) - pad)))
        y1 = int(max(0, np.floor(float(np.min(pts[:, 1])) - pad)))
        x2 = int(min(w - 1, np.ceil(float(np.max(pts[:, 0])) + pad)))
        y2 = int(min(h - 1, np.ceil(float(np.max(pts[:, 1])) + pad)))
        side = PAINT_WINDOW_MAX_SIDE
        if windowed and (x2 - x1 + 1 > side or y2 - y1 + 1 > side):
            # Keep HQ-SAM prompts local: a window centered on the stroke.
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            half = side // 2
            x1 = int(max(0, min(w - side, round(cx - half)))) if w > side else 0
            y1 = int(max(0, min(h - side, round(cy - half)))) if h > side else 0
            x2 = int(min(w - 1, x1 + side - 1))
            y2 = int(min(h - 1, y1 + side - 1))
        inside = pts[(pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)]
        if len(inside) == 0:
            inside = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
        if len(inside) > MAX_PROMPT_POINTS:
            inside = inside[np.linspace(0, len(inside) - 1, MAX_PROMPT_POINTS).astype(np.int32)]
        return _Prompt(False, inside.astype(np.float32), (x1, y1, x2, y2), brush)

    # ------------------------------------------------------------- selection
    def _select_with_hqsam(self, image_rgb, image_key, prompt: _Prompt, brush_size, report) -> _Selection:
        h, w = image_rgb.shape[:2]
        predictor = self._hqsam_predictor()
        if not prompt.box_mode and self.guided_policy.crop_small_objects:
            crop_box = small_object_crop_box((h, w), prompt.points, brush_size)
            if crop_box is not None:
                report(35, "Đang nhận diện vật thể nhỏ (crop)")
                crop_raw, score = predictor.predict_crop(image_rgb, crop_box, prompt.points)
                chosen = select_prompt_instance(crop_raw, prompt.points, brush_size,
                                                crop_box=crop_box, image_hw=(h, w))
                if chosen is not None:
                    return self._with_support(_Selection(chosen, score, 'crop', crop_raw),
                                              brush_size)
                # A crop fragment was rejected: fall back to the full image.

        if self._guided_image_key != image_key:
            report(43, "Đang mã hóa ảnh cho HQ-SAM (lần đầu với ảnh này)")
            predictor.set_image(image_rgb)
            self._guided_image_key = image_key
        report(60, "Đang dự đoán mask bằng HQ-SAM")
        x1, y1, x2, y2 = prompt.window
        if prompt.box_mode:
            raw, score = predictor.predict(box=np.array([x1, y1, x2, y2], np.float32))
            window = np.zeros((h, w), dtype=bool)
            window[y1:y2 + 1, x1:x2 + 1] = True
            return _Selection(raw & window, score, 'box', raw)
        raw, score = predictor.predict(point_coords=prompt.points)
        chosen = select_prompt_instance(raw, prompt.points, brush_size)
        if chosen is None:
            return _Selection(np.zeros((h, w), dtype=bool), score, 'none', raw)
        return self._with_support(_Selection(chosen, score, 'full', raw), brush_size)

    @staticmethod
    def _with_support(selection: _Selection, brush_size: int) -> _Selection:
        """Paint edits may change pixels around the whole instance, not the window."""
        selection.support_window = instance_support_box(selection.mask, brush_size)
        return selection

    # --------------------------------------------------------------- matting
    def _matte_selection(self, image_pil, selection_mask, roi_mask, *, for_hard_erase: bool):
        """ViTMatte-SAM on the selected *instance*, not the full-image alpha.

        Large Restore instances use the notebook's 17/33 px trimap; thin
        instances (and every Erase) use adaptive kernels, otherwise matting
        can lose every sure-foreground pixel. The ROI protects unrelated
        distant objects.
        """
        letterbox, _, meta = ops.letterbox_rgb(image_pil, interpolation=ops.SAM_MATTING_INTERPOLATION)
        mask_768 = ops.to_letterbox(selection_mask, meta, cv2.INTER_NEAREST)
        adaptive = selection_trimap_kernels(mask_768)
        kernels = adaptive if for_hard_erase or adaptive[0] <= 7 else ops.SAM_TRIMAP_KERNELS
        trimap = ops.mask_trimap(mask_768, *kernels)

        soft = ops.unletterbox(self._predict_vitmatte_sam_letterbox(letterbox, trimap), meta)
        allowed = ops.unletterbox((trimap > 0).astype(np.uint8) * 255, meta) > 0
        soft[~(allowed & roi_mask)] = 0.0
        return soft

    # ------------------------------------------------------------- lifecycle
    def edit_source(self, image_path) -> Image.Image:
        """Decoded RGB of the image being edited: decoded once per Edit session.

        Guided strokes, Matting, Manual strokes and Undo/Redo all rebuild the
        cutout from the same pixels; only a different (or re-saved) file is
        decoded again. Callers only read the image, never modify it.
        """
        key = _file_key(image_path)
        cached = self._edit_source
        if cached is None or cached[0] != key:
            cached = (key, ops.load_rgb(image_path))
            self._edit_source = cached
        return cached[1]

    def release_guided_resources(self):
        """Called when leaving Edit (never while a Guided worker runs)."""
        self._edit_source = None  # decoded image: always freed, even if models stay
        if self.settings.keep_model_loaded:
            return
        predictor = getattr(self, '_hqsam', None)
        if predictor is not None:
            predictor.close()
        self._hqsam = None
        self._guided_image_key = None
        self._vitmatte_sam_sess = None
        gc.collect()  # only on leaving Edit, never between strokes

    # ------------------------------------------------------------------ main
    def apply_guided_edit(
        self,
        image_path: str,
        current_alpha: np.ndarray,
        stroke_points: list[tuple[float, float]],
        brush_size: int,
        operation: str,
        prompt_box: tuple[int, int, int, int] | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict:
        """Run one Guided stroke/box; returns a prediction dict with
        ``final_alpha`` and ``guided_edit_info``. Progress values are stages."""
        started = time.time()

        def report(percentage: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(percentage, message)

        op = operation.lower().strip()
        if op not in ("erase", "restore"):
            raise ValueError("operation phải là 'erase' hoặc 'restore'.")
        report(5, "Đang chuẩn bị ảnh và nét vẽ")
        image_path = str(image_path)
        image_pil = self.edit_source(image_path)
        image_rgb = np.asarray(image_pil)  # read-only view for HQ-SAM
        h, w = image_rgb.shape[:2]
        alpha = np.asarray(current_alpha).astype(np.uint8).copy()
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

        policy = self.guided_policy
        prompt = self._build_prompt((h, w), stroke_points, brush_size, prompt_box, windowed=True)
        report(15, "Đang nạp HQ-SAM nếu cần")
        self.release_auto_sessions()
        selection = self._select_with_hqsam(image_rgb, _file_key(image_path), prompt,
                                            brush_size, report)

        roi = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = prompt.window
        if selection.support_window is not None:
            sx0, sy0, sx1, sy1 = selection.support_window
            roi[sy0:sy1, sx0:sx1] = True
        else:
            roi[y1:y2 + 1, x1:x2 + 1] = True

        report(72, "Đang tinh chỉnh vùng chọn")
        selected = selection.mask
        has_selection = bool(selected.any())
        refine = policy.refine_each_stroke and has_selection
        soft = None
        if refine:
            report(77, "Đang tinh chỉnh biên bằng ViTMatte-SAM")
            soft = self._matte_selection(image_pil, selected, roi, for_hard_erase=(op == 'erase'))

        edge_expand_px = 0
        if op == "erase":
            erase_mask = hard_erase_mask(selected, soft, roi) if soft is not None else selected
            report(89, "Đang làm mượt mép vùng xóa")
            # Remove an antialiased rim without leaving a hard rectangular edge.
            edge_expand_px = max(1, min(4, int(round(max(h, w) / 1200.0 * 2))))
            final = erase_with_feather(alpha, erase_mask, edge_expand_px=edge_expand_px)
        elif soft is not None:
            final = blend_soft_selection(alpha, soft, prompt.brush_mask & selected, op)
            # A subpixel object may vanish in the 768 letterbox: keep the mask.
            if not np.any(soft[selected] >= 0.5):
                final[selected] = 255
        else:
            final = alpha.copy()
            final[selected] = 255

        if prompt.box_mode:
            # The drawn rectangle is an explicit edit boundary.
            final[~roi] = alpha[~roi]

        report(95, "Đang cập nhật ảnh và thông tin mask")
        score_text = "" if selection.score is None else f"_score={selection.score:.4f}"
        prediction = self.build_prediction_from_alpha(
            image_path=image_path, final_alpha=final, elapsed_sec=time.time() - started,
            source=f"guided_{op}_hqsam{score_text}")
        if selection.raw_mask is not None:
            # Compressed PNGs for Debug save, instead of extra full-size arrays.
            for key, mask in (("sam_raw_mask_png", selection.raw_mask),
                              ("sam_selected_mask_png", selected)):
                ok, encoded = cv2.imencode('.png', mask.astype(np.uint8) * 255,
                                           [cv2.IMWRITE_PNG_COMPRESSION, 9])
                if ok:
                    prediction[key] = encoded.tobytes()
        prediction["guided_edit_info"] = {
            "operation": op,
            "input_mode": "box" if prompt.box_mode else "paint",
            "brush_size": int(brush_size),
            "stroke_points_count": len(stroke_points),
            "prompt_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "score": selection.score,
            "inference_source": selection.source,
            "matting_refined": bool(refine),
            "erase_edge_expand_px": edge_expand_px,
            "selected_pixels": int(selected.sum()),
            "selection_failed": not has_selection,
        }
        return prediction
