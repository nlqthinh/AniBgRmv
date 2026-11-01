from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np
from PIL import Image


# HQ2 Erase: M7 may extend the raw instance by a *small* supported border,
# but its soft values must never leave half-erased outlines in final alpha.
HQ2_ERASE_M7_THRESHOLD = 0.50
HQ2_ERASE_BORDER_RADIUS_AT_768 = 2.0
HQ2_ERASE_MAX_BORDER_RADIUS = 8


class SamGuidedEditMixin:
    def _load_sam_predictor(
        self,
        sam_checkpoint_path: str,
        model_type: str = "vit_b",
        device: str = "auto",
    ):
        """Lazy-load the selected Guided predictor when first used.

        The old ``sam_checkpoint_path`` argument stays unchanged for the UI;
        it means the HQ2 public base .pt or the legacy ViT-B encoder .onnx.
        """
        from ..app_config import (SAM_BACKEND, SAM_DECODER_ONNX_PATH,
                                  HQ2_BEST_DECODER_PATH, HQ3_BEST_DECODER_PATH,
                                  HQ3_ENCODER_ONNX_PATH, HQ3_DECODER_ONNX_PATH)

        if not hasattr(self, "_sam_cache"):
            self._sam_cache = {}
        encoder = self._resolve_asset_path(sam_checkpoint_path)
        if SAM_BACKEND == "hq3_onnx":
            if model_type != "hq3_large":
                raise RuntimeError("HQ3 ONNX yêu cầu model_type=hq3_large.")
            from .hq3_onnx_runtime import Hq3OnnxPredictor
            encoder = self._resolve_asset_path(HQ3_ENCODER_ONNX_PATH)
            decoder = self._resolve_asset_path(HQ3_DECODER_ONNX_PATH)
            factory = Hq3OnnxPredictor
        elif SAM_BACKEND in ("hq2_torch", "hq3_torch"):
            expected_type = "hq3_large" if SAM_BACKEND == "hq3_torch" else "hq2_large"
            if model_type != expected_type:
                raise RuntimeError("HQ BEST yêu cầu HQ-SAM2 Large, không phải SAM ViT-B.")
            from .hq2_torch_runtime import Hq2TorchPredictor
            # HQ3 notebook retains the HQ-SAM2 decoder checkpoint format;
            # the base architecture/predictor class is unchanged.
            decoder = self._resolve_asset_path(
                HQ3_BEST_DECODER_PATH if SAM_BACKEND == "hq3_torch" else HQ2_BEST_DECODER_PATH)
            factory = Hq2TorchPredictor
        else:
            if model_type != "vit_b":
                raise RuntimeError("Backend SAM ONNX cũ chỉ hỗ trợ vit_b.")
            from .sam_onnx_runtime import SamOnnxPredictor
            decoder = self._resolve_asset_path(SAM_DECODER_ONNX_PATH)
            factory = SamOnnxPredictor
        for path in (encoder, decoder):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy checkpoint {SAM_BACKEND}: {path}. "
                    "HQ2 cần đủ sam2.1_hq_hiera_large.pt và hq2_best_decoder.pt; "
                    "backend cũ cần đủ hai file SAM ViT-B ONNX."
                )
        forced_device = getattr(self, "sam_device", None)
        if forced_device is not None:
            device = forced_device
        elif device == "auto":
            device = "cpu"
        key = (SAM_BACKEND, str(encoder.resolve()), str(decoder.resolve()), model_type, device)
        if key not in self._sam_cache:
            self._sam_cache[key] = factory(encoder, decoder, device)
            self._sam_current_image_key = None
        return self._sam_cache[key]

    def release_sam_resources(self):
        """Called on leaving Edit, never during an active Guided worker.

        ONNX Runtime and PyTorch retain native/GPU memory while models live.
        The optional HQ2 weight cache trades between-Edit memory for a faster
        next Edit. Other backends still unload on exit as before.
        """
        import gc
        from ..app_config import APP_CONFIG

        keep_hq2 = bool(APP_CONFIG['sam'].get('hq2_keep_model_loaded', False))
        cache = getattr(self, '_sam_cache', {})
        for key, predictor in list(cache.items()):
            if keep_hq2 and key[0] in ('hq2_torch', 'hq3_torch', 'hq3_onnx'):
                continue
            try:
                predictor.close()
            except Exception:
                pass
            del cache[key]
        if not keep_hq2 and getattr(self, "_m71_sess", None) is not None:
            self._m71_sess = None
            self._m71_input_name = None
        if not cache:
            self._sam_current_image_key = None
            gc.collect()  # only at transition out of Edit; not during strokes
        if hasattr(self, 'log_sam_vram'):
            self.log_sam_vram('after leaving Edit / release HQ3 and M71')

    # GrabCut is a classical CPU segmentation algorithm. Running it on a
    # multi-megapixel ROI per stroke makes CPU Guided feel frozen. Limit only
    # its *working resolution*, never the output mask or original image.
    @staticmethod
    def _cpu_grabcut_working_view(roi, strokes=None, max_side=896):
        """Return compact RGB/stroke arrays and original ROI shape.

        INTER_AREA preserves thin positive paint seeds when downscaling;
        binary nearest-neighbor is used later when restoring label masks.
        The model is classical GrabCut, not ViTMatte/M71.
        """
        h, w = roi.shape[:2]
        limit = max(64, int(max_side))
        scale = min(1.0, limit / max(h, w))
        if scale == 1.0:
            return roi, strokes, (h, w)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        rgb_small = cv2.resize(roi, new_size, interpolation=cv2.INTER_AREA)
        stroke_small = None
        if strokes is not None:
            # For binary strokes, positive partial coverage must not vanish.
            stroke_small = (cv2.resize(strokes.astype(np.uint8) * 255,
                         new_size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        return rgb_small, stroke_small, (h, w)

    @staticmethod
    def _cpu_grabcut_box_mask(image_rgb, box_xyxy):
        """Choose foreground inside a user box, bounding CPU work on large ROIs.

        A failed GrabCut returns an empty mask, NOT a full rectangular erase.
        Manual Box is the explicit tool for modifying an entire rectangle.
        """
        h, w = image_rgb.shape[:2]
        x0, y0, x1, y1 = (int(v) for v in box_xyxy)
        result = np.zeros((h, w), dtype=bool)
        if x1 <= x0 or y1 <= y0:
            return result
        margin = max(8, min(64, max(x1-x0+1, y1-y0+1)//5))
        rx0, ry0 = max(0, x0-margin), max(0, y0-margin)
        rx1, ry1 = min(w, x1+margin+1), min(h, y1+margin+1)
        roi = np.ascontiguousarray(image_rgb[ry0:ry1, rx0:rx1], dtype=np.uint8)
        bx, by = x0-rx0, y0-ry0
        bw, bh = x1-x0+1, y1-y0+1
        if min(roi.shape[:2]) < 8 or bw < 3 or bh < 3:
            return result
        if bx <= 0 or by <= 0 or bx+bw >= roi.shape[1] or by+bh >= roi.shape[0]:
            # A box touching the image boundary can still be editable by
            # Manual Box; GrabCut needs known background outside its box.
            return result
        working, _, orig_hw = SamGuidedEditMixin._cpu_grabcut_working_view(roi)
        sh, sw = working.shape[:2]
        ratio_x = sw / orig_hw[1]
        ratio_y = sh / orig_hw[0]
        xb = max(1, round(bx * ratio_x))
        yb = max(1, round(by * ratio_y))
        xend = min(sw - 1, round((bx+bw) * ratio_x))
        yend = min(sh - 1, round((by+bh) * ratio_y))
        if xend - xb < 3 or yend - yb < 3:
            return result
        gc = np.zeros(working.shape[:2], dtype=np.uint8)
        bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
        try:
            cv2.grabCut(working, gc, (xb, yb, xend-xb, yend-yb),
                        bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return result
        selected = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        if orig_hw != working.shape[:2]:
            selected = cv2.resize(selected, (orig_hw[1], orig_hw[0]),
                                  interpolation=cv2.INTER_NEAREST)
        selected[:by, :] = 0
        selected[by+bh:, :] = 0
        selected[:, :bx] = 0
        selected[:, bx+bw:] = 0
        result[ry0:ry1, rx0:rx1] = selected.astype(bool)
        return result

    @staticmethod
    def _cpu_grabcut_mask(image_rgb, brush_mask, box_xyxy):
        """CPU Paint GrabCut on a bounded working raster; preserve all strokes.

        Unlike HQ3, GrabCut can miss large/non-uniform objects. Even if it
        fails, manual brush coverage is still enforced in this legacy branch.
        """
        x1, y1, x2, y2 = (int(v) for v in box_xyxy)
        roi = np.ascontiguousarray(image_rgb[y1:y2+1, x1:x2+1], dtype=np.uint8)
        strokes = np.asarray(brush_mask[y1:y2+1, x1:x2+1], dtype=np.uint8)
        output = np.zeros(brush_mask.shape, dtype=bool)
        working, seed, orig_hw = SamGuidedEditMixin._cpu_grabcut_working_view(roi, strokes)
        h, w = working.shape[:2]
        if min(h, w) < 8 or seed is None or not seed.any():
            output[y1:y2+1, x1:x2+1] = strokes.astype(bool)
            return output
        # Seed probable FG around the paint; enforce positive seed pixels.
        r = max(2, min(12, int(max(1, np.count_nonzero(seed)**0.5) / 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        probable_fg = cv2.dilate(seed, kernel) > 0
        trimap = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        trimap[probable_fg] = cv2.GC_PR_FGD
        trimap[seed > 0] = cv2.GC_FGD
        # Reserve background at the perimeter, without overwriting explicit
        # paint seeds. Otherwise long strokes touching the image edge vanish.
        for ys, xs in ((slice(0,2),slice(None)), (slice(-2,None),slice(None)),
                       (slice(None),slice(0,2)), (slice(None),slice(-2,None))):
            region = trimap[ys, xs]
            region[seed[ys, xs] == 0] = cv2.GC_BGD
        if not np.any(trimap == cv2.GC_BGD) or not np.any(trimap == cv2.GC_FGD):
            output[y1:y2+1, x1:x2+1] = strokes.astype(bool)
            return output
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(working, trimap, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
            mask = ((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)).astype(np.uint8)
            count, labels = cv2.connectedComponents(mask, connectivity=8)
            touched = np.unique(labels[seed > 0])
            touched = touched[touched != 0]
            mask = np.isin(labels, touched) if len(touched) else seed.astype(bool)
        except cv2.error:
            mask = seed.astype(bool)
        if mask.shape != orig_hw:
            mask = cv2.resize(mask.astype(np.uint8), (orig_hw[1],orig_hw[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        # Resizing must not discard the user's exact input pixels.
        output[y1:y2+1, x1:x2+1] = mask | strokes.astype(bool)
        return output

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

        # Legacy optional path: use the same centered letterbox transform as
        # the real F3/M7 pipeline; _predict_m6_alpha_768 no longer exists.
        letterbox, valid, meta = self._letterbox_rgb_for_m7(image_pil)
        x, y = meta["pad_left"], meta["pad_top"]
        rw, rh = meta["resized_width"], meta["resized_height"]
        alpha_768 = np.zeros((self.size, self.size), np.uint8)
        alpha_768[y:y + rh, x:x + rw] = cv2.resize(
            alpha, (rw, rh), interpolation=cv2.INTER_LINEAR
        )
        trimap_768 = self.make_loose_trimap(alpha_768.astype(np.float32) / 255.0,
                                             valid_mask=valid)
        m7_alpha_768 = self._predict_m7_alpha_letterbox(letterbox, trimap_768)
        final = self._restore_alpha_from_letterbox(m7_alpha_768, meta)

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

    def _refine_hq2_selection_with_m7(
        self,
        image_pil: Image.Image,
        selected_mask: np.ndarray,
        local_box_mask: np.ndarray,
        *,
        for_hard_erase: bool = False,
    ) -> np.ndarray:
        """V0 SAM+M7: refine the selected *instance*, not the full-image alpha.

        Large Restore instances retain V0's 17/33px trimap. Tiny/thin masks
        get smaller kernels for both operations; otherwise M7 can lose every
        sure-foreground pixel and grow a very wide uncertain border.
        The padded instance ROI protects unrelated distant characters.
        """
        letterbox, valid, meta = self._letterbox_rgb_for_m7(image_pil)
        rw, rh = meta["resized_width"], meta["resized_height"]
        x, y = meta["pad_left"], meta["pad_top"]
        mask_768 = np.zeros((self.size, self.size), np.uint8)
        mask_768[y:y + rh, x:x + rw] = cv2.resize(
            selected_mask.astype(np.uint8), (rw, rh),
            interpolation=cv2.INTER_NEAREST,
        )
        adaptive_erode, adaptive_dilate = self._hq2_erase_trimap_kernel_sizes(mask_768)
        erode_size, dilate_size = (
            (adaptive_erode, adaptive_dilate)
            if for_hard_erase or adaptive_erode <= 7 else (17, 33)
        )
        inside = cv2.erode(mask_768, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size, erode_size))) > 0
        support = cv2.dilate(mask_768, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))) > 0
        trimap = np.full((self.size, self.size), 128, dtype=np.uint8)
        trimap[inside] = 255
        trimap[~support] = 0
        trimap[valid == 0] = 0

        soft_768 = (self._predict_m71_alpha_letterbox(letterbox, trimap)
                    if getattr(self, "sam_backend", None) in ("hq3_torch", "hq3_onnx")
                    else self._predict_m7_alpha_letterbox(letterbox, trimap))
        soft = self._restore_alpha_from_letterbox(soft_768, meta).astype(np.float32) / 255.0
        allowed = self._restore_alpha_from_letterbox(
            (support.astype(np.uint8) * 255), meta) > 0
        soft[~(allowed & local_box_mask)] = 0.0
        return np.clip(soft, 0.0, 1.0)

    @staticmethod
    def _hq2_erase_trimap_kernel_sizes(mask_768: np.ndarray) -> tuple[int, int]:
        """Adapt the Erase trimap to thin instances; Restore stays at 17/33.

        The maximum inscribed radius is more useful than a bounding box for
        narrow objects, e.g. a sword crossing the entire image diagonally.
        """
        if not np.any(mask_768):
            return 1, 5
        distance = cv2.distanceTransform(mask_768.astype(np.uint8), cv2.DIST_L2, 3)
        radius = float(distance.max())
        erode_radius = max(0, min(8, int(radius - 1.0)))
        dilate_radius = max(2, min(16, int(round(radius * 2.0))))
        return erode_radius * 2 + 1, dilate_radius * 2 + 1

    @staticmethod
    def _hq2_m7_hard_erase_mask(
        hq2_and_brush_mask: np.ndarray,
        m7_soft_selection: np.ndarray,
        local_box_mask: np.ndarray,
    ) -> np.ndarray:
        """Keep all HQ2/brush pixels; add only confident M7 pixels near them.

        M7 supports the boundary, never decides a fractional amount of Erase.
        Limiting its extension avoids swallowing an unrelated nearby object.
        """
        core = np.asarray(hq2_and_brush_mask, dtype=bool)
        soft = np.asarray(m7_soft_selection, dtype=np.float32)
        roi = np.asarray(local_box_mask, dtype=bool)
        if core.ndim != 2 or soft.shape != core.shape or roi.shape != core.shape:
            raise ValueError("Mask HQ2, M7 và ROI phải cùng kích thước H×W.")
        if not np.isfinite(soft).all():
            raise ValueError("M7 trả về alpha không hữu hạn.")
        if not core.any():
            return core.copy()

        height, width = core.shape
        radius = max(1, min(HQ2_ERASE_MAX_BORDER_RADIUS, int(round(
            HQ2_ERASE_BORDER_RADIUS_AT_768 * max(height, width) / 768.0))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * radius + 1, 2 * radius + 1))
        # M47 memory rule: never dilate an entire 4K/8K mask for a small
        # Guided stroke. Limit all extra buffers to the established local ROI.
        rows = np.flatnonzero(roi.any(axis=1))
        cols = np.flatnonzero(roi.any(axis=0))
        if not rows.size or not cols.size:
            return core.copy()
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        region = np.s_[y0:y1, x0:x1]
        nearby = cv2.dilate(core[region].astype(np.uint8), kernel) > 0
        result = core.copy()
        result[region] |= (soft[region] >= HQ2_ERASE_M7_THRESHOLD) & nearby & roi[region]
        return result

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
        progress_callback: Callable[[int, str], None] | None = None,
        prompt_box: tuple[int, int, int, int] | None = None,
    ) -> dict:
        """
        Guided Erase / Restore:
        1. Use the configured backend to select the prompted object.
        2. For HQ2, refine that instance with M7 (when enabled).
        3. Apply hard-center/antialiased-boundary Erase or soft Restore.

        HQ2 speed strategy: infer small dab crops first; a valid crop skips the
        extra full-image encoder pass. Longer strokes and rejected crop masks
        use the cached full-image embedding (computed lazily on first need).
        Progress percentages describe stages, not measured GPU completion.
        """
        start = time.time()

        def report(percentage: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(percentage, message)

        report(5, "Đang chuẩn bị ảnh và nét vẽ")

        image_path = str(image_path)
        image_pil = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image_pil, dtype=np.uint8, copy=True)
        h, w = image_rgb.shape[:2]

        alpha = np.asarray(current_alpha).astype(np.uint8).copy()
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

        # CPU GrabCut needs the whole Paint stroke, not the 512px HQ3 ROI cap.
        use_fast_cpu = (
            not getattr(self, 'strict_gpu', False)
            and getattr(self, 'cpu_guided_backend', 'sam') == 'grabcut'
        )
        box_mode = prompt_box is not None
        brush_mask = self._stroke_mask((h, w), stroke_points, brush_size)
        if box_mode:
            coords = [int(v) for v in prompt_box]
            if len(coords) != 4:
                raise ValueError("Guided Box yêu cầu x0,y0,x1,y1.")
            x1, x2 = sorted((max(0, min(w-1, coords[0])), max(0, min(w-1, coords[2]))))
            y1, y2 = sorted((max(0, min(h-1, coords[1])), max(0, min(h-1, coords[3]))))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Guided Box cần vùng có chiều rộng và cao > 0.")
            # In Box/Guided, no synthetic center click or fake brush area is
            # sent to HQ3. The two corners are the actual SAM box prompt.
            point_coords = np.empty((0, 2), dtype=np.float32)
            point_labels = np.empty((0,), dtype=np.int32)
        else:
            pts_full = np.array(stroke_points, dtype=np.float32).reshape(-1, 2)
            if pts_full.size == 0:
                raise ValueError("Guided Paint cần ít nhất một điểm/stroke.")
            pts_full[:, 0] = np.clip(pts_full[:, 0], 0, w - 1)
            pts_full[:, 1] = np.clip(pts_full[:, 1], 0, h - 1)

            pad = max(48, int(brush_size * 4))
            x1 = int(max(0, np.floor(float(np.min(pts_full[:, 0])) - pad)))
            y1 = int(max(0, np.floor(float(np.min(pts_full[:, 1])) - pad)))
            x2 = int(min(w - 1, np.ceil(float(np.max(pts_full[:, 0])) + pad)))
            y2 = int(min(h - 1, np.ceil(float(np.max(pts_full[:, 1])) + pad)))

            # Retain the existing Paint 512 px locality limit ONLY for paint.
            max_box_side = 512
            if (not use_fast_cpu and
                    (x2 - x1 + 1 > max_box_side or y2 - y1 + 1 > max_box_side)):
                cx = float(np.mean(pts_full[:, 0]))
                cy = float(np.mean(pts_full[:, 1]))
                half = max_box_side // 2
                x1 = int(max(0, min(w - max_box_side, round(cx - half)))) if w > max_box_side else 0
                y1 = int(max(0, min(h - max_box_side, round(cy - half)))) if h > max_box_side else 0
                x2 = int(min(w - 1, x1 + max_box_side - 1))
                y2 = int(min(h - 1, y1 + max_box_side - 1))
            pts_prompt = pts_full[
                (pts_full[:, 0] >= x1) & (pts_full[:, 0] <= x2) &
                (pts_full[:, 1] >= y1) & (pts_full[:, 1] <= y2)
            ]
            if len(pts_prompt) == 0:
                pts_prompt = np.array([[(x1+x2)/2, (y1+y2)/2]], dtype=np.float32)
            if len(pts_prompt) > 8:
                pts_prompt = pts_prompt[np.linspace(0, len(pts_prompt)-1, 8).astype(np.int32)]
            point_coords = pts_prompt.astype(np.float32)
            point_labels = np.ones((len(point_coords),), dtype=np.int32)

        local_box_mask = np.zeros((h, w), dtype=bool)
        local_box_mask[y1:y2+1, x1:x2+1] = True
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Cache SAM image embedding for the current image. This is the real speed win.
        try:
            image_stat = Path(image_path).stat()
            image_key = (str(Path(image_path).resolve()), int(image_stat.st_mtime_ns), int(w), int(h))
        except Exception:
            image_key = (str(image_path), int(w), int(h))

        from ..app_config import APP_CONFIG, SAM_BACKEND
        embedding_reused = False
        hq2_source = None
        raw_hq2_mask = None
        crop_selected = None
        if use_fast_cpu:
            report(30, "Đang chọn vật thể bằng GrabCut (CPU)")
            selected_backend = 'grabcut_cpu_fast'
            masks = (self._cpu_grabcut_box_mask(image_rgb, (x1, y1, x2, y2))
                     if box_mode else self._cpu_grabcut_mask(
                         image_rgb, brush_mask, (x1, y1, x2, y2)))[None, ...]
            scores = np.array([0.0], dtype=np.float32)
        else:
            selected_backend = SAM_BACKEND
            report(15, "Đang nạp mô hình SAM nếu cần")
            # Before allocating HQ-SAM2 Large, release already idle CUDA
            # F3/M7 sessions. The finished image alpha and previews remain.
            if selected_backend in ('hq3_torch', 'hq3_onnx') and hasattr(self, '_release_auto_sessions_for_hq3'):
                self._release_auto_sessions_for_hq3()
            predictor = self._load_sam_predictor(
                sam_checkpoint_path=sam_checkpoint_path,
                model_type=sam_model_type,
                device=sam_device,
            )
            if hasattr(self, 'log_sam_vram'):
                self.log_sam_vram('after HQ3 model load' if selected_backend in ('hq3_torch', 'hq3_onnx') else 'after SAM load')
            embedding_reused = getattr(self, "_sam_current_image_key", None) == image_key

            if (not box_mode and selected_backend in ('hq2_torch', 'hq3_torch', 'hq3_onnx')
                    and APP_CONFIG['sam'].get('hq2_small_object_crop', True)
                    and hasattr(predictor, 'predict_crop')):
                from .hq2_guided_selection import (
                    select_prompt_instance, small_object_crop_box,
                )
                crop_box = small_object_crop_box((h, w), point_coords, brush_size)
                if crop_box is not None:
                    # For a tiny star, an independently magnified crop is
                    # already the preferred HQ2 result. No need to encode
                    # the full photo first and discard that prediction.
                    report(35, "Đang nhận diện vật thể nhỏ (HQ2 crop)")
                    crop_masks, crop_scores = predictor.predict_crop(
                        image_rgb, crop_box, point_coords, point_labels)
                    crop_raw = np.asarray(crop_masks[0], dtype=bool)
                    raw_hq2_mask = crop_raw
                    crop_selected = select_prompt_instance(
                        crop_raw, point_coords, brush_size,
                        crop_box=crop_box, image_hw=(h, w))
                    if crop_selected is not None:
                        masks, scores = crop_masks, crop_scores
                        hq2_source = 'crop'

            if crop_selected is None:
                if not embedding_reused:
                    report(43, "Đang mã hóa ảnh cho SAM (lần đầu có thể lâu)")
                    from ..disk_cache import load_sam_embedding, save_sam_embedding
                    # ONNX supports disk cache; HQ2 stores full-image features
                    # in its predictor only while that image/Edit is active.
                    cached = (load_sam_embedding(image_path, str(predictor.encoder_path))
                              if hasattr(predictor, 'encoder_path') else None)
                    if cached is not None and tuple(cached[1]) == (h, w):
                        predictor.set_cached_embedding(*cached)
                    else:
                        predictor.set_image(image_rgb)
                        if hasattr(predictor, 'encoder_path'):
                            save_sam_embedding(image_path, str(predictor.encoder_path), predictor)
                    self._sam_current_image_key = image_key
                report(60, "Đang dự đoán mask bằng SAM")
                masks, scores, _ = predictor.predict(
                    point_coords=None if box_mode else point_coords,
                    point_labels=None if box_mode else point_labels,
                    box=box if box_mode or selected_backend not in (
                        'hq2_torch', 'hq3_torch', 'hq3_onnx') else None,
                    multimask_output=True,
                )

        if masks.ndim != 3 or masks.shape[0] == 0:
            raise RuntimeError(f"SAM trả mask không hợp lệ: {masks.shape}")

        idx = 0
        if selected_backend in ('hq2_torch', 'hq3_torch', 'hq3_onnx'):
            from .hq2_guided_selection import (
                instance_support_box, select_prompt_instance,
            )
            # Box uses the actual SAM box tokens; do not run point-based
            # instance selection (there are no brush points in Box mode).
            if crop_selected is not None:
                selected = crop_selected
            else:
                full_raw_mask = masks[0].astype(bool)
                selected = (full_raw_mask & local_box_mask) if box_mode else (
                    select_prompt_instance(full_raw_mask, point_coords, brush_size))
                hq2_source = 'box' if box_mode else ('full' if selected is not None else 'none')
                if selected is not None or raw_hq2_mask is None:
                    raw_hq2_mask = full_raw_mask

            sam_mask = (selected if selected is not None
                        else np.zeros((h, w), dtype=bool))
            if sam_mask.any() and not box_mode:
                # Preserve old paint behavior, but Box stays inside the
                # rectangle explicitly drawn by the user.
                x0, y0, x3, y3 = instance_support_box(sam_mask, brush_size)
                local_box_mask[:] = False
                local_box_mask[y0:y3, x0:x3] = True
                x1, y1, x2, y2 = x0, y0, x3 - 1, y3 - 1
        else:
            # CPU GrabCut and original ViT-B ONNX retain their proven behavior.
            best_rank = None
            for cand_idx in range(masks.shape[0]):
                cand = masks[cand_idx].astype(bool) & local_box_mask
                overlap = int((cand & (local_box_mask if box_mode else brush_mask)).sum())
                area = int(cand.sum())
                rank = (overlap, float(scores[cand_idx]), -area)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    idx = cand_idx
            sam_mask = masks[idx].astype(bool) & local_box_mask
            if sam_mask.any() and brush_mask.any() and not box_mode:
                num_labels, labels = cv2.connectedComponents(
                    sam_mask.astype(np.uint8), connectivity=8)
                keep = np.zeros_like(sam_mask, dtype=bool)
                for label_id in range(1, num_labels):
                    comp = labels == label_id
                    if np.any(comp & brush_mask):
                        keep |= comp
                if keep.any():
                    sam_mask = keep

        if selected_backend in ('hq3_torch', 'hq3_onnx') and hasattr(self, 'log_sam_vram'):
            self.log_sam_vram('after HQ3 crop/full inference, before M71')
        report(72, "Đang tinh chỉnh vùng chọn")
        # HQ2's instance is authoritative: an oversized circular click cannot
        # alter pixels outside a correctly predicted tiny star. If no valid
        # HQ2 instance exists, leave the alpha unchanged instead of stamping a
        # circle and pretending it came from SAM. Manual brush remains exact.
        edit_mask = (sam_mask if selected_backend in ('hq2_torch', 'hq3_torch', 'hq3_onnx')
                     else (sam_mask | brush_mask))
        hard_constraint_mask = (brush_mask & sam_mask
                                if selected_backend in ('hq2_torch', 'hq3_torch', 'hq3_onnx') else brush_mask)

        op = operation.lower().strip()
        edge_expand_px = 0
        # Both operations use HQ2 + M7 when enabled; their final alpha differs.
        # Erase binarizes the supported selection, Restore keeps the soft matte.
        hq2_m7 = (selected_backend in ('hq2_torch', 'hq3_torch', 'hq3_onnx') and edit_mask.any() and
                  bool(APP_CONFIG['sam'].get('hq2_refine_with_m7', True)))
        if hq2_m7:
            matte_label = "M71" if selected_backend in ("hq3_torch", "hq3_onnx") else "M7"
            report(77, f"Đang tinh chỉnh biên bằng {matte_label}")
        soft_selection = (self._refine_hq2_selection_with_m7(
            image_pil, edit_mask, local_box_mask, for_hard_erase=(op == 'erase'))
                          if hq2_m7 else None)
        if op == "erase":
            erase_mask = (self._hq2_m7_hard_erase_mask(
                edit_mask, soft_selection, local_box_mask)
                if soft_selection is not None else edit_mask)
            report(89, "Đang làm mượt mép vùng xóa")
            if selected_backend in ('hq2_torch', 'hq3_torch', 'hq3_onnx'):
                from .hq2_erase_edges import erase_hq2_alpha
                # HQ2/M7 may stop just inside the real antialiased contour.
                # Remove that narrow original-color rim before softening the
                # *new* outside boundary; cap growth to avoid nearby objects.
                edge_expand_px = max(1, min(4, int(round(max(h, w) / 1200.0 * 2))))
                edited = erase_hq2_alpha(
                    alpha, erase_mask, edge_expand_px=edge_expand_px)
            else:
                edited = alpha.copy()
                edited[erase_mask] = 0
            # Even an explicitly requested legacy ViTMatte pass must not run
            # for HQ2 Erase. Preserve that option for older Guided backends.
            final = self._refine_alpha_from_user_edit(
                image_path, edited, force_zero_mask=edit_mask,
            ) if refine_with_vitmatte and selected_backend not in ('hq2_torch', 'hq3_torch') else edited
            source = f"guided_erase_{selected_backend}_score={float(scores[idx]):.4f}"
        elif op == "restore":
            edited = alpha.copy()
            if soft_selection is not None:
                from .hq2_torch_runtime import blend_hq2_m7_edit
                edited = blend_hq2_m7_edit(alpha, soft_selection, hard_constraint_mask, op)
                # If a subpixel-sized object vanished in M7's 768 letterbox,
                # retain HQ2's actual object rather than restoring nothing.
                if not np.any(soft_selection[sam_mask] >= 0.5):
                    edited[sam_mask] = 255
                final = edited
            else:
                edited[edit_mask] = 255
                final = self._refine_alpha_from_user_edit(
                    image_path, edited, force_one_mask=edit_mask,
                ) if refine_with_vitmatte else edited
            source = f"guided_restore_{selected_backend}_score={float(scores[idx]):.4f}"
        else:
            raise ValueError("operation phải là 'erase' hoặc 'restore'.")

        if box_mode:
            # Box is an explicit edit boundary. Edge expansion/soft matte must
            # not alter pixels just outside the rectangle drawn by the user.
            final = np.asarray(final).copy()
            final[~local_box_mask] = alpha[~local_box_mask]

        report(95, "Đang cập nhật ảnh và thông tin mask")
        prediction = self.build_prediction_from_alpha(
            image_path=image_path,
            final_alpha=final,
            elapsed_sec=time.time() - start,
            source=source,
            include_debug=False,
        )
        if raw_hq2_mask is not None:
            # Store compressed PNGs, not extra 4K/8K bool arrays, until the
            # user clicks Save with Debug ON. Both are in full image coords.
            for key, mask in (("hq2_raw_mask_png", raw_hq2_mask),
                              ("hq2_selected_mask_png", sam_mask)):
                ok, encoded = cv2.imencode(
                    '.png', mask.astype(np.uint8) * 255,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
                if ok:
                    prediction[key] = encoded.tobytes()
        prediction["guided_edit_info"] = {
            "operation": op,
            "guided_backend": selected_backend,
            "sam_score": float(scores[idx]) if not use_fast_cpu else None,
            "sam_selected_index": idx,
            "sam_all_scores": [float(x) for x in scores],
            "brush_size": int(brush_size),
            "stroke_points_count": len(stroke_points),
            "input_mode": "box" if box_mode else "paint",
            "user_box_xyxy": list(map(int, box)) if box_mode else None,
            "sam_prompt_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "sam_prompt_size": [int(x2 - x1 + 1), int(y2 - y1 + 1)],
            "sam_embedding_cached": embedding_reused,
            "hq2_m7_refined": hq2_m7,
            "hq2_m7_hard_erase": bool(hq2_m7 and op == 'erase' and selected_backend == "hq2_torch"),
            "hq3_m71_refined": bool(hq2_m7 and selected_backend in ("hq3_torch", "hq3_onnx")),
            "hq3_m71_hard_erase": bool(hq2_m7 and op == 'erase' and selected_backend in ("hq3_torch", "hq3_onnx")),
            "hq2_erase_edge_expand_px": edge_expand_px,
            "hq2_inference_source": hq2_source,
            "hq2_selection_failed": bool(selected_backend in ('hq2_torch', 'hq3_torch', 'hq3_onnx') and
                                         not edit_mask.any()),
        }
        return prediction

