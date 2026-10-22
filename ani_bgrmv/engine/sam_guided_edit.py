from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


class SamGuidedEditMixin:
    def _load_sam_predictor(
        self,
        sam_checkpoint_path: str,
        model_type: str = "vit_b",
        device: str = "auto",
    ):
        """Lazy-load SAM ONNX encoder/decoder when Guided Edit is first used.

        The public parameter name sam_checkpoint_path is preserved for the
        unchanged UI/worker API; it now points to sam.encoder_onnx_path.
        """
        from ..app_config import SAM_DECODER_ONNX_PATH
        from .sam_onnx_runtime import SamOnnxPredictor

        if model_type != "vit_b":
            raise RuntimeError('Bản ONNX này hỗ trợ SAM vit_b. Hãy export model vit_b tương ứng.')
        if not hasattr(self, "_sam_cache"):
            self._sam_cache = {}
        encoder = self._resolve_asset_path(sam_checkpoint_path)
        decoder = self._resolve_asset_path(SAM_DECODER_ONNX_PATH)
        for path in (encoder, decoder):
            if not path.is_file():
                raise FileNotFoundError(
                    f'Không tìm thấy SAM ONNX: {path}. '
                    'Hãy export đủ sam_vit_b_encoder.onnx và sam_vit_b_decoder.onnx.'
                )
        forced_device = getattr(self, "sam_device", None)
        if forced_device is not None:
            device = forced_device
        elif device == "auto":
            device = "cpu"
        key = (str(encoder.resolve()), str(decoder.resolve()), model_type, device)
        if key not in self._sam_cache:
            self._sam_cache[key] = SamOnnxPredictor(encoder, decoder, device)
            self._sam_current_image_key = None
        return self._sam_cache[key]

    def release_sam_resources(self):
        """Called on leaving Edit, never during an active Guided worker.

        ONNX Runtime keeps CUDA/CPU memory arenas as long as sessions live.
        Unloading SAM here preserves F3/ViTMatte and frees the large encoder
        sessions. SAM is lazy-loaded again on next Edit.
        """
        import gc
        cache = getattr(self, '_sam_cache', {})
        for predictor in cache.values():
            try:
                predictor.close()
            except Exception:
                pass
        cache.clear()
        self._sam_current_image_key = None
        gc.collect()  # only at transition out of Edit; not during brush strokes

    @staticmethod
    def _cpu_grabcut_mask(image_rgb, brush_mask, box_xyxy):
        """Fast approximate Guided selector for CPU-only/weak-PC builds.

        GrabCut learns foreground/background colors inside a small ROI. It is
        *not* SAM and can produce different shapes; the brush itself is always
        included by the caller. It never infers on the full 1024x1024 ViT.
        """
        x1, y1, x2, y2 = (int(v) for v in box_xyxy)
        roi = np.ascontiguousarray(image_rgb[y1:y2+1, x1:x2+1], dtype=np.uint8)
        strokes = np.asarray(brush_mask[y1:y2+1, x1:x2+1], dtype=np.uint8)
        h, w = roi.shape[:2]
        output = np.zeros(brush_mask.shape, dtype=bool)
        if min(h, w) < 8 or not strokes.any():
            output[y1:y2+1, x1:x2+1] = strokes.astype(bool)
            return output
        # Use definite FG only where the user brushed; probable FG in a
        # narrow neighborhood; definite background at ROI perimeter.
        r = max(3, min(15, int(max(1, np.count_nonzero(strokes)**0.5) / 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        probable_fg = cv2.dilate(strokes, kernel) > 0
        trimap = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        trimap[probable_fg] = cv2.GC_PR_FGD
        trimap[strokes > 0] = cv2.GC_FGD
        # Border represents outside of the selected local region.
        trimap[:2, :] = cv2.GC_BGD
        trimap[-2:, :] = cv2.GC_BGD
        trimap[:, :2] = cv2.GC_BGD
        trimap[:, -2:] = cv2.GC_BGD
        if not np.any(trimap == cv2.GC_FGD):
            output[y1:y2+1, x1:x2+1] = strokes.astype(bool)
            return output
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(roi, trimap, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
            mask = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
            # Keep only components touching the brush. Avoid swallowing a
            # distant similarly-colored object inside the prompt box.
            n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            touched = np.unique(labels[strokes > 0])
            touched = touched[touched != 0]
            mask = np.isin(labels, touched) if len(touched) else strokes.astype(bool)
        except cv2.error:
            # Even a flat-color image remains editable instead of throwing.
            mask = strokes.astype(bool)
        output[y1:y2+1, x1:x2+1] = mask
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

        use_fast_cpu = (
            not getattr(self, 'strict_gpu', False)
            and getattr(self, 'cpu_guided_backend', 'sam') == 'grabcut'
        )
        if use_fast_cpu:
            selected_backend = 'grabcut_cpu_fast'
            masks = self._cpu_grabcut_mask(image_rgb, brush_mask, (x1, y1, x2, y2))[None, ...]
            scores = np.array([0.0], dtype=np.float32)
        else:
            selected_backend = 'sam_onnx'
            predictor = self._load_sam_predictor(
                sam_checkpoint_path=sam_checkpoint_path,
                model_type=sam_model_type,
                device=sam_device,
            )
            if getattr(self, "_sam_current_image_key", None) != image_key:
                from ..disk_cache import load_sam_embedding, save_sam_embedding
                # Reuse GPU encoder outputs across Edit sessions without keeping
                # the full ONNX encoder alive in RAM while editing is closed.
                cached = load_sam_embedding(image_path, str(predictor.encoder_path)) if hasattr(predictor, 'encoder_path') else None
                if cached is not None and tuple(cached[1]) == (h, w):
                    predictor.set_cached_embedding(*cached)
                else:
                    predictor.set_image(image_rgb)
                    if hasattr(predictor, 'encoder_path'):
                        save_sam_embedding(image_path, str(predictor.encoder_path), predictor)
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
            source = f"guided_erase_{selected_backend}_score={float(scores[idx]):.4f}"
        elif op == "restore":
            edited[edit_mask] = 255
            final = self._refine_alpha_from_user_edit(
                image_path,
                edited,
                force_zero_mask=None,
                force_one_mask=edit_mask,
            ) if refine_with_vitmatte else edited
            source = f"guided_restore_{selected_backend}_score={float(scores[idx]):.4f}"
        else:
            raise ValueError("operation phải là 'erase' hoặc 'restore'.")

        prediction = self.build_prediction_from_alpha(
            image_path=image_path,
            final_alpha=final,
            elapsed_sec=time.time() - start,
            source=source,
            include_debug=False,
        )
        prediction["guided_edit_info"] = {
            "operation": op,
            "guided_backend": selected_backend,
            "sam_score": float(scores[idx]) if not use_fast_cpu else None,
            "sam_selected_index": idx,
            "sam_all_scores": [float(x) for x in scores],
            "brush_size": int(brush_size),
            "stroke_points_count": len(stroke_points),
            "sam_prompt_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "sam_prompt_size": [int(x2 - x1 + 1), int(y2 - y1 + 1)],
            "sam_embedding_cached": not use_fast_cpu,
        }
        return prediction

