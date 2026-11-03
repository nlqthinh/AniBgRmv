"""CPU Guided selection with OpenCV GrabCut (classical, no neural network).

GrabCut is fast on weak CPUs but cannot match HQ-SAM on hair or thin
details. Work runs on a bounded working raster; output masks keep the
original image resolution.
"""
from __future__ import annotations

import cv2
import numpy as np

WORKING_MAX_SIDE = 896


def _working_view(roi, strokes=None, max_side=WORKING_MAX_SIDE):
    """Return compact RGB/stroke arrays and the original ROI shape.

    INTER_AREA preserves thin positive paint seeds when downscaling;
    nearest-neighbor is used later when restoring label masks.
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


def grabcut_box_mask(image_rgb, box_xyxy):
    """Choose foreground inside a user box (inclusive xyxy).

    A failed GrabCut returns an empty mask, NOT a full rectangle:
    Manual Box is the explicit tool for changing an entire rectangle.
    """
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box_xyxy)
    result = np.zeros((h, w), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return result
    margin = max(8, min(64, max(x1 - x0 + 1, y1 - y0 + 1) // 5))
    rx0, ry0 = max(0, x0 - margin), max(0, y0 - margin)
    rx1, ry1 = min(w, x1 + margin + 1), min(h, y1 + margin + 1)
    roi = np.ascontiguousarray(image_rgb[ry0:ry1, rx0:rx1], dtype=np.uint8)
    bx, by = x0 - rx0, y0 - ry0
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if min(roi.shape[:2]) < 8 or bw < 3 or bh < 3:
        return result
    if bx <= 0 or by <= 0 or bx + bw >= roi.shape[1] or by + bh >= roi.shape[0]:
        # GrabCut needs known background outside its box.
        return result
    working, _, orig_hw = _working_view(roi)
    sh, sw = working.shape[:2]
    ratio_x = sw / orig_hw[1]
    ratio_y = sh / orig_hw[0]
    xb = max(1, round(bx * ratio_x))
    yb = max(1, round(by * ratio_y))
    xend = min(sw - 1, round((bx + bw) * ratio_x))
    yend = min(sh - 1, round((by + bh) * ratio_y))
    if xend - xb < 3 or yend - yb < 3:
        return result
    labels = np.zeros(working.shape[:2], dtype=np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(working, labels, (xb, yb, xend - xb, yend - yb),
                    bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return result
    selected = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    if orig_hw != working.shape[:2]:
        selected = cv2.resize(selected, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)
    selected[:by, :] = 0
    selected[by + bh:, :] = 0
    selected[:, :bx] = 0
    selected[:, bx + bw:] = 0
    result[ry0:ry1, rx0:rx1] = selected.astype(bool)
    return result


def grabcut_paint_mask(image_rgb, brush_mask, box_xyxy):
    """Grow a Paint stroke into the object under it; strokes are always kept."""
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    roi = np.ascontiguousarray(image_rgb[y1:y2 + 1, x1:x2 + 1], dtype=np.uint8)
    strokes = np.asarray(brush_mask[y1:y2 + 1, x1:x2 + 1], dtype=np.uint8)
    output = np.zeros(brush_mask.shape, dtype=bool)
    working, seed, orig_hw = _working_view(roi, strokes)
    h, w = working.shape[:2]
    if min(h, w) < 8 or seed is None or not seed.any():
        output[y1:y2 + 1, x1:x2 + 1] = strokes.astype(bool)
        return output
    # Seed probable FG around the paint; enforce positive seed pixels.
    r = max(2, min(12, int(max(1, np.count_nonzero(seed) ** 0.5) / 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    probable_fg = cv2.dilate(seed, kernel) > 0
    trimap = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    trimap[probable_fg] = cv2.GC_PR_FGD
    trimap[seed > 0] = cv2.GC_FGD
    # Reserve background at the perimeter, without overwriting explicit
    # paint seeds. Otherwise long strokes touching the image edge vanish.
    for ys, xs in ((slice(0, 2), slice(None)), (slice(-2, None), slice(None)),
                   (slice(None), slice(0, 2)), (slice(None), slice(-2, None))):
        region = trimap[ys, xs]
        region[seed[ys, xs] == 0] = cv2.GC_BGD
    if not np.any(trimap == cv2.GC_BGD) or not np.any(trimap == cv2.GC_FGD):
        output[y1:y2 + 1, x1:x2 + 1] = strokes.astype(bool)
        return output
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(working, trimap, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
        mask = ((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        touched = np.unique(labels[seed > 0])
        touched = touched[touched != 0]
        mask = np.isin(labels, touched) if len(touched) else seed.astype(bool)
    except cv2.error:
        mask = seed.astype(bool)
    if mask.shape != orig_hw:
        mask = cv2.resize(mask.astype(np.uint8), (orig_hw[1], orig_hw[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    # Resizing must not discard the user's exact input pixels.
    output[y1:y2 + 1, x1:x2 + 1] = mask | strokes.astype(bool)
    return output
