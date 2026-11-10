"""Pure alpha operations used by Guided Edit (no models, no UI).

All functions take/return uint8 alpha (H, W) or bool masks of the same shape.
Work is limited to a local bounding rectangle whenever possible, so a small
stroke on a 4K/8K image does not allocate full-size temporary arrays.
"""

import cv2
import numpy as np

from . import matting_ops as ops

# Hard Erase: matting may extend the selected instance by a *small* border,
# but its soft values must never leave half-erased outlines in final alpha.
HARD_ERASE_MATTE_THRESHOLD = 0.50
HARD_ERASE_BORDER_RADIUS_AT_768 = 2.0
HARD_ERASE_MAX_BORDER_RADIUS = 8


def stroke_mask(shape_hw: tuple[int, int], stroke_points, brush_size: int) -> np.ndarray:
    """Rasterize a Paint stroke (round caps) into a bool mask."""
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


def erase_with_feather(alpha: np.ndarray, erase_mask: np.ndarray, *,
                       edge_expand_px: int = 2) -> np.ndarray:
    """Erase a selection plus a narrow halo guard, with a subpixel outer edge.

    A predicted mask can stop just inside the object's antialiased outline;
    feathering it alone leaves a colored rim. Expand by ``edge_expand_px``
    first, then a 3x3 Gaussian adds at most one pixel of feathering outside.
    """
    original = np.asarray(alpha, dtype=np.uint8)
    selected = np.asarray(erase_mask, dtype=bool)
    if original.ndim != 2 or original.shape != selected.shape:
        raise ValueError("Alpha và vùng Erase phải cùng kích thước H×W.")
    radius = int(edge_expand_px)
    if not 0 <= radius <= 6:
        raise ValueError("Độ mở rộng viền Erase phải nằm trong 0..6 pixel.")

    edited = original.copy()
    rows = np.flatnonzero(selected.any(axis=1))
    cols = np.flatnonzero(selected.any(axis=0))
    if not rows.size or not cols.size:
        return edited

    height, width = original.shape
    # Radius pixels of guaranteed Erase + one-pixel feather + spare border.
    pad = radius + 2
    y0, y1 = max(0, int(rows[0]) - pad), min(height, int(rows[-1]) + pad + 1)
    x0, x1 = max(0, int(cols[0]) - pad), min(width, int(cols[-1]) + pad + 1)
    region = np.s_[y0:y1, x0:x1]
    local_mask = selected[region]
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (radius * 2 + 1, radius * 2 + 1))
        local_mask = cv2.dilate(local_mask.astype(np.uint8), kernel) > 0
    coverage = cv2.GaussianBlur(local_mask.astype(np.uint8) * 255,
                                (3, 3), sigmaX=0.65)
    coverage[local_mask] = 255  # absolutely no residual outline within the guard

    keep = (255 - coverage).astype(np.uint16)
    current = edited[region].astype(np.uint16)
    edited[region] = ((current * keep + 127) // 255).astype(np.uint8)
    return edited


def blend_soft_selection(current_alpha: np.ndarray, soft_selection: np.ndarray,
                         brush_mask: np.ndarray, operation: str) -> np.ndarray:
    """Apply a soft (0..1) matted selection; the brush stays a hard constraint."""
    alpha = np.asarray(current_alpha, dtype=np.uint8)
    selection = np.asarray(soft_selection, dtype=np.float32)
    brush = np.asarray(brush_mask, dtype=bool)
    if alpha.ndim != 2 or selection.shape != alpha.shape or brush.shape != alpha.shape:
        raise ValueError("Alpha, vùng chọn mềm và brush phải cùng kích thước H×W.")
    selection = np.clip(selection, 0.0, 1.0)
    if not np.isfinite(selection).all():
        raise ValueError("Matting trả alpha không hữu hạn.")
    if operation == "erase":
        edited = np.rint(alpha.astype(np.float32) * (1.0 - selection)).astype(np.uint8)
        edited[brush] = 0
    elif operation == "restore":
        edited = np.maximum(alpha, np.rint(selection * 255.0).astype(np.uint8))
        edited[brush] = 255
    else:
        raise ValueError("operation phải là erase hoặc restore.")
    return edited


def hard_erase_mask(selection: np.ndarray, soft_selection: np.ndarray,
                    roi_mask: np.ndarray) -> np.ndarray:
    """Keep every selected pixel; add only confident matted pixels next to it.

    Matting supports the boundary but never decides a fractional Erase;
    limiting its extension avoids swallowing an unrelated nearby object.
    """
    core = np.asarray(selection, dtype=bool)
    soft = np.asarray(soft_selection, dtype=np.float32)
    roi = np.asarray(roi_mask, dtype=bool)
    if core.ndim != 2 or soft.shape != core.shape or roi.shape != core.shape:
        raise ValueError("Vùng chọn, alpha matting và ROI phải cùng kích thước H×W.")
    if not np.isfinite(soft).all():
        raise ValueError("Matting trả alpha không hữu hạn.")
    if not core.any():
        return core.copy()

    height, width = core.shape
    radius = max(1, min(HARD_ERASE_MAX_BORDER_RADIUS, int(round(
        HARD_ERASE_BORDER_RADIUS_AT_768 * max(height, width) / float(ops.LETTERBOX_SIZE)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # Never dilate an entire 4K/8K mask for a small stroke: stay inside ROI.
    rows = np.flatnonzero(roi.any(axis=1))
    cols = np.flatnonzero(roi.any(axis=0))
    if not rows.size or not cols.size:
        return core.copy()
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    region = np.s_[y0:y1, x0:x1]
    nearby = cv2.dilate(core[region].astype(np.uint8), kernel) > 0
    result = core.copy()
    result[region] |= (soft[region] >= HARD_ERASE_MATTE_THRESHOLD) & nearby & roi[region]
    return result


def selection_trimap_kernels(mask_768: np.ndarray) -> tuple[int, int]:
    """Erode/dilate kernel sizes adapted to thin instances.

    The maximum inscribed radius suits narrow objects better than a bounding
    box, e.g. a sword crossing the entire image diagonally.
    """
    if not np.any(mask_768):
        return 1, 5
    distance = cv2.distanceTransform(mask_768.astype(np.uint8), cv2.DIST_L2, 3)
    radius = float(distance.max())
    erode_radius = max(0, min(8, int(radius - 1.0)))
    dilate_radius = max(2, min(16, int(round(radius * 2.0))))
    return erode_radius * 2 + 1, dilate_radius * 2 + 1
