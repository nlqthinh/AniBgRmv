"""HQ-SAM Guided instance selection, independent of the Edit UI and matting.

All boxes are (left, top, right, bottom) with an exclusive right/bottom.
Never clip a predicted object to a rectangular prompt ROI: doing so produces
the square-shaped Erase/Restore artifacts that this module prevents.
"""

from __future__ import annotations

import cv2
import numpy as np


MAX_INSTANCE_SIDE = 1536
MAX_AREA_BRUSH_FACTOR = 500


def small_object_crop_box(
    image_hw: tuple[int, int], points: np.ndarray, brush_size: int,
) -> tuple[int, int, int, int] | None:
    """Offer higher effective HQ-SAM resolution for clicks/very short strokes.

    Zoom is a display transform only, so the source-image pixel coordinates
    determine the crop. Long strokes retain full-image inference unchanged.
    """
    h, w = image_hw
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not len(pts) or min(h, w) < 64:
        return None
    span = np.ptp(pts, axis=0)
    if float(np.max(span)) > max(6, brush_size * 0.75):
        return None
    side = max(128, min(384, int(brush_size * 8)))
    if max(h, w) <= side * 1.25:
        return None  # Cropping would not appreciably magnify the input.
    cx, cy = pts.mean(axis=0)
    crop_w, crop_h = min(w, side), min(h, side)
    x0 = max(0, min(w - crop_w, int(round(float(cx) - crop_w / 2))))
    y0 = max(0, min(h - crop_h, int(round(float(cy) - crop_h / 2))))
    return x0, y0, x0 + crop_w, y0 + crop_h


def select_prompt_instance(
    raw_mask: np.ndarray,
    prompt_points: np.ndarray,
    brush_size: int,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    image_hw: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Accept only a plausible full instance containing a positive point.

    Reject cropped instances touching an *internal* crop border rather than
    applying a rectangular cut. Reject huge predictions rather than erasing a
    distant character; manual brush remains available for deliberate edits.
    """
    mask = np.asarray(raw_mask, dtype=bool)
    if mask.ndim != 2:
        return None
    h, w = mask.shape
    pts = np.asarray(prompt_points, dtype=np.float32).reshape(-1, 2)
    if not len(pts):
        return None

    # Limit connected-component labeling to a bounded neighborhood around
    # positive points; a huge erroneous SAM foreground must never allocate
    # a full 8K int32 label array. Touching an internal search edge rejects
    # the instance instead of creating a straight-edged edited mask.
    search_pad = 1024
    x0 = max(0, int(np.floor(pts[:, 0].min())) - search_pad)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - search_pad)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + search_pad + 1)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + search_pad + 1)
    clipped = mask[y0:y1, x0:x1].astype(np.uint8)
    if not clipped.any():
        return None
    _, labels = cv2.connectedComponents(clipped, connectivity=8)
    hit_labels: set[int] = set()
    for px, py in pts:
        x, y = int(round(float(px))) - x0, int(round(float(py))) - y0
        # Three source-image pixels of tolerance for antialiased thin details.
        xa, xb = max(0, x - 3), min(clipped.shape[1], x + 4)
        ya, yb = max(0, y - 3), min(clipped.shape[0], y + 4)
        if xa < xb and ya < yb:
            hit_labels.update(int(k) for k in np.unique(labels[ya:yb, xa:xb]) if k != 0)
    if not hit_labels:
        return None

    chosen = np.isin(labels, list(hit_labels))
    sy, sx = np.nonzero(chosen)
    left, right = x0 + int(sx.min()), x0 + int(sx.max()) + 1
    top, bottom = y0 + int(sy.min()), y0 + int(sy.max()) + 1
    area = int(chosen.sum())
    if ((left <= x0 and x0 > 0) or
            (top <= y0 and y0 > 0) or
            (right >= x1 and x1 < w) or
            (bottom >= y1 and y1 < h) or
            right - left > MAX_INSTANCE_SIDE or
            bottom - top > MAX_INSTANCE_SIDE or
            area > max(100_000, MAX_AREA_BRUSH_FACTOR * brush_size ** 2)):
        return None

    if crop_box is not None:
        if image_hw is None:
            raise ValueError("image_hw cần được cung cấp khi kiểm tra crop HQ-SAM.")
        ih, iw = image_hw
        cx0, cy0, cx1, cy1 = crop_box
        # A crop mask that runs off-screen is only a fragment of a larger
        # object. The full-image prediction is safer for that case.
        if ((left <= cx0 and cx0 > 0) or
                (top <= cy0 and cy0 > 0) or
                (right >= cx1 and cx1 < iw) or
                (bottom >= cy1 and cy1 < ih)):
            return None

    result = np.zeros_like(mask, dtype=bool)
    result[y0:y1, x0:x1] = chosen
    return result


def instance_support_box(mask: np.ndarray, brush_size: int) -> tuple[int, int, int, int]:
    """Pad around the *whole* instance for matting, never cut through its body."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Không thể tạo ROI từ mask HQ-SAM rỗng.")
    # The matting trimap's maximum 33px dilation at 768 has radius 16; add headroom for
    # upscaling/downscaling when restoring its support to source resolution.
    pad = max(48, brush_size * 2, int(np.ceil(20 * max(h, w) / 768)))
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))
