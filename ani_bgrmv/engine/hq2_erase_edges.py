"""Anti-alias HQ2 Erase while removing the original object's edge pixels.

HQ2 + M7's binary mask can stop just inside the original antialiased outline.
Feathering that mask alone leaves a visible colored rim. Expand the definite
Erase by a very small radius first, then feather outside the *expanded* mask.
All processing stays inside a local bounding rectangle to control memory.
"""

import cv2
import numpy as np


def erase_hq2_alpha(alpha: np.ndarray, erase_mask: np.ndarray, *,
                    edge_expand_px: int = 2) -> np.ndarray:
    """Erase HQ2/M7 plus a narrow halo guard, with a subpixel outer edge.

    Original and expanded selected pixels stay alpha=0. A 3x3 Gaussian
    provides at most one pixel of feathering beyond that expansion, where
    the object's former outline should already have been removed.
    """
    original = np.asarray(alpha, dtype=np.uint8)
    selected = np.asarray(erase_mask, dtype=bool)
    if original.ndim != 2 or original.shape != selected.shape:
        raise ValueError("Alpha và vùng HQ2 Erase phải cùng kích thước H×W.")
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
