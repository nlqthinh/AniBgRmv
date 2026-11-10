"""Foreground colours for soft edges, so a cutout does not carry its old background.

An edge pixel of the source is a mix: I = alpha * F + (1 - alpha) * B_old.
Keeping I as the cutout colour leaves a fringe of B_old (a light halo from a
white background, a dark one from a dark background) once the cutout is put
on a new background. This module estimates F for 0 < alpha < 1 with two
"blur fusion" passes from M. Forte, "Approximate Fast Foreground Colour
Estimation", ICIP 2021 (doi 10.1109/ICIP42928.2021.9506164):

    F_mean, B_mean = local means of F weighted by alpha, of B by (1 - alpha)
    F = F_mean + alpha * (I - alpha * F_mean - (1 - alpha) * B_mean)

Box sizes 90 then 6, as in the paper. Opaque and fully transparent pixels are
never changed. Not part of any trained model: output.foreground_estimation
turns it off (images and video).
"""
from __future__ import annotations

import cv2
import numpy as np

BLUR_SIZES = (90, 6)
# A soft pixel only sees pixels within the boxes of both passes.
_REACH = sum(size // 2 + 1 for size in BLUR_SIZES)
_EPS = 1e-5


def _blur_fusion(image, fg, bg, alpha, size):
    box = (size, size)
    mean_alpha = cv2.blur(alpha, box)[:, :, None]
    mean_fg = cv2.blur(fg * alpha, box) / (mean_alpha + _EPS)
    mean_bg = cv2.blur(bg * (1.0 - alpha), box) / (1.0 - mean_alpha + _EPS)
    fg = mean_fg + alpha * (image - alpha * mean_fg - (1.0 - alpha) * mean_bg)
    return np.clip(fg, 0.0, 1.0), mean_bg


def estimate_foreground(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Copy of ``image`` whose soft-edge pixels hold the estimated foreground.

    image: uint8 (H, W, 3), RGB or BGR (channels are independent).
    alpha: float (H, W) in 0..1. Returns ``image`` itself when no pixel is soft.
    """
    alpha = np.asarray(alpha, dtype=np.float32)
    soft = (alpha > 0.0) & (alpha < 1.0)
    if not soft.any():
        return image
    # Work on the box around the soft pixels plus the filters' reach: same
    # result as the whole image, much less work when the subject is small.
    x, y, w, h = cv2.boundingRect(soft.astype(np.uint8))
    x0, y0 = max(0, x - _REACH), max(0, y - _REACH)
    x1, y1 = min(image.shape[1], x + w + _REACH), min(image.shape[0], y + h + _REACH)
    crop = image[y0:y1, x0:x1].astype(np.float32) / 255.0
    crop_alpha = alpha[y0:y1, x0:x1, None]
    fg = bg = crop
    for size in BLUR_SIZES:
        fg, bg = _blur_fusion(crop, fg, bg, crop_alpha, size)
    result = image.copy()
    region, region_soft = result[y0:y1, x0:x1], soft[y0:y1, x0:x1]
    region[region_soft] = np.rint(fg[region_soft] * 255.0).astype(np.uint8)
    return result
