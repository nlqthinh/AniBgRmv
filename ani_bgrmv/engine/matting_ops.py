"""Stateless image math shared by the automatic cut and Guided Edit.

Matches the training notebooks: aspect-preserving letterbox to 768x768 with
RGB padding 114 (alpha padding 0), ImageNet normalization, and the loose
trimap that turns a UNet/HQ-SAM mask into ViTMatte input.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

LETTERBOX_SIZE = 768
PAD_RGB = 114
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_rgb(image_pil: Image.Image, size: int = LETTERBOX_SIZE):
    """RGB -> keep aspect ratio -> constant gray padding -> size x size.

    Returns (letterboxed PIL image, valid_mask uint8 255/0, meta).
    """
    image_np = np.array(image_pil.convert("RGB"))
    orig_h, orig_w = image_np.shape[:2]
    scale = min(size / orig_w, size / orig_h)
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_np, (resized_w, resized_h), interpolation=interpolation)

    pad_left = (size - resized_w) // 2
    pad_right = size - resized_w - pad_left
    pad_top = (size - resized_h) // 2
    pad_bottom = size - resized_h - pad_top
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=(PAD_RGB,) * 3)
    valid_mask = np.zeros((size, size), dtype=np.uint8)
    valid_mask[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = 255
    meta = {
        "original_width": int(orig_w), "original_height": int(orig_h),
        "resized_width": int(resized_w), "resized_height": int(resized_h),
        "pad_left": int(pad_left), "pad_top": int(pad_top),
        "pad_right": int(pad_right), "pad_bottom": int(pad_bottom),
        "scale": float(scale), "target_size": int(size),
    }
    return Image.fromarray(padded.astype(np.uint8)), valid_mask, meta


def restore_from_letterbox(alpha_square: np.ndarray, meta: dict) -> np.ndarray:
    """Crop the padding away and resize to the original image size (uint8)."""
    alpha = np.asarray(alpha_square)
    if alpha.dtype != np.uint8:
        alpha = (np.clip(alpha, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    x1, y1 = int(meta["pad_left"]), int(meta["pad_top"])
    x2, y2 = x1 + int(meta["resized_width"]), y1 + int(meta["resized_height"])
    restored = cv2.resize(alpha[y1:y2, x1:x2],
                          (int(meta["original_width"]), int(meta["original_height"])),
                          interpolation=cv2.INTER_LINEAR)
    return restored.astype(np.uint8)


def normalized_chw(image_pil: Image.Image) -> np.ndarray:
    """RGB -> ImageNet-normalized float32 [3, H, W]."""
    arr = np.asarray(image_pil.convert("RGB")).astype(np.float32) / 255.0
    return np.transpose((arr - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1))


def unet_input(image_letterbox_pil: Image.Image) -> np.ndarray:
    return normalized_chw(image_letterbox_pil)[None].astype(np.float32)


def vitmatte_input(image_letterbox_pil: Image.Image, trimap: np.ndarray,
                   size: int = LETTERBOX_SIZE) -> np.ndarray:
    """ViTMatte pixel_values: normalized RGB + trimap/255 as a 4th channel."""
    image = image_letterbox_pil.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    if trimap.shape != (size, size):
        trimap = cv2.resize(trimap, (size, size), interpolation=cv2.INTER_NEAREST)
    x = np.concatenate([normalized_chw(image), trimap.astype(np.float32)[None] / 255.0], axis=0)
    return x[None].astype(np.float32)


def loose_trimap(alpha_01: np.ndarray, valid_mask: np.ndarray | None = None, *,
                 fg_thresh: float = 0.995, rough_thresh: float = 0.03,
                 erode_k: int = 21, dilate_k: int = 35) -> np.ndarray:
    """Trimap (0 / 128 / 255) from a coarse alpha; padding forced to background."""
    alpha = np.clip(alpha_01, 0.0, 1.0).astype(np.float32)
    strong_fg = (alpha >= fg_thresh).astype(np.uint8) * 255
    rough_fg = (alpha >= rough_thresh).astype(np.uint8) * 255
    sure_fg = cv2.erode(strong_fg, np.ones((int(erode_k), int(erode_k)), np.uint8), iterations=1)
    possible_fg = cv2.dilate(rough_fg, np.ones((int(dilate_k), int(dilate_k)), np.uint8), iterations=1)
    trimap = np.full(alpha.shape, 128, dtype=np.uint8)
    trimap[possible_fg == 0] = 0
    trimap[sure_fg == 255] = 255
    if valid_mask is not None:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape != trimap.shape:
            valid = cv2.resize(valid.astype(np.uint8) * 255, (trimap.shape[1], trimap.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
        trimap[~valid] = 0
    return trimap


def trimap_ratios(trimap: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    """Fraction of background / unknown / foreground (inside valid_mask if given)."""
    arr = np.asarray(trimap)
    if valid_mask is not None:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape == arr.shape and valid.any():
            arr = arr[valid]
    total = max(1, int(arr.size))
    return {"bg_ratio": float((arr == 0).sum() / total),
            "unknown_ratio": float((arr == 128).sum() / total),
            "fg_ratio": float((arr == 255).sum() / total)}


def array_stats(x: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    values = np.asarray(x, dtype=np.float32)
    if valid_mask is not None:
        valid = np.asarray(valid_mask).astype(bool)
        if valid.shape == values.shape and valid.any():
            values = values[valid]
    finite = np.isfinite(values)
    keys = ("min", "max", "mean", "p01", "p05", "p50", "p95", "p99", "dynamic_p01_p99")
    if not finite.any():
        return dict.fromkeys(keys)
    values = values[finite]
    p01, p05, p50, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
    return dict(zip(keys, (float(values.min()), float(values.max()), float(values.mean()),
                           float(p01), float(p05), float(p50), float(p95), float(p99),
                           float(p99 - p01))))


def stretch_if_low_contrast(alpha_01: np.ndarray, valid_mask: np.ndarray | None,
                            reason_prefix: str) -> tuple[np.ndarray, dict]:
    """Deployment guard against a 'gray mask' ONNX output.

    A low-contrast alpha makes the loose trimap almost all unknown, and
    ViTMatte then returns a foggy cutout. Only in that failure mode the alpha
    is percentile-stretched; normal outputs are returned unchanged.
    """
    alpha = np.clip(alpha_01.astype(np.float32), 0.0, 1.0)
    valid = None
    if valid_mask is not None:
        valid = np.asarray(valid_mask).astype(bool)
        if not (valid.shape == alpha.shape and valid.any()):
            valid = None

    stats_before = array_stats(alpha, valid)
    p01, p99, dynamic = stats_before["p01"], stats_before["p99"], stats_before["dynamic_p01_p99"]
    if p01 is None or p99 is None or dynamic is None or dynamic < 1e-6:
        return alpha, {"applied": False, "reason": f"{reason_prefix}_not_enough_dynamic_range",
                       "stats_before": stats_before}

    trimap_before = trimap_ratios(loose_trimap(alpha, valid), valid)
    low_contrast = (p99 < 0.90) or (p01 > 0.08) or (dynamic < 0.55)
    trimap_too_open = trimap_before["bg_ratio"] < 0.08 or trimap_before["unknown_ratio"] > 0.70
    if not (low_contrast or trimap_too_open):
        return alpha, {"applied": False, "reason": f"{reason_prefix}_already_sane",
                       "stats_before": stats_before, "trimap_before": trimap_before}

    lo, hi = float(p01), float(p99)
    stretched = np.clip((alpha - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    if valid is not None:
        stretched[~valid] = 0.0
    # Keep soft edges, but make confident bg/fg actually confident.
    stretched[stretched < 0.015] = 0.0
    stretched[stretched > 0.985] = 1.0
    return stretched, {
        "applied": True, "reason": f"{reason_prefix}_low_contrast_or_open_trimap",
        "stats_before": stats_before, "stats_after": array_stats(stretched, valid),
        "trimap_before": trimap_before,
        "trimap_after": trimap_ratios(loose_trimap(stretched, valid), valid),
        "stretch_p01": lo, "stretch_p99": hi,
    }


def unet_alpha_for_trimap(raw_output: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Turn the UNet ONNX output into the alpha used to build the trimap.

    Exports may return logits (notebook applies sigmoid) or an alpha already
    in 0..1; never apply sigmoid twice. Then apply the low-contrast guard.
    """
    raw = np.nan_to_num(np.asarray(raw_output, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw_stats = array_stats(raw, valid_mask)
    looks_like_logits = (raw_stats["min"] is not None and
                         (raw_stats["min"] < 0.0 or raw_stats["max"] > 1.0))
    if looks_like_logits:
        mode, alpha = "sigmoid_from_logits", 1.0 / (1.0 + np.exp(-raw))
    else:
        mode, alpha = "direct_alpha", raw
    before_fix = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    used, fix_stats = stretch_if_low_contrast(before_fix, valid_mask, mode)
    trimap = loose_trimap(used, valid_mask)
    return {"raw_output": raw, "alpha_before_fix": before_fix, "alpha_used": used,
            "mode": mode, "raw_stats": raw_stats, "fix_stats": fix_stats,
            "trimap": trimap, "trimap_ratios_valid": trimap_ratios(trimap, valid_mask)}
