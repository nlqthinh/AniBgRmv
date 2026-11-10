"""Stateless image math shared by the automatic cut, Guided Edit and video export.

Every constant here copies a training notebook. Change it only together with
the notebook that trained the model, otherwise the model sees inputs it was
never trained on:

  letterbox   768x768, aspect ratio kept, RGB padding 114, alpha padding 0 (F3, M7, M71)
              downscale filter: INTER_AREA (F3, M7), INTER_LINEAR (M71)
  UNet        RGB normalized with the ImageNet mean/std (F3)
  ViTMatte    RGB normalized with mean = std = 0.5, the VitMatteImageProcessor
              config of hustvl/vitmatte-small-distinctions-646; trimap/255 is
              the 4th channel (M7, M71)
  trimaps     loose trimap from the UNet alpha (M7); ellipse erode 17 /
              dilate 33 around a binary HQ-SAM mask (M71)

This module only needs numpy, OpenCV and Pillow: the video matting sidecar
imports it as a plain sibling file.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

LETTERBOX_SIZE = 768
# ONNX contract of the letterbox models (F3, M7, M71): one float32 input,
# RGB (UNet) or RGB + trimap (ViTMatte), and one float32 alpha output.
UNET_CHANNELS, MATTING_CHANNELS = 3, 4
ALPHA_OUTPUT_SHAPE = (1, 1, LETTERBOX_SIZE, LETTERBOX_SIZE)


def model_input_shape(channels: int) -> tuple[int, int, int, int]:
    return (1, channels, LETTERBOX_SIZE, LETTERBOX_SIZE)
PAD_RGB = 114
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VITMATTE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
VITMATTE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
# M71 notebook: cv2.resize(..., INTER_LINEAR) for every letterbox.
SAM_MATTING_INTERPOLATION = cv2.INTER_LINEAR
SAM_TRIMAP_KERNELS = (17, 33)  # (erode, dilate) ellipse diameters, M71 notebook


def load_rgb(image_path) -> Image.Image:
    """Decode an image file to RGB once; every later step reuses this image."""
    with Image.open(image_path) as source:
        return source.convert("RGB")


def letterbox_rgb(image_pil: Image.Image, size: int = LETTERBOX_SIZE, interpolation: int | None = None):
    """RGB -> keep aspect ratio -> constant gray padding -> size x size.

    ``interpolation=None`` is the F3/M7 rule: INTER_AREA when shrinking,
    INTER_LINEAR when enlarging. Returns (PIL image, valid_mask 255/0, meta).
    """
    image_np = np.array(image_pil.convert("RGB"))
    orig_h, orig_w = image_np.shape[:2]
    scale = min(size / orig_w, size / orig_h)
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))
    if interpolation is None:
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


def to_letterbox(plane: np.ndarray, meta: dict, interpolation: int) -> np.ndarray:
    """2-D uint8 array in original pixels -> uint8 square laid out like ``meta``, padding 0.

    Binary masks use INTER_NEAREST (as the notebooks), 8-bit alphas INTER_LINEAR.
    """
    size = int(meta["target_size"])
    rw, rh = int(meta["resized_width"]), int(meta["resized_height"])
    x, y = int(meta["pad_left"]), int(meta["pad_top"])
    square = np.zeros((size, size), np.uint8)
    square[y:y + rh, x:x + rw] = cv2.resize(np.asarray(plane, np.uint8), (rw, rh),
                                            interpolation=interpolation)
    return square


def unletterbox(alpha_square: np.ndarray, meta: dict) -> np.ndarray:
    """Crop the padding away and resize to the original image size (float32 0..1).

    uint8 input is read as 0..255; float input as 0..1.
    """
    alpha = np.asarray(alpha_square)
    alpha = alpha.astype(np.float32) / 255.0 if alpha.dtype == np.uint8 else alpha.astype(np.float32)
    x1, y1 = int(meta["pad_left"]), int(meta["pad_top"])
    x2, y2 = x1 + int(meta["resized_width"]), y1 + int(meta["resized_height"])
    restored = cv2.resize(np.clip(alpha[y1:y2, x1:x2], 0.0, 1.0),
                          (int(meta["original_width"]), int(meta["original_height"])),
                          interpolation=cv2.INTER_LINEAR)
    return np.clip(restored, 0.0, 1.0)


def restore_from_letterbox(alpha_square: np.ndarray, meta: dict) -> np.ndarray:
    """``unletterbox`` as an 8-bit alpha."""
    return np.rint(unletterbox(alpha_square, meta) * 255.0).astype(np.uint8)


def _normalized_chw(image_pil: Image.Image, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_pil.convert("RGB")).astype(np.float32) / 255.0
    return np.transpose((arr - mean) / std, (2, 0, 1))


def unet_input(image_letterbox_pil: Image.Image) -> np.ndarray:
    """UNet image[1,3,768,768], ImageNet-normalized like the F3 notebook."""
    return _normalized_chw(image_letterbox_pil, IMAGENET_MEAN, IMAGENET_STD)[None].astype(np.float32)


def unet_alpha(raw_output: np.ndarray) -> np.ndarray:
    """Check the UNet ONNX contract: an alpha in 0..1 (sigmoid inside the graph).

    tools/export_unet_onnx.py always exports the sigmoid. A graph that returns
    logits is rejected instead of guessed at.
    """
    alpha = np.asarray(raw_output, dtype=np.float32)
    if not np.isfinite(alpha).all():
        raise ValueError("UNet ONNX trả giá trị NaN/Inf.")
    if alpha.min() < -1e-4 or alpha.max() > 1.0 + 1e-4:
        raise ValueError(
            "UNet ONNX phải trả alpha 0..1 (đã có sigmoid), nhưng nhận "
            f"[{alpha.min():.3f}, {alpha.max():.3f}]. Export lại bằng tools/export_unet_onnx.py.")
    return np.clip(alpha, 0.0, 1.0)


def vitmatte_input(image_letterbox_pil: Image.Image, trimap: np.ndarray,
                   size: int = LETTERBOX_SIZE) -> np.ndarray:
    """ViTMatte pixel_values[1,4,H,W], identical to VitMatteImageProcessor.

    RGB: (x/255 - 0.5) / 0.5. Channel 4: trimap/255 (rescaled, not normalized).
    """
    image = image_letterbox_pil.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    if trimap.shape != (size, size):
        trimap = cv2.resize(trimap, (size, size), interpolation=cv2.INTER_NEAREST)
    rgb = _normalized_chw(image, VITMATTE_MEAN, VITMATTE_STD)
    x = np.concatenate([rgb, trimap.astype(np.float32)[None] / 255.0], axis=0)
    return x[None].astype(np.float32)


def loose_trimap(alpha_01: np.ndarray, valid_mask: np.ndarray | None = None, *,
                 fg_thresh: float = 0.995, rough_thresh: float = 0.03,
                 erode_k: int = 21, dilate_k: int = 35) -> np.ndarray:
    """M7 trimap (0 / 128 / 255) from a UNet alpha; padding forced to background.

    The M7 notebook trained and tested ViTMatte on trimaps built from the F3
    alpha saved as an 8-bit PNG, so the thresholds apply to the 8-bit value.
    """
    alpha = np.rint(np.clip(alpha_01, 0.0, 1.0) * 255.0).astype(np.uint8).astype(np.float32) / 255.0
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


def mask_trimap(mask_square: np.ndarray, erode_size: int = SAM_TRIMAP_KERNELS[0],
                dilate_size: int = SAM_TRIMAP_KERNELS[1]) -> np.ndarray:
    """M71 trimap around a binary mask: dilated band = 128, eroded core = 255.

    Like the notebook, the padding is not forced to 0: a band that reaches
    the image border may continue into the gray padding.
    """
    mask = (np.asarray(mask_square) > 0).astype(np.uint8)
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode_size), int(erode_size)))
    dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate_size), int(dilate_size)))
    trimap = np.zeros(mask.shape, np.uint8)
    trimap[cv2.dilate(mask, dilate) > 0] = 128
    trimap[cv2.erode(mask, erode) > 0] = 255
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
