"""Held pictures: the same drawing of the object on consecutive frames.

Anime keeps a character drawing for 2-3 frames while the background, lights
or a small camera shake change. Tracked and matted one by one, those frames
get a slightly different mask and alpha each time, so hair flickers. A frame
that *holds* an earlier frame (its source) reuses that frame's mask (after
Track) and alpha (at Export), moved by the camera shake.

The test compares the Track proxies (what SAM2 sees):
- structure, not brightness: grayscale minus its blur (STRUCTURE_SIGMA), so
  lights and shading that change smoothly behind or on the character do not
  count; lines and hair strands do;
- only in the band where ViTMatte decides alpha: the M71 trimap band
  (matting_ops.SAM_TRIMAP_KERNELS) around either mask, at proxy scale; the
  inside of the body is not compared (a moving mouth does not change alpha);
- a shake up to MAX_SHIFT px is measured (phase correlation) and compensated
  when the plain comparison fails;
- held when at most HOLD_TOLERANCE of the band changes by > STRUCTURE_THRESHOLD.
Each frame is compared with its source, not with the previous frame, so a slow
drift cannot pile up along a run.

Measured on 24frame.mp4 (App_15, 1280x720 proxies): held drawings with lights
changing behind them: 0-1 band pixels above 16; frames 11->13 (0.5 px shake):
268 -> 55 pixels (0.05%) after compensation; moving drawings: > 0.5%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from .matting_ops import LETTERBOX_SIZE, SAM_TRIMAP_KERNELS
from .video_pipeline import PROXY_LONG_EDGE

STRUCTURE_SIGMA = 4.0       # px at PROXY_LONG_EDGE
STRUCTURE_THRESHOLD = 16
MAX_SHIFT = 3.0             # px at PROXY_LONG_EDGE: a shake, not a pan
MIN_SHIFT_RESPONSE = 0.3    # phase-correlation peak below this: no clear shift
HOLD_TOLERANCE = 5e-4
MIN_TOLERANCE_PIXELS = 8


class Hold(NamedTuple):
    """A frame shows its ``source`` frame moved by (dx, dy) proxy pixels."""
    source: int
    dx: float = 0.0
    dy: float = 0.0


@dataclass(frozen=True)
class _View:
    structure: np.ndarray  # float32, lines and strands of the proxy
    mask: np.ndarray       # bool, object pixels


def _scaled(distance: float, shape) -> float:
    return distance * max(shape) / PROXY_LONG_EDGE


def _view(path, mask) -> _View:
    picture = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if picture is None:
        raise FileNotFoundError('Không đọc được frame proxy ' + str(path))
    mask = np.asarray(mask) > 0
    if picture.shape[:2] != mask.shape:
        raise ValueError(f'Mask {mask.shape} khác frame proxy {picture.shape[:2]}.')
    gray = cv2.cvtColor(cv2.GaussianBlur(picture, (3, 3), 0), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return _View(gray - cv2.GaussianBlur(gray, (0, 0), _scaled(STRUCTURE_SIGMA, mask.shape)), mask)


def _band(earlier: _View, later: _View) -> np.ndarray:
    """The trimap band of M71 around both masks, at proxy scale.

    Distances instead of dilate/erode with a disk: same band, 3x faster.
    """
    to_proxy = max(later.mask.shape) / LETTERBOX_SIZE
    inside, outside = (size // 2 * to_proxy for size in SAM_TRIMAP_KERNELS)
    outside_union = (~(earlier.mask | later.mask)).astype(np.uint8)
    both = (earlier.mask & later.mask).astype(np.uint8)
    near = cv2.distanceTransform(outside_union, cv2.DIST_L2, 5) <= outside
    deep = cv2.distanceTransform(both, cv2.DIST_L2, 5) > inside
    return near & ~deep


def _changed(a: np.ndarray, b: np.ndarray, band: np.ndarray) -> int:
    return np.count_nonzero((np.abs(a - b) > STRUCTURE_THRESHOLD) & band)


def _shake(a: np.ndarray, b: np.ndarray, band: np.ndarray):
    """(dx, dy) moving ``a`` onto ``b`` around the object, or None."""
    x, y, w, h = cv2.boundingRect(band.astype(np.uint8))
    if w < 16 or h < 16:
        return None
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a[y:y + h, x:x + w], b[y:y + h, x:x + w], window)
    if response < MIN_SHIFT_RESPONSE or math.hypot(dx, dy) > _scaled(MAX_SHIFT, a.shape):
        return None
    return dx, dy


def shifted(plane: np.ndarray, dx: float, dy: float, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """``plane`` moved by (dx, dy) pixels; edges repeat (the subject goes on past the frame)."""
    if dx == 0 and dy == 0:
        return plane
    h, w = plane.shape[:2]
    return cv2.warpAffine(plane, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                          flags=interpolation, borderMode=cv2.BORDER_REPLICATE)


def _match(earlier: _View, later: _View) -> tuple[float, float] | None:
    """The shift with which ``later`` holds ``earlier``, or None."""
    band = _band(earlier, later)
    allowed = max(MIN_TOLERANCE_PIXELS, HOLD_TOLERANCE * np.count_nonzero(band))
    if _changed(earlier.structure, later.structure, band) <= allowed:
        return 0.0, 0.0
    shake = _shake(earlier.structure, later.structure, band)
    if shake is None:
        return None
    moved = shifted(earlier.structure, *shake, interpolation=cv2.INTER_LINEAR)
    return shake if _changed(moved, later.structure, band) <= allowed else None


def held_frames(frame_paths, masks, keep=frozenset()) -> tuple[Hold, ...]:
    """For every frame, the earlier frame it holds (itself, shift 0, if none).

    ``masks``: per frame, nonzero = object (label maps or 0/255 masks).
    Frames in ``keep`` (user keyframes) always stand for themselves.
    """
    if len(frame_paths) != len(masks):
        raise ValueError('Số frame proxy khác số mask.')
    holds, reference = [], None
    for index, (path, mask) in enumerate(zip(frame_paths, masks)):
        view = _view(path, mask)
        shake = None if reference is None or index in keep else _match(reference, view)
        if shake is None:
            holds.append(Hold(index))
            reference = view
        else:
            holds.append(Hold(holds[-1].source, round(shake[0], 2), round(shake[1], 2)))
    return tuple(holds)
