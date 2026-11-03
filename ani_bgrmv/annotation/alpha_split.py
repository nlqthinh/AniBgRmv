"""Split one alpha matte into separated objects, no model needed.

Objects that do not touch are separated by background, so the connected
regions of the solid alpha (>= SOLID_ALPHA) give one region per object. Soft
edge pixels (0 < alpha < SOLID_ALPHA) join the nearest region, so every
instance keeps the original 256-level edge and all regions add up exactly to
the input alpha. Touching objects stay one region: they need HQ-SAM prompts.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SOLID_ALPHA = 128
# Regions smaller than this share of the image are fragments, not objects
# (0.1% = a 32x32 px patch on a 1024x1024 image).
DEFAULT_MIN_OBJECT_RATIO = 0.001


@dataclass(frozen=True)
class AlphaSplit:
    labels: np.ndarray       # int32 HxW: 0 = background; 1..object_count = objects
                             # (largest first), then fragments
    object_count: int
    areas: tuple[int, ...]   # solid pixels of labels 1..N

    @property
    def fragment_count(self) -> int:
        return len(self.areas) - self.object_count


def split_alpha(alpha: np.ndarray, min_object_ratio: float = DEFAULT_MIN_OBJECT_RATIO) -> AlphaSplit:
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or alpha.dtype != np.uint8:
        raise ValueError('alpha phải là ảnh xám uint8 (H,W).')
    solid = (alpha >= SOLID_ALPHA).astype(np.uint8)
    count, regions, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas, kind='stable')
    relabel = np.zeros(count, np.int32)
    relabel[order + 1] = np.arange(1, count, dtype=np.int32)
    labels = relabel[regions]
    if count > 1:
        _attach_soft_pixels(labels, alpha)
    sorted_areas = tuple(int(a) for a in areas[order])
    object_count = sum(a >= min_object_ratio * alpha.size for a in sorted_areas)
    return AlphaSplit(labels, int(object_count), sorted_areas)


def _attach_soft_pixels(labels: np.ndarray, alpha: np.ndarray) -> None:
    """Give every 0 < alpha < SOLID_ALPHA pixel the label of its nearest region."""
    soft = (alpha > 0) & (labels == 0)
    if not soft.any():
        return
    seeds = labels > 0
    # Each seed pixel gets its own id; every other pixel gets its nearest seed's id.
    _, nearest = cv2.distanceTransformWithLabels((~seeds).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    seed_label = np.zeros(int(nearest.max()) + 1, np.int32)
    seed_label[nearest[seeds]] = labels[seeds]
    labels[soft] = seed_label[nearest[soft]]


def instance_alpha(alpha: np.ndarray, split: AlphaSplit, label: int) -> np.ndarray:
    """Original 256-level alpha of one region, zero elsewhere."""
    return np.where(split.labels == label, alpha, 0).astype(np.uint8)
