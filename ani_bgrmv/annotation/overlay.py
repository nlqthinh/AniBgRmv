"""Colored object maps: pilot sheets, package previews and the annotator canvas."""
from __future__ import annotations

import cv2
import numpy as np

from .partition import internal_edges

# 20 distinct colors; gray is kept for pixels of no object.
PALETTE = np.array([(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128),
                    (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
                    (128, 128, 0), (255, 215, 180), (0, 0, 128), (255, 225, 25), (100, 100, 255)], np.float32)
IGNORED_COLOR = (128, 128, 128)
EDGE_COLOR = (255, 255, 255)


def object_color(object_id: int) -> tuple[int, int, int]:
    return tuple(int(v) for v in PALETTE[(object_id - 1) % len(PALETTE)])


def _draw_numbers(image: np.ndarray, labels: np.ndarray) -> None:
    """Each object's id at its center (white on a dark outline)."""
    ys, xs = np.nonzero(labels > 0)
    if not len(ys):
        return
    ids = labels[ys, xs]
    count = np.bincount(ids)
    center_y, center_x = np.bincount(ids, ys) / np.maximum(count, 1), np.bincount(ids, xs) / np.maximum(count, 1)
    scale = max(0.6, max(labels.shape) / 900)
    thickness = max(1, round(2 * scale))
    for object_id in np.flatnonzero(count):
        text = str(object_id)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        origin = (int(center_x[object_id] - w / 2), int(center_y[object_id] + h / 2))
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def overlay(rgb: np.ndarray, alpha: np.ndarray, labels: np.ndarray, *, edges: bool = False,
            highlight: int | None = None, numbers: bool = False) -> np.ndarray:
    """Image dimmed to 40%, each object tinted with its own color by its alpha
    (gray = no object). ``edges`` draws object-object boundaries in white;
    ``highlight`` shows one object in full color; ``numbers`` writes the ids."""
    count = max(int(labels.max()), 0)
    colors = np.zeros((count + 2, 3), np.float32)   # index = label + 1
    colors[0] = IGNORED_COLOR
    for label in range(1, count + 1):
        colors[label + 1] = object_color(label)
    weight = (alpha.astype(np.float32) / 255.0)[..., None] * 0.6
    if highlight is not None:
        weight = np.where((labels == highlight)[..., None], weight / 0.6 * 0.9, weight * 0.5)
    out = rgb.astype(np.float32) * 0.4 + colors[labels + 1] * weight
    if edges:
        out[internal_edges(labels)] = EDGE_COLOR
    out = out.clip(0, 255).astype(np.uint8)
    if numbers:
        _draw_numbers(out, labels)
    return out


def thumbnail(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)
