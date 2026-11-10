"""Share one alpha matte among its objects: separated regions + HQ-SAM objects.

The cutout already has the right outer edge, so the only question is which
object each visible pixel belongs to. No model runs here; the caller decodes
each SamObject's prompt and stores the logits.

1. Separated regions (alpha_split) are objects; tiny regions are IGNORED.
2. A SAM piece takes the pixels where its logit is > 0 and the highest, only
   inside the regions it was clicked in (plus tiny fragments). Pieces of one
   group form one object.
3. What is left of a clicked region stays an object when it is big and thick
   enough (e.g. the second person of a pair); other leftovers are edge slivers
   the 256x256 SAM mask missed and join the nearest object.
4. Boundaries between two objects move onto the image's lineart: watershed
   in a band of SNAP_BAND_CELLS SAM cells on each side.
5. instance_alphas() anti-aliases those boundaries (1-2 px) and splits the
   original alpha, so each object keeps 256 levels and all objects add up
   exactly to the original alpha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .alpha_split import SOLID_ALPHA, AlphaSplit, fill_from_nearest

IGNORED = -1
SAM_GRID = 256          # decoder logits are SAM_GRID x SAM_GRID for the whole image
# A real object is at least this many SAM cells thick; thinner leftovers are
# slivers along an edge that the coarse SAM mask missed.
MIN_DEPTH_CELLS = 4.0
# How far (in SAM cells) a boundary may move to reach the lineart.
SNAP_BAND_CELLS = 2.5
_EDGE_KERNEL = np.ones((3, 3), np.uint8)
_SOFT_RADIUS, _SOFT_SIGMA = 2, 0.7   # anti-aliasing of object-object boundaries


@dataclass
class SamObject:
    """One piece defined by clicks; ``logits`` is the decoder output for prompt().
    Pieces with the same ``group`` are one object (e.g. a body and its legs)."""
    points: list[tuple[float, float]]                       # positive clicks, image pixels
    negatives: list[tuple[float, float]] = field(default_factory=list)
    box: tuple[float, float, float, float] | None = None    # x0, y0, x1, y1
    logits: np.ndarray | None = None
    group: int | None = None                                # None: an object on its own

    def to_json(self) -> dict:
        """The prompt only; logits are stored separately (arrays)."""
        return {'points': [list(p) for p in self.points], 'negatives': [list(p) for p in self.negatives],
                'box': list(self.box) if self.box else None, 'group': self.group}

    @classmethod
    def from_json(cls, data: dict, logits: np.ndarray) -> SamObject:
        return cls([tuple(p) for p in data['points']], [tuple(p) for p in data['negatives']],
                   tuple(data['box']) if data['box'] else None, np.asarray(logits, np.float32),
                   data['group'])


def same_object(a: SamObject, b: SamObject) -> bool:
    return a is b or (a.group is not None and a.group == b.group)


def prompt(obj: SamObject, neighbors: list[SamObject]) -> tuple[list, list]:
    """Decoder prompt: the object's clicks, plus the positive clicks of every
    other object as negatives so HQ-SAM does not swallow a touching neighbor."""
    others = [p for n in neighbors if not same_object(n, obj) for p in n.points]
    coords = [*obj.points, *obj.negatives, *others]
    labels = [1] * len(obj.points) + [0] * (len(obj.negatives) + len(others))
    if obj.box is not None:
        x0, y0, x1, y1 = obj.box
        coords += [(x0, y0), (x1, y1)]
        labels += [2, 3]
    return coords, labels


def sam_cell_px(shape: tuple[int, int]) -> float:
    """Image pixels covered by one decoder logit (the SAM mask resolution)."""
    return max(shape) / SAM_GRID


def min_object_depth(shape: tuple[int, int]) -> float:
    """Smallest distance from inside an object to its edge, in image pixels."""
    return MIN_DEPTH_CELLS * sam_cell_px(shape)


def _clicked_regions(obj: SamObject, split: AlphaSplit) -> list[int]:
    h, w = split.labels.shape
    regions = set()
    for x, y in obj.points:
        if 0 <= int(y) < h and 0 <= int(x) < w:
            regions.add(int(split.labels[int(y), int(x)]))
    if obj.box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in obj.box)
        regions.update(np.unique(split.labels[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]).tolist())
    return [r for r in regions if 1 <= r <= split.object_count]


def _claims(alpha: np.ndarray, split: AlphaSplit, objects: list[SamObject]) -> np.ndarray:
    """Index of the SAM object owning each pixel, or -1."""
    h, w = alpha.shape
    owner = np.full((h, w), -1, np.int32)
    best = np.zeros((h, w), np.float32)   # only logits > 0 can win
    open_to_all = (alpha > 0) & (split.labels > split.object_count)   # tiny fragments
    for index, obj in enumerate(objects):
        allowed = open_to_all | np.isin(split.labels, _clicked_regions(obj, split))
        logits = cv2.resize(obj.logits, (w, h), interpolation=cv2.INTER_LINEAR)
        win = allowed & (logits > best)
        best[win] = logits[win]
        owner[win] = index
    return owner


def internal_edges(labels: np.ndarray) -> np.ndarray:
    """Object pixels with a different object (label > 0) among their 8 neighbors."""
    objects = labels > 0
    value = labels.astype(np.float32)
    up = cv2.dilate(np.where(objects, value, 0), _EDGE_KERNEL)
    down = cv2.erode(np.where(objects, value, np.float32(1e9)), _EDGE_KERNEL)
    return objects & ((up > value) | (down < value))


def _snap_band_px(shape: tuple[int, int]) -> int:
    return max(2, math.ceil(SNAP_BAND_CELLS * sam_cell_px(shape)))


def _snap_to_lineart(image_rgb: np.ndarray, labels: np.ndarray, band_px: int) -> np.ndarray:
    edges = internal_edges(labels)
    if not edges.any():
        return labels
    size = 2 * band_px + 1
    band = cv2.dilate(edges.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    band = (band > 0) & (labels > 0)
    background = int(labels.max()) + 1
    markers = np.where(labels > 0, labels, background).astype(np.int32)
    markers[band] = 0                     # watershed decides these from the image gradient
    cv2.watershed(np.ascontiguousarray(image_rgb, dtype=np.uint8), markers)
    snapped = labels.copy()
    take = band & (markers > 0) & (markers < background)
    snapped[take] = markers[take]          # lines (-1) and background keep the SAM decision
    return snapped


def partition(image_rgb: np.ndarray, alpha: np.ndarray, split: AlphaSplit,
              objects: list[SamObject], min_object_ratio: float) -> np.ndarray:
    """int32 map: 0 background, IGNORED tiny fragments, 1..N objects (largest first)."""
    owner = _claims(alpha, split, objects)
    groups: dict = {}
    object_label = np.array([groups.setdefault(('group', o.group) if o.group is not None else ('own', i),
                                               len(groups) + 1) for i, o in enumerate(objects)] + [0], np.int32)
    labels = object_label[owner]          # owner -1 picks the trailing 0
    # Unclaimed parts of each separated region: big and thick -> own object,
    # otherwise -> sliver.
    rest = (owner < 0) & (split.labels >= 1) & (split.labels <= split.object_count)
    rest_u8 = rest.astype(np.uint8)
    _, pieces = cv2.connectedComponents(rest_u8, connectivity=8)
    keys = pieces.astype(np.int64) * (split.object_count + 1) + split.labels
    unique, inverse = np.unique(keys[rest], return_inverse=True)
    solid_area = np.bincount(inverse, weights=(alpha[rest] >= SOLID_ALPHA), minlength=len(unique))
    deep = cv2.distanceTransform(rest_u8, cv2.DIST_L2, 5)[rest] >= min_object_depth(alpha.shape)
    big = (solid_area >= min_object_ratio * alpha.size) & (np.bincount(inverse, weights=deep, minlength=len(unique)) > 0)
    new_ids = np.zeros(len(unique), np.int32)
    new_ids[big] = len(groups) + 1 + np.arange(int(big.sum()), dtype=np.int32)
    labels[rest] = new_ids[inverse]
    fill_from_nearest(labels, rest & (labels == 0))
    labels = _snap_to_lineart(image_rgb, labels, _snap_band_px(alpha.shape))
    return renumber_largest_first(labels, alpha)


def renumber_largest_first(labels: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Objects renumbered 1..N by visible alpha, largest first; visible pixels
    without an object (leftover fragments) become IGNORED."""
    objects = labels > 0
    top = int(labels.max()) if objects.any() else 0
    weight = np.bincount(labels[objects], weights=alpha[objects], minlength=top + 1)
    present = [k for k in np.argsort(-weight, kind='stable') if k > 0 and weight[k] > 0]
    renumber = np.zeros(top + 1, np.int32)
    renumber[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    final = np.where(objects, renumber[np.maximum(labels, 0)], 0).astype(np.int32)
    final[(final == 0) & (alpha > 0)] = IGNORED
    return final


def refine_to_lineart(image_rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Move the edge of a coarse (256x256) HQ-SAM mask onto the image's lineart,
    inside the visible alpha."""
    visible = alpha > 0
    labels = np.where(visible, np.where(mask, 1, 2), 0).astype(np.int32)
    return _snap_to_lineart(image_rgb, labels, _snap_band_px(alpha.shape)) == 1


def instance_alphas(alpha: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    """One uint8 alpha per object (labels 1..N). Object-object boundaries are
    anti-aliased; at every pixel the objects add up to the original alpha."""
    count = int(labels.max())
    out = [np.where(labels == k, alpha, 0).astype(np.uint8) for k in range(1, count + 1)]
    size = 2 * _SOFT_RADIUS + 1
    band = cv2.dilate(internal_edges(labels).astype(np.uint8), np.ones((size, size), np.uint8))
    band = (band > 0) & (labels > 0)
    if not band.any():
        return out
    ys, xs = np.nonzero(band)
    y0, x0 = max(0, ys.min() - size), max(0, xs.min() - size)
    window = labels[y0:ys.max() + size + 1, x0:xs.max() + size + 1]
    present = np.unique(labels[band])
    weights = np.empty((len(present), len(ys)), np.float32)
    for row, k in enumerate(present):
        blurred = cv2.GaussianBlur((window == k).astype(np.float32), (size, size), _SOFT_SIGMA)
        weights[row] = blurred[ys - y0, xs - x0]
    weights /= weights.sum(axis=0, keepdims=True)
    total = alpha[ys, xs].astype(np.int32)
    shares = np.floor(total * weights).astype(np.int32)
    own = np.searchsorted(present, labels[ys, xs])
    columns = np.arange(len(ys))
    shares[own, columns] += total - shares.sum(axis=0)   # rounding rest stays with the pixel's object
    for row, k in enumerate(present):
        out[k - 1][ys, xs] = shares[row]
    return out
