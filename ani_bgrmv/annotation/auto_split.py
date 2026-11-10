"""Propose HQ-SAM objects for groups of touching objects, conservatively.

Greedy peeling, one separated region at a time: click the point deepest
inside the part nobody owns yet. The click is first decoded alone:
  * if HQ-SAM's mask covers most of ONE object already found, the click is
    a part of it (head, legs, the rest of a creature) -> same group;
  * otherwise it is decoded with the other objects' clicks as negatives. A
    piece smaller than SMALL_PART_RATIO of the biggest object that touches
    exactly one object is a part of it (blood, an effect, a held weapon);
    anything else is a new object.
Then every piece is decoded again with the other objects' clicks as negatives.
A region whose only object covers SINGLE_OBJECT_COVERAGE of it keeps its
exact cutout. When in doubt nothing is split: one more click in the annotator
is cheaper than merging pieces back together.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..engine.hqsam_onnx import mask_from_logits
from .alpha_split import SOLID_ALPHA, AlphaSplit
from .partition import SamObject, min_object_depth, prompt

MAX_OBJECTS_PER_REGION = 30
SINGLE_OBJECT_COVERAGE = 0.8
SAME_OBJECT_OVERLAP = 0.5   # share of a found object inside a lone click's mask
SMALL_PART_RATIO = 0.25     # vs the biggest object of the region
_TOUCH = np.ones((5, 5), np.uint8)

Decode = Callable[[list, list], np.ndarray]   # (coords, labels) -> logits [256,256]


def _decode(obj: SamObject, neighbors: list[SamObject], decode: Decode, shape) -> np.ndarray:
    obj.logits = decode(*prompt(obj, neighbors))
    return mask_from_logits(obj.logits, shape)


class _Region:
    """Peeling state of one separated region: its pieces and their masks."""

    def __init__(self, region: np.ndarray):
        self.region = region
        self.unowned = region.astype(np.uint8)
        self.pieces: list[SamObject] = []
        self.masks: list[np.ndarray] = []      # each piece's mask inside the region

    def group_masks(self) -> dict[int, np.ndarray]:
        union: dict[int, np.ndarray] = {}
        for piece, mask in zip(self.pieces, self.masks):
            union[piece.group] = union[piece.group] | mask if piece.group in union else mask
        return union

    def group_covering(self, mask: np.ndarray) -> int | None:
        """The one group most of whose area lies inside ``mask``."""
        groups = {g for g, m in self.group_masks().items()
                  if np.count_nonzero(mask & m) >= SAME_OBJECT_OVERLAP * np.count_nonzero(m)}
        return groups.pop() if len(groups) == 1 else None

    def small_part_of(self, piece: np.ndarray) -> int | None:
        """The group a small piece belongs to, if it touches exactly one group."""
        unions = self.group_masks()
        if not unions or np.count_nonzero(piece) >= SMALL_PART_RATIO * max(
                np.count_nonzero(m) for m in unions.values()):
            return None
        near = cv2.dilate(piece.astype(np.uint8), _TOUCH) > 0
        touched = [g for g, m in unions.items() if (near & m).any()]
        return touched[0] if len(touched) == 1 else None


def _peel(region: np.ndarray, shape, decode: Decode, min_area: float) -> list[SamObject]:
    state = _Region(region)
    min_depth = min_object_depth(shape)   # thinner leftovers are slivers of found objects
    groups = 0
    for _ in range(2 * MAX_OBJECTS_PER_REGION):
        if groups == MAX_OBJECTS_PER_REGION or np.count_nonzero(state.unowned) < min_area:
            break
        depth = cv2.distanceTransform(state.unowned, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
        if depth[y, x] < min_depth:
            break
        obj = SamObject(points=[(float(x), float(y))])
        piece = _decode(obj, [], decode, shape) & region
        obj.group = state.group_covering(piece)
        if obj.group is None and state.pieces:
            piece = _decode(obj, state.pieces, decode, shape) & region
        gained = piece & (state.unowned > 0)
        if np.count_nonzero(gained) < min_area:
            # HQ-SAM returned nothing new here: never click this spot again.
            cv2.circle(state.unowned, (int(x), int(y)), int(depth[y, x]) + 1, 0, -1)
            continue
        if obj.group is None:
            obj.group = state.small_part_of(gained)
        if obj.group is None:
            obj.group, groups = groups, groups + 1
        state.pieces.append(obj)
        state.masks.append(piece)
        state.unowned[piece] = 0
    if groups == 1 and np.count_nonzero(state.group_masks()[0]) >= (
            SINGLE_OBJECT_COVERAGE * np.count_nonzero(region)):
        return []   # one object with its parts: keep the exact cutout
    return state.pieces


def auto_split(alpha: np.ndarray, split: AlphaSplit, decode: Decode,
               min_object_ratio: float) -> list[SamObject]:
    min_area = min_object_ratio * alpha.size
    solid = alpha >= SOLID_ALPHA
    proposals: list[SamObject] = []
    for region in range(1, split.object_count + 1):
        pieces = _peel(solid & (split.labels == region), alpha.shape, decode, min_area)
        if len({p.group for p in pieces}) > 1:
            for piece in pieces:
                _decode(piece, pieces, decode, alpha.shape)
        first_group = 1 + max((p.group for p in proposals), default=-1)
        for piece in pieces:
            piece.group += first_group   # groups are unique across regions
        proposals += pieces
    return proposals
