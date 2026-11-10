"""One image in the annotator, managed object by object (one color each).

The edit starts from a label map (the proposal: 0 background, IGNORED = gray,
no object; 1..N objects) and replays operations in the order they were made.
Every operation works on ONE object and never changes another object's pixels:
  ['restore', id, how]  gray pixels inside the region become object ``id``
  ['erase', id, how]    pixels of object ``id`` inside the region become gray
  ['new', id]           an empty object with a new color
  ['merge', id, other]  the whole object ``other`` becomes part of ``id``
  ['drop', id]          every pixel of ``id`` becomes gray
``how`` is the region: {'tool': 'brush', 'points', 'radius'} or {'tool': 'box',
'box'} draw it by hand; {'tool': 'sam', 'prompt', 'mask'} is HQ-SAM's mask for
the prompt, moved onto the lineart (stored as packed bits, so a reopened edit
is exactly the same). To move pixels from object A to object B: erase them
from A (they turn gray), then restore them into B.
Object ids never change while editing, so colors stay put; the saved files
are numbered largest first. No Qt here: the window runs this on its worker thread.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..engine.hqsam_onnx import BOX_LABELS, mask_from_logits
from .partition import IGNORED, instance_alphas, refine_to_lineart, renumber_largest_first, sam_cell_px

EDIT_FORMAT = 'anibgrmv-annotation-edit-v2'
RESTORE, ERASE = 'restore', 'erase'
MAX_STROKE_POINTS = 8
# The active object's own anchor click joins a guided prompt (positive to
# restore, negative to erase) only when it is this many SAM cells from the clicks.
ANCHOR_MIN_CELLS = 2.0

Decode = Callable[[list, list], np.ndarray]   # (coords, labels) -> logits [256,256]


def _spread(points: list, count: int) -> list:
    """At most ``count`` points evenly spread along a stroke."""
    if len(points) <= count:
        return list(points)
    return [points[round(i * (len(points) - 1) / (count - 1))] for i in range(count)]


class ImageEdit:
    def __init__(self, rgb: np.ndarray, alpha: np.ndarray, base: np.ndarray, *,
                 decode: Decode | None = None, ops=(), masks=()):
        self.rgb, self.alpha = rgb, alpha
        self.base = np.asarray(base, np.int32)
        self._decode = decode            # None: no HQ-SAM features for this image
        self.ops: list[list] = [list(op) for op in ops]
        self._masks: list[np.ndarray] = [np.asarray(m, np.uint8) for m in masks]   # packed bits
        self._redo: list[tuple] = []
        self._replay()
        self.active = self._first_object()

    @property
    def can_use_sam(self) -> bool:
        return self._decode is not None

    # -- replay -----------------------------------------------------------------
    def _region(self, how: dict) -> np.ndarray:
        shape = self.alpha.shape
        if how['tool'] == 'sam':
            return np.unpackbits(self._masks[how['mask']], count=self.alpha.size).reshape(shape).astype(bool)
        region = np.zeros(shape, np.uint8)
        if how['tool'] == 'box':
            x0, y0, x1, y1 = (int(round(v)) for v in how['box'])
            region[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
        else:
            radius = max(1, int(round(how['radius'])))
            path = np.round(np.asarray(how['points'], np.float32)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(region, [path], False, 1, thickness=2 * radius)
            for x, y in path[:, 0]:
                cv2.circle(region, (int(x), int(y)), radius, 1, -1)
        return region > 0

    def _apply(self, labels: np.ndarray, alive: set, op: list) -> None:
        kind = op[0]
        if kind == RESTORE:
            labels[self._region(op[2]) & (self.alpha > 0) & (labels <= 0)] = op[1]
        elif kind == ERASE:
            labels[self._region(op[2]) & (labels == op[1])] = IGNORED
        elif kind == 'new':
            alive.add(op[1])
        elif kind == 'merge':
            labels[labels == op[2]] = op[1]
            alive.discard(op[2])
        elif kind == 'drop':
            labels[labels == op[1]] = IGNORED
            alive.discard(op[1])
        else:
            raise ValueError(f'Thao tác không hợp lệ: {kind}')

    def _replay(self) -> None:
        labels = self.base.copy()
        alive = set(np.unique(labels[labels > 0]).tolist())
        for op in self.ops:
            self._apply(labels, alive, op)
        self._labels, self._alive = labels, alive

    def _push(self, op: list, mask: np.ndarray | None = None) -> None:
        if mask is not None:
            op[2]['mask'] = len(self._masks)
            self._masks.append(np.packbits(mask.ravel()))
        self.ops.append(op)
        self._apply(self._labels, self._alive, op)

    def _first_object(self) -> int | None:
        return min(self._alive) if self._alive else None

    # -- objects ------------------------------------------------------------------
    def objects(self) -> list[tuple[int, float]]:
        """(id, share of the visible alpha) of every object, by id."""
        visible = max(int(np.count_nonzero(self.alpha)), 1)
        counts = np.bincount(np.maximum(self._labels, 0).ravel(), minlength=max(self._alive, default=0) + 1)
        return [(i, counts[i] / visible) for i in sorted(self._alive)]

    def object_at(self, point) -> int | None:
        h, w = self._labels.shape
        x, y = int(point[0]), int(point[1])
        label = int(self._labels[y, x]) if 0 <= x < w and 0 <= y < h else 0
        return label if label > 0 else None

    def select(self, object_id: int) -> bool:
        if object_id not in self._alive:
            return False
        self.active = object_id
        return True

    def new_object(self) -> int:
        used = [int(self.base.max())] + [v for op in self.ops for v in op[1:] if isinstance(v, int)]
        object_id = max(used + [0]) + 1
        self._redo.clear()
        self._push(['new', object_id])
        self.active = object_id
        return object_id

    def merge_into_active(self, other: int | None) -> bool:
        if self.active is None or other is None or other == self.active or other not in self._alive:
            return False
        self._redo.clear()
        self._push(['merge', self.active, other])
        return True

    def drop_active(self) -> bool:
        if self.active is None:
            return False
        self._redo.clear()
        self._push(['drop', self.active])
        self.active = self._first_object()
        return True

    # -- restore / erase --------------------------------------------------------
    def _anchor(self) -> list[float] | None:
        """The pixel deepest inside the active object."""
        depth = cv2.distanceTransform((self._labels == self.active).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
        return [float(x), float(y)] if depth[y, x] > 0 else None

    def _prompt(self, mode: str, tool: str, geometry) -> tuple[list, list]:
        if tool == 'box':
            x0, y0, x1, y1 = geometry
            return [[x0, y0], [x1, y1]], BOX_LABELS.tolist()
        points = [list(geometry)] if tool == 'point' else [list(p) for p in _spread(geometry, MAX_STROKE_POINTS)]
        coords, labels = points, [1] * len(points)
        anchor = self._anchor()
        if anchor is not None and min(np.hypot(anchor[0] - x, anchor[1] - y) for x, y in points) > (
                ANCHOR_MIN_CELLS * sam_cell_px(self.alpha.shape)):
            coords, labels = coords + [anchor], labels + [1 if mode == RESTORE else 0]
        return coords, labels

    def _changes(self, mode: str, region: np.ndarray) -> bool:
        target = ((self.alpha > 0) & (self._labels <= 0)) if mode == RESTORE else (self._labels == self.active)
        return bool((region & target).any())

    def _need_active(self) -> None:
        if self.active is None:
            raise RuntimeError('Chọn hoặc tạo một object trước.')

    def guided(self, mode: str, tool: str, geometry) -> bool:
        """HQ-SAM decides the region from a point, a brush stroke or a box.
        Returns False when it would change nothing."""
        if not self.can_use_sam:
            raise RuntimeError('Ảnh này không có dữ liệu HQ-SAM: dùng chế độ Manual.')
        self._need_active()
        coords, labels = self._prompt(mode, tool, geometry)
        mask = mask_from_logits(self._decode(coords, labels), self.alpha.shape) & (self.alpha > 0)
        mask = refine_to_lineart(self.rgb, self.alpha, mask)
        if not self._changes(mode, mask):
            return False
        self._redo.clear()
        self._push([mode, self.active, {'tool': 'sam', 'prompt': {'coords': coords, 'labels': labels}}], mask)
        return True

    def manual(self, mode: str, tool: str, geometry, radius: float = 0.0) -> bool:
        """Exactly the drawn brush stroke or box. Returns False when it would change nothing."""
        self._need_active()
        how = ({'tool': 'box', 'box': [float(v) for v in geometry]} if tool == 'box' else
               {'tool': 'brush', 'points': [[float(x), float(y)] for x, y in geometry], 'radius': float(radius)})
        if not self._changes(mode, self._region(how)):
            return False
        self._redo.clear()
        self._push([mode, self.active, how])
        return True

    # -- undo -------------------------------------------------------------------
    def undo(self) -> bool:
        if not self.ops:
            return False
        op = self.ops.pop()
        mask = self._masks.pop() if op[0] in (RESTORE, ERASE) and op[2]['tool'] == 'sam' else None
        self._redo.append((op, mask, self.active))
        self._replay()
        if self.active not in self._alive:
            self.active = self._first_object()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        op, mask, active = self._redo.pop()
        if mask is not None:
            self._masks.append(mask)
        self.ops.append(op)
        self._apply(self._labels, self._alive, op)
        self.active = active if active in self._alive else self._first_object()
        return True

    # -- results ----------------------------------------------------------------
    def labels(self) -> np.ndarray:
        """Stable object ids (colors) for display: 0 background, IGNORED gray."""
        return self._labels

    def gray_share(self) -> float:
        """Share of the visible alpha that belongs to no object (left out of the files)."""
        visible = self.alpha > 0
        return float(np.count_nonzero(visible & (self._labels <= 0)) / max(np.count_nonzero(visible), 1))

    def object_count(self) -> int:
        return len([i for i, share in self.objects() if share > 0])

    def object_alphas(self) -> list[np.ndarray]:
        """Saved files: 8-bit alpha per object, largest first."""
        return instance_alphas(self.alpha, renumber_largest_first(self._labels, self.alpha))

    def recipe(self) -> tuple[dict, dict[str, np.ndarray]]:
        """JSON-able decisions + arrays (base map, guided masks), to save and reopen."""
        packed = (self.alpha.size + 7) // 8
        masks = np.stack(self._masks) if self._masks else np.zeros((0, packed), np.uint8)
        return ({'format': EDIT_FORMAT, 'ops': self.ops, 'active': self.active},
                {'base': self.base.astype(np.int16), 'masks': masks})
