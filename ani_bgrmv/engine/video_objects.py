"""Objects of the video editor, as in the annotator (Gan_nhan). Pure numpy.

Each frame is a label map: 0 = no object, k = Object k; a pixel belongs to at
most one object. Restore gives only unowned pixels to the selected object,
Erase only takes the selected object's own pixels: no edit ever touches
another object.

A keyframe stores the whole label map of one edited frame, as the user sees
it. Track prompts every object on every keyframe with exactly that, so an
object not painted on a keyframe counts as absent there. A span Track
(track_span) re-tracks only the frames between two keyframes.

ObjectEdits is immutable: every change returns a new one that shares the
unchanged label maps, so Undo keeps whole states cheaply. Merging or dropping
an object after a Track relabels the tracked frames through a 256-entry
table instead of rewriting the whole video.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

NO_OBJECT = 0
MAX_OBJECTS = 255  # labels are uint8


@dataclass(frozen=True, eq=False)  # holds arrays: compare states by identity
class ObjectEdits:
    keyframes: Mapping[int, np.ndarray] = field(default_factory=dict)  # frame -> labels
    objects: tuple[int, ...] = (1,)
    relabel: Mapping[int, int] = field(default_factory=dict)  # tracked label -> shown label
    active: int | None = 1

    def __post_init__(self):
        # Read-only views: a change must go through replace(), never in place.
        object.__setattr__(self, 'keyframes', MappingProxyType(dict(self.keyframes)))
        object.__setattr__(self, 'relabel', MappingProxyType(dict(self.relabel)))


def labels_at(edits: ObjectEdits, tracked: Sequence[np.ndarray] | None, frame: int,
              shape: tuple[int, int]) -> np.ndarray:
    """The label map shown (and exported) for ``frame``. Do not modify it."""
    if frame in edits.keyframes:
        return edits.keyframes[frame]
    if tracked is None:
        return np.zeros(shape, np.uint8)
    labels = tracked[frame]
    if not edits.relabel:
        return labels
    table = np.arange(256, dtype=np.uint8)
    for old, new in edits.relabel.items():
        table[old] = new
    return table[labels]


def edit_region(edits: ObjectEdits, tracked, frame: int, shape, region: np.ndarray,
                operation: str) -> ObjectEdits | None:
    """Restore: unowned pixels of ``region`` join the active object. Erase: the
    active object's pixels of ``region`` become unowned. None: nothing changed."""
    if edits.active is None:
        return None
    labels = labels_at(edits, tracked, frame, shape)
    owner = NO_OBJECT if operation == 'restore' else edits.active
    changed = np.asarray(region, bool) & (labels == owner)
    if not changed.any():
        return None
    result = labels.copy()
    result[changed] = edits.active if operation == 'restore' else NO_OBJECT
    return replace(edits, keyframes={**edits.keyframes, frame: result})


def drop_keyframe(edits: ObjectEdits, frame: int) -> ObjectEdits | None:
    """The frame shows its tracked mask again (None: not a keyframe)."""
    if frame not in edits.keyframes:
        return None
    return replace(edits, keyframes={k: v for k, v in edits.keyframes.items() if k != frame})


def track_span(keyframes, frame: int, count: int, forward: bool) -> tuple[int, ...]:
    """Frames from keyframe ``frame`` to the nearest keyframe after it
    (``forward``) or before it, else to the clip's end; in tracking order."""
    if forward:
        end = min((k for k in keyframes if k > frame), default=count - 1)
        return tuple(range(frame, end + 1))
    end = max((k for k in keyframes if k < frame), default=0)
    return tuple(range(frame, end - 1, -1))


def object_mask(labels: np.ndarray, object_id: int | None) -> np.ndarray:
    """0/255 mask of one object (the alpha the image editor's Guided works on)."""
    return (labels == object_id).astype(np.uint8) * 255


def guided_region(labels: np.ndarray, object_id: int, guided_alpha: np.ndarray,
                  operation: str) -> np.ndarray:
    """Pixels a Guided stroke changed for one object: added ones (Restore) or
    removed ones (Erase), from the alpha Guided returned for that object."""
    before = labels == object_id
    after = np.asarray(guided_alpha) >= 128
    return after & ~before if operation == 'restore' else before & ~after


def next_object_id(edits: ObjectEdits) -> int:
    """Never reuses an id that tracked frames may still show through relabel."""
    return max((*edits.objects, *edits.relabel, 0)) + 1


def new_object(edits: ObjectEdits) -> ObjectEdits:
    object_id = next_object_id(edits)
    if object_id > MAX_OBJECTS:
        raise ValueError(f'Tối đa {MAX_OBJECTS} object.')
    return replace(edits, objects=(*edits.objects, object_id), active=object_id)


def select(edits: ObjectEdits, object_id: int) -> ObjectEdits:
    return replace(edits, active=object_id)


def _move_label(edits: ObjectEdits, source: int, target: int) -> dict:
    """Keyframes and relabel table with every ``source`` pixel shown as ``target``."""
    keyframes = {frame: np.where(labels == source, np.uint8(target), labels)
                 if (labels == source).any() else labels
                 for frame, labels in edits.keyframes.items()}
    relabel = {old: (target if new == source else new) for old, new in edits.relabel.items()}
    relabel[source] = target
    return {'keyframes': keyframes, 'relabel': relabel}


def merge_into_active(edits: ObjectEdits, source: int) -> ObjectEdits | None:
    """Object ``source`` becomes part of the active object on every frame."""
    if edits.active is None or source in (NO_OBJECT, edits.active) or source not in edits.objects:
        return None
    return replace(edits, objects=tuple(o for o in edits.objects if o != source),
                   **_move_label(edits, source, edits.active))


def drop_active(edits: ObjectEdits) -> ObjectEdits | None:
    """Remove the active object from every frame; the next object becomes active."""
    if edits.active is None:
        return None
    index = edits.objects.index(edits.active)
    objects = tuple(o for o in edits.objects if o != edits.active)
    active = objects[min(index, len(objects) - 1)] if objects else None
    return replace(edits, objects=objects, active=active,
                   **_move_label(edits, edits.active, NO_OBJECT))


def after_track(edits: ObjectEdits) -> ObjectEdits:
    """A new Track already uses the merged/dropped labels of the keyframes."""
    return replace(edits, relabel={})


def shares(labels: np.ndarray, objects: Sequence[int]) -> dict[int, float]:
    """Share of the frame each object covers (for the object list)."""
    counts = np.bincount(labels.ravel(), minlength=256)
    return {object_id: counts[object_id] / labels.size for object_id in objects}


def track_agreement(old: Sequence[np.ndarray], new: Sequence[np.ndarray]) -> dict:
    """How close two Tracks are, per object: IoU on frames where either has it.

    Lets a setting change be judged by numbers (same keyframes, Track again).
    """
    if len(old) != len(new):
        raise ValueError('Hai lần Track có số frame khác nhau.')
    present = sum(np.bincount(f.ravel(), minlength=256) > 0 for f in (*old, *new))
    result = {}
    for object_id in map(int, np.flatnonzero(present[1:]) + 1):
        ious = {}
        for frame, (a, b) in enumerate(zip(old, new)):
            a, b = a == object_id, b == object_id
            union = np.count_nonzero(a | b)
            if union:
                ious[frame] = np.count_nonzero(a & b) / union
        worst = min(ious, key=ious.get)
        result[object_id] = {'iou_min': round(ious[worst], 4), 'worst_frame': worst,
                             'iou_mean': round(sum(ious.values()) / len(ious), 4)}
    return result


def export_alpha(labels: np.ndarray) -> np.ndarray:
    """Export keeps every object: one 0/255 alpha for the whole frame."""
    return (labels > NO_OBJECT).astype(np.uint8) * 255
