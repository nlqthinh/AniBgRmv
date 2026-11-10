"""Clip a UI box to an image; reject zero-area intersections."""
from __future__ import annotations


def normalized_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Return inclusive xyxy coordinates or None if the clipped box is empty.

    Box corners may be outside the image; a completely outside drag or a
    single-pixel line is not a valid SAM selection.
    """
    if box is None or len(box) != 4 or width < 2 or height < 2:
        return None
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = sorted((max(0, min(width - 1, x0)), max(0, min(width - 1, x1))))
    y0, y1 = sorted((max(0, min(height - 1, y0)), max(0, min(height - 1, y1))))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
