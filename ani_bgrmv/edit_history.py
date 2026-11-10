"""Undo/Redo history shared by the Single-image and Video edit pages (no Qt).

A state is whatever a page needs to restore: an alpha array (Single) or a
(frame, keyframe-or-None) pair (Video). Each list keeps at most ``max_items``
states; with ``budget_bytes`` it also drops the oldest while its arrays
exceed the budget, always keeping the newest state.

undo()/redo() take ``counterpart``: given the state being restored, it
returns the state to put on the other list, read *before* the page applies
the restored one (Video: the current keyframe of that same frame).
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar('T')


def _nbytes(state) -> int:
    if isinstance(state, tuple):
        return sum(_nbytes(item) for item in state)
    return int(getattr(state, 'nbytes', 0))


class EditHistory(Generic[T]):
    def __init__(self, max_items: int, budget_bytes: int | None = None):
        self._max_items = int(max_items)
        self._budget = budget_bytes
        self._undo: list[T] = []
        self._redo: list[T] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, state: T) -> None:
        """A new edit: remember the state before it and drop the Redo branch."""
        self._undo.append(state)
        self._trim(self._undo)
        self._redo.clear()

    def discard_last(self) -> T | None:
        """Take back the last push when its edit failed or changed nothing."""
        return self._undo.pop() if self._undo else None

    def undo(self, counterpart: Callable[[T], T]) -> T | None:
        return self._move(self._undo, self._redo, counterpart)

    def redo(self, counterpart: Callable[[T], T]) -> T | None:
        return self._move(self._redo, self._undo, counterpart)

    def _move(self, source: list[T], target: list[T], counterpart) -> T | None:
        if not source:
            return None
        state = source.pop()
        target.append(counterpart(state))
        self._trim(target)
        return state

    def _trim(self, states: list[T]) -> None:
        while len(states) > self._max_items:
            states.pop(0)
        if self._budget is not None:
            while len(states) > 1 and sum(_nbytes(s) for s in states) > self._budget:
                states.pop(0)
