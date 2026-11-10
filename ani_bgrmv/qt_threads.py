"""Safe Qt thread-state queries for UI actions after worker completion.

PySide6 can retain a Python wrapper after its underlying QThread is deleted.
"""
from shiboken6 import isValid


def thread_is_valid(thread) -> bool:
    """Check the C++ object rather than only its remaining Python wrapper."""
    if thread is None:
        return False
    try:
        return bool(isValid(thread))
    except RuntimeError:
        return False


def thread_is_running(thread) -> bool:
    """Return False for absent/deleted threads without touching a dead C++ object."""
    if not thread_is_valid(thread):
        return False
    try:
        return bool(thread.isRunning())
    except RuntimeError:  # C++ QObject was destroyed between validation and read.
        return False
