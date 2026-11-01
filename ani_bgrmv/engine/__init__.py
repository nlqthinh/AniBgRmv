"""Engine package exports (defer heavy ONNX imports until image engine is used).

Video IO and HQ-SAM2 subprocess launcher can be imported independently of
ONNX Runtime; the old ``from ani_bgrmv.engine import BgRemoveEngine`` still works.
"""

__all__ = ['BgRemoveEngine']


def __getattr__(name):
    if name == 'BgRemoveEngine':
        from .bg_remove_engine import BgRemoveEngine
        return BgRemoveEngine
    raise AttributeError(name)
