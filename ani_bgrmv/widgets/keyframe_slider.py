"""Video timeline: a horizontal slider with every keyframe marked in yellow.

Click a mark to go to that keyframe. With the timeline focused on a keyframe,
the Delete key asks to remove it (deleteRequested); everywhere else Delete
keeps its page meaning (the video page: drop the selected object).
"""
from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider

KEYFRAME_COLOR = QColor(255, 204, 0)
MARK_WIDTH = 4         # px
HIT_DISTANCE = 6       # px: a click this close to a mark goes to its keyframe


class KeyframeSlider(QSlider):
    deleteRequested = Signal(int)  # the keyframe the timeline is on

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._keyframes: tuple[int, ...] = ()

    def set_keyframes(self, frames):
        self._keyframes = tuple(sorted(frames))
        self.update()

    def mark_x(self, frame: int) -> float:
        """Centre of the handle when the slider is at ``frame`` (widget pixels)."""
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderHandle, self)
        offset = QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), frame,
                                                groove.width() - handle.width())
        return groove.x() + handle.width() / 2 + offset

    def _on_keyframe(self) -> bool:
        return self.value() in self._keyframes

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._keyframes:
            return
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(KEYFRAME_COLOR)
        top, height = self.height() * 0.2, self.height() * 0.6
        for frame in self._keyframes:
            painter.drawRoundedRect(QRectF(self.mark_x(frame) - MARK_WIDTH / 2, top,
                                           MARK_WIDTH, height), 1.5, 1.5)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._keyframes:
            x = event.position().x()
            nearest = min(self._keyframes, key=lambda frame: abs(self.mark_x(frame) - x))
            if abs(self.mark_x(nearest) - x) <= HIT_DISTANCE:
                self.setFocus(Qt.FocusReason.MouseFocusReason)
                self.setValue(nearest)
                event.accept()
                return
        super().mousePressEvent(event)

    def event(self, event):
        # Take Delete from the page's shortcut while the timeline is on a keyframe.
        if (event.type() == QEvent.Type.ShortcutOverride
                and event.key() == Qt.Key.Key_Delete and self._on_keyframe()):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self._on_keyframe():
            self.deleteRequested.emit(self.value())
            event.accept()
            return
        super().keyPressEvent(event)
