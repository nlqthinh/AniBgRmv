"""Two handles on one track: the part [start, end) of a video to keep.

Drag a handle to move that end, drag the highlighted part to move both, click
the track elsewhere to bring the nearest handle there (as video trimming on a
phone). The part is never longer than ``max_span``: pulling one handle further
drags the other along.
"""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

HANDLE_WIDTH = 10   # px
TRACK_HEIGHT = 8    # px
TRACK_COLOR = QColor('#3A4250')
RANGE_COLOR = QColor('#4C8DF6')
HANDLE_COLOR = QColor('#F0F0F0')


class RangeSlider(QWidget):
    rangeChanged = Signal(int, int)  # start, end (end excluded)
    handleMoved = Signal(int)        # value under the dragged handle, for a preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self._maximum = 1
        self._max_span = 1
        self._start, self._end = 0, 1
        self._drag: str | None = None   # 'start', 'end' or 'range'
        self._grab_offset = 0
        self.setMinimumHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_limits(self, maximum: int, max_span: int):
        """Values 0..maximum; at most ``max_span`` kept. Starts at the first ``max_span``."""
        self._maximum, self._max_span = max(1, int(maximum)), max(1, int(max_span))
        self.set_values(0, self._max_span)

    def values(self) -> tuple[int, int]:
        return self._start, self._end

    def set_values(self, start: int, end: int):
        end = max(1, min(int(end), self._maximum))
        start = max(0, min(int(start), end - 1))
        start = max(start, end - self._max_span)
        if (start, end) != (self._start, self._end):
            self._start, self._end = start, end
            self.update()
            self.rangeChanged.emit(start, end)

    # -- geometry -----------------------------------------------------------------
    def _x(self, value: float) -> float:
        usable = max(1, self.width() - HANDLE_WIDTH)
        return HANDLE_WIDTH / 2 + usable * value / self._maximum

    def _value(self, x: float) -> int:
        usable = max(1, self.width() - HANDLE_WIDTH)
        return round((x - HANDLE_WIDTH / 2) * self._maximum / usable)

    # -- drawing ----------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        middle = self.height() / 2
        painter.setBrush(TRACK_COLOR)
        painter.drawRoundedRect(QRectF(0, middle - TRACK_HEIGHT / 2, self.width(), TRACK_HEIGHT), 3, 3)
        left, right = self._x(self._start), self._x(self._end)
        painter.setBrush(RANGE_COLOR)
        painter.drawRect(QRectF(left, middle - TRACK_HEIGHT / 2, right - left, TRACK_HEIGHT))
        painter.setBrush(HANDLE_COLOR)
        for x in (left, right):
            painter.drawRoundedRect(QRectF(x - HANDLE_WIDTH / 2, 2, HANDLE_WIDTH, self.height() - 4), 3, 3)

    # -- mouse ------------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            return super().mousePressEvent(event)
        x = event.position().x()
        left, right = self._x(self._start), self._x(self._end)
        if abs(x - left) <= HANDLE_WIDTH and abs(x - left) <= abs(x - right):
            self._drag = 'start'
        elif abs(x - right) <= HANDLE_WIDTH:
            self._drag = 'end'
        elif left < x < right:
            self._drag, self._grab_offset = 'range', self._value(x) - self._start
        else:
            self._drag = 'start' if x < left else 'end'
            self._move_to(self._value(x))
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag:
            self._move_to(self._value(event.position().x()))
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag = None
        super().mouseReleaseEvent(event)

    def _move_to(self, value: int):
        if self._drag == 'start':
            start = max(0, min(value, self._end - 1))
            self.set_values(start, min(self._end, start + self._max_span))
            self.handleMoved.emit(self._start)
        elif self._drag == 'end':
            end = max(self._start + 1, min(value, self._maximum))
            self.set_values(max(self._start, end - self._max_span), end)
            self.handleMoved.emit(self._end - 1)
        elif self._drag == 'range':
            span = self._end - self._start
            start = max(0, min(value - self._grab_offset, self._maximum - span))
            self.set_values(start, start + span)
            self.handleMoved.emit(self._start)
