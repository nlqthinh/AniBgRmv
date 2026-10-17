"""Four display-only background choices used by cut previews and SAM Edit.

A background choice never changes the prediction's RGBA pixels or final alpha.
"""

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


BACKGROUND_MODES = ("transparent", "black", "white", "green")
BACKGROUND_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "green": (0, 255, 0),  # pure green screen: #00FF00
}
SWATCH_SIZE = 38


def paint_checker(painter: QPainter, rect, tile: int = 16) -> None:
    """Paint a checkerboard *only* on the preview surface."""
    painter.fillRect(rect, QColor("#FFFFFF"))
    for y in range(rect.top(), rect.bottom() + 1, tile):
        for x in range(rect.left(), rect.right() + 1, tile):
            if ((x - rect.left()) // tile + (y - rect.top()) // tile) % 2 == 0:
                painter.fillRect(x, y, min(tile, rect.right() - x + 1),
                                 min(tile, rect.bottom() - y + 1), QColor("#DCDCDC"))


def checker_icon() -> QIcon:
    canvas = QPixmap(26, 26)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    paint_checker(painter, canvas.rect(), tile=13)
    painter.end()
    return QIcon(canvas)


class BackgroundSwatches(QWidget):
    """Four 38x38 swatches. `backgroundChanged` fires on user clicks only."""

    backgroundChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "transparent"
        self.buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        for mode, tooltip, fill in (
            ("transparent", "Transparent (nền caro)", "#FFFFFF"),
            ("black", "Nền đen", "#000000"),
            ("white", "Nền trắng", "#FFFFFF"),
            ("green", "Green screen (#00FF00)", "#00FF00"),
        ):
            btn = QPushButton(self)
            btn.setObjectName(f"background_{mode}")
            btn.setAccessibleName(tooltip)
            btn.setToolTip(tooltip + " — ảnh xuất vẫn trong suốt")
            btn.setCheckable(True)
            btn.setFixedSize(SWATCH_SIZE, SWATCH_SIZE)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(
                "QPushButton {"
                f" background: {fill}; border: 2px solid #777777;"
                " border-radius: 7px; padding: 0px;"
                "}"
                "QPushButton:hover {border: 2px solid #1976D2;}"
                "QPushButton:checked {border: 4px solid #1976D2;}"
            )
            if mode == "transparent":
                btn.setIcon(checker_icon())
                btn.setIconSize(QSize(26, 26))
            btn.clicked.connect(lambda checked=False, value=mode: self._user_select(value))
            row.addWidget(btn)
            self.buttons[mode] = btn

        self.set_mode("transparent")

    @property
    def mode(self) -> str:
        return self._mode

    def _user_select(self, mode: str):
        if mode != self._mode:
            self.set_mode(mode)
            self.backgroundChanged.emit(mode)

    def set_mode(self, mode: str):
        if mode not in BACKGROUND_MODES:
            raise ValueError(f"Unknown preview background: {mode}")
        self._mode = mode
        for name, button in self.buttons.items():
            button.setChecked(name == mode)
