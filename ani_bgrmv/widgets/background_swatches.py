"""Preview background of the image editor: four swatches + an opacity slider.

A background choice is display-only: it never changes the prediction's RGBA
pixels, the final alpha or the saved PNG.
"""

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


BACKGROUND_MODES = ("transparent", "black", "green", "white")
BACKGROUND_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "green": (0, 255, 0),  # pure green screen: #00FF00
}
DEFAULT_BACKGROUND_OPACITY = 100  # 100%: background fully replaces removed pixels
SWATCH_SIZE = 38


def checker_icon() -> QIcon:
    canvas = QPixmap(26, 26)
    canvas.fill(QColor("#FFFFFF"))
    painter = QPainter(canvas)
    for y in range(2):
        for x in range(2):
            if (x + y) % 2 == 0:
                painter.fillRect(x * 13, y * 13, 13, 13, QColor("#DCDCDC"))
    painter.end()
    return QIcon(canvas)


class BackgroundControls(QWidget):
    """Swatches, then 'Độ mờ nền' and its slider. Each background keeps its
    own opacity. ``changed(mode, opacity)`` fires on user changes only."""

    changed = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "transparent"
        self._opacities = {mode: DEFAULT_BACKGROUND_OPACITY for mode in BACKGROUND_MODES}
        self.buttons: dict[str, QPushButton] = {}
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for mode, tooltip, fill in (
            ("transparent", "Transparent (nền caro)", "#FFFFFF"),
            ("black", "Nền đen", "#000000"),
            ("green", "Green screen (#00FF00)", "#00FF00"),
            ("white", "Nền trắng", "#FFFFFF"),
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
                "QPushButton:hover {border: 2px solid #3299FF;}"
                "QPushButton:checked {border: 4px solid #3299FF;}"
            )
            if mode == "transparent":
                btn.setIcon(checker_icon())
                btn.setIconSize(QSize(26, 26))
            btn.clicked.connect(lambda checked=False, value=mode: self._select(value))
            row.addWidget(btn)
            self.buttons[mode] = btn
        row.addStretch(1)

        self.opacity_label = QLabel()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setToolTip("0%: thấy ảnh gốc; 100%: nền thay hẳn phần đã cắt. "
                                       "Chỉ đổi phần xem, không đổi ảnh lưu.")
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(row)
        layout.addWidget(self.opacity_label)
        layout.addWidget(self.opacity_slider)
        self._show_mode()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def opacity(self) -> int:
        return self._opacities[self._mode]

    def _show_mode(self):
        for name, button in self.buttons.items():
            button.setChecked(name == self._mode)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(self.opacity)
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"Độ mờ nền: {self.opacity}%")

    def _select(self, mode: str):
        self._mode = mode
        self._show_mode()
        self.changed.emit(mode, self.opacity)

    def _opacity_changed(self, value: int):
        self._opacities[self._mode] = int(value)
        self.opacity_label.setText(f"Độ mờ nền: {value}%")
        self.changed.emit(self._mode, int(value))
