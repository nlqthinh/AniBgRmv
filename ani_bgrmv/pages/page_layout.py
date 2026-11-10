"""Layout pieces shared by the image and video pages (as the annotator):
a top bar, then the picture on the left and a fixed-width tool panel on the right.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout,
                               QWidget)

from ..styles import MUTED_TEXT_COLOR
from ..widgets.elided_label import ElidedLabel

PANEL_WIDTH = 290  # the annotator's side panel
# Inside MainWindow's 11 px margins this gives the picture the annotator's
# size (948 x 673 at 1280 x 760).
PAGE_MARGIN = 7


def button_row(*texts: str, checkable: bool = False) -> tuple[list[QPushButton], QHBoxLayout]:
    """Buttons side by side; a page keeps exactly one of a checkable mode on."""
    row, buttons = QHBoxLayout(), []
    for text in texts:
        button = QPushButton(text)
        button.setCheckable(checkable)
        row.addWidget(button)
        buttons.append(button)
    return buttons, row


def side_panel(tools: QWidget, actions: QWidget) -> QWidget:
    """The tool panel at PANEL_WIDTH: ``tools`` scroll when the window is
    shorter than them (instead of making the window taller than the screen),
    ``actions`` (Dừng, Track, Export...) stay visible under them."""
    area = QScrollArea()
    area.setWidget(tools)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    panel = QWidget()
    column = QVBoxLayout(panel)
    column.setContentsMargins(0, 0, 0, 0)
    column.addWidget(area, 1)
    column.addWidget(actions)
    panel.setFixedWidth(PANEL_WIDTH)
    return panel


def top_bar(title: str) -> tuple[QHBoxLayout, QPushButton, ElidedLabel]:
    """← Trang chủ, the page title, then the one-line status (full text in its tooltip)."""
    back = QPushButton("← Trang chủ")
    title_label = QLabel(title)
    title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
    status = ElidedLabel()
    status.setStyleSheet(f"color: {MUTED_TEXT_COLOR};")
    bar = QHBoxLayout()
    bar.addWidget(back)
    bar.addWidget(title_label)
    bar.addSpacing(16)
    bar.addWidget(status, 1)
    return bar, back, status
