from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QProgressBar, QVBoxLayout, QHBoxLayout

from ..styles import CANCEL_BUTTON_STYLE, DANGER_BUTTON_STYLE
from ..utils import pil_to_pixmap
from .background_swatches import BackgroundSwatches, BACKGROUND_COLORS, SWATCH_SIZE, paint_checker

class UploadPreviewCanvas(QFrame):
    """
    Một layout duy nhất cho cả upload và preview.

    State:
    - upload: hiện vùng upload.
    - loading: hiện thanh tiến trình + nút Dừng cắt.
    - preview: hiện ảnh preview + nút X.
    """

    clicked = Signal()
    filesDropped = Signal(list)
    clearRequested = Signal()
    cancelRequested = Signal()
    backgroundChanged = Signal(str)

    def __init__(self, title: str):
        super().__init__()
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(360)
        self._mode = "upload"
        self._pixmap: QPixmap | None = None
        self._is_rgba_cutout = False
        self._background_mode = "transparent"

        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #8A8A8A;
                border-radius: 18px;
                background: #FFFFFF;
            }
            QFrame:hover {
                border: 2px dashed #3D7EFF;
                background: #F8FAFF;
            }
            QLabel {
                border: none;
                background: transparent;
            }
            """
        )

        self.btn_clear = QPushButton("X")
        self.btn_clear.setStyleSheet(DANGER_BUTTON_STYLE)
        self.btn_clear.setFixedSize(SWATCH_SIZE, SWATCH_SIZE)
        self.btn_clear.setVisible(False)
        self.btn_clear.clicked.connect(self.clearRequested.emit)

        self.background_swatches = BackgroundSwatches(self)
        self.background_swatches.setVisible(False)
        self.background_swatches.backgroundChanged.connect(self.set_background_mode)

        clear_row = QHBoxLayout()
        clear_row.setSpacing(6)
        clear_row.addWidget(self.background_swatches)
        clear_row.addStretch(1)
        clear_row.addWidget(self.btn_clear)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            """
            QLabel {
                font-size: 30px;
                font-weight: 900;
                color: white;
                background: #111111;
                border: 2px solid #000000;
                border-radius: 12px;
                padding: 10px 24px;
            }
            """
        )

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setVisible(False)

        self.loading_label = QLabel("Đang xử lý...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 17px; font-weight: 700; color: #333333;")
        self.loading_label.setVisible(False)

        self.progress = QProgressBar()
        self.progress.setMinimumWidth(420)
        self.progress.setVisible(False)

        self.btn_cancel = QPushButton("Dừng cắt")
        self.btn_cancel.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancelRequested.emit)

        center_box = QVBoxLayout()
        center_box.setSpacing(16)
        center_box.addStretch(1)
        center_box.addWidget(self.title_label, alignment=Qt.AlignCenter)
        center_box.addWidget(self.loading_label)
        center_box.addWidget(self.progress, alignment=Qt.AlignCenter)
        center_box.addWidget(self.btn_cancel, alignment=Qt.AlignCenter)
        center_box.addWidget(self.preview_label)
        center_box.addStretch(1)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.addLayout(clear_row)
        layout.addLayout(center_box)
        self.setLayout(layout)

    def set_upload_mode(self):
        self._mode = "upload"
        self._pixmap = None
        self._is_rgba_cutout = False
        self.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setVisible(False)
        self.background_swatches.setVisible(False)
        self.title_label.setVisible(True)
        self.loading_label.setVisible(False)
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)
        self.preview_label.clear()
        self.preview_label.setVisible(False)
        self.setEnabled(True)

    def set_loading_mode(
        self,
        text: str,
        current: int | None = None,
        total: int | None = None,
        allow_cancel: bool = True,
    ):
        self._mode = "loading"
        self.setCursor(Qt.ArrowCursor)
        self.btn_clear.setVisible(False)
        self.background_swatches.setVisible(False)
        self.title_label.setVisible(False)
        self.preview_label.setVisible(False)
        self.loading_label.setVisible(True)
        self.loading_label.setText(text)
        self.progress.setVisible(True)
        self.btn_cancel.setVisible(allow_cancel)
        self.btn_cancel.setEnabled(allow_cancel)

        if current is None or total is None or total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(current)

    def set_cancel_waiting_mode(self, text: str = "Đang hủy thao tác..."):
        self.set_loading_mode(text, allow_cancel=False)

    def set_preview_pil(self, pil_image, caption: str = ""):
        self._mode = "preview"
        self.setCursor(Qt.ArrowCursor)
        self.btn_clear.setVisible(True)
        self.background_swatches.setVisible(True)
        self._is_rgba_cutout = False
        self.title_label.setVisible(False)
        self.loading_label.setVisible(False)
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.preview_label.setVisible(True)
        self._pixmap = pil_to_pixmap(pil_image)
        self._update_scaled_preview()
        if caption:
            self.preview_label.setToolTip(caption)

    def set_preview_cutout(self, cutout_pil, caption: str = ""):
        """Display the real RGBA cutout with a selectable *visual-only* backing."""
        self._mode = "preview"
        self.setCursor(Qt.ArrowCursor)
        self.btn_clear.setVisible(True)
        self.background_swatches.setVisible(True)
        self.title_label.setVisible(False)
        self.loading_label.setVisible(False)
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.preview_label.setVisible(True)
        self._pixmap = pil_to_pixmap(cutout_pil)  # retains original alpha
        self._is_rgba_cutout = True
        self._update_scaled_preview()
        self.preview_label.setToolTip(caption)

    def set_background_mode(self, mode: str):
        if mode == self._background_mode:
            return
        self.background_swatches.set_mode(mode)
        self._background_mode = mode
        self._update_scaled_preview()
        self.backgroundChanged.emit(mode)

    def _update_scaled_preview(self):
        if self._pixmap is None:
            return
        max_w = max(100, self.width() - 60)
        max_h = max(100, self.height() - 90)
        scaled = self._pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if not self._is_rgba_cutout:
            self.preview_label.setPixmap(scaled)
            return

        # Compose at screen resolution to keep buttons responsive on big images.
        # The original RGBA pixmap is never modified; saving uses prediction['cutout_pil'].
        display = QPixmap(scaled.size())
        display.fill(Qt.transparent)
        painter = QPainter(display)
        rect = display.rect()
        if self._background_mode == "transparent":
            paint_checker(painter, rect)
        else:
            painter.fillRect(rect, QColor(*BACKGROUND_COLORS[self._background_mode]))
        painter.drawPixmap(0, 0, scaled)
        painter.end()
        self.preview_label.setPixmap(display)

    def resizeEvent(self, event):
        self._update_scaled_preview()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self._mode == "upload" and event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if self._mode != "loading" and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self._mode == "loading":
            return

        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())

        if paths:
            self.filesDropped.emit(paths)

        event.acceptProposedAction()


