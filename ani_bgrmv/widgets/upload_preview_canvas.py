from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from ..styles import CANCEL_BUTTON_STYLE, CANVAS_BG, MUTED_TEXT_COLOR


class UploadPreviewCanvas(QFrame):
    """Left picture area while no file is open.

    States:
    - upload: drop zone, a button to choose a file (more with add_button);
      clicking anywhere in the zone does the same as the first button.
    - loading: progress bar and, when allowed, the Stop button.
    """

    clicked = Signal()
    filesDropped = Signal(list)
    cancelRequested = Signal()

    def __init__(self, title: str, hint: str, button_text: str, cancel_text: str = "Dừng cắt"):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(360)
        self._mode = "upload"
        self._cancel_text = cancel_text
        # Only this frame (by name) gets the dashed border and its blue hover:
        # a QLabel is also a QFrame and would get them too.
        self.setObjectName("uploadArea")
        self.setStyleSheet(
            f"""
            QFrame#uploadArea {{
                border: 2px dashed #5A6474;
                border-radius: 12px;
                background: {CANVAS_BG};
            }}
            QFrame#uploadArea:hover {{
                border: 2px dashed #3D7EFF;
            }}
            """
        )

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 28px; font-weight: 800; color: white;")
        self.hint_label = QLabel(hint)
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(f"font-size: 14px; color: {MUTED_TEXT_COLOR};")
        self._buttons = QHBoxLayout()
        self._buttons.addStretch(1)
        self._buttons.addStretch(1)
        self.add_button(button_text).clicked.connect(self.clicked.emit)

        self.loading_label = QLabel("Đang xử lý...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setWordWrap(True)
        self.loading_label.setStyleSheet("font-size: 17px; font-weight: 700; color: white;")
        self.progress = QProgressBar()
        self.progress.setMinimumWidth(420)
        self.btn_cancel = QPushButton(cancel_text)
        self.btn_cancel.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.cancelRequested.emit)

        center = QVBoxLayout()
        center.setSpacing(16)
        center.addStretch(1)
        center.addWidget(self.title_label)
        center.addWidget(self.hint_label)
        center.addLayout(self._buttons)
        center.addWidget(self.loading_label)
        center.addWidget(self.progress, alignment=Qt.AlignCenter)
        center.addWidget(self.btn_cancel, alignment=Qt.AlignCenter)
        center.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.addLayout(center)
        self.set_upload_mode()

    def add_button(self, text: str) -> QPushButton:
        """One more choose-button under the hint (e.g. 'Chọn folder…')."""
        button = QPushButton(text)
        button.setMinimumWidth(140)
        self._buttons.insertWidget(self._buttons.count() - 1, button)
        return button

    def _show_upload(self, upload: bool):
        for widget in (self.title_label, self.hint_label):
            widget.setVisible(upload)
        for index in range(self._buttons.count()):
            button = self._buttons.itemAt(index).widget()
            if button is not None:
                button.setVisible(upload)
        self.loading_label.setVisible(not upload)
        self.progress.setVisible(not upload)

    def set_upload_mode(self):
        self._mode = "upload"
        self.setCursor(Qt.PointingHandCursor)
        self._show_upload(True)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)

    def set_loading_mode(
        self,
        text: str,
        current: int | None = None,
        total: int | None = None,
        allow_cancel: bool = True,
        cancel_text: str | None = None,
    ):
        """``cancel_text``: the Stop button's text for this task (default:
        the one given to the constructor)."""
        self._mode = "loading"
        self.setCursor(Qt.ArrowCursor)
        self._show_upload(False)
        self.loading_label.setText(text)
        self.btn_cancel.setText(cancel_text or self._cancel_text)
        self.btn_cancel.setVisible(allow_cancel)
        self.btn_cancel.setEnabled(allow_cancel)
        if current is None or total is None or total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(current)

    def set_cancel_waiting_mode(self, text: str = "Đang hủy thao tác..."):
        self.set_loading_mode(text, allow_cancel=False)

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
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
        event.acceptProposedAction()
