"""Layout shared by the image pages (1 hình and nhiều hình), as the annotator:
the picture on the left, the tools in a fixed-width panel on the right.

Panel, top to bottom: image name; Erase · Clear Mask · Restore; Guided ·
Manual; Paint · Box; Fit (· Matting on CPU); brush size; preview background
and its opacity; description of the current tool; Stop (while Guided runs);
Debug; Undo · Redo; then the page's own save/navigation buttons.
"""
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QStackedWidget, QVBoxLayout, QWidget)

from ..app_config import SAVE_DEBUG_OUTPUTS, SHOW_DEBUG_BUTTON
from ..styles import CANCEL_BUTTON_STYLE, MUTED_TEXT_COLOR
from ..widgets.background_swatches import BackgroundControls
from ..widgets.edit_mask_canvas import EditMaskCanvas
from ..widgets.upload_preview_canvas import UploadPreviewCanvas
from .page_layout import PAGE_MARGIN, PANEL_WIDTH, button_row, top_bar
from .single_edit_controller import SingleEditControllerMixin, new_edit_history


class ImageEditPage(SingleEditControllerMixin, QWidget):
    backRequested = Signal()

    def __init__(self, engine_getter, title: str, upload_canvas: UploadPreviewCanvas, output_dir: str):
        super().__init__()
        self.engine_getter = engine_getter
        self.output_dir = output_dir
        self.current_prediction: dict | None = None
        self.in_edit_mode = False
        self.edit_operation = "erase"
        self.edit_submode = "guided"
        self.edit_draw_mode = "paint"
        self.edit_stroke_active = False
        self.edit_dirty = False
        self.edit_history = new_edit_history()
        self.edit_thread = None
        self.edit_worker = None
        self.edit_job_kind = "guided"
        self.guided_ignore_result = False
        self.guided_running = False
        self._manual_last_point = None
        self.guided_timeout_timer = QTimer(self)
        self.guided_timeout_timer.setSingleShot(True)
        self.guided_timeout_timer.timeout.connect(self.on_guided_edit_timeout)

        top, self.btn_back, self.status_label = top_bar(title)

        # Left: the upload/progress area until an image is open, then the editor.
        self.canvas = upload_canvas
        self.edit_canvas = EditMaskCanvas(file_actions=True)
        self.view = QStackedWidget()
        self.view.addWidget(self.canvas)
        self.view.addWidget(self.edit_canvas)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self.view, 1)
        body_layout.addWidget(self._build_panel())
        # The batch page adds its overview grid as a second page here.
        self.content = QStackedWidget()
        self.content.addWidget(body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*(PAGE_MARGIN,) * 4)
        layout.addLayout(top)
        layout.addWidget(self.content, 1)

        self.btn_back.clicked.connect(self.handle_back_clicked)
        self.btn_erase.clicked.connect(lambda: self.set_edit_operation("erase"))
        self.btn_restore.clicked.connect(lambda: self.set_edit_operation("restore"))
        self.btn_clear_mask.clicked.connect(self.clear_edit_mask)
        self.btn_guided.clicked.connect(lambda: self.set_edit_submode("guided"))
        self.btn_manual.clicked.connect(lambda: self.set_edit_submode("manual"))
        self.btn_paint_mode.clicked.connect(lambda: self.set_edit_draw_mode("paint"))
        self.btn_box_mode.clicked.connect(lambda: self.set_edit_draw_mode("box"))
        self.btn_fit.clicked.connect(self.edit_canvas.reset_view)
        self.btn_matting.clicked.connect(self.start_matting)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        self.background_controls.changed.connect(self.edit_canvas.set_background)
        self.btn_stop_guided.clicked.connect(self.cancel_guided_edit)
        self.btn_debug.toggled.connect(self.on_debug_toggled)
        self.btn_undo.clicked.connect(self.undo_edit)
        self.btn_redo.clicked.connect(self.redo_edit)
        self.btn_config.clicked.connect(self.choose_output_dir)
        self.btn_open_output.clicked.connect(lambda: self.open_folder(self.output_dir))
        self.edit_canvas.strokeStarted.connect(self.on_edit_stroke_started)
        self.edit_canvas.strokePoint.connect(self.on_edit_stroke_point)
        self.edit_canvas.strokeFinished.connect(self.on_edit_stroke_finished)
        self.edit_canvas.boxFinished.connect(self.on_edit_box_finished)
        for keys, handler in (("Ctrl+Z", self.undo_edit), ("Ctrl+Y", self.redo_edit)):
            QShortcut(QKeySequence(keys), self).activated.connect(handler)

        self.on_debug_toggled(self.btn_debug.isChecked())
        self.on_brush_size_changed(self.brush_slider.value())
        self.set_tools_enabled(False)
        self.update_description()

    # -- layout -----------------------------------------------------------------
    def _build_panel(self) -> QWidget:
        panel = QVBoxLayout()
        self.name_label = QLabel("Chưa có ảnh")
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("font-weight: 700;")
        panel.addWidget(self.name_label)

        (self.btn_erase, self.btn_clear_mask, self.btn_restore), row = button_row(
            "Erase", "Clear Mask", "Restore", checkable=True)
        self.btn_clear_mask.setCheckable(False)
        self.btn_clear_mask.setToolTip("Xóa toàn bộ alpha mask (có thể Undo).")
        panel.addLayout(row)
        (self.btn_guided, self.btn_manual), row = button_row("Guided", "Manual", checkable=True)
        panel.addLayout(row)
        (self.btn_paint_mode, self.btn_box_mode), row = button_row("Paint mode", "Box mode", checkable=True)
        panel.addLayout(row)
        self.btn_erase.setChecked(True)
        self.btn_guided.setChecked(True)
        self.btn_paint_mode.setChecked(True)
        self.btn_fit = QPushButton("Fit")
        self.btn_matting = QPushButton("Matting")
        self.btn_matting.setToolTip("CPU: chạy ViTMatte một lần trên toàn bộ mask hiện tại sau khi chỉnh Paint/Box.")
        self.btn_matting.hide()  # shown for CPU editing only
        row = QHBoxLayout()
        row.addWidget(self.btn_fit)
        row.addWidget(self.btn_matting)
        panel.addLayout(row)

        self.brush_label = QLabel()
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(4, 120)
        self.brush_slider.setValue(28)
        panel.addWidget(self.brush_label)
        panel.addWidget(self.brush_slider)
        self.background_controls = BackgroundControls()
        panel.addWidget(self.background_controls)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(f"color: {MUTED_TEXT_COLOR};")
        panel.addWidget(self.description_label)
        panel.addStretch(1)

        self.btn_stop_guided = QPushButton("Dừng")
        self.btn_stop_guided.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.btn_stop_guided.hide()
        panel.addWidget(self.btn_stop_guided)
        self.btn_debug = QPushButton()
        self.btn_debug.setCheckable(True)
        # Release builds hide Debug (output.show_debug_button = false).
        self.btn_debug.setChecked(SAVE_DEBUG_OUTPUTS and SHOW_DEBUG_BUTTON)
        self.btn_debug.setVisible(SHOW_DEBUG_BUTTON)
        panel.addWidget(self.btn_debug)
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        row = QHBoxLayout()
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_redo)
        panel.addLayout(row)
        self.btn_config = QPushButton("Đường dẫn lưu")
        self.btn_open_output = QPushButton("Mở folder đã lưu")
        panel.addLayout(self._build_file_buttons())
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet(f"color: {MUTED_TEXT_COLOR}; font-size: 11px;")
        panel.addWidget(self.path_label)
        self._show_output_dir()

        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(PANEL_WIDTH)
        return side

    def _build_file_buttons(self) -> QGridLayout:
        """The page's save (and navigation) buttons, under Undo/Redo."""
        raise NotImplementedError

    def _sync_file_buttons(self, enabled: bool):
        """Enable the page's save/navigation buttons for an open, idle image."""
        raise NotImplementedError

    def showEvent(self, event):
        self.sync_engine_controls()
        super().showEvent(event)

    # -- shared actions ----------------------------------------------------------
    def on_debug_toggled(self, enabled: bool):
        # A hidden button must never enable debug saving (release safety).
        enabled = bool(enabled) and SHOW_DEBUG_BUTTON
        if self.btn_debug.isChecked() != enabled:
            self.btn_debug.blockSignals(True)
            self.btn_debug.setChecked(enabled)
            self.btn_debug.blockSignals(False)
        self.btn_debug.setText("Debug: ON" if enabled else "Debug: OFF")

    def debug_enabled(self) -> bool:
        return SHOW_DEBUG_BUTTON and self.btn_debug.isChecked()

    def handle_back_clicked(self):
        raise NotImplementedError

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn folder lưu ảnh", self.output_dir)
        if folder:
            self.output_dir = folder
            self._show_output_dir()
            self.status_label.setText("Đã đổi đường dẫn lưu.")

    def _show_output_dir(self):
        self.path_label.setText(f"Lưu vào: {self.output_dir}")
