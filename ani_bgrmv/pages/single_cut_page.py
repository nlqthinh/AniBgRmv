from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QHBoxLayout,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..app_config import SAVE_DEBUG_OUTPUTS, SHOW_DEBUG_BUTTON, SINGLE_OUTPUT_DIR
from ..styles import (
    CANCEL_BUTTON_STYLE,
    DEBUG_BUTTON_STYLE,
    EDIT_BLACK_BUTTON_STYLE,
    EDIT_ERASE_BUTTON_STYLE,
    EDIT_RESTORE_BUTTON_STYLE,
    apply_button_style,
)
from ..utils import center_button_row, collect_image_files_from_paths
from ..widgets.edit_mask_canvas import EditMaskCanvas
from ..widgets.upload_preview_canvas import UploadPreviewCanvas
from ..workers import GuidedEditWorker, SingleCutWorker
from .single_edit_controller import SingleEditControllerMixin

class SingleCutPage(SingleEditControllerMixin, QWidget):
    backRequested = Signal()

    def __init__(self, engine_getter):
        super().__init__()
        self.engine_getter = engine_getter

        self.current_prediction: dict | None = None
        self.current_thread: QThread | None = None
        self.current_worker: SingleCutWorker | None = None
        self.cancel_requested = False
        self.output_dir = SINGLE_OUTPUT_DIR

        self.in_edit_mode = False
        self.edit_start_prediction: dict | None = None
        self.edit_operation = "erase"
        self.edit_submode = "guided"
        self.edit_stroke_active = False
        self.edit_dirty = False
        self.edit_undo_stack: list[np.ndarray] = []
        self.edit_redo_stack: list[np.ndarray] = []
        self.edit_thread: QThread | None = None
        self.edit_worker: GuidedEditWorker | None = None
        self.guided_ignore_result = False
        self.guided_running = False
        self.guided_timeout_timer = QTimer(self)
        self.guided_timeout_timer.setSingleShot(True)
        self.guided_timeout_timer.timeout.connect(self.on_guided_edit_timeout)

        self.btn_back = QPushButton("← Trang chủ")
        apply_button_style(self.btn_back)

        self.title = QLabel("Cắt nền 1 hình")
        self.title.setStyleSheet("font-size: 22px; font-weight: 700; color: #FFFFFF;")

        self.status_label = QLabel("Upload hoặc kéo-thả ảnh vào khung chính.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #FFFFFF;")

        self.instruction_label = QLabel("Kéo-thả ảnh vào đây hoặc bấm vào khung để chọn ảnh.")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 14px; color: #FFFFFF;")

        self.path_label = QLabel(f"Đường dẫn lưu: {self.output_dir}")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size: 12px; color: #F0F0F0;")

        top = QHBoxLayout()
        top.addWidget(self.btn_back)
        top.addStretch(1)
        top.addWidget(self.title)
        top.addStretch(2)

        self.canvas = UploadPreviewCanvas("Upload ảnh")
        self.edit_canvas = EditMaskCanvas()
        self.edit_canvas.setVisible(False)

        # Edit top bar: Undo/Redo left, Guided/Manual right.
        self.edit_top_widget = QWidget()
        edit_top = QHBoxLayout()
        edit_top.setContentsMargins(0, 0, 0, 0)
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_guided = QPushButton("Guided")
        self.btn_manual = QPushButton("Manual")
        for btn in [self.btn_undo, self.btn_redo, self.btn_guided, self.btn_manual]:
            btn.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
            btn.setCheckable(btn in [self.btn_guided, self.btn_manual])
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_guided.setChecked(True)
        edit_top.addWidget(self.btn_undo)
        edit_top.addWidget(self.btn_redo)
        edit_top.addStretch(1)
        edit_top.addWidget(self.btn_guided)
        edit_top.addWidget(self.btn_manual)
        self.edit_top_widget.setLayout(edit_top)
        self.edit_top_widget.setFixedHeight(40)
        self.edit_top_widget.setVisible(False)

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(4)
        self.brush_slider.setMaximum(120)
        self.brush_slider.setValue(28)
        self.brush_slider.setTickInterval(4)
        self.brush_slider.setStyleSheet(
            """
            QSlider::groove:horizontal { height: 8px; background: #DDDDDD; border-radius: 4px; }
            QSlider::handle:horizontal { width: 20px; margin: -6px 0; background: #FF3030; border-radius: 10px; }
            """
        )
        self.btn_close_brush = QPushButton("X")
        self.btn_close_brush.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
        self.btn_close_brush.setFixedWidth(48)
        self.brush_panel_widget = QWidget()
        brush_panel = QHBoxLayout()
        brush_panel.setContentsMargins(80, 0, 80, 0)
        brush_panel.setSpacing(10)
        brush_panel.addWidget(self.brush_slider, stretch=1)
        brush_panel.addWidget(self.btn_close_brush)
        self.brush_panel_widget.setLayout(brush_panel)
        self.brush_panel_widget.setFixedHeight(40)
        self.brush_panel_widget.setVisible(False)

        self.btn_save = QPushButton("Lưu ảnh")
        self.btn_debug = QPushButton()
        self.btn_debug.setCheckable(True)
        self.btn_debug.setChecked(SAVE_DEBUG_OUTPUTS if SHOW_DEBUG_BUTTON else False)
        self.btn_debug.setStyleSheet(DEBUG_BUTTON_STYLE)
        if not SHOW_DEBUG_BUTTON:
            self.btn_debug.hide()
        self.btn_config = QPushButton("Đường dẫn lưu")
        self.btn_edit = QPushButton("Edit")
        self.btn_open_output = QPushButton("Mở folder đã lưu")

        apply_button_style(self.btn_save, primary=True)
        apply_button_style(self.btn_config)
        apply_button_style(self.btn_edit)
        apply_button_style(self.btn_open_output)

        self.btn_save.setEnabled(False)
        self.btn_edit.setEnabled(False)

        self.normal_buttons_widget = QWidget()
        self.normal_buttons_widget.setLayout(center_button_row(
            self.btn_debug,
            self.btn_save,
            self.btn_edit,
            self.btn_config,
            self.btn_open_output,
        ))

        self.edit_buttons_widget = QWidget()
        self.btn_erase = QPushButton("Erase")
        self.btn_fit = QPushButton("Fit")
        self.btn_restore = QPushButton("Restore")
        self.btn_exit_edit = QPushButton("Exit")
        self.btn_brush_size = QPushButton("Brush size")
        self.btn_edit_save = QPushButton("Lưu ảnh")
        self.btn_edit_debug = QPushButton()
        self.btn_edit_debug.setCheckable(True)
        self.btn_edit_debug.setChecked(self.btn_debug.isChecked())
        self.btn_edit_debug.setStyleSheet(DEBUG_BUTTON_STYLE)
        if not SHOW_DEBUG_BUTTON:
            self.btn_edit_debug.hide()
        self.btn_stop_guided = QPushButton("Dừng")
        self.btn_stop_guided.setVisible(False)

        self.btn_erase.setCheckable(True)
        self.btn_restore.setCheckable(True)
        self.btn_erase.setChecked(True)

        self.btn_erase.setStyleSheet(EDIT_ERASE_BUTTON_STYLE)
        self.btn_restore.setStyleSheet(EDIT_RESTORE_BUTTON_STYLE)
        for btn in [self.btn_fit, self.btn_exit_edit, self.btn_brush_size, self.btn_edit_save]:
            btn.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
        self.btn_stop_guided.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.btn_edit_save.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)

        for btn, width in [(self.btn_erase, 88), (self.btn_fit, 64),
                           (self.btn_restore, 98), (self.btn_exit_edit, 84),
                           (self.btn_brush_size, 122), (self.btn_edit_save, 104),
                           (self.btn_edit_debug, 112), (self.btn_stop_guided, 84)]:
            btn.setMinimumWidth(width)
            btn.setFixedHeight(42)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.edit_buttons_widget.setFixedHeight(58)

        edit_buttons_row = center_button_row(
            self.btn_fit,
            self.btn_erase,
            self.btn_restore,
            self.btn_exit_edit,
            self.btn_brush_size,
            self.btn_edit_debug,
            self.btn_edit_save,
            self.btn_stop_guided,
        )
        edit_buttons_row.setContentsMargins(0, 4, 0, 6)
        edit_buttons_row.setSpacing(7)  # Fit added without squeezing bottom buttons
        self.edit_buttons_widget.setLayout(edit_buttons_row)
        self.edit_buttons_widget.setVisible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self.status_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.edit_top_widget)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.edit_canvas, stretch=1)
        layout.addWidget(self.brush_panel_widget)
        layout.addWidget(self.normal_buttons_widget)
        layout.addWidget(self.edit_buttons_widget)
        self.setLayout(layout)

        self.btn_back.clicked.connect(self.handle_back_clicked)
        self.canvas.clicked.connect(self.select_image)
        self.canvas.filesDropped.connect(self.handle_drop)
        self.canvas.clearRequested.connect(self.clear_current_preview)
        self.canvas.cancelRequested.connect(self.cancel_current_cut)
        # Two separate canvases share one preview-only background choice.
        self.canvas.backgroundChanged.connect(self.edit_canvas.set_background_mode)
        self.edit_canvas.backgroundChanged.connect(self.canvas.set_background_mode)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_debug.toggled.connect(self.on_debug_toggled)
        self.btn_edit_debug.toggled.connect(self.on_debug_toggled)
        self.on_debug_toggled(self.btn_debug.isChecked())
        self.btn_config.clicked.connect(self.choose_output_dir)
        self.btn_open_output.clicked.connect(lambda: self.open_folder(self.output_dir))
        self.btn_edit.clicked.connect(self.enter_edit_mode)

        self.btn_erase.clicked.connect(lambda: self.set_edit_operation("erase"))
        self.btn_fit.clicked.connect(self.edit_canvas.reset_view)
        self.btn_restore.clicked.connect(lambda: self.set_edit_operation("restore"))
        self.btn_guided.clicked.connect(lambda: self.set_edit_submode("guided"))
        self.btn_manual.clicked.connect(lambda: self.set_edit_submode("manual"))
        self.btn_exit_edit.clicked.connect(self.exit_edit_mode_confirm)
        self.btn_brush_size.clicked.connect(self.toggle_brush_slider)
        self.btn_close_brush.clicked.connect(lambda: self.set_brush_panel_visible(False))
        self.btn_edit_save.clicked.connect(self.save_current)
        self.btn_stop_guided.clicked.connect(self.cancel_guided_edit)
        self.btn_undo.clicked.connect(self.undo_edit)
        self.btn_redo.clicked.connect(self.redo_edit)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)

        self.edit_canvas.strokeStarted.connect(self.on_edit_stroke_started)
        self.edit_canvas.strokePoint.connect(self.on_edit_stroke_point)
        self.edit_canvas.strokeFinished.connect(self.on_edit_stroke_finished)

        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_edit)
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.redo_edit)

    def on_debug_toggled(self, enabled: bool):
        # Keep the normal Save row and the SAM Edit Save row synchronized.
        # Hidden buttons must never enable debug saving (release safety).
        enabled = bool(enabled) and SHOW_DEBUG_BUTTON
        for button in (self.btn_debug, self.btn_edit_debug):
            if button.isChecked() != enabled:
                button.blockSignals(True)
                button.setChecked(enabled)
                button.blockSignals(False)
            button.setText("Debug: ON" if enabled else "Debug: OFF")

    def set_busy_buttons(self, busy: bool):
        self.btn_back.setEnabled(not busy)
        self.btn_save.setEnabled(False if busy else self.current_prediction is not None)
        self.btn_edit.setEnabled(False if busy else self.current_prediction is not None)
        self.btn_config.setEnabled(not busy)
        self.btn_open_output.setEnabled(not busy)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self.start_cut(path)

    def handle_drop(self, paths: list[str]):
        image_paths = collect_image_files_from_paths(paths, recursive=True)
        if not image_paths:
            QMessageBox.warning(self, "Không có ảnh", "Không tìm thấy file ảnh hợp lệ.")
            return
        self.start_cut(image_paths[0])

    def start_cut(self, image_path: str):
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.critical(self, "Model chưa sẵn sàng", "Pipeline chưa load xong.")
            return

        self.cancel_requested = False
        self.current_prediction = None
        self.set_busy_buttons(True)
        self.canvas.set_loading_mode("Đang cắt ảnh bằng F2_A1 → ViTMatte M6...")
        self.status_label.setText(f"Đang xử lý: {Path(image_path).name}")

        self.current_thread = QThread()
        self.current_worker = SingleCutWorker(engine, image_path)
        self.current_worker.moveToThread(self.current_thread)

        self.current_thread.started.connect(self.current_worker.run)
        self.current_worker.finished.connect(self.on_cut_finished)
        self.current_worker.canceled.connect(self.on_cut_canceled)
        self.current_worker.error.connect(self.on_worker_error)

        self.current_worker.finished.connect(self.current_thread.quit)
        self.current_worker.canceled.connect(self.current_thread.quit)
        self.current_worker.error.connect(self.current_thread.quit)
        self.current_thread.finished.connect(self.current_thread.deleteLater)

        self.current_thread.start()

    def cancel_current_cut(self):
        self.cancel_requested = True
        if self.current_worker is not None:
            self.current_worker.cancel()
        self.canvas.set_cancel_waiting_mode("Đang hủy thao tác cắt...")
        self.status_label.setText("Đã yêu cầu dừng. App sẽ bỏ kết quả khi tác vụ hiện tại trả về.")

    def on_cut_finished(self, prediction: dict):
        if self.cancel_requested:
            self.on_cut_canceled()
            return

        self.current_prediction = prediction
        self.set_busy_buttons(False)

        self.canvas.set_preview_cutout(
            prediction["cutout_pil"],
            caption=Path(prediction["input_path"]).name,
        )
        self.status_label.setText(
            f"Cắt xong. Chưa lưu. Time: {prediction['elapsed_sec']:.2f}s | "
            f"Ảnh: {Path(prediction['input_path']).name}"
        )

    def on_cut_canceled(self):
        self.cancel_requested = False
        self.current_prediction = None
        self.current_worker = None
        self.set_busy_buttons(False)
        self.canvas.set_upload_mode()
        self.status_label.setText("Đã dừng cắt. Bạn có thể upload ảnh khác.")

    def on_worker_error(self, err: str):
        self.cancel_requested = False
        self.current_worker = None
        self.canvas.set_upload_mode()
        self.current_prediction = None
        self.set_busy_buttons(False)
        QMessageBox.critical(self, "Lỗi", err)
        self.status_label.setText("Có lỗi xảy ra khi cắt ảnh.")

    def clear_current_preview(self):
        if self.in_edit_mode:
            self.force_exit_edit(discard=True)
        self.current_prediction = None
        self.set_busy_buttons(False)
        self.canvas.set_upload_mode()
        self.status_label.setText("Đã xóa preview. Bạn có thể upload ảnh khác.")

    def save_current(self):
        if not self.current_prediction:
            return

        engine = self.engine_getter()
        if engine is None:
            return

        saved = engine.save_prediction(
            self.current_prediction,
            self.output_dir,
            save_debug=SHOW_DEBUG_BUTTON and self.btn_debug.isChecked(),
        )

        self.status_label.setText(f"Đã lưu: {saved['cutout_path']}")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu ảnh đã cắt:\n{saved['cutout_path']}")

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn folder lưu ảnh", self.output_dir)
        if not folder:
            return
        self.output_dir = folder
        self.path_label.setText(f"Đường dẫn lưu: {self.output_dir}")
        self.status_label.setText("Đã đổi đường dẫn lưu.")
