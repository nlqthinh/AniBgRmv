import os
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from ..app_config import BATCH_OUTPUT_DIR, MAX_BATCH_IMAGES, SAVE_DEBUG_OUTPUTS, SHOW_DEBUG_BUTTON
from ..styles import DEBUG_BUTTON_STYLE, apply_button_style
from ..utils import center_button_row, collect_image_files_from_paths
from ..widgets.upload_preview_canvas import UploadPreviewCanvas
from ..workers import BatchCutWorker, BatchSaveWorker

class BatchCutPage(QWidget):
    backRequested = Signal()

    def __init__(self, engine_getter):
        super().__init__()
        self.engine_getter = engine_getter

        self.selected_paths: list[str] = []
        self.predictions: list[dict] = []
        self.output_dir = BATCH_OUTPUT_DIR

        self.cut_thread: QThread | None = None
        self.cut_worker: BatchCutWorker | None = None
        self.save_thread: QThread | None = None
        self.save_worker: BatchSaveWorker | None = None
        self.cancel_requested = False
        self._save_debug_for_current_batch = False

        self.btn_back = QPushButton("← Trang chủ")
        apply_button_style(self.btn_back)

        self.title = QLabel("Cắt nhiều hình")
        self.title.setStyleSheet("font-size: 22px; font-weight: 700; color: #FFFFFF;")

        self.status_label = QLabel("Upload nhiều ảnh, kéo-thả ảnh/folder, hoặc chọn cắt cả folder.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #FFFFFF;")

        self.instruction_label = QLabel("Kéo-thả nhiều ảnh/folder vào đây hoặc bấm vào khung để chọn nhiều ảnh.")
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

        self.btn_save_all = QPushButton("Lưu ảnh")
        self.btn_debug = QPushButton("Debug: ON" if SAVE_DEBUG_OUTPUTS else "Debug: OFF")
        self.btn_debug.setCheckable(True)
        self.btn_debug.setChecked(SAVE_DEBUG_OUTPUTS if SHOW_DEBUG_BUTTON else False)
        self.btn_debug.setStyleSheet(DEBUG_BUTTON_STYLE)
        if not SHOW_DEBUG_BUTTON:
            self.btn_debug.hide()
        self.btn_debug.setText("Debug: ON" if self.btn_debug.isChecked() else "Debug: OFF")
        self.btn_select_folder = QPushButton("Cắt cả folder")
        self.btn_config = QPushButton("Đường dẫn lưu")
        self.btn_open_output = QPushButton("Mở folder đã lưu")

        apply_button_style(self.btn_save_all, primary=True)
        apply_button_style(self.btn_select_folder)
        apply_button_style(self.btn_config)
        apply_button_style(self.btn_open_output)

        self.btn_save_all.setEnabled(False)

        buttons = center_button_row(
            self.btn_debug,
            self.btn_save_all,
            self.btn_select_folder,
            self.btn_config,
            self.btn_open_output,
        )

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.status_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self.btn_back.clicked.connect(self.backRequested.emit)
        self.canvas.clicked.connect(self.select_images)
        self.canvas.filesDropped.connect(self.handle_drop)
        self.canvas.clearRequested.connect(self.clear_current_preview)
        self.canvas.cancelRequested.connect(self.cancel_current_batch)
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_debug.toggled.connect(
            lambda checked: self.btn_debug.setText("Debug: ON" if checked else "Debug: OFF")
        )
        self.btn_config.clicked.connect(self.choose_output_dir)
        self.btn_open_output.clicked.connect(lambda: self.open_folder(self.output_dir))

    def set_busy_buttons(self, busy: bool):
        self.btn_back.setEnabled(not busy)
        self.btn_save_all.setEnabled(False if busy else len(self.predictions) > 0)
        self.btn_select_folder.setEnabled(not busy)
        self.btn_config.setEnabled(not busy)
        self.btn_open_output.setEnabled(not busy)

    def validate_batch_paths(self, image_paths: list[str], source_label: str = "batch") -> list[str] | None:
        if not image_paths:
            QMessageBox.warning(self, "Không có ảnh", "Không tìm thấy file ảnh hợp lệ.")
            return None

        if len(image_paths) > MAX_BATCH_IMAGES:
            QMessageBox.warning(
                self,
                "Quá nhiều ảnh",
                f"{source_label} có {len(image_paths)} ảnh. "
                f"App hiện giới hạn tối đa {MAX_BATCH_IMAGES} ảnh/lần.",
            )
            return None

        return image_paths


    def select_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn nhiều ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not paths:
            return

        image_paths = self.validate_batch_paths(paths, "Bạn đã chọn")
        if image_paths is None:
            return

        self.start_batch(image_paths)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn folder ảnh")
        if not folder:
            return

        image_paths = collect_image_files_from_paths([folder], recursive=True)
        image_paths = self.validate_batch_paths(image_paths, "Folder này")
        if image_paths is None:
            return

        folder_name = Path(folder).name or folder
        confirm_box = QMessageBox(self)
        confirm_box.setWindowTitle("Xác nhận cắt folder")
        confirm_box.setIcon(QMessageBox.Question)
        confirm_box.setText(f'Xác nhận cắt folder "{folder_name}"?')
        confirm_box.setInformativeText(
            f"Tìm thấy {len(image_paths)} ảnh hợp lệ. App sẽ bắt đầu cắt sau khi bạn xác nhận."
        )
        btn_confirm = confirm_box.addButton("Xác nhận", QMessageBox.AcceptRole)
        confirm_box.addButton("Hủy", QMessageBox.RejectRole)
        confirm_box.exec()

        if confirm_box.clickedButton() is not btn_confirm:
            self.status_label.setText("Đã hủy cắt folder.")
            return

        self.predictions = []
        self.selected_paths = []
        self.btn_save_all.setEnabled(False)
        self.canvas.set_upload_mode()
        self.status_label.setText(f"Đã xác nhận folder: {folder} | {len(image_paths)} ảnh.")
        self.start_batch(image_paths)

    def handle_drop(self, paths: list[str]):
        image_paths = collect_image_files_from_paths(paths, recursive=True)
        image_paths = self.validate_batch_paths(image_paths, "Dữ liệu kéo-thả")
        if image_paths is None:
            return

        self.start_batch(image_paths)

    def start_batch(self, image_paths: list[str]):
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.critical(self, "Model chưa sẵn sàng", "Pipeline chưa load xong.")
            return

        self.cancel_requested = False
        self.selected_paths = image_paths
        self.predictions = []
        self.set_busy_buttons(True)
        self.canvas.set_loading_mode(f"Đang cắt {len(image_paths)} ảnh...", 0, len(image_paths))
        self.status_label.setText(f"Đang tự động cắt {len(image_paths)} ảnh...")

        self.cut_thread = QThread()
        self.cut_worker = BatchCutWorker(engine, image_paths)
        self.cut_worker.moveToThread(self.cut_thread)

        self.cut_thread.started.connect(self.cut_worker.run)
        self.cut_worker.progress.connect(self.on_cut_progress)
        self.cut_worker.finished.connect(self.on_batch_cut_finished)
        self.cut_worker.canceled.connect(self.on_batch_cut_canceled)
        self.cut_worker.error.connect(self.on_worker_error)

        self.cut_worker.finished.connect(self.cut_thread.quit)
        self.cut_worker.canceled.connect(self.cut_thread.quit)
        self.cut_worker.error.connect(self.cut_thread.quit)
        self.cut_thread.finished.connect(self.cut_thread.deleteLater)

        self.cut_thread.start()

    def cancel_current_batch(self):
        self.cancel_requested = True
        if self.cut_worker is not None:
            self.cut_worker.cancel()
        self.canvas.set_cancel_waiting_mode("Đang hủy batch...")
        self.status_label.setText("Đã yêu cầu dừng batch. App sẽ dừng sau ảnh đang xử lý.")

    def on_cut_progress(self, current: int, total: int, message: str):
        self.canvas.set_loading_mode(message, current, total)
        self.status_label.setText(f"{message} ({current}/{total})")

    def on_batch_cut_finished(self, predictions: list[dict]):
        if self.cancel_requested:
            self.on_batch_cut_canceled()
            return

        self.predictions = predictions
        self.set_busy_buttons(False)

        if predictions:
            first = predictions[0]
            self.canvas.set_preview_cutout(
                first["cutout_pil"],
                caption=f"Preview ảnh đầu tiên: {Path(first['input_path']).name}",
            )
        else:
            self.canvas.set_upload_mode()

        self.status_label.setText(
            f"Cắt xong {len(predictions)} ảnh. Preview đang hiển thị ảnh đầu tiên. Chưa lưu."
        )

    def on_batch_cut_canceled(self):
        self.cancel_requested = False
        self.cut_worker = None
        self.selected_paths = []
        self.predictions = []
        self.set_busy_buttons(False)
        self.canvas.set_upload_mode()
        self.status_label.setText("Đã dừng batch. Bạn có thể upload lại.")

    def on_worker_error(self, err: str):
        self.btn_debug.setEnabled(True)
        self.cancel_requested = False
        self.cut_worker = None
        self.canvas.set_upload_mode()
        self.selected_paths = []
        self.predictions = []
        self.set_busy_buttons(False)
        QMessageBox.critical(self, "Lỗi", err)
        self.status_label.setText("Có lỗi xảy ra.")

    def clear_current_preview(self):
        self.selected_paths = []
        self.predictions = []
        self.set_busy_buttons(False)
        self.canvas.set_upload_mode()
        self.status_label.setText("Đã xóa preview batch. Bạn có thể upload lại.")

    def save_all(self):
        if not self.predictions:
            return

        engine = self.engine_getter()
        if engine is None:
            return

        self.btn_save_all.setEnabled(False)
        self.btn_debug.setEnabled(False)
        self.btn_select_folder.setEnabled(False)
        self.btn_config.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.btn_back.setEnabled(False)
        self._save_debug_for_current_batch = SHOW_DEBUG_BUTTON and self.btn_debug.isChecked()

        self.canvas.set_loading_mode(
            f"Đang lưu {len(self.predictions)} ảnh...",
            0,
            len(self.predictions),
            allow_cancel=False,
        )
        self.status_label.setText("Đang lưu tất cả ảnh đã cắt...")

        self.save_thread = QThread()
        self.save_worker = BatchSaveWorker(
            engine,
            self.predictions,
            self.output_dir,
            save_debug=self._save_debug_for_current_batch,
        )
        self.save_worker.moveToThread(self.save_thread)

        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.progress.connect(self.on_save_progress)
        self.save_worker.finished.connect(self.on_save_finished)
        self.save_worker.error.connect(self.on_worker_error)

        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_worker.error.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.save_thread.deleteLater)

        self.save_thread.start()

    def on_save_progress(self, current: int, total: int, message: str):
        self.canvas.set_loading_mode(message, current, total, allow_cancel=False)
        self.status_label.setText(f"{message} ({current}/{total})")

    def on_save_finished(self, saved_path: str):
        self.btn_debug.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        self.btn_config.setEnabled(True)
        self.btn_open_output.setEnabled(True)
        self.btn_back.setEnabled(True)
        self.btn_save_all.setEnabled(True)

        if self.predictions:
            first = self.predictions[0]
            self.canvas.set_preview_cutout(
                first["cutout_pil"],
                caption=f"Preview ảnh đầu tiên: {Path(first['input_path']).name}",
            )

        if self._save_debug_for_current_batch:
            self.status_label.setText(f"Đã lưu batch. Log: {saved_path}")
            QMessageBox.information(self, "Batch đã lưu", f"Đã lưu batch.\nLog:\n{saved_path}")
        else:
            self.status_label.setText(f"Đã lưu ảnh PNG tại: {saved_path}")
            QMessageBox.information(self, "Batch đã lưu", f"Đã lưu ảnh PNG trong suốt tại:\n{saved_path}")

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn folder lưu ảnh", self.output_dir)
        if not folder:
            return
        self.output_dir = folder
        self.path_label.setText(f"Đường dẫn lưu: {self.output_dir}")
        self.status_label.setText("Đã đổi đường dẫn lưu.")

    def open_folder(self, folder: str):
        p = Path(folder).resolve()
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(p)
