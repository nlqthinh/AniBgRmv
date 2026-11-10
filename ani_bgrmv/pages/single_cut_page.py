from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QFileDialog, QGridLayout, QMessageBox, QPushButton

from ..app_config import SINGLE_OUTPUT_DIR
from ..utils import collect_image_files_from_paths
from ..qt_threads import thread_is_valid
from ..widgets.upload_preview_canvas import UploadPreviewCanvas
from ..workers import SingleCutWorker
from .image_edit_page import ImageEditPage


class SingleCutPage(ImageEditPage):
    """One image: upload → automatic cut → the editor opens at once.

    Dropping another image (or ✕) closes this one; unsaved edits are asked
    about first. Going Home keeps the image open for when you come back.
    """

    def __init__(self, engine_getter):
        canvas = UploadPreviewCanvas("Upload ảnh", "Kéo-thả ảnh vào đây hoặc bấm nút để chọn ảnh.",
                                     "Chọn ảnh…")
        super().__init__(engine_getter, "Cắt nền 1 hình", canvas, SINGLE_OUTPUT_DIR)
        self.current_thread: QThread | None = None
        self.current_worker: SingleCutWorker | None = None
        self.cancel_requested = False
        self.status_label.setText("Upload hoặc kéo-thả ảnh vào khung bên trái.")

        self.canvas.clicked.connect(self.select_image)
        self.canvas.filesDropped.connect(self.handle_drop)
        self.canvas.cancelRequested.connect(self.cancel_current_cut)
        self.edit_canvas.filesDropped.connect(self.handle_drop)
        self.edit_canvas.closeRequested.connect(self.close_image)
        self.btn_save.clicked.connect(self.save_current)

    def _build_file_buttons(self) -> QGridLayout:
        self.btn_save = QPushButton("Lưu ảnh")
        grid = QGridLayout()
        grid.addWidget(self.btn_save, 0, 0)
        grid.addWidget(self.btn_config, 0, 1)
        grid.addWidget(self.btn_open_output, 1, 0, 1, 2)
        return grid

    def _sync_file_buttons(self, enabled: bool):
        self.btn_save.setEnabled(enabled)

    def set_busy_buttons(self, busy: bool):
        """While cutting: no Home, no saving, no output folder change."""
        self.btn_back.setEnabled(not busy)
        self.btn_config.setEnabled(not busy)
        self.btn_open_output.setEnabled(not busy)

    def handle_back_clicked(self):
        if self.editor_busy():
            return
        if self.in_edit_mode:
            # As leaving Edit did: free HQ-SAM. The image stays open.
            engine = self.engine_getter()
            if engine is not None:
                engine.release_guided_resources()
        self.backRequested.emit()

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

    def clear_current_preview(self):
        """Close the image without asking (MainWindow: the models changed)."""
        if self.in_edit_mode:
            self.close_editor()
        self.current_prediction = None
        self.name_label.setText("Chưa có ảnh")

    def close_image(self):
        """✕ on the picture."""
        if self.editor_busy() or not self.confirm_discard_unsaved():
            return
        self.clear_current_preview()
        self.status_label.setText("Đã đóng ảnh. Bạn có thể upload ảnh khác.")

    def start_cut(self, image_path: str):
        # A prior worker may have emitted its result but not finished quitting.
        if thread_is_valid(self.current_thread):
            return
        self.current_thread = self.current_worker = None
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.critical(self, "Model chưa sẵn sàng", "Pipeline chưa load xong.")
            return
        if self.in_edit_mode:
            if self.editor_busy() or not self.confirm_discard_unsaved():
                return
            self.close_editor()

        self.cancel_requested = False
        self.current_prediction = None
        self.name_label.setText(Path(image_path).name)
        self.set_busy_buttons(True)
        self.canvas.set_loading_mode("Đang cắt ảnh bằng UNet → ViTMatte-UNet...")
        self.status_label.setText(f"Đang xử lý: {Path(image_path).name}")

        self.current_thread = QThread()
        self.current_worker = SingleCutWorker(engine, image_path, lightweight=not self.debug_enabled())
        self.current_worker.moveToThread(self.current_thread)

        self.current_thread.started.connect(self.current_worker.run)
        self.current_worker.finished.connect(self.on_cut_finished)
        self.current_worker.canceled.connect(self.on_cut_canceled)
        self.current_worker.error.connect(self.on_worker_error)

        self.current_worker.finished.connect(self.current_thread.quit)
        self.current_worker.canceled.connect(self.current_thread.quit)
        self.current_worker.error.connect(self.current_thread.quit)
        self.current_thread.finished.connect(self.current_worker.deleteLater)
        self.current_thread.finished.connect(self._on_cut_thread_finished)
        self.current_thread.finished.connect(self.current_thread.deleteLater)

        self.current_thread.start()

    def _on_cut_thread_finished(self):
        finished = self.sender()
        if finished is not self.current_thread:
            return
        if thread_is_valid(finished):
            finished.wait()
        self.current_thread = None
        self.current_worker = None
        self.set_busy_buttons(False)

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
        self.enter_edit_mode()
        self.status_label.setText(
            f"Cắt xong. Chưa lưu. Time: {prediction['elapsed_sec']:.2f}s | "
            f"Ảnh: {Path(prediction['input_path']).name}"
        )

    def on_cut_canceled(self):
        self.cancel_requested = False
        self.current_prediction = None
        self.set_busy_buttons(False)
        self.canvas.set_upload_mode()
        self.name_label.setText("Chưa có ảnh")
        self.status_label.setText("Đã dừng cắt. Bạn có thể upload ảnh khác.")

    def on_worker_error(self, err: str):
        self.cancel_requested = False
        self.canvas.set_upload_mode()
        self.current_prediction = None
        self.set_busy_buttons(False)
        self.name_label.setText("Chưa có ảnh")
        QMessageBox.critical(self, "Lỗi", err)
        self.status_label.setText("Có lỗi xảy ra khi cắt ảnh.")

    def save_current(self):
        engine = self.engine_getter()
        if not self.current_prediction or engine is None:
            return
        saved = engine.save_prediction(self.current_prediction, self.output_dir,
                                       save_debug=self.debug_enabled())
        self.edit_dirty = False
        self.status_label.setText(f"Đã lưu: {saved['cutout_path']}")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu ảnh đã cắt:\n{saved['cutout_path']}")
