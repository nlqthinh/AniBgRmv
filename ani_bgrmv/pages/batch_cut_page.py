from pathlib import Path

from PySide6.QtCore import QSize, QThread, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListView, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from ..app_config import BATCH_OUTPUT_DIR, MAX_BATCH_IMAGES
from ..disk_cache import new_batch_cache
from ..engine.save_prediction import THUMBNAIL_SIZE, SavePredictionMixin
from ..qt_threads import thread_is_valid
from ..utils import collect_image_files_from_paths
from ..widgets.upload_preview_canvas import UploadPreviewCanvas
from ..workers import BatchCutWorker, BatchSaveWorker, save_staged
from .image_edit_page import ImageEditPage

SAVE_STOP_TEXT = "Dừng lưu"
# State of each image of the batch, shown in the overview (as the annotator grid).
CUT, EDITED, SAVED = "cut", "edited", "saved"
STATE_TEXT = {CUT: "chưa sửa", EDITED: "đã sửa, chưa lưu", SAVED: "đã lưu"}
STATE_COLOR = {CUT: QColor(40, 40, 40), EDITED: QColor(35, 60, 90), SAVED: QColor(30, 80, 45)}


class BatchCutPage(ImageEditPage):
    """Many images: cut them all (progress bar), then edit them one by one.

    Every result waits in a temp folder (disk_cache), not in RAM. Leaving an
    image (◀/▶, Tổng quan, Lưu) writes its edits there; Lưu ảnh saves the
    open image, Lưu tất cả every image. Home, ✕ or a new batch drop the batch
    and its temp files, asking first when an edited image is not saved.
    """

    UNSAVED_MESSAGE = "Có ảnh đã chỉnh sửa nhưng chưa lưu. Bỏ các chỉnh sửa đó?"

    def __init__(self, engine_getter):
        # Set before the base __init__: it already syncs the navigation buttons.
        self.predictions: list[dict] = []   # staged results (stage_prediction)
        self.states: list[str] = []
        self.index = -1                     # image open in the editor
        canvas = UploadPreviewCanvas("Upload nhiều ảnh",
                                     "Kéo-thả nhiều ảnh hoặc folder vào đây, hoặc bấm nút để chọn.",
                                     "Chọn ảnh…")
        folder_button = canvas.add_button("Chọn folder…")
        super().__init__(engine_getter, "Cắt nhiều hình", canvas, BATCH_OUTPUT_DIR)
        self.btn_select_folder = folder_button
        self.cut_thread: QThread | None = None
        self.cut_worker: BatchCutWorker | None = None
        self.save_thread: QThread | None = None
        self.save_worker: BatchSaveWorker | None = None
        self.cancel_requested = False
        self._save_debug_for_current_batch = False
        self._batch_temp_cache = None
        self.content.addWidget(self._build_overview())
        self.edit_canvas.btn_close.setToolTip("Đóng tất cả ảnh của lượt này")
        self.status_label.setText("Upload nhiều ảnh, kéo-thả ảnh/folder, hoặc chọn cả folder.")

        self.canvas.clicked.connect(self.select_images)
        self.canvas.filesDropped.connect(self.handle_drop)
        self.canvas.cancelRequested.connect(self.stop_task)
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.edit_canvas.filesDropped.connect(self.handle_drop)
        self.edit_canvas.closeRequested.connect(self.close_batch)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_overview.clicked.connect(self.show_overview)

    # -- layout -----------------------------------------------------------------
    def _build_file_buttons(self) -> QGridLayout:
        self.btn_save = QPushButton("Lưu ảnh")
        self.btn_save_all = QPushButton("Lưu tất cả")
        self.btn_prev = QPushButton("◀ Ảnh trước")
        self.btn_next = QPushButton("Ảnh sau ▶")
        self.btn_overview = QPushButton("Tổng quan")
        grid = QGridLayout()
        for index, button in enumerate((self.btn_save, self.btn_save_all, self.btn_config,
                                        self.btn_open_output, self.btn_prev, self.btn_next)):
            grid.addWidget(button, index // 2, index % 2)
        grid.addWidget(self.btn_overview, 3, 0, 1, 2)
        return grid

    def _build_overview(self) -> QWidget:
        """Every image of the batch; click one to edit it."""
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.grid.setGridSize(QSize(THUMBNAIL_SIZE + 24, THUMBNAIL_SIZE + 44))
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setWordWrap(True)
        self.grid.itemClicked.connect(lambda item: self.open_from_overview(item.data(Qt.ItemDataRole.UserRole)))
        self.btn_overview_back = QPushButton("← Quay lại ảnh đang sửa")
        self.btn_overview_back.clicked.connect(lambda: self.content.setCurrentIndex(0))
        self.overview_info = QLabel()
        bar = QHBoxLayout()
        bar.addWidget(self.btn_overview_back)
        bar.addWidget(self.overview_info)
        bar.addStretch(1)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.grid, 1)
        return page

    def _sync_file_buttons(self, enabled: bool):
        enabled = enabled and bool(self.predictions)
        for button in (self.btn_save, self.btn_save_all, self.btn_overview):
            button.setEnabled(enabled)
        self.btn_prev.setEnabled(enabled and self.index > 0)
        self.btn_next.setEnabled(enabled and self.index < len(self.predictions) - 1)

    def set_busy_buttons(self, busy: bool):
        """While cutting or saving everything."""
        for button in (self.btn_back, self.btn_config, self.btn_open_output):
            button.setEnabled(not busy)
        # Debug saving needs images cut with Debug ON.
        self.btn_debug.setEnabled(not busy and self._has_debug_data())

    def _show_item_name(self):
        name = Path(self.predictions[self.index]["input_path"]).name
        self.name_label.setText(f"{self.index + 1}/{len(self.predictions)} · {name}")

    # -- choosing images ------------------------------------------------------------
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
        if image_paths is not None:
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
        self.status_label.setText(f"Đã xác nhận folder: {folder} | {len(image_paths)} ảnh.")
        self.start_batch(image_paths)

    def handle_drop(self, paths: list[str]):
        image_paths = collect_image_files_from_paths(paths, recursive=True)
        image_paths = self.validate_batch_paths(image_paths, "Dữ liệu kéo-thả")
        if image_paths is not None:
            self.start_batch(image_paths)

    # -- the batch ------------------------------------------------------------------
    def has_unsaved_edits(self) -> bool:
        return super().has_unsaved_edits() or EDITED in self.states

    def _has_debug_data(self) -> bool:
        """True when the staged batch was cut with Debug ON (debug .npz on disk)."""
        return all('cached_debug_path' in p for p in self.predictions)

    def _clear_batch_cache(self):
        cache = self._batch_temp_cache
        self._batch_temp_cache = None
        if cache is not None:
            cache.cleanup()

    def clear_current_preview(self):
        """Drop the batch without asking (MainWindow: the models changed)."""
        if self.in_edit_mode:
            self.close_editor()
        self._clear_batch_cache()
        self.predictions, self.states, self.index = [], [], -1
        self.current_prediction = None
        self.content.setCurrentIndex(0)
        self.name_label.setText("Chưa có ảnh")
        self.canvas.set_upload_mode()
        self.set_busy_buttons(False)
        self._sync_file_buttons(False)

    def close_batch(self) -> bool:
        """✕, Home or a new batch. False when the user keeps the batch."""
        if not self.predictions:
            return True
        if self.editor_busy() or not self.confirm_discard_unsaved():
            return False
        self.clear_current_preview()
        self.status_label.setText("Đã đóng lượt ảnh. Bạn có thể upload lại.")
        return True

    def handle_back_clicked(self):
        # Returning Home should not leave batch PNGs occupying disk/RAM.
        if self.close_batch():
            self.backRequested.emit()

    def start_batch(self, image_paths: list[str]):
        if thread_is_valid(self.cut_thread):
            return  # Previous batch is still finishing its QThread cleanup.
        self.cut_thread = self.cut_worker = None
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.critical(self, "Model chưa sẵn sàng", "Pipeline chưa load xong.")
            return
        if not self.close_batch():
            return

        self.cancel_requested = False
        self.set_busy_buttons(True)
        self.canvas.set_loading_mode(f"Đang cắt {len(image_paths)} ảnh...", 0, len(image_paths))
        self.status_label.setText(f"Đang tự động cắt {len(image_paths)} ảnh...")

        self.cut_thread = QThread()
        # Results wait for Save on disk, not in RAM. Debug ON also stages the
        # model intermediates needed for the 99_debug output folders.
        self._batch_temp_cache = new_batch_cache()
        self.cut_worker = BatchCutWorker(engine, image_paths, self._batch_temp_cache.name,
                                         keep_debug=self.debug_enabled())
        self.cut_worker.moveToThread(self.cut_thread)

        self.cut_thread.started.connect(self.cut_worker.run)
        self.cut_worker.progress.connect(self.on_cut_progress)
        self.cut_worker.finished.connect(self.on_batch_cut_finished)
        self.cut_worker.canceled.connect(self.on_batch_cut_canceled)
        self.cut_worker.error.connect(self.on_worker_error)

        self.cut_worker.finished.connect(self.cut_thread.quit)
        self.cut_worker.canceled.connect(self.cut_thread.quit)
        self.cut_worker.error.connect(self.cut_thread.quit)
        self.cut_thread.finished.connect(self.cut_worker.deleteLater)
        self.cut_thread.finished.connect(self._on_cut_thread_finished)
        self.cut_thread.finished.connect(self.cut_thread.deleteLater)

        self.cut_thread.start()

    def _on_cut_thread_finished(self):
        finished = self.sender()
        if finished is not self.cut_thread:
            return
        if thread_is_valid(finished):
            finished.wait()
        self.cut_thread = None
        self.cut_worker = None

    def stop_task(self):
        """The Stop button of the left area: cutting or Lưu tất cả."""
        if self.save_worker is not None:
            self.save_worker.cancel()
            self.canvas.set_cancel_waiting_mode("Đang dừng lưu...")
            self.status_label.setText("Đã yêu cầu dừng lưu. App sẽ dừng sau ảnh đang lưu.")
        else:
            self.cancel_current_batch()

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
        self.states = [CUT] * len(predictions)
        self.set_busy_buttons(False)
        if not predictions:
            self.clear_current_preview()
            return
        self.open_item(0)
        self.status_label.setText(f"Cắt xong {len(predictions)} ảnh. Chưa lưu.")

    def on_batch_cut_canceled(self):
        self.cancel_requested = False
        self.clear_current_preview()
        self.status_label.setText("Đã dừng batch. Bạn có thể upload lại.")

    def on_worker_error(self, err: str):
        self.cancel_requested = False
        self.clear_current_preview()
        QMessageBox.critical(self, "Lỗi", err)
        self.status_label.setText("Có lỗi xảy ra.")

    # -- one image of the batch ---------------------------------------------------------
    def store_current(self):
        """Write the open image's edits into the batch's temp folder, so the
        overview, Lưu tất cả and coming back to this image all see them."""
        if not (self.in_edit_mode and self.edit_dirty):
            return
        engine = self.engine_getter()
        self.predictions[self.index] = engine.stage_prediction(
            self.current_prediction, self._batch_temp_cache.name, self.index + 1, self._has_debug_data())
        self.states[self.index] = EDITED
        self.edit_dirty = False

    def open_item(self, index: int):
        self.store_current()
        self.index = index
        self.current_prediction = SavePredictionMixin.load_staged_prediction(self.predictions[index])
        self.enter_edit_mode()
        self._show_item_name()

    def step(self, offset: int):
        index = self.index + offset
        if 0 <= index < len(self.predictions) and not self.editor_busy():
            self.open_item(index)

    def show_overview(self):
        if self.editor_busy():
            return
        self.store_current()
        self.grid.clear()
        for index, staged in enumerate(self.predictions):
            # Thumbnails are rewritten after an edit: read the file, no QPixmap cache.
            thumb = QPixmap()
            thumb.loadFromData(Path(staged["cached_thumb_path"]).read_bytes())
            name = Path(staged["input_path"]).name
            entry = QListWidgetItem(QIcon(thumb), f"{index + 1:03d} {STATE_TEXT[self.states[index]]}\n{name}")
            entry.setData(Qt.ItemDataRole.UserRole, index)
            entry.setBackground(STATE_COLOR[self.states[index]])
            self.grid.addItem(entry)
        self.grid.setCurrentRow(self.index)
        self.overview_info.setText(
            f"{len(self.states)} ảnh · đã lưu {self.states.count(SAVED)} · "
            f"đã sửa chưa lưu {self.states.count(EDITED)} · chưa sửa {self.states.count(CUT)}")
        self.content.setCurrentIndex(1)

    def open_from_overview(self, index: int):
        self.content.setCurrentIndex(0)
        if index != self.index:
            self.open_item(index)

    # -- saving ---------------------------------------------------------------------
    def _save_debug(self) -> bool:
        return self.debug_enabled() and self._has_debug_data()

    def save_current(self):
        engine = self.engine_getter()
        if not self.in_edit_mode or engine is None or self.editor_busy():
            return
        self.store_current()
        try:
            saved = save_staged(engine, self.predictions[self.index], self.output_dir, self._save_debug())
        except (OSError, RuntimeError) as exc:
            QMessageBox.critical(self, "Lỗi lưu ảnh", str(exc))
            return
        self.states[self.index] = SAVED
        self.status_label.setText(f"Đã lưu: {saved['cutout_path']}")

    def save_all(self):
        engine = self.engine_getter()
        if (not self.predictions or engine is None or thread_is_valid(self.save_thread)
                or self.editor_busy()):
            return
        self.save_thread = self.save_worker = None
        self.store_current()
        self._save_debug_for_current_batch = self._save_debug()
        self.set_busy_buttons(True)
        self.set_tools_enabled(False)
        self.view.setCurrentWidget(self.canvas)
        self.canvas.set_loading_mode(f"Đang lưu {len(self.predictions)} ảnh...", 0,
                                     len(self.predictions), cancel_text=SAVE_STOP_TEXT)
        self.status_label.setText("Đang lưu tất cả ảnh đã cắt...")

        self.save_thread = QThread()
        self.save_worker = BatchSaveWorker(engine, self.predictions, self.output_dir,
                                           save_debug=self._save_debug_for_current_batch)
        self.save_worker.moveToThread(self.save_thread)

        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.progress.connect(self.on_save_progress)
        self.save_worker.finished.connect(self.on_save_finished)
        self.save_worker.canceled.connect(self.on_save_canceled)
        self.save_worker.error.connect(self.on_save_error)

        for signal in (self.save_worker.finished, self.save_worker.canceled, self.save_worker.error):
            signal.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self._on_save_thread_finished)
        self.save_thread.finished.connect(self.save_thread.deleteLater)

        self.save_thread.start()

    def _on_save_thread_finished(self):
        finished = self.sender()
        if finished is not self.save_thread:
            return
        if thread_is_valid(finished):
            finished.wait()
        self.save_thread = None
        self.save_worker = None

    def _back_to_editor(self):
        self.set_busy_buttons(False)
        self.canvas.set_upload_mode()
        self.view.setCurrentWidget(self.edit_canvas)
        self.set_tools_enabled(True)

    def on_save_progress(self, current: int, total: int, message: str):
        if self.save_worker is not None and self.save_worker.cancel_requested:
            return  # keep "Đang dừng lưu..." until the worker stops
        self.canvas.set_loading_mode(message, current, total, cancel_text=SAVE_STOP_TEXT)
        self.status_label.setText(f"{message} ({current}/{total})")

    def on_save_error(self, err: str):
        # The staged images stay: choose another destination and save again.
        self._back_to_editor()
        QMessageBox.critical(self, 'Lỗi lưu ảnh', err)
        self.status_label.setText('Lưu lỗi; kết quả vẫn được giữ để bạn thử lại.')

    def on_save_canceled(self, saved: int):
        # Images are saved in order: the first ``saved`` ones are on disk.
        self.states[:saved] = [SAVED] * saved
        self._back_to_editor()
        self.status_label.setText(f"Đã dừng lưu: đã lưu {saved}/{len(self.predictions)} ảnh. "
                                  "Bấm Lưu tất cả để lưu lại cả lượt.")

    def on_save_finished(self, saved_path: str):
        self.states = [SAVED] * len(self.predictions)
        self._back_to_editor()
        if self._save_debug_for_current_batch:
            self.status_label.setText(f"Đã lưu batch. Log: {saved_path}")
            QMessageBox.information(self, "Batch đã lưu", f"Đã lưu batch.\nLog:\n{saved_path}")
        else:
            self.status_label.setText(f"Đã lưu ảnh PNG tại: {saved_path}")
            QMessageBox.information(self, "Batch đã lưu", f"Đã lưu ảnh PNG trong suốt tại:\n{saved_path}")
