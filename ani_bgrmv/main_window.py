"""Main navigation and asynchronous, strict CPU/GPU model selection."""
from PySide6.QtCore import QObject, QPoint, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from .app_config import get_config_path
from .model_settings import current_models, save_models, validate_choice
from .qt_threads import thread_is_running, thread_is_valid
from .compute_selection import current_selection, cuda_index, save_selection
from .engine import BgRemoveEngine
from .pages.batch_cut_page import BatchCutPage
from .pages.compute_status_page import ComputeStatusPage
from .pages.home_page import HomePage
from .pages.single_cut_page import SingleCutPage
from .pages.video_cut_page import VideoCutPage
from .widgets.elided_label import ElidedLabel


class ModelLoadWorker(QObject):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, selection):
        super().__init__()
        self.selection = dict(selection)

    @Slot()
    def run(self):
        try:
            self.loaded.emit(BgRemoveEngine(self.selection))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("AniBgRmvMainWindow")
        self.setWindowTitle("Anime Background Remover - UNet → ViTMatte")
        # Same start size as the annotator, so the picture areas match it.
        screen_rect = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, screen_rect.width()), min(760, screen_rect.height()))
        self.setMinimumSize(min(900, screen_rect.width()), min(620, screen_rect.height()))

        self.engine: BgRemoveEngine | None = None
        self.selection = current_selection()
        self.load_thread = None
        self.load_worker = None
        self._rollback_models = None
        self._reload_previous_models = False

        self.status_label = ElidedLabel(f"Đang khởi động... | Config: {get_config_path()}")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 13px; color: #F0F0F0;")

        self.stack = QStackedWidget()
        self.home_page = HomePage()
        self.single_page = SingleCutPage(self.get_engine)
        self.batch_page = BatchCutPage(self.get_engine)
        self.video_page = VideoCutPage(self.get_engine)
        self.compute_page = ComputeStatusPage()
        # Stable page selectors for dark backgrounds, independent of OS theme.
        self.home_page.setObjectName("AniBgRmvHomePage")
        self.single_page.setObjectName("AniBgRmvSinglePage")
        self.batch_page.setObjectName("AniBgRmvBatchPage")
        self.compute_page.setObjectName("AniBgRmvComputePage")
        self.video_page.setObjectName("AniBgRmvVideoPage")

        for page in (self.home_page, self.single_page, self.batch_page, self.video_page, self.compute_page):
            self.stack.addWidget(page)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.stack)
        self.setLayout(layout)

        self.home_page.singleRequested.connect(lambda: self.stack.setCurrentWidget(self.single_page))
        self.home_page.batchRequested.connect(lambda: self.stack.setCurrentWidget(self.batch_page))
        self.home_page.videoRequested.connect(lambda: self.stack.setCurrentWidget(self.video_page))
        self.home_page.computeRequested.connect(self.open_compute_page)
        self.single_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.batch_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.video_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.compute_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.compute_page.selectionRequested.connect(self.change_compute)
        self.compute_page.modelsRequested.connect(self.change_models)

        self.set_pages_enabled(False)
        self.load_model()

    def get_engine(self):
        return self.engine

    def set_pages_enabled(self, enabled: bool):
        self.home_page.btn_single.setEnabled(enabled)
        self.home_page.btn_batch.setEnabled(enabled)
        self.home_page.btn_video.setEnabled(enabled)

    def open_compute_page(self):
        self.compute_page.set_selection(self.selection)
        self.compute_page.set_models(current_models())
        if self.load_thread is not None:
            self.compute_page.show_runtime("Đang nạp model theo cấu hình đã chọn...")
        elif self.engine is not None:
            self.compute_page.show_runtime("App đang sử dụng: " + self.engine.runtime_info)
        else:
            mode = self.selection["mode"]
            self.compute_page.show_runtime("GPU chưa sẵn sàng. Hãy GPU Check và chọn GPU tương thích." if mode == "gpu" else "CPU chưa sẵn sàng.")
        self.stack.setCurrentWidget(self.compute_page)

    def change_compute(self, selection):
        if self.video_page.is_busy or any(map(thread_is_running, (
            self.single_page.current_thread, self.single_page.edit_thread,
            self.batch_page.cut_thread, self.batch_page.save_thread, self.batch_page.edit_thread,
        ))):
            QMessageBox.warning(self, "Đang xử lý", "Dừng tác vụ ảnh/video trước khi đổi CPU/GPU.")
            self.compute_page.set_selection(self.selection)
            return
        if self.load_thread is not None:
            return
        previous_selection = dict(self.selection)
        try:
            self.selection = save_selection(selection)
            self.compute_page.set_models(current_models())
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Lỗi lưu cấu hình CPU/GPU", str(exc))
            self.compute_page.set_selection(self.selection)
            return
        # Masks produced on another device must not be treated as a fresh Track.
        if previous_selection != self.selection and self.video_page.tracked is not None:
            self.video_page.forget_track('Đã đổi CPU/GPU. Hãy Track lại video trên thiết bị mới.')
        # Immediately stop using the old engine after the user changes mode.
        self.engine = None
        self.set_pages_enabled(False)
        if self.selection["mode"] == "gpu":
            try:
                cuda_index(self.selection)
            except (ValueError, RuntimeError) as exc:
                text = str(exc)
                self.status_label.setText("GPU không sẵn sàng — " + text)
                self.compute_page.show_runtime("GPU không sẵn sàng — " + text)
                return
        self.load_model()

    def change_models(self, models):
        if self.load_thread is not None or self.compute_page.import_thread is not None:
            return
        if self.video_page.is_busy or any(map(thread_is_running, (
            self.single_page.current_thread, self.single_page.edit_thread,
            self.batch_page.cut_thread, self.batch_page.save_thread, self.batch_page.edit_thread,
        ))):
            QMessageBox.warning(self, 'Đang xử lý', 'Hoàn tất/dừng tác vụ ảnh hoặc video trước khi đổi model.')
            return
        before = current_models()
        try:
            validate_choice(models)
            if models == before:
                self.compute_page.model_feedback.setText('Model Setting không thay đổi.')
                return
            save_models(models)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, 'Không thể lưu Model Setting', str(exc))
            self.compute_page.set_models(before)
            return
        self.compute_page.set_models(current_models())
        self.compute_page.model_feedback.setText('Đã lưu model vào app_config.json.')

        # Masks tracked with another SAM checkpoint must never be reused.
        if (before['video_sam'] != models['video_sam']
                or before['video_backend'] != models['video_backend']):
            self.video_page.forget_track('Model SAM đã đổi. Cần Track lại video.')

        single_changed = any(before[name] != models[name] for name in
                             ('single_auto', 'single_vit', 'single_sam', 'single_sam_vit'))
        batch_changed = any(before[name] != models[name] for name in
                            ('batch_auto', 'batch_vit'))
        if single_changed or batch_changed:
            if single_changed:
                self.single_page.clear_current_preview()
            if batch_changed:
                self.batch_page.clear_current_preview()
            # Old sessions may be using old ONNX files; discard before loading.
            self._rollback_models = before
            self.engine = None
            self.set_pages_enabled(False)
            import gc
            gc.collect()
            self.load_model()

    def load_model(self):
        if self.load_thread is not None:
            return
        if self.selection["mode"] == "gpu":
            try:
                cuda_index(self.selection)
            except (ValueError, RuntimeError) as exc:
                self.status_label.setText(str(exc))
                self.compute_page.show_runtime("GPU chưa sẵn sàng — " + str(exc))
                return
        self.set_pages_enabled(False)
        self.compute_page.set_applying(True)
        self.status_label.setText("Đang nạp model trên GPU đã chọn..." if self.selection["mode"] == "gpu" else "Đang nạp model trên CPU...")
        self.compute_page.show_runtime(self.status_label.text())

        self.load_thread = QThread(self)
        self.load_worker = ModelLoadWorker(self.selection)
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.loaded.connect(self._on_model_loaded)
        self.load_worker.failed.connect(self._on_model_failed)
        self.load_worker.loaded.connect(self.load_thread.quit)
        self.load_worker.failed.connect(self.load_thread.quit)
        self.load_thread.finished.connect(self.load_worker.deleteLater)
        self.load_thread.finished.connect(self._on_loader_cleanup)
        self.load_thread.start()

    def _on_model_loaded(self, engine):
        self.engine = engine
        message = f"App đang sử dụng: {engine.runtime_info} | Config: {get_config_path()}"
        self.status_label.setText(message)
        self.compute_page.show_runtime("App đang sử dụng: " + engine.runtime_info)
        self.set_pages_enabled(True)
        self._rollback_models = None

    def _on_model_failed(self, message):
        self.engine = None
        self.set_pages_enabled(False)
        if self._rollback_models is not None:
            message += '\nCấu hình Model Setting sẽ được phục hồi.'
            previous = self._rollback_models
            self._rollback_models = None
            try:
                save_models(previous)
                self.compute_page.set_models(previous)
                self._reload_previous_models = True
            except Exception as rollback_exc:
                message += '\nKhông thể phục hồi cấu hình trước: ' + str(rollback_exc)
        self.status_label.setText("Không thể tải model bằng thiết bị đã chọn.")
        self.compute_page.show_runtime("Không thể sử dụng thiết bị đã chọn: " + message)
        if self.stack.currentWidget() == self.compute_page:
            QMessageBox.warning(self, "CPU/GPU không sẵn sàng", message + "\n\nKhông chuyển sang CPU tự động.")
        else:
            QMessageBox.warning(self, "Lỗi nạp model", message + "\n\nMở 'Setting' để chọn lại.")

    def _on_loader_cleanup(self):
        thread = self.sender()
        if thread is self.load_thread:
            # Worker deletion is already connected to QThread.finished.
            # Release Python references only after this loader has stopped.
            if thread_is_valid(thread):
                thread.wait()
                thread.deleteLater()
            self.load_worker = None
            self.load_thread = None
        self.compute_page.set_applying(False)
        if self._reload_previous_models:
            self._reload_previous_models = False
            self.load_model()

    # -- window on screen ----------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._keep_title_bar_on_screen)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._keep_title_bar_on_screen)

    def moveEvent(self, event):
        super().moveEvent(event)
        QTimer.singleShot(0, self._keep_title_bar_on_screen)

    def _keep_title_bar_on_screen(self):
        """The title bar (─ □ ✕, dragging) must stay reachable: a window taller
        than its screen's free area, or whose title bar is on no screen, comes
        back inside. Minimized, maximized and full-screen windows are left alone."""
        hidden_frame = (Qt.WindowState.WindowMinimized | Qt.WindowState.WindowMaximized
                        | Qt.WindowState.WindowFullScreen)
        if not self.isVisible() or self.screen() is None or self.windowState() & hidden_frame:
            return
        free = self.screen().availableGeometry()
        title = self.frameGeometry().height() - self.height()  # title bar + borders
        if self.frameGeometry().height() > free.height():
            self.setMinimumHeight(min(self.minimumHeight(), free.height() - title))
            self.resize(self.width(), free.height() - title)
        frame = self.frameGeometry()
        if QGuiApplication.screenAt(QPoint(frame.center().x(), frame.top() + title // 2)) is None:
            left = max(free.left(), min(frame.left(), free.right() + 1 - frame.width()))
            self.move(left, free.top())  # a window's pos includes its frame

    def closeEvent(self, event):
        if any(map(thread_is_running, (
            self.single_page.current_thread, self.single_page.edit_thread,
            self.batch_page.cut_thread, self.batch_page.save_thread, self.batch_page.edit_thread,
        ))):
            QMessageBox.information(self, "Đang xử lý ảnh", "Bấm Dừng hoặc chờ tác vụ ảnh kết thúc trước khi đóng app.")
            event.ignore()
            return
        if self.video_page.is_busy:
            QMessageBox.information(self, "Đang xử lý video", "Bấm Dừng rồi chờ kết thúc tác vụ video trước khi đóng app.")
            event.ignore()
            return
        if thread_is_running(self.load_thread):
            QMessageBox.information(self, "Đang nạp model", "Vui lòng chờ hoàn tất nạp model trước khi thoát.")
            event.ignore()
            return
        if thread_is_running(self.compute_page.import_thread):
            QMessageBox.information(self, 'Đang kiểm tra model', 'Kiểm tra Import chưa hoàn tất.')
            event.ignore()
            return
        if thread_is_running(self.compute_page.scan_thread):
            self.compute_page._stop_scan()
            QMessageBox.information(self, "Đang tìm GPU", "Đã yêu cầu dừng GPU Check. Vui lòng chờ trước khi thoát.")
            event.ignore()
            return
        self.video_page.shutdown()
        super().closeEvent(event)
