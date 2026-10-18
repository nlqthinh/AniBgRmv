"""Main navigation and asynchronous, strict CPU/GPU model selection."""
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from bg_engine import BgRemoveEngine

from .app_config import F2_A1_ONNX_PATH, M6_ONNX_PATH, get_config_path
from .device_management import current_selection, cuda_index, save_selection
from .pages.batch_cut_page import BatchCutPage
from .pages.compute_status_page import ComputeStatusPage
from .pages.home_page import HomePage
from .pages.single_cut_page import SingleCutPage


class ModelLoadWorker(QObject):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, selection):
        super().__init__()
        self.selection = dict(selection)

    @Slot()
    def run(self):
        try:
            engine = BgRemoveEngine(
                f2a1_model_path=F2_A1_ONNX_PATH,
                m6_model_path=M6_ONNX_PATH,
                size=768,
                compute_selection=self.selection,
            )
            self.loaded.emit(engine)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anime Background Remover - F3 → ViTMatte M7")
        screen_rect = QGuiApplication.primaryScreen().availableGeometry()
        start_w = min(1120, max(960, int(screen_rect.width() * 0.90)))
        start_h = min(760, max(640, int(screen_rect.height() * 0.88)))
        self.resize(start_w, start_h)
        self.setMinimumSize(900, 620)

        self.engine: BgRemoveEngine | None = None
        self.selection = current_selection()
        self.load_thread = None
        self.load_worker = None

        self.status_label = QLabel(f"Đang khởi động... | Config: {get_config_path()}")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; color: #555555;")

        self.stack = QStackedWidget()
        self.home_page = HomePage()
        self.single_page = SingleCutPage(self.get_engine)
        self.batch_page = BatchCutPage(self.get_engine)
        self.compute_page = ComputeStatusPage()

        for page in (self.home_page, self.single_page, self.batch_page, self.compute_page):
            self.stack.addWidget(page)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.stack)
        self.setLayout(layout)

        self.home_page.singleRequested.connect(lambda: self.stack.setCurrentWidget(self.single_page))
        self.home_page.batchRequested.connect(lambda: self.stack.setCurrentWidget(self.batch_page))
        self.home_page.computeRequested.connect(self.open_compute_page)
        self.single_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.batch_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.compute_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.compute_page.selectionRequested.connect(self.change_compute)

        self.set_pages_enabled(False)
        self.load_model()

    def get_engine(self):
        return self.engine

    def set_pages_enabled(self, enabled: bool):
        self.home_page.btn_single.setEnabled(enabled)
        self.home_page.btn_batch.setEnabled(enabled)

    def open_compute_page(self):
        self.compute_page.set_selection(self.selection)
        if self.load_thread is not None:
            self.compute_page.show_runtime("Đang nạp model theo cấu hình đã chọn...")
        elif self.engine is not None:
            self.compute_page.show_runtime("App đang sử dụng: " + self.engine.runtime_info)
        else:
            mode = self.selection["mode"]
            self.compute_page.show_runtime("GPU chưa sẵn sàng. Hãy GPU Check và chọn GPU tương thích." if mode == "gpu" else "CPU chưa sẵn sàng.")
        self.stack.setCurrentWidget(self.compute_page)

    def change_compute(self, selection):
        if self.load_thread is not None:
            return
        try:
            self.selection = save_selection(selection)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Lỗi lưu cấu hình CPU/GPU", str(exc))
            self.compute_page.set_selection(self.selection)
            return
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
        self.load_thread.finished.connect(self._on_loader_cleanup)
        self.load_thread.start()

    def _on_model_loaded(self, engine):
        self.engine = engine
        message = f"App đang sử dụng: {engine.runtime_info} | Config: {get_config_path()}"
        self.status_label.setText(message)
        self.compute_page.show_runtime("App đang sử dụng: " + engine.runtime_info)
        self.set_pages_enabled(True)

    def _on_model_failed(self, message):
        self.engine = None
        self.set_pages_enabled(False)
        self.status_label.setText("Không thể tải model bằng thiết bị đã chọn.")
        self.compute_page.show_runtime("Không thể sử dụng thiết bị đã chọn: " + message)
        if self.stack.currentWidget() == self.compute_page:
            QMessageBox.warning(self, "CPU/GPU không sẵn sàng", message + "\n\nKhông chuyển sang CPU tự động.")
        else:
            QMessageBox.warning(self, "Lỗi nạp model", message + "\n\nMở 'Trạng thái CPU/GPU' để chọn lại.")

    def _on_loader_cleanup(self):
        self.load_worker.deleteLater()
        self.load_thread.deleteLater()
        self.load_worker = None
        self.load_thread = None
        self.compute_page.set_applying(False)

    def closeEvent(self, event):
        if self.load_thread is not None and self.load_thread.isRunning():
            QMessageBox.information(self, "Đang nạp model", "Vui lòng chờ hoàn tất nạp model trước khi thoát.")
            event.ignore()
            return
        if self.compute_page.scan_thread is not None and self.compute_page.scan_thread.isRunning():
            self.compute_page._stop_scan()
            QMessageBox.information(self, "Đang tìm GPU", "Đã yêu cầu dừng GPU Check. Vui lòng chờ trước khi thoát.")
            event.ignore()
            return
        super().closeEvent(event)
