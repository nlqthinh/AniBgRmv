from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from bg_engine import BgRemoveEngine

from .app_config import F2_A1_ONNX_PATH, M6_ONNX_PATH, get_config_path
from .pages.batch_cut_page import BatchCutPage
from .pages.home_page import HomePage
from .pages.single_cut_page import SingleCutPage

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Anime Background Remover - F2_A1 → ViTMatte M6")
        screen_rect = QGuiApplication.primaryScreen().availableGeometry()
        start_w = min(1120, max(960, int(screen_rect.width() * 0.90)))
        start_h = min(760, max(640, int(screen_rect.height() * 0.88)))
        self.resize(start_w, start_h)
        self.setMinimumSize(900, 620)

        self.engine: BgRemoveEngine | None = None

        self.status_label = QLabel(f"Đang load pipeline F2_A1 → ViTMatte M6... | Config: {get_config_path()}")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 13px; color: #555555;")

        self.stack = QStackedWidget()
        self.home_page = HomePage()
        self.single_page = SingleCutPage(self.get_engine)
        self.batch_page = BatchCutPage(self.get_engine)

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.single_page)
        self.stack.addWidget(self.batch_page)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.stack)
        self.setLayout(layout)

        self.home_page.singleRequested.connect(lambda: self.stack.setCurrentWidget(self.single_page))
        self.home_page.batchRequested.connect(lambda: self.stack.setCurrentWidget(self.batch_page))
        self.single_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))
        self.batch_page.backRequested.connect(lambda: self.stack.setCurrentWidget(self.home_page))

        self.set_pages_enabled(False)
        self.load_model()

    def get_engine(self):
        return self.engine

    def set_pages_enabled(self, enabled: bool):
        self.home_page.btn_single.setEnabled(enabled)
        self.home_page.btn_batch.setEnabled(enabled)

    def load_model(self):
        try:
            self.engine = BgRemoveEngine(
                f2a1_model_path=F2_A1_ONNX_PATH,
                m6_model_path=M6_ONNX_PATH,
                size=768,
            )
            self.status_label.setText(
                f"Pipeline sẵn sàng: F2_A1 → M6 | Runtime: {self.engine.runtime_info} | Config: {get_config_path()}"
            )
            self.set_pages_enabled(True)
        except Exception as e:
            detail = (
                f"{e}\n\n"
                f"Config đang đọc:\n{get_config_path()}\n\n"
                f"UNet ONNX path:\n{F2_A1_ONNX_PATH}\n\n"
                f"ViTMatte ONNX path:\n{M6_ONNX_PATH}"
            )
            QMessageBox.critical(self, "Lỗi load model", detail)
            self.status_label.setText("Load pipeline thất bại. Kiểm tra app_config.json cạnh exe/app.py.")
            self.set_pages_enabled(False)

