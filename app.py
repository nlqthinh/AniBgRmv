import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QFrame,
    QSizePolicy,
)

from bg_engine import BgRemoveEngine


# =========================
# APP CONFIG
# =========================

CONFIG_FILE_NAME = "app_config.json"

DEFAULT_APP_CONFIG = {
    "_note": (
        "Relative paths are resolved from the folder containing app.py "
        "or AniBGRmv.exe. On Windows, using forward slash paths is recommended, "
        "for example: D:/AniBGRmv/output/single_saved"
    ),
    "model": {
        "unet_onnx_path": "models/f2a1_unet_mit_b1_768.onnx",
        "vitmatte_onnx_path": "models/m6_vitmatte_m6_768.onnx",
    },
    "output": {
        "single_output_dir": "output/single_saved",
        "batch_output_dir": "output/batch_saved",
        "save_debug_outputs": True,
    },
    "batch": {
        "max_batch_images": 30,
    },
}


# Source mode: config nằm cạnh app.py.
# PyInstaller exe mode: config nằm cạnh AniBGRmv.exe.
def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_BASE_DIR = get_app_base_dir()


def deep_merge_config(default: dict, user: dict) -> dict:
    """
    Merge config để user có thể thiếu vài field mà app vẫn có default an toàn.
    """
    merged = dict(default)
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_external_config_path() -> Path:
    """
    Config người dùng sửa sau khi build.

    Source mode:
        project/app_config.json

    Exe mode:
        dist/AniBGRmv/app_config.json, nằm cạnh AniBGRmv.exe
    """
    return APP_BASE_DIR / CONFIG_FILE_NAME


def get_bundled_config_path() -> Path:
    """
    Config được PyInstaller bundle vào _internal/_MEIPASS.
    Chỉ dùng làm seed lần đầu nếu chưa có external config cạnh exe.
    """
    return Path(getattr(sys, "_MEIPASS", APP_BASE_DIR)) / CONFIG_FILE_NAME


def get_config_path() -> Path:
    # Hàm này dùng để hiển thị path config thật sự user nên sửa.
    return get_external_config_path()


def write_default_config(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(DEFAULT_APP_CONFIG, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_external_config_exists() -> Path:
    """
    Quy tắc:
    1. Nếu đã có app_config.json cạnh exe/app.py -> dùng file đó.
    2. Nếu chưa có, nhưng có config được bundle -> copy bundle ra cạnh exe/app.py.
    3. Nếu cả hai đều không có -> tạo default config.

    Lý do:
    - Người dùng cần sửa config sau khi build mà không rebuild.
    - PyInstaller onedir có thể đặt data trong _internal, không phải cạnh exe.
    """
    external_config = get_external_config_path()
    bundled_config = get_bundled_config_path()

    if external_config.exists():
        return external_config

    external_config.parent.mkdir(parents=True, exist_ok=True)

    if bundled_config.exists() and bundled_config.resolve() != external_config.resolve():
        shutil.copy2(bundled_config, external_config)
        return external_config

    write_default_config(external_config)
    return external_config


def load_app_config() -> dict:
    config_path = ensure_external_config_exists()

    try:
        user_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"File cấu hình {config_path} không phải JSON hợp lệ. "
            f"Hãy sửa JSON hoặc xóa file để app tự tạo lại. Chi tiết: {e}"
        ) from e

    if not isinstance(user_config, dict):
        raise ValueError(f"File cấu hình {config_path} phải là JSON object.")

    return deep_merge_config(DEFAULT_APP_CONFIG, user_config)


def resolve_app_path(path_value: str) -> Path:
    """
    Đường dẫn tuyệt đối: dùng nguyên.
    Đường dẫn tương đối: hiểu là nằm dưới APP_BASE_DIR.
    """
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return APP_BASE_DIR / p


APP_CONFIG = load_app_config()

F2_A1_ONNX_PATH = str(resolve_app_path(APP_CONFIG["model"]["unet_onnx_path"]))
M6_ONNX_PATH = str(resolve_app_path(APP_CONFIG["model"]["vitmatte_onnx_path"]))

MAX_BATCH_IMAGES = int(APP_CONFIG["batch"]["max_batch_images"])

SINGLE_OUTPUT_DIR = str(resolve_app_path(APP_CONFIG["output"]["single_output_dir"]))
BATCH_OUTPUT_DIR = str(resolve_app_path(APP_CONFIG["output"]["batch_output_dir"]))

SAVE_DEBUG_OUTPUTS = bool(APP_CONFIG["output"]["save_debug_outputs"])

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# =========================
# STYLE
# =========================

BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 700;
    padding: 10px 18px;
    min-height: 42px;
    border-radius: 10px;
    border: 1px solid #B8B8B8;
    color: #000000;
    background: #F6F6F6;
}
QPushButton:hover {
    color: #000000;
    background: #ECECEC;
}
QPushButton:disabled {
    color: #777777;
    background: #EEEEEE;
    border: 1px solid #DDDDDD;
}
"""

PRIMARY_BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 900;
    padding: 10px 22px;
    min-height: 44px;
    border-radius: 10px;
    border: 1px solid #159447;
    color: #000000;
    background: #35D06F;
}
QPushButton:hover {
    color: #000000;
    background: #22C55E;
}
QPushButton:disabled {
    color: #777777;
    background: #BDEECC;
    border: 1px solid #BDEECC;
}
"""

DANGER_BUTTON_STYLE = """
QPushButton {
    font-size: 18px;
    font-weight: 900;
    min-width: 38px;
    min-height: 34px;
    border-radius: 8px;
    border: 1px solid #A0A0A0;
    color: white;
    background: #111111;
}
QPushButton:hover {
    background: #C62828;
    border: 1px solid #C62828;
}
"""

CANCEL_BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 900;
    padding: 9px 22px;
    min-height: 40px;
    border-radius: 10px;
    border: 1px solid #000000;
    color: #FF2B2B;
    background: #000000;
}
QPushButton:hover {
    color: #FF4D4D;
    background: #171717;
}
QPushButton:disabled {
    color: #B54B4B;
    background: #222222;
    border: 1px solid #222222;
}
"""


def apply_button_style(button: QPushButton, primary: bool = False):
    button.setStyleSheet(PRIMARY_BUTTON_STYLE if primary else BUTTON_STYLE)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


# =========================
# SMALL HELPERS
# =========================


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def collect_image_files_from_paths(paths: list[str], recursive: bool = True) -> list[str]:
    """
    Input có thể là file hoặc folder do kéo-thả.
    - File ảnh: lấy trực tiếp.
    - Folder: lấy ảnh trong folder. Mặc định recursive cho batch.
    """
    found: list[str] = []

    for raw in paths:
        p = Path(raw)

        if p.is_file() and is_image_file(p):
            found.append(str(p))

        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(p.glob(pattern)):
                if child.is_file() and is_image_file(child):
                    found.append(str(child))

    seen = set()
    unique = []
    for p in found:
        key = str(Path(p).resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def pil_to_pixmap(pil_image):
    rgba = pil_image.convert("RGBA")
    w, h = rgba.size
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def center_button_row(*buttons: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(12)
    row.addStretch(1)
    for btn in buttons:
        row.addWidget(btn)
    row.addStretch(1)
    return row


# =========================
# WORKERS
# =========================


class SingleCutWorker(QObject):
    finished = Signal(dict)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_path: str):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if self.cancel_requested:
                self.canceled.emit()
                return

            result = self.engine.process_for_preview(self.image_path)

            if self.cancel_requested:
                self.canceled.emit()
                return

            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchCutWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_paths: list[str]):
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            results = []
            total = len(self.image_paths)

            for idx, path in enumerate(self.image_paths, start=1):
                if self.cancel_requested:
                    self.canceled.emit()
                    return

                self.progress.emit(idx - 1, total, f"Đang cắt: {Path(path).name}")
                result = self.engine.process_for_preview(path)

                if self.cancel_requested:
                    self.canceled.emit()
                    return

                results.append(result)
                self.progress.emit(idx, total, f"Đã cắt: {Path(path).name}")

            self.finished.emit(results)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchSaveWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        engine: BgRemoveEngine,
        predictions: list[dict],
        output_dir: str,
        save_debug: bool = True,
    ):
        super().__init__()
        self.engine = engine
        self.predictions = predictions
        self.output_dir = output_dir
        self.save_debug = save_debug

    def run(self):
        try:
            logs = []
            total = len(self.predictions)

            for idx, pred in enumerate(self.predictions, start=1):
                self.progress.emit(idx - 1, total, f"Đang lưu: {Path(pred['input_path']).name}")
                saved = self.engine.save_prediction(
                    pred,
                    self.output_dir,
                    save_debug=self.save_debug,
                )
                logs.append(saved)
                self.progress.emit(idx, total, f"Đã lưu: {Path(pred['input_path']).name}")

            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "batch_log.csv"
            pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")

            self.finished.emit(str(log_path))
        except Exception:
            self.error.emit(traceback.format_exc())


# =========================
# REUSABLE UI COMPONENTS
# =========================


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

    def __init__(self, title: str):
        super().__init__()
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(520)
        self._mode = "upload"
        self._pixmap: QPixmap | None = None

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
        self.btn_clear.setVisible(False)
        self.btn_clear.clicked.connect(self.clearRequested.emit)

        clear_row = QHBoxLayout()
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
        layout.setContentsMargins(18, 14, 18, 18)
        layout.addLayout(clear_row)
        layout.addLayout(center_box)
        self.setLayout(layout)

    def set_upload_mode(self):
        self._mode = "upload"
        self._pixmap = None
        self.setCursor(Qt.PointingHandCursor)
        self.btn_clear.setVisible(False)
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
        self.title_label.setVisible(False)
        self.loading_label.setVisible(False)
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.preview_label.setVisible(True)
        self._pixmap = pil_to_pixmap(pil_image)
        self._update_scaled_preview()
        if caption:
            self.preview_label.setToolTip(caption)

    def _update_scaled_preview(self):
        if self._pixmap is None:
            return
        max_w = max(100, self.width() - 60)
        max_h = max(100, self.height() - 90)
        scaled = self._pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

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


class HomePage(QWidget):
    singleRequested = Signal()
    batchRequested = Signal()

    def __init__(self):
        super().__init__()

        title = QLabel("Anime Background Remover")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: 800;")

        subtitle = QLabel("Chọn tác vụ cần làm")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 15px; color: #555555;")

        self.btn_single = QPushButton("Cắt nền 1 hình")
        self.btn_batch = QPushButton("Cắt nhiều hình")
        self.btn_video = QPushButton("Cắt video (sau)")
        self.btn_object = QPushButton("Xóa vật thể (sau)")

        for btn in [self.btn_single, self.btn_batch, self.btn_video, self.btn_object]:
            btn.setMinimumWidth(260)
            apply_button_style(btn)

        self.btn_video.setEnabled(False)
        self.btn_object.setEnabled(False)

        box = QVBoxLayout()
        box.setSpacing(14)
        box.addWidget(self.btn_single, alignment=Qt.AlignCenter)
        box.addWidget(self.btn_batch, alignment=Qt.AlignCenter)
        box.addWidget(self.btn_video, alignment=Qt.AlignCenter)
        box.addWidget(self.btn_object, alignment=Qt.AlignCenter)

        outer = QVBoxLayout()
        outer.addStretch(1)
        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addSpacing(30)
        outer.addLayout(box)
        outer.addStretch(2)

        self.setLayout(outer)

        self.btn_single.clicked.connect(self.singleRequested.emit)
        self.btn_batch.clicked.connect(self.batchRequested.emit)


class SingleCutPage(QWidget):
    backRequested = Signal()

    def __init__(self, engine_getter):
        super().__init__()
        self.engine_getter = engine_getter

        self.current_prediction: dict | None = None
        self.current_thread: QThread | None = None
        self.current_worker: SingleCutWorker | None = None
        self.cancel_requested = False
        self.output_dir = SINGLE_OUTPUT_DIR

        self.btn_back = QPushButton("← Trang chủ")
        apply_button_style(self.btn_back)

        self.title = QLabel("Cắt nền 1 hình")
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")

        self.status_label = QLabel("Upload hoặc kéo-thả ảnh vào khung chính.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #444444;")

        self.instruction_label = QLabel("Kéo-thả ảnh vào đây hoặc bấm vào khung để chọn ảnh.")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 14px; color: #333333;")

        self.path_label = QLabel(f"Đường dẫn lưu: {self.output_dir}")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size: 12px; color: #666666;")

        top = QHBoxLayout()
        top.addWidget(self.btn_back)
        top.addStretch(1)
        top.addWidget(self.title)
        top.addStretch(2)

        self.canvas = UploadPreviewCanvas("Upload ảnh")

        self.btn_save = QPushButton("Lưu ảnh")
        self.btn_config = QPushButton("Đường dẫn lưu")
        self.btn_edit = QPushButton("Edit")
        self.btn_open_output = QPushButton("Mở folder đã lưu")

        apply_button_style(self.btn_save, primary=True)
        apply_button_style(self.btn_config)
        apply_button_style(self.btn_edit)
        apply_button_style(self.btn_open_output)

        self.btn_save.setEnabled(False)
        self.btn_edit.setEnabled(False)

        buttons = center_button_row(
            self.btn_save,
            self.btn_edit,
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
        self.canvas.clicked.connect(self.select_image)
        self.canvas.filesDropped.connect(self.handle_drop)
        self.canvas.clearRequested.connect(self.clear_current_preview)
        self.canvas.cancelRequested.connect(self.cancel_current_cut)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_config.clicked.connect(self.choose_output_dir)
        self.btn_open_output.clicked.connect(lambda: self.open_folder(self.output_dir))
        self.btn_edit.clicked.connect(self.edit_placeholder)

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

        self.canvas.set_preview_pil(
            prediction["preview_checker_pil"],
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
            save_debug=SAVE_DEBUG_OUTPUTS,
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

    def edit_placeholder(self):
        QMessageBox.information(self, "Edit", "Tính năng Edit/SAM sẽ được thêm ở bước sau.")

    def open_folder(self, folder: str):
        p = Path(folder).resolve()
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(p)


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

        self.btn_back = QPushButton("← Trang chủ")
        apply_button_style(self.btn_back)

        self.title = QLabel("Cắt nhiều hình")
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")

        self.status_label = QLabel("Upload nhiều ảnh, kéo-thả ảnh/folder, hoặc chọn cắt cả folder.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #444444;")

        self.instruction_label = QLabel("Kéo-thả nhiều ảnh/folder vào đây hoặc bấm vào khung để chọn nhiều ảnh.")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 14px; color: #333333;")

        self.path_label = QLabel(f"Đường dẫn lưu: {self.output_dir}")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size: 12px; color: #666666;")

        top = QHBoxLayout()
        top.addWidget(self.btn_back)
        top.addStretch(1)
        top.addWidget(self.title)
        top.addStretch(2)

        self.canvas = UploadPreviewCanvas("Upload ảnh")

        self.btn_save_all = QPushButton("Lưu ảnh")
        self.btn_select_folder = QPushButton("Cắt cả folder")
        self.btn_config = QPushButton("Đường dẫn lưu")
        self.btn_open_output = QPushButton("Mở folder đã lưu")

        apply_button_style(self.btn_save_all, primary=True)
        apply_button_style(self.btn_select_folder)
        apply_button_style(self.btn_config)
        apply_button_style(self.btn_open_output)

        self.btn_save_all.setEnabled(False)

        buttons = center_button_row(
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
            self.canvas.set_preview_pil(
                first["preview_checker_pil"],
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
        self.btn_select_folder.setEnabled(False)
        self.btn_config.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.btn_back.setEnabled(False)

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
            save_debug=SAVE_DEBUG_OUTPUTS,
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

    def on_save_finished(self, log_path: str):
        self.btn_select_folder.setEnabled(True)
        self.btn_config.setEnabled(True)
        self.btn_open_output.setEnabled(True)
        self.btn_back.setEnabled(True)
        self.btn_save_all.setEnabled(True)

        if self.predictions:
            first = self.predictions[0]
            self.canvas.set_preview_pil(
                first["preview_checker_pil"],
                caption=f"Preview ảnh đầu tiên: {Path(first['input_path']).name}",
            )

        self.status_label.setText(f"Đã lưu batch. Log: {log_path}")
        QMessageBox.information(self, "Batch đã lưu", f"Đã lưu batch.\nLog:\n{log_path}")

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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Anime Background Remover - F2_A1 → ViTMatte M6")
        self.resize(1120, 780)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
