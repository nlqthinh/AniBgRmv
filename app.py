import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from PySide6.QtCore import QObject, QThread, Signal, Qt, QRectF, QTimer
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QShortcut, QKeySequence, QGuiApplication
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
    QSlider,
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
    "sam": {
        "checkpoint_path": "models/sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "device": "auto",
        "guided_timeout_ms": 0,
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

SAM_CHECKPOINT_PATH = str(resolve_app_path(APP_CONFIG["sam"]["checkpoint_path"]))
SAM_MODEL_TYPE = str(APP_CONFIG["sam"].get("model_type", "vit_b"))
SAM_DEVICE = str(APP_CONFIG["sam"].get("device", "auto"))
GUIDED_EDIT_TIMEOUT_MS = int(APP_CONFIG.get("sam", {}).get("guided_timeout_ms", 5000))

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



EDIT_BLACK_BUTTON_STYLE = """
QPushButton {
    font-size: 14px;
    font-weight: 900;
    padding: 5px 14px;
    min-height: 30px;
    border-radius: 9px;
    border: 1px solid #FFFFFF;
    color: #FFFFFF;
    background: #000000;
}
QPushButton:hover {
    background: #151515;
}
QPushButton:checked {
    border: 2px solid #FFFFFF;
    background: #222222;
}
QPushButton:disabled {
    color: #999999;
    background: #222222;
    border: 1px solid #777777;
}
"""

EDIT_ERASE_BUTTON_STYLE = EDIT_BLACK_BUTTON_STYLE.replace("color: #FFFFFF;", "color: #FF3030;")
EDIT_RESTORE_BUTTON_STYLE = EDIT_BLACK_BUTTON_STYLE.replace("color: #FFFFFF;", "color: #20D060;")

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
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
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



class GuidedEditWorker(QObject):
    finished = Signal(dict)
    canceled = Signal()
    error = Signal(str)

    def __init__(
        self,
        engine: BgRemoveEngine,
        image_path: str,
        current_alpha: np.ndarray,
        stroke_points: list[tuple[float, float]],
        brush_size: int,
        operation: str,
        sam_checkpoint_path: str,
        sam_model_type: str,
        sam_device: str,
    ):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.current_alpha = current_alpha
        self.stroke_points = stroke_points
        self.brush_size = brush_size
        self.operation = operation
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_model_type = sam_model_type
        self.sam_device = sam_device
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if self.cancel_requested:
                self.canceled.emit()
                return

            result = self.engine.apply_sam_guided_edit(
                image_path=self.image_path,
                current_alpha=self.current_alpha,
                stroke_points=self.stroke_points,
                brush_size=self.brush_size,
                operation=self.operation,
                sam_checkpoint_path=self.sam_checkpoint_path,
                sam_model_type=self.sam_model_type,
                sam_device=self.sam_device,
                # M22: do not run ViTMatte on every guided stroke.
                # It made edit mode extremely slow on CPU.
                # Guided edit now uses SAM for local region selection, then applies alpha directly.
                refine_with_vitmatte=False,
            )

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
        self.setMinimumHeight(360)
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
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
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



class EditMaskCanvas(QFrame):
    strokeStarted = Signal()
    strokePoint = Signal(float, float)
    strokeFinished = Signal(list)

    def __init__(self):
        super().__init__()
        # M22: lower minimum height so bottom edit buttons do not get pushed
        # outside the window until the user maximizes it.
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet(
            """
            QFrame {
                border: 2px solid #111111;
                border-radius: 18px;
                background: #FFFFFF;
            }
            """
        )

        self.image_rgb: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.operation = "erase"
        self.brush_size = 28
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_drawing = False
        self.is_panning = False
        self.last_pan_pos = None
        self.stroke_points: list[tuple[float, float]] = []
        self.processing = False

        # M22 performance cache: do not rebuild PIL/QPixmap on every paint event.
        self._preview_cache: QPixmap | None = None
        self._hover_pos = None
        self._hover_inside = False

    def invalidate_preview_cache(self):
        self._preview_cache = None

    def set_image_alpha(self, image_path: str, alpha: np.ndarray):
        image = Image.open(image_path).convert("RGB")
        self.image_rgb = np.asarray(image)
        self.alpha = np.asarray(alpha).astype(np.uint8).copy()
        if self.alpha.shape[:2] != self.image_rgb.shape[:2]:
            self.alpha = cv2.resize(
                self.alpha,
                (self.image_rgb.shape[1], self.image_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        self.stroke_points = []
        self.invalidate_preview_cache()
        self.update()

    def set_alpha_array(self, alpha: np.ndarray, copy: bool = False):
        if self.image_rgb is None:
            return
        arr = np.asarray(alpha).astype(np.uint8)
        if arr.shape[:2] != self.image_rgb.shape[:2]:
            arr = cv2.resize(
                arr,
                (self.image_rgb.shape[1], self.image_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        self.alpha = arr.copy() if copy else arr
        self.invalidate_preview_cache()
        self.update()

    def set_operation(self, operation: str):
        self.operation = operation
        self.invalidate_preview_cache()
        self.update()

    def set_brush_size(self, brush_size: int):
        self.brush_size = max(1, int(brush_size))
        self.update()

    def set_processing(self, processing: bool):
        self.processing = processing
        self.update()

    def clear_stroke(self):
        self.stroke_points = []
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _base_scale(self) -> float:
        if self.image_rgb is None:
            return 1.0
        h, w = self.image_rgb.shape[:2]
        available_w = max(1, self.width() - 32)
        available_h = max(1, self.height() - 32)
        return min(available_w / w, available_h / h)

    def _image_rect(self):
        if self.image_rgb is None:
            return QRectF()
        h, w = self.image_rgb.shape[:2]
        scale = self._base_scale() * self.zoom
        draw_w = w * scale
        draw_h = h * scale
        x = (self.width() - draw_w) / 2.0 + self.pan_x
        y = (self.height() - draw_h) / 2.0 + self.pan_y
        return QRectF(x, y, draw_w, draw_h)

    def _widget_to_image(self, x: float, y: float):
        if self.image_rgb is None:
            return None
        rect = self._image_rect()
        scale = self._base_scale() * self.zoom
        if scale <= 0:
            return None
        ix = (x - rect.left()) / scale
        iy = (y - rect.top()) / scale
        h, w = self.image_rgb.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return float(ix), float(iy)

    def _image_to_widget(self, x: float, y: float):
        rect = self._image_rect()
        scale = self._base_scale() * self.zoom
        return rect.left() + x * scale, rect.top() + y * scale

    def _render_preview(self) -> Image.Image:
        assert self.image_rgb is not None and self.alpha is not None
        rgb = self.image_rgb.astype(np.float32)
        a = (self.alpha.astype(np.float32) / 255.0)[..., None]

        # Erase mode: background/cut area looks strongly white.
        # Restore mode: background is less white so missing parts can be seen.
        bg_visibility = 0.12 if self.operation == "erase" else 0.48
        visible = a + (1.0 - a) * bg_visibility
        white = np.full_like(rgb, 255.0)
        out = rgb * visible + white * (1.0 - visible)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")

    def _preview_pixmap(self) -> QPixmap:
        if self._preview_cache is None:
            pil = self._render_preview()
            self._preview_cache = pil_to_pixmap(pil)
        return self._preview_cache

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))

        if self.image_rgb is None or self.alpha is None:
            painter.setPen(QColor("#333333"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Không có ảnh để edit")
            return

        rect = self._image_rect()
        pix = self._preview_pixmap()
        painter.drawPixmap(rect, pix, QRectF(pix.rect()))

        if self.stroke_points:
            scale = self._base_scale() * self.zoom
            pen = QPen(QColor(255, 0, 0, 210))
            pen.setWidth(max(1, int(self.brush_size * scale)))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)

            if len(self.stroke_points) == 1:
                x, y = self._image_to_widget(*self.stroke_points[0])
                painter.drawPoint(int(x), int(y))
            else:
                for p1, p2 in zip(self.stroke_points[:-1], self.stroke_points[1:]):
                    x1, y1 = self._image_to_widget(*p1)
                    x2, y2 = self._image_to_widget(*p2)
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Red translucent brush cursor.
        if self._hover_inside and self._hover_pos is not None and not self.processing:
            point = self._widget_to_image(self._hover_pos.x(), self._hover_pos.y())
            if point is not None:
                scale = self._base_scale() * self.zoom
                r = max(2.0, (self.brush_size * scale) / 2.0)
                painter.setPen(QPen(QColor(255, 0, 0, 170), 2))
                painter.setBrush(QColor(255, 0, 0, 45))
                painter.drawEllipse(QRectF(self._hover_pos.x() - r, self._hover_pos.y() - r, r * 2, r * 2))

        mode_text = "ERASE" if self.operation == "erase" else "RESTORE"
        painter.setPen(QColor("black"))
        painter.drawText(18, 28, f"{mode_text} | zoom {self.zoom:.2f}x | brush {self.brush_size}px")

        if self.processing:
            painter.fillRect(self.rect(), QColor(255, 255, 255, 70))
            painter.setPen(QColor("black"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Đang xử lý SAM...")

    def enterEvent(self, event):
        self._hover_inside = True
        self.setCursor(Qt.BlankCursor)
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover_inside = False
        self._hover_pos = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._hover_pos = event.position()
        if self.image_rgb is None or self.alpha is None or self.processing:
            return

        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier):
            self.is_panning = True
            self.last_pan_pos = event.position()
            return

        if event.button() == Qt.LeftButton:
            point = self._widget_to_image(event.position().x(), event.position().y())
            if point is None:
                return
            self.is_drawing = True
            self.stroke_points = [point]
            self.strokeStarted.emit()
            self.strokePoint.emit(point[0], point[1])
            self.update()

    def mouseMoveEvent(self, event):
        self._hover_pos = event.position()

        if self.is_panning and self.last_pan_pos is not None:
            delta = event.position() - self.last_pan_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.last_pan_pos = event.position()
            self.update()
            return

        if not self.is_drawing or self.processing:
            self.update()
            return

        point = self._widget_to_image(event.position().x(), event.position().y())
        if point is None:
            self.update()
            return

        if not self.stroke_points:
            self.stroke_points.append(point)
        else:
            px, py = self.stroke_points[-1]
            if abs(point[0] - px) + abs(point[1] - py) >= 2:
                self.stroke_points.append(point)
                self.strokePoint.emit(point[0], point[1])
        self.update()

    def mouseReleaseEvent(self, event):
        self._hover_pos = event.position()
        if self.is_panning:
            self.is_panning = False
            self.last_pan_pos = None
            return

        if self.is_drawing and event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.strokeFinished.emit(list(self.stroke_points))
            self.update()

    def wheelEvent(self, event):
        if self.image_rgb is None:
            return

        cursor_pos = event.position()
        before = self._widget_to_image(cursor_pos.x(), cursor_pos.y())

        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom = min(8.0, self.zoom * 1.15)
        else:
            self.zoom = max(0.25, self.zoom / 1.15)

        # M23: zoom around cursor, not around image center.
        if before is not None:
            rect = self._image_rect()
            scale = self._base_scale() * self.zoom
            target_x = rect.left() + before[0] * scale
            target_y = rect.top() + before[1] * scale
            self.pan_x += cursor_pos.x() - target_x
            self.pan_y += cursor_pos.y() - target_y

        self.update()

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
            self.btn_save,
            self.btn_edit,
            self.btn_config,
            self.btn_open_output,
        ))

        self.edit_buttons_widget = QWidget()
        self.btn_erase = QPushButton("Erase")
        self.btn_restore = QPushButton("Restore")
        self.btn_exit_edit = QPushButton("Exit")
        self.btn_brush_size = QPushButton("Brush size")
        self.btn_edit_save = QPushButton("Lưu ảnh")
        self.btn_stop_guided = QPushButton("Dừng")
        self.btn_stop_guided.setVisible(False)

        self.btn_erase.setCheckable(True)
        self.btn_restore.setCheckable(True)
        self.btn_erase.setChecked(True)

        self.btn_erase.setStyleSheet(EDIT_ERASE_BUTTON_STYLE)
        self.btn_restore.setStyleSheet(EDIT_RESTORE_BUTTON_STYLE)
        for btn in [self.btn_exit_edit, self.btn_brush_size, self.btn_edit_save]:
            btn.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
        self.btn_stop_guided.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.btn_edit_save.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)

        for btn, width in [(self.btn_erase, 104), (self.btn_restore, 112), (self.btn_exit_edit, 92), (self.btn_brush_size, 132), (self.btn_edit_save, 104), (self.btn_stop_guided, 104)]:
            btn.setMinimumWidth(width)
            btn.setFixedHeight(42)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.edit_buttons_widget.setFixedHeight(58)

        edit_buttons_row = center_button_row(
            self.btn_erase,
            self.btn_restore,
            self.btn_exit_edit,
            self.btn_brush_size,
            self.btn_edit_save,
            self.btn_stop_guided,
        )
        edit_buttons_row.setContentsMargins(0, 4, 0, 6)
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
        self.btn_save.clicked.connect(self.save_current)
        self.btn_config.clicked.connect(self.choose_output_dir)
        self.btn_open_output.clicked.connect(lambda: self.open_folder(self.output_dir))
        self.btn_edit.clicked.connect(self.enter_edit_mode)

        self.btn_erase.clicked.connect(lambda: self.set_edit_operation("erase"))
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

    def enter_edit_mode(self):
        if not self.current_prediction:
            return

        self.in_edit_mode = True
        self.edit_start_prediction = self.clone_prediction_with_alpha(self.current_prediction, self.current_prediction["final_alpha"])
        self.edit_dirty = False
        self.edit_undo_stack = []
        self.edit_redo_stack = []
        self.edit_operation = "erase"
        self.edit_submode = "guided"

        self.canvas.setVisible(False)
        self.edit_canvas.setVisible(True)
        self.edit_top_widget.setVisible(True)
        self.normal_buttons_widget.setVisible(False)
        self.edit_buttons_widget.setVisible(True)
        self.path_label.setVisible(False)
        self.btn_back.setEnabled(True)
        self.instruction_label.setText("Edit: Erase/Restore, Guided/Manual. Giữ Shift + kéo chuột để pan, cuộn chuột để zoom.")

        self.btn_erase.setChecked(True)
        self.btn_restore.setChecked(False)
        self.btn_guided.setChecked(True)
        self.btn_manual.setChecked(False)
        self.set_brush_panel_visible(False)
        self.btn_stop_guided.setVisible(False)

        self.edit_canvas.set_operation("erase")
        self.edit_canvas.set_brush_size(self.brush_slider.value())
        self.edit_canvas.reset_view()
        self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        self.status_label.setText("Đang ở Edit mode. Guided dùng SAM để chọn vùng nhanh; Manual sửa alpha trực tiếp.")
        self.refresh_single_layout()

    def force_exit_edit(self, discard: bool = False):
        if discard and self.edit_start_prediction is not None:
            self.current_prediction = self.edit_start_prediction
        self.in_edit_mode = False
        self.edit_start_prediction = None
        self.edit_dirty = False
        self.edit_undo_stack = []
        self.edit_redo_stack = []
        self.edit_canvas.setVisible(False)
        self.canvas.setVisible(True)
        self.edit_top_widget.setVisible(False)
        self.normal_buttons_widget.setVisible(True)
        self.edit_buttons_widget.setVisible(False)
        self.btn_stop_guided.setVisible(False)
        self.path_label.setVisible(True)
        self.set_brush_panel_visible(False)
        self.btn_back.setEnabled(True)
        self.instruction_label.setText("Kéo-thả ảnh vào đây hoặc bấm vào khung để chọn ảnh.")
        if self.current_prediction:
            self.canvas.set_preview_pil(self.current_prediction["preview_checker_pil"])
            self.btn_save.setEnabled(True)
            self.btn_edit.setEnabled(True)
        self.refresh_single_layout()

    def handle_back_clicked(self):
        if self.in_edit_mode:
            self.exit_edit_mode_confirm(go_home=True)
        else:
            self.backRequested.emit()

    def refresh_single_layout(self):
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.updateGeometry()
        self.repaint()

    def set_brush_panel_visible(self, visible: bool):
        self.brush_panel_widget.setVisible(bool(visible))
        self.refresh_single_layout()

    def exit_edit_mode_confirm(self, go_home: bool = False):
        if not self.in_edit_mode:
            if go_home:
                self.backRequested.emit()
            return

        if self.edit_canvas.processing:
            QMessageBox.information(self, "Đang xử lý", "Vui lòng chờ Guided Edit xử lý xong rồi hãy thoát.")
            return

        # No edit happened -> exit directly, no confirmation dialog.
        if not self.edit_dirty:
            self.force_exit_edit(discard=False)
            self.status_label.setText("Đã thoát Edit. Không có thay đổi mask.")
            if go_home:
                self.backRequested.emit()
            return

        box = QMessageBox(self)
        box.setWindowTitle("Xác nhận lưu mask")
        box.setIcon(QMessageBox.Question)
        box.setText("Bạn có muốn giữ mask đã chỉnh sửa không?")
        btn_keep = box.addButton("Giữ mask", QMessageBox.AcceptRole)
        btn_discard = box.addButton("Không giữ", QMessageBox.DestructiveRole)
        box.addButton("Hủy", QMessageBox.RejectRole)
        box.exec()

        clicked = box.clickedButton()
        if clicked is btn_keep:
            self.force_exit_edit(discard=False)
            self.status_label.setText("Đã giữ mask chỉnh sửa. Bạn có thể bấm Lưu ảnh.")
            if go_home:
                self.backRequested.emit()
        elif clicked is btn_discard:
            self.force_exit_edit(discard=True)
            self.status_label.setText("Đã thoát Edit và bỏ thay đổi mask.")
            if go_home:
                self.backRequested.emit()
        else:
            self.status_label.setText("Tiếp tục Edit.")

    def clone_prediction_with_alpha(self, prediction: dict, alpha: np.ndarray) -> dict:
        engine = self.engine_getter()
        if engine is None:
            return dict(prediction)
        return engine.build_prediction_from_alpha(
            image_path=prediction["input_path"],
            final_alpha=np.asarray(alpha).astype(np.uint8),
            elapsed_sec=prediction.get("elapsed_sec", 0.0),
            source="ui_edit_clone",
        )

    def refresh_edit_canvas(self):
        if not self.current_prediction:
            return
        self.edit_canvas.set_image_alpha(
            self.current_prediction["input_path"],
            self.current_prediction["final_alpha"],
        )

    def set_edit_operation(self, operation: str):
        self.edit_operation = operation
        self.btn_erase.setChecked(operation == "erase")
        self.btn_restore.setChecked(operation == "restore")
        self.edit_canvas.set_operation(operation)
        if operation == "erase":
            self.status_label.setText("Erase: vùng nền sẽ trắng mạnh hơn để dễ thấy phần cần xóa.")
        else:
            self.status_label.setText("Restore: nền bị cắt bớt trắng lại để dễ thấy phần cần phục hồi.")

    def set_edit_submode(self, submode: str):
        self.edit_submode = submode
        self.btn_guided.setChecked(submode == "guided")
        self.btn_manual.setChecked(submode == "manual")
        self.status_label.setText("Guided: SAM chọn vùng nhanh, không chạy ViTMatte mỗi nét để tránh lag." if submode == "guided" else "Manual: vẽ trực tiếp lên alpha, không chạy SAM.")

    def toggle_brush_slider(self):
        self.set_brush_panel_visible(not self.brush_panel_widget.isVisible())

    def on_brush_size_changed(self, value: int):
        self.edit_canvas.set_brush_size(value)
        self.status_label.setText(f"Brush size: {value}px")

    def push_undo_state(self):
        if not self.current_prediction:
            return
        self.edit_undo_stack.append(np.asarray(self.current_prediction["final_alpha"]).copy())
        if len(self.edit_undo_stack) > 30:
            self.edit_undo_stack.pop(0)
        self.edit_redo_stack.clear()
        self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        self.btn_undo.setEnabled(len(self.edit_undo_stack) > 0)
        self.btn_redo.setEnabled(len(self.edit_redo_stack) > 0)

    def undo_edit(self):
        if not self.in_edit_mode or not self.current_prediction or not self.edit_undo_stack:
            return
        self.edit_redo_stack.append(np.asarray(self.current_prediction["final_alpha"]).copy())
        alpha = self.edit_undo_stack.pop()
        self.current_prediction = self.clone_prediction_with_alpha(self.current_prediction, alpha)
        self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        self.status_label.setText("Undo mask edit.")

    def redo_edit(self):
        if not self.in_edit_mode or not self.current_prediction or not self.edit_redo_stack:
            return
        self.edit_undo_stack.append(np.asarray(self.current_prediction["final_alpha"]).copy())
        alpha = self.edit_redo_stack.pop()
        self.current_prediction = self.clone_prediction_with_alpha(self.current_prediction, alpha)
        self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        self.status_label.setText("Redo mask edit.")

    def on_edit_stroke_started(self):
        if not self.current_prediction:
            return
        self.edit_stroke_active = True
        self.push_undo_state()

    def on_edit_stroke_point(self, x: float, y: float):
        if self.edit_submode != "manual" or not self.current_prediction:
            return
        self.apply_manual_point_to_alpha(x, y)

    def on_edit_stroke_finished(self, points: list):
        self.edit_stroke_active = False
        if not self.current_prediction or not points:
            return

        if self.edit_submode == "manual":
            self.edit_dirty = True
            self.current_prediction = self.clone_prediction_with_alpha(
                self.current_prediction,
                self.current_prediction["final_alpha"],
            )
            self.refresh_edit_canvas()
            self.status_label.setText("Manual edit đã áp dụng.")
            return

        self.start_guided_edit(points)

    def apply_manual_point_to_alpha(self, x: float, y: float):
        if not self.current_prediction:
            return
        # M22: mutate current alpha in-place. The previous version copied the
        # full alpha array on every mouse-move event, which made brush drawing lag.
        alpha = np.asarray(self.current_prediction["final_alpha"])
        radius = max(1, int(self.brush_slider.value() // 2))
        mask = np.zeros(alpha.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1, lineType=cv2.LINE_AA)
        if self.edit_operation == "erase":
            alpha[mask > 0] = 0
        else:
            alpha[mask > 0] = 255
        self.current_prediction["final_alpha"] = alpha
        self.edit_canvas.set_alpha_array(alpha, copy=False)

    def set_edit_controls_enabled(self, enabled: bool):
        for btn in [self.btn_erase, self.btn_restore, self.btn_exit_edit, self.btn_brush_size, self.btn_edit_save, self.btn_guided, self.btn_manual, self.btn_close_brush]:
            btn.setEnabled(enabled)
        self.brush_slider.setEnabled(enabled)
        self.btn_stop_guided.setVisible(not enabled)
        self.btn_stop_guided.setEnabled(not enabled)
        if not enabled:
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)
        else:
            self.update_undo_redo_buttons()
        self.refresh_single_layout()

    def is_guided_thread_running(self) -> bool:
        """
        Safe QThread status check.
        PySide can delete the C++ QThread object before Python reference becomes None.
        Calling isRunning() on that stale object raises RuntimeError.
        """
        if self.edit_thread is None:
            return False
        try:
            return self.edit_thread.isRunning()
        except RuntimeError:
            self.edit_thread = None
            self.edit_worker = None
            return False

    def on_guided_thread_finished_cleanup(self):
        self.edit_thread = None
        self.edit_worker = None

    def start_guided_edit(self, points: list):
        engine = self.engine_getter()
        if engine is None or not self.current_prediction:
            self.edit_canvas.clear_stroke()
            return

        if self.is_guided_thread_running():
            self.edit_canvas.clear_stroke()
            self.status_label.setText("Guided Edit trước vẫn đang xử lý. Hãy bấm Dừng hoặc chờ xong rồi bôi tiếp.")
            return

        self.guided_ignore_result = False
        self.guided_running = True
        self.set_edit_controls_enabled(False)
        self.edit_canvas.set_processing(True)
        if GUIDED_EDIT_TIMEOUT_MS > 0:
            self.status_label.setText(f"Đang chạy Guided Edit bằng SAM... tối đa {GUIDED_EDIT_TIMEOUT_MS / 1000:.0f}s")
            self.guided_timeout_timer.start(GUIDED_EDIT_TIMEOUT_MS)
        else:
            self.status_label.setText("Đang chạy Guided Edit bằng SAM... Lần đầu trên ảnh này có thể chậm, các lần sau sẽ nhanh hơn do cache embedding.")
            self.guided_timeout_timer.stop()
        QApplication.processEvents()

        self.edit_thread = QThread()
        self.edit_worker = GuidedEditWorker(
            engine=engine,
            image_path=self.current_prediction["input_path"],
            current_alpha=np.asarray(self.current_prediction["final_alpha"]).copy(),
            stroke_points=[(float(x), float(y)) for x, y in points],
            brush_size=int(self.brush_slider.value()),
            operation=self.edit_operation,
            sam_checkpoint_path=SAM_CHECKPOINT_PATH,
            sam_model_type=SAM_MODEL_TYPE,
            sam_device=SAM_DEVICE,
        )
        self.edit_worker.moveToThread(self.edit_thread)

        self.edit_thread.started.connect(self.edit_worker.run)
        self.edit_worker.finished.connect(self.on_guided_edit_finished)
        self.edit_worker.canceled.connect(self.on_guided_edit_canceled)
        self.edit_worker.error.connect(self.on_guided_edit_error)
        self.edit_worker.finished.connect(self.edit_thread.quit)
        self.edit_worker.canceled.connect(self.edit_thread.quit)
        self.edit_worker.error.connect(self.edit_thread.quit)
        self.edit_thread.finished.connect(self.on_guided_thread_finished_cleanup)
        self.edit_thread.finished.connect(self.edit_thread.deleteLater)
        self.edit_thread.start()

    def cancel_guided_edit(self):
        if not self.guided_running and not self.is_guided_thread_running():
            return
        self.guided_ignore_result = True
        self.guided_running = False
        self.guided_timeout_timer.stop()
        if self.edit_worker is not None:
            self.edit_worker.cancel()
        self.edit_canvas.set_processing(False)
        self.edit_canvas.clear_stroke()
        self.set_edit_controls_enabled(True)
        self.status_label.setText("Đã dừng chờ Guided Edit. Nếu SAM trả kết quả trễ, app sẽ bỏ qua kết quả đó.")

    def on_guided_edit_timeout(self):
        if not self.guided_running:
            return
        self.guided_ignore_result = True
        self.guided_running = False
        if self.edit_worker is not None:
            self.edit_worker.cancel()
        self.edit_canvas.set_processing(False)
        self.edit_canvas.clear_stroke()
        self.set_edit_controls_enabled(True)
        self.status_label.setText("Guided Edit quá thời gian cấu hình nên đã bỏ qua. Bạn có thể dùng Manual hoặc bấm Guided lại ở vùng nhỏ hơn.")

    def on_guided_edit_canceled(self):
        self.guided_timeout_timer.stop()
        self.guided_running = False
        was_ignored = self.guided_ignore_result
        self.guided_ignore_result = False
        self.edit_canvas.set_processing(False)
        self.edit_canvas.clear_stroke()
        self.set_edit_controls_enabled(True)
        self.status_label.setText("Guided Edit đã dừng." if not was_ignored else "Guided Edit đã bị bỏ qua theo yêu cầu dừng.")

    def on_guided_edit_finished(self, prediction: dict):
        self.guided_timeout_timer.stop()
        self.guided_running = False

        if self.guided_ignore_result or not self.in_edit_mode:
            self.guided_ignore_result = False
            self.edit_canvas.set_processing(False)
            self.edit_canvas.clear_stroke()
            self.set_edit_controls_enabled(True)
            self.status_label.setText("SAM đã trả kết quả sau khi bị dừng, app đã bỏ qua kết quả đó.")
            return

        self.current_prediction = prediction
        self.edit_dirty = True
        self.edit_canvas.set_processing(False)
        self.edit_canvas.clear_stroke()
        self.refresh_edit_canvas()
        self.set_edit_controls_enabled(True)
        self.update_undo_redo_buttons()
        self.status_label.setText(f"Guided {self.edit_operation} xong. Time: {prediction.get('elapsed_sec', 0):.2f}s")
        self.refresh_single_layout()

    def on_guided_edit_error(self, err: str):
        self.guided_timeout_timer.stop()
        self.guided_running = False
        self.guided_ignore_result = False
        self.edit_canvas.set_processing(False)
        self.edit_canvas.clear_stroke()
        self.set_edit_controls_enabled(True)
        if self.edit_undo_stack and self.current_prediction:
            alpha = self.edit_undo_stack.pop()
            self.current_prediction = self.clone_prediction_with_alpha(self.current_prediction, alpha)
            self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        QMessageBox.critical(self, "Lỗi Guided Edit", err)
        self.status_label.setText("Guided Edit lỗi. Đã rollback về mask trước đó nếu có.")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
