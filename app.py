import os
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
    QListWidget,
)

from bg_engine import BgRemoveEngine


# =========================
# APP CONFIG
# =========================

F2_A1_ONNX_PATH = "models/f2a1_unet_mit_b1_768.onnx"
M6_ONNX_PATH = "models/m6_vitmatte_m6_768.onnx"

OUTPUT_ROOT = "outputs"
SINGLE_OUTPUT_DIR = str(Path(OUTPUT_ROOT) / "single_saved")
BATCH_OUTPUT_DIR = str(Path(OUTPUT_ROOT) / "batch_saved")

SAVE_DEBUG_OUTPUTS = True

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# =========================
# SMALL HELPERS
# =========================

def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def collect_image_files_from_paths(paths: list[str], recursive: bool = True) -> list[str]:
    """
    Input có thể là file hoặc folder do kéo-thả.
    - File ảnh: lấy trực tiếp.
    - Folder: lấy ảnh trong folder. Mặc định recursive để chọn folder batch thuận tiện.
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

    # Giữ thứ tự nhưng bỏ duplicate.
    seen = set()
    unique = []
    for p in found:
        key = str(Path(p).resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def pil_to_pixmap(pil_image):
    """
    Hiển thị PIL image lên QLabel mà không cần ghi file tạm.
    """
    rgba = pil_image.convert("RGBA")
    w, h = rgba.size
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


# =========================
# WORKERS
# =========================

class SingleCutWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_path: str):
        super().__init__()
        self.engine = engine
        self.image_path = image_path

    def run(self):
        try:
            result = self.engine.process_for_preview(self.image_path)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchCutWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_paths: list[str]):
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths

    def run(self):
        try:
            results = []
            total = len(self.image_paths)

            for idx, path in enumerate(self.image_paths, start=1):
                self.progress.emit(idx - 1, total, f"Đang cắt: {Path(path).name}")
                result = self.engine.process_for_preview(path)
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

class DropUploadArea(QFrame):
    clicked = Signal()
    filesDropped = Signal(list)

    def __init__(self, title: str, subtitle: str):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(280)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #8A8A8A;
                border-radius: 18px;
                background: #FAFAFA;
            }
            QFrame:hover {
                border: 2px dashed #3D7EFF;
                background: #F3F7FF;
            }
            QLabel {
                border: none;
                background: transparent;
            }
            """
        )

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: 700;")

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("font-size: 14px; color: #555555;")

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addStretch(1)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
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
            btn.setMinimumHeight(54)
            btn.setStyleSheet("font-size: 17px;")

        self.btn_video.setEnabled(False)
        self.btn_object.setEnabled(False)

        box = QVBoxLayout()
        box.setSpacing(14)
        box.addWidget(self.btn_single)
        box.addWidget(self.btn_batch)
        box.addWidget(self.btn_video)
        box.addWidget(self.btn_object)

        outer = QVBoxLayout()
        outer.addStretch(1)
        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addSpacing(30)
        outer.addLayout(box)
        outer.addStretch(2)

        wrapper = QHBoxLayout()
        wrapper.addStretch(1)
        wrapper.addLayout(outer, 2)
        wrapper.addStretch(1)

        self.setLayout(wrapper)

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

        self.btn_back = QPushButton("← Trang chủ")
        self.title = QLabel("Cắt nền 1 hình")
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.status_label = QLabel("Bấm vào vùng upload hoặc kéo thả ảnh vào giữa màn hình.")
        self.status_label.setAlignment(Qt.AlignCenter)

        top = QHBoxLayout()
        top.addWidget(self.btn_back)
        top.addStretch(1)
        top.addWidget(self.title)
        top.addStretch(2)

        self.upload_area = DropUploadArea(
            "Upload ảnh",
            "Bấm để chọn ảnh hoặc kéo-thả vào đây. Nếu kéo nhiều ảnh, app chỉ lấy ảnh đầu tiên.",
        )

        self.preview_label = QLabel("Preview sẽ hiện ở đây sau khi cắt.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(360)
        self.preview_label.setStyleSheet("border: 1px solid #CCCCCC; background: white;")

        self.btn_save = QPushButton("Lưu ảnh đã cắt")
        self.btn_open_output = QPushButton("Mở folder đã lưu")
        self.btn_save.setEnabled(False)
        self.btn_open_output.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        row = QHBoxLayout()
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_open_output)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.status_label)
        layout.addWidget(self.upload_area)
        layout.addWidget(self.preview_label)
        layout.addLayout(row)
        layout.addWidget(self.progress)

        self.setLayout(layout)

        self.btn_back.clicked.connect(self.backRequested.emit)
        self.upload_area.clicked.connect(self.select_image)
        self.upload_area.filesDropped.connect(self.handle_drop)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_open_output.clicked.connect(lambda: self.open_folder(SINGLE_OUTPUT_DIR))

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

        # Single mode: nếu lỡ thả nhiều ảnh thì chỉ lấy ảnh đầu tiên.
        self.start_cut(image_paths[0])

    def start_cut(self, image_path: str):
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.critical(self, "Model chưa sẵn sàng", "Pipeline chưa load xong.")
            return

        self.current_prediction = None
        self.btn_save.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.upload_area.setEnabled(False)
        self.status_label.setText(f"Đang tự động cắt: {image_path}")
        self.progress.setRange(0, 0)

        self.current_thread = QThread()
        self.current_worker = SingleCutWorker(engine, image_path)
        self.current_worker.moveToThread(self.current_thread)

        self.current_thread.started.connect(self.current_worker.run)
        self.current_worker.finished.connect(self.on_cut_finished)
        self.current_worker.error.connect(self.on_worker_error)

        self.current_worker.finished.connect(self.current_thread.quit)
        self.current_worker.error.connect(self.current_thread.quit)
        self.current_thread.finished.connect(self.current_thread.deleteLater)

        self.current_thread.start()

    def on_cut_finished(self, prediction: dict):
        self.current_prediction = prediction
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.upload_area.setEnabled(True)
        self.btn_save.setEnabled(True)

        pix = pil_to_pixmap(prediction["preview_checker_pil"])
        scaled = pix.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

        self.status_label.setText(
            f"Cắt xong. Chưa lưu. Time: {prediction['elapsed_sec']:.2f}s | "
            f"Ảnh: {Path(prediction['input_path']).name}"
        )

    def on_worker_error(self, err: str):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.upload_area.setEnabled(True)
        self.btn_save.setEnabled(False)
        QMessageBox.critical(self, "Lỗi", err)
        self.status_label.setText("Có lỗi xảy ra khi cắt ảnh.")

    def save_current(self):
        if not self.current_prediction:
            return

        engine = self.engine_getter()
        if engine is None:
            return

        saved = engine.save_prediction(
            self.current_prediction,
            SINGLE_OUTPUT_DIR,
            save_debug=SAVE_DEBUG_OUTPUTS,
        )

        self.btn_open_output.setEnabled(True)
        self.status_label.setText(f"Đã lưu: {saved['cutout_path']}")
        QMessageBox.information(self, "Đã lưu", f"Đã lưu ảnh đã cắt:\n{saved['cutout_path']}")

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

        self.cut_thread: QThread | None = None
        self.cut_worker: BatchCutWorker | None = None
        self.save_thread: QThread | None = None
        self.save_worker: BatchSaveWorker | None = None

        self.btn_back = QPushButton("← Trang chủ")
        self.title = QLabel("Cắt nhiều hình")
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.status_label = QLabel("Bấm vùng upload, kéo-thả nhiều ảnh, hoặc chọn folder.")
        self.status_label.setAlignment(Qt.AlignCenter)

        top = QHBoxLayout()
        top.addWidget(self.btn_back)
        top.addStretch(1)
        top.addWidget(self.title)
        top.addStretch(2)

        self.upload_area = DropUploadArea(
            "Upload nhiều ảnh",
            "Bấm để chọn nhiều ảnh hoặc kéo-thả file/folder vào đây.",
        )

        self.btn_select_folder = QPushButton("Chọn folder và cắt cả folder")
        self.btn_save_all = QPushButton("Lưu tất cả ảnh đã cắt")
        self.btn_open_output = QPushButton("Mở folder đã lưu")

        self.btn_save_all.setEnabled(False)
        self.btn_open_output.setEnabled(False)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(220)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_select_folder)
        buttons.addWidget(self.btn_save_all)
        buttons.addWidget(self.btn_open_output)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.status_label)
        layout.addWidget(self.upload_area)
        layout.addLayout(buttons)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.progress)

        self.setLayout(layout)

        self.btn_back.clicked.connect(self.backRequested.emit)
        self.upload_area.clicked.connect(self.select_images)
        self.upload_area.filesDropped.connect(self.handle_drop)
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_open_output.clicked.connect(lambda: self.open_folder(BATCH_OUTPUT_DIR))

    def select_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn nhiều ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if paths:
            self.start_batch(paths)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn folder ảnh")
        if not folder:
            return

        image_paths = collect_image_files_from_paths([folder], recursive=True)
        if not image_paths:
            QMessageBox.warning(self, "Không có ảnh", "Folder này không có ảnh hợp lệ.")
            return

        self.start_batch(image_paths)

    def handle_drop(self, paths: list[str]):
        image_paths = collect_image_files_from_paths(paths, recursive=True)
        if not image_paths:
            QMessageBox.warning(self, "Không có ảnh", "Không tìm thấy file ảnh hợp lệ.")
            return

        # Batch mode: kéo nhiều ảnh/folder thì cắt hết.
        self.start_batch(image_paths)

    def start_batch(self, image_paths: list[str]):
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.critical(self, "Model chưa sẵn sàng", "Pipeline chưa load xong.")
            return

        self.selected_paths = image_paths
        self.predictions = []

        self.list_widget.clear()
        for p in image_paths:
            self.list_widget.addItem(f"Chờ cắt: {Path(p).name}")

        self.btn_save_all.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.upload_area.setEnabled(False)
        self.btn_select_folder.setEnabled(False)

        self.progress.setRange(0, len(image_paths))
        self.progress.setValue(0)
        self.status_label.setText(f"Đang tự động cắt {len(image_paths)} ảnh...")

        self.cut_thread = QThread()
        self.cut_worker = BatchCutWorker(engine, image_paths)
        self.cut_worker.moveToThread(self.cut_thread)

        self.cut_thread.started.connect(self.cut_worker.run)
        self.cut_worker.progress.connect(self.on_cut_progress)
        self.cut_worker.finished.connect(self.on_batch_cut_finished)
        self.cut_worker.error.connect(self.on_worker_error)

        self.cut_worker.finished.connect(self.cut_thread.quit)
        self.cut_worker.error.connect(self.cut_thread.quit)
        self.cut_thread.finished.connect(self.cut_thread.deleteLater)

        self.cut_thread.start()

    def on_cut_progress(self, current: int, total: int, message: str):
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.status_label.setText(f"{message} ({current}/{total})")

    def on_batch_cut_finished(self, predictions: list[dict]):
        self.predictions = predictions

        self.list_widget.clear()
        for pred in predictions:
            self.list_widget.addItem(
                f"Đã cắt, chưa lưu: {Path(pred['input_path']).name} | "
                f"{pred['elapsed_sec']:.2f}s"
            )

        self.progress.setValue(len(predictions))
        self.upload_area.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        self.btn_save_all.setEnabled(len(predictions) > 0)
        self.status_label.setText(
            f"Cắt xong {len(predictions)} ảnh. Chưa lưu. Bấm 'Lưu tất cả ảnh đã cắt'."
        )

    def on_worker_error(self, err: str):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.upload_area.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        self.btn_save_all.setEnabled(False)
        QMessageBox.critical(self, "Lỗi", err)
        self.status_label.setText("Có lỗi xảy ra.")

    def save_all(self):
        if not self.predictions:
            return

        engine = self.engine_getter()
        if engine is None:
            return

        self.btn_save_all.setEnabled(False)
        self.upload_area.setEnabled(False)
        self.btn_select_folder.setEnabled(False)

        self.progress.setRange(0, len(self.predictions))
        self.progress.setValue(0)
        self.status_label.setText("Đang lưu tất cả ảnh đã cắt...")

        self.save_thread = QThread()
        self.save_worker = BatchSaveWorker(
            engine,
            self.predictions,
            BATCH_OUTPUT_DIR,
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
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.status_label.setText(f"{message} ({current}/{total})")

    def on_save_finished(self, log_path: str):
        self.upload_area.setEnabled(True)
        self.btn_select_folder.setEnabled(True)
        self.btn_save_all.setEnabled(True)
        self.btn_open_output.setEnabled(True)

        self.status_label.setText(f"Đã lưu batch. Log: {log_path}")
        QMessageBox.information(self, "Batch đã lưu", f"Đã lưu batch.\nLog:\n{log_path}")

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

        self.status_label = QLabel("Đang load pipeline F2_A1 → ViTMatte M6...")
        self.status_label.setAlignment(Qt.AlignCenter)

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
                f"Pipeline sẵn sàng: F2_A1 → M6 | Runtime: {self.engine.runtime_info}"
            )
            self.set_pages_enabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi load model", str(e))
            self.status_label.setText("Load pipeline thất bại.")
            self.set_pages_enabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
