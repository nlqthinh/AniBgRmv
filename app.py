import sys
from pathlib import Path
import traceback
import pandas as pd

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtGui import QPixmap
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
)

from bg_engine import BgRemoveEngine


# M9 default pipeline:
# image -> F2_A1 ONNX -> loose trimap -> ViTMatte M6 ONNX -> final alpha -> cutout PNG
F2_A1_ONNX_PATH = "models/f2a1_unet_mit_b1_768.onnx"
M6_ONNX_PATH = "models/m6_vitmatte_m6_768.onnx"
OUTPUT_DIR = "outputs"


class SingleImageWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, engine, image_path, output_dir):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.output_dir = output_dir

    def run(self):
        try:
            result = self.engine.process_one(self.image_path, self.output_dir)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, engine, image_paths, output_dir):
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths
        self.output_dir = output_dir

    def run(self):
        try:
            logs = []
            total = len(self.image_paths)

            for idx, path in enumerate(self.image_paths, start=1):
                result = self.engine.process_one(path, self.output_dir)
                logs.append(result)
                self.progress.emit(idx, total)

            log_path = Path(self.output_dir) / "batch_log.csv"
            pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")

            self.finished.emit(str(log_path))
        except Exception:
            self.error.emit(traceback.format_exc())


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Anime Background Remover - F2_A1 → ViTMatte M6")
        self.resize(1000, 700)

        self.engine = None
        self.current_image_path = None
        self.current_result = None

        self.status_label = QLabel("Đang load model...")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("Chưa chọn ảnh")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(480)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.btn_select = QPushButton("Chọn ảnh")
        self.btn_cut = QPushButton("Tự động cắt nền")
        self.btn_open_output = QPushButton("Mở folder output")
        self.btn_batch = QPushButton("Batch nhiều ảnh")

        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.btn_cut.setEnabled(False)
        self.btn_open_output.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.btn_select.setEnabled(False)

        row = QHBoxLayout()
        row.addWidget(self.btn_select)
        row.addWidget(self.btn_cut)
        row.addWidget(self.btn_open_output)
        row.addWidget(self.btn_batch)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.image_label)
        layout.addLayout(row)
        layout.addWidget(self.progress)

        self.setLayout(layout)

        self.btn_select.clicked.connect(self.select_image)
        self.btn_cut.clicked.connect(self.cut_current_image)
        self.btn_open_output.clicked.connect(self.open_output_folder)
        self.btn_batch.clicked.connect(self.batch_images)

        self.load_model()

    def load_model(self):
        try:
            self.engine = BgRemoveEngine(
                f2a1_model_path=F2_A1_ONNX_PATH,
                m6_model_path=M6_ONNX_PATH,
                size=768,
            )
            self.status_label.setText(
                f"Pipeline đã sẵn sàng: F2_A1 → M6 | Runtime: {self.engine.runtime_info}"
            )
            self.btn_select.setEnabled(True)
            self.btn_batch.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi load model", str(e))
            self.status_label.setText("Load model thất bại.")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )

        if not path:
            return

        self.current_image_path = path
        self.current_result = None
        self.btn_cut.setEnabled(True)
        self.btn_open_output.setEnabled(False)

        self.show_image(path)
        self.status_label.setText(f"Đã chọn: {path}")

    def show_image(self, path):
        pix = QPixmap(path)
        if pix.isNull():
            self.image_label.setText("Không đọc được ảnh.")
            return

        scaled = pix.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def cut_current_image(self):
        if not self.current_image_path:
            return

        self.btn_cut.setEnabled(False)
        self.status_label.setText("Đang cắt nền bằng F2_A1 → ViTMatte M6...")
        self.progress.setRange(0, 0)

        self.thread = QThread()
        self.worker = SingleImageWorker(
            self.engine,
            self.current_image_path,
            OUTPUT_DIR,
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_single_finished)
        self.worker.error.connect(self.on_worker_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_single_finished(self, result):
        self.current_result = result
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        self.btn_cut.setEnabled(True)
        self.btn_open_output.setEnabled(True)

        preview_path = result["preview_checker_path"]
        self.show_image(preview_path)

        self.status_label.setText(
            f"Xong. Time: {result['elapsed_sec']:.2f}s | Cutout: {result['cutout_path']}"
        )

    def on_worker_error(self, err):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.btn_cut.setEnabled(True)
        self.btn_batch.setEnabled(True)
        QMessageBox.critical(self, "Lỗi", err)
        self.status_label.setText("Có lỗi xảy ra.")

    def open_output_folder(self):
        output = Path(OUTPUT_DIR).resolve()
        output.mkdir(exist_ok=True)
        import os
        os.startfile(output)

    def batch_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn nhiều ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )

        if not paths:
            return

        self.btn_batch.setEnabled(False)
        self.btn_cut.setEnabled(False)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.status_label.setText(f"Đang batch {len(paths)} ảnh bằng F2_A1 → M6...")

        batch_output = str(Path(OUTPUT_DIR) / "batch")

        self.batch_thread = QThread()
        self.batch_worker = BatchWorker(
            self.engine,
            paths,
            batch_output,
        )

        self.batch_worker.moveToThread(self.batch_thread)
        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.error.connect(self.on_worker_error)

        self.batch_worker.finished.connect(self.batch_thread.quit)
        self.batch_worker.error.connect(self.batch_thread.quit)
        self.batch_thread.finished.connect(self.batch_thread.deleteLater)

        self.batch_thread.start()

    def on_batch_progress(self, current, total):
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.status_label.setText(f"Đang batch: {current}/{total}")

    def on_batch_finished(self, log_path):
        self.btn_batch.setEnabled(True)
        self.btn_cut.setEnabled(self.current_image_path is not None)
        self.status_label.setText(f"Batch xong. Log: {log_path}")
        QMessageBox.information(self, "Batch xong", f"Đã lưu log:\n{log_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
