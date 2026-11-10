"""Home-style CPU/GPU settings screen. All GPU selections are strict."""
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from pathlib import Path

from ..compute_selection import current_selection
from ..device_management import GpuScanThread
from ..model_settings import current_models, default_models, stored_model, validate_import
from ..qt_threads import thread_is_valid
from ..portable_device_settings import build_variant
from ..styles import CANCEL_BUTTON_STYLE, apply_button_style

CPU_EDITION_NOTE = 'CPU Edition: dùng HQ-SAM trên CPU (cần encoder/decoder ONNX).'
# Video rows that pick an option, not a model file (no "Import..."): (text, value), tooltip.
VIDEO_OPTIONS = {
    'video_export_background': (
        (('Nền xanh lá', 'green'), ('Nền xanh biển', 'blue'), ('PNG trong suốt', 'transparent')),
        'MP4 nền màu, hoặc thư mục PNG trong suốt'),
    'video_track_keyframes': (
        (('Nhớ mọi keyframe', 0), ('8 keyframe gần nhất', 8), ('4 keyframe gần nhất', 4),
         ('2 keyframe gần nhất', 2)),
        'Khi Track, mỗi frame nhìn lại bao nhiêu keyframe. Ít hơn = nhanh hơn khi có nhiều '
        'keyframe; track_report.json ghi mask khác lần Track trước bao nhiêu.'),
    'video_held_frames': (
        (('Giữ hình: dùng chung', True), ('Tính riêng từng frame', False)),
        'Frame lặp lại hình của frame trước quanh viền object dùng chung mask (sau Track) '
        'và alpha (khi Export), để tóc không nhấp nháy.'),
}


class NoWheelComboBox(QComboBox):
    """Prevent accidental selection changes while scrolling the Settings page."""

    def wheelEvent(self, event):
        event.ignore()  # Let the surrounding scroll area handle the wheel.


class ModelImportWorker(QObject):
    completed = Signal()
    failed = Signal(str)

    def __init__(self, role, value):
        super().__init__()
        self.role = role
        self.value = value

    @Slot()
    def run(self):
        try:
            validate_import(self.role, self.value)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit()


def model_label(value):
    if isinstance(value, (tuple, list)):
        return ' + '.join(Path(str(part)).name for part in value)
    return Path(str(value)).name



class ComputeStatusPage(QWidget):
    backRequested = Signal()
    selectionRequested = Signal(dict)
    modelsRequested = Signal(dict)

    def __init__(self):
        super().__init__()
        self.selection = current_selection()
        self.scan_thread = None
        self.last_found = []
        self.is_applying = False
        self.import_thread = None
        self.import_worker = None
        self.models = current_models()
        self._model_combos = {}

        title = QLabel("Trạng thái CPU/GPU")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: 800; color: #F5F7FA;")
        self.status = QLabel("Đang kiểm tra trạng thái xử lý...")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 16px; color: #F5F7FA;")

        self.mode_box = NoWheelComboBox()
        self.mode_box.addItem("Sử dụng CPU", "cpu")
        self.mode_box.addItem("Sử dụng GPU", "gpu")
        if build_variant() == 'cpu':
            self.mode_box.model().item(1).setEnabled(False)
        self.mode_box.setFixedWidth(260)
        self.mode_box.setMinimumHeight(46)
        self.mode_box.setStyleSheet("font-size: 15px; padding: 6px; color: #F5F7FA; background-color: #252E3B; border: 1px solid #8A96A7; border-radius: 8px;")

        self.btn_check = QPushButton("GPU Check")
        self.btn_stop = QPushButton("Dừng")
        self.btn_exit = QPushButton("Exit")
        for btn in (self.btn_check, self.btn_stop, self.btn_exit):
            btn.setMinimumWidth(125)
            apply_button_style(btn)
        self.btn_stop.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.btn_stop.setVisible(False)

        self.gpu_box = NoWheelComboBox()
        self.gpu_box.setFixedWidth(260)
        self.gpu_box.setMinimumHeight(46)
        self.gpu_box.setStyleSheet("font-size: 15px; padding: 6px; color: #F5F7FA; background-color: #252E3B; border: 1px solid #8A96A7; border-radius: 8px;")
        self.gpu_box.setVisible(False)
        self.check_status = QLabel(
            CPU_EDITION_NOTE
            if build_variant() == 'cpu' else ''
        )
        self.check_status.setAlignment(Qt.AlignCenter)
        self.check_status.setWordWrap(True)
        self.check_status.setStyleSheet("font-size: 13px; color: #D6DEE8;")

        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(12)
        gpu_row.addStretch(1)
        gpu_row.addWidget(self.mode_box)
        gpu_row.addWidget(self.btn_check)
        gpu_row.addWidget(self.btn_stop)
        gpu_row.addStretch(1)
        self.gpu_row_widget = QWidget()
        self.gpu_row_widget.setLayout(gpu_row)

        self.gpu_label = QLabel('Chọn GPU')
        self.gpu_label.setAlignment(Qt.AlignCenter)
        self.gpu_label.setStyleSheet('color: white; font-size: 14px; font-weight: 600;')
        body = QVBoxLayout()
        body.setSpacing(14)
        body.addWidget(self.gpu_row_widget)
        body.addWidget(self.check_status)
        body.addWidget(self.gpu_label)
        body.addWidget(self.gpu_box, alignment=Qt.AlignCenter)
        self.gpu_spacing = QWidget()
        self.gpu_spacing.setFixedHeight(20)
        body.addWidget(self.gpu_spacing)
        self.model_title = QLabel('Model Setting')
        self.model_title.setAlignment(Qt.AlignCenter)
        self.model_title.setStyleSheet('font-size: 24px; font-weight: 750; color: #F5F7FA;')
        body.addWidget(self.model_title)
        body.addWidget(self._create_model_table(), alignment=Qt.AlignHCenter)
        self.model_feedback = QLabel('Chọn model từ JSON hoặc Import. Nhấn Áp dụng để lưu.')
        self.model_feedback.setWordWrap(True)
        self.model_feedback.setAlignment(Qt.AlignCenter)
        self.model_feedback.setStyleSheet('color: #D6DEE8; font-size: 12px;')
        body.addWidget(self.model_feedback)
        self.btn_apply = QPushButton('Áp dụng Model Setting')
        apply_button_style(self.btn_apply)
        body.addWidget(self.btn_apply, alignment=Qt.AlignCenter)
        body.addWidget(self.btn_exit, alignment=Qt.AlignCenter)

        content = QWidget()
        container = QVBoxLayout(content)
        container.setContentsMargins(18, 12, 18, 12)
        container.addWidget(title)
        container.addWidget(self.status)
        container.addSpacing(10)
        container.addLayout(body)
        container.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.gpu_box.currentIndexChanged.connect(self._gpu_changed)
        self.btn_check.clicked.connect(self._start_scan)
        self.btn_stop.clicked.connect(self._stop_scan)
        self.btn_exit.clicked.connect(self.backRequested.emit)
        self.btn_apply.clicked.connect(self._apply_models)
        self.set_selection(self.selection)
        self.set_models(self.models)

    def _create_model_table(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(9)

        def header(row, values):
            for column, text in enumerate(values):
                if not text:
                    continue
                label = QLabel(text)
                label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter if column == 0
                                   else Qt.AlignCenter)
                label.setWordWrap(True)
                label.setStyleSheet('font-size: 13px; font-weight: 700; color: white;')
                grid.addWidget(label, row, column)

        def combo(row, column, key):
            widget = NoWheelComboBox()
            widget.setMinimumHeight(34)
            widget.setFixedWidth(166)
            widget.setStyleSheet('font-size: 12px; padding: 4px; color: white; background: #252E3B; border: 1px solid #8A96A7; border-radius: 6px;')
            self._model_combos[key] = widget
            widget.currentIndexChanged.connect(
                lambda idx, role=key: self._model_changed(role, idx))
            grid.addWidget(widget, row, column)

        header(0, ['Loại ảnh', 'Cắt nền ảnh', 'Tinh chỉnh viền', 'Chọn vùng', 'Tinh chỉnh viền (vùng chọn)'])
        for row, name in ((1, 'Cắt hình đơn'), (2, 'Cắt nhiều hình')):
            label = QLabel(name)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setStyleSheet('color: white; font-weight: 600;')
            grid.addWidget(label, row, 0)
        combo(1, 1, 'single_auto')
        combo(1, 2, 'single_vit')
        combo(1, 3, 'single_sam')
        combo(1, 4, 'single_sam_vit')
        combo(2, 1, 'batch_auto')
        combo(2, 2, 'batch_vit')
        
        header(3, ['Loại video', 'Cắt nền video', 'Tinh chỉnh viền (vùng chọn)', 'Chọn vùng', ''])
        label = QLabel('Cắt nền video')
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setStyleSheet('color: white; font-weight: 600;')
        grid.addWidget(label, 4, 0)
        combo(4, 1, 'video_backend')
        combo(4, 2, 'video_sam_vit')
        combo(4, 3, 'video_sam')
        export_label = QLabel('Tùy chọn xuất video:')
        export_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        export_label.setStyleSheet('color: white; font-weight: 600;')
        grid.addWidget(export_label, 5, 0)
        combo(5, 1, 'video_export_background')
        track_label = QLabel('Tùy chọn Track video:')
        track_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        track_label.setStyleSheet('color: white; font-weight: 600;')
        grid.addWidget(track_label, 6, 0)
        combo(6, 1, 'video_track_keyframes')
        combo(6, 2, 'video_held_frames')
        future = QLabel('Xóa object (tương lai)')
        future.setStyleSheet('font-size: 14px; color: #8A96A7;')
        grid.addWidget(future, 7, 0, 1, 5, Qt.AlignCenter)
        frame = QWidget()
        frame.setLayout(grid)
        return frame

    def set_models(self, models):
        self.models = dict(models)
        defaults = default_models()
        for role, box in self._model_combos.items():
            value = self.models[role]
            # OFF preserves the selected checkpoint on disk, but must not create
            # two confusing entries ("Đã lưu" and "Mặc định") for the same file.
            stored = stored_model(role) if value is None else value
            box.blockSignals(True)
            box.clear()
            if role == 'video_backend':
                box.addItem('HQ-SAM2', 'hqsam2')
            elif role in VIDEO_OPTIONS:
                for text, data in VIDEO_OPTIONS[role][0]:
                    box.addItem(text, data)
            elif role == 'video_sam_vit':
                _, path = value
                box.addItem('OFF', ('binary', path))
                box.addItem('M71 (Auto: GPU)', ('auto', path))
                box.addItem('M71 (ON)', ('vitmatte_sam', path))
            else:
                if role in ('single_vit', 'batch_vit', 'single_sam_vit'):
                    box.addItem('OFF', None)
                if stored is not None:
                    box.addItem('Đang dùng: ' + model_label(stored), stored)
                if defaults[role] != stored:
                    box.addItem('Mặc định: ' + model_label(defaults[role]), defaults[role])
            if role not in VIDEO_OPTIONS:
                box.addItem('Import...', '__import__')
            for index in range(box.count()):
                if box.itemData(index) == value:
                    box.setCurrentIndex(index)
                    break
            box.blockSignals(False)
            box.setToolTip(VIDEO_OPTIONS[role][1] if role in VIDEO_OPTIONS
                           else model_label(stored) if stored is not None else 'OFF')

    def _model_changed(self, role, index):
        box = self._model_combos[role]
        value = box.itemData(index)
        if value == '__import__':
            self._import_model(role)
            return
        self.models[role] = value

    def _import_model(self, role):
        actual = role
        before = dict(self.models)
        self.set_models(before)  # Reset the Import... entry pending validation.
        if actual == 'video_backend':
            QMessageBox.warning(self, 'Backend chưa hỗ trợ',
                                'Bản app hiện chỉ chạy HQ-SAM2 Video Predictor. '
                                'Không thể import EfficientTAM hoặc backend video khác trước khi tích hợp.')
            return
        if actual in ('single_sam', 'video_sam'):
            ext = 'ONNX (*.onnx)' if actual == 'single_sam' else 'PyTorch (*.pt *.pth)'
            first, _ = QFileDialog.getOpenFileName(self, 'Chọn encoder/base', '', ext)
            if not first:
                return
            second, _ = QFileDialog.getOpenFileName(self, 'Chọn decoder', str(Path(first).parent), ext)
            if not second:
                return
            imported = (first, second)
        else:
            imported, _ = QFileDialog.getOpenFileName(self, 'Import model ONNX', '', 'ONNX (*.onnx)')
            if not imported:
                return
        if self.import_thread is not None:
            return
        self.model_feedback.setText('Đang kiểm tra model import và chữ ký input/output...')
        self.btn_apply.setEnabled(False)
        self.import_thread = QThread(self)
        self.import_worker = ModelImportWorker(actual, imported)
        self.import_worker.moveToThread(self.import_thread)
        self.import_thread.started.connect(self.import_worker.run)

        def complete():
            if actual == 'video_sam_vit':
                self.models[actual] = ('vitmatte_sam', imported)
            else:
                self.models[actual] = imported
            self.set_models(self.models)
            self.model_feedback.setText('Model hợp lệ. Nhấn Áp dụng để lưu.')

        def fail(message):
            QMessageBox.warning(self, 'Model không tương thích', message)
            self.model_feedback.setText('Import thất bại. Giữ model cũ.')

        self.import_worker.completed.connect(complete)
        self.import_worker.failed.connect(fail)
        self.import_worker.completed.connect(self.import_thread.quit)
        self.import_worker.failed.connect(self.import_thread.quit)
        self.import_thread.finished.connect(self.import_worker.deleteLater)
        self.import_thread.finished.connect(self._import_cleanup)
        self.import_thread.start()

    def _import_cleanup(self):
        thread = self.sender()
        if thread is self.import_thread:
            if thread_is_valid(thread):
                thread.wait()
                thread.deleteLater()
            self.import_worker = self.import_thread = None
        self.btn_apply.setEnabled(not self.is_applying)

    def _apply_models(self):
        if self.import_thread is not None:
            return
        self.modelsRequested.emit(dict(self.models))

    def set_selection(self, selection):
        self.selection = dict(selection)
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentIndex(1 if selection["mode"] == "gpu" else 0)
        self.mode_box.blockSignals(False)
        self.btn_check.setVisible(selection["mode"] == "gpu")
        self.btn_stop.setVisible(selection["mode"] == "gpu" and self.scan_thread is not None)
        self.gpu_label.setVisible(selection["mode"] == "gpu")
        self.gpu_spacing.setVisible(selection["mode"] == "gpu")
        if selection["mode"] == "gpu" and selection.get("gpu_id") and not self.last_found:
            self.gpu_box.blockSignals(True)
            self.gpu_box.clear()
            self.gpu_box.addItem(selection.get("gpu_name") or selection["gpu_id"], selection["gpu_id"])
            self.gpu_box.blockSignals(False)
            self.gpu_box.setVisible(True)
        if selection["mode"] != "gpu":
            self.gpu_box.setVisible(False)
            self.check_status.setText(
                CPU_EDITION_NOTE
                if build_variant() == 'cpu' else ''
            )

    def set_applying(self, active):
        self.is_applying = active
        self.mode_box.setEnabled(not active)
        self.gpu_box.setEnabled(not active)
        self.btn_check.setEnabled(not active and self.scan_thread is None)
        self.btn_exit.setEnabled(True)
        self.btn_apply.setEnabled(not active and self.import_thread is None)
        for box in self._model_combos.values():
            box.setEnabled(not active)

    def show_runtime(self, message):
        self.status.setText(message)

    def _mode_changed(self):
        mode = self.mode_box.currentData()
        self.btn_check.setVisible(mode == "gpu")
        self.btn_stop.setVisible(mode == "gpu" and self.scan_thread is not None)
        self.gpu_label.setVisible(mode == "gpu")
        self.gpu_spacing.setVisible(mode == "gpu")
        if mode == "cpu":
            if self.scan_thread:
                self._stop_scan()
            self.gpu_box.setVisible(False)
            self.check_status.setText(
                CPU_EDITION_NOTE
                if build_variant() == 'cpu' else ''
            )
            self.selectionRequested.emit({"mode": "cpu", "gpu_id": None, "gpu_name": None})
        else:
            self.gpu_box.setVisible(bool(self.gpu_box.count()))
            self.check_status.setText("Bấm GPU Check để chọn GPU. Mọi model phải sử dụng đúng GPU đó.")
            # The prior CPU engine must not keep running after GPU mode selected.
            self.selectionRequested.emit({"mode": "gpu", "gpu_id": None, "gpu_name": None})

    def _gpu_changed(self, index):
        if index < 0 or self.is_applying or self.mode_box.currentData() != "gpu":
            return
        selection = {"mode": "gpu", "gpu_id": self.gpu_box.itemData(index),
                     "gpu_name": self.gpu_box.itemText(index)}
        if not str(selection["gpu_id"]).startswith("cuda:"):
            self.check_status.setText("GPU này không hỗ trợ chế độ GPU: mọi model phải chạy trên cùng NVIDIA GPU. Không fallback CPU.")
        else:
            self.check_status.setText("")
        self.selectionRequested.emit(selection)

    def _start_scan(self):
        if self.scan_thread is not None:
            return
        self.check_status.setText("Đang check ... vui lòng đợi")
        self.btn_check.setEnabled(False)
        self.btn_stop.setVisible(True)
        self.gpu_box.setVisible(False)
        self.scan_thread = GpuScanThread(self)
        self.scan_thread.scanned.connect(self._scan_done)
        self.scan_thread.scanFailed.connect(self._scan_failed)
        self.scan_thread.scanStopped.connect(self._scan_stopped)
        self.scan_thread.finished.connect(self._scan_cleanup)
        self.scan_thread.start()

    def _stop_scan(self):
        if self.scan_thread is not None:
            self.scan_thread.stop()
            self.check_status.setText("Đang dừng tìm GPU...")
            self.btn_stop.setEnabled(False)

    def _scan_done(self, devices):
        if self.mode_box.currentData() != "gpu":
            return
        self.last_found = devices
        self.gpu_box.blockSignals(True)
        self.gpu_box.clear()
        for device in devices:
            self.gpu_box.addItem(device["name"], device["id"])
        self.gpu_box.setCurrentIndex(0)  # First detected adapter is selected automatically.
        self.gpu_box.blockSignals(False)
        self.gpu_box.setVisible(True)
        self.check_status.setText(f"Tìm thấy {len(devices)} GPU. GPU không hỗ trợ CUDA sẽ báo lỗi ở chế độ A.")
        self._gpu_changed(0)

    def _scan_failed(self, message):
        self.gpu_box.setVisible(False)
        self.check_status.setText(message)

    def _scan_stopped(self):
        self.check_status.setText("Đã dừng tìm GPU.")

    def _scan_cleanup(self):
        thread = self.sender()
        if thread is self.scan_thread:
            if thread_is_valid(thread):
                thread.wait()
                thread.deleteLater()
            self.scan_thread = None
        self.btn_stop.setVisible(False)
        self.btn_stop.setEnabled(True)
        self.btn_check.setEnabled(not self.is_applying)
