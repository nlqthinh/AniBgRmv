"""Home-style CPU/GPU settings screen. All GPU selections are strict."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..device_management import GpuScanThread, current_selection
from ..portable_device_settings import build_variant
from ..styles import apply_button_style


class ComputeStatusPage(QWidget):
    backRequested = Signal()
    selectionRequested = Signal(dict)

    def __init__(self):
        super().__init__()
        self.selection = current_selection()
        self.scan_thread = None
        self.last_found = []
        self.is_applying = False

        title = QLabel("Trạng thái CPU/GPU")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: 800; color: #F5F7FA;")
        self.status = QLabel("Đang kiểm tra trạng thái xử lý...")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 16px; color: #F5F7FA;")

        self.mode_box = QComboBox()
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
        self.btn_stop.setVisible(False)

        self.gpu_box = QComboBox()
        self.gpu_box.setFixedWidth(260)
        self.gpu_box.setMinimumHeight(46)
        self.gpu_box.setStyleSheet("font-size: 15px; padding: 6px; color: #F5F7FA; background-color: #252E3B; border: 1px solid #8A96A7; border-radius: 8px;")
        self.gpu_box.setVisible(False)
        self.check_status = QLabel(
            'CPU Lite: không chứa CUDA hoặc SAM ViT-B. Guided sử dụng GrabCut.'
            if build_variant() == 'cpu' else ''
        )
        self.check_status.setAlignment(Qt.AlignCenter)
        self.check_status.setWordWrap(True)
        self.check_status.setStyleSheet("font-size: 13px; color: #D6DEE8;")

        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(12)
        gpu_row.addStretch(1)
        gpu_row.addWidget(self.btn_check)
        gpu_row.addWidget(self.btn_stop)
        gpu_row.addStretch(1)
        self.gpu_row_widget = QWidget()
        self.gpu_row_widget.setLayout(gpu_row)

        body = QVBoxLayout()
        body.setSpacing(14)
        body.addWidget(self.mode_box, alignment=Qt.AlignCenter)
        body.addWidget(self.gpu_row_widget)
        body.addWidget(self.check_status)
        body.addWidget(self.gpu_box, alignment=Qt.AlignCenter)
        body.addSpacing(15)
        body.addWidget(self.btn_exit, alignment=Qt.AlignCenter)
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        outer.addWidget(title)
        outer.addWidget(self.status)
        outer.addSpacing(30)
        outer.addLayout(body)
        outer.addStretch(2)

        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.gpu_box.currentIndexChanged.connect(self._gpu_changed)
        self.btn_check.clicked.connect(self._start_scan)
        self.btn_stop.clicked.connect(self._stop_scan)
        self.btn_exit.clicked.connect(self.backRequested.emit)
        self.set_selection(self.selection)

    def set_selection(self, selection):
        self.selection = dict(selection)
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentIndex(1 if selection["mode"] == "gpu" else 0)
        self.mode_box.blockSignals(False)
        self.gpu_row_widget.setVisible(selection["mode"] == "gpu")
        if selection["mode"] == "gpu" and selection.get("gpu_id") and not self.last_found:
            self.gpu_box.blockSignals(True)
            self.gpu_box.clear()
            self.gpu_box.addItem(selection.get("gpu_name") or selection["gpu_id"], selection["gpu_id"])
            self.gpu_box.blockSignals(False)
            self.gpu_box.setVisible(True)
        if selection["mode"] != "gpu":
            self.gpu_box.setVisible(False)
            self.check_status.setText(
                'CPU Lite: không chứa CUDA hoặc SAM ViT-B. Guided sử dụng GrabCut.'
                if build_variant() == 'cpu' else ''
            )

    def set_applying(self, active):
        self.is_applying = active
        self.mode_box.setEnabled(not active)
        self.gpu_box.setEnabled(not active)
        self.btn_check.setEnabled(not active and self.scan_thread is None)
        self.btn_exit.setEnabled(True)

    def show_runtime(self, message):
        self.status.setText(message)

    def _mode_changed(self):
        mode = self.mode_box.currentData()
        self.gpu_row_widget.setVisible(mode == "gpu")
        if mode == "cpu":
            if self.scan_thread:
                self._stop_scan()
            self.gpu_box.setVisible(False)
            self.check_status.setText(
                'CPU Lite: không chứa CUDA hoặc SAM ViT-B. Guided sử dụng GrabCut.'
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
            self.check_status.setText("GPU này không hỗ trợ chế độ A: F3/M7/SAM ONNX phải dùng cùng GPU. Không chạy CPU thay thế.")
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
        self.btn_stop.setVisible(False)
        self.btn_stop.setEnabled(True)
        self.scan_thread = None
        self.btn_check.setEnabled(not self.is_applying)
