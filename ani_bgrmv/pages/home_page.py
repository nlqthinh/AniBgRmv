from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ..styles import apply_button_style

class HomePage(QWidget):
    singleRequested = Signal()
    batchRequested = Signal()
    computeRequested = Signal()

    def __init__(self):
        super().__init__()

        title = QLabel("Anime Background Remover")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: 800; color: #F5F7FA;")

        subtitle = QLabel("Chọn tác vụ cần làm")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 15px; color: #D6DEE8;")

        self.btn_single = QPushButton("Cắt nền 1 hình")
        self.btn_batch = QPushButton("Cắt nhiều hình")
        self.btn_video = QPushButton("Cắt video (sau)")
        self.btn_object = QPushButton("Xóa vật thể (sau)")
        self.btn_compute = QPushButton("Trạng thái CPU/GPU")

        for btn in [self.btn_single, self.btn_batch, self.btn_video, self.btn_object, self.btn_compute]:
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
        box.addWidget(self.btn_compute, alignment=Qt.AlignCenter)

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
        self.btn_compute.clicked.connect(self.computeRequested.emit)

