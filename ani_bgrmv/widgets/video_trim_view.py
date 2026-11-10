"""Choose the part of a video longer than 10 s to keep, then ✂ Cắt.

Laid out as the video editor: the picture, then the button where Play is and
the timeline, here a RangeSlider of at most 10 s. The picture follows the
handle being dragged.
"""
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ..styles import CANVAS_BG
from .range_slider import RangeSlider

PREVIEW_DELAY_MS = 60  # decode the picture once the handle rests, not at every pixel


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f'{int(minutes):02d}:{rest:05.2f}'


class VideoTrimView(QWidget):
    cutRequested = Signal(int, int)  # source frames [start, end) to keep
    canceled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = None   # video_pipeline.FramePreview of the open video
        self._fps = 1.0
        self._pixmap = QPixmap()
        self._wanted = 0
        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(f'background: {CANVAS_BG};')
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.btn_cut = QPushButton('✂ Cắt')
        self.btn_cut.setToolTip('Giữ đoạn đã chọn (tối đa 10 giây) rồi mở trình cắt nền video.')
        self.range = RangeSlider()
        self.range_label = QLabel()
        self.btn_cancel = QPushButton('Hủy')
        row = QHBoxLayout()
        row.addWidget(self.btn_cut)
        row.addWidget(self.range, 1)
        row.addWidget(self.range_label)
        row.addWidget(self.btn_cancel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preview, 1)
        layout.addLayout(row)

        self._preview_timer = QTimer(self, singleShot=True, interval=PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(lambda: self._show(self._wanted))
        self.range.handleMoved.connect(self._schedule_preview)
        self.range.rangeChanged.connect(self._describe)
        self.btn_cut.clicked.connect(lambda: self.cutRequested.emit(*self.range.values()))
        self.btn_cancel.clicked.connect(self.canceled)

    def open(self, info, frames):
        """``info``: video_pipeline.VideoInfo; ``frames``: its FramePreview (closed by close_video)."""
        self.close_video()
        self._frames, self._fps = frames, info.fps
        self.range.set_limits(info.frame_count, info.max_clip_frames)
        self._describe(*self.range.values())
        self._show(0)

    def close_video(self):
        self._preview_timer.stop()
        if self._frames is not None:
            self._frames.close()
            self._frames = None
        self._pixmap = QPixmap()
        self.preview.clear()

    def _schedule_preview(self, frame: int):
        self._wanted = frame
        self._preview_timer.start()

    def _describe(self, start: int, end: int):
        self.range_label.setText(f'{_clock(start / self._fps)} – {_clock(end / self._fps)} '
                                 f'· giữ {(end - start) / self._fps:.2f}s')

    def _show(self, frame: int):
        rgb = None if self._frames is None else self._frames.frame(frame)
        if rgb is None:
            return
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self._pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())
        self._fit()

    def _fit(self):
        if not self._pixmap.isNull():
            self.preview.setPixmap(self._pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()
