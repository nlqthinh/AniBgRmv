"""Video editor, laid out as the image pages and the annotator: the frame and
its timeline on the left, the tools on the right.

Objects work as in the annotator (Gan_nhan): the object list, + Mới (N),
Gộp vào (G), Bỏ (Del); Ctrl+click on the frame or Tab selects. Restore gives
unowned pixels to the selected object, Erase takes only its pixels, so no
edit touches another object. Every edited frame becomes a keyframe (yellow
mark on the timeline; click it, then Delete removes it); Track (HQ-SAM2 Video
Predictor, its own process) follows each object with its own memory, over the
whole clip or, with ◀ Track lùi / Track tiến ▶, only from the current keyframe
to the next one; Export keeps all objects as one alpha. See
engine/video_objects.py.
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSlider,
                               QStackedWidget, QVBoxLayout, QWidget)

from ..annotation.overlay import object_color
from ..app_config import APP_BASE_DIR, APP_CONFIG
from ..edit_history import EditHistory
from ..engine import video_holds as vh
from ..engine import video_objects as vo
from ..engine.box_prompt import normalized_box
from ..engine.video_diagnostics import create_run
from ..engine.video_pipeline import (FramePreview, VideoCanceled, VideoClip, VideoFormatError,
                                     blank_alpha, cleanup_legacy_matting_cache, export_destination,
                                     export_video, extract_clip, paint_alpha, probe_video,
                                     release_clip_temp, track_masks)
from ..qt_threads import thread_is_running, thread_is_valid
from ..styles import CANCEL_BUTTON_STYLE, MUTED_TEXT_COLOR
from ..widgets.edit_mask_canvas import DEFAULT_TINT_OPACITY, EditMaskCanvas
from ..widgets.keyframe_slider import KeyframeSlider
from ..widgets.upload_preview_canvas import UploadPreviewCanvas
from ..widgets.video_trim_view import VideoTrimView
from ..workers import GuidedEditWorker
from .page_layout import PAGE_MARGIN, button_row, side_panel, top_bar

UNDO_STEPS = 20  # whole object states; they share every unchanged label map
NOTHING_CHANGED = ('Không có gì thay đổi: Restore chỉ lấy vùng chưa thuộc object nào; '
                   'Erase chỉ xóa object đang chọn.')
JOB_TITLES = {'track': 'Đang Track', 'export': 'Đang Export', 'guided': 'Đang xử lý Guided Edit'}
LONG_VIDEO_HINT = 'Video dài hơn 10 giây: chọn đoạn ≤10 giây để cắt.'
UPLOAD_HINT = 'Kéo-thả video hoặc bấm nút để chọn. ' + LONG_VIDEO_HINT


class VideoJob(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, action: str, *args, **kwargs):
        super().__init__()
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    @Slot()
    def run(self):
        try:
            tasks = {'upload': extract_clip, 'track': track_masks, 'export': export_video}
            result = tasks[self.action](*self.args, progress=self.progress.emit,
                                        cancel=lambda: self.cancel_requested, **self.kwargs)
            if self.cancel_requested:
                if self.action == 'upload':
                    release_clip_temp(result.temp_dir)
                self.canceled.emit()
            else:
                self.finished.emit(result)
        except VideoCanceled:
            self.canceled.emit()
        except (VideoFormatError, ValueError) as exc:
            self.error.emit(str(exc))
        except Exception:
            self.error.emit(traceback.format_exc())


def _swatch(object_id: int) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*object_color(object_id)))
    return QIcon(pixmap)


class VideoCutPage(QWidget):
    backRequested = Signal()

    def __init__(self, engine_getter):
        super().__init__()
        self.engine_getter = engine_getter
        self.clip: VideoClip | None = None
        self.current_frame = 0
        self.edits = vo.ObjectEdits()
        self.tracked: tuple[np.ndarray, ...] | None = None  # label maps from Track
        self.held: dict[int, vh.Hold] = {}  # frames that hold an earlier picture (Track)
        self._tracked_keyframes: dict[int, np.ndarray] = {}  # what the last full Track was given
        # (frame shown, objects before the change): Undo/Redo restore both.
        self.history: EditHistory[tuple[int, vo.ObjectEdits]] = EditHistory(UNDO_STEPS)
        self.operation = 'restore'  # a new object starts empty: give it pixels first
        self.submode = 'guided'
        self.draw_mode = 'paint'
        self.merge_pick = False
        self._guided_object: int | None = None
        self.output_dir = str((APP_BASE_DIR / APP_CONFIG.get('output', {}).get(
            'video_output_dir', 'output/video_saved')).resolve())
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._job_name = ''
        self._busy = False
        self._cache_output_dirs = {self.output_dir}

        top, self.btn_back, self.status_label = top_bar('Cắt nền video (≤10 giây)')
        self.status_label.setText(UPLOAD_HINT)
        self.upload_canvas = UploadPreviewCanvas(
            'Upload video', 'Kéo-thả video vào đây hoặc bấm nút để chọn. ' + LONG_VIDEO_HINT,
            'Chọn video…', cancel_text='Dừng Up')
        self.edit_canvas = EditMaskCanvas(file_actions=True, object_labels=True)
        self.edit_canvas.btn_close.setToolTip('Xóa video này khỏi trang Video')
        self.view = QStackedWidget()
        self.editor_view = self._build_editor()
        self.trim_view = VideoTrimView()  # a video over 10 s: choose the part to keep first
        self._trim_path: str | None = None
        for view in (self.upload_canvas, self.editor_view, self.trim_view):
            self.view.addWidget(view)

        body = QHBoxLayout()
        self.panel = self._build_panel()
        body.addWidget(self.view, 1)
        body.addWidget(self.panel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*(PAGE_MARGIN,) * 4)
        layout.addLayout(top)
        layout.addLayout(body, 1)

        self.btn_back.clicked.connect(self._back)
        self.upload_canvas.clicked.connect(self._choose_video)
        self.upload_canvas.filesDropped.connect(self._dropped)
        self.upload_canvas.cancelRequested.connect(self.stop_task)
        self.edit_canvas.filesDropped.connect(self._dropped)
        self.edit_canvas.closeRequested.connect(self._close_video_clicked)
        self.trim_view.cutRequested.connect(self._cut)
        self.trim_view.canceled.connect(self._cancel_trim)
        self.edit_canvas.picked.connect(self._on_pick)
        self.edit_canvas.strokeFinished.connect(lambda points: self._edit(points=points))
        self.edit_canvas.boxFinished.connect(lambda box: self._edit(box=box))
        self.timeline.valueChanged.connect(self._set_frame)
        self.timeline.deleteRequested.connect(self.delete_keyframe)
        self.btn_play.clicked.connect(self._toggle_play)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self.object_list.itemClicked.connect(
            lambda item: self._select(item.data(Qt.ItemDataRole.UserRole)))
        self.btn_new_object.clicked.connect(self.new_object)
        self.btn_merge.clicked.connect(self.start_merge)
        self.btn_drop.clicked.connect(self.drop_object)
        self.btn_erase.clicked.connect(lambda: self._set_operation('erase'))
        self.btn_restore.clicked.connect(lambda: self._set_operation('restore'))
        self.btn_clear.clicked.connect(self.clear_mask)
        self.btn_guided.clicked.connect(lambda: self._set_submode('guided'))
        self.btn_manual.clicked.connect(lambda: self._set_submode('manual'))
        self.btn_paint.clicked.connect(lambda: self._set_draw_mode('paint'))
        self.btn_box.clicked.connect(lambda: self._set_draw_mode('box'))
        self.btn_fit.clicked.connect(self.edit_canvas.reset_view)
        self.brush_slider.valueChanged.connect(self._brush_changed)
        self.tint_slider.valueChanged.connect(self._tint_changed)
        self.btn_stop.clicked.connect(self.stop_task)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_track.clicked.connect(lambda: self.track())
        self.btn_track_back.clicked.connect(lambda: self.track_span(forward=False))
        self.btn_track_forward.clicked.connect(lambda: self.track_span(forward=True))
        self.btn_export.clicked.connect(self.export)
        self.btn_config.clicked.connect(self.choose_output)
        self.btn_open_folder.clicked.connect(self.open_output)
        for keys, handler in (('Ctrl+Z', self.undo), ('Ctrl+Y', self.redo), ('N', self.new_object),
                              ('G', self.start_merge), ('Delete', self.drop_object),
                              ('Tab', self.next_object)):
            QShortcut(QKeySequence(keys), self).activated.connect(handler)
        self._brush_changed(self.brush_slider.value())
        self._tint_changed(self.tint_slider.value())
        self._describe()
        self._sync_actions()

    # -- layout -----------------------------------------------------------------
    def _build_editor(self) -> QWidget:
        """The frame, then Play and the timeline under it."""
        self.btn_play = QPushButton('▶ Play')
        self.timeline = KeyframeSlider()
        self.timeline.setRange(0, 0)
        self.timeline.setToolTip('Vạch vàng = keyframe. Click vạch để tới keyframe đó, '
                                 'rồi nhấn Delete để xóa keyframe (Undo được).')
        self.timeline_label = QLabel('00:00 / 00:00 · Frame 0/0')
        row = QHBoxLayout()
        row.addWidget(self.btn_play)
        row.addWidget(self.timeline, 1)
        row.addWidget(self.timeline_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        editor = QWidget()
        layout = QVBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit_canvas, 1)
        layout.addLayout(row)
        layout.addWidget(self.progress_bar)
        return editor

    def _build_panel(self) -> QWidget:
        panel = QVBoxLayout()
        self.name_label = QLabel('Chưa có video')
        self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet('font-weight: 700;')
        panel.addWidget(self.name_label)
        panel.addWidget(QLabel('Object đang sửa (Ctrl+click trên ảnh, Tab):'))
        self.object_list = QListWidget()
        self.object_list.setMaximumHeight(120)
        panel.addWidget(self.object_list)
        (self.btn_new_object, self.btn_merge, self.btn_drop), row = button_row(
            '+ Mới (N)', 'Gộp vào (G)', 'Bỏ (Del)')
        panel.addLayout(row)
        (self.btn_erase, self.btn_clear, self.btn_restore), row = button_row(
            'Erase', 'Clear Mask', 'Restore', checkable=True)
        self.btn_clear.setCheckable(False)
        self.btn_clear.setToolTip('Xóa object đang chọn trên frame này (có thể Undo).')
        panel.addLayout(row)
        (self.btn_guided, self.btn_manual), row = button_row('Guided', 'Manual', checkable=True)
        panel.addLayout(row)
        (self.btn_paint, self.btn_box), row = button_row('Paint mode', 'Box mode', checkable=True)
        panel.addLayout(row)
        self.btn_restore.setChecked(True)
        self.btn_guided.setChecked(True)
        self.btn_paint.setChecked(True)
        self.btn_fit = QPushButton('Fit')
        panel.addWidget(self.btn_fit)
        self.brush_label = QLabel()
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(4, 120)
        self.brush_slider.setValue(28)
        panel.addWidget(self.brush_label)
        panel.addWidget(self.brush_slider)
        self.tint_label = QLabel()
        self.tint_slider = QSlider(Qt.Horizontal)
        self.tint_slider.setRange(0, 100)
        self.tint_slider.setValue(DEFAULT_TINT_OPACITY)
        self.tint_slider.setToolTip('Chỉ đổi phần xem; không đổi mask Track/Export.')
        panel.addWidget(self.tint_label)
        panel.addWidget(self.tint_slider)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(f'color: {MUTED_TEXT_COLOR};')
        panel.addWidget(self.description_label)
        panel.addStretch(1)
        tools = QWidget()
        tools.setLayout(panel)

        panel = QVBoxLayout()  # always visible under the tools
        panel.setContentsMargins(0, 0, 0, 0)
        self.btn_stop = QPushButton('Dừng')
        self.btn_stop.setStyleSheet(CANCEL_BUTTON_STYLE)
        self.btn_stop.hide()
        panel.addWidget(self.btn_stop)
        (self.btn_undo, self.btn_redo), row = button_row('Undo', 'Redo')
        panel.addLayout(row)
        self.btn_track = QPushButton('Track')
        self.btn_export = QPushButton('Export')
        self.btn_track_back = QPushButton('◀ Track lùi')
        self.btn_track_forward = QPushButton('Track tiến ▶')
        self.btn_track_back.setToolTip('Đứng ở một keyframe: chỉ Track lùi tới keyframe gần nhất '
                                       'bên trái (không có thì tới đầu video).')
        self.btn_track_forward.setToolTip('Đứng ở một keyframe: chỉ Track tiến tới keyframe gần nhất '
                                          'bên phải (không có thì tới cuối video).')
        self.btn_config = QPushButton('Đường dẫn lưu')
        self.btn_open_folder = QPushButton('Mở folder đã lưu')
        grid = QGridLayout()
        for index, button in enumerate((self.btn_track, self.btn_export,
                                        self.btn_track_back, self.btn_track_forward,
                                        self.btn_config, self.btn_open_folder)):
            grid.addWidget(button, index // 2, index % 2)
        panel.addLayout(grid)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet(f'color: {MUTED_TEXT_COLOR}; font-size: 11px;')
        panel.addWidget(self.path_label)
        self._show_output_dir()
        actions = QWidget()
        actions.setLayout(panel)
        return side_panel(tools, actions)

    def _describe(self):
        """What the current tool does, as the annotator's help text."""
        if self.merge_pick:
            text = 'GỘP: click object cần gộp vào object đang chọn (trên mọi frame).'
        else:
            what = ('Restore: vùng chưa thuộc object nào → object đang chọn.' if self.operation == 'restore'
                    else 'Erase: vùng của object đang chọn → không thuộc object nào.')
            how = 'Kéo cọ lên vùng.' if self.draw_mode == 'paint' else 'Kéo khung quanh vùng.'
            way = 'HQ-SAM tự tìm vùng.' if self.submode == 'guided' else 'Đúng nét vẽ.'
            text = (f'{what} {how} {way} Không bao giờ đụng object khác.\n'
                    'Frame đã sửa là keyframe: Track theo đúng các object thấy trên frame đó '
                    '(object không tô = không có mặt).')
        self.description_label.setText(text)

    # -- busy state -----------------------------------------------------------------
    @property
    def is_busy(self):
        return self._busy or thread_is_running(self._thread)

    def _sync_actions(self):
        """Buttons enabled for the current state (nothing while a job runs)."""
        idle = not self.is_busy
        has_clip = idle and self.clip is not None
        has_object = has_clip and self.edits.active is not None
        for widget in (self.btn_new_object, self.object_list, self.btn_guided, self.btn_manual,
                       self.btn_paint, self.btn_box, self.btn_fit, self.btn_play, self.timeline,
                       self.btn_track, self.edit_canvas.btn_close, self.tint_slider):
            widget.setEnabled(has_clip)
        for widget in (self.btn_merge, self.btn_drop, self.btn_erase, self.btn_clear, self.btn_restore):
            widget.setEnabled(has_object)
        self.brush_slider.setEnabled(has_clip and self.draw_mode == 'paint')
        self.btn_undo.setEnabled(has_clip and self.history.can_undo)
        self.btn_redo.setEnabled(has_clip and self.history.can_redo)
        self.btn_export.setEnabled(has_clip and self.tracked is not None)
        for button, forward in ((self.btn_track_back, False), (self.btn_track_forward, True)):
            button.setEnabled(has_clip and self._span(forward) is not None)
        for widget in (self.btn_back, self.btn_config, self.btn_open_folder):
            widget.setEnabled(idle)

    def _busy_ui(self, busy: bool):
        self.btn_stop.setVisible(busy and self.clip is not None)
        self.btn_stop.setEnabled(busy)
        self.edit_canvas.set_processing(busy and self.clip is not None,
                                        JOB_TITLES.get(self._job_name, ''))
        self.progress_bar.setVisible(busy and self.clip is not None)
        self.progress_bar.setValue(0)
        self._sync_actions()

    def _back(self):
        if self.is_busy:
            QMessageBox.information(self, 'Đang xử lý', 'Hãy bấm Dừng và chờ tác vụ kết thúc.')
            return
        self._play_timer.stop()
        self.backRequested.emit()

    # -- video file -------------------------------------------------------------------
    def _choose_video(self):
        if self.is_busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Chọn video', '',
                                              'Videos (*.mp4 *.mov *.mkv *.webm *.avi *.m4v)')
        if path:
            self.open_video(path)

    def _dropped(self, paths: list[str]):
        if not paths:
            return
        if len(paths) != 1 or not Path(paths[0]).is_file():
            QMessageBox.warning(self, 'Sai video', 'Chỉ cho phép upload một file video mỗi lần.')
            return
        self.open_video(paths[0])

    def open_video(self, path: str):
        if self.is_busy:
            return
        if self.clip:
            if not self.confirm_close_video():
                return
            self.clear_video()
        self._end_trim()
        try:
            info = probe_video(path)
        except VideoFormatError as exc:
            QMessageBox.warning(self, 'Lỗi video', str(exc))
            return
        if not info.too_long:
            self._upload(path)
        elif self._confirm_trim():
            self._start_trim(info)

    def _upload(self, path: str, **part):
        """Read the video (``part``: start/end source frames of a longer one)."""
        self.view.setCurrentWidget(self.upload_canvas)
        self.upload_canvas.set_loading_mode('Đang kiểm tra và đọc video...', allow_cancel=True)
        self._run_job('upload', VideoJob('upload', path, **part))

    # -- a video over 10 seconds: keep a part of it -----------------------------------------
    def _confirm_trim(self) -> bool:
        box = QMessageBox(QMessageBox.Icon.Information, 'Video dài hơn 10 giây',
                          'Video vượt quá 10 giây. Bạn phải cắt video còn 10 giây để sử dụng '
                          'tính năng này', parent=self)
        agree = box.addButton('Đồng ý', QMessageBox.ButtonRole.AcceptRole)
        box.addButton('Hủy', QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(agree)
        box.exec()
        return box.clickedButton() is agree

    def _start_trim(self, info):
        self._trim_path = info.path
        self.trim_view.open(info, FramePreview(info.path))
        self.view.setCurrentWidget(self.trim_view)
        self.name_label.setText(f'{Path(info.path).name} · {info.duration:.2f}s')
        self.status_label.setText('Kéo 2 đầu (hoặc cả đoạn) trên thanh thời gian để chọn tối đa '
                                  '10 giây, rồi bấm ✂ Cắt.')

    def _cut(self, start: int, end: int):
        path = self._trim_path
        self._end_trim()
        self._upload(path, start=start, end=end)

    def _cancel_trim(self):
        self._end_trim()
        self.view.setCurrentWidget(self.upload_canvas)
        self.name_label.setText('Chưa có video')
        self.status_label.setText(UPLOAD_HINT)

    def _end_trim(self):
        self.trim_view.close_video()
        self._trim_path = None

    def confirm_close_video(self) -> bool:
        """Edits are lost unless exported: ask first (✕, or dropping another video)."""
        box = QMessageBox(QMessageBox.Icon.Warning, 'Xóa video',
                          'Xóa video này khỏi trang Video?\n'
                          'Mask, keyframe và kết quả Track chưa Export sẽ mất. '
                          'File video gốc trên máy không bị xóa.', parent=self)
        confirm = box.addButton('Xác nhận', QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(box.addButton('Hủy', QMessageBox.ButtonRole.RejectRole))
        box.exec()
        return box.clickedButton() is confirm

    def _close_video_clicked(self):
        if self.clip and not self.is_busy and self.confirm_close_video():
            self.clear_video()

    def clear_video(self):
        if self.is_busy:
            return
        self._play_timer.stop()
        if self.clip:
            release_clip_temp(self.clip.temp_dir)
        self._cleanup_legacy_caches()
        self.clip = None
        self.current_frame = 0
        self.edits = vo.ObjectEdits()
        self._forget_track_result()
        self.history.clear()
        self.merge_pick = self.edit_canvas.pick_next = False
        self.edit_canvas.clear_image()
        self.upload_canvas.set_upload_mode()
        self.view.setCurrentWidget(self.upload_canvas)
        self.name_label.setText('Chưa có video')
        self.object_list.clear()
        self.status_label.setText(UPLOAD_HINT)
        self._describe()
        self._sync_actions()

    # -- jobs (upload, Guided, Track, Export) ---------------------------------------------
    def _run_job(self, name: str, worker: QObject):
        if self.is_busy:
            return
        self._play_timer.stop()
        self._job_name = name
        self._busy = True
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_job_done)
        worker.canceled.connect(self._on_cancel)
        worker.error.connect(self._on_error)
        for signal in (worker.finished, worker.canceled, worker.error):
            signal.connect(self._thread.quit)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._busy_ui(True)
        self._thread.start()

    def _on_thread_finished(self):
        thread = self.sender()
        if thread is not self._thread:
            return
        if thread_is_valid(thread):
            thread.wait()
        self._thread = None
        self._worker = None
        if thread:
            thread.deleteLater()
        self._busy = False
        self._busy_ui(False)

    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(max(0, min(100, value)))
        self.status_label.setText(message)
        if self.clip:
            self.edit_canvas.set_processing_progress(value, message)
        else:
            self.upload_canvas.set_loading_mode(message, max(0, value), 100, allow_cancel=True)

    def stop_task(self):
        if self._worker is None:
            return
        self._worker.cancel()
        self.status_label.setText('Đã yêu cầu Dừng. Chờ worker kết thúc frame hiện tại...')
        self.btn_stop.setEnabled(False)
        if self._job_name == 'upload':
            self.upload_canvas.set_cancel_waiting_mode()

    def _diagnostics(self) -> str:
        return str(getattr(self, '_last_track_diagnostics' if self._job_name == 'track'
                           else '_last_export_diagnostics', ''))

    def _on_cancel(self):
        self.status_label.setText('Đã dừng. Không áp dụng kết quả dở dang. Log: ' + self._diagnostics())
        if self._job_name == 'upload':
            self.upload_canvas.set_upload_mode()

    def _on_error(self, message: str):
        self.status_label.setText('Tác vụ video thất bại. Log: ' + self._diagnostics())
        if self._job_name == 'upload':
            self.upload_canvas.set_upload_mode()
        QMessageBox.warning(self, 'Lỗi video', message)

    def _on_job_done(self, result):
        if self._job_name == 'upload':
            self.clip = result
            self.current_frame = 0
            self.edits = vo.ObjectEdits()
            self._forget_track_result()
            self.history.clear()
            self.view.setCurrentWidget(self.editor_view)
            self.timeline.blockSignals(True)
            self.timeline.setRange(0, self.clip.count - 1)
            self.timeline.setValue(0)
            self.timeline.blockSignals(False)
            self.name_label.setText(f'{Path(self.clip.source).name} · {self.clip.duration:.2f}s · '
                                    f'{self.clip.count} frame @ {self.clip.output_fps:g} fps')
            self._set_operation('restore')
            self.edit_canvas.reset_view()
            self._show_frame()
            self.status_label.setText('Tô từng người vào một object (+ Mới cho người tiếp theo), '
                                      'rồi bấm Track.')
        elif self._job_name == 'track':
            self._apply_track(result)
            self._show_frame()
            self.status_label.setText(self._track_summary(result))
        elif self._job_name == 'export':
            self.status_label.setText(f'Đã export: {result}. Log: {self._diagnostics()}')
            saved = (f'Đã lưu chuỗi PNG trong suốt, {self.clip.output_fps:g} fps '
                     '(khi nhập vào phần mềm edit, đặt đúng fps này)'
                     if Path(result).is_dir() else 'Đã lưu MP4 nền màu (MP4 không có alpha)')
            QMessageBox.information(self, 'Đã export video', f'{saved}:\n{result}')
        elif self._job_name == 'guided':
            if (result.get('guided_edit_info') or {}).get('selection_failed'):
                self.status_label.setText('HQ-SAM chưa nhận diện được vật thể ở nét này; mask không đổi. '
                                          'Thử nét khác, hoặc dùng Manual.')
                return
            labels = self._labels()
            region = vo.guided_region(labels, self._guided_object, result['final_alpha'], self.operation)
            self._commit(vo.edit_region(self.edits, self.tracked, self.current_frame,
                                        labels.shape, region, self.operation))

    # -- frames -------------------------------------------------------------------------
    def _labels(self, frame: int | None = None) -> np.ndarray:
        """Label map shown for ``frame`` (default: the current one). Read only."""
        return vo.labels_at(self.edits, self.tracked,
                            self.current_frame if frame is None else frame,
                            (self.clip.proxy_height, self.clip.proxy_width))

    def _show_frame(self):
        if not self.clip:
            return
        idx = self.current_frame
        labels = self._labels()
        self.edit_canvas.set_image_alpha(self.clip.frame_paths[idx], labels)
        self.edit_canvas.set_object_view(self.tint_slider.value(), self.edits.active)
        self.timeline.set_keyframes(self.edits.keyframes)
        self.timeline_label.setText(f'{idx / self.clip.output_fps:05.2f}s / {self.clip.duration:.2f}s · '
                                    f'Frame {idx + 1}/{self.clip.count}'
                                    + (' · keyframe' if idx in self.edits.keyframes else ''))
        self._show_objects(labels)
        self._sync_actions()

    def _show_objects(self, labels: np.ndarray):
        """The object list, with each object's share of the current frame."""
        self.object_list.clear()
        for object_id, share in vo.shares(labels, self.edits.objects).items():
            entry = QListWidgetItem(_swatch(object_id), f'Object {object_id} · {share:.0%}')
            entry.setData(Qt.ItemDataRole.UserRole, object_id)
            self.object_list.addItem(entry)
            if object_id == self.edits.active:
                self.object_list.setCurrentItem(entry)

    def _track_summary(self, result) -> str:
        first, last = result.frames[0], result.frames[-1]
        text = (f'HQ-SAM2 Track {len(result.frames)} frame' if len(result.frames) == self.clip.count
                else f'HQ-SAM2 Track frame {first + 1}–{last + 1} ({len(result.frames)} frame)')
        if result.held:
            text += f', {len(result.held)} frame giữ hình dùng chung mask'
        if result.agreement:
            object_id, worst = min(result.agreement.items(), key=lambda item: item[1]['iou_min'])
            text += (f'. So với lần Track trước: IoU thấp nhất {worst["iou_min"]:.3f} '
                     f'(Object {object_id}, frame {worst["worst_frame"] + 1})')
        return text + f'. Log: {self._diagnostics()}. Có thể Export.'

    def _same_keyframes_as_track(self) -> bool:
        """Track again with nothing edited: its masks can be compared with the last ones."""
        keyframes = self.edits.keyframes
        return (self.tracked is not None and keyframes.keys() == self._tracked_keyframes.keys()
                and all(np.array_equal(keyframes[i], self._tracked_keyframes[i]) for i in keyframes))

    def _forget_track_result(self):
        self.tracked, self.held, self._tracked_keyframes = None, {}, {}

    def _apply_track(self, result):
        """A Track of the whole clip replaces the tracked frames; a span Track
        replaces only its frames (and their holds). Editing is blocked while a
        job runs, so the keyframes now are the ones it tracked."""
        if len(result.frames) == self.clip.count:
            self.tracked, self.held = result.masks, dict(result.held)
            self._tracked_keyframes = dict(self.edits.keyframes)
            self.edits = vo.after_track(self.edits)
            return
        # Frames outside the span keep their labels and the relabel table of
        # Gộp/Bỏ; the span's masks come from keyframes that already use the
        # current object ids, which the table never maps.
        span = set(result.frames)
        blank = np.zeros((self.clip.proxy_height, self.clip.proxy_width), np.uint8)
        tracked = list(self.tracked or (blank,) * self.clip.count)
        for frame, labels in zip(result.frames, result.masks):
            tracked[frame] = labels
        self.tracked = tuple(tracked)
        self.held = {i: hold for i, hold in self.held.items() if i not in span} | result.held
        self._tracked_keyframes = {}  # no longer one Track of these keyframes

    def forget_track(self, message: str):
        """MainWindow: the SAM model changed, tracked frames are no longer valid."""
        self._forget_track_result()
        if self.clip is not None:
            self._show_frame()
            self.status_label.setText(message)

    def _set_frame(self, idx: int):
        if self.clip is None or self.is_busy:
            return
        self.current_frame = max(0, min(int(idx), self.clip.count - 1))
        self._show_frame()

    def _toggle_play(self):
        if not self.clip or self.is_busy:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText('▶ Play')
            return
        if self.current_frame >= self.clip.count - 1:
            self.timeline.setValue(0)  # at the end: play again from the start
        self._play_timer.start(max(40, round(1000 / min(12.0, self.clip.output_fps))))
        self.btn_play.setText('❚❚ Pause')

    def _play_tick(self):
        if not self.clip or self.is_busy or self.current_frame + 1 >= self.clip.count:
            self._play_timer.stop()
            self.btn_play.setText('▶ Play')
            return
        self.timeline.setValue(min(self.clip.count - 1,
                                   self.current_frame + max(1, round(self.clip.output_fps / 12))))

    # -- tools ----------------------------------------------------------------------------
    def _set_operation(self, op: str):
        self.operation = op
        self.btn_erase.setChecked(op == 'erase')
        self.btn_restore.setChecked(op == 'restore')
        self.edit_canvas.set_operation(op)
        self._describe()

    def _set_draw_mode(self, mode: str):
        self.draw_mode = mode
        self.btn_paint.setChecked(mode == 'paint')
        self.btn_box.setChecked(mode == 'box')
        self.edit_canvas.set_draw_mode(mode)
        self._describe()
        self._sync_actions()

    def _set_submode(self, mode: str):
        self.submode = mode
        self.btn_guided.setChecked(mode == 'guided')
        self.btn_manual.setChecked(mode == 'manual')
        self._describe()

    def _brush_changed(self, value: int):
        self.edit_canvas.set_brush_size(value)
        self.brush_label.setText(f'Cỡ cọ: {value} px')

    def _tint_changed(self, value: int):
        self.tint_label.setText(f'Độ mờ màu tô: {value}%')
        self.edit_canvas.set_object_view(value, self.edits.active)

    # -- edits ----------------------------------------------------------------------------
    def _commit(self, edits: vo.ObjectEdits | None, nothing: str = NOTHING_CHANGED):
        """Apply a change as one Undo step; ``None`` means nothing changed."""
        if edits is None:
            self.status_label.setText(nothing)
            return
        self.history.push((self.current_frame, self.edits))
        self.edits = edits
        self._show_frame()
        self.status_label.setText(f'Đã sửa frame {self.current_frame + 1}. Track lại để lan truyền, '
                                  'hoặc Export.')

    def _edit(self, points=None, box=None):
        if not self.clip or self.is_busy or (not points and box is None):
            return
        self.edit_canvas.clear_stroke()
        if self.edits.active is None:
            self.status_label.setText('Chưa có object: bấm + Mới (N) để tạo.')
            return
        if box is not None:
            box = normalized_box(box, self.clip.proxy_width, self.clip.proxy_height)
            if box is None:
                self.status_label.setText('Box không giao với video hoặc quá nhỏ; không chạy Guided Edit.')
                return
        labels = self._labels()
        if self.submode == 'manual':
            region = paint_alpha(blank_alpha(self.clip), points or [], self.brush_slider.value(),
                                 'restore', box) > 0
            self._commit(vo.edit_region(self.edits, self.tracked, self.current_frame,
                                        labels.shape, region, self.operation))
            return
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.warning(self, 'Chưa tải model', 'Model chưa sẵn sàng.')
            return
        # Guided works on the selected object alone, as on a single image.
        self._guided_object = self.edits.active
        worker = GuidedEditWorker(engine=engine,
                                  image_path=self.clip.frame_paths[self.current_frame],
                                  current_alpha=vo.object_mask(labels, self.edits.active),
                                  stroke_points=points or [],
                                  prompt_box=box,
                                  brush_size=self.brush_slider.value(),
                                  operation=self.operation)
        self._run_job('guided', worker)

    def clear_mask(self):
        """The selected object leaves the current frame."""
        if self.clip and not self.is_busy:
            labels = self._labels()
            self._commit(vo.edit_region(self.edits, self.tracked, self.current_frame, labels.shape,
                                        np.ones(labels.shape, bool), 'erase'))

    def delete_keyframe(self, frame: int):
        """The timeline's Delete on a keyframe: that frame shows its Track result again."""
        if self.clip and not self.is_busy and frame in self.edits.keyframes:
            self._commit(vo.drop_keyframe(self.edits, frame))
            self.status_label.setText(f'Đã xóa keyframe frame {frame + 1} (Undo để lấy lại).')

    def undo(self):
        if self.clip and not self.is_busy and self.history.can_undo:
            self._restore(self.history.undo(lambda state: (state[0], self.edits)))

    def redo(self):
        if self.clip and not self.is_busy and self.history.can_redo:
            self._restore(self.history.redo(lambda state: (state[0], self.edits)))

    def _restore(self, state: tuple[int, vo.ObjectEdits]):
        """Back to a state, shown on the frame where it changed."""
        frame, self.edits = state
        self.timeline.blockSignals(True)
        self.timeline.setValue(frame)
        self.timeline.blockSignals(False)
        self.current_frame = frame
        self._show_frame()

    # -- objects (as the annotator) ------------------------------------------------------
    def _select(self, object_id):
        if object_id in self.edits.objects and not self.is_busy:
            self.edits = vo.select(self.edits, object_id)
            self._show_frame()

    def next_object(self):
        objects = self.edits.objects
        if self.clip and objects:
            current = objects.index(self.edits.active) if self.edits.active in objects else -1
            self._select(objects[(current + 1) % len(objects)])

    def new_object(self):
        if not self.clip or self.is_busy:
            return
        try:
            self._commit(vo.new_object(self.edits))
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._set_operation('restore')  # a new object starts empty: give it pixels
        self.status_label.setText(f'Object {self.edits.active}: Restore lên người/vật cần giữ.')

    def drop_object(self):
        if self.clip and not self.is_busy and self.edits.active is not None:
            dropped = self.edits.active
            self._commit(vo.drop_active(self.edits))
            self.status_label.setText(f'Đã bỏ Object {dropped} trên mọi frame (Undo để lấy lại).')

    def start_merge(self):
        if self.clip and not self.is_busy and self.edits.active is not None:
            self.merge_pick = self.edit_canvas.pick_next = True
            self._describe()

    def _on_pick(self, x: float, y: float):
        if not self.clip or self.is_busy:
            return
        labels = self._labels()
        object_id = int(labels[min(int(y), labels.shape[0] - 1), min(int(x), labels.shape[1] - 1)])
        if self.merge_pick:
            self.merge_pick = False
            self._describe()
            self._commit(vo.merge_into_active(self.edits, object_id),
                         nothing='Click vào một object KHÁC object đang chọn để gộp.')
        elif object_id:
            self._select(object_id)

    # -- Track / Export -------------------------------------------------------------------
    def _free_image_models(self, action: str) -> bool:
        """A video worker is about to use the GPU from another process: free the
        image sessions first. False (after a warning) when that fails."""
        engine = self.engine_getter()
        if engine is None:
            return True
        try:
            engine.release_for_video_worker()
        except Exception as exc:
            QMessageBox.warning(self, 'Không giải phóng được tài nguyên',
                                f'Không bắt đầu {action} khi session ảnh chưa được giải phóng: {exc}')
            return False
        return True

    def _span(self, forward: bool) -> tuple[int, ...] | None:
        """Frames a span Track would cover from the current frame, or None when
        it is not on a keyframe or there is nothing between it and the next one."""
        keyframes = self.edits.keyframes
        if self.clip is None or self.current_frame not in keyframes:
            return None
        span = vo.track_span(keyframes, self.current_frame, self.clip.count, forward)
        return span if any(frame not in keyframes for frame in span) else None

    def track(self, frames: tuple[int, ...] | None = None):
        """Track the whole clip, or only ``frames`` (a span, in tracking order)."""
        if not self.clip or self.is_busy:
            return
        if not any(np.any(labels) for labels in self.edits.keyframes.values()):
            QMessageBox.warning(self, 'Chưa có mask',
                                'Hãy Restore để tô object trên ít nhất một frame, rồi bấm Track.')
            return
        # HQ-SAM2 runs in its own process, whatever the CPU/CUDA choice.
        if not self._free_image_models('Track'):
            return
        report_dir = create_run(self.output_dir, 'track')
        self._last_track_diagnostics = str(report_dir)
        previous = self.tracked if frames is None and self._same_keyframes_as_track() else None
        self._run_job('track', VideoJob('track', self.clip, dict(self.edits.keyframes),
                                        report_dir=str(report_dir), previous=previous,
                                        frames=frames))

    def track_span(self, forward: bool):
        """◀ Track lùi / Track tiến ▶: from the current keyframe to the next one."""
        span = self._span(forward)
        if span is not None:
            self.track(span)

    def export(self):
        if not self.clip or self.is_busy:
            return
        if self.tracked is None:
            QMessageBox.warning(self, 'Chưa Track', 'Hãy Track toàn video trước khi Export.')
            return
        # The object tint is display only; Settings controls the export format.
        bg_mode = APP_CONFIG['video']['export_background']
        path = export_destination(self.output_dir, self.clip.source, bg_mode)
        engine = self.engine_getter()
        gpu_active = bool(engine is not None and getattr(engine, 'compute_mode', '') == 'gpu')
        if engine is None:
            from ..engine.video_sidecar import selected_video_device
            gpu_active = selected_video_device().startswith('cuda:')
        mode = str(APP_CONFIG.get('video', {}).get('matting_export', 'auto')).lower()
        matting = ('vitmatte_sam' if gpu_active else 'binary') if mode == 'auto' else mode
        if not gpu_active and matting == 'vitmatte_sam':
            QMessageBox.warning(self, 'Matting không tương thích',
                                'Đã chọn CPU nhưng tinh chỉnh viền video yêu cầu GPU. '
                                'Hãy chuyển lựa chọn tinh chỉnh viền video sang Auto hoặc OFF.')
            return
        # ViTMatte-SAM runs on the GPU in its own process (Guided Edit may have loaded HQ-SAM here).
        if matting == 'vitmatte_sam' and not self._free_image_models('Export'):
            return
        report_dir = create_run(self.output_dir, 'export')
        self._last_export_diagnostics = str(report_dir)
        export_masks = tuple(vo.export_alpha(self._labels(i)) for i in range(self.clip.count))
        holds = self._unchanged_holds() if APP_CONFIG['video']['reuse_held_frames'] else {}
        self._run_job('export', VideoJob('export', self.clip, export_masks,
                                         str(path), background=bg_mode,
                                         matting=matting, report_dir=str(report_dir),
                                         foreground=APP_CONFIG['output']['foreground_estimation'],
                                         holds=holds))

    def _unchanged_holds(self) -> dict[int, vh.Hold]:
        """Holds whose frame still shows its source's mask moved by the shake:
        an edit after Track (keyframe, Gộp/Bỏ) breaks the holds it touches."""
        return {frame: hold for frame, hold in self.held.items()
                if np.array_equal(self._labels(frame),
                                  vh.shifted(self._labels(hold.source), hold.dx, hold.dy))}

    # -- output folder --------------------------------------------------------------------
    def _show_output_dir(self):
        self.path_label.setText(f'Lưu vào: {self.output_dir}')

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, 'Chọn folder lưu video', self.output_dir)
        if folder:
            self.output_dir = folder
            self._cache_output_dirs.add(folder)
            self._show_output_dir()

    def open_output(self):
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if os.name == 'nt':
            os.startfile(path)
        else:
            import subprocess
            subprocess.Popen(['xdg-open', str(path)])

    def _cleanup_legacy_caches(self):
        errors = []
        for folder in self._cache_output_dirs:
            try:
                cleanup_legacy_matting_cache(folder)
            except OSError as exc:
                errors.append(f'{folder}: {exc}')
        if errors:
            QMessageBox.warning(self, 'Không thể xóa cache matting cũ', '\n'.join(errors))

    def shutdown(self):
        """Called by MainWindow only after all workers have stopped."""
        self._play_timer.stop()
        self._end_trim()
        if self.clip and not self.is_busy:
            release_clip_temp(self.clip.temp_dir)
            self.clip = None
        self._cleanup_legacy_caches()
