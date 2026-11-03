"""Video editor: Guided keyframes and the native HQ-SAM2 Video Predictor."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QProgressBar, QSlider, QVBoxLayout, QWidget)

from ..app_config import APP_BASE_DIR, APP_CONFIG
from ..engine.video_diagnostics import create_run
from ..engine.video_pipeline import (VideoCanceled, VideoClip, VideoFormatError,
                                     blank_alpha, export_video, extract_clip,
                                     paint_alpha, release_clip_temp, track_masks)
from ..styles import (CANCEL_BUTTON_STYLE, EDIT_BLACK_BUTTON_STYLE,
                      EDIT_ERASE_BUTTON_STYLE, EDIT_MODE_BUTTON_STYLE,
                      EDIT_RESTORE_BUTTON_STYLE, apply_button_style)
from ..widgets.edit_mask_canvas import EditMaskCanvas
from ..widgets.upload_preview_canvas import UploadPreviewCanvas
from ..workers import GuidedEditWorker


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


class VideoCutPage(QWidget):
    backRequested = Signal()

    def __init__(self, engine_getter):
        super().__init__()
        self.engine_getter = engine_getter
        self.clip: VideoClip | None = None
        self.current_frame = 0
        self.keyframes: dict[int, np.ndarray] = {}
        self.tracked_masks: tuple[np.ndarray, ...] | None = None
        self.undo_stack: list[tuple[int, np.ndarray]] = []
        self.redo_stack: list[tuple[int, np.ndarray]] = []
        self.operation = 'restore'  # A blank mask needs a foreground selection first.
        self.submode = 'guided'
        self.draw_mode = 'paint'
        self.output_dir = str((APP_BASE_DIR / APP_CONFIG.get('output', {}).get(
            'video_output_dir', 'output/video_saved')).resolve())
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._job_name = ''
        self._busy = False

        self.btn_back = QPushButton('← Trang chủ')
        self.btn_back.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
        self.title = QLabel('Cắt nền video (≤10 giây)')
        self.title.setStyleSheet('font-size: 22px; font-weight: 750; color: white;')
        head = QHBoxLayout()
        head.addWidget(self.btn_back)
        head.addStretch(1)
        head.addWidget(self.title)
        head.addStretch(1)

        self.status = QLabel('Kéo-thả video hoặc bấm khung để chọn (tối đa 10 giây).')
        self.status.setStyleSheet('color: white; font-size: 13px;')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.path_info = QLabel(f'Đường dẫn lưu: {self.output_dir}')
        self.path_info.setStyleSheet('color: #DCE5EE; font-size: 11px;')
        self.path_info.setAlignment(Qt.AlignCenter)
        self.path_info.setWordWrap(True)

        self.upload_canvas = UploadPreviewCanvas('Upload video')
        self.edit_canvas = EditMaskCanvas()
        # Video adds a timeline + export row; do not push controls offscreen
        # on the existing 900x620 minimum window. Image Edit remains unchanged.
        self.edit_canvas.setMinimumHeight(220)
        self.edit_canvas.hide()

        self.btn_undo = QPushButton('Undo')
        self.btn_redo = QPushButton('Redo')
        self.btn_paint = QPushButton('Paint mode')
        self.btn_box = QPushButton('Box mode')
        self.btn_guided = QPushButton('Guided')
        self.btn_manual = QPushButton('Manual')
        self.btn_clear_video = QPushButton('Xóa video')
        for b in (self.btn_paint, self.btn_box, self.btn_guided, self.btn_manual):
            b.setCheckable(True)
            b.setStyleSheet(EDIT_MODE_BUTTON_STYLE)
        for b in (self.btn_undo, self.btn_redo, self.btn_clear_video):
            b.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
        self.btn_paint.setChecked(True)
        self.btn_guided.setChecked(True)
        top_tools = QHBoxLayout()
        top_tools.addWidget(self.btn_undo)
        top_tools.addWidget(self.btn_redo)
        top_tools.addWidget(self.btn_clear_video)
        top_tools.addStretch(1)
        top_tools.addWidget(self.btn_paint)
        top_tools.addWidget(self.btn_box)
        top_tools.addStretch(1)
        top_tools.addWidget(self.btn_guided)
        top_tools.addWidget(self.btn_manual)
        self.top_tools = QWidget()
        self.top_tools.setLayout(top_tools)
        self.top_tools.hide()

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline_label = QLabel('00:00 / 00:00 · Frame 0/0')
        self.timeline_label.setStyleSheet('color: #FFFFFF;')
        self.btn_play = QPushButton('▶ Play')
        self.btn_play.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
        timeline_row = QHBoxLayout()
        timeline_row.addWidget(self.btn_play)
        timeline_row.addWidget(self.timeline, 1)
        timeline_row.addWidget(self.timeline_label)
        self.timeline_widget = QWidget()
        self.timeline_widget.setLayout(timeline_row)
        self.timeline_widget.hide()

        self.btn_fit = QPushButton('Fit')
        self.btn_erase = QPushButton('Erase')
        self.btn_clear = QPushButton('Clear Mask')
        self.btn_restore = QPushButton('Restore')
        self.btn_brush = QPushButton('Brush size')
        for b in (self.btn_erase, self.btn_restore):
            b.setCheckable(True)
        self.btn_restore.setChecked(True)
        self.btn_erase.setStyleSheet(EDIT_ERASE_BUTTON_STYLE)
        self.btn_clear.setStyleSheet(EDIT_ERASE_BUTTON_STYLE)
        self.btn_restore.setStyleSheet(EDIT_RESTORE_BUTTON_STYLE)
        for b in (self.btn_fit, self.btn_brush):
            b.setStyleSheet(EDIT_BLACK_BUTTON_STYLE)
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        for b in (self.btn_fit, self.btn_erase, self.btn_clear, self.btn_restore, self.btn_brush):
            action_row.addWidget(b)
        action_row.addStretch(1)
        self.edit_actions = QWidget()
        self.edit_actions.setLayout(action_row)
        self.edit_actions.hide()

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(4, 120)
        self.brush_slider.setValue(28)
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel('Brush size'))
        brush_row.addWidget(self.brush_slider, 1)
        self.brush_widget = QWidget()
        self.brush_widget.setLayout(brush_row)
        self.brush_widget.hide()

        self.btn_track = QPushButton('Track')
        self.btn_export = QPushButton('Export')
        self.btn_config = QPushButton('Đường dẫn lưu')
        self.btn_open_folder = QPushButton('Mở folder đã lưu')
        self.btn_stop = QPushButton('Dừng')
        self.btn_stop.setStyleSheet(CANCEL_BUTTON_STYLE)
        for b in (self.btn_track, self.btn_export, self.btn_config, self.btn_open_folder):
            apply_button_style(b, primary=b is self.btn_track)
        self.btn_stop.hide()
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        for b in (self.btn_track, self.btn_export, self.btn_config, self.btn_open_folder, self.btn_stop):
            export_row.addWidget(b)
        export_row.addStretch(1)
        self.export_actions = QWidget()
        self.export_actions.setLayout(export_row)
        self.export_actions.hide()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 10)
        outer.setSpacing(5)
        outer.addLayout(head)
        outer.addWidget(self.status)
        outer.addWidget(self.path_info)
        outer.addWidget(self.top_tools)
        outer.addWidget(self.upload_canvas, 1)
        outer.addWidget(self.edit_canvas, 1)
        outer.addWidget(self.timeline_widget)
        outer.addWidget(self.brush_widget)
        outer.addWidget(self.edit_actions)
        outer.addWidget(self.export_actions)
        outer.addWidget(self.progress_bar)

        self.btn_back.clicked.connect(self._back)
        self.upload_canvas.clicked.connect(self._choose_video)
        self.upload_canvas.filesDropped.connect(self._dropped)
        self.upload_canvas.cancelRequested.connect(self.stop_task)
        self.btn_clear_video.clicked.connect(self.clear_video)
        self.timeline.valueChanged.connect(self._set_frame)
        self.btn_play.clicked.connect(self._toggle_play)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self.btn_erase.clicked.connect(lambda: self._set_operation('erase'))
        self.btn_restore.clicked.connect(lambda: self._set_operation('restore'))
        self.btn_clear.clicked.connect(self.clear_mask)
        self.btn_fit.clicked.connect(self.edit_canvas.reset_view)
        self.btn_brush.clicked.connect(self._toggle_brush)
        self.brush_slider.valueChanged.connect(self.edit_canvas.set_brush_size)
        self.btn_paint.clicked.connect(lambda: self._set_draw_mode('paint'))
        self.btn_box.clicked.connect(lambda: self._set_draw_mode('box'))
        self.btn_guided.clicked.connect(lambda: self._set_submode('guided'))
        self.btn_manual.clicked.connect(lambda: self._set_submode('manual'))
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.edit_canvas.strokeFinished.connect(lambda points: self._edit(points=points))
        self.edit_canvas.boxFinished.connect(lambda box: self._edit(box=box))
        self.btn_track.clicked.connect(self.track)
        self.btn_export.clicked.connect(self.export)
        self.btn_config.clicked.connect(self.choose_output)
        self.btn_open_folder.clicked.connect(self.open_output)
        self.btn_stop.clicked.connect(self.stop_task)
        shortcut_undo = QShortcut(QKeySequence('Ctrl+Z'), self)
        shortcut_undo.activated.connect(self.undo)
        shortcut_redo = QShortcut(QKeySequence('Ctrl+Y'), self)
        shortcut_redo.activated.connect(self.redo)
        self._shortcut_undo = shortcut_undo
        self._shortcut_redo = shortcut_redo
        self._sync_actions()

    @property
    def is_busy(self):
        return self._busy or (self._thread is not None and self._thread.isRunning())

    def _back(self):
        if self.is_busy:
            QMessageBox.information(self, 'Đang xử lý', 'Hãy bấm Dừng và chờ tác vụ kết thúc.')
            return
        self._play_timer.stop()
        self.backRequested.emit()

    def _choose_video(self):
        if self.is_busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Chọn video tối đa 10 giây', '',
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
            self.clear_video()
        self.upload_canvas.set_loading_mode('Đang kiểm tra và đọc video...', allow_cancel=True)
        self._run_job('upload', VideoJob('upload', path))

    def clear_video(self):
        if self.is_busy:
            return
        self._play_timer.stop()
        if self.clip:
            release_clip_temp(self.clip.temp_dir)
        self.clip = None
        self.current_frame = 0
        self.keyframes.clear()
        self.tracked_masks = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.edit_canvas.clear_image()
        self.edit_canvas.hide()
        self.upload_canvas.set_upload_mode()
        self.upload_canvas.show()
        for widget in (self.top_tools, self.timeline_widget, self.edit_actions, self.export_actions):
            widget.hide()
        self.brush_widget.hide()
        self.status.setText('Kéo-thả video hoặc bấm khung để chọn (tối đa 10 giây).')
        self._sync_actions()

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
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread:
            thread.deleteLater()
        self._busy = False
        self._busy_ui(False)

    def _busy_ui(self, busy: bool):
        self.btn_stop.setVisible(busy and self.clip is not None)
        if busy:
            self.btn_stop.setEnabled(True)
        self.btn_back.setEnabled(not busy)
        self.btn_clear_video.setEnabled(not busy)
        self.timeline.setEnabled(not busy)
        self.btn_play.setEnabled(not busy)
        for btn in (self.btn_track, self.btn_export, self.btn_config, self.btn_open_folder,
                    self.btn_undo, self.btn_redo, self.btn_erase, self.btn_restore,
                    self.btn_clear, self.btn_paint, self.btn_box, self.btn_manual,
                    self.btn_guided, self.btn_brush):
            btn.setEnabled(not busy)
        self.edit_canvas.set_processing(busy and self.clip is not None)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setValue(0)
        else:
            self._sync_actions()

    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(max(0, min(100, value)))
        self.status.setText(message)
        if self.clip:
            self.edit_canvas.set_processing_progress(value, message)
        else:
            self.upload_canvas.set_loading_mode(message, max(0, value), 100, allow_cancel=True)

    def stop_task(self):
        if self._worker is None:
            return
        self._worker.cancel()
        self.status.setText('Đã yêu cầu Dừng. Chờ worker kết thúc frame hiện tại...')
        self.btn_stop.setEnabled(False)
        if self._job_name == 'upload':
            self.upload_canvas.set_cancel_waiting_mode()

    def _on_cancel(self):
        diag=getattr(self, '_last_track_diagnostics' if self._job_name=='track'
                     else '_last_export_diagnostics', '')
        self.status.setText('Đã dừng. Không áp dụng kết quả dở dang. Log: '+str(diag))
        if self._job_name == 'upload':
            self.upload_canvas.set_upload_mode()

    def _on_error(self, message: str):
        diag=getattr(self, '_last_track_diagnostics' if self._job_name=='track'
                     else '_last_export_diagnostics', '')
        self.status.setText('Tác vụ video thất bại. Log: '+str(diag))
        if self._job_name == 'upload':
            self.upload_canvas.set_upload_mode()
        QMessageBox.warning(self, 'Lỗi video', message)

    def _on_job_done(self, result):
        if self._job_name == 'upload':
            self.clip = result
            self.current_frame = 0
            self.keyframes.clear()
            self.tracked_masks = None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.upload_canvas.hide()
            self.edit_canvas.show()
            for widget in (self.top_tools, self.timeline_widget, self.edit_actions, self.export_actions):
                widget.show()
            self.timeline.blockSignals(True)
            self.timeline.setRange(0, self.clip.count - 1)
            self.timeline.setValue(0)
            self.timeline.blockSignals(False)
            self._set_operation('restore')
            self._show_frame()
            self.status.setText(f'{Path(self.clip.source).name}: {self.clip.duration:.2f}s, '
                                f'{self.clip.count} frame @ {self.clip.output_fps:.2f}fps. '
                                'Restore vùng muốn giữ, sau đó bấm Track.')
        elif self._job_name == 'track':
            self.tracked_masks = result
            self._show_frame()
            self.status.setText(f'HQ-SAM2 Track {len(result)} frame. Log: '
                                f'{getattr(self, "_last_track_diagnostics", "")}. Có thể Export.')
        elif self._job_name == 'export':
            self.status.setText(f'Đã export: {result}. Log: '
                                f'{getattr(self, "_last_export_diagnostics", "")}')
            QMessageBox.information(self, 'Đã export video',
                                    f'Đã lưu MP4 nền màu (không có alpha trong MP4):\n{result}')
        elif self._job_name == 'guided':
            alpha = np.asarray(result['final_alpha'], dtype=np.uint8)
            self._commit_mask(alpha)
            self.status.setText('Guided Edit xong. Bấm Track để cập nhật các frame.')
        self._sync_actions()

    def _current_mask(self):
        if self.clip is None:
            return None
        if self.current_frame in self.keyframes:
            return self.keyframes[self.current_frame]
        if self.tracked_masks is not None:
            return self.tracked_masks[self.current_frame]
        return blank_alpha(self.clip)

    def _show_frame(self):
        if not self.clip:
            return
        idx = self.current_frame
        self.edit_canvas.set_image_alpha(self.clip.frame_paths[idx], self._current_mask())
        t = idx / self.clip.output_fps
        self.timeline_label.setText(f'{t:05.2f}s / {self.clip.duration:.2f}s · Frame {idx+1}/{self.clip.count}')
        self._sync_actions()

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
        else:
            self._play_timer.start(max(40, round(1000 / min(12.0, self.clip.output_fps))))
            self.btn_play.setText('❚❚ Pause')

    def _play_tick(self):
        if not self.clip or self.is_busy or self.current_frame + 1 >= self.clip.count:
            self._play_timer.stop()
            self.btn_play.setText('▶ Play')
            return
        self.timeline.setValue(min(self.clip.count - 1,
                                   self.current_frame + max(1, round(self.clip.output_fps / 12))))

    def _set_operation(self, op: str):
        self.operation = op
        self.btn_erase.setChecked(op == 'erase')
        self.btn_restore.setChecked(op == 'restore')
        self.edit_canvas.set_operation(op)
        self._sync_actions()

    def _set_draw_mode(self, mode: str):
        self.draw_mode = mode
        self.btn_paint.setChecked(mode == 'paint')
        self.btn_box.setChecked(mode == 'box')
        self.edit_canvas.set_draw_mode(mode)
        if mode == 'box':
            self.brush_widget.hide()
        self.btn_brush.setVisible(mode == 'paint')

    def _set_submode(self, mode: str):
        self.submode = mode
        self.btn_guided.setChecked(mode == 'guided')
        self.btn_manual.setChecked(mode == 'manual')

    def _toggle_brush(self):
        if self.draw_mode == 'paint':
            self.brush_widget.setVisible(not self.brush_widget.isVisible())

    def _edit(self, points=None, box=None):
        if not self.clip or self.is_busy:
            return
        if not points and box is None:
            return
        if self.submode == 'manual':
            new_alpha = paint_alpha(self._current_mask(), points or [],
                                    self.brush_slider.value(), self.operation, box)
            self._commit_mask(new_alpha)
            self.edit_canvas.clear_stroke()
            return
        engine = self.engine_getter()
        if engine is None:
            QMessageBox.warning(self, 'Chưa tải model', 'Model chưa sẵn sàng.')
            self.edit_canvas.clear_stroke()
            return
        worker = GuidedEditWorker(engine=engine,
                                  image_path=self.clip.frame_paths[self.current_frame],
                                  current_alpha=self._current_mask().copy(),
                                  stroke_points=points or [],
                                  prompt_box=box,
                                  brush_size=self.brush_slider.value(),
                                  operation=self.operation)
        self.edit_canvas.clear_stroke()
        self._run_job('guided', worker)

    def _commit_mask(self, alpha: np.ndarray):
        if self.clip is None:
            return
        old = self._current_mask().copy()
        if np.array_equal(old, alpha):
            self.status.setText('Mask không thay đổi.')
            return
        self.undo_stack.append((self.current_frame, old))
        self.undo_stack = self.undo_stack[-20:]
        self.redo_stack.clear()
        self.keyframes[self.current_frame] = np.asarray(alpha, np.uint8).copy()
        self.tracked_masks = None
        self.edit_canvas.set_alpha_array(alpha, copy=True)
        self.status.setText(f'Đã sửa frame {self.current_frame+1}. Cần Track lại.')
        self._sync_actions()

    def clear_mask(self):
        if self.clip and not self.is_busy:
            self._commit_mask(blank_alpha(self.clip))

    def undo(self):
        if not self.clip or self.is_busy or not self.undo_stack:
            return
        frame, old = self.undo_stack.pop()
        current = self.keyframes.get(frame, blank_alpha(self.clip)).copy()
        self.redo_stack.append((frame, current))
        self.keyframes[frame] = old
        self.tracked_masks = None
        self.timeline.setValue(frame)
        self._show_frame()

    def redo(self):
        if not self.clip or self.is_busy or not self.redo_stack:
            return
        frame, state = self.redo_stack.pop()
        current = self.keyframes.get(frame, blank_alpha(self.clip)).copy()
        self.undo_stack.append((frame, current))
        self.keyframes[frame] = state
        self.tracked_masks = None
        self.timeline.setValue(frame)
        self._show_frame()

    def track(self):
        if not self.clip or self.is_busy:
            return
        if not self.keyframes or not any(np.any(m > 0) for m in self.keyframes.values()):
            QMessageBox.warning(self, 'Chưa có mask',
                                'Hãy dùng Restore chọn vùng cần giữ trên ít nhất một frame, rồi bấm Track.')
            return
        # The GPU video worker owns a separate CUDA context: release the image
        # engine's Guided and automatic sessions first (reloaded lazily later).
        engine = self.engine_getter()
        if engine is not None:
            try:
                engine.release_guided_resources()
                engine.release_auto_sessions()
            except Exception as exc:
                QMessageBox.warning(self, 'Không giải phóng được VRAM',
                                    f'Không bắt đầu Track khi session GPU chưa được giải phóng: {exc}')
                return
        snapshot = {k: v.copy() for k, v in self.keyframes.items()}
        report_dir = create_run(self.output_dir, 'track')
        self._last_track_diagnostics = str(report_dir)
        self._run_job('track', VideoJob('track', self.clip, snapshot,
                                        report_dir=str(report_dir)))

    def export(self):
        if not self.clip or self.is_busy:
            return
        if self.tracked_masks is None:
            QMessageBox.warning(self, 'Chưa Track', 'Hãy Track toàn video trước khi Export.')
            return
        folder = Path(self.output_dir)
        base = Path(self.clip.source).stem + '_cut'
        path = folder / f'{base}.mp4'
        suffix = 2
        while path.exists():
            path = folder / f'{base}_{suffix}.mp4'
            suffix += 1
        bg_mode = self.edit_canvas._background_mode
        # MP4 has no reliable RGBA alpha; green is explicit chroma-key default.
        bg_mode = 'green' if bg_mode == 'transparent' else bg_mode
        report_dir = create_run(self.output_dir, 'export')
        engine = self.engine_getter()
        gpu_active = bool(engine is not None and getattr(engine, 'compute_mode', '') == 'gpu')
        if engine is None:
            from ..portable_device_settings import load_machine_selection
            gpu_active = (load_machine_selection().get('mode') == 'gpu' or
                          APP_CONFIG.get('compute',{}).get('mode') == 'gpu')
        mode = str(APP_CONFIG.get('video', {}).get('matting_export', 'auto')).lower()
        matting = ('vitmatte_sam' if gpu_active else 'binary') if mode == 'auto' else mode
        self._last_export_diagnostics = str(report_dir)
        self._run_job('export', VideoJob('export', self.clip, self.tracked_masks,
                                         str(path), background=bg_mode,
                                         matting=matting, report_dir=str(report_dir)))

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, 'Chọn folder lưu video', self.output_dir)
        if folder:
            self.output_dir = folder
            self.path_info.setText(f'Đường dẫn lưu: {self.output_dir}')

    def open_output(self):
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if os.name == 'nt':
            os.startfile(path)
        else:
            import subprocess
            subprocess.Popen(['xdg-open', str(path)])

    def _sync_actions(self):
        if self._busy:
            return
        self.btn_undo.setEnabled(bool(self.undo_stack))
        self.btn_redo.setEnabled(bool(self.redo_stack))
        self.btn_export.setEnabled(self.tracked_masks is not None)

    def shutdown(self):
        """Called by MainWindow only after all workers have stopped."""
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self.clip and not self.is_busy:
            release_clip_temp(self.clip.temp_dir)
            self.clip = None
