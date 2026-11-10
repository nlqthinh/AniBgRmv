import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMessageBox

from ..app_config import GUIDED_EDIT_TIMEOUT_MS
from ..edit_history import EditHistory
from ..workers import GuidedEditWorker, MattingWorker
from ..qt_threads import thread_is_running, thread_is_valid

# Model inference outputs kept on a prediction after Edit/Guided/Matting:
# editing changes final_alpha, not what the automatic models produced.
INFERENCE_DEBUG_KEYS = (
    "unet_alpha_768",
    "unet_alpha_original",
    "letterbox_meta",
    "image_letterbox_pil",
    "valid_mask_768",
    "trimap_768",
    "matte_alpha_768",
    "trimap_stats",
    "trimap_stats_valid",
)
# Undo keeps at most 30 full-resolution alphas, and ~96 MiB of them (4K/8K images).
UNDO_STEPS, UNDO_BUDGET_BYTES = 30, 96 * 1024 * 1024


def new_edit_history() -> EditHistory:
    return EditHistory(UNDO_STEPS, UNDO_BUDGET_BYTES)


def guided_description(engine) -> str:
    policy = engine.guided_policy
    if policy.refine_each_stroke:
        return "HQ-SAM chọn vùng, ViTMatte-SAM tinh chỉnh biên sau mỗi nét"
    return "HQ-SAM chọn vùng; bấm Matting một lần sau khi sửa"


def matting_available(engine) -> bool:
    """The Matting button exists for CPU, where strokes skip per-stroke matting."""
    return (engine is not None and not engine.strict_gpu
            and engine.cpu_matting_model != "off")


class SingleEditControllerMixin:
    """Edits the alpha of ONE open image: the single page's image or the
    current image of a batch. ImageEditPage provides the widgets; a page
    provides ``_sync_file_buttons(enabled)`` for its save/navigation buttons."""

    UNSAVED_MESSAGE = "Ảnh đang mở có chỉnh sửa chưa lưu. Bỏ các chỉnh sửa đó?"

    def enter_edit_mode(self):
        """Open ``current_prediction`` in the editor (right after a cut).

        The chosen tools (Erase/Restore, Guided/Manual, Paint/Box) carry over
        from the previous image; Undo/Redo starts empty."""
        if not self.current_prediction:
            return
        self.in_edit_mode = True
        self.edit_dirty = False
        self.edit_history.clear()
        self._manual_last_point = None
        self.view.setCurrentWidget(self.edit_canvas)
        self.edit_canvas.set_operation(self.edit_operation)
        self.edit_canvas.set_draw_mode(self.edit_draw_mode)
        self.edit_canvas.set_stroke_preview_enabled(self.edit_submode == "guided")
        self.edit_canvas.set_brush_size(self.brush_slider.value())
        self.edit_canvas.reset_view()
        self.refresh_edit_canvas()
        self.set_tools_enabled(True)
        self.sync_engine_controls()

    def sync_engine_controls(self):
        """Matting (CPU only) and the Guided description follow the current
        engine, which Settings (CPU/GPU, models) may have replaced."""
        engine = self.engine_getter()
        self.btn_matting.setVisible(self.in_edit_mode and engine is not None and not engine.strict_gpu)
        self.btn_matting.setEnabled(self.btn_erase.isEnabled() and matting_available(engine))
        self.update_description()

    def close_editor(self):
        """Close the open image: free its buffers and, as leaving Edit always
        did, the Guided models (never while a Guided worker still runs)."""
        self.in_edit_mode = False
        self.edit_dirty = False
        self.edit_history.clear()
        self.edit_canvas.clear_image()
        if not self.is_guided_thread_running():
            engine = self.engine_getter()
            if engine is not None:
                engine.release_guided_resources()
        self.canvas.set_upload_mode()
        self.view.setCurrentWidget(self.canvas)
        self.btn_stop_guided.setVisible(False)
        self.set_tools_enabled(False)

    def editor_busy(self) -> bool:
        """A Guided/Matting job runs: the open image must stay open."""
        busy = self.edit_canvas.processing or self.guided_running or self.is_guided_thread_running()
        if busy:
            QMessageBox.information(self, "Đang xử lý", "Vui lòng chờ Guided Edit xử lý xong (hoặc bấm Dừng).")
        return busy

    def has_unsaved_edits(self) -> bool:
        return self.in_edit_mode and self.edit_dirty

    def confirm_discard_unsaved(self) -> bool:
        """True when no unsaved edit would be lost, or the user agrees to lose it."""
        if not self.has_unsaved_edits():
            return True
        box = QMessageBox(QMessageBox.Icon.Warning, "Chưa lưu", self.UNSAVED_MESSAGE, parent=self)
        discard = box.addButton("Bỏ chỉnh sửa", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(box.addButton("Hủy", QMessageBox.ButtonRole.RejectRole))
        box.exec()
        return box.clickedButton() is discard

    def preserve_inference_debug(self, edited: dict, previous: dict) -> dict:
        """Keep diagnostics from the actual UNet -> ViTMatte-UNet inference."""
        for key in INFERENCE_DEBUG_KEYS:
            if key in previous:
                edited[key] = previous[key]

        # Keep edit-specific notes as well as the underlying inference notes.
        notes = list(previous.get("safety_notes") or [])
        for note in edited.get("safety_notes") or []:
            if note not in notes:
                notes.append(note)
        edited["safety_notes"] = notes
        return edited

    def clone_prediction_with_alpha(self, prediction: dict, alpha: np.ndarray) -> dict:
        engine = self.engine_getter()
        if engine is None:
            return dict(prediction)
        edited = engine.build_prediction_from_alpha(
            image_path=prediction["input_path"],
            final_alpha=np.asarray(alpha).astype(np.uint8),
            elapsed_sec=prediction.get("elapsed_sec", 0.0),
            source="ui_edit_clone",
        )
        return self.preserve_inference_debug(edited, prediction)

    def refresh_edit_canvas(self):
        if not self.current_prediction:
            return
        self.edit_canvas.set_image_alpha(
            self.current_prediction["input_path"],
            self.current_prediction["final_alpha"],
        )

    def update_description(self):
        """What the current tool does, under the background controls."""
        what = ("Erase: xóa vùng khỏi ảnh cắt." if self.edit_operation == "erase" else
                "Restore: lấy lại vùng bị cắt mất (vùng đã cắt hiện mờ để dễ thấy).")
        how = "Kéo cọ lên vùng." if self.edit_draw_mode == "paint" else "Kéo khung quanh vùng."
        engine = self.engine_getter()
        if self.edit_submode == "manual":
            way = "Manual: sửa đúng nét vẽ, không chạy model."
        elif engine is not None:
            way = f"Guided: {guided_description(engine)}."
        else:
            way = "Guided: HQ-SAM tự tìm vùng."
        self.description_label.setText(
            f"{what} {how} {way}\nCuộn chuột: zoom · Shift + kéo hoặc chuột giữa: di chuyển ảnh.")

    def set_edit_operation(self, operation: str):
        self.edit_operation = operation
        self.btn_erase.setChecked(operation == "erase")
        self.btn_restore.setChecked(operation == "restore")
        self.edit_canvas.set_operation(operation)
        self._manual_last_point = None
        self.update_description()

    def set_edit_submode(self, submode: str):
        self.edit_submode = submode
        self._manual_last_point = None
        self.edit_canvas.set_stroke_preview_enabled(submode == "guided")
        self.btn_guided.setChecked(submode == "guided")
        self.btn_manual.setChecked(submode == "manual")
        self.update_description()

    def set_edit_draw_mode(self, mode: str):
        """Paint vs rectangle input; independent of Guided/Manual segmentation."""
        if mode not in ("paint", "box"):
            raise ValueError("Unknown drawing mode")
        if (self.edit_canvas.processing or self.guided_running
                or self.edit_canvas.is_drawing or self.is_guided_thread_running()):
            self.btn_paint_mode.setChecked(self.edit_draw_mode == "paint")
            self.btn_box_mode.setChecked(self.edit_draw_mode == "box")
            return
        self.edit_draw_mode = mode
        self.edit_canvas.set_draw_mode(mode)
        self.btn_paint_mode.setChecked(mode == "paint")
        self.btn_box_mode.setChecked(mode == "box")
        self.brush_slider.setEnabled(self.in_edit_mode and mode == "paint")
        self.update_description()

    def on_brush_size_changed(self, value: int):
        self.edit_canvas.set_brush_size(value)
        self.brush_label.setText(f"Cỡ cọ: {value} px")

    def _current_alpha_copy(self) -> np.ndarray:
        return np.asarray(self.current_prediction["final_alpha"]).copy()

    def push_undo_state(self):
        if not self.current_prediction:
            return
        self.edit_history.push(self._current_alpha_copy())
        self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        self.btn_undo.setEnabled(self.edit_history.can_undo)
        self.btn_redo.setEnabled(self.edit_history.can_redo)

    def clear_edit_mask(self):
        """Clear the full alpha matte, not the source image or SAM's model cache.

        A single Undo snapshot restores the exact previous alpha (including
        soft edges). Do not interrupt a running Guided worker or an active
        brush stroke: either could subsequently overwrite the cleared mask.
        """
        if (not self.in_edit_mode or self.current_prediction is None
                or self.edit_canvas.processing or self.guided_running
                or self.edit_stroke_active or self.edit_canvas.is_drawing
                or self.is_guided_thread_running()):
            return

        alpha = np.asarray(self.current_prediction["final_alpha"])
        if not np.any(alpha):
            self.status_label.setText("Mask hiện đã trống (alpha = 0).")
            return

        # Build first, so a failed disk read does not modify Undo/Redo history.
        cleared = self.clone_prediction_with_alpha(
            self.current_prediction, np.zeros_like(alpha, dtype=np.uint8)
        )
        self.push_undo_state()  # exactly one Undo step, clears the Redo branch
        self.current_prediction = cleared
        self.edit_dirty = True
        self._manual_last_point = None
        self.edit_canvas.clear_stroke()
        self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        self.status_label.setText("Đã xóa toàn bộ mask. Bấm Undo (Ctrl+Z) để khôi phục.")

    def undo_edit(self):
        if (not self.in_edit_mode or not self.current_prediction or
                not self.edit_history.can_undo or self.guided_running or
                self.is_guided_thread_running()):
            return
        alpha = self.edit_history.undo(lambda _restored: self._current_alpha_copy())
        self.current_prediction = self.clone_prediction_with_alpha(self.current_prediction, alpha)
        self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        self.status_label.setText("Undo mask edit.")

    def redo_edit(self):
        if (not self.in_edit_mode or not self.current_prediction or
                not self.edit_history.can_redo or self.guided_running or
                self.is_guided_thread_running()):
            return
        alpha = self.edit_history.redo(lambda _restored: self._current_alpha_copy())
        self.current_prediction = self.clone_prediction_with_alpha(self.current_prediction, alpha)
        self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        self.status_label.setText("Redo mask edit.")

    def on_edit_box_finished(self, box: tuple):
        """Commit one rectangle as one Undo step; never turn it into a brush stroke."""
        if (not self.in_edit_mode or not self.current_prediction
                or self.edit_draw_mode != "box" or self.guided_running
                or self.is_guided_thread_running()):
            return
        h, w = np.asarray(self.current_prediction["final_alpha"]).shape[:2]
        x0, y0, x1, y1 = (int(v) for v in box)
        x0, x1 = sorted((max(0, min(w-1, x0)), max(0, min(w-1, x1))))
        y0, y1 = sorted((max(0, min(h-1, y0)), max(0, min(h-1, y1))))
        if x1 <= x0 or y1 <= y0:
            return
        rect = (x0, y0, x1, y1)  # inclusive coordinates
        if self.edit_submode == "guided":
            self.push_undo_state()
            self.start_guided_edit([], box=rect)
            return

        original = np.asarray(self.current_prediction["final_alpha"])
        edited = original.copy()
        value = 0 if self.edit_operation == "erase" else 255
        if np.all(edited[y0:y1+1, x0:x1+1] == value):
            return
        edited[y0:y1+1, x0:x1+1] = value
        updated = self.clone_prediction_with_alpha(self.current_prediction, edited)
        self.push_undo_state()
        self.current_prediction = updated
        self.edit_dirty = True
        self.refresh_edit_canvas()
        self.status_label.setText(f"Manual Box {self.edit_operation}: ({x0},{y0})–({x1},{y1}).")

    def on_edit_stroke_started(self):
        if not self.current_prediction or self.edit_draw_mode != "paint":
            return
        self.edit_stroke_active = True
        self.push_undo_state()  # one full alpha snapshot per stroke, NOT per mouse event
        self._manual_last_point = None
        if self.edit_submode == "manual":
            alpha = np.asarray(self.current_prediction["final_alpha"])
            if not alpha.flags.writeable or not alpha.flags.c_contiguous:
                alpha = np.ascontiguousarray(alpha).copy()
                self.current_prediction["final_alpha"] = alpha
            self.edit_canvas.set_alpha_reference(alpha)

    def on_edit_stroke_point(self, x: float, y: float):
        if self.edit_submode != "manual" or not self.current_prediction:
            return
        self.apply_manual_point_to_alpha(x, y)

    def on_edit_stroke_finished(self, points: list):
        self.edit_stroke_active = False
        self._manual_last_point = None
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
        """Paint a connected brush segment directly on alpha, not a full-image mask.

        Earlier code allocated `np.zeros(alpha.shape)` and recomposited the entire
        preview for *every* mouse event. At fast mouse speed this blocked Qt's event
        loop and yielded isolated circular dabs. Connecting the previous point to
        the latest event fills any samples that were skipped while Qt was busy.
        """
        if not self.current_prediction:
            return
        alpha = np.asarray(self.current_prediction["final_alpha"])
        point = (int(round(x)), int(round(y)))
        radius = max(1, int(self.brush_slider.value() // 2))
        color = 0 if self.edit_operation == "erase" else 255
        previous = self._manual_last_point

        if previous is None:
            cv2.circle(alpha, point, radius, color, thickness=-1, lineType=cv2.LINE_8)
            x0 = x1 = point[0]
            y0 = y1 = point[1]
        else:
            # Connected segment, round endpoints from previous/current brush dabs.
            cv2.line(alpha, previous, point, color, thickness=2 * radius,
                     lineType=cv2.LINE_8)
            cv2.circle(alpha, point, radius, color, thickness=-1, lineType=cv2.LINE_8)
            x0, x1 = sorted((previous[0], point[0]))
            y0, y1 = sorted((previous[1], point[1]))

        self._manual_last_point = point
        pad = radius + 2
        self.edit_canvas.refresh_alpha_region(x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    def set_tools_enabled(self, enabled: bool):
        """Tools that need an open, idle image (none while a job runs)."""
        enabled = bool(enabled) and self.in_edit_mode
        for widget in (self.btn_erase, self.btn_clear_mask, self.btn_restore, self.btn_guided,
                       self.btn_manual, self.btn_paint_mode, self.btn_box_mode, self.btn_fit,
                       self.background_controls, self.edit_canvas.btn_close):
            widget.setEnabled(enabled)
        self.btn_matting.setEnabled(enabled and matting_available(self.engine_getter()))
        self.brush_slider.setEnabled(enabled and self.edit_draw_mode == "paint")
        self._sync_file_buttons(enabled)
        if enabled:
            self.update_undo_redo_buttons()
        else:
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)

    def set_edit_controls_enabled(self, enabled: bool):
        """Around a Guided/Matting job: tools off and the Stop button shown."""
        self.set_tools_enabled(enabled)
        self.btn_stop_guided.setVisible(not enabled)
        self.btn_stop_guided.setEnabled(not enabled)

    def is_guided_thread_running(self) -> bool:
        """Block new strokes until the previous worker is fully cleaned up."""
        if thread_is_running(self.edit_thread):
            return True
        if thread_is_valid(self.edit_thread):
            return True  # QThread finished; queued cleanup slot is still pending.
        self.edit_thread = None
        self.edit_worker = None
        return False

    def on_guided_thread_finished_cleanup(self):
        # This queued slot can arrive after the user already started the next
        # stroke: only clean up the thread that actually finished, never the
        # new one (dropping a running QThread aborts the app). finished is
        # emitted just BEFORE the thread stops, so wait() for the real stop.
        finished = self.sender()
        if finished is None or finished is not self.edit_thread:
            return  # stale signal of an earlier, already replaced thread
        if thread_is_valid(finished):
            finished.wait()
        self.edit_thread = None
        self.edit_worker = None
        # Leaving Edit before the worker's QThread fully quits must not leave
        # its large Guided sessions allocated for the rest of the app lifetime.
        if not self.in_edit_mode:
            engine = self.engine_getter()
            if engine is not None:
                engine.release_guided_resources()

    def start_matting(self):
        """Run ViTMatte once after several CPU strokes, without blocking Qt."""
        engine = self.engine_getter()
        if not self.in_edit_mode or self.current_prediction is None or not matting_available(engine):
            return
        if self.edit_stroke_active or self.edit_canvas.is_drawing or \
                self.guided_running or self.is_guided_thread_running():
            self.status_label.setText('Hãy hoàn tất nét vẽ hoặc chờ Guided trước khi Matting.')
            return
        alpha = np.asarray(self.current_prediction['final_alpha'], dtype=np.uint8)
        if not np.any(alpha) or np.all(alpha == 255):
            self.status_label.setText('Mask hiện không có ranh giới để Matting.')
            return

        self.guided_ignore_result = False
        self.guided_running = True
        self.edit_job_kind = 'matting'
        self.guided_timeout_timer.stop()  # A CPU pass may exceed SAM timeout.
        self.set_edit_controls_enabled(False)
        self.edit_canvas.set_processing(True)
        self.edit_canvas.set_processing_progress(5, 'Đang chuẩn bị Matting CPU')
        self.status_label.setText('Đang chạy Matting một lần trên toàn bộ mask...')
        try:
            self.edit_thread = QThread()
            self.edit_worker = MattingWorker(
                engine, self.current_prediction['input_path'], alpha)
            self.edit_worker.moveToThread(self.edit_thread)
            self.edit_thread.started.connect(self.edit_worker.run)
            self.edit_worker.finished.connect(self.on_guided_edit_finished)
            self.edit_worker.progress.connect(self.on_guided_edit_progress)
            self.edit_worker.canceled.connect(self.on_guided_edit_canceled)
            self.edit_worker.error.connect(self.on_guided_edit_error)
            for signal in (self.edit_worker.finished, self.edit_worker.canceled,
                           self.edit_worker.error):
                signal.connect(self.edit_thread.quit)
            self.edit_thread.finished.connect(self.edit_worker.deleteLater)
            self.edit_thread.finished.connect(self.on_guided_thread_finished_cleanup)
            self.edit_thread.finished.connect(self.edit_thread.deleteLater)
            self.edit_thread.start()
        except Exception:
            import traceback
            error = traceback.format_exc()
            self.guided_running = False
            self.guided_ignore_result = False
            self.edit_canvas.set_processing(False)
            self.set_edit_controls_enabled(True)
            self.edit_thread = None
            self.edit_worker = None
            QMessageBox.critical(self, 'Lỗi khởi động Matting', error)

    def start_guided_edit(self, points: list, box: tuple | None = None):
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
        self.edit_job_kind = 'guided'
        self.set_edit_controls_enabled(False)
        self.edit_canvas.set_processing(True)
        if GUIDED_EDIT_TIMEOUT_MS > 0:
            self.status_label.setText(f"Đang chạy Guided Edit... tối đa {GUIDED_EDIT_TIMEOUT_MS / 1000:.0f}s")
            self.guided_timeout_timer.start(GUIDED_EDIT_TIMEOUT_MS)
        else:
            self.status_label.setText("Đang chạy Guided Edit... Lần đầu trên ảnh này có thể chậm (mã hóa ảnh), các nét sau nhanh hơn.")
            self.guided_timeout_timer.stop()
        QApplication.processEvents()

        # A mismatched worker signal must never lock the canvas in processing state.
        try:
            self.edit_thread = QThread()
            self.edit_worker = GuidedEditWorker(
                engine=engine,
                image_path=self.current_prediction["input_path"],
                current_alpha=np.asarray(self.current_prediction["final_alpha"]).copy(),
                stroke_points=[(float(x), float(y)) for x, y in points],
                prompt_box=box,
                brush_size=int(self.brush_slider.value()),
                operation=self.edit_operation,
            )
            self.edit_worker.moveToThread(self.edit_thread)

            self.edit_thread.started.connect(self.edit_worker.run)
            self.edit_worker.finished.connect(self.on_guided_edit_finished)
            self.edit_worker.progress.connect(self.on_guided_edit_progress)
            self.edit_worker.canceled.connect(self.on_guided_edit_canceled)
            self.edit_worker.error.connect(self.on_guided_edit_error)
            self.edit_worker.finished.connect(self.edit_thread.quit)
            self.edit_worker.canceled.connect(self.edit_thread.quit)
            self.edit_worker.error.connect(self.edit_thread.quit)
            self.edit_thread.finished.connect(self.edit_worker.deleteLater)
            self.edit_thread.finished.connect(self.on_guided_thread_finished_cleanup)
            self.edit_thread.finished.connect(self.edit_thread.deleteLater)
            self.edit_thread.start()

        except Exception:
            # Failure happened before a worker could start. Reset the Edit UI,
            # preserving mask/pixels and removing this empty Undo snapshot.
            import traceback
            err = traceback.format_exc()
            self.guided_timeout_timer.stop()
            self.guided_running = False
            self.guided_ignore_result = False
            self.edit_canvas.set_processing(False)
            self.edit_canvas.clear_stroke()
            self.set_edit_controls_enabled(True)
            self.edit_history.discard_last()
            self.update_undo_redo_buttons()
            self.edit_worker = None
            self.edit_thread = None
            self.status_label.setText("Không thể khởi động Guided Edit. Có thể thử lại.")
            QMessageBox.critical(self, "Lỗi khởi động Guided Edit", err)

    def on_guided_edit_progress(self, percentage: int, stage: str):
        """Queued worker updates; only the main GUI thread paints progress."""
        if self.guided_ignore_result or not self.guided_running or not self.in_edit_mode:
            return
        self.edit_canvas.set_processing_progress(percentage, stage)
        label = 'Matting CPU' if self.edit_job_kind == 'matting' else 'Guided Edit'
        self.status_label.setText(f"{label} ~{percentage}% — {stage}")

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
        label = 'Matting' if self.edit_job_kind == 'matting' else 'Guided Edit'
        self.status_label.setText(f'Đã yêu cầu Dừng {label}; sẽ bỏ qua kết quả nếu tiến trình trả về trễ.')

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
        label = 'Matting' if self.edit_job_kind == 'matting' else 'Guided Edit'
        self.status_label.setText(f'{label} đã dừng.' if not was_ignored
                                  else f'{label} đã bị bỏ qua theo yêu cầu dừng.')

    def on_guided_edit_finished(self, prediction: dict):
        self.guided_timeout_timer.stop()
        self.guided_running = False

        if self.guided_ignore_result or not self.in_edit_mode:
            self.guided_ignore_result = False
            self.edit_canvas.set_processing(False)
            self.edit_canvas.clear_stroke()
            self.set_edit_controls_enabled(True)
            self.status_label.setText("Guided Edit đã trả kết quả sau khi bị dừng, app đã bỏ qua kết quả đó.")
            return

        if self.edit_job_kind == 'matting':
            original = self.current_prediction
            old_alpha = np.asarray(original['final_alpha'])
            new_alpha = np.asarray(prediction['final_alpha'])
            if not np.array_equal(old_alpha, new_alpha):
                # Commit Undo *after* a successful, non-empty matting result.
                # An unsuccessful/no-op Matting must not clear Redo history.
                self.push_undo_state()
                prediction = self.preserve_inference_debug(prediction, original)
                self.current_prediction = prediction
                self.edit_dirty = True
                self.refresh_edit_canvas()
            self.edit_canvas.set_processing(False)
            self.edit_canvas.clear_stroke()
            self.set_edit_controls_enabled(True)
            self.update_undo_redo_buttons()
            info = prediction.get('matting_info') or {}
            changed = int(info.get('changed_pixels', 0))
            self.status_label.setText(
                f'Matting {info.get("model", "")} hoàn tất: '
                f'{float(info.get("elapsed_sec", 0)):.2f}s, '
                f'{changed} pixel thay đổi.' if changed else
                'Matting hoàn tất; mask không thay đổi.')
            return

        # A failed HQ-SAM prompt must never stamp the brush's circle as if
        # it were an inferred object. Keep the image unchanged and make the
        # reason visible; Manual still provides deliberate pixel-level edits.
        info = prediction.get("guided_edit_info") or {}
        if info.get("selection_failed"):
            if self.current_prediction is not None:
                for debug_key in ("sam_raw_mask_png", "sam_selected_mask_png"):
                    if debug_key in prediction:
                        self.current_prediction[debug_key] = prediction[debug_key]
            self.edit_history.discard_last()
            self.edit_canvas.set_processing(False)
            self.edit_canvas.clear_stroke()
            self.set_edit_controls_enabled(True)
            self.update_undo_redo_buttons()
            self.status_label.setText(
                "HQ-SAM chưa nhận diện được vật thể tại điểm chấm; mask không thay đổi. "
                "Hãy thử nét ngắn khác, hoặc dùng Manual để sửa đúng nét cọ.")
            return

        # SAM returns a newly built prediction; restore the original inference
        # debug fields while keeping SAM's edited cutout/final alpha unchanged.
        if self.current_prediction is not None:
            prediction = self.preserve_inference_debug(prediction, self.current_prediction)
        self.current_prediction = prediction
        self.edit_dirty = True
        self.edit_canvas.set_processing(False)
        self.edit_canvas.clear_stroke()
        self.refresh_edit_canvas()
        self.set_edit_controls_enabled(True)
        self.update_undo_redo_buttons()
        self.status_label.setText(f"Guided {self.edit_operation} xong. Time: {prediction.get('elapsed_sec', 0):.2f}s")

    def on_guided_edit_error(self, err: str):
        self.guided_timeout_timer.stop()
        self.guided_running = False
        self.guided_ignore_result = False
        self.edit_canvas.set_processing(False)
        self.edit_canvas.clear_stroke()
        self.set_edit_controls_enabled(True)
        alpha = (self.edit_history.discard_last()
                 if self.edit_job_kind != 'matting' and self.current_prediction else None)
        if alpha is not None:  # the stroke failed: back to the alpha before it
            self.current_prediction = self.clone_prediction_with_alpha(self.current_prediction, alpha)
            self.refresh_edit_canvas()
        self.update_undo_redo_buttons()
        label = 'Matting CPU' if self.edit_job_kind == 'matting' else 'Guided Edit'
        QMessageBox.critical(self, f'Lỗi {label}', err)
        self.status_label.setText('Matting lỗi; alpha hiện tại được giữ nguyên.' if
                                  self.edit_job_kind == 'matting' else
                                  'Guided Edit lỗi. Đã rollback về mask trước đó nếu có.')

    def open_folder(self, folder: str):
        p = Path(folder).resolve()
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(p)

