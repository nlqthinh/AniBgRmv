import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMessageBox

from ..app_config import (
    GUIDED_EDIT_TIMEOUT_MS,
    SAM_CHECKPOINT_PATH,
    SAM_DEVICE,
    SAM_MODEL_TYPE,
)
from ..workers import GuidedEditWorker


class SingleEditControllerMixin:
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

