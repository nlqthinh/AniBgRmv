from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from bg_engine import BgRemoveEngine

class SingleCutWorker(QObject):
    finished = Signal(dict)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_path: str):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if self.cancel_requested:
                self.canceled.emit()
                return

            result = self.engine.process_for_preview(self.image_path)

            if self.cancel_requested:
                self.canceled.emit()
                return

            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())



class GuidedEditWorker(QObject):
    finished = Signal(dict)
    canceled = Signal()
    error = Signal(str)

    def __init__(
        self,
        engine: BgRemoveEngine,
        image_path: str,
        current_alpha: np.ndarray,
        stroke_points: list[tuple[float, float]],
        brush_size: int,
        operation: str,
        sam_checkpoint_path: str,
        sam_model_type: str,
        sam_device: str,
    ):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.current_alpha = current_alpha
        self.stroke_points = stroke_points
        self.brush_size = brush_size
        self.operation = operation
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_model_type = sam_model_type
        self.sam_device = sam_device
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if self.cancel_requested:
                self.canceled.emit()
                return

            result = self.engine.apply_sam_guided_edit(
                image_path=self.image_path,
                current_alpha=self.current_alpha,
                stroke_points=self.stroke_points,
                brush_size=self.brush_size,
                operation=self.operation,
                sam_checkpoint_path=self.sam_checkpoint_path,
                sam_model_type=self.sam_model_type,
                sam_device=self.sam_device,
                # M22: do not run ViTMatte on every guided stroke.
                # It made edit mode extremely slow on CPU.
                # Guided edit now uses SAM for local region selection, then applies alpha directly.
                refine_with_vitmatte=False,
            )

            if self.cancel_requested:
                self.canceled.emit()
                return

            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchCutWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_paths: list[str]):
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            results = []
            total = len(self.image_paths)

            for idx, path in enumerate(self.image_paths, start=1):
                if self.cancel_requested:
                    self.canceled.emit()
                    return

                self.progress.emit(idx - 1, total, f"Đang cắt: {Path(path).name}")
                result = self.engine.process_for_preview(path)

                if self.cancel_requested:
                    self.canceled.emit()
                    return

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

            if self.save_debug:
                # Keep the original batch report when debugging.
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                log_path = out_dir / "batch_log.csv"
                pd.DataFrame(logs).to_csv(log_path, index=False, encoding="utf-8-sig")
                self.finished.emit(str(log_path))
            else:
                # Release-like save: PNG cutouts only; no batch_log.csv.
                self.finished.emit(str(Path(self.output_dir) / "01_cutouts"))
        except Exception:
            self.error.emit(traceback.format_exc())


# =========================
# REUSABLE UI COMPONENTS
# =========================

