from __future__ import annotations

from pathlib import Path
import csv
import shutil
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:  # annotations only: importing the engine loads ONNX Runtime
    from .engine.bg_remove_engine import BgRemoveEngine

class SingleCutWorker(QObject):
    finished = Signal(dict)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_path: str, lightweight: bool = False):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.lightweight = bool(lightweight)
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if self.cancel_requested:
                self.canceled.emit()
                return

            result = self.engine.process_for_preview(self.image_path, include_debug=not self.lightweight)

            if self.cancel_requested:
                self.canceled.emit()
                return

            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())



class GuidedEditWorker(QObject):
    progress = Signal(int, str)
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
        prompt_box: tuple[int, int, int, int] | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.current_alpha = current_alpha
        self.stroke_points = stroke_points
        self.brush_size = brush_size
        self.operation = operation
        self.prompt_box = prompt_box
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if self.cancel_requested:
                self.canceled.emit()
                return

            # Paint and Box share the same stage progress, queued to the GUI thread.
            result = self.engine.apply_guided_edit(
                image_path=self.image_path,
                current_alpha=self.current_alpha,
                stroke_points=self.stroke_points,
                prompt_box=self.prompt_box,
                brush_size=self.brush_size,
                operation=self.operation,
                progress_callback=self.progress.emit,
            )

            if self.cancel_requested:
                self.canceled.emit()
                return

            self.progress.emit(100, "Đã hoàn tất")
            self.finished.emit(result)
        except ValueError as exc:
            # Invalid user prompts should be explained, not show a traceback.
            self.error.emit(str(exc))
        except Exception:
            self.error.emit(traceback.format_exc())


class MattingWorker(QObject):
    """One on-demand CPU matting pass; same signals as GuidedEditWorker.

    Cancellation is cooperative: ONNX Runtime may finish its current run,
    but a canceled result is never applied to the GUI alpha.
    """
    progress = Signal(int, str)
    finished = Signal(dict)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_path: str, current_alpha: np.ndarray):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.current_alpha = np.asarray(current_alpha, dtype=np.uint8).copy()
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            if self.cancel_requested:
                self.canceled.emit()
                return
            prediction = self.engine.matte_current_alpha(
                self.image_path, self.current_alpha,
                progress_callback=self.progress.emit,
            )
            if self.cancel_requested:
                self.canceled.emit()
                return
            self.progress.emit(100, 'Matting hoàn tất')
            self.finished.emit(prediction)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchCutWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_paths: list[str], cache_dir: str,
                 keep_debug: bool = False):
        """Every result is staged in ``cache_dir`` right after its cut, so RAM
        holds only small dicts (see SavePredictionMixin.stage_prediction)."""
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths
        self.cache_dir = Path(cache_dir)
        self.keep_debug = bool(keep_debug)
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
                result = self.engine.process_for_preview(path, include_debug=self.keep_debug,
                                                         profile='batch')

                if self.cancel_requested:
                    self.canceled.emit()
                    return

                results.append(self.engine.stage_prediction(result, self.cache_dir, idx,
                                                            self.keep_debug))
                self.progress.emit(idx, total, f"Đã cắt: {Path(path).name}")

            self.finished.emit(results)
        except Exception:
            self.error.emit(traceback.format_exc())


def save_staged(engine: BgRemoveEngine, staged: dict, output_dir: str, save_debug: bool) -> dict:
    """Save one staged batch image (stage_prediction) into ``output_dir``.

    Debug: cutout + previews + 99_debug from the staged .npz. Otherwise the
    staged PNG is copied as is: lossless RGBA, no recompression.
    """
    if save_debug:
        if 'cached_debug_path' not in staged:
            raise RuntimeError('Debug ON cần dữ liệu gốc; hãy cắt lại với Debug ON.')
        return engine.save_prediction(engine.load_staged_prediction(staged), output_dir, save_debug=True)
    dst = engine._unique_path(Path(output_dir) / '01_cutouts', f"{Path(staged['input_path']).stem}_cutout.png")
    shutil.copyfile(staged['cached_cutout_path'], dst)
    return {'input_path': staged['input_path'], 'cutout_path': str(dst),
            'elapsed_sec': staged.get('elapsed_sec')}


class BatchSaveWorker(QObject):
    """Saves the staged images in order; Stop takes effect after the image
    being saved. ``canceled(n)``: the first n images were saved."""
    progress = Signal(int, int, str)
    finished = Signal(str)
    canceled = Signal(int)
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
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            logs = []
            total = len(self.predictions)

            for idx, pred in enumerate(self.predictions, start=1):
                if self.cancel_requested:
                    if self.save_debug:
                        self._write_log(logs)
                    self.canceled.emit(len(logs))
                    return
                self.progress.emit(idx - 1, total, f"Đang lưu: {Path(pred['input_path']).name}")
                logs.append(save_staged(self.engine, pred, self.output_dir, self.save_debug))
                self.progress.emit(idx, total, f"Đã lưu: {Path(pred['input_path']).name}")

            if self.save_debug:
                # Keep the original batch report when debugging.
                self.finished.emit(self._write_log(logs))
            else:
                # Release-like save: PNG cutouts only; no batch_log.csv.
                self.finished.emit(str(Path(self.output_dir) / "01_cutouts"))
        except Exception:
            self.error.emit(traceback.format_exc())

    def _write_log(self, logs: list[dict]) -> str:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "batch_log.csv"
        # stdlib csv (no pandas); columns in first-seen order, including
        # fields that appear only in later saved predictions.
        columns = list(dict.fromkeys(key for item in logs for key in item))
        with log_path.open('w', newline='', encoding='utf-8-sig') as stream:
            if columns:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(logs)
        return str(log_path)
