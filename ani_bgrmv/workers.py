from pathlib import Path
import csv
import shutil
import traceback

import numpy as np
from PySide6.QtCore import QObject, Signal

from bg_engine import BgRemoveEngine

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
        sam_checkpoint_path: str,
        sam_model_type: str,
        sam_device: str,
        prompt_box: tuple[int, int, int, int] | None = None,
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
        self.prompt_box = prompt_box
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
                prompt_box=self.prompt_box,
                brush_size=self.brush_size,
                operation=self.operation,
                sam_checkpoint_path=self.sam_checkpoint_path,
                sam_model_type=self.sam_model_type,
                sam_device=self.sam_device,
                # M22: do not run ViTMatte on every guided stroke.
                # It made edit mode extremely slow on CPU.
                # Guided edit now uses SAM for local region selection, then applies alpha directly.
                refine_with_vitmatte=False,
                # Both Guided Paint and Guided Box share the same stage progress.
                # This signal is queued to the GUI thread by Qt.
                progress_callback=self.progress.emit,
            )

            if self.cancel_requested:
                self.canceled.emit()
                return

            self.progress.emit(100, "Đã hoàn tất")
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchCutWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    canceled = Signal()
    error = Signal(str)

    def __init__(self, engine: BgRemoveEngine, image_paths: list[str], lightweight: bool = False,
                 batch_cache_dir: str | None = None):
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths
        self.lightweight = bool(lightweight)
        self.batch_cache_dir = Path(batch_cache_dir) if batch_cache_dir else None
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
                result = self.engine.process_for_preview(path, include_debug=not self.lightweight)

                if self.cancel_requested:
                    self.canceled.emit()
                    return

                if self.batch_cache_dir is not None:
                    # Release mode: do not keep N full-resolution RGBA PIL
                    # images and masks in the Qt signal/result list.
                    self.batch_cache_dir.mkdir(parents=True, exist_ok=True)
                    cached_path = self.batch_cache_dir / f'{idx:04d}_cutout.png'
                    compact_cutout = self.engine.optimize_cutout_png(result['cutout_pil'])
                    compact_cutout.save(cached_path, 'PNG', compress_level=1)
                    result = {
                        'input_path': result['input_path'],
                        'cached_cutout_path': str(cached_path),
                        'elapsed_sec': result.get('elapsed_sec'),
                    }
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
                if 'cached_cutout_path' in pred:
                    if self.save_debug:
                        raise RuntimeError('Debug ON cần dữ liệu gốc; hãy cắt lại với Debug ON.')
                    src = Path(pred['cached_cutout_path'])
                    dst = self.engine._unique_path(Path(self.output_dir) / '01_cutouts',
                                                    f"{Path(pred['input_path']).stem}_cutout.png")
                    shutil.copyfile(src, dst)  # lossless RGBA, no recompression
                    saved = {'input_path': pred['input_path'], 'cutout_path': str(dst),
                             'elapsed_sec': pred.get('elapsed_sec')}
                else:
                    saved = self.engine.save_prediction(
                        pred, self.output_dir, save_debug=self.save_debug,
                    )
                logs.append(saved)
                self.progress.emit(idx, total, f"Đã lưu: {Path(pred['input_path']).name}")

            if self.save_debug:
                # Keep the original batch report when debugging.
                out_dir = Path(self.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                log_path = out_dir / "batch_log.csv"
                # The batch report does not require pandas. Removing its only
                # import avoids bundling pandas and optional scientific stacks.
                # DictWriter preserves the old column insertion order and
                # handles fields appearing only in later saved predictions.
                columns = list(dict.fromkeys(key for item in logs for key in item))
                with log_path.open('w', newline='', encoding='utf-8-sig') as stream:
                    if columns:
                        writer = csv.DictWriter(stream, fieldnames=columns,
                                                extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(logs)
                self.finished.emit(str(log_path))
            else:
                # Release-like save: PNG cutouts only; no batch_log.csv.
                self.finished.emit(str(Path(self.output_dir) / "01_cutouts"))
        except Exception:
            self.error.emit(traceback.format_exc())



