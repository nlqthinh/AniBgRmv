from pathlib import Path

import numpy as np
from PIL import Image

# (prediction key, debug subfolder, file suffix). Written only when present:
# an edited or lightweight prediction never gets fabricated intermediates.
DEBUG_IMAGES = (
    ("final_alpha", "01_final_alpha", "final_alpha"),
    ("unet_alpha_768", "02_unet_alpha_768", "unet_alpha_768"),
    ("unet_alpha_original", "03_unet_alpha_original", "unet_alpha_original"),
    ("trimap_768", "04_trimap_768", "trimap_768"),
    ("matte_alpha_768", "05_vitmatte_unet_alpha_768", "vitmatte_unet_alpha_768"),
    ("valid_mask_768", "06_valid_mask_768", "valid_mask_768"),
    ("image_letterbox_pil", "07_image_letterbox_768", "image_letterbox_768"),
)
# Guided Edit diagnostics, kept as encoded PNG bytes (no extra full-size arrays).
DEBUG_PNG_BYTES = (
    ("sam_raw_mask_png", "08_sam_raw_mask", "sam_raw_mask"),
    ("sam_selected_mask_png", "09_sam_selected_instance", "sam_selected_instance"),
)
# Small fields written to batch_log.csv; kept in RAM when a batch is staged.
LOG_FIELDS = ("trimap_stats", "trimap_stats_valid", "letterbox_meta", "safety_notes")
THUMBNAIL_SIZE = 176  # overview grid of the batch page (as the annotator grid)


class SavePredictionMixin:
    def _unique_path(self, folder: Path, filename: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        candidate = folder / filename
        if not candidate.exists():
            return candidate
        idx = 2
        while True:
            p = folder / f"{candidate.stem}_{idx}{candidate.suffix}"
            if not p.exists():
                return p
            idx += 1

    def optimize_cutout_png(self, cutout_pil: Image.Image) -> Image.Image:
        """Zero the RGB of fully transparent pixels: smaller PNG, same image."""
        arr = np.asarray(cutout_pil.convert("RGBA")).copy()
        arr[arr[:, :, 3] == 0, :3] = 0
        return Image.fromarray(arr, mode="RGBA")

    def save_prediction(self, prediction: dict, output_dir: str, save_debug: bool = True) -> dict:
        """Save the cutout, plus previews and debug images when save_debug is on."""
        input_path = Path(prediction["input_path"])
        output_dir = Path(output_dir)
        stem = input_path.stem

        cutout_path = self._unique_path(output_dir / "01_cutouts", f"{stem}_cutout.png")
        self.optimize_cutout_png(prediction["cutout_pil"]).save(
            cutout_path, format="PNG", optimize=True, compress_level=9)
        saved = {
            "input_path": str(input_path),
            "cutout_path": str(cutout_path),
            "elapsed_sec": prediction.get("elapsed_sec"),
            "trimap_stats": prediction.get("trimap_stats", {}),
            "trimap_stats_valid": prediction.get("trimap_stats_valid", {}),
            "letterbox_meta": prediction.get("letterbox_meta", {}),
            "safety_notes": ";".join(prediction.get("safety_notes", [])),
        }
        if not save_debug:
            return saved

        # Previews exist only on disk: built here, never kept in a prediction.
        checker = self.make_preview_checker(prediction["cutout_pil"])
        white = self.make_preview_white(prediction["cutout_pil"])
        checker_path = self._unique_path(output_dir / "02_preview_checker", f"{stem}_preview_checker.jpg")
        white_path = self._unique_path(output_dir / "03_preview_white", f"{stem}_preview_white.jpg")
        checker.save(checker_path, quality=95, optimize=True)
        white.save(white_path, quality=95, optimize=True)
        saved["preview_checker_path"] = str(checker_path)
        saved["preview_white_path"] = str(white_path)

        # Debug files overwrite by name to avoid stale masks when testing.
        debug_root = output_dir / "99_debug"
        for key, folder, suffix in DEBUG_IMAGES:
            data = prediction.get(key)
            if data is None:
                continue
            (debug_root / folder).mkdir(parents=True, exist_ok=True)
            path = debug_root / folder / f"{stem}_{suffix}.png"
            image = data if isinstance(data, Image.Image) else Image.fromarray(np.asarray(data).astype(np.uint8))
            image.save(path, optimize=True, compress_level=9)
            saved[f"{suffix}_path"] = str(path)
        for key, folder, suffix in DEBUG_PNG_BYTES:
            encoded = prediction.get(key)
            if encoded is None:
                continue
            (debug_root / folder).mkdir(parents=True, exist_ok=True)
            path = debug_root / folder / f"{stem}_{suffix}.png"
            path.write_bytes(encoded)
            saved[f"{suffix}_path"] = str(path)
        return saved

    # ------------------------------------------------------------ batch staging
    def stage_prediction(self, prediction: dict, cache_dir: Path, index: int,
                         keep_debug: bool) -> dict:
        """Move one batch result to ``cache_dir``; return the small dict kept in RAM.

        The cutout becomes a PNG, plus a small thumbnail for the overview grid;
        with ``keep_debug`` the DEBUG_IMAGES arrays (and the Guided debug PNGs
        of an edited image) go to one .npz, so Debug ON no longer holds every
        image's data until Save. Staging the same index again (an edited
        image) replaces its files.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cutout = self.optimize_cutout_png(prediction["cutout_pil"])
        cutout_path = cache_dir / f"{index:04d}_cutout.png"
        cutout.save(cutout_path, "PNG", compress_level=1)
        thumb_path = cache_dir / f"{index:04d}_thumb.png"
        cutout.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        cutout.save(thumb_path, "PNG", compress_level=1)
        staged = {"input_path": prediction["input_path"], "cached_cutout_path": str(cutout_path),
                  "cached_thumb_path": str(thumb_path), "elapsed_sec": prediction.get("elapsed_sec")}
        if keep_debug:
            debug_path = cache_dir / f"{index:04d}_debug.npz"
            arrays = {key: np.asarray(prediction[key]) for key, _, _ in DEBUG_IMAGES
                      if prediction.get(key) is not None}
            arrays.update({key: np.frombuffer(prediction[key], np.uint8) for key, _, _ in DEBUG_PNG_BYTES
                           if prediction.get(key) is not None})
            np.savez_compressed(debug_path, **arrays)
            staged["cached_debug_path"] = str(debug_path)
            staged.update({key: prediction[key] for key in LOG_FIELDS if key in prediction})
        return staged

    @staticmethod
    def load_staged_prediction(staged: dict) -> dict:
        """Inverse of stage_prediction: a dict save_prediction and the editor accept."""
        with Image.open(staged["cached_cutout_path"]) as image:
            cutout = image.convert("RGBA")
        prediction = {**staged, "cutout_pil": cutout,
                      "final_alpha": np.asarray(cutout.getchannel("A")).copy()}
        if "cached_debug_path" in staged:
            png_keys = {key for key, _, _ in DEBUG_PNG_BYTES}
            with np.load(staged["cached_debug_path"]) as arrays:
                prediction.update({key: arrays[key].tobytes() if key in png_keys else arrays[key]
                                   for key in arrays.files})
        return prediction
