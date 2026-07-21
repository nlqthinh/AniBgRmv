from pathlib import Path

import numpy as np
from PIL import Image


class SavePredictionMixin:
    def _unique_path(self, folder: Path, filename: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)

        candidate = folder / filename
        if not candidate.exists():
            return candidate

        stem = candidate.stem
        suffix = candidate.suffix

        idx = 2
        while True:
            p = folder / f"{stem}_{idx}{suffix}"
            if not p.exists():
                return p
            idx += 1

    def optimize_cutout_png(self, cutout_pil: Image.Image) -> Image.Image:
        rgba = cutout_pil.convert("RGBA")
        arr = np.asarray(rgba).copy()

        alpha = arr[:, :, 3]
        transparent = alpha == 0

        arr[transparent, 0] = 0
        arr[transparent, 1] = 0
        arr[transparent, 2] = 0

        return Image.fromarray(arr, mode="RGBA")

    def _write_debug_png(self, folder: Path, filename: str, data) -> str:
        """Debug files overwrite by name to avoid stale masks when testing."""
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename

        if isinstance(data, Image.Image):
            data.save(path, optimize=True, compress_level=9)
        else:
            Image.fromarray(np.asarray(data).astype(np.uint8)).save(path, optimize=True, compress_level=9)

        return str(path)

    def save_prediction(self, prediction: dict, output_dir: str, save_debug: bool = True) -> dict:
        """
        Save final cutout and optional debug images when the user clicks Save.
        """
        input_path = Path(prediction["input_path"])
        output_dir = Path(output_dir)
        stem = input_path.stem

        cutout_dir = output_dir / "01_cutouts"
        preview_checker_dir = output_dir / "02_preview_checker"
        preview_white_dir = output_dir / "03_preview_white"

        cutout_path = self._unique_path(cutout_dir, f"{stem}_cutout.png")
        preview_checker_path = self._unique_path(preview_checker_dir, f"{stem}_preview_checker.jpg")
        preview_white_path = self._unique_path(preview_white_dir, f"{stem}_preview_white.jpg")

        cutout_optimized = self.optimize_cutout_png(prediction["cutout_pil"])
        cutout_optimized.save(cutout_path, format="PNG", optimize=True, compress_level=9)
        prediction["preview_checker_pil"].save(preview_checker_path, quality=95, optimize=True)
        prediction["preview_white_pil"].save(preview_white_path, quality=95, optimize=True)

        saved = {
            "input_path": str(input_path),
            "cutout_path": str(cutout_path),
            "preview_checker_path": str(preview_checker_path),
            "preview_white_path": str(preview_white_path),
            "elapsed_sec": prediction.get("elapsed_sec", None),
            "trimap_stats": prediction.get("trimap_stats", {}),
            "alpha_fix_stats": prediction.get("alpha_fix_stats", {}),
            "vitmatte_edge_stats": prediction.get("vitmatte_edge_stats", {}),
            "trimap_stats_valid": prediction.get("trimap_stats_valid", {}),
            "f3_letterbox_meta": prediction.get("f3_letterbox_meta", {}),
            "safety_notes": ";".join(prediction.get("safety_notes", [])),
        }

        if not save_debug:
            return saved

        debug_root = output_dir / "99_debug"

        saved["final_alpha_path"] = self._write_debug_png(
            debug_root / "01_final_alpha_used_for_cutout",
            f"{stem}_final_alpha_used_for_cutout.png",
            prediction["final_alpha"],
        )

        if prediction.get("f3_raw_output_letterbox_debug") is not None:
            saved["f3_raw_output_letterbox_path"] = self._write_debug_png(
                debug_root / "02_f3_raw_output_letterbox_768_debug",
                f"{stem}_f3_raw_output_letterbox_768_debug.png",
                prediction["f3_raw_output_letterbox_debug"],
            )

        if prediction.get("f3_auto_alpha_before_fix_768") is not None:
            saved["f3_auto_alpha_before_fix_path"] = self._write_debug_png(
                debug_root / "03_f3_auto_alpha_before_fix_letterbox_768",
                f"{stem}_f3_auto_alpha_before_fix_letterbox_768.png",
                prediction["f3_auto_alpha_before_fix_768"],
            )

        saved["f3_auto_alpha_used_letterbox_path"] = self._write_debug_png(
            debug_root / "04_f3_auto_alpha_used_for_trimap_letterbox_768",
            f"{stem}_f3_auto_alpha_used_for_trimap_letterbox_768.png",
            prediction["f2_alpha_768"],
        )

        if prediction.get("unet_alpha_original") is not None:
            saved["f3_auto_alpha_original_preview_path"] = self._write_debug_png(
                debug_root / "05_f3_auto_alpha_original_preview",
                f"{stem}_f3_auto_alpha_original_preview.png",
                prediction["unet_alpha_original"],
            )

        saved["trimap_letterbox_path"] = self._write_debug_png(
            debug_root / "06_m7_loose_trimap_letterbox_768",
            f"{stem}_m7_loose_trimap_letterbox_768.png",
            prediction["trimap_768"],
        )

        if prediction.get("m6_raw_alpha_768") is not None:
            saved["vitmatte_m7_raw_alpha_letterbox_path"] = self._write_debug_png(
                debug_root / "07_vitmatte_m7_raw_alpha_letterbox_768",
                f"{stem}_vitmatte_m7_raw_alpha_letterbox_768.png",
                prediction["m6_raw_alpha_768"],
            )

        saved["vitmatte_m7_alpha_used_letterbox_path"] = self._write_debug_png(
            debug_root / "08_vitmatte_m7_alpha_used_letterbox_768",
            f"{stem}_vitmatte_m7_alpha_used_letterbox_768.png",
            prediction["m6_alpha_768"],
        )

        if prediction.get("valid_mask_768") is not None:
            saved["valid_mask_letterbox_path"] = self._write_debug_png(
                debug_root / "09_valid_mask_letterbox_768",
                f"{stem}_valid_mask_letterbox_768.png",
                prediction["valid_mask_768"],
            )

        if prediction.get("image_letterbox_pil") is not None:
            saved["image_letterbox_path"] = self._write_debug_png(
                debug_root / "10_image_letterbox_768",
                f"{stem}_image_letterbox_768.png",
                prediction["image_letterbox_pil"],
            )

        if prediction.get("unet_alpha_final_768") is not None:
            saved["final_alpha_letterbox_path"] = self._write_debug_png(
                debug_root / "11_final_alpha_letterbox_768_before_crop",
                f"{stem}_final_alpha_letterbox_768_before_crop.png",
                prediction["unet_alpha_final_768"],
            )

        return saved
