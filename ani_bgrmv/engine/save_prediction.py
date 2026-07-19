from pathlib import Path

import numpy as np
from PIL import Image


class SavePredictionMixin:
    def _unique_path(self, folder: Path, filename: str) -> Path:
        """
        Avoid overwriting when batch has duplicate image names.
        """
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
        """
        Optimize cutout PNG without changing visible pixels.

        If alpha == 0, RGB is invisible, so set invisible RGB to 0 to make PNG
        compression easier. Soft alpha edges are untouched.
        """
        rgba = cutout_pil.convert("RGBA")
        arr = np.asarray(rgba).copy()

        alpha = arr[:, :, 3]
        transparent = alpha == 0

        arr[transparent, 0] = 0
        arr[transparent, 1] = 0
        arr[transparent, 2] = 0

        return Image.fromarray(arr, mode="RGBA")

    def save_prediction(self, prediction: dict, output_dir: str, save_debug: bool = True) -> dict:
        """
        Only call this when the user clicks Save.
        """
        input_path = Path(prediction["input_path"])
        output_dir = Path(output_dir)
        stem = input_path.stem

        cutout_dir = output_dir / "01_cutouts"
        preview_checker_dir = output_dir / "02_preview_checker"
        preview_white_dir = output_dir / "03_preview_white"

        final_alpha_dir = output_dir / "99_debug" / "01_final_alpha"
        f2_alpha_dir = output_dir / "99_debug" / "02_unet_c7_auto_alpha_768"
        trimap_dir = output_dir / "99_debug" / "03_m6_loose_trimap_768"
        m6_alpha_dir = output_dir / "99_debug" / "04_vitmatte_m6_alpha_768"
        m6_raw_alpha_dir = output_dir / "99_debug" / "05_vitmatte_m6_raw_alpha_768"

        cutout_path = self._unique_path(cutout_dir, f"{stem}_cutout.png")
        preview_checker_path = self._unique_path(preview_checker_dir, f"{stem}_preview_checker.jpg")
        preview_white_path = self._unique_path(preview_white_dir, f"{stem}_preview_white.jpg")

        cutout_optimized = self.optimize_cutout_png(prediction["cutout_pil"])
        cutout_optimized.save(
            cutout_path,
            format="PNG",
            optimize=True,
            compress_level=9,
        )
        prediction["preview_checker_pil"].save(preview_checker_path, quality=95, optimize=True)
        prediction["preview_white_pil"].save(preview_white_path, quality=95, optimize=True)

        saved = {
            "input_path": str(input_path),
            "cutout_path": str(cutout_path),
            "preview_checker_path": str(preview_checker_path),
            "preview_white_path": str(preview_white_path),
            "elapsed_sec": prediction.get("elapsed_sec", None),
            "trimap_stats": prediction.get("trimap_stats", {}),
            "safety_notes": ";".join(prediction.get("safety_notes", [])),
        }

        if save_debug:
            final_alpha_path = self._unique_path(final_alpha_dir, f"{stem}_final_alpha.png")
            f2_alpha_path = self._unique_path(f2_alpha_dir, f"{stem}_unet_c7_auto_alpha_768.png")
            trimap_path = self._unique_path(trimap_dir, f"{stem}_m6_loose_trimap_768.png")
            m6_alpha_path = self._unique_path(m6_alpha_dir, f"{stem}_vitmatte_m6_alpha_768.png")
            m6_raw_alpha_path = self._unique_path(m6_raw_alpha_dir, f"{stem}_vitmatte_m6_raw_alpha_768.png")

            Image.fromarray(prediction["final_alpha"]).save(final_alpha_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["f2_alpha_768"]).save(f2_alpha_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["trimap_768"]).save(trimap_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["m6_alpha_768"]).save(m6_alpha_path, optimize=True, compress_level=9)
            Image.fromarray(prediction["m6_raw_alpha_768"]).save(m6_raw_alpha_path, optimize=True, compress_level=9)

            saved.update({
                "final_alpha_path": str(final_alpha_path),
                "f2_alpha_path": str(f2_alpha_path),
                "trimap_path": str(trimap_path),
                "m6_alpha_path": str(m6_alpha_path),
                "m6_raw_alpha_path": str(m6_raw_alpha_path),
            })

        return saved
