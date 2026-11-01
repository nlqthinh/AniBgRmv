"""M84: packaged MP4 I/O self-test; native SAM2 needs external PyTorch/assets."""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

from .engine.video_pipeline import extract_clip, paint_alpha, release_clip_temp, export_video


def main() -> int:
    target = (Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False)
              else Path.cwd()) / 'video_codec_selftest.json'
    record = {'passed': False, 'mode': 'video_selftest', 'opencv': cv2.__version__}
    try:
        # GPU edition must contain the standalone sidecar script for the
        # user-configured external PyTorch Python. CPU Lite does not ship it.
        if getattr(sys, 'frozen', False):
            bundle = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
            flavor_path = bundle / 'build_flavor.json'
            if flavor_path.is_file():
                flavor = json.loads(flavor_path.read_text(encoding='utf-8')).get('variant')
                if flavor == 'gpu':
                    sidecar = bundle / 'ani_bgrmv/engine/video_hq3_worker.py'
                    if not sidecar.is_file():
                        raise FileNotFoundError('Thiếu HQ3 video sidecar trong GPU EXE: ' + str(sidecar))
                    for name in ('video_diagnostics.py','video_m71_worker.py'):
                        other = bundle / 'ani_bgrmv/engine' / name
                        if not other.is_file():
                            raise FileNotFoundError('Thiếu M87 video sidecar: ' + str(other))
        with tempfile.TemporaryDirectory(prefix='anibgrmv_video_codec_') as d:
            input_path = Path(d) / 'input.mp4'
            out_path = Path(d) / 'output.mp4'
            encoder = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*'mp4v'), 6.0, (96, 64))
            if not encoder.isOpened():
                raise RuntimeError('Không có MP4 encoder (mp4v).')
            for i in range(9):
                frame = np.zeros((64, 96, 3), np.uint8)
                frame[:, :] = (i * 5, 40, 70)
                cv2.rectangle(frame, (15+i, 12), (45+i, 48), (20, 240, 40), -1)
                encoder.write(frame)
            encoder.release()
            clip = extract_clip(str(input_path))
            try:
                alpha = paint_alpha(np.zeros((64, 96), np.uint8), [], 8,
                                    'restore', box=(15, 12, 45, 48))
                # Codec self-test must NOT invoke native SAM2 video tracking:
                # the ONNX-only GPU EXE deliberately has no bundled PyTorch or
                # HQ3 .pt. Actual video Track is checked separately on the host.
                masks = tuple(alpha.copy() for _ in range(clip.count))
                export_video(clip, masks, str(out_path))
                output = cv2.VideoCapture(str(out_path))
                opened = output.isOpened()
                ok, frame = output.read() if opened else (False, None)
                output.release()
                if not opened or not ok or frame.shape != (64, 96, 3):
                    raise RuntimeError('MP4 đã export không thể giải mã.')
                record.update(passed=True, frames=clip.count, codec='mp4v',
                              output_bytes=out_path.stat().st_size)
            finally:
                release_clip_temp(clip.temp_dir)
    except Exception:
        record['error'] = traceback.format_exc()
    try:
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    except OSError:
        pass
    if record['passed']:
        print('[M84 PASS] Packaged MP4 read/write verified; native HQ3 Track requires separate preflight', flush=True)
        return 0
    print('[M83 FAIL] Video codec self-test: ' + record.get('error', 'Unknown error'), flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
