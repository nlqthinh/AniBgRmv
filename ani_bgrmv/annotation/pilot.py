"""Pilots: split a fixed random sample of cutouts and draw contact sheets.

Without HQ-SAM (bước 0) only separated regions become objects. With a
``splitter`` (pilot --sam, owner PC) touching groups are split by auto_split.
Output folder:
  sheet_NN.png   20 thumbnails per page: one color per object, gray = ignored
                 tiny fragments, caption '#NNN obj=K frag=F [sam=S]'
  objects/NNN_<name>/object_01.png ...  8-bit alpha (256 levels) per object,
                 largest first, only for images with 2+ objects
  pilot.csv      one row per image (Excel-friendly UTF-8)
The numbers #NNN are the same for every pilot run on the same index.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .alpha_split import split_alpha
from .dataset_index import read_alpha, read_rgb, sample_entries
from .overlay import overlay, thumbnail
from .partition import IGNORED, SamObject, instance_alphas, partition

THUMB, PAD, CAPTION, COLUMNS, ROWS = 256, 6, 22, 5, 4
# Measured in bước 0 on the annotator's Pentium G4400: one decoder run.
WORKER_DECODE_SECONDS = 0.45
CSV_FIELDS = ('stt', 'trang', 'vi_tri', 'anh', 'rong', 'cao', 'so_object', 'so_object_sam',
              'so_manh_bo_qua', 'so_muc_alpha', 'thoi_gian_ms')

Splitter = Callable[[np.ndarray, np.ndarray, object], list[SamObject]]


def _tile(image: np.ndarray, caption: str) -> Image.Image:
    small = thumbnail(image, THUMB - 2 * PAD)
    tile = Image.new('RGB', (THUMB, THUMB + CAPTION), (24, 24, 24))
    tile.paste(Image.fromarray(small), ((THUMB - small.shape[1]) // 2, (THUMB - small.shape[0]) // 2))
    ImageDraw.Draw(tile).text((6, THUMB + 5), caption, fill=(235, 235, 235))
    return tile


def _save_sheets(tiles: list[Image.Image], out_dir: Path) -> None:
    per_sheet = COLUMNS * ROWS
    for start in range(0, len(tiles), per_sheet):
        sheet = Image.new('RGB', (COLUMNS * THUMB, ROWS * (THUMB + CAPTION)), (0, 0, 0))
        for slot, tile in enumerate(tiles[start:start + per_sheet]):
            sheet.paste(tile, ((slot % COLUMNS) * THUMB, (slot // COLUMNS) * (THUMB + CAPTION)))
        sheet.save(out_dir / f'sheet_{start // per_sheet + 1:02d}.png')


def save_objects(alphas: list[np.ndarray], folder: Path) -> None:
    """object_01.png, object_02.png ...: 8-bit alpha, largest object first."""
    folder.mkdir(parents=True, exist_ok=True)
    for number, alpha in enumerate(alphas, 1):
        Image.fromarray(alpha, 'L').save(folder / f'object_{number:02d}.png')


def run_pilot(root: Path, index: dict, out_dir: Path, *, count: int, min_object_ratio: float,
              splitter: Splitter | None = None, seed: int = 0) -> dict:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, rows, timings = [], [], []
    for number, entry in enumerate(sample_entries(index, count, seed), 1):
        rgb, alpha = read_rgb(root / entry['image']), read_alpha(root / entry['cutout'])
        started = time.perf_counter()
        split = split_alpha(alpha, min_object_ratio)
        objects = splitter(rgb, alpha, split) if splitter else []
        labels = partition(rgb, alpha, split, objects, min_object_ratio)
        alphas = instance_alphas(alpha, labels)
        timings.append((time.perf_counter() - started) * 1000)
        ignored = len(np.unique(split.labels[labels == IGNORED]))
        caption = f'#{number:03d} obj={len(alphas)} frag={ignored}' + (f' sam={len(objects)}' if splitter else '')
        tiles.append(_tile(overlay(rgb, alpha, labels), caption))
        if len(alphas) >= 2:
            save_objects(alphas, out_dir / 'objects' / f'{number:03d}_{Path(entry["image"]).stem}')
        slot = number - 1
        rows.append({'stt': number, 'trang': slot // (COLUMNS * ROWS) + 1,
                     'vi_tri': slot % (COLUMNS * ROWS) + 1, 'anh': entry['image'],
                     'rong': entry['size'][0], 'cao': entry['size'][1], 'so_object': len(alphas),
                     'so_object_sam': len(objects), 'so_manh_bo_qua': ignored,
                     'so_muc_alpha': entry['alpha_levels'], 'thoi_gian_ms': round(timings[-1], 1)})
    _save_sheets(tiles, out_dir)
    with (out_dir / 'pilot.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    objects = [r['so_object'] for r in rows]
    sam = [r['so_object_sam'] for r in rows] or [0]
    return {'images': len(rows), 'zero': objects.count(0), 'one': objects.count(1),
            'many': sum(n >= 2 for n in objects), 'many_max': max(objects, default=0),
            'with_fragments': sum(r['so_manh_bo_qua'] > 0 for r in rows),
            'used_sam': splitter is not None, 'sam_median': float(np.median(sam)), 'sam_max': max(sam),
            'ms_median': float(np.median(timings)) if timings else 0.0,
            'min_object_ratio': min_object_ratio}


def pilot_summary(result: dict) -> str:
    one = ('HQ-SAM xem là 1 object' if result['used_sam']
           else 'có thể vẫn là nhiều object CHẠM nhau -> cần SAM')
    lines = [f"Số ảnh thử: {result['images']} (chọn ngẫu nhiên, cố định: số #NNN giống mọi lần chạy)",
             f"  1 object ({one}): {result['one']}",
             f"  2+ object: {result['many']} (nhiều nhất {result['many_max']} object/ảnh)",
             f"  0 object (alpha quá mờ hoặc quá nhỏ): {result['zero']}",
             f"  Có mảnh nhỏ bị bỏ qua, màu xám (< {result['min_object_ratio']:.2%} diện tích ảnh): "
             f"{result['with_fragments']}",
             f"Thời gian xử lý trên máy này: {result['ms_median']:.0f} ms/ảnh (trung vị)"]
    if result['used_sam']:
        lines += [f"Object do HQ-SAM tách: trung vị {result['sam_median']:.0f}, nhiều nhất "
                  f"{result['sam_max']} / ảnh -> mở ảnh trên máy người làm thuê tối đa ~"
                  f"{result['sam_max'] * WORKER_DECODE_SECONDS:.0f} giây"]
    lines += ['',
              'Cách xem: mở sheet_01.png ... Mỗi màu là 1 object, xám là mảnh nhỏ bị bỏ qua.',
              'Ghi lại số #NNN của ảnh tách SAI (thiếu/thừa object, ranh giới lệch) gửi cho Claude.',
              'Ảnh 2+ object có file object_01.png ... trong thư mục objects/ '
              '(alpha 8-bit, 256 mức, mở bằng trình xem ảnh để soi viền).']
    return '\n'.join(lines) + '\n'
