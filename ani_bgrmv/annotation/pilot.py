"""Bước 0 pilot: split a random sample of cutouts and draw contact sheets.

Output folder:
  sheet_NN.png   20 thumbnails per page: every object has its own color,
                 fragments are gray, caption '#NNN obj=K frag=F'
  objects/NNN_<name>/object_K.png   RGBA per object, only for images with
                 2+ objects (same 256-level alpha as the cutout)
  pilot.csv      one row per image (Excel-friendly UTF-8)
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .alpha_split import AlphaSplit, instance_alpha, split_alpha
from .dataset_index import read_alpha, read_rgb, sample_entries

PALETTE = np.array([(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
                    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)], np.float32)
FRAGMENT_COLOR = (128, 128, 128)
THUMB, PAD, CAPTION, COLUMNS, ROWS = 256, 6, 22, 5, 4
CSV_FIELDS = ('stt', 'trang', 'vi_tri', 'anh', 'rong', 'cao', 'so_object', 'so_manh_nho',
              'dien_tich_object', 'so_muc_alpha', 'thoi_gian_ms')


def overlay(rgb: np.ndarray, alpha: np.ndarray, split: AlphaSplit) -> np.ndarray:
    """Image dimmed to 40%; each object tinted with its own color by its alpha."""
    colors = np.zeros((len(split.areas) + 1, 3), np.float32)
    for label in range(1, len(split.areas) + 1):
        colors[label] = (PALETTE[(label - 1) % len(PALETTE)] if label <= split.object_count
                         else FRAGMENT_COLOR)
    weight = (alpha.astype(np.float32) / 255.0)[..., None]
    out = rgb.astype(np.float32) * 0.4 + colors[split.labels] * 0.6 * weight
    return out.clip(0, 255).astype(np.uint8)


def _thumbnail(image: np.ndarray, caption: str) -> Image.Image:
    h, w = image.shape[:2]
    scale = (THUMB - 2 * PAD) / max(h, w)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
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


def _save_objects(rgb, alpha, split: AlphaSplit, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for label in range(1, split.object_count + 1):
        rgba = np.dstack([rgb, instance_alpha(alpha, split, label)])
        Image.fromarray(rgba, 'RGBA').save(folder / f'object_{label}.png')


def run_pilot(root: Path, index: dict, out_dir: Path, *, count: int, min_object_ratio: float,
              seed: int = 0) -> dict:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, rows, timings = [], [], []
    for number, entry in enumerate(sample_entries(index, count, seed), 1):
        rgb, alpha = read_rgb(root / entry['image']), read_alpha(root / entry['cutout'])
        started = time.perf_counter()
        split = split_alpha(alpha, min_object_ratio)
        timings.append((time.perf_counter() - started) * 1000)
        tiles.append(_thumbnail(overlay(rgb, alpha, split),
                                f'#{number:03d} obj={split.object_count} frag={split.fragment_count}'))
        if split.object_count >= 2:
            _save_objects(rgb, alpha, split, out_dir / 'objects' / f'{number:03d}_{Path(entry["image"]).stem}')
        slot = number - 1
        rows.append({'stt': number, 'trang': slot // (COLUMNS * ROWS) + 1,
                     'vi_tri': slot % (COLUMNS * ROWS) + 1, 'anh': entry['image'],
                     'rong': entry['size'][0], 'cao': entry['size'][1],
                     'so_object': split.object_count, 'so_manh_nho': split.fragment_count,
                     'dien_tich_object': ' '.join(map(str, split.areas[:split.object_count])),
                     'so_muc_alpha': entry['alpha_levels'], 'thoi_gian_ms': round(timings[-1], 1)})
    _save_sheets(tiles, out_dir)
    with (out_dir / 'pilot.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    objects = [r['so_object'] for r in rows]
    return {'images': len(rows), 'zero': objects.count(0), 'one': objects.count(1),
            'many': sum(n >= 2 for n in objects),
            'with_fragments': sum(r['so_manh_nho'] > 0 for r in rows),
            'split_ms_median': float(np.median(timings)) if timings else 0.0,
            'min_object_ratio': min_object_ratio}


def pilot_summary(result: dict) -> str:
    return '\n'.join([
        f"Số ảnh thử: {result['images']} (chọn ngẫu nhiên, cố định)",
        f"  1 object (có thể vẫn là nhiều object CHẠM nhau -> cần SAM): {result['one']}",
        f"  2+ object tách rời (tách xong không cần SAM): {result['many']}",
        f"  0 object (alpha quá mờ hoặc quá nhỏ): {result['zero']}",
        f"  Có mảnh nhỏ màu xám (< {result['min_object_ratio']:.2%} diện tích ảnh): "
        f"{result['with_fragments']}",
        f"Thời gian tách trung bình: {result['split_ms_median']:.0f} ms/ảnh",
        '',
        'Cách xem: mở sheet_01.png ... Mỗi màu là 1 object, xám là mảnh nhỏ.',
        'Ghi lại số thứ tự (#NNN) của ảnh tách SAI để gửi lại cho Claude.',
        'Ảnh 2+ object có file RGBA từng object trong thư mục objects/ (alpha giữ nguyên 256 mức).',
    ]) + '\n'
