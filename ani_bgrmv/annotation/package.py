"""Per-ID annotation packages: built on the owner's GPU PC, read by the annotator.

goi_ID_<ID>/
  package.json              one item per image
  <decoder>.onnx (+ sidecars)  the HQ-SAM decoder matching the features
  features/<sha>.npz        FP16 encoder features (~8.4 MB each)
  proposals/<sha>.npz       auto split: decoder logits of each proposed piece (FP16)
  preview/<sha>.png         small colored preview of the proposal for the grid
The owner's file names start with their ID ('7_2D_Chibi (3)' is an image of ID 7),
and the annotator's PC uses the same file names, in one folder per ID. Items are
keyed by the SHA-256 of the image pixels: the annotator's app finds each image by
content wherever it is, the same names only make that fast.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from ..engine.hqsam_onnx import HqSamDecoder, HqSamOnnxPredictor
from ..hqsam_manifest import sha256_file
from .alpha_split import split_alpha
from .auto_split import auto_split
from .dataset_index import read_alpha, read_rgb, usable_entries
from .output_dir import copy_decoder, fresh_output_dir
from .overlay import overlay, thumbnail
from .partition import partition

PACKAGE_FORMAT = 'anibgrmv-annotation-package-v1'
PACKAGE_FILE = 'package.json'
PREVIEW_SIDE = 256
OTHER_ID = 'khac'   # the ID of owner images whose file name does not start with '<number>_'
_ID_PREFIX = re.compile(r'(\d+)_')


@dataclass(frozen=True)
class Group:
    """The owner's images of one ID: one package."""
    folder: str      # the ID; results go to annotations/<ID> on the annotator's PC
    entries: tuple   # owner index entries, in file name order

    @property
    def package(self) -> str:
        return f'goi_ID_{self.folder}'


def _id_order(folder: str):
    return (0, int(folder), '') if folder.isdigit() else (1, 0, folder)   # 1, 2, ..., 10, then names


def _natural(path: str) -> list:
    """Sort key putting 'a (2)' before 'a (10)'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', path)]


def _owner_id(image: str) -> str:
    """'images/7_2D_Chibi (3).jpg' -> '7'."""
    match = _ID_PREFIX.match(PurePosixPath(image).name)
    return str(int(match.group(1))) if match else OTHER_ID


def package_groups(owner_index: dict) -> list[Group]:
    """The owner's usable images by ID, in ID order (an image the owner has twice goes in once)."""
    ids: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for entry in sorted(usable_entries(owner_index), key=lambda e: _natural(e['image'])):
        if entry['image_sha256'] not in seen:
            seen.add(entry['image_sha256'])
            ids.setdefault(_owner_id(entry['image']), []).append(entry)
    return [Group(folder, tuple(entries)) for folder, entries in sorted(ids.items(), key=lambda p: _id_order(p[0]))]


def packaged_images(out_root: Path) -> dict[str, str]:
    """SHA-256 -> folder name of the finished package holding that image.
    Packages are recognised by content, so they still count after folders or
    images are renamed (a package interrupted while being built has no
    package.json yet)."""
    held: dict[str, str] = {}
    for manifest_file in sorted(Path(out_root).glob(f'goi_ID_*/{PACKAGE_FILE}')):
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        held.update(dict.fromkeys((item['image_sha256'] for item in manifest['items']), manifest_file.parent.name))
    return held


def parse_ids(text: str) -> list[str]:
    """'4-10', '1,3,5-8', 'khac' -> those IDs in that order, without repeats."""
    chosen: list[str] = []
    for token in (t.strip() for t in text.split(',')):
        if not token:
            continue
        first, dash, last = (part.strip() for part in token.partition('-'))
        if dash and not (first.isdigit() and last.isdigit()):
            raise ValueError(f'Không hiểu dải ID "{token}": hãy viết kiểu 4-10.')
        names = ([str(i) for i in range(int(first), int(last) + 1)] if dash
                 else [str(int(token)) if token.isdigit() else token])
        chosen += [n for n in names if n not in chosen]
    return chosen


def plan_groups(groups: list[Group], out_root: Path, text: str) -> list[Group]:
    """The IDs to package: those asked for, or (empty text) every ID none of
    whose images is in a package yet."""
    if not text.strip():
        held = packaged_images(out_root)
        return [g for g in groups if not any(e['image_sha256'] in held for e in g.entries)]
    by_id = {group.folder: group for group in groups}
    names = parse_ids(text)
    unknown = [name for name in names if name not in by_id]
    if unknown:
        raise ValueError(f"Dataset máy chủ không có ảnh của ID {', '.join(unknown)} "
                         f"(chỉ có ID {', '.join(by_id)}).")
    return [by_id[name] for name in names]


def _package_note(shas: list[str], held: dict[str, str]) -> str:
    packages = sorted({held[sha] for sha in shas if sha in held}, key=_natural)
    if not packages:
        return 'chưa có gói'
    names = ', '.join(packages)
    inside = sum(sha in held for sha in shas)
    if inside == len(shas):
        return f'đã có gói ({names})'
    return f'{inside}/{len(shas)} ảnh đã có gói ({names}), {len(shas) - inside} ảnh mới chưa có gói'


def groups_overview(groups: list[Group], out_root: Path) -> str:
    held = packaged_images(out_root)
    lines = [f'Dataset máy chủ: {sum(len(g.entries) for g in groups)} ảnh, chia ID theo số ở đầu tên file '
             f'(7_xxx (3) thuộc ID 7; tên không bắt đầu bằng số_ thuộc ID {OTHER_ID}):']
    for group in groups:
        first, last = (PurePosixPath(e['image']).stem for e in (group.entries[0], group.entries[-1]))
        lines.append(f'  ID {group.folder:<4} {len(group.entries)} ảnh ({first} … {last}) · '
                     f"{_package_note([e['image_sha256'] for e in group.entries], held)}")
    return '\n'.join(lines) + '\n'


def check_owner_files(root: Path, groups: list[Group]) -> None:
    """Before hours of GPU work: ``root`` must be the dataset the owner's index was made from."""
    missing = [e['image'] for group in groups for e in group.entries
               if not ((Path(root) / e['image']).is_file() and (Path(root) / e['cutout']).is_file())]
    if missing:
        raise ValueError(f'Dataset {root} không khớp index máy chủ: thiếu {len(missing)} ảnh, ví dụ '
                         f'{missing[0]}. Hãy kéo đúng thư mục dataset đã dùng ở bước 0.')


def _package_item(root: Path, entry: dict, out: Path, predictor: HqSamOnnxPredictor,
                  decoder: HqSamDecoder, min_object_ratio: float) -> dict:
    sha = entry['image_sha256']
    rgb, alpha = read_rgb(root / entry['image']), read_alpha(root / entry['cutout'])
    features = predictor.encode(rgb).half()
    features.save(out / 'features' / f'{sha}.npz')
    split = split_alpha(alpha, min_object_ratio)
    pieces = auto_split(alpha, split, lambda coords, labels: decoder.logits(features, coords, labels)[0],
                        min_object_ratio)
    labels = partition(rgb, alpha, split, pieces, min_object_ratio)
    if pieces:
        with (out / 'proposals' / f'{sha}.npz').open('wb') as stream:
            np.savez(stream, logits=np.stack([p.logits for p in pieces]).astype(np.float16))
    Image.fromarray(thumbnail(overlay(rgb, alpha, labels, edges=True), PREVIEW_SIDE)).save(
        out / 'preview' / f'{sha}.png')
    # worker_image / worker_cutout: where the annotator's app looks first (same file names).
    return {'image_sha256': sha, 'worker_image': entry['image'], 'worker_cutout': entry['cutout'],
            'features': f'features/{sha}.npz', 'proposal': f'proposals/{sha}.npz' if pieces else None,
            'preview': f'preview/{sha}.png', 'pieces': [p.to_json() for p in pieces],
            'objects': int(labels.max())}


def build_package(root: Path, group: Group, out_root: Path, *, predictor: HqSamOnnxPredictor,
                  decoder: HqSamDecoder, min_object_ratio: float, progress=None) -> dict:
    """Owner PC: the package of one ID; returns a summary."""
    out = Path(out_root) / group.package
    fresh_output_dir(out, PACKAGE_FILE, ('features', 'proposals', 'preview'))
    items = []
    for done, entry in enumerate(group.entries, 1):
        items.append(_package_item(Path(root), entry, out, predictor, decoder, min_object_ratio))
        if progress:
            progress(done, len(group.entries))
    decoder_sha256 = copy_decoder(predictor.encoder_path, predictor.decoder_path, out)
    manifest = {'format': PACKAGE_FORMAT, 'folder': group.folder, 'decoder_filename': predictor.decoder_path.name,
                'decoder_sha256': decoder_sha256, 'min_object_ratio': min_object_ratio, 'items': items}
    (out / PACKAGE_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    return {'package': group.package, 'images': len(items), 'multi': sum(i['objects'] >= 2 for i in items)}


def read_package(path: Path) -> dict:
    """The manifest of a package folder, after checking its decoder file."""
    path = Path(path)
    manifest_file = path / PACKAGE_FILE
    if not manifest_file.is_file():
        raise ValueError(f'Không phải thư mục gói (thiếu {PACKAGE_FILE}): {path}')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('format') != PACKAGE_FORMAT:
        raise ValueError(f'{PACKAGE_FILE} không đúng định dạng gói gán nhãn: {path}')
    decoder = path / manifest['decoder_filename']
    if not decoder.is_file() or sha256_file(decoder) != manifest['decoder_sha256']:
        raise ValueError(f'Decoder trong gói bị thiếu hoặc hỏng, hãy chép lại gói: {path}')
    return manifest


def package_report_text(reports: list[dict]) -> str:
    return ''.join(f"Gói {r['package']}: {r['images']} ảnh, đề xuất 2+ object {r['multi']}\n" for r in reports)
