"""The annotator's work on one ID package: images, status and output files.

<dataset>/annotations/
  _vi_tri_anh.json            where each image is (found by pixel content), cached
<dataset>/annotations/<ID>/
  _trang_thai.json            status of every image: todo | single | split | hard
  _edits/<name>.json + .npz   the edit (operations, base map, guided masks) to reopen it
  <name>/object_01.png ...    8-bit alpha per object, only for split images
Files are replaced only once their new version is complete (temporary file or
folder, then rename), so a failed write never loses the previous result.
An image renamed after its package was made keeps its work: opening the
package moves that work to the new name. Everything else in the dataset is
only read. Runs on the annotator window's worker thread (no Qt here).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from ..engine.hqsam_onnx import HqSamDecoder, HqSamFeatures
from ..engine.onnx_device import OnnxDevice
from ..app_config import IMAGE_EXTS
from .dataset_index import IMAGES_DIR, locate_by_content, read_alpha, read_rgb
from .alpha_split import split_alpha
from .image_edit import EDIT_FORMAT, ImageEdit
from .package import read_package
from .partition import SamObject, partition

STATUS_FORMAT = 'anibgrmv-annotation-status-v1'
ANNOTATIONS_DIR = 'annotations'
STATUS_FILE, EDITS_DIR = '_trang_thai.json', '_edits'
LOCATION_CACHE = '_vi_tri_anh.json'   # shared by all packages of one dataset
TODO, SINGLE, SPLIT, HARD = 'todo', 'single', 'split', 'hard'
WORKER_CPU_THREADS = 2   # the annotator's Pentium G4400


@dataclass(frozen=True)
class Item:
    index: int
    name: str              # output folder / file name, unique in the package
    image: str             # path inside the dataset folder
    cutout: str
    image_sha256: str
    features: Path | None
    proposal: Path | None
    preview: Path | None
    pieces: tuple          # proposed pieces (JSON)
    proposed_objects: int | None


def _item_name(image: str) -> str:
    parts = PurePosixPath(image).with_suffix('').parts
    lowered = [p.lower() for p in parts]
    inside = parts[lowered.index(IMAGES_DIR) + 1:] if IMAGES_DIR in lowered else parts[-1:]
    return '__'.join(inside)


def _load_logits(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data['logits']


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


class Workspace:
    def __init__(self, dataset_root: Path, package_path: Path):
        self.root, self.package_path = Path(dataset_root), Path(package_path)
        manifest = read_package(self.package_path)
        self.folder = manifest['folder']
        self.min_object_ratio = float(manifest['min_object_ratio'])
        self._decoder_path = self.package_path / manifest['decoder_filename']
        self._decoder: HqSamDecoder | None = None
        package, entries = self.package_path, manifest['items']
        # The package was made for the annotator's file names; this dataset may
        # name and arrange them differently, so images are found by content.
        located = locate_by_content(self.root, {e['image_sha256']: e['worker_image'] for e in entries},
                                    IMAGE_EXTS, self.root / ANNOTATIONS_DIR / LOCATION_CACHE)
        if not located:
            raise ValueError(f'Không tìm thấy ảnh nào của gói ID {self.folder} trong {self.root} '
                             '(ảnh được so theo nội dung, không theo tên). Hãy chọn thư mục dataset '
                             'chứa đúng các ảnh mà gói này được tạo cho.')
        self.items = []
        for index, entry in enumerate(entries):
            image, cutout = located.get(entry['image_sha256'], (entry['worker_image'], entry['worker_cutout']))
            self.items.append(Item(index, _item_name(image), image, cutout, entry['image_sha256'],
                                   package / entry['features'] if entry['features'] else None,
                                   package / entry['proposal'] if entry['proposal'] else None,
                                   package / entry['preview'] if entry['preview'] else None,
                                   tuple(entry['pieces']), entry['objects']))
        self.missing_files = [e['worker_image'] for e in entries if e['image_sha256'] not in located]
        self.out_dir = self.root / ANNOTATIONS_DIR / self.folder
        (self.out_dir / EDITS_DIR).mkdir(parents=True, exist_ok=True)
        status_file = self.out_dir / STATUS_FILE
        saved = json.loads(status_file.read_text(encoding='utf-8')) if status_file.is_file() else {}
        self._status: dict[str, dict] = saved.get('images', {})
        self._keep_work_of_renamed({_item_name(e['worker_image']): item for e, item in zip(entries, self.items)})

    # -- images renamed after the package was made ----------------------------
    def _has_work(self, name: str) -> bool:
        return name in self._status or self._edit_files(name)[0].is_file() or (self.out_dir / name).is_dir()

    def _keep_work_of_renamed(self, by_package_name: dict[str, Item]) -> None:
        """Work saved under the name an image had when the package was made
        (status, draft, object files) moves to its name today. Never overwrites:
        a move whose new name already holds work is left as it is."""
        moves = {old: item for old, item in by_package_name.items() if old != item.name and self._has_work(old)}
        moves = {old: item for old, item in moves.items()
                 if not self._has_work(item.name) and item.name not in moves}
        for old, item in moves.items():
            for source, target in zip(self._edit_files(old), self._edit_files(item.name)):
                if source.is_file():
                    os.replace(source, target)
            if (self.out_dir / old).is_dir():
                os.replace(self.out_dir / old, self.out_dir / item.name)
            if old in self._status:
                self._status[item.name] = {**self._status.pop(old), 'image': item.image}
        if moves:
            self._save_status()

    # -- status ---------------------------------------------------------------
    def status(self, item: Item) -> str:
        return self._status.get(item.name, {}).get('status', TODO)

    def progress(self) -> dict[str, int]:
        counts = {TODO: 0, SINGLE: 0, SPLIT: 0, HARD: 0}
        for item in self.items:
            counts[self.status(item)] += 1
        return counts

    def _save_status(self) -> None:
        _write_json(self.out_dir / STATUS_FILE, {'format': STATUS_FORMAT, 'folder': self.folder,
                                                 'images': self._status})

    # -- open / save ----------------------------------------------------------
    def _decode_for(self, item: Item):
        """HQ-SAM decode for this image (found by content), or None without features."""
        if item.features is None:
            return None
        if self._decoder is None:
            self._decoder = HqSamDecoder(self._decoder_path, OnnxDevice('cpu', cpu_threads=WORKER_CPU_THREADS))
        features = HqSamFeatures.load(item.features)
        decoder = self._decoder
        return lambda coords, labels: decoder.logits(features, coords, labels)[0]

    def _edit_files(self, name: str) -> tuple[Path, Path]:
        # Append, never with_suffix: 'sample.v1' and 'sample.v2' are different images.
        folder = self.out_dir / EDITS_DIR
        return folder / f'{name}.json', folder / f'{name}.npz'

    def _proposal(self, item: Item, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """The package's auto split as a label map (separated regions only without it)."""
        pieces = ([SamObject.from_json(d, l) for d, l in zip(item.pieces, _load_logits(item.proposal))]
                  if item.proposal is not None else [])
        return partition(rgb, alpha, split_alpha(alpha, self.min_object_ratio), pieces, self.min_object_ratio)

    def open(self, item: Item) -> ImageEdit:
        rgb, alpha = read_rgb(self.root / item.image), read_alpha(self.root / item.cutout)
        decode = self._decode_for(item)
        recipe_file, arrays_file = self._edit_files(item.name)
        recipe = json.loads(recipe_file.read_text(encoding='utf-8')) if recipe_file.is_file() else {}
        if recipe.get('format') != EDIT_FORMAT:   # no draft yet (or one from an older app version)
            return ImageEdit(rgb, alpha, self._proposal(item, rgb, alpha), decode=decode)
        with np.load(arrays_file, allow_pickle=False) as arrays:
            edit = ImageEdit(rgb, alpha, arrays['base'], decode=decode, ops=recipe['ops'], masks=arrays['masks'])
        if recipe['active'] is not None:
            edit.select(recipe['active'])
        return edit

    def save_draft(self, item: Item, edit: ImageEdit) -> None:
        recipe, arrays = edit.recipe()
        recipe_file, arrays_file = self._edit_files(item.name)
        _write_npz(arrays_file, arrays)
        _write_json(recipe_file, recipe)

    def finish(self, item: Item, edit: ImageEdit, mark: str, seconds: float = 0.0) -> str:
        """mark: 'save' (split if 2+ objects, else single), 'single' or 'hard'."""
        self.save_draft(item, edit)
        alphas = edit.object_alphas() if mark == 'save' else []
        status = SPLIT if len(alphas) >= 2 else HARD if mark == 'hard' else SINGLE
        folder = self.out_dir / item.name
        written = self._write_objects(item.name, alphas) if status == SPLIT else None
        # Only now, with the new result complete, the old one goes away.
        old = self.out_dir / EDITS_DIR / f'{item.name}.objects_old'
        shutil.rmtree(old, ignore_errors=True)
        if folder.is_dir():
            os.replace(folder, old)
        if written is not None:
            os.replace(written, folder)
        objects = {SPLIT: len(alphas), SINGLE: 1, HARD: 0}[status]
        self._status[item.name] = {'status': status, 'objects': objects,
                                   'image': item.image, 'seconds': round(seconds, 1),
                                   'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        self._save_status()
        shutil.rmtree(old, ignore_errors=True)
        return status

    def _write_objects(self, name: str, alphas: list) -> Path:
        """object_01.png ... in a temporary folder; removed again if any write fails."""
        folder = self.out_dir / EDITS_DIR / f'{name}.objects_tmp'
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir()
        try:
            for number, alpha in enumerate(alphas, 1):
                Image.fromarray(alpha, 'L').save(folder / f'object_{number:02d}.png')
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return folder
