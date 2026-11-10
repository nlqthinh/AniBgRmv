"""Pair dataset images with their cutouts and fingerprint them by pixel content.

The owner's PC and the annotator's PC arrange and name files differently
(one 'images' folder vs one folder per ID), so images are matched by a SHA-256
of their decoded pixels, never by file name. Layout rule on both PCs: every
folder named 'images' has a sibling folder 'cutout' with a file of the same
relative path and name (any image extension).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

INDEX_FORMAT = 'anibgrmv-dataset-index-v1'
IMAGES_DIR, CUTOUT_DIR = 'images', 'cutout'
JAGGED_ALPHA_LEVELS = 2  # a cutout with only 0/255 has jagged (aliased) edges
_READ_ERRORS = (OSError, ValueError, Image.DecompressionBombError)


@dataclass(frozen=True)
class Pair:
    image: Path
    cutout: Path | None


def _image_files(folder: Path, exts: set[str]) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in exts)


def _stem_key(path: Path, folder: Path) -> str:
    return path.relative_to(folder).with_suffix('').as_posix().lower()


def find_pairs(root: Path, image_exts) -> tuple[list[Pair], list[Path]]:
    """Every image under an 'images' folder with its cutout, plus orphan cutouts."""
    exts = {e.lower() for e in image_exts}
    pairs, orphans = [], []
    for images_dir in sorted(p for p in Path(root).rglob('*') if p.is_dir() and p.name.lower() == IMAGES_DIR):
        cutout_dir = images_dir.parent / CUTOUT_DIR
        cutouts = {_stem_key(p, cutout_dir): p for p in _image_files(cutout_dir, exts)}
        for image in _image_files(images_dir, exts):
            pairs.append(Pair(image, cutouts.pop(_stem_key(image, images_dir), None)))
        orphans.extend(sorted(cutouts.values()))
    return pairs, orphans


def _listing_of(folder: Path, limit: int) -> list[str]:
    """First names in a folder, sub-folders marked with '/'."""
    names = sorted(os.scandir(folder), key=lambda e: e.name.lower())
    return [e.name + ('/' if e.is_dir() else '') for e in names[:limit]]


def describe_layout(root: Path, limit: int = 15) -> dict:
    """What the dataset folder really contains, so a scan that finds no image
    explains itself (wrong folder names, .zip or shortcut folders, extensions,
    folders that cannot be read)."""
    root = Path(root)
    extensions, errors = Counter(), []
    folders = images_dirs = cutout_dirs = 0
    for folder, _, files in os.walk(root, onerror=lambda e: errors.append(f'{e.filename}: {e.strerror}')):
        folders += 1
        name = Path(folder).name.lower()
        images_dirs += name == IMAGES_DIR
        cutout_dirs += name == CUTOUT_DIR
        extensions.update(Path(f).suffix.lower() or '(không có đuôi)' for f in files)
    top = _listing_of(root, limit)
    first = next((n for n in top if n.endswith('/')), None)
    return {'folders': folders - 1, 'images_dirs': images_dirs, 'cutout_dirs': cutout_dirs,
            'extensions': dict(extensions.most_common(10)), 'top_level': top,
            'first_folder': {first: _listing_of(root / first, limit)} if first else {},
            'errors': errors[:limit]}


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def read_alpha(path: Path) -> np.ndarray:
    """8-bit alpha of a cutout: its alpha channel, or the gray value of an L mask."""
    with Image.open(path) as image:
        if image.mode in ('RGBA', 'LA', 'PA') or 'transparency' in image.info:
            return np.asarray(image.convert('RGBA'))[:, :, 3]
        if image.mode == 'L':
            return np.asarray(image)
        raise ValueError(f'cutout không có kênh alpha 8-bit (mode {image.mode})')


def pixel_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{array.shape}|{array.dtype.str}|'.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def alpha_stats(alpha: np.ndarray) -> dict:
    """Number of distinct alpha values and the share of soft (0 < a < 255) pixels."""
    counts = np.bincount(alpha.ravel(), minlength=256)
    visible = int(alpha.size - counts[0])
    soft = visible - int(counts[255])
    return {'alpha_levels': int((counts > 0).sum()),
            'soft_ratio': round(soft / visible, 4) if visible else 0.0}


def _entry(root: Path, pair: Pair) -> tuple[dict, str | None]:
    rgb = read_rgb(pair.image)
    entry = {'image': pair.image.relative_to(root).as_posix(),
             'image_sha256': pixel_sha256(rgb),
             'size': [int(rgb.shape[1]), int(rgb.shape[0])],
             'cutout': None}
    if pair.cutout is None:
        return entry, 'thiếu cutout'
    try:
        alpha = read_alpha(pair.cutout)
    except _READ_ERRORS as exc:
        return entry, f'cutout không đọc được: {exc}'
    entry.update(cutout=pair.cutout.relative_to(root).as_posix(),
                 cutout_sha256=pixel_sha256(alpha), **alpha_stats(alpha))
    if alpha.shape != rgb.shape[:2]:
        return entry, f'cutout {alpha.shape[1]}x{alpha.shape[0]} khác kích thước ảnh'
    return entry, None


def build_index(root: Path, image_exts, progress=None) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Không tìm thấy thư mục dataset: {root}')
    pairs, orphans = find_pairs(root, image_exts)
    entries, problems = [], []
    for done, pair in enumerate(pairs, 1):
        rel = pair.image.relative_to(root).as_posix()
        try:
            entry, problem = _entry(root, pair)
            entries.append(entry)
        except _READ_ERRORS as exc:
            problem = f'ảnh không đọc được: {exc}'
        if problem:
            problems.append({'file': rel, 'problem': problem})
        if progress:
            progress(done, len(pairs))
    problems += [{'file': p.relative_to(root).as_posix(), 'problem': 'cutout không có ảnh gốc'}
                 for p in orphans]
    return {'format': INDEX_FORMAT, 'root_name': root.name, 'entries': entries, 'problems': problems,
            'image_exts': sorted(image_exts), 'layout': describe_layout(root)}


def locate_by_content(root: Path, wanted: dict[str, str], image_exts, cache_file: Path) -> dict[str, tuple[str, str]]:
    """Find images by the SHA-256 of their pixels, whatever their names here.

    ``wanted`` maps sha -> the path the image is expected at; files with that
    path, then with that file name (any folder or extension), then in that top
    folder are read first, so usually only the wanted files are read. Returns
    sha -> (image, cutout) inside ``root``. Hashes are cached in ``cache_file``
    by (path, size, mtime), so only new or changed files are read again. The
    cache is written only when something was found (a wrongly chosen folder
    stays untouched)."""
    root, cache_file = Path(root), Path(cache_file)
    cache = json.loads(cache_file.read_text(encoding='utf-8')) if cache_file.is_file() else {}
    hints = set(wanted.values())
    stems = {PurePosixPath(path).stem.lower() for path in hints}
    folders = {top_folder(path) for path in hints}

    def order(pair: Pair):
        rel = pair.image.relative_to(root).as_posix()
        return rel not in hints, pair.image.stem.lower() not in stems, top_folder(rel) not in folders, rel

    found: dict[str, tuple[str, str]] = {}
    pairs = [p for p in find_pairs(root, image_exts)[0] if p.cutout is not None]
    for pair in sorted(pairs, key=order):
        if len(found) == len(wanted):
            break
        rel, stat = pair.image.relative_to(root).as_posix(), pair.image.stat()
        record = cache.get(rel)
        if record is None or record[:2] != [stat.st_size, stat.st_mtime_ns]:
            try:
                record = cache[rel] = [stat.st_size, stat.st_mtime_ns, pixel_sha256(read_rgb(pair.image))]
            except _READ_ERRORS:
                continue
        if record[2] in wanted and record[2] not in found:
            found[record[2]] = (rel, pair.cutout.relative_to(root).as_posix())
    if found:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache), encoding='utf-8')
    return found


def check_index(index: dict, source) -> dict:
    if not isinstance(index, dict) or index.get('format') != INDEX_FORMAT:
        raise ValueError(f'Không phải file index của bước 0: {source}')
    return index


def usable_entries(index: dict) -> list[dict]:
    """Entries whose cutout exists, can be read and matches the image size."""
    bad = {p['file'] for p in index['problems']}
    return [e for e in index['entries'] if e['cutout'] and e['image'] not in bad]


def sample_entries(index: dict, count: int, seed: int = 0) -> list[dict]:
    usable = usable_entries(index)
    return random.Random(seed).sample(usable, min(count, len(usable)))


def _by_sha(index: dict) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for entry in index['entries']:
        groups[entry['image_sha256']].append(entry)
    return groups


def top_folder(rel: str) -> str:
    return rel.split('/', 1)[0] if '/' in rel else '.'


def compare_indexes(owner: dict, worker: dict) -> dict:
    """Match the two PCs by pixel content; report what only one side has."""
    mine, theirs = _by_sha(owner), _by_sha(worker)
    matched = []
    for sha in sorted(mine.keys() & theirs.keys()):
        a, b = mine[sha][0], theirs[sha][0]
        matched.append({'image_sha256': sha, 'owner_image': a['image'], 'worker_image': b['image'],
                        'cutout_same': bool(a.get('cutout_sha256'))
                        and a.get('cutout_sha256') == b.get('cutout_sha256')})
    totals = Counter(top_folder(e['image']) for e in worker['entries'])
    found = Counter(top_folder(row['worker_image']) for row in matched)
    return {
        'matched': matched,
        'only_owner': sorted(e['image'] for sha in mine.keys() - theirs.keys() for e in mine[sha]),
        'only_worker': sorted(e['image'] for sha in theirs.keys() - mine.keys() for e in theirs[sha]),
        'owner_duplicates': [[e['image'] for e in g] for g in mine.values() if len(g) > 1],
        'worker_duplicates': [[e['image'] for e in g] for g in theirs.values() if len(g) > 1],
        'worker_folders': {name: [found[name], totals[name]] for name in sorted(totals)},
    }


def _listing(title: str, items: list, limit: int = 30) -> list[str]:
    if not items:
        return []
    lines = [f'{title} ({len(items)}):']
    lines += [f'  - {item}' for item in items[:limit]]
    if len(items) > limit:
        lines.append(f'  ... và {len(items) - limit} mục khác')
    return lines


def _layout_lines(index: dict) -> list[str]:
    layout = index['layout']
    lines = ['Cấu trúc đã thấy:',
             f"  Thư mục con: {layout['folders']} (tên 'images': {layout['images_dirs']}, "
             f"tên 'cutout': {layout['cutout_dirs']})",
             '  Đuôi file: ' + (', '.join(f'{ext} x{n}' for ext, n in layout['extensions'].items())
                               or 'không có file nào'),
             f"  Trong {index['root_name']}: " + (', '.join(layout['top_level']) or '(trống)')]
    lines += [f"  Trong {name}: " + (', '.join(children) or '(trống)')
              for name, children in layout['first_folder'].items()]
    lines += [f'  Không đọc được: {error}' for error in layout['errors']]
    return lines + ['']


def index_summary(index: dict) -> str:
    entries = index['entries']
    jagged = [e['cutout'] for e in entries if e['cutout'] and e['alpha_levels'] <= JAGGED_ALPHA_LEVELS]
    duplicates = [', '.join(e['image'] for e in g) for g in _by_sha(index).values() if len(g) > 1]
    lines = []
    if not usable_entries(index):
        lines += ['KHÔNG TÌM THẤY CẶP ẢNH + CUTOUT NÀO.',
                  "Cần: mỗi thư mục tên 'images' có thư mục 'cutout' cùng cấp (ví dụ 1/images và "
                  "1/cutout), file cùng tên, đuôi " + ' '.join(index['image_exts']) + '.',
                  'Thư mục nén (.zip/.rar) phải giải nén; shortcut (.lnk) không dùng được.', '']
    lines += [f"Thư mục dataset: {index['root_name']}",
              f'Số ảnh đọc được: {len(entries)}',
              f'Số ảnh dùng được (có cutout đúng kích thước): {len(usable_entries(index))}',
              f'Số vấn đề: {len(index["problems"])}', '']
    lines += _layout_lines(index)
    lines += _listing('Cutout chỉ có 2 mức alpha (viền sẽ răng cưa)', jagged)
    lines += _listing('Ảnh trùng nội dung', duplicates)
    lines += _listing('Vấn đề', [f"{p['file']}: {p['problem']}" for p in index['problems']])
    return '\n'.join(lines) + '\n'


def compare_summary(result: dict) -> str:
    matched = result['matched']
    changed = [f"{r['owner_image']}  <->  {r['worker_image']}" for r in matched if not r['cutout_same']]
    lines = [f'Ảnh khớp nội dung giữa 2 máy: {len(matched)}',
             f'  trong đó cutout cũng giống hệt: {len(matched) - len(changed)}',
             f"Chỉ có trên máy chủ: {len(result['only_owner'])}",
             f"Chỉ có trên máy người làm thuê: {len(result['only_worker'])}", '',
             'Theo thư mục cấp 1 trên máy người làm thuê (khớp / tổng):']
    lines += [f'  {name}: {pair[0]} / {pair[1]}' for name, pair in result['worker_folders'].items()]
    lines.append('')
    lines += _listing('Ảnh khớp nhưng cutout KHÁC nhau (cần thống nhất trước khi gán nhãn)', changed)
    lines += _listing('Chỉ có trên máy chủ', result['only_owner'])
    lines += _listing('Chỉ có trên máy người làm thuê', result['only_worker'])
    lines += _listing('Trùng nội dung trên máy chủ', [', '.join(g) for g in result['owner_duplicates']])
    lines += _listing('Trùng nội dung trên máy người làm thuê',
                      [', '.join(g) for g in result['worker_duplicates']])
    return '\n'.join(lines) + '\n'
