"""Small caches private to this Windows user; never ship them with the EXE.

- Batch cache: keeps cutout PNGs off RAM while a multi-image job awaits Save.
- File digests: SHA-256 of large model files, remembered by (path, size,
  mtime) so a ~1 GB HQ-SAM encoder is hashed once, not on every Edit session.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from .hqsam_manifest import sha256_file


def cache_root() -> Path:
    base = os.getenv('LOCALAPPDATA')
    if base:
        root = Path(base) / 'AniBgRmv' / 'cache'
    else:
        root = Path.home() / '.cache' / 'AniBgRmv'
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_batch_cache() -> tempfile.TemporaryDirectory:
    parent = cache_root() / 'batch'
    parent.mkdir(parents=True, exist_ok=True)
    # Crashed or forcibly closed jobs leave temporary PNGs behind. Prune only
    # old staging folders; never touch current sessions or final outputs.
    cutoff = time.time() - 3 * 24 * 3600
    for old in parent.glob('job_*'):
        try:
            if old.is_dir() and old.stat().st_mtime < cutoff:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return tempfile.TemporaryDirectory(prefix='job_', dir=parent)


def _digest_store() -> Path:
    return cache_root() / 'file_digests.json'


def cached_sha256(path: Path) -> str:
    """SHA-256 of ``path``; recomputed only when its size or mtime changes."""
    path = Path(path).resolve()
    info = path.stat()
    key = f'{path}|{info.st_size}|{info.st_mtime_ns}'
    store = _digest_store()
    try:
        known = json.loads(store.read_text(encoding='utf-8'))
        if not isinstance(known, dict):
            known = {}
    except (OSError, ValueError):
        known = {}
    if isinstance(known.get(key), str):
        return known[key]
    digest = sha256_file(path)
    # Drop stale entries for the same file (older size/mtime).
    known = {k: v for k, v in known.items() if not k.startswith(f'{path}|')}
    known[key] = digest
    try:
        temporary = store.with_name(store.name + '.tmp')
        temporary.write_text(json.dumps(known, indent=1), encoding='utf-8')
        os.replace(temporary, store)
    except OSError:
        pass  # caching is optional; a read-only profile must not block Edit
    return digest


def clear_cache(kind: str = 'all') -> None:
    """Optional manual cleanup; only known cache entries can be removed."""
    if kind not in ('all', 'batch', 'file_digests'):
        raise ValueError('Unknown cache type')
    root = cache_root()
    if kind in ('all', 'batch'):
        shutil.rmtree(root / 'batch', ignore_errors=True)
    if kind in ('all', 'file_digests'):
        _digest_store().unlink(missing_ok=True)
    # Embedding cache written by older releases (SAM ViT-B); no longer read.
    if kind == 'all':
        shutil.rmtree(root / 'sam_embeddings', ignore_errors=True)
