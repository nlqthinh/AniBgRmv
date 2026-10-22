"""Small local, private-to-this-Windows-user caches; never ship these with EXE.

The batch cache keeps PNGs off RAM while a multi-image job awaits Save.
SAM cache stores *embeddings* (not source images) to avoid another 1024px
encoder inference after SAM sessions are unloaded on leaving Edit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


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


def _sam_cache_key(image_path: str, encoder_path: str) -> str:
    """Invalidate when an image or encoder changes; never cache just by name."""
    parts = []
    for path in (image_path, encoder_path):
        p = Path(path).resolve()
        info = p.stat()
        parts.append((str(p), info.st_size, info.st_mtime_ns))
    payload = json.dumps(['sam-vitb-embedding-v1', parts], ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def load_sam_embedding(image_path: str, encoder_path: str):
    try:
        path = cache_root() / 'sam_embeddings' / (_sam_cache_key(image_path, encoder_path) + '.npz')
        with np.load(path, allow_pickle=False) as data:
            embedding = np.asarray(data['embedding'], dtype=np.float32)
            original_hw = tuple(int(x) for x in data['original_hw'])
            resized_hw = tuple(int(x) for x in data['resized_hw'])
        if embedding.shape != (1, 256, 64, 64):
            return None
        if len(original_hw) != 2 or len(resized_hw) != 2 or min(*original_hw, *resized_hw) < 1:
            return None
        os.utime(path, None)
        return embedding, original_hw, resized_hw
    except (OSError, ValueError, KeyError, EOFError, RuntimeError):
        return None  # cache is optional; a corrupt cache must never break Edit


def save_sam_embedding(image_path: str, encoder_path: str, predictor, max_mib: int = 128):
    if getattr(predictor, 'embedding', None) is None:
        return
    target_dir = cache_root() / 'sam_embeddings'
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / (_sam_cache_key(image_path, encoder_path) + '.npz')
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target_dir, suffix='.npz', delete=False) as fp:
            temp_path = Path(fp.name)
            np.savez_compressed(
                fp,
                embedding=predictor.embedding,
                original_hw=np.asarray(predictor.original_hw, dtype=np.int32),
                resized_hw=np.asarray(predictor.resized_hw, dtype=np.int32),
            )
        os.replace(temp_path, destination)
        limit = max(16, max_mib) * 1024 * 1024
        files = sorted(target_dir.glob('*.npz'), key=lambda p: p.stat().st_mtime)
        used = sum(p.stat().st_size for p in files)
        for old in files:
            if used <= limit:
                break
            if old == destination and len(files) == 1:
                break
            n = old.stat().st_size
            try:
                old.unlink()
                used -= n
            except OSError:
                pass
    except (OSError, ValueError):
        pass  # a full disk must not block Guided Edit
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def clear_cache(kind: str = 'all') -> None:
    """Optional manual cleanup; only known subdirectories can be removed."""
    if kind not in ('all', 'batch', 'sam_embeddings'):
        raise ValueError('Unknown cache type')
    root = cache_root()
    names = ('batch', 'sam_embeddings') if kind == 'all' else (kind,)
    for name in names:
        shutil.rmtree(root / name, ignore_errors=True)
