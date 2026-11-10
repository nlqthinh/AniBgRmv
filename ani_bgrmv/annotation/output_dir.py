"""Generated folders (bench pack, ID packages): rebuilt on every run, but a
folder that holds anything else is never deleted."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..hqsam_manifest import check_hqsam_manifest, sha256_file


def fresh_output_dir(path: Path, marker: str, subdirs: tuple[str, ...] = ()) -> None:
    """Empty ``path`` if it is an earlier output (it contains ``marker``)."""
    path = Path(path)
    if path.exists():
        if not (path / marker).is_file() and any(path.iterdir()):
            raise ValueError(f'Thư mục đã có file khác, không ghi đè: {path}')
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for name in subdirs:
        (path / name).mkdir()


def copy_decoder(encoder_path: Path, decoder_path: Path, dest: Path) -> str:
    """Copy the HQ-SAM decoder with its sidecar files; return its SHA-256."""
    decoder_path = Path(decoder_path)
    manifest = check_hqsam_manifest(encoder_path, decoder_path)
    sidecars = [n for n in (manifest.get('sidecars') or {}) if n.startswith(decoder_path.name + '.')]
    for name in (decoder_path.name, *sidecars):
        shutil.copy2(decoder_path.parent / name, Path(dest) / name)
    return sha256_file(decoder_path)
