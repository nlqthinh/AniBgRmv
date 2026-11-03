"""Validate the HQ-SAM ONNX export manifest (standard library only).

tools/export_hqsam_onnx.py writes '<encoder stem>.manifest.json' after it has
verified the two graphs against PyTorch. The app, the PyInstaller spec and the
bundle audit all use this single check, so an encoder and decoder from
different exports (or a half-copied model folder) are never paired.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

# File-format identifier written by existing exports; not a model name.
MANIFEST_FORMAT = "anibgrmv-hq3-sam2-point-onnx-v1"


def manifest_path_for(encoder_path: Path) -> Path:
    return Path(encoder_path).with_suffix(".manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hqsam_manifest(encoder_path: Path, decoder_path: Path, *,
                         digest: Callable[[Path], str] | None = None) -> dict:
    """Return the manifest, or raise ValueError listing every problem.

    Sizes and filenames are always checked. Pass ``digest`` (for example a
    cached SHA-256) to also compare file contents.
    """
    encoder, decoder = Path(encoder_path), Path(decoder_path)
    manifest_file = manifest_path_for(encoder)
    for path in (encoder, decoder, manifest_file):
        if not path.is_file():
            raise ValueError(f"Thiếu file HQ-SAM ONNX: {path}. Chạy bat/prepare_hqsam_onnx.bat.")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Manifest HQ-SAM không đọc được: {manifest_file}: {exc}") from exc

    problems = []
    if manifest.get("format") != MANIFEST_FORMAT or manifest.get("verified") is not True:
        problems.append("manifest chưa được exporter xác minh")
    if manifest.get("encoder_filename") != encoder.name or manifest.get("decoder_filename") != decoder.name:
        problems.append("tên encoder/decoder khác manifest")
    files = [(encoder, manifest.get("encoder_size"), manifest.get("encoder_sha256")),
             (decoder, manifest.get("decoder_size"), manifest.get("decoder_sha256"))]
    for name, info in (manifest.get("sidecars") or {}).items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(info, dict):
            problems.append(f"tên sidecar không an toàn: {name!r}")
            continue
        files.append((encoder.parent / name, info.get("size"), info.get("sha256")))
    for path, size, sha in files:
        if not path.is_file() or path.stat().st_size != size:
            problems.append(f"thiếu hoặc sai kích thước: {path.name}")
        elif digest is not None and digest(path) != sha:
            problems.append(f"nội dung khác manifest (SHA-256): {path.name}")
    if problems:
        raise ValueError(f"HQ-SAM ONNX không khớp manifest {manifest_file.name}: " + "; ".join(problems)
                         + ". Hãy export lại bằng bat/prepare_hqsam_onnx.bat.")
    return manifest
