"""Which model files a release build ships, and the start-up check that they are present.

The GPU release is two archives (tools/package_release.py): the app and its
models, because together they pass GitHub's 2 GiB limit per file. The build
(AniBGRmv.spec) writes MODEL_LIST next to the EXE; on start the frozen app
checks every listed file, so a missing models archive gives one clear message
instead of a model error later. In source mode there is no list: nothing to check.
"""
from __future__ import annotations

import json
from pathlib import Path

MODEL_LIST = 'model_files.json'


def write_model_list(path: Path, relative_paths) -> Path:
    """Sorted POSIX paths, relative to the EXE folder."""
    names = sorted({Path(p).as_posix() for p in relative_paths})
    Path(path).write_text(json.dumps(names, indent=2) + '\n', encoding='utf-8')
    return Path(path)


def read_model_list(base_dir: Path) -> list[str]:
    listing = Path(base_dir) / MODEL_LIST
    if not listing.is_file():
        return []
    return json.loads(listing.read_text(encoding='utf-8'))


def missing_model_files(base_dir: Path) -> list[str]:
    return [name for name in read_model_list(base_dir) if not (Path(base_dir) / name).is_file()]


def missing_models_message(missing: list[str]) -> str:
    shown = '\n'.join(f'  • {name}' for name in missing[:6])
    more = f'\n  … và {len(missing) - 6} file khác' if len(missing) > 6 else ''
    return ('Thiếu file model của bản phát hành:\n' + shown + more + '\n\n'
            'Bản GPU gồm 2 file: …_App.7z và …_Models.7z. Hãy giải nén CẢ HAI '
            'vào cùng một chỗ (cùng thư mục AniBGRmv), rồi mở lại app.')
