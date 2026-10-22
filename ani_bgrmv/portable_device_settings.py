"""Computer-local compute settings: never put a build machine's GPU into an EXE release.

This module deliberately uses only the standard library; easy to test without Qt/ORT.
"""
import json
import os
import sys
from pathlib import Path


def build_variant():
    """Return 'cpu' for a CPU-only release; 'gpu' for development/GPU release."""
    if not getattr(sys, 'frozen', False):
        return 'gpu'
    bundled = Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
    marker = bundled / 'build_flavor.json'
    try:
        value = json.loads(marker.read_text(encoding='utf-8')).get('variant')
        return value if value in ('cpu', 'gpu') else 'gpu'
    except (OSError, ValueError, TypeError):
        return 'gpu'


def machine_config_path():
    if os.name == 'nt':
        base = Path(os.environ.get('LOCALAPPDATA') or (Path.home() / 'AppData' / 'Local'))
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or (Path.home() / '.config'))
    return base / 'AniBgRmv' / 'device_selection.json'


def _normalize(selection):
    value = selection if isinstance(selection, dict) else {}
    mode = value.get('mode')
    if mode != 'gpu':
        return {'mode': 'cpu', 'gpu_id': None, 'gpu_name': None}
    gpu_id = value.get('gpu_id')
    gpu_name = value.get('gpu_name')
    # Only an explicitly named NVIDIA CUDA device may be persisted as GPU.
    if not isinstance(gpu_id, str) or not gpu_id.startswith('cuda:'):
        return {'mode': 'cpu', 'gpu_id': None, 'gpu_name': None}
    return {'mode': 'gpu', 'gpu_id': gpu_id,
            'gpu_name': gpu_name if isinstance(gpu_name, str) else None}


def load_machine_selection():
    # Crucial: missing local config means CPU, regardless of the JSON carried
    # in the distributed zip. A user's selection travels with their PC, not EXE.
    if build_variant() == 'cpu':
        return _normalize(None)
    try:
        return _normalize(json.loads(machine_config_path().read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError):
        return _normalize(None)


def save_machine_selection(selection):
    selection = _normalize(selection)
    if build_variant() == 'cpu' and selection['mode'] != 'cpu':
        raise RuntimeError('Bản CPU không chứa CUDA; hãy dùng AniBgRmv GPU Edition.')
    destination = machine_config_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, destination)
    return selection
