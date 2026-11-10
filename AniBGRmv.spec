# -*- mode: python ; coding: utf-8 -*-
"""AniBgRmv release build (PyInstaller onedir).

ANIBGRMV_BUILD_VARIANT=cpu : CPU edition — UNet + ViTMatte-UNet + HQ-SAM ONNX.
ANIBGRMV_BUILD_VARIANT=gpu : NVIDIA edition — adds HQ-SAM + ViTMatte-SAM ONNX
                             and a verified CUDA/cuDNN runtime. No PyTorch inside;
                             video Track uses an external Python (see README).
No checkpoints are downloaded and no model is modified here.
"""
import json
import os
import sys
import sysconfig
from pathlib import Path

# SPECPATH is supplied by PyInstaller; batch builders already cd to ROOT.
ROOT = Path(globals().get("SPECPATH") or Path.cwd()).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from ani_bgrmv.hqsam_manifest import check_hqsam_manifest, manifest_path_for
from ani_bgrmv.model_registry import MODELS, model_value
from ani_bgrmv.release_models import MODEL_LIST, write_model_list

SOURCE_CONFIG = ROOT / 'ani_bgrmv' / 'app_config.json'
if not SOURCE_CONFIG.is_file():
    raise RuntimeError(f'Config not found: {SOURCE_CONFIG}')
config = json.loads(SOURCE_CONFIG.read_text(encoding='utf-8'))
VARIANT = os.environ.get('ANIBGRMV_BUILD_VARIANT', 'gpu').strip().lower()
if VARIANT not in ('gpu', 'cpu'):
    raise ValueError('ANIBGRMV_BUILD_VARIANT must be cpu or gpu')
EXE_NAME = 'AniBGRmv_CPU' if VARIANT == 'cpu' else 'AniBGRmv'

# Never publish the build machine's GPU or output paths: the user picks a GPU
# on their PC (stored in LOCALAPPDATA) and starts in CPU mode.
seed = json.loads(json.dumps(config))
seed['compute'] = {'mode': 'cpu', 'gpu_id': None, 'gpu_name': None, 'cpu_threads': 2}
seed.setdefault('sam', {}).pop('cpu_guided_backend', None)
seed.setdefault('output', {}).update({
    'single_output_dir': 'output/single_saved',
    'batch_output_dir': 'output/batch_saved',
    'save_debug_outputs': False,
    'show_debug_button': False,
})
if VARIANT == 'cpu':
    seed.setdefault('batch', {})['max_batch_images'] = min(
        8, int(seed.get('batch', {}).get('max_batch_images', 8)))
META_DIR = ROOT / '.anibgrmv_build_seed' / VARIANT
META_DIR.mkdir(parents=True, exist_ok=True)
SAFE_CONFIG = META_DIR / 'app_config.json'
SAFE_CONFIG.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
FLAVOR_FILE = META_DIR / 'build_flavor.json'
FLAVOR_FILE.write_text(json.dumps({'variant': VARIANT}), encoding='utf-8')

# Every ONNX role of model_registry, including separate Batch/Video choices.
selected = {}
for role, spec in MODELS.items():
    if not spec.bundled:
        continue  # PyTorch checkpoints stay outside the EXE (video worker)
    raw = model_value(role, config)
    if not raw:
        raise RuntimeError(f'Missing {spec.group}.{spec.key} in {SOURCE_CONFIG}')
    rel = Path(raw)
    if rel.is_absolute() or rel.drive or '..' in rel.parts or rel.suffix.lower() != '.onnx':
        raise RuntimeError(f'{spec.group}.{spec.key} must be a safe relative .onnx path: {raw}')
    src = next((p for p in (ROOT / 'ani_bgrmv' / rel, ROOT / rel) if p.is_file()), None)
    if src is None:
        raise RuntimeError(f'Model missing ({spec.group}.{spec.key}): {raw}')
    selected[rel.as_posix()] = src

model_assets = dict(selected)
# Validate and bundle guided ONNX on both build flavors.
encoder_rel = Path(model_value('hqsam_encoder', config))
decoder_rel = Path(model_value('hqsam_decoder', config))
if encoder_rel.parent != decoder_rel.parent:
    raise RuntimeError('HQ-SAM encoder/decoder must share a model directory')
encoder_src, decoder_src = selected[encoder_rel.as_posix()], selected[decoder_rel.as_posix()]
try:
    manifest = check_hqsam_manifest(encoder_src, decoder_src)
except ValueError as exc:
    raise RuntimeError(str(exc)) from exc
manifest_src = manifest_path_for(encoder_src)
model_assets[str(encoder_rel.with_name(manifest_src.name))] = manifest_src
# The exporter lists every external tensor file of both graphs.
for filename in manifest.get('sidecars', {}):
    if not filename.startswith((encoder_src.name + '.', decoder_src.name + '.')):
        raise RuntimeError(f'HQ-SAM manifest sidecar does not belong to its graphs: {filename!r}')
    model_assets[str(encoder_rel.with_name(filename))] = manifest_src.parent / filename

for relative, src in sorted(selected.items()):
    # Exports of UNet/ViTMatte may produce one '<model>.onnx.data' sidecar.
    external = Path(str(src) + '.data')
    if external.is_file():
        model_assets[str(Path(relative).with_name(external.name))] = external

VIDEO_SIDECAR_FILES = ('video_track_worker.py', 'video_matting_worker.py',
                       'video_diagnostics.py', 'matting_ops.py')

# The packaged app checks these files on start; package_release.py puts them
# in the separate Models archive of the GPU release.
MODEL_LIST_FILE = write_model_list(META_DIR / MODEL_LIST, model_assets)

# Bundle the canonical seed config, not the developer's app_config.json.
datas = [(str(SAFE_CONFIG), '.'), (str(FLAVOR_FILE), '.'), (str(MODEL_LIST_FILE), '.')]
for relative, src in sorted(model_assets.items()):
    datas.append((str(src), str(Path(relative).parent)))
# Video sidecars run as plain .py files in the external PyTorch environment;
# matting_ops.py is the shared preprocessing imported by the matting sidecar.
for name in VIDEO_SIDECAR_FILES:
    datas.append((str(ROOT / 'ani_bgrmv' / 'engine' / name), 'ani_bgrmv/engine'))

from cuda_bundle_smoke import verified_selection
from cuda_bundle_optimizer import prune_automatic_cuda_copies
from python_native_bundle import find_base_python_native_dlls, write_native_manifest
import onnxruntime as ort

# A Miniconda-based venv needs the base Python's native stdlib DLLs (_ssl,
# _ctypes, _lzma...). Bundle only those, never all of Conda's Library/bin.
python_native_dlls = find_base_python_native_dlls(Path(sys.base_prefix))
native_manifest = write_native_manifest(META_DIR / 'python_native_manifest.json', python_native_dlls)
datas.append((str(native_manifest), '.'))
binaries = [(str(source), '.') for source in python_native_dlls.values()]
print('Base Python native DLLs:', ', '.join(python_native_dlls) or '(none)')

provider_dir = Path(ort.__file__).resolve().parent / 'capi'
cuda_libs = {}
cudnn_family = None
if VARIANT == 'gpu':
    site_packages = Path(sysconfig.get_paths()['purelib'])
    provider = provider_dir / 'onnxruntime_providers_cuda.dll'
    cuda_libs, _excluded, cudnn_family = verified_selection(ROOT, site_packages, provider)
    print('Verified CUDA/cuDNN bundle:', cudnn_family)
    binaries += [(str(src), 'gpu_runtime') for src in cuda_libs.values()]
    # The CUDA EP/bridge must be visible BEFORE Analysis so their non-CUDA
    # native dependencies are collected too.
    for filename in ('onnxruntime_providers_cuda.dll', 'onnxruntime_providers_shared.dll'):
        provider_file = provider_dir / filename
        if filename == 'onnxruntime_providers_cuda.dll' and not provider_file.is_file():
            raise RuntimeError(f'ONNX Runtime GPU provider missing: {provider_file}')
        if provider_file.is_file():
            binaries.append((str(provider_file), 'onnxruntime/capi'))
print('Variant:', VARIANT, '| seed config:', SAFE_CONFIG)
for rel, src in selected.items():
    print('  Model:', rel, '->', src)

analysis = Analysis(
    [str(ROOT / 'app.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=([str(ROOT / 'tools' / 'pyi_rth_cuda_bundle.py')] if VARIANT == 'gpu' else []),
    # The app runs every model on ONNX Runtime: no training/export/test stack.
    excludes=['torch', 'torchvision', 'torchaudio', 'sam2', 'segment_anything', 'onnx',
              'segmentation_models_pytorch', 'timm', 'albumentations', 'transformers',
              'pandas', 'matplotlib', 'scipy', 'sympy', 'pytest', '_pytest'],
    noarchive=False,
)
if VARIANT == 'gpu':
    # PyInstaller may still auto-collect another cuDNN copy (e.g. torch/lib);
    # ship ONLY the provider-verified gpu_runtime selection.
    _kept, _removed = prune_automatic_cuda_copies(
        analysis.binaries, cuda_libs, allow_shadowed_torch_cuda=True,
        allow_shadowed_nvidia_cudnn=(cudnn_family == 'torch'))
else:
    # CPU edition: drop every CUDA-dependent native DLL and any torch binary.
    _kept, _removed = [], []
    for entry in analysis.binaries:
        dest = str(entry[0]).replace('\\', '/').lower()
        leaf = dest.rsplit('/', 1)[-1]
        if (dest.startswith('torch/') or leaf == 'onnxruntime_providers_cuda.dll'
                or leaf.startswith(('cudnn', 'cublas', 'cudart', 'nvrtc', 'nvjitlink',
                                    'cufft', 'curand', 'cusparse', 'cusolver'))):
            _removed.append(entry)
        else:
            _kept.append(entry)
analysis.binaries[:] = _kept

if VARIANT == 'gpu':
    # Hook behavior varies across PyInstaller versions: the CUDA provider and
    # its bridge must stay inside onnxruntime/capi, not only gpu_runtime.
    names = {str(x[0]).replace('\\', '/').lower() for x in analysis.binaries}
    for filename in ('onnxruntime_providers_cuda.dll', 'onnxruntime_providers_shared.dll'):
        src = provider_dir / filename
        if src.is_file() and ('onnxruntime/capi/' + filename).lower() not in names:
            analysis.binaries.append(('onnxruntime/capi/' + filename, str(src), 'BINARY'))
    # No TensorRT EP: its DLL would only produce missing nvinfer warnings.
    for group in (analysis.binaries, analysis.datas):
        group[:] = [entry for entry in group
                    if not str(entry[0]).lower().endswith('onnxruntime_providers_tensorrt.dll')]
else:
    # The CUDA provider may be collected as BINARY or DATA; remove both.
    analysis.datas[:] = [entry for entry in analysis.datas
                         if not str(entry[0]).lower().endswith('onnxruntime_providers_cuda.dll')]
    if any(str(x[0]).lower().endswith('onnxruntime_providers_cuda.dll')
           for x in analysis.binaries + analysis.datas):
        raise RuntimeError('CPU release still contains ONNX CUDA provider')
print('Auto-collected binaries removed:', len(_removed), '| variant:', VARIANT)

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [],
    exclude_binaries=True,
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Set ANIBGRMV_BUILD_CONSOLE=1 for first-release diagnosis, default hidden.
    console=os.environ.get('ANIBGRMV_BUILD_CONSOLE', '0') == '1',
    contents_directory='.',
)
coll = COLLECT(
    exe, analysis.binaries, analysis.datas,
    strip=False, upx=False,
    name=EXE_NAME,
)
