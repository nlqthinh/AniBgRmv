# -*- mode: python ; coding: utf-8 -*-
"""AniBgRmv release build: CPU GrabCut or NVIDIA CUDA HQ3 ONNX.

Preserves optional legacy torch backends; ONNX-only GPU needs no PyTorch.
Do not add development checkpoints or download external dependencies.
"""
import json
import os
import sys
import sysconfig
from pathlib import Path

# SPECPATH is supplied by PyInstaller; batch builders already cd to ROOT.
# Using the spec directory also supports calling PyInstaller from elsewhere.
ROOT = Path(globals().get("SPECPATH") or Path.cwd()).resolve()
SOURCE_CONFIG = ROOT / 'ani_bgrmv' / 'app_config.json'
if not SOURCE_CONFIG.is_file():
    raise RuntimeError(f'Config not found: {SOURCE_CONFIG}')
config = json.loads(SOURCE_CONFIG.read_text(encoding='utf-8'))
config.setdefault('sam', {})
config['sam'].setdefault('backend', 'hq3_torch')
config['sam'].setdefault('hq2_base_checkpoint_path', 'models/sam2.1_hq_hiera_large.pt')
config['sam'].setdefault('hq2_decoder_checkpoint_path', 'models/hq2_best_decoder.pt')
config['sam'].setdefault('hq3_decoder_checkpoint_path', 'models/hq3_best_overall.pt')
config['sam'].setdefault('m71_onnx_path', 'models/m71_hq3_d0d_best.onnx')
config['sam'].setdefault('hq3_encoder_onnx_path', 'models/hq3_hqsam2_encoder.onnx')
config['sam'].setdefault('hq3_decoder_onnx_path', 'models/hq3_hqsam2_decoder.onnx')
config['sam'].setdefault('hq2_refine_with_m7', True)
SAM_BACKEND = str(config['sam']['backend']).lower()
if SAM_BACKEND not in ('hq2_torch', 'hq3_torch', 'hq3_onnx', 'sam_vit_b_onnx'):
    raise ValueError(f'Unsupported sam.backend: {SAM_BACKEND}')
VARIANT = os.environ.get('ANIBGRMV_BUILD_VARIANT', 'gpu').strip().lower()
if VARIANT not in ('gpu', 'cpu'):
    raise ValueError('ANIBGRMV_BUILD_VARIANT must be cpu or gpu')
EXE_NAME = 'AniBGRmv_CPU' if VARIANT == 'cpu' else 'AniBGRmv'
# Never publish the developer PC's saved GPU name or output paths. The
# customer chooses their own GPU and stores it in LOCALAPPDATA (M46).
seed = json.loads(json.dumps(config))
seed['compute'] = {'mode': 'cpu', 'gpu_id': None, 'gpu_name': None,
                   'cpu_threads': 2}
seed.setdefault('sam', {})['cpu_guided_backend'] = 'grabcut'
seed.setdefault('output', {}).update({
    'single_output_dir': 'output/single_saved',
    'batch_output_dir': 'output/batch_saved',
    'save_debug_outputs': False,
    'show_debug_button': False,
})
if VARIANT == 'cpu':
    seed.setdefault('batch', {})['max_batch_images'] = min(
        8, int(seed.get('batch', {}).get('max_batch_images', 8)))
    # Lightweight CPU edition uses GrabCut instead of SAM ViT-B; no SAM ONNX
    # checkpoints are required. F3 and M7 retain their original accuracy.
META_DIR = ROOT / '.anibgrmv_build_seed' / VARIANT
META_DIR.mkdir(parents=True, exist_ok=True)
SAFE_CONFIG = META_DIR / 'app_config.json'
SAFE_CONFIG.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
FLAVOR_FILE = META_DIR / 'build_flavor.json'
FLAVOR_FILE.write_text(json.dumps({'variant': VARIANT}), encoding='utf-8')

required = [
    ('model', 'unet_onnx_path', '.onnx'),
    ('model', 'vitmatte_onnx_path', '.onnx'),
]
if VARIANT == 'gpu':
    if SAM_BACKEND == 'hq3_onnx':
        required.extend([
            ('sam', 'hq3_encoder_onnx_path', '.onnx'),
            ('sam', 'hq3_decoder_onnx_path', '.onnx'),
            ('sam', 'm71_onnx_path', '.onnx'),
        ])
    elif SAM_BACKEND == 'hq3_torch':
        required.extend([
            ('sam', 'hq2_base_checkpoint_path', '.pt'),
            ('sam', 'hq3_decoder_checkpoint_path', '.pt'),
            ('sam', 'm71_onnx_path', '.onnx'),
        ])
    elif SAM_BACKEND == 'hq2_torch':
        required.extend([
            ('sam', 'hq2_base_checkpoint_path', '.pt'),
            ('sam', 'hq2_decoder_checkpoint_path', '.pt'),
        ])
    else:
        required.extend([
            ('sam', 'encoder_onnx_path', '.onnx'),
            ('sam', 'decoder_onnx_path', '.onnx'),
        ])
selected = {}
for group, key, extension in required:
    raw = config.get(group, {}).get(key)
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f'Missing {group}.{key} in {SOURCE_CONFIG}')
    rel = Path(raw)
    if rel.is_absolute() or rel.drive or '..' in rel.parts or rel.suffix.lower() != extension:
        raise RuntimeError(f'{group}.{key} must be a safe relative {extension} path: {raw}')
    src = next((p for p in (ROOT / 'ani_bgrmv' / rel, ROOT / rel) if p.is_file()), None)
    if src is None:
        raise RuntimeError(f'Model missing ({group}.{key}): {raw}')
    selected[rel.as_posix()] = src

# Bundle the canonical config, not the outdated root app_config.json.
datas = [(str(SAFE_CONFIG), '.'), (str(FLAVOR_FILE), '.')]
# M84: GPU EXE's VIDEO tracker is a standalone Python sidecar. Keep its source
# visible to the user-configured external PyTorch Python. Do not bundle torch
# into HQ3 ONNX GPU EXE: massive size and risk to proven cuDNN 9.10.2 runtime.
if VARIANT == 'gpu':
    datas.append((str(ROOT / 'ani_bgrmv/engine/video_hq3_worker.py'), 'ani_bgrmv/engine'))
    datas.append((str(ROOT / 'ani_bgrmv/engine/video_diagnostics.py'), 'ani_bgrmv/engine'))
    datas.append((str(ROOT / 'ani_bgrmv/engine/video_m71_worker.py'), 'ani_bgrmv/engine'))

model_assets = dict(selected)
if VARIANT == 'gpu' and SAM_BACKEND == 'hq3_onnx':
    encoder_rel = Path(config['sam']['hq3_encoder_onnx_path']).as_posix()
    decoder_rel = Path(config['sam']['hq3_decoder_onnx_path']).as_posix()
    encoder_src, decoder_src = selected[encoder_rel], selected[decoder_rel]
    if Path(encoder_rel).parent != Path(decoder_rel).parent:
        raise RuntimeError('HQ3 encoder/decoder must share a model directory')
    manifest = encoder_src.with_suffix('.manifest.json')
    if not manifest.is_file():
        raise RuntimeError(f'Missing HQ3 ONNX verified manifest: {manifest}')
    info = json.loads(manifest.read_text(encoding='utf-8'))
    if (info.get('verified') is not True
        or info.get('format') != 'anibgrmv-hq3-sam2-point-onnx-v1'
        or info.get('encoder_filename') != encoder_src.name
        or info.get('decoder_filename') != decoder_src.name):
        raise RuntimeError(f'HQ3 ONNX manifest missing verification or graph names incorrect: {manifest}')
    for attr, src in (('encoder_size', encoder_src), ('decoder_size', decoder_src)):
        if info.get(attr) != src.stat().st_size:
            raise RuntimeError(f'HQ3 ONNX manifest model size mismatch: {src}')
    model_assets[str(Path(encoder_rel).with_name(manifest.name))] = manifest
    # The exporter includes every external tensor sidecar in this manifest.
    # Do not assume the only possible ONNX filename is '<graph>.onnx.data'.
    for filename, metadata in info.get('sidecars', {}).items():
        if (not isinstance(filename, str) or Path(filename).name != filename
            or not filename.startswith((encoder_src.name + '.', decoder_src.name + '.'))
            or not isinstance(metadata, dict)):
            raise RuntimeError(f'HQ3 ONNX manifest has invalid sidecar entry: {filename!r}')
        sidecar = manifest.parent / filename
        if not sidecar.is_file() or sidecar.stat().st_size != metadata.get('size'):
            raise RuntimeError(f'HQ3 ONNX external tensor is missing/incorrect: {sidecar}')
        model_assets[str(Path(encoder_rel).with_name(filename))] = sidecar

for relative, src in sorted(selected.items()):
    if src.suffix.lower() == '.onnx':
        # Legacy/F3/M7/M71 exports usually produce one .onnx.data sidecar.
        # HQ3 sidecars above are sourced from its verified export manifest.
        external = Path(str(src) + '.data')
        if external.is_file():
            model_assets[str(Path(relative).with_name(external.name))] = external

for relative, src in sorted(model_assets.items()):
    datas.append((str(src), str(Path(relative).parent)))

# HQ-SAM2 uses Hydra YAML package resources. These are not model files.
hiddenimports = []
if VARIANT == 'gpu' and SAM_BACKEND in ('hq2_torch', 'hq3_torch'):
    from PyInstaller.utils.hooks import collect_data_files, collect_submodules
    datas += collect_data_files('sam2', includes=['configs/**/*.yaml', 'configs/**/*.yml'])
    hiddenimports += collect_submodules('sam2')

# M45: Select runtime DLLs via verified PE-import dependency closure.
# All cuDNN plugins are retained because cuDNN can load them dynamically.
# No model is modified or quantized.
sys.path.insert(0, str(ROOT / 'tools'))
from cuda_bundle_optimizer import select_cuda_libs, prune_automatic_cuda_copies
from cuda_bundle_smoke import verified_selection
from python_native_bundle import find_base_python_native_dlls, write_native_manifest
import onnxruntime as ort

# M80: Miniconda base stdlib extensions (_ssl, _ctypes, _lzma, etc.)
# need DLLs from base/Library/bin. When pandas/scipy are excluded there is
# no longer an incidental hook adding that directory to the binder's search
# path. Include only their dependencies, not the whole Conda Library/bin,
# to avoid shadowing the deliberately selected CUDA/cuDNN family.
python_native_dlls = find_base_python_native_dlls(Path(sys.base_prefix))
manifest_path = write_native_manifest(
    META_DIR / 'python_native_manifest.json', python_native_dlls
)
datas.append((str(manifest_path), '.'))
binaries = [(str(source), '.') for source in python_native_dlls.values()]
print('[M80] Conda base stdlib native DLLs:', ', '.join(python_native_dlls) or '(none)')
excluded_cuda_libs = {}
if VARIANT == 'gpu':
    site_packages = Path(sysconfig.get_paths()['purelib'])
    provider = Path(ort.__file__).resolve().parent / 'capi' / 'onnxruntime_providers_cuda.dll'
    if SAM_BACKEND in ('hq3_onnx', 'sam_vit_b_onnx'):
        cuda_libs, excluded_cuda_libs, cudnn_family = verified_selection(
            ROOT, site_packages, provider
        )
        print('[M77] Verified CUDA/cuDNN bundle:', cudnn_family)
    else:
        cuda_libs, excluded_cuda_libs = select_cuda_libs(site_packages, provider)
        cudnn_family = 'nvidia'
    binaries += [(str(src), 'gpu_runtime') for src in cuda_libs.values()]
    # PyInstaller must see the CUDA EP/bridge BEFORE Analysis, so the PE
    # dependency scanner can include their non-CUDA native dependencies.
    provider_dir = Path(ort.__file__).resolve().parent / 'capi'
    for filename in ('onnxruntime_providers_cuda.dll', 'onnxruntime_providers_shared.dll'):
        provider_file = provider_dir / filename
        if filename == 'onnxruntime_providers_cuda.dll' and not provider_file.is_file():
            raise RuntimeError(f'ONNX Runtime GPU provider missing: {provider_file}')
        if provider_file.is_file():
            binaries.append((str(provider_file), 'onnxruntime/capi'))
    print('M75 GPU: CUDA bundle + ONNX native providers:', len(binaries))
else:
    cuda_libs = {}
    print('CPU Lite: no CUDA/cuDNN DLLs or SAM/HQ2 checkpoints')
print('M46 variant:', VARIANT)
print('M46 portable config:', SAFE_CONFIG)
for rel, src in selected.items():
    print('  Model:', rel, '->', src)

analysis = Analysis(
    [str(ROOT / 'app.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=([str(ROOT / "tools" / "pyi_rth_cuda_bundle.py")] if VARIANT == "gpu" else []),
    excludes=([
        'torch', 'torchvision', 'torchaudio', 'sam2', 'segment_anything',
        'onnx', 'segmentation_models_pytorch', 'timm', 'albumentations', 'transformers',
        'pandas', 'matplotlib', 'scipy', 'sympy',
    ] if VARIANT == 'cpu' or SAM_BACKEND not in ('hq2_torch', 'hq3_torch') else [
        'segment_anything', 'onnx', 'segmentation_models_pytorch',
        'albumentations', 'transformers',
    ]),
    noarchive=False,
)
# Keep GPU Edition M45 exact DLL-selection behavior. For the CPU Edition,
# eliminate *every* CUDA-dependent native DLL and any torch binary. Build
# flavors have different output directories to avoid stale DLLs leaking in.
if VARIANT == 'gpu' and SAM_BACKEND in ('sam_vit_b_onnx', 'hq3_onnx'):
    # This edition excludes torch from Python imports. PyInstaller may still
    # auto-collect a different cuDNN build from torch/lib; drop that unused
    # copy and ship ONLY the provider-verified gpu_runtime selection.
    # All non-torch conflicts remain strict errors (no mixed CUDA distribution).
    _kept, _removed = prune_automatic_cuda_copies(
        analysis.binaries, cuda_libs, allow_shadowed_torch_cuda=True,
        allow_shadowed_nvidia_cudnn=(cudnn_family == 'torch')
    )
elif VARIANT == 'gpu':
    # IMPORTANT: the ONNX-only M45 pruner drops torch/c10 binaries. HQ2 needs
    # those DLLs. Keep the PyInstaller torch collection intact and audit the
    # resulting Windows build on a clean machine before publishing.
    _kept, _removed = list(analysis.binaries), []
else:
    _kept = []
    _removed = []
    for entry in analysis.binaries:
        dest, source, kind = entry
        leaf = str(dest).replace('\\', '/').lower().rsplit('/', 1)[-1]
        if (str(dest).replace('\\','/').lower().startswith('torch/')
            or leaf == 'onnxruntime_providers_cuda.dll'
            or leaf.startswith(('cudnn', 'cublas', 'cudart', 'nvrtc', 'nvjitlink',
                                'cufft', 'curand', 'cusparse', 'cusolver'))):
            _removed.append(entry)
        else:
            _kept.append(entry)
analysis.binaries[:] = _kept
# PyInstaller hook behavior varies across versions. The CUDA provider and
# its bridge must stay inside onnxruntime/capi, never just gpu_runtime.
if VARIANT == 'gpu':
    provider_dir = Path(ort.__file__).resolve().parent / 'capi'
    for filename in ('onnxruntime_providers_cuda.dll', 'onnxruntime_providers_shared.dll'):
        src = provider_dir / filename
        if filename == 'onnxruntime_providers_cuda.dll' and not src.is_file():
            raise RuntimeError(f'Missing ONNX Runtime CUDA provider: {src}')
        if src.is_file():
            dst = 'onnxruntime/capi/' + filename
            binary_names = {str(x[0]).replace('\\', '/').lower() for x in analysis.binaries}
            if dst.lower() not in binary_names:
                analysis.binaries.append((dst, str(src), 'BINARY'))

# CUDA Edition intentionally does not use TensorRT EP. PyInstaller's ORT
# hook may bundle its DLL and trigger missing nvinfer_10.dll warnings.
# Remove that optional provider, but never drop the required CUDA EP.
if VARIANT == 'gpu' and SAM_BACKEND in ('hq3_onnx', 'sam_vit_b_onnx'):
    for group in (analysis.binaries, analysis.datas):
        group[:] = [entry for entry in group if
                    not str(entry[0]).lower().endswith('onnxruntime_providers_tensorrt.dll')]

# PyInstaller may collect the CUDA provider as BINARY or DATA. A CPU build
# must remove both types, then assert no stray CUDA provider remains.
if VARIANT == 'cpu':
    analysis.datas[:] = [entry for entry in analysis.datas
                         if not str(entry[0]).lower().endswith('onnxruntime_providers_cuda.dll')]
    if any(str(x[0]).lower().endswith('onnxruntime_providers_cuda.dll')
           for x in analysis.binaries + analysis.datas):
        raise RuntimeError('CPU release still contains ONNX CUDA provider')
print('M75: Auto binaries removed:', len(_removed), '| variant:', VARIANT)

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
