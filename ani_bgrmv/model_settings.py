"""Settings page model choices: read, validate and save them in app_config.json.

A pair represents the two *different* assets needed by HQ-SAM ONNX or video
HQ-SAM2. Reject incompatible imports before changing the app configuration.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import app_config as config
from .engine import matting_ops as ops
from .model_registry import MODELS, model_value

MODEL_KEYS = (
    'single_auto', 'single_vit', 'single_sam', 'single_sam_vit',
    'batch_auto', 'batch_vit',
    'video_backend', 'video_sam', 'video_sam_vit', 'video_export_background',
    'video_track_keyframes', 'video_held_frames',
)
# Settings choice -> model_registry role(s) whose file it selects.
CHOICE_ROLES = {
    'single_auto': 'unet', 'single_vit': 'vitmatte_unet',
    'single_sam': ('hqsam_encoder', 'hqsam_decoder'), 'single_sam_vit': 'vitmatte_sam',
    'batch_auto': 'batch_unet', 'batch_vit': 'batch_vitmatte_unet',
    'video_sam': ('hqsam_base_checkpoint', 'hqsam_decoder_checkpoint'),
}


def stored_model(choice: str, data: dict | None = None):
    """File(s) of a Settings choice in the config, kept even while it is OFF."""
    data = config.APP_CONFIG if data is None else data
    roles = CHOICE_ROLES[choice]
    if isinstance(roles, tuple):
        return tuple(model_value(role, data) for role in roles)
    return model_value(roles, data)


def _models(data: dict) -> dict:
    switches = data['model_settings']
    batch_vit = switches.get('batch_vit_enabled', switches['image_vit_enabled'])
    enabled = {'single_vit': switches['image_vit_enabled'], 'batch_vit': batch_vit,
               'single_sam_vit': switches['image_sam_vit_enabled']}
    models = {choice: stored_model(choice, data) if enabled.get(choice, True) else None
              for choice in CHOICE_ROLES}
    models.update(video_backend=switches['video_backend'],
                  video_sam_vit=(data['video']['matting_export'], model_value('video_vitmatte_sam', data)),
                  video_export_background=data['video']['export_background'],
                  video_track_keyframes=data['video']['track_keyframes_in_memory'],
                  video_held_frames=data['video']['reuse_held_frames'])
    return models


def current_models() -> dict:
    return _models(config.APP_CONFIG)


def default_models() -> dict:
    """Show an explicit revert-to-default option even after a custom import."""
    return _models(config.DEFAULT_APP_CONFIG)


def _check_onnx(path: str, kind: str) -> None:
    source = config.resolve_app_path(path)
    if source.suffix.lower() != '.onnx' or not source.is_file():
        raise ValueError(f'{kind}: cần file .onnx tồn tại: {source}')
    try:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(source), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        inputs, outputs = session.get_inputs(), session.get_outputs()
    except Exception as exc:
        raise ValueError(f'{kind}: không mở được graph ONNX {source}: {exc}') from exc
    expected_in = ops.model_input_shape(ops.UNET_CHANNELS if kind == 'UNet' else ops.MATTING_CHANNELS)
    if (len(inputs) != 1 or inputs[0].type != 'tensor(float)'
            or tuple(inputs[0].shape) != expected_in
            or len(outputs) != 1 or outputs[0].type != 'tensor(float)'
            or tuple(outputs[0].shape) != ops.ALPHA_OUTPUT_SHAPE):
        raise ValueError(f'{kind}: graph ONNX sai input/output; cần float32 '
                         f'{list(expected_in)} → {list(ops.ALPHA_OUTPUT_SHAPE)}.')


def _check_image_sam(pair) -> None:
    from .disk_cache import cached_sha256
    from .hqsam_manifest import check_hqsam_manifest
    encoder, decoder = (config.resolve_app_path(x) for x in pair)
    check_hqsam_manifest(encoder, decoder, digest=cached_sha256)
    # The manifest proves the exported encoder/decoder pair was validated
    # together. Also check the required graph inputs/outputs.
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for file, inputs, outputs in (
        (encoder, {'image'}, {'high_res_feats_0', 'high_res_feats_1', 'image_embed'}),
        (decoder, {'high_res_feats_0', 'high_res_feats_1', 'image_embed',
                   'point_coords', 'point_labels'}, {'masks', 'iou_predictions'}),
    ):
        session = ort.InferenceSession(str(file), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        if not inputs <= {i.name for i in session.get_inputs()} or not outputs <= {o.name for o in session.get_outputs()}:
            raise ValueError(f'HQ-SAM ONNX sai graph: {file}')


_PT_INSPECT = r'''
import json, sys, torch
base = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
decoder = torch.load(sys.argv[2], map_location='cpu', weights_only=True)
if not isinstance(base, dict) or not isinstance(base.get('model'), dict):
    raise ValueError('Base HQ-SAM2 phải chứa model state_dict')
if (not isinstance(decoder, dict)
    or decoder.get('format') != 'HQ2_HQSAM2_decoder_state_v2'
    or not isinstance(decoder.get('decoder'), dict)):
    raise ValueError('Decoder phải có format HQ2_HQSAM2_decoder_state_v2')
print('CHECKPOINTS_OK')
'''


def _check_video_sam(pair) -> None:
    base, decoder = (config.resolve_app_path(x) for x in pair)
    if any(p.suffix.lower() not in ('.pt', '.pth') or not p.is_file() for p in (base, decoder)):
        raise ValueError('Video HQ-SAM2 cần đủ 2 checkpoint .pt/.pth (base + decoder).')
    runner = config.APP_CONFIG.get('video', {}).get('tracker_python_path', '').strip()
    python = Path(runner).expanduser() if runner else Path(sys.executable)
    if getattr(sys, 'frozen', False) and not runner:
        raise ValueError('Hãy cấu hình video.tracker_python_path để kiểm tra checkpoint PyTorch.')
    try:
        result = subprocess.run([str(python), '-c', _PT_INSPECT, str(base), str(decoder)],
                                capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f'Không thể kiểm tra checkpoint PyTorch: {exc}') from exc
    if result.returncode or 'CHECKPOINTS_OK' not in result.stdout:
        raise ValueError('Checkpoint video không tương thích hoặc không đọc được với PyTorch: '
                         + (result.stderr.strip()[-600:] or result.stdout.strip()[-600:]))


def validate_import(role: str, value) -> None:
    """Imported ONNXs are checked against the *actual* inference contract."""
    if role in ('single_auto', 'batch_auto', 'single_vit', 'batch_vit', 'single_sam_vit', 'video_sam_vit'):
        _check_onnx(value, 'UNet' if role in ('single_auto', 'batch_auto') else 'ViTMatte')
    elif role == 'single_sam':
        _check_image_sam(value)
    elif role == 'video_sam':
        _check_video_sam(value)
    else:
        raise ValueError(f'Chưa hỗ trợ import backend {role!r}.')


def validate_choice(models: dict) -> None:
    if set(models) != set(MODEL_KEYS):
        raise ValueError('Bộ thiết lập model thiếu hoặc thừa trường.')
    for name in ('single_auto', 'batch_auto'):
        if not isinstance(models[name], str) or not models[name]:
            raise ValueError(f'{name} bắt buộc có UNet ONNX.')
    if models['video_backend'] != 'hqsam2':
        raise ValueError('Hiện tại chỉ hỗ trợ HQ-SAM2 cho video; EfficientTAM chưa được tích hợp.')
    for name in ('single_sam', 'video_sam'):
        value = models[name]
        if not isinstance(value, (tuple, list)) or len(value) != 2 or not all(isinstance(x, str) and x for x in value):
            raise ValueError(f'{name} yêu cầu cặp encoder/decoder hoặc base/decoder.')
    for name in ('single_vit', 'batch_vit', 'single_sam_vit'):
        if models[name] is not None and not isinstance(models[name], str):
            raise ValueError(f'{name} phải là đường dẫn ONNX hoặc OFF.')
    mode, path = models['video_sam_vit']
    if mode not in ('auto', 'binary', 'vitmatte_sam') or not isinstance(path, str):
        raise ValueError('Video ViTMatte-SAM phải là Auto / OFF / ON.')
    if models['video_export_background'] not in config.ALLOWED_VALUES[('video', 'export_background')]:
        raise ValueError('Xuất video chỉ hỗ trợ nền xanh lá, nền xanh biển hoặc PNG trong suốt.')
    if models['video_track_keyframes'] not in config.ALLOWED_VALUES[('video', 'track_keyframes_in_memory')]:
        raise ValueError('Track: số keyframe nhớ lại chỉ nhận Tất cả, 8, 4 hoặc 2.')
    if not isinstance(models['video_held_frames'], bool):
        raise ValueError('Frame giữ hình phải là bật/tắt.')


def _to_config(models: dict, original: dict) -> dict:
    """Return an updated JSON tree, preserving all unrelated user fields."""
    validate_choice(models)
    updated = json.loads(json.dumps(original))
    current = config.deep_merge_config(config.DEFAULT_APP_CONFIG, original)

    def write(role: str, value: str) -> None:
        spec = MODELS[role]
        updated.setdefault(spec.group, {})[spec.key] = value

    for choice, roles in CHOICE_ROLES.items():
        value = models[choice]
        if isinstance(roles, tuple):
            for role, path in zip(roles, value):
                write(role, path)
        else:
            # OFF keeps the stored file, so switching ON again restores it.
            write(roles, value or model_value(roles, current))
    video = updated.setdefault('video', {})
    video['matting_export'], video_matte = models['video_sam_vit']
    write('video_vitmatte_sam', video_matte)
    video['export_background'] = models['video_export_background']
    video['track_keyframes_in_memory'] = models['video_track_keyframes']
    video['reuse_held_frames'] = models['video_held_frames']
    updated.setdefault('sam', {}).pop('cpu_guided_backend', None)  # unused legacy key
    updated.setdefault('model_settings', {}).update(
        batch_vit_enabled=models['batch_vit'] is not None,
        image_vit_enabled=models['single_vit'] is not None,
        image_sam_vit_enabled=models['single_sam_vit'] is not None,
        video_backend='hqsam2')
    return updated


def save_models(models: dict) -> None:
    """Atomic save; the in-memory config follows, so the next model load uses it."""
    path = config.get_external_config_path()
    original = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(original, dict):
        raise ValueError('app_config.json phải là JSON object.')
    updated = _to_config(models, original)
    config.validate_config(updated, path)
    tmp = path.with_name(path.name + '.model-settings.tmp')
    try:
        tmp.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    config.APP_CONFIG.clear()
    config.APP_CONFIG.update(config.deep_merge_config(config.DEFAULT_APP_CONFIG, updated))
