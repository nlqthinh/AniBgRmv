"""Every model file the app uses, in one table.

Code asks for a file by role, e.g. ``model_path('hqsam_encoder')``; it never
reads a model path from app_config.json itself. The EXE build (AniBGRmv.spec)
bundles every ``bundled`` role, and the Settings page writes these same keys.

To add a model: add one ModelSpec below and its default path in
app_config.DEFAULT_APP_CONFIG, then use model_path('<role>') where it runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    label: str              # shown in error messages
    group: str              # section of app_config.json
    key: str                # key inside that section
    bundled: bool = True    # ONNX shipped inside the EXE; False = PyTorch checkpoint for the video worker
    inherits: str = ''      # empty value in the config -> use this role's file


MODELS = {
    'unet': ModelSpec('UNet ONNX', 'model', 'unet_onnx_path'),
    'vitmatte_unet': ModelSpec('ViTMatte-UNet ONNX', 'model', 'vitmatte_unet_onnx_path'),
    'batch_unet': ModelSpec('UNet ONNX (batch)', 'batch', 'unet_onnx_path', inherits='unet'),
    'batch_vitmatte_unet': ModelSpec('ViTMatte-UNet ONNX (batch)', 'batch', 'vitmatte_unet_onnx_path',
                                     inherits='vitmatte_unet'),
    'hqsam_encoder': ModelSpec('HQ-SAM encoder ONNX', 'sam', 'hqsam_encoder_onnx_path'),
    'hqsam_decoder': ModelSpec('HQ-SAM decoder ONNX', 'sam', 'hqsam_decoder_onnx_path'),
    'vitmatte_sam': ModelSpec('ViTMatte-SAM ONNX', 'sam', 'vitmatte_sam_onnx_path'),
    'video_vitmatte_sam': ModelSpec('ViTMatte-SAM ONNX (video)', 'video', 'vitmatte_sam_onnx_path',
                                    inherits='vitmatte_sam'),
    'hqsam_base_checkpoint': ModelSpec('HQ-SAM2 base checkpoint', 'sam', 'hqsam_base_checkpoint_path',
                                       bundled=False),
    'hqsam_decoder_checkpoint': ModelSpec('HQ-SAM2 decoder checkpoint', 'sam',
                                          'hqsam_decoder_checkpoint_path', bundled=False),
}


def model_value(role: str, config: dict) -> str:
    """Path as written in the config (may be relative); '' if it is not set."""
    spec = MODELS[role]
    value = (config.get(spec.group) or {}).get(spec.key) or ''
    return value or (model_value(spec.inherits, config) if spec.inherits else '')


def model_path(role: str) -> Path:
    """Absolute path of a model file of the current configuration."""
    from .app_config import APP_CONFIG, resolve_app_path
    return resolve_app_path(model_value(role, APP_CONFIG))


def guided_model_assets() -> tuple[tuple[str, Path], ...]:
    """Files required by HQ-SAM Guided Edit, shared by the engine and GPU Check."""
    from .app_config import APP_CONFIG
    from .hqsam_manifest import manifest_path_for
    encoder = model_path('hqsam_encoder')
    assets = [(MODELS['hqsam_encoder'].label, encoder),
              (MODELS['hqsam_decoder'].label, model_path('hqsam_decoder')),
              ('HQ-SAM export manifest', manifest_path_for(encoder))]
    if APP_CONFIG['model_settings']['image_sam_vit_enabled']:
        assets.append((MODELS['vitmatte_sam'].label, model_path('vitmatte_sam')))
    return tuple(assets)
