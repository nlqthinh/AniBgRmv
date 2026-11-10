"""What the image engine reads from app_config.json, read once when the engine is created.

The engine never looks at the global APP_CONFIG afterwards. A change made in
Settings or GPU Check takes effect through a new engine (main_window reloads
it), so a Guided session never mixes old and new values. Tests can build one
with dataclasses.replace(EngineSettings.from_config(...), ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

# model_registry roles the engine opens (Single, Batch, Guided Edit, CPU Matting).
ENGINE_MODELS = ('unet', 'vitmatte_unet', 'batch_unet', 'batch_vitmatte_unet',
                 'hqsam_encoder', 'hqsam_decoder', 'vitmatte_sam')


@dataclass(frozen=True)
class EngineSettings:
    models: Mapping[str, Path]          # role -> absolute model file
    image_vit: bool                     # model_settings.image_vit_enabled (Single)
    batch_vit: bool                     # model_settings.batch_vit_enabled (default: Single's)
    image_sam_vit: bool                 # model_settings.image_sam_vit_enabled
    foreground_estimation: bool         # output.foreground_estimation
    cpu_matting_model: str              # sam.cpu_matting_model
    small_object_crop: bool             # sam.small_object_crop (GPU Guided)
    refine_each_stroke: bool            # sam.refine_each_stroke (GPU Guided)
    keep_model_loaded: bool             # sam.keep_model_loaded
    unload_auto_models_during_guided: bool
    shrink_gpu_arena_after_run: bool
    log_timing: bool
    compute: Mapping                    # 'compute': cpu_threads, CUDA arena options

    @classmethod
    def from_config(cls, config: dict) -> EngineSettings:
        from ..app_config import resolve_app_path
        from ..model_registry import model_value
        switches, sam = config['model_settings'], config['sam']
        image_vit = bool(switches['image_vit_enabled'])
        return cls(
            models=MappingProxyType({role: resolve_app_path(model_value(role, config))
                                     for role in ENGINE_MODELS}),
            image_vit=image_vit,
            batch_vit=bool(switches.get('batch_vit_enabled', image_vit)),
            image_sam_vit=bool(switches['image_sam_vit_enabled']),
            foreground_estimation=bool(config['output']['foreground_estimation']),
            cpu_matting_model=str(sam['cpu_matting_model']).lower(),
            small_object_crop=bool(sam['small_object_crop']),
            refine_each_stroke=bool(sam['refine_each_stroke']),
            keep_model_loaded=bool(sam['keep_model_loaded']),
            unload_auto_models_during_guided=bool(sam['unload_auto_models_during_guided']),
            shrink_gpu_arena_after_run=bool(sam['shrink_gpu_arena_after_run']),
            log_timing=bool(sam['log_timing']),
            compute=MappingProxyType(dict(config['compute'])),
        )
