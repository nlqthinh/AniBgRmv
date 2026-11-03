"""Application configuration.

The user-editable ``app_config.json`` lives next to ``app.py`` (source mode)
or ``AniBGRmv.exe`` (frozen). Missing fields fall back to DEFAULT_APP_CONFIG.

Model names describe architecture + role, not experiment IDs:
  UNet           automatic foreground segmentation
  ViTMatte-UNet  matting refinement for UNet masks (automatic cut)
  HQ-SAM         HQ-SAM2 Large, prompt (Paint/Box) segmentation for Guided Edit
  ViTMatte-SAM   matting refinement for HQ-SAM masks (Guided Edit, video export)
"""
import json
import shutil
import sys
from pathlib import Path

CONFIG_FILE_NAME = "app_config.json"

DEFAULT_APP_CONFIG = {
    "_note": (
        "Relative paths are resolved from the folder containing app.py "
        "or AniBGRmv.exe. On Windows, using forward slash paths is recommended, "
        "for example: D:/AniBGRmv/output/single_saved"
    ),
    "model": {
        "unet_onnx_path": "models/F3_A1_letterbox_constant_768_best.onnx",
        "vitmatte_unet_onnx_path": "models/m7_vitmatte_letterbox_best.onnx",
    },
    "sam": {
        "hqsam_encoder_onnx_path": "models/hq3_hqsam2_encoder.onnx",
        "hqsam_decoder_onnx_path": "models/hq3_hqsam2_decoder.onnx",
        "vitmatte_sam_onnx_path": "models/m71_hq3_d0d_best.onnx",
        # Video Track only (PyTorch sidecar): public base + fine-tuned decoder.
        "hqsam_base_checkpoint_path": "models/sam2.1_hq_hiera_large.pt",
        "hqsam_decoder_checkpoint_path": "models/hq3_best_overall.pt",
        "guided_timeout_ms": 0,
        # GPU only. CPU never runs these per stroke (see GuidedPolicy).
        "refine_each_stroke": True,
        "small_object_crop": True,
        # Keep HQ-SAM/ViTMatte-SAM loaded between Edit sessions (more memory).
        "keep_model_loaded": False,
        # GPU 4 GB: unload UNet/ViTMatte-UNet while Guided Edit runs.
        "unload_auto_models_during_guided": True,
        # ONNX Runtime returns idle CUDA arena memory after each run.
        "shrink_gpu_arena_after_run": True,
        "log_timing": False,
        # CPU Guided selector: "grabcut" (fast) or "hqsam" (best quality,
        # needs one HQ-SAM encoder pass per image).
        "cpu_guided_backend": "grabcut",
        # CPU "Matting" button model: "off", "vitmatte_unet" or "vitmatte_sam".
        "cpu_matting_model": "vitmatte_unet",
    },
    "output": {
        "single_output_dir": "output/single_saved",
        "batch_output_dir": "output/batch_saved",
        "video_output_dir": "output/video_saved",
        "save_debug_outputs": True,
        "show_debug_button": True,
    },
    "video": {
        # Video Track runs in a separate Python that has PyTorch + HQ-SAM2.
        # Source mode uses sys.executable; the GPU EXE needs an absolute path.
        "tracker_python_path": "",
        # "auto" = vitmatte_sam on GPU, binary on CPU; or "vitmatte_sam"/"binary".
        "matting_export": "auto",
    },
    "batch": {
        "max_batch_images": 30,
    },
    "compute": {"mode": "cpu", "gpu_id": None, "gpu_name": None, "cpu_threads": 2},
}

# Keys from older releases. A stale key is rejected with its new name rather
# than silently ignored (a custom model path would otherwise be lost).
LEGACY_KEYS = {
    ("model", "vitmatte_onnx_path"): "model.vitmatte_unet_onnx_path",
    ("sam", "hq3_encoder_onnx_path"): "sam.hqsam_encoder_onnx_path",
    ("sam", "hq3_decoder_onnx_path"): "sam.hqsam_decoder_onnx_path",
    ("sam", "m71_onnx_path"): "sam.vitmatte_sam_onnx_path",
    ("sam", "hq2_base_checkpoint_path"): "sam.hqsam_base_checkpoint_path",
    ("sam", "hq3_decoder_checkpoint_path"): "sam.hqsam_decoder_checkpoint_path",
    ("sam", "hq2_refine_with_m7"): "sam.refine_each_stroke",
    ("sam", "hq2_small_object_crop"): "sam.small_object_crop",
    ("sam", "hq2_keep_model_loaded"): "sam.keep_model_loaded",
    ("sam", "hq3_unload_f3_m7_during_guided"): "sam.unload_auto_models_during_guided",
    ("sam", "m71_shrink_gpu_arena_after_run"): "sam.shrink_gpu_arena_after_run",
    ("sam", "hq3_onnx_timing"): "sam.log_timing",
    ("sam", "cpu_grabcut_matting"): "sam.cpu_matting_model",
    ("sam", "backend"): None,
    ("sam", "model_type"): None,
    ("sam", "device"): None,
    ("sam", "encoder_onnx_path"): None,
    ("sam", "decoder_onnx_path"): None,
    ("sam", "hq2_decoder_checkpoint_path"): None,
    ("sam", "sam_vram_diagnostics"): None,
    ("sam", "hq3_trim_torch_before_m71"): None,
    ("sam", "hq3_trim_torch_min_reclaim_mib"): None,
    ("sam", "hq3_onnx_reuse_encoder_session"): None,
    ("sam", "hq3_onnx_encoder_shrink_gpu_arena_after_run"): None,
    ("sam", "hq3_onnx_cpu_threads"): None,
}

ALLOWED_VALUES = {
    ("sam", "cpu_guided_backend"): ("grabcut", "hqsam"),
    ("sam", "cpu_matting_model"): ("off", "vitmatte_unet", "vitmatte_sam"),
    ("video", "matting_export"): ("auto", "vitmatte_sam", "binary"),
}


# Source mode: config next to app_config.py; EXE mode: next to the EXE.
def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_BASE_DIR = get_app_base_dir()


def deep_merge_config(default: dict, user: dict) -> dict:
    """Merge so that a user file may omit fields and still get safe defaults."""
    merged = dict(default)
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_external_config_path() -> Path:
    """The config the user edits after building (next to the EXE/app)."""
    return APP_BASE_DIR / CONFIG_FILE_NAME


def get_bundled_config_path() -> Path:
    """PyInstaller-bundled seed, copied out only when no external config exists."""
    return Path(getattr(sys, "_MEIPASS", APP_BASE_DIR)) / CONFIG_FILE_NAME


def get_config_path() -> Path:
    return get_external_config_path()


def write_default_config(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(DEFAULT_APP_CONFIG, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_external_config_exists() -> Path:
    """Use the external config; else copy the bundled seed; else write defaults."""
    external_config = get_external_config_path()
    bundled_config = get_bundled_config_path()

    if external_config.exists():
        return external_config

    external_config.parent.mkdir(parents=True, exist_ok=True)

    if bundled_config.exists() and bundled_config.resolve() != external_config.resolve():
        shutil.copy2(bundled_config, external_config)
        return external_config

    write_default_config(external_config)
    return external_config


def validate_config(config: dict, source: Path) -> None:
    """Reject legacy keys and unknown option values with an actionable message."""
    problems = []
    for (group, key), new_name in LEGACY_KEYS.items():
        if key in (config.get(group) or {}):
            problems.append(f"{group}.{key} → đổi thành {new_name}" if new_name
                            else f"{group}.{key} → không còn dùng, hãy xóa")
    for (group, key), allowed in ALLOWED_VALUES.items():
        value = (config.get(group) or {}).get(key)
        if value is not None and value not in allowed:
            problems.append(f"{group}.{key}={value!r} → chỉ nhận {', '.join(allowed)}")
    if problems:
        raise ValueError(f"Cấu hình {source} cần cập nhật:\n- " + "\n- ".join(problems))


def load_app_config() -> dict:
    config_path = ensure_external_config_exists()

    try:
        user_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"File cấu hình {config_path} không phải JSON hợp lệ. "
            f"Hãy sửa JSON hoặc xóa file để app tự tạo lại. Chi tiết: {e}"
        ) from e

    if not isinstance(user_config, dict):
        raise ValueError(f"File cấu hình {config_path} phải là JSON object.")
    validate_config(user_config, config_path)
    return deep_merge_config(DEFAULT_APP_CONFIG, user_config)


def resolve_app_path(path_value: str) -> Path:
    """Absolute paths are kept; relative paths are under APP_BASE_DIR."""
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return APP_BASE_DIR / p


APP_CONFIG = load_app_config()

UNET_ONNX_PATH = str(resolve_app_path(APP_CONFIG["model"]["unet_onnx_path"]))
VITMATTE_UNET_ONNX_PATH = str(resolve_app_path(APP_CONFIG["model"]["vitmatte_unet_onnx_path"]))

HQSAM_ENCODER_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hqsam_encoder_onnx_path"]))
HQSAM_DECODER_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hqsam_decoder_onnx_path"]))
VITMATTE_SAM_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["vitmatte_sam_onnx_path"]))
HQSAM_BASE_CHECKPOINT_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hqsam_base_checkpoint_path"]))
HQSAM_DECODER_CHECKPOINT_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hqsam_decoder_checkpoint_path"]))
GUIDED_EDIT_TIMEOUT_MS = int(APP_CONFIG["sam"]["guided_timeout_ms"])

MAX_BATCH_IMAGES = int(APP_CONFIG["batch"]["max_batch_images"])

SINGLE_OUTPUT_DIR = str(resolve_app_path(APP_CONFIG["output"]["single_output_dir"]))
BATCH_OUTPUT_DIR = str(resolve_app_path(APP_CONFIG["output"]["batch_output_dir"]))

# Release builds set output.show_debug_button=false and save_debug_outputs=false.
SAVE_DEBUG_OUTPUTS = bool(APP_CONFIG["output"]["save_debug_outputs"])
SHOW_DEBUG_BUTTON = bool(APP_CONFIG["output"]["show_debug_button"])


def hqsam_manifest_path() -> Path:
    """The exporter writes '<encoder stem>.manifest.json' next to the encoder."""
    return Path(HQSAM_ENCODER_ONNX_PATH).with_suffix(".manifest.json")


def guided_model_assets() -> tuple[tuple[str, str], ...]:
    """Files required by HQ-SAM Guided Edit, shared by engine and GPU check."""
    return (("HQ-SAM encoder ONNX", HQSAM_ENCODER_ONNX_PATH),
            ("HQ-SAM decoder ONNX", HQSAM_DECODER_ONNX_PATH),
            ("HQ-SAM export manifest", str(hqsam_manifest_path())),
            ("ViTMatte-SAM ONNX", VITMATTE_SAM_ONNX_PATH))


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
