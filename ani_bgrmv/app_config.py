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
        "vitmatte_onnx_path": "models/m7_vitmatte_letterbox_best.onnx",
    },
    "sam": {
        # HQ2 is the default for new source builds. Explicitly set
        # "sam_vit_b_onnx" to roll back without changing the Edit UI.
        "backend": "hq3_torch",
        "hq2_base_checkpoint_path": "models/sam2.1_hq_hiera_large.pt",
        "hq2_decoder_checkpoint_path": "models/hq2_best_decoder.pt",
        "hq3_decoder_checkpoint_path": "models/hq3_best_overall.pt",
        "hq3_encoder_onnx_path": "models/hq3_hqsam2_encoder.onnx",
        "hq3_decoder_onnx_path": "models/hq3_hqsam2_decoder.onnx",
        "m71_onnx_path": "models/m71_hq3_d0d_best.onnx",
        "hq2_refine_with_m7": True,
        "hq2_small_object_crop": True,  # toggle extra HQ2 crop pass if VRAM is tight
        # Optional: avoid reloading ~900MB HQ2 when reopening Edit; consumes
        # substantial GPU/CPU memory between Edit sessions. Off by default.
        "hq2_keep_model_loaded": False,
        # GPU 4GB: unload F3/M7 during HQ3 Guided; restore on next Single/Batch cut.
        "hq3_unload_f3_m7_during_guided": True,
        # Print per-stage GPU/torch VRAM snapshots. Set False after benchmarking.
        "sam_vram_diagnostics": True,
        # GPU 4GB memory handoff: release PyTorch's UNUSED cached blocks
        # before the first/next M71 inference; model weights/features survive.
        "hq3_trim_torch_before_m71": True,
        "hq3_trim_torch_min_reclaim_mib": 256,
        # ONNX Runtime RunOptions: return idle M71 CUDA arena regions after Run.
        # May increase latency; toggle off for speed A/B comparison.
        "m71_shrink_gpu_arena_after_run": True,
        "encoder_onnx_path": "models/sam_vit_b_encoder.onnx",
        "decoder_onnx_path": "models/sam_vit_b_decoder.onnx",
        "model_type": "vit_b",
        "device": "auto",
        "guided_timeout_ms": 0,
        "cpu_guided_backend": "grabcut",  # nhanh, chất lượng khác SAM; dùng SAM nếu đổi thành "sam"
    },
    "output": {
        "single_output_dir": "output/single_saved",
        "batch_output_dir": "output/batch_saved",
        "video_output_dir": "output/video_saved",
        "save_debug_outputs": True,
        "show_debug_button": True,
    },
    "video": {
        # Video temporal tracking runs in a separate installed PyTorch Python.
        # Source mode uses sys.executable; GPU EXE needs an absolute path here.
        # A standalone ONNX-only EXE cannot provide SAM2 temporal memory.
        "tracker_python_path": "",
        "matting_export": "auto",  # GPU -> M71 ONNX, CPU -> binary; options auto/m71/binary
    },
    "batch": {
        "max_batch_images": 30,
    },
    "compute": {"mode": "cpu", "gpu_id": None, "gpu_name": None, "cpu_threads": 2},
}


# Source mode: config nằm cạnh app.py.
# PyInstaller exe mode: config nằm cạnh AniBGRmv.exe.
def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_BASE_DIR = get_app_base_dir()


def deep_merge_config(default: dict, user: dict) -> dict:
    """
    Merge config để user có thể thiếu vài field mà app vẫn có default an toàn.
    """
    merged = dict(default)
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_external_config_path() -> Path:
    """
    Config người dùng sửa sau khi build.

    Source mode:
        project/app_config.json

    Exe mode:
        dist/AniBGRmv/app_config.json, nằm cạnh AniBGRmv.exe
    """
    return APP_BASE_DIR / CONFIG_FILE_NAME


def get_bundled_config_path() -> Path:
    """
    Config được PyInstaller bundle vào _internal/_MEIPASS.
    Chỉ dùng làm seed lần đầu nếu chưa có external config cạnh exe.
    """
    return Path(getattr(sys, "_MEIPASS", APP_BASE_DIR)) / CONFIG_FILE_NAME


def get_config_path() -> Path:
    # Hàm này dùng để hiển thị path config thật sự user nên sửa.
    return get_external_config_path()


def write_default_config(config_path: Path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(DEFAULT_APP_CONFIG, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_external_config_exists() -> Path:
    """
    Quy tắc:
    1. Nếu đã có app_config.json cạnh exe/app.py -> dùng file đó.
    2. Nếu chưa có, nhưng có config được bundle -> copy bundle ra cạnh exe/app.py.
    3. Nếu cả hai đều không có -> tạo default config.

    Lý do:
    - Người dùng cần sửa config sau khi build mà không rebuild.
    - PyInstaller onedir có thể đặt data trong _internal, không phải cạnh exe.
    """
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

    return deep_merge_config(DEFAULT_APP_CONFIG, user_config)


def resolve_app_path(path_value: str) -> Path:
    """
    Đường dẫn tuyệt đối: dùng nguyên.
    Đường dẫn tương đối: hiểu là nằm dưới APP_BASE_DIR.
    """
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return APP_BASE_DIR / p


APP_CONFIG = load_app_config()

F2_A1_ONNX_PATH = str(resolve_app_path(APP_CONFIG["model"]["unet_onnx_path"]))
M6_ONNX_PATH = str(resolve_app_path(APP_CONFIG["model"]["vitmatte_onnx_path"]))

MAX_BATCH_IMAGES = int(APP_CONFIG["batch"]["max_batch_images"])

SINGLE_OUTPUT_DIR = str(resolve_app_path(APP_CONFIG["output"]["single_output_dir"]))
BATCH_OUTPUT_DIR = str(resolve_app_path(APP_CONFIG["output"]["batch_output_dir"]))

# Development: show the Debug toggle and initially check it.
# Release: set output.show_debug_button=false and
# output.save_debug_outputs=false in ani_bgrmv/app_config.json.
SAVE_DEBUG_OUTPUTS = bool(APP_CONFIG["output"]["save_debug_outputs"])
SHOW_DEBUG_BUTTON = bool(APP_CONFIG["output"]["show_debug_button"])

SAM_BACKEND = str(APP_CONFIG["sam"].get("backend", "hq2_torch")).lower()
if SAM_BACKEND not in ("hq2_torch", "hq3_torch", "hq3_onnx", "sam_vit_b_onnx"):
    raise ValueError(f"sam.backend không được hỗ trợ: {SAM_BACKEND}")
SAM_ENCODER_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["encoder_onnx_path"]))
SAM_DECODER_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["decoder_onnx_path"]))
HQ2_BASE_CHECKPOINT_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hq2_base_checkpoint_path"]))
HQ2_BEST_DECODER_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hq2_decoder_checkpoint_path"]))
HQ3_BEST_DECODER_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hq3_decoder_checkpoint_path"]))
M71_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["m71_onnx_path"]))
HQ3_ENCODER_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hq3_encoder_onnx_path"]))
HQ3_DECODER_ONNX_PATH = str(resolve_app_path(APP_CONFIG["sam"]["hq3_decoder_onnx_path"]))
# Existing GuidedEditWorker argument name is preserved. For HQ2 it now points
# to the *full public HQ-SAM2 base*, never to the decoder-only BEST weights.
SAM_CHECKPOINT_PATH = (HQ2_BASE_CHECKPOINT_PATH if SAM_BACKEND in ("hq2_torch", "hq3_torch")
                       else HQ3_ENCODER_ONNX_PATH if SAM_BACKEND == "hq3_onnx"
                       else SAM_ENCODER_ONNX_PATH)
SAM_MODEL_TYPE = ("hq3_large" if SAM_BACKEND in ("hq3_torch", "hq3_onnx")
                  else "hq2_large" if SAM_BACKEND == "hq2_torch"
                  else str(APP_CONFIG["sam"].get("model_type", "vit_b")))
SAM_DEVICE = str(APP_CONFIG["sam"].get("device", "auto"))
GUIDED_EDIT_TIMEOUT_MS = int(APP_CONFIG.get("sam", {}).get("guided_timeout_ms", 5000))


def sam_runtime_assets() -> tuple[tuple[str, str], ...]:
    """Model-specific required files, shared by engine and GPU verification."""
    if SAM_BACKEND == "hq3_onnx":
        return (("HQ3 HQ-SAM2 encoder ONNX", HQ3_ENCODER_ONNX_PATH),
                ("HQ3 HQ-SAM2 decoder ONNX", HQ3_DECODER_ONNX_PATH),
                ("HQ3 ONNX verified manifest", str(Path(HQ3_ENCODER_ONNX_PATH).with_suffix('.manifest.json'))),
                ("M71 ViTMatte ONNX (SAM only)", M71_ONNX_PATH))
    if SAM_BACKEND == "hq3_torch":
        return (("HQ-SAM2 Large base", HQ2_BASE_CHECKPOINT_PATH),
                ("HQ3 BEST decoder", HQ3_BEST_DECODER_PATH),
                ("M71 ViTMatte ONNX (SAM only)", M71_ONNX_PATH))
    if SAM_BACKEND == "hq2_torch":
        return (("HQ-SAM2 Large base", HQ2_BASE_CHECKPOINT_PATH),
                ("HQ2 BEST decoder", HQ2_BEST_DECODER_PATH))
    return (("SAM ViT-B encoder ONNX", SAM_ENCODER_ONNX_PATH),
            ("SAM ViT-B decoder ONNX", SAM_DECODER_ONNX_PATH))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
