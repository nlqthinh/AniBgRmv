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
        "unet_onnx_path": "models/f2a1_unet_mit_b1_768.onnx",
        "vitmatte_onnx_path": "models/m6_vitmatte_m6_768.onnx",
    },
    "sam": {
        "checkpoint_path": "models/sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "device": "auto",
        "guided_timeout_ms": 0,
    },
    "output": {
        "single_output_dir": "output/single_saved",
        "batch_output_dir": "output/batch_saved",
        "save_debug_outputs": True,
    },
    "batch": {
        "max_batch_images": 30,
    },
    "compute": {"mode": "cpu", "gpu_id": None, "gpu_name": None},
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

SAVE_DEBUG_OUTPUTS = bool(APP_CONFIG["output"]["save_debug_outputs"])

SAM_CHECKPOINT_PATH = str(resolve_app_path(APP_CONFIG["sam"]["checkpoint_path"]))
SAM_MODEL_TYPE = str(APP_CONFIG["sam"].get("model_type", "vit_b"))
SAM_DEVICE = str(APP_CONFIG["sam"].get("device", "auto"))
GUIDED_EDIT_TIMEOUT_MS = int(APP_CONFIG.get("sam", {}).get("guided_timeout_ms", 5000))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
