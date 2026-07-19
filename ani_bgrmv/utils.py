from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QPushButton

from .app_config import IMAGE_EXTS

def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def collect_image_files_from_paths(paths: list[str], recursive: bool = True) -> list[str]:
    """
    Input có thể là file hoặc folder do kéo-thả.
    - File ảnh: lấy trực tiếp.
    - Folder: lấy ảnh trong folder. Mặc định recursive cho batch.
    """
    found: list[str] = []

    for raw in paths:
        p = Path(raw)

        if p.is_file() and is_image_file(p):
            found.append(str(p))

        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(p.glob(pattern)):
                if child.is_file() and is_image_file(child):
                    found.append(str(child))

    seen = set()
    unique = []
    for p in found:
        key = str(Path(p).resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def pil_to_pixmap(pil_image):
    rgba = pil_image.convert("RGBA")
    w, h = rgba.size
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def center_button_row(*buttons: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    row.addStretch(1)
    for btn in buttons:
        row.addWidget(btn)
    row.addStretch(1)
    return row


# =========================
# WORKERS
# =========================
