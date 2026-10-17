from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QSizePolicy
from ..utils import pil_to_pixmap
from .background_swatches import BackgroundSwatches, BACKGROUND_COLORS

class EditMaskCanvas(QFrame):
    strokeStarted = Signal()
    strokePoint = Signal(float, float)
    strokeFinished = Signal(list)
    backgroundChanged = Signal(str)

    def __init__(self):
        super().__init__()
        # M22: lower minimum height so bottom edit buttons do not get pushed
        # outside the window until the user maximizes it.
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.BlankCursor)
        self.setStyleSheet(
            """
            QFrame {
                border: 2px solid #111111;
                border-radius: 18px;
                background: #FFFFFF;
            }
            """
        )

        self.image_rgb: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.operation = "erase"
        self.brush_size = 28
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_drawing = False
        self.is_panning = False
        self.last_pan_pos = None
        self.stroke_points: list[tuple[float, float]] = []
        self.processing = False
        self._background_mode = "transparent"
        # A floating, clickable row, anchored at the same top-left offset as
        # the normal preview's swatches. Always visible while Edit is open.
        self.background_swatches = BackgroundSwatches(self)
        self.background_swatches.move(18, 14)
        self.background_swatches.adjustSize()
        self.background_swatches.raise_()
        self.background_swatches.backgroundChanged.connect(self.set_background_mode)

        # M22 performance cache: do not rebuild PIL/QPixmap on every paint event.
        self._preview_cache: QPixmap | None = None
        self._hover_pos = None
        self._hover_inside = False

    def invalidate_preview_cache(self):
        self._preview_cache = None

    def set_image_alpha(self, image_path: str, alpha: np.ndarray):
        image = Image.open(image_path).convert("RGB")
        self.image_rgb = np.asarray(image)
        self.alpha = np.asarray(alpha).astype(np.uint8).copy()
        if self.alpha.shape[:2] != self.image_rgb.shape[:2]:
            self.alpha = cv2.resize(
                self.alpha,
                (self.image_rgb.shape[1], self.image_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        self.stroke_points = []
        self.invalidate_preview_cache()
        self.update()

    def set_alpha_array(self, alpha: np.ndarray, copy: bool = False):
        if self.image_rgb is None:
            return
        arr = np.asarray(alpha).astype(np.uint8)
        if arr.shape[:2] != self.image_rgb.shape[:2]:
            arr = cv2.resize(
                arr,
                (self.image_rgb.shape[1], self.image_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        self.alpha = arr.copy() if copy else arr
        self.invalidate_preview_cache()
        self.update()

    def set_background_mode(self, mode: str):
        if mode == self._background_mode:
            return
        self.background_swatches.set_mode(mode)
        self._background_mode = mode
        self.invalidate_preview_cache()
        self.update()
        self.backgroundChanged.emit(mode)

    def set_operation(self, operation: str):
        self.operation = operation
        self.invalidate_preview_cache()
        self.update()

    def set_brush_size(self, brush_size: int):
        self.brush_size = max(1, int(brush_size))
        self.update()

    def set_processing(self, processing: bool):
        self.processing = processing
        self.update()

    def clear_stroke(self):
        self.stroke_points = []
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _base_scale(self) -> float:
        if self.image_rgb is None:
            return 1.0
        h, w = self.image_rgb.shape[:2]
        available_w = max(1, self.width() - 32)
        available_h = max(1, self.height() - 32)
        return min(available_w / w, available_h / h)

    def _image_rect(self):
        if self.image_rgb is None:
            return QRectF()
        h, w = self.image_rgb.shape[:2]
        scale = self._base_scale() * self.zoom
        draw_w = w * scale
        draw_h = h * scale
        x = (self.width() - draw_w) / 2.0 + self.pan_x
        y = (self.height() - draw_h) / 2.0 + self.pan_y
        return QRectF(x, y, draw_w, draw_h)

    def _widget_to_image(self, x: float, y: float):
        if self.image_rgb is None:
            return None
        rect = self._image_rect()
        scale = self._base_scale() * self.zoom
        if scale <= 0:
            return None
        ix = (x - rect.left()) / scale
        iy = (y - rect.top()) / scale
        h, w = self.image_rgb.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return float(ix), float(iy)

    def _image_to_widget(self, x: float, y: float):
        rect = self._image_rect()
        scale = self._base_scale() * self.zoom
        return rect.left() + x * scale, rect.top() + y * scale

    def _render_preview(self) -> Image.Image:
        assert self.image_rgb is not None and self.alpha is not None
        rgb = self.image_rgb.astype(np.float32)
        a = (self.alpha.astype(np.float32) / 255.0)[..., None]

        if self._background_mode == "transparent":
            # Checkerboard represents transparent PNG, without changing alpha.
            h, w = self.alpha.shape[:2]
            rows = np.arange(h)[:, None] // 24
            cols = np.arange(w)[None, :] // 24
            gray = np.where((rows + cols) % 2 == 0, 220, 255).astype(np.float32)
            bg = gray[..., None]  # broadcast grayscale to 3 RGB channels
        else:
            bg = np.array(BACKGROUND_COLORS[self._background_mode], dtype=np.float32)

        # Restore needs a faint view of the original, including deleted pixels,
        # so users can paint missing parts back in. In Erase the selected backing
        # is exact and shows the actual cutout (no white/fake alpha pixels).
        ghost = 0.48 if self.operation == "restore" else 0.0
        visible = a + (1.0 - a) * ghost
        out = rgb * visible + bg * (1.0 - visible)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")

    def _preview_pixmap(self) -> QPixmap:
        if self._preview_cache is None:
            pil = self._render_preview()
            self._preview_cache = pil_to_pixmap(pil)
        return self._preview_cache

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))

        if self.image_rgb is None or self.alpha is None:
            painter.setPen(QColor("#333333"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Không có ảnh để edit")
            return

        rect = self._image_rect()
        pix = self._preview_pixmap()
        painter.drawPixmap(rect, pix, QRectF(pix.rect()))

        if self.stroke_points:
            scale = self._base_scale() * self.zoom
            pen = QPen(QColor(255, 0, 0, 210))
            pen.setWidth(max(1, int(self.brush_size * scale)))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)

            if len(self.stroke_points) == 1:
                x, y = self._image_to_widget(*self.stroke_points[0])
                painter.drawPoint(int(x), int(y))
            else:
                for p1, p2 in zip(self.stroke_points[:-1], self.stroke_points[1:]):
                    x1, y1 = self._image_to_widget(*p1)
                    x2, y2 = self._image_to_widget(*p2)
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Red translucent brush cursor.
        if self._hover_inside and self._hover_pos is not None and not self.processing:
            point = self._widget_to_image(self._hover_pos.x(), self._hover_pos.y())
            if point is not None:
                scale = self._base_scale() * self.zoom
                r = max(2.0, (self.brush_size * scale) / 2.0)
                painter.setPen(QPen(QColor(255, 0, 0, 170), 2))
                painter.setBrush(QColor(255, 0, 0, 45))
                painter.drawEllipse(QRectF(self._hover_pos.x() - r, self._hover_pos.y() - r, r * 2, r * 2))

        mode_text = "ERASE" if self.operation == "erase" else "RESTORE"
        painter.setPen(QColor("black"))
        painter.drawText(18, 75, f"{mode_text} | zoom {self.zoom:.2f}x | brush {self.brush_size}px")

        if self.processing:
            painter.fillRect(self.rect(), QColor(255, 255, 255, 70))
            painter.setPen(QColor("black"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Đang xử lý SAM...")

    def enterEvent(self, event):
        self._hover_inside = True
        self.setCursor(Qt.BlankCursor)
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover_inside = False
        self._hover_pos = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._hover_pos = event.position()
        if self.image_rgb is None or self.alpha is None or self.processing:
            return

        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier):
            self.is_panning = True
            self.last_pan_pos = event.position()
            return

        if event.button() == Qt.LeftButton:
            point = self._widget_to_image(event.position().x(), event.position().y())
            if point is None:
                return
            self.is_drawing = True
            self.stroke_points = [point]
            self.strokeStarted.emit()
            self.strokePoint.emit(point[0], point[1])
            self.update()

    def mouseMoveEvent(self, event):
        self._hover_pos = event.position()

        if self.is_panning and self.last_pan_pos is not None:
            delta = event.position() - self.last_pan_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.last_pan_pos = event.position()
            self.update()
            return

        if not self.is_drawing or self.processing:
            self.update()
            return

        point = self._widget_to_image(event.position().x(), event.position().y())
        if point is None:
            self.update()
            return

        if not self.stroke_points:
            self.stroke_points.append(point)
        else:
            px, py = self.stroke_points[-1]
            if abs(point[0] - px) + abs(point[1] - py) >= 2:
                self.stroke_points.append(point)
                self.strokePoint.emit(point[0], point[1])
        self.update()

    def mouseReleaseEvent(self, event):
        self._hover_pos = event.position()
        if self.is_panning:
            self.is_panning = False
            self.last_pan_pos = None
            return

        if self.is_drawing and event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.strokeFinished.emit(list(self.stroke_points))
            self.update()

    def wheelEvent(self, event):
        if self.image_rgb is None:
            return

        cursor_pos = event.position()
        before = self._widget_to_image(cursor_pos.x(), cursor_pos.y())

        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom = min(8.0, self.zoom * 1.15)
        else:
            self.zoom = max(0.25, self.zoom / 1.15)

        # M23: zoom around cursor, not around image center.
        if before is not None:
            rect = self._image_rect()
            scale = self._base_scale() * self.zoom
            target_x = rect.left() + before[0] * scale
            target_y = rect.top() + before[1] * scale
            self.pan_x += cursor_pos.x() - target_x
            self.pan_y += cursor_pos.y() - target_y

        self.update()
