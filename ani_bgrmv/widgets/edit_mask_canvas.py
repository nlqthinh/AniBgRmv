from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QSizePolicy
from ..utils import pil_to_pixmap
from .background_swatches import BackgroundSwatches, BACKGROUND_COLORS

# M37 — chỉnh độ mờ của con trỏ ở đây (0 = trong suốt, 255 = đậm hoàn toàn).
# Màu: ERASE_CURSOR_RGB / RESTORE_CURSOR_RGB là (đỏ, xanh lá).
ERASE_CURSOR_RGB = (255, 0, 0)
RESTORE_CURSOR_RGB = (0, 220, 75)
BRUSH_CURSOR_OUTLINE_ALPHA = 75  # trước đây 170; nay mờ hơn một nửa
BRUSH_CURSOR_FILL_ALPHA = 18     # trước đây 45; nay mờ hơn một nửa
GUIDED_STROKE_ALPHA = 124       # M37.2: nét màu Guided/SAM nhạt hơn ~25% so với 165 (0..255)

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
        self._loaded_image_path: str | None = None
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
        self._stroke_preview_enabled = True  # Guided: preview đường đi; Manual: xem alpha cập nhật thật

    def invalidate_preview_cache(self):
        self._preview_cache = None

    def clear_image(self):
        """Release large RGB, alpha and QPixmap buffers after Edit ends."""
        self.image_rgb = None
        self._loaded_image_path = None
        self.alpha = None
        self._preview_cache = None
        self.stroke_points = []
        self._hover_inside = False
        self._hover_pos = None
        self.update()

    def set_image_alpha(self, image_path: str, alpha: np.ndarray):
        if self.image_rgb is None or self._loaded_image_path != str(image_path):
            with Image.open(image_path) as source:
                self.image_rgb = np.asarray(source.convert('RGB'))
            self._loaded_image_path = str(image_path)
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

    def set_alpha_reference(self, alpha: np.ndarray):
        """Share the writable prediction-alpha during Manual stroke; keep the preview cache.

        Don't invalidate the full preview for every mouse movement.  The controller
        calls refresh_alpha_region() after changing only the brushed rectangle.
        """
        if self.image_rgb is not None:
            self.alpha = alpha

    def set_stroke_preview_enabled(self, enabled: bool):
        self._stroke_preview_enabled = bool(enabled)
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

    def _render_preview_region(self, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
        """Memory-efficient alpha composite, preserving M37.2 backgrounds.

        Previously rgb.astype(float32) and multiple RGB-sized temporaries
        could briefly allocate hundreds of MiB on a 4K/8K image. Work per
        channel; manual strokes continue updating only the brushed rectangle.
        """
        assert self.image_rgb is not None and self.alpha is not None
        alpha = self.alpha[y0:y1, x0:x1].astype(np.float32)
        ghost = 0.48 if self.operation == 'restore' else 0.0
        visible = alpha * ((1.0 - ghost) / 255.0)
        visible += ghost
        invisible = 1.0 - visible

        if self._background_mode == 'transparent':
            rows = np.arange(y0, y1)[:, None] // 24
            cols = np.arange(x0, x1)[None, :] // 24
            checker = np.where((rows + cols) % 2 == 0, 220.0, 255.0).astype(np.float32)
            bg_multiplier = checker * invisible
            background = None
        else:
            background = BACKGROUND_COLORS[self._background_mode]
            bg_multiplier = None

        out = np.empty((y1-y0, x1-x0, 3), dtype=np.uint8)
        for channel in range(3):
            color = self.image_rgb[y0:y1, x0:x1, channel].astype(np.float32)
            color *= visible
            if bg_multiplier is None:
                color += float(background[channel]) * invisible
            else:
                color += bg_multiplier
            out[...,channel] = np.clip(color, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode='RGB')

    def _render_preview(self) -> Image.Image:
        assert self.image_rgb is not None
        h, w = self.image_rgb.shape[:2]
        return self._render_preview_region(0, 0, w, h)

    def refresh_alpha_region(self, x0: int, y0: int, x1: int, y1: int):
        """Update just the brushed ROI of the *cached* pixmap (no full-frame PIL conversion).

        Bounding coordinates are inclusive. The alpha itself was modified in-place
        by the controller; this method only changes the visible on-screen preview.
        """
        if self.alpha is None or self.image_rgb is None:
            return
        h, w = self.alpha.shape[:2]
        left, top = max(0, int(x0)), max(0, int(y0))
        right, bottom = min(w, int(x1) + 1), min(h, int(y1) + 1)
        if right <= left or bottom <= top:
            return
        if self._preview_cache is not None:
            patch = pil_to_pixmap(self._render_preview_region(left, top, right, bottom))
            painter = QPainter(self._preview_cache)
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.drawPixmap(left, top, patch)
            painter.end()
        self.update()

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

        if self.stroke_points and self._stroke_preview_enabled:
            scale = self._base_scale() * self.zoom
            red, green, blue = ERASE_CURSOR_RGB if self.operation == "erase" else RESTORE_CURSOR_RGB
            pen = QPen(QColor(red, green, blue, GUIDED_STROKE_ALPHA))
            pen.setWidth(max(1, int(self.brush_size * scale)))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)

            if len(self.stroke_points) == 1:
                x, y = self._image_to_widget(*self.stroke_points[0])
                painter.drawPoint(int(x), int(y))
            else:
                # M37.2: draw the whole stroke in ONE path. Painting separate
                # segments re-composited their translucent caps/joins, making
                # fast strokes noticeably darker where consecutive segments meet.
                x0, y0 = self._image_to_widget(*self.stroke_points[0])
                stroke_path = QPainterPath()
                stroke_path.moveTo(x0, y0)
                for point in self.stroke_points[1:]:
                    wx, wy = self._image_to_widget(*point)
                    stroke_path.lineTo(wx, wy)
                painter.drawPath(stroke_path)

        # Brush cursor is translucent and follows the selected operation.
        if self._hover_inside and self._hover_pos is not None and not self.processing:
            point = self._widget_to_image(self._hover_pos.x(), self._hover_pos.y())
            if point is not None:
                scale = self._base_scale() * self.zoom
                r = max(2.0, (self.brush_size * scale) / 2.0)
                red, green, blue = ERASE_CURSOR_RGB if self.operation == "erase" else RESTORE_CURSOR_RGB
                painter.setPen(QPen(QColor(red, green, blue, BRUSH_CURSOR_OUTLINE_ALPHA), 2))
                painter.setBrush(QColor(red, green, blue, BRUSH_CURSOR_FILL_ALPHA))
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
