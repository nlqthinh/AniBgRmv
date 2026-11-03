import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRectF, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSlider, QSizePolicy, QVBoxLayout, QWidget
from ..utils import pil_to_pixmap
from .background_swatches import BackgroundSwatches, BACKGROUND_COLORS, BACKGROUND_MODES

# Chỉnh độ mờ của con trỏ ở đây (0 = trong suốt, 255 = đậm hoàn toàn).
# Màu: ERASE_CURSOR_RGB / RESTORE_CURSOR_RGB là (đỏ, xanh lá).
ERASE_CURSOR_RGB = (255, 0, 0)
RESTORE_CURSOR_RGB = (0, 220, 75)
BRUSH_CURSOR_OUTLINE_ALPHA = 75  # trước đây 170; nay mờ hơn một nửa
BRUSH_CURSOR_FILL_ALPHA = 18     # trước đây 45; nay mờ hơn một nửa
GUIDED_STROKE_ALPHA = 124       # Nét màu Guided nhạt hơn ~25% so với 165 (0..255)
DEFAULT_BACKGROUND_OPACITY = 100  # 100% retains the existing background preview.

class EditMaskCanvas(QFrame):
    strokeStarted = Signal()
    strokePoint = Signal(float, float)
    strokeFinished = Signal(list)
    boxFinished = Signal(tuple)  # image-pixel xyxy, inclusive; only valid drags
    backgroundChanged = Signal(str)

    def __init__(self):
        super().__init__()
        # Lower minimum height so bottom edit buttons do not get pushed
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
        self.draw_mode = "paint"
        self._box_start = None
        self._box_end = None
        self.processing = False
        self.processing_percent = 0
        self.processing_stage = "Đang chuẩn bị Guided Edit"
        self._background_mode = "transparent"
        # Display-only setting: never affects final_alpha or exported PNG.
        self._background_opacities = {
            mode: DEFAULT_BACKGROUND_OPACITY for mode in BACKGROUND_MODES
        }
        # A floating, clickable row, anchored at the same top-left offset as
        # the normal preview's swatches. Always visible while Edit is open.
        self.background_swatches = BackgroundSwatches(self)
        self.background_swatches.move(18, 14)
        self.background_swatches.adjustSize()
        self.background_swatches.raise_()
        self.background_swatches.backgroundChanged.connect(self.set_background_mode)
        self.background_swatches.swatchClicked.connect(self._on_background_swatch_clicked)

        # A second click on the selected swatch toggles a compact overlay,
        # independently for checker/black/white/green. It stays inside the
        # canvas so the established page layout and brush controls do not move.
        self.background_opacity_panel = QWidget(self)
        self.background_opacity_panel.setObjectName("edit_background_opacity_panel")
        # The canvas hides the OS pointer while painting the red/green brush.
        # Child controls must explicitly restore it for precise slider dragging.
        self.background_opacity_panel.setCursor(Qt.ArrowCursor)
        self.background_opacity_panel.setStyleSheet(
            "QWidget#edit_background_opacity_panel {"
            "background-color: rgba(26, 29, 35, 235);"
            "border: 1px solid #777777; border-radius: 8px; }"
        )
        opacity_layout = QVBoxLayout(self.background_opacity_panel)
        opacity_layout.setContentsMargins(10, 7, 10, 9)
        opacity_layout.setSpacing(4)
        self.background_opacity_label = QLabel("Độ mờ nền: 100%")
        self.background_opacity_label.setCursor(Qt.ArrowCursor)
        self.background_opacity_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        self.background_opacity_slider = QSlider(Qt.Horizontal)
        self.background_opacity_slider.setCursor(Qt.ArrowCursor)
        self.background_opacity_slider.setRange(0, 100)
        self.background_opacity_slider.setValue(DEFAULT_BACKGROUND_OPACITY)
        self.background_opacity_slider.setToolTip(
            "0%: thấy ảnh gốc; 100%: nền preview như trước. Không đổi mask đã lưu."
        )
        self.background_opacity_slider.setStyleSheet(
            "QSlider::groove:horizontal {height: 7px; background: #D9D9D9; border-radius: 3px;}"
            "QSlider::handle:horizontal {width: 16px; margin: -5px 0;"
            "background: #FF3030; border-radius: 8px;}"
        )
        opacity_layout.addWidget(self.background_opacity_label)
        opacity_layout.addWidget(self.background_opacity_slider)
        self.background_opacity_panel.setFixedWidth(self.background_swatches.sizeHint().width())
        self.background_opacity_panel.adjustSize()
        self.background_opacity_panel.move(
            self.background_swatches.x(),
            self.background_swatches.y() + self.background_swatches.height() + 8,
        )
        self.background_opacity_panel.hide()
        self.background_opacity_slider.valueChanged.connect(self._on_background_opacity_changed)
        # Coalesce heavy image recomposition during fast slider dragging.
        self._background_opacity_timer = QTimer(self)
        self._background_opacity_timer.setSingleShot(True)
        self._background_opacity_timer.timeout.connect(self._refresh_background_opacity)
        self.background_opacity_slider.sliderReleased.connect(self._refresh_background_opacity)

        # Performance cache: do not rebuild PIL/QPixmap on every paint event.
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
        self._box_start = self._box_end = None
        self.is_drawing = False
        self._hover_inside = False
        self._hover_pos = None
        self._background_opacity_timer.stop()
        self.background_opacity_panel.hide()
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
        self.background_opacity_panel.hide()
        self._sync_background_opacity_slider()
        self.invalidate_preview_cache()
        self.update()
        self.backgroundChanged.emit(mode)

    def _sync_background_opacity_slider(self):
        opacity = self._background_opacities[self._background_mode]
        self.background_opacity_slider.blockSignals(True)
        self.background_opacity_slider.setValue(opacity)
        self.background_opacity_slider.blockSignals(False)
        self.background_opacity_label.setText(f"Độ mờ nền: {opacity}%")

    def _on_background_swatch_clicked(self, mode: str, was_selected: bool):
        if not was_selected:
            # Switching colors selects the new background; a second click opens
            # its own slider, avoiding accidental edits to a different mode.
            self.background_opacity_panel.hide()
            return
        if self.background_opacity_panel.isVisible():
            self.background_opacity_panel.hide()
        else:
            self._sync_background_opacity_slider()
            self.background_opacity_panel.show()
            self.background_opacity_panel.raise_()
        self.update()

    def _on_background_opacity_changed(self, value: int):
        self._background_opacities[self._background_mode] = int(value)
        self.background_opacity_label.setText(f"Độ mờ nền: {value}%")
        if not self._background_opacity_timer.isActive():
            self._background_opacity_timer.start(60)

    def _refresh_background_opacity(self):
        self._background_opacity_timer.stop()
        self.invalidate_preview_cache()
        self.update()

    def set_operation(self, operation: str):
        self.operation = operation
        self.invalidate_preview_cache()
        self.update()

    def set_draw_mode(self, mode: str):
        if mode not in ("paint", "box"):
            raise ValueError("Unknown edit drawing mode")
        if self.is_drawing or self.processing:
            return
        self.draw_mode = mode
        self.stroke_points = []
        self._box_start = self._box_end = None
        self.setCursor(Qt.CrossCursor if mode == "box" else (Qt.BlankCursor if self._hover_inside else Qt.ArrowCursor))
        self.update()

    def _clamped_image_point(self, x: float, y: float):
        """Keep a dragged rectangle inside the image even when the pointer leaves it."""
        if self.image_rgb is None:
            return None
        rect = self._image_rect()
        scale = self._base_scale() * self.zoom
        h, w = self.image_rgb.shape[:2]
        if scale <= 0:
            return None
        return (max(0.0, min(float(w-1), (x - rect.left()) / scale)),
                max(0.0, min(float(h-1), (y - rect.top()) / scale)))

    def set_brush_size(self, brush_size: int):
        self.brush_size = max(1, int(brush_size))
        self.update()

    def set_processing(self, processing: bool):
        self.processing = bool(processing)
        if processing:
            self.background_opacity_panel.hide()
            # Apply any pending slider change before pausing the preview.
            if self._background_opacity_timer.isActive():
                self._refresh_background_opacity()
            self.processing_percent = 0
            self.processing_stage = "Đang chuẩn bị Guided Edit"
        self.update()

    def set_processing_progress(self, percentage: int, stage: str):
        """Stage-based progress shared by Guided Paint and Guided Box.

        Only update from the main Qt thread via the worker's progress signal.
        The progress is an estimate based on stages, not precise GPU progress.
        """
        if not self.processing:
            return
        self.processing_percent = max(
            self.processing_percent, max(0, min(100, int(percentage))))
        self.processing_stage = str(stage)
        self.update()

    def clear_stroke(self):
        self.stroke_points = []
        self._box_start = self._box_end = None
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
        """Memory-efficient alpha composite, preserving the selected backgrounds.

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
        opacity = getattr(self, '_background_opacities', {}).get(
            self._background_mode, DEFAULT_BACKGROUND_OPACITY) / 100.0
        # Reuse the two existing H×W float buffers. At the 100% default this
        # executes the exact old compositing math with no extra allocation.
        if opacity != 1.0:
            visible += invisible * (1.0 - opacity)
            invisible *= opacity

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

        if self.draw_mode == "box" and self._box_start is not None and self._box_end is not None:
            x0, y0 = self._image_to_widget(*self._box_start)
            x1, y1 = self._image_to_widget(*self._box_end)
            color = ERASE_CURSOR_RGB if self.operation == "erase" else RESTORE_CURSOR_RGB
            painter.setPen(QPen(QColor(*color, 230), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(min(x0, x1), min(y0, y1), abs(x1-x0), abs(y1-y0)))

        if self.draw_mode == "paint" and self.stroke_points and self._stroke_preview_enabled:
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
                # Draw the whole stroke in ONE path. Painting separate
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
        if self.draw_mode == "paint" and self._hover_inside and self._hover_pos is not None and not self.processing:
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
        tool_text = (f"brush {self.brush_size}px" if self.draw_mode == "paint" else "BOX")
        status_y = 136 if self.background_opacity_panel.isVisible() else 75
        painter.drawText(18, status_y, f"{mode_text} | zoom {self.zoom:.2f}x | {tool_text}")

        if self.processing:
            self._paint_processing_progress(painter)

    def _paint_processing_progress(self, painter: QPainter):
        """Paint the original guided stage progress overlay on cached image."""
        painter.save()
        painter.fillRect(self.rect(), QColor(0, 0, 0, 65))
        card_width = min(430.0, max(200.0, float(self.width()) - 32.0))
        card = QRectF((self.width() - card_width) / 2.0,
                      (self.height() - 116.0) / 2.0, card_width, 116.0)
        painter.setPen(QPen(QColor("#8B919C"), 1))
        painter.setBrush(QColor("#1C232F"))
        painter.drawRoundedRect(card, 13, 13)

        painter.setPen(QColor("#FFFFFF"))
        heading = QRectF(card.left() + 16, card.top() + 11,
                         card.width() - 32, 24)
        painter.drawText(heading, Qt.AlignCenter, "Đang xử lý Guided Edit")
        stage = QRectF(card.left() + 13, card.top() + 37,
                       card.width() - 26, 22)
        painter.setPen(QColor("#D6DFEA"))
        painter.drawText(stage, Qt.AlignCenter, self.processing_stage)

        track = QRectF(card.left() + 24, card.top() + 68,
                       card.width() - 48, 13)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#455063"))
        painter.drawRoundedRect(track, 6, 6)
        fill_width = track.width() * self.processing_percent / 100.0
        if fill_width > 0:
            painter.setBrush(QColor("#38CF87"))
            painter.drawRoundedRect(QRectF(track.left(), track.top(),
                                           fill_width, track.height()), 6, 6)
        painter.setPen(QColor("#FFFFFF"))
        footer = QRectF(card.left() + 12, card.top() + 87,
                        card.width() - 24, 20)
        painter.drawText(footer, Qt.AlignCenter,
                         f"Khoảng {self.processing_percent}% · tiến độ theo giai đoạn")
        painter.restore()

    def enterEvent(self, event):
        self._hover_inside = True
        self.setCursor(Qt.CrossCursor if self.draw_mode == "box" else Qt.BlankCursor)
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
            if self.draw_mode == "box":
                self._box_start = self._box_end = point
                self.update()
                return
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

        if self.draw_mode == "box":
            self._box_end = self._clamped_image_point(event.position().x(), event.position().y())
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
            if self.draw_mode == "box":
                end = self._clamped_image_point(event.position().x(), event.position().y())
                start = self._box_start
                self._box_start = self._box_end = None
                if start is not None and end is not None:
                    x0, x1 = sorted((int(round(start[0])), int(round(end[0]))))
                    y0, y1 = sorted((int(round(start[1])), int(round(end[1]))))
                    if x1 > x0 and y1 > y0:
                        self.boxFinished.emit((x0, y0, x1, y1))
                self.update()
                return
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

        # Zoom around cursor, not around image center.
        if before is not None:
            rect = self._image_rect()
            scale = self._base_scale() * self.zoom
            target_x = rect.left() + before[0] * scale
            target_y = rect.top() + before[1] * scale
            self.pan_x += cursor_pos.x() - target_x
            self.pan_y += cursor_pos.y() - target_y

        self.update()
