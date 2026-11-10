"""Annotator app: AniBGRmv_CPU.exe --annotator (or python app.py --annotator).

Grid page: every image of one ID package with its proposal and status.
  Select tiles, then A = accept the proposals as they are; O / double click =
  open in the editor; K = mark hard.
Editor page, object by object (one color each):
  Object: pick it in the list, Ctrl+click it on the image or Tab; N new object,
    G then click another object = merge it into this one, Del = drop this object.
  Khôi phục (R) gives gray pixels to the object, Xóa (E) turns its pixels gray;
    neither ever touches another object. Guided (HQ-SAM) with Chấm (point),
    Cọ (brush) or Khung (box); Manual (M) = exactly the brush or box drawn.
  Ctrl+Z / Ctrl+Y, Enter save + next, S one object + next, K hard + next,
  H colors on/off, F fit, [ ] brush size, Esc back to the grid.
Models and files only run on one worker thread, so the window never freezes;
while it works, new actions are ignored instead of queued.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QKeySequence, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QGridLayout, QHBoxLayout, QLabel, QListView,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
                               QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from .image_edit import ERASE, RESTORE
from .overlay import object_color, overlay
from .workspace import HARD, SINGLE, SPLIT, TODO, Workspace

POINT, BRUSH, BOX = 'point', 'brush', 'box'
STATUS_TEXT = {TODO: 'chưa làm', SINGLE: '1 object', SPLIT: 'đã tách', HARD: 'khó'}
STATUS_COLOR = {TODO: QColor(40, 40, 40), SINGLE: QColor(35, 60, 90), SPLIT: QColor(30, 80, 45),
                HARD: QColor(100, 70, 20)}
NOTHING_CHANGED = 'Không có gì thay đổi: Khôi phục chỉ lấy vùng XÁM; Xóa chỉ xóa object đang chọn.'
DRAG_PX = 6           # screen pixels before a press becomes a drag
ICON = 176
ERROR_LOG = Path(sys.argv[0]).resolve().parent / 'annotator_loi.txt'   # next to the EXE / app.py


def _qimage(rgb: np.ndarray) -> QImage:
    h, w = rgb.shape[:2]
    data = np.ascontiguousarray(rgb)
    return QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _swatch(object_id: int) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*object_color(object_id)))
    return QIcon(pixmap)


class _Worker(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)

    @Slot(int, object)
    def run(self, token: int, job) -> None:
        try:
            result = job()
        except Exception:
            self.failed.emit(token, traceback.format_exc())
            return
        self.done.emit(token, result)


class Canvas(QGraphicsView):
    """Image view: wheel zoom, middle-drag pan; reports, in image pixels,
    Ctrl+clicks (pick), clicks, brush strokes and boxes for the current shape."""
    picked = Signal(float, float)
    clicked = Signal(float, float, int)      # x, y, 1 = left / 2 = right
    stroked = Signal(list)
    boxed = Signal(float, float, float, float)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._pixmap = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(18, 18, 18)))
        self.shape = POINT
        self.brush_radius = 8
        self._press = None      # (screen pos, image pos, button, ctrl)
        self._points: list = []
        self._pan = None
        self._preview = None    # brush path or box being drawn

    def show_image(self, image: QImage, fit: bool) -> None:
        self._pixmap.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(QRectF(0, 0, image.width(), image.height()))
        if fit:
            self.fit()

    def fit(self) -> None:
        self.fitInView(self._pixmap, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def _image_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _clear_preview(self) -> None:
        if self._preview is not None:
            self.scene().removeItem(self._preview)
            self._preview = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan = event.position()
            return
        button = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.RightButton: 2}.get(event.button())
        if button:
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._press = (event.position(), self._image_pos(event), button, ctrl)
            self._points = [self._image_pos(event)]

    def mouseMoveEvent(self, event) -> None:
        if self._pan is not None:
            delta = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            return
        if self._press is None or self._press[3] or self._press[2] != 1:
            return
        here = self._image_pos(event)
        self._clear_preview()
        if self.shape == BRUSH:
            self._points.append(here)
            path = QPainterPath(self._points[0])
            for point in self._points[1:]:
                path.lineTo(point)
            self._preview = QGraphicsPathItem(path)
            pen = QPen(QColor(255, 255, 255, 140), 2 * self.brush_radius)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._preview.setPen(pen)
        elif self.shape == BOX:
            self._preview = QGraphicsRectItem(QRectF(self._press[1], here).normalized())
            self._preview.setPen(QPen(QColor(255, 255, 0), 0))
        if self._preview is not None:
            self.scene().addItem(self._preview)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan = None
            return
        if self._press is None:
            return
        _, start, button, ctrl = self._press
        end = self._image_pos(event)
        self._press = None
        self._clear_preview()
        if ctrl:
            self.picked.emit(start.x(), start.y())
        elif button == 1 and self.shape == BRUSH:
            self.stroked.emit([(p.x(), p.y()) for p in self._points + [end]])
        elif button == 1 and self.shape == BOX and self._dragged(start, end):
            rect = QRectF(start, end).normalized()
            self.boxed.emit(rect.left(), rect.top(), rect.right(), rect.bottom())
        else:
            self.clicked.emit(start.x(), start.y(), button)

    def _dragged(self, start: QPointF, end: QPointF) -> bool:
        a, b = self.mapFromScene(start), self.mapFromScene(end)
        return abs(a.x() - b.x()) + abs(a.y() - b.y()) > DRAG_PX


class AnnotatorWindow(QMainWindow):
    _submit = Signal(int, object)

    def __init__(self, dataset: str | None = None, package: str | None = None):
        super().__init__()
        self.setWindowTitle('AniBgRmv – Gán nhãn object')
        self.resize(1280, 760)
        self._settings = QSettings('AniBgRmv', 'Annotator')
        self._workspace: Workspace | None = None
        self._status: dict[str, str] = {}
        self._edit = None
        self._item = None
        self._labels = None
        self._objects: list = []
        self._opened_at = 0.0
        self._merge_pick = False
        self._callbacks: dict[int, object] = {}
        self._token = 0
        self._start_worker()
        self._build_ui()
        self._build_shortcuts()
        dataset = dataset or self._settings.value('dataset', '')
        package = package or self._settings.value('package', '')
        if dataset and package:
            self._load(dataset, package)

    # -- worker thread ----------------------------------------------------------
    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._thread.finished.connect(self._worker.deleteLater)
        self._submit.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _busy(self) -> bool:
        if self._callbacks:
            self.statusBar().showMessage('Đang tính, đợi một chút…')
        return bool(self._callbacks)

    def _call(self, job, on_done, message: str) -> None:
        self._token += 1
        self._callbacks[self._token] = on_done
        self.statusBar().showMessage(message)
        self._submit.emit(self._token, job)

    @Slot(int, object)
    def _on_done(self, token: int, result) -> None:
        callback = self._callbacks.pop(token)
        self.statusBar().clearMessage()
        callback(result)

    @Slot(int, str)
    def _on_failed(self, token: int, error: str) -> None:
        self._callbacks.pop(token, None)
        self.statusBar().clearMessage()
        with ERROR_LOG.open('a', encoding='utf-8') as log:   # the EXE has no console
            log.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{error}\n")
        QMessageBox.warning(self, 'Lỗi', f"{error.strip().splitlines()[-1]}\n\nChi tiết: {ERROR_LOG}")

    def closeEvent(self, event) -> None:
        self._thread.quit()
        self._thread.wait()            # the running job finishes first
        if self._edit is not None:     # worker stopped: safe to save here
            self._workspace.save_draft(self._item, self._edit)
        super().closeEvent(event)

    # -- layout -----------------------------------------------------------------
    def _build_ui(self) -> None:
        top = QHBoxLayout()
        self._paths = QLabel('Chọn thư mục dataset và thư mục gói (goi_ID_…).')
        self._progress = QLabel('')
        choose_dataset = QPushButton('Chọn dataset…')
        choose_dataset.clicked.connect(self._choose_dataset)
        choose_package = QPushButton('Chọn gói…')
        choose_package.clicked.connect(self._choose_package)
        for widget in (choose_dataset, choose_package, self._paths):
            top.addWidget(widget)
        top.addStretch(1)
        top.addWidget(self._progress)
        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_grid())
        self._pages.addWidget(self._build_editor())
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(self._pages, 1)
        self.setCentralWidget(root)

    def _build_grid(self) -> QWidget:
        self._grid = QListWidget()
        self._grid.setViewMode(QListView.ViewMode.IconMode)
        self._grid.setIconSize(QSize(ICON, ICON))
        self._grid.setGridSize(QSize(ICON + 24, ICON + 44))
        self._grid.setResizeMode(QListView.ResizeMode.Adjust)
        self._grid.setMovement(QListView.Movement.Static)
        self._grid.setWordWrap(True)
        self._grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._grid.itemDoubleClicked.connect(lambda item: self._open(item.data(Qt.ItemDataRole.UserRole)))
        self._filter = QComboBox()
        self._filter.addItems(['Tất cả', 'Chưa làm', 'Đề xuất 2+ object', 'Khó'])
        self._filter.currentIndexChanged.connect(self._apply_filter)
        bar = QHBoxLayout()
        bar.addWidget(QLabel('Lọc:'))
        bar.addWidget(self._filter)
        for text, handler in (('Mở (O)', self._open_selected), ('Chấp nhận đề xuất (A)', self._accept_selected),
                              ('Khó (K)', self._hard_selected)):
            button = QPushButton(text)
            button.clicked.connect(handler)
            bar.addWidget(button)
        bar.addStretch(1)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(bar)
        layout.addWidget(self._grid, 1)
        return page

    @staticmethod
    def _choice(texts: list[str], on_pick, columns: int) -> tuple[QButtonGroup, QGridLayout]:
        """Exclusive buttons in a grid; ``on_pick(index)`` when one is clicked."""
        group, grid = QButtonGroup(), QGridLayout()
        for index, text in enumerate(texts):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setChecked(index == 0)
            group.addButton(button, index)
            grid.addWidget(button, index // columns, index % columns)
        group.idClicked.connect(on_pick)
        return group, grid

    def _build_editor(self) -> QWidget:
        self._canvas = Canvas()
        self._canvas.picked.connect(self._on_pick)
        self._canvas.clicked.connect(self._on_click)
        self._canvas.stroked.connect(self._on_stroke)
        self._canvas.boxed.connect(self._on_box)
        panel = QVBoxLayout()
        self._title = QLabel('')
        self._title.setWordWrap(True)
        panel.addWidget(self._title)
        panel.addWidget(QLabel('Object đang sửa (Ctrl+click trên ảnh, Tab):'))
        self._palette = QListWidget()
        self._palette.setMaximumHeight(150)
        self._palette.itemClicked.connect(lambda entry: self._select(entry.data(Qt.ItemDataRole.UserRole)))
        panel.addWidget(self._palette)
        row = QHBoxLayout()
        for text, handler in (('+ Mới (N)', self._new_object), ('Gộp vào (G)', self._start_merge),
                              ('Bỏ (Del)', self._drop_object)):
            button = QPushButton(text)
            button.clicked.connect(handler)
            row.addWidget(button)
        panel.addLayout(row)
        self._modes, grid = self._choice(['Khôi phục (R)', 'Xóa (E)'], lambda _: self._update_help(), 2)
        panel.addLayout(grid)
        self._ways, grid = self._choice(['Guided – HQ-SAM', 'Manual – tay (M)'], lambda _: self._sync_tools(), 2)
        panel.addLayout(grid)
        self._shapes, grid = self._choice(['Chấm (1)', 'Cọ (2)', 'Khung (3)'], lambda _: self._sync_tools(), 3)
        panel.addLayout(grid)
        self._brush = QSpinBox()
        self._brush.setRange(1, 300)
        self._brush.setValue(8)
        self._brush.setPrefix('Cỡ cọ: ')
        self._brush.setSuffix(' px')
        self._brush.valueChanged.connect(lambda value: setattr(self._canvas, 'brush_radius', value))
        panel.addWidget(self._brush)
        self._colors = QCheckBox('Hiện màu (H)')
        self._colors.setChecked(True)
        self._colors.toggled.connect(lambda _: self._refresh())
        panel.addWidget(self._colors)
        self._help = QLabel('')
        self._help.setWordWrap(True)
        panel.addWidget(self._help)
        self._info = QLabel('')
        self._info.setWordWrap(True)
        panel.addWidget(self._info)
        panel.addStretch(1)
        buttons = QGridLayout()
        for index, (text, handler) in enumerate((
                ('Hoàn tác (Ctrl+Z)', self._undo), ('Làm lại (Ctrl+Y)', self._redo),
                ('Lưu, ảnh tiếp (Enter)', lambda: self._finish('save')),
                ('1 object (S)', lambda: self._finish('single')), ('Khó (K)', lambda: self._finish('hard')),
                ('Về lưới (Esc)', self._back_to_grid),
                ('◀ Ảnh trước', lambda: self._step(-1)), ('Ảnh sau ▶', lambda: self._step(1)))):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button, index // 2, index % 2)
        panel.addLayout(buttons)
        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(290)
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(side)
        self._sync_tools()
        return page

    def _build_shortcuts(self) -> None:
        editor = {'R': lambda: self._click_choice(self._modes, 0), 'E': lambda: self._click_choice(self._modes, 1),
                  'M': lambda: self._click_choice(self._ways, 1 - self._ways.checkedId()),
                  '1': lambda: self._click_choice(self._shapes, 0), '2': lambda: self._click_choice(self._shapes, 1),
                  '3': lambda: self._click_choice(self._shapes, 2),
                  'N': self._new_object, 'G': self._start_merge, 'Delete': self._drop_object,
                  'Tab': self._next_object, 'Ctrl+Z': self._undo, 'Ctrl+Y': self._redo,
                  'Return': lambda: self._finish('save'), 'Enter': lambda: self._finish('save'),
                  'S': lambda: self._finish('single'), 'K': lambda: self._finish('hard'),
                  'H': lambda: self._colors.toggle(), 'F': lambda: self._canvas.fit(),
                  '[': lambda: self._brush.setValue(self._brush.value() - 2),
                  ']': lambda: self._brush.setValue(self._brush.value() + 2),
                  'Escape': self._back_to_grid, 'PgDown': lambda: self._step(1), 'PgUp': lambda: self._step(-1)}
        grid = {'O': self._open_selected, 'A': self._accept_selected, 'K': self._hard_selected}
        for key in set(editor) | set(grid):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda key=key: self._key(key, editor, grid))

    def _key(self, key: str, editor: dict, grid: dict) -> None:
        handler = (editor if self._pages.currentIndex() == 1 else grid).get(key)
        if handler is not None:
            handler()

    # -- tool state ---------------------------------------------------------------
    def _click_choice(self, group: QButtonGroup, index: int) -> None:
        button = group.button(index)
        if button.isEnabled():
            button.click()

    def _mode(self) -> str:
        return RESTORE if self._modes.checkedId() == 0 else ERASE

    def _guided(self) -> bool:
        return self._ways.checkedId() == 0

    def _sync_tools(self) -> None:
        """Manual has no 'point'; images without HQ-SAM are manual only."""
        sam = self._edit is None or self._edit.can_use_sam
        self._ways.button(0).setEnabled(sam)
        if not sam:
            self._ways.button(1).setChecked(True)
        self._shapes.button(0).setEnabled(self._guided())
        if not self._guided() and self._shapes.checkedId() == 0:
            self._shapes.button(1).setChecked(True)
        self._canvas.shape = (POINT, BRUSH, BOX)[self._shapes.checkedId()]
        self._update_help()

    def _update_help(self) -> None:
        if self._merge_pick:
            text = 'GỘP: click object cần gộp vào object đang chọn (click phải: hủy).'
        else:
            what = ('Khôi phục: vùng XÁM thành object đang chọn.' if self._mode() == RESTORE
                    else 'Xóa: vùng của object đang chọn thành XÁM.')
            how = {POINT: 'Click lên vùng.', BRUSH: 'Kéo cọ lên vùng.', BOX: 'Kéo khung quanh vùng.'}
            way = 'HQ-SAM tự tìm vùng.' if self._guided() else 'Đúng nét vẽ.'
            text = (f'{what} {how[self._canvas.shape]} {way} Không bao giờ đụng object khác: muốn chuyển '
                    'vùng từ A sang B thì Xóa ở A rồi Khôi phục ở B.')
        self._help.setText(text)

    # -- workspace ----------------------------------------------------------------
    def _choose_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Thư mục dataset (chứa các folder 1, 2, 3…)',
                                                self._settings.value('dataset', ''))
        if path:
            self._settings.setValue('dataset', path)
            self._reload()

    def _choose_package(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Thư mục gói goi_ID_…', self._settings.value('package', ''))
        if path:
            self._settings.setValue('package', path)
            self._reload()

    def _reload(self) -> None:
        dataset, package = self._settings.value('dataset', ''), self._settings.value('package', '')
        if dataset and package and not self._busy():
            self._load(dataset, package)

    def _load(self, dataset: str, package: str) -> None:
        self._close_edit()

        def job():
            workspace = Workspace(Path(dataset), Path(package))
            return workspace, {item.name: workspace.status(item) for item in workspace.items}

        def loaded(result) -> None:
            self._workspace, self._status = result
            self._settings.setValue('dataset', dataset)
            self._settings.setValue('package', package)
            self._paths.setText(f'Dataset: {dataset}   |   Gói: ID {self._workspace.folder}')
            self._fill_grid()
            self._pages.setCurrentIndex(0)
            if self._workspace.missing_files:
                QMessageBox.information(self, 'Thiếu ảnh', f'{len(self._workspace.missing_files)} ảnh của gói '
                                        'không tìm thấy trong dataset này (so theo nội dung), bỏ qua các ảnh đó.')
        self._call(job, loaded, 'Đang mở gói (lần đầu: tìm ảnh theo nội dung, có thể mất 1–2 phút)…')

    def _fill_grid(self) -> None:
        self._grid.clear()
        missing = set(self._workspace.missing_files)
        for item in self._workspace.items:
            icon = QIcon(str(item.preview)) if item.preview else QIcon()
            entry = QListWidgetItem(icon, '')
            entry.setData(Qt.ItemDataRole.UserRole, item.index)
            if item.image in missing:
                entry.setFlags(Qt.ItemFlag.NoItemFlags)
            self._grid.addItem(entry)
            self._show_status(item)
        self._apply_filter()

    def _show_status(self, item) -> None:
        entry = self._grid.item(item.index)
        status = self._status.get(item.name, TODO)
        proposed = '' if item.proposed_objects is None else f' · đề xuất {item.proposed_objects}'
        sam = '' if item.features else ' · không SAM'
        missing = ' · KHÔNG THẤY ẢNH' if item.image in self._workspace.missing_files else ''
        entry.setText(f'{item.index + 1:03d} {STATUS_TEXT[status]}{proposed}{sam}{missing}\n{item.name}')
        entry.setBackground(STATUS_COLOR[status])
        statuses = [self._status.get(i.name, TODO) for i in self._workspace.items]
        self._progress.setText(f'Đã xong {len(statuses) - statuses.count(TODO)}/{len(statuses)} · '
                               f'đã tách {statuses.count(SPLIT)} · 1 object {statuses.count(SINGLE)} · '
                               f'khó {statuses.count(HARD)}')

    def _apply_filter(self) -> None:
        if self._workspace is None:
            return
        choice = self._filter.currentIndex()
        for item in self._workspace.items:
            status = self._status.get(item.name, TODO)
            visible = (choice == 0 or (choice == 1 and status == TODO) or (choice == 3 and status == HARD)
                       or (choice == 2 and (item.proposed_objects or 0) >= 2))
            self._grid.item(item.index).setHidden(not visible)

    def _selected_items(self) -> list:
        return [self._workspace.items[e.data(Qt.ItemDataRole.UserRole)] for e in self._grid.selectedItems()]

    def _open_selected(self) -> None:
        selected = self._selected_items()
        if selected:
            self._open(selected[0].index)

    def _batch(self, mark: str, only_todo: bool) -> None:
        if self._workspace is None or self._busy():
            return
        workspace = self._workspace
        items = [i for i in self._selected_items() if not only_todo or self._status.get(i.name, TODO) == TODO]
        if not items:
            return

        def job():
            return {item.name: workspace.finish(item, workspace.open(item), mark) for item in items}

        def finished(statuses) -> None:
            self._status.update(statuses)
            for item in items:
                self._show_status(item)
            self._apply_filter()
        self._call(job, finished, f'Đang lưu {len(items)} ảnh…')

    def _accept_selected(self) -> None:
        self._batch('save', only_todo=True)

    def _hard_selected(self) -> None:
        self._batch('hard', only_todo=False)

    # -- editor: images -------------------------------------------------------------
    @staticmethod
    def _render(edit, colors: bool) -> dict:
        """Worker thread: everything the editor shows about an edit."""
        labels = edit.labels()
        image = (overlay(edit.rgb, edit.alpha, labels, edges=True, highlight=edit.active, numbers=True)
                 if colors else edit.rgb)
        return {'image': _qimage(image), 'labels': labels, 'objects': edit.objects(),
                'active': edit.active, 'gray': edit.gray_share()}

    def _show(self, view: dict, fit: bool = False) -> None:
        self._labels, self._objects = view['labels'], view['objects']
        self._canvas.show_image(view['image'], fit)
        self._palette.blockSignals(True)
        self._palette.clear()
        for object_id, share in self._objects:
            entry = QListWidgetItem(_swatch(object_id), f'Object {object_id} · {share:.0%}')
            entry.setData(Qt.ItemDataRole.UserRole, object_id)
            self._palette.addItem(entry)
            if object_id == view['active']:
                entry.setSelected(True)
                self._palette.setCurrentItem(entry)
        self._palette.blockSignals(False)
        sam = 'có HQ-SAM' if self._edit.can_use_sam else 'KHÔNG có HQ-SAM (chỉ Manual)'
        count = sum(share > 0 for _, share in self._objects)
        self._info.setText(f'{count} object · vùng xám (không thuộc object nào): {view["gray"]:.1%} · {sam}')

    def _open(self, index: int) -> None:
        if self._workspace is None or self._busy():
            return
        item = self._workspace.items[index]
        if item.image in self._workspace.missing_files:
            return
        self._close_edit()
        workspace, colors = self._workspace, self._colors.isChecked()

        def job():
            edit = workspace.open(item)
            return edit, self._render(edit, colors)

        def opened(result) -> None:
            self._edit, view = result
            self._item, self._merge_pick = item, False
            self._opened_at = time.monotonic()
            status = STATUS_TEXT[self._status.get(item.name, TODO)]
            self._title.setText(f'{item.index + 1}/{len(workspace.items)} · {item.name} · {status}')
            self._pages.setCurrentIndex(1)
            self._sync_tools()
            self._show(view, fit=True)
        self._call(job, opened, 'Đang mở ảnh…')

    def _close_edit(self) -> None:
        """Keep unsaved work as a draft before another image or package is opened."""
        if self._edit is None:
            return
        workspace, item, edit = self._workspace, self._item, self._edit
        self._edit = self._item = None
        self._call(lambda: workspace.save_draft(item, edit), lambda _: None, 'Đang lưu nháp…')

    def _back_to_grid(self) -> None:
        if self._busy():
            return
        if self._item is not None:
            self._grid.setCurrentRow(self._item.index)
        self._close_edit()
        self._pages.setCurrentIndex(0)

    def _step(self, offset: int) -> None:
        if self._item is None or self._busy():
            return
        index = self._item.index + offset
        if 0 <= index < len(self._workspace.items):
            self._open(index)

    def _finish(self, mark: str) -> None:
        if self._edit is None or self._busy():
            return
        workspace, item, edit = self._workspace, self._item, self._edit
        seconds = time.monotonic() - self._opened_at
        self._edit = None

        def finished(status: str) -> None:
            self._status[item.name] = status
            self._show_status(item)
            self._apply_filter()
            if item.index + 1 < len(workspace.items):
                self._open(item.index + 1)
            else:
                self._item = None
                self._pages.setCurrentIndex(0)
        self._call(lambda: workspace.finish(item, edit, mark, seconds), finished, 'Đang lưu…')

    # -- editor: actions --------------------------------------------------------------
    def _edit_action(self, action, refused: str = NOTHING_CHANGED) -> None:
        """Run ``action(edit)`` on the worker thread, then redraw. When it
        returns False nothing changed: show ``refused``."""
        if self._edit is None or self._busy():
            return
        edit, colors = self._edit, self._colors.isChecked()

        def job():
            return action(edit), self._render(edit, colors)

        def done(result) -> None:
            changed, view = result
            self._show(view)
            if changed is False:
                self.statusBar().showMessage(refused, 6000)
        self._call(job, done, 'Đang tính…')

    def _refresh(self) -> None:
        self._edit_action(lambda edit: None)

    def _undo(self) -> None:
        self._edit_action(lambda e: e.undo(), refused='Không còn gì để hoàn tác.')

    def _redo(self) -> None:
        self._edit_action(lambda e: e.redo(), refused='Không còn gì để làm lại.')

    def _select(self, object_id) -> None:
        self._edit_action(lambda e: e.select(object_id), refused='Chọn object trong danh sách hoặc Ctrl+click.')

    def _next_object(self) -> None:
        ids = [i for i, _ in self._objects]
        if self._edit is not None and ids:
            current = ids.index(self._edit.active) if self._edit.active in ids else -1
            self._select(ids[(current + 1) % len(ids)])

    def _new_object(self) -> None:
        self._edit_action(lambda e: e.new_object())
        self._click_choice(self._modes, 0)   # a new object starts empty: restore into it

    def _drop_object(self) -> None:
        self._edit_action(lambda e: e.drop_active(), refused='Chưa chọn object.')

    def _start_merge(self) -> None:
        if self._edit is not None and self._edit.active is not None:
            self._merge_pick = True
            self._update_help()

    def _end_merge(self) -> None:
        self._merge_pick = False
        self._update_help()

    def _region_action(self, tool: str, geometry) -> None:
        mode, radius = self._mode(), self._canvas.brush_radius
        if self._guided():
            self._edit_action(lambda e: e.guided(mode, tool, geometry))
        else:
            self._edit_action(lambda e: e.manual(mode, tool, geometry, radius))

    def _on_pick(self, x: float, y: float) -> None:
        if self._edit is not None:
            self._select(self._edit.object_at((x, y)))

    def _on_click(self, x: float, y: float, button: int) -> None:
        if self._edit is None:
            return
        if self._merge_pick:
            self._end_merge()
            if button == 1:
                other = self._edit.object_at((x, y))
                self._edit_action(lambda e: e.merge_into_active(other),
                                  refused='Click vào một object KHÁC object đang chọn.')
        elif button == 1 and self._canvas.shape == POINT:
            self._region_action(POINT, (x, y))

    def _on_stroke(self, points: list) -> None:
        if self._edit is not None and not self._merge_pick:
            self._region_action(BRUSH, points)

    def _on_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self._edit is not None and not self._merge_pick:
            self._region_action(BOX, (x0, y0, x1, y1))


def main() -> int:
    from ..styles import apply_fixed_dark_theme
    app = QApplication(sys.argv)
    apply_fixed_dark_theme(app)
    window = AnnotatorWindow()
    window.show()
    return app.exec()
