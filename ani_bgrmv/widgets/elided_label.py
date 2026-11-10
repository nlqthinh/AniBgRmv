"""One-line status label that can never make its window bigger.

A word-wrapped QLabel asks for more height (and width) when a long message
arrives, e.g. a Track summary with its log path. Inside a window that already
fills a small screen, that pushes the layout past the screen. This label keeps
one line: what does not fit is replaced by "…" in the middle (the start of the
message and the end of a path stay visible); the full text is the tooltip.
"""
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QSizePolicy


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        # The layout decides the width; the text only fills what it is given.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def text(self) -> str:
        """The full message, not the shortened one on screen."""
        return self._full_text

    def setText(self, text: str):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._show_fitting_text()

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_fitting_text()

    def _show_fitting_text(self):
        one_line = " ".join(self._full_text.split())
        super().setText(self.fontMetrics().elidedText(
            one_line, Qt.TextElideMode.ElideMiddle, max(0, self.contentsRect().width())))
