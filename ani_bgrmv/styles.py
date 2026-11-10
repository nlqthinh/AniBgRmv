"""Dark theme of every window, the same look as the annotator (Gan_nhan).

Buttons use the dark palette (gray, white text); a checked button (the mode
that is on) is blue. Only the Home/Settings buttons and the Stop/✕ buttons
keep their own, larger style.
"""
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QPushButton, QSizePolicy

APP_DARK_BG = "#171B22"
APP_TEXT_COLOR = "#F5F7FA"
MUTED_TEXT_COLOR = "#D6DEE8"
CANVAS_BG = "#121212"         # picture areas, as the annotator canvas
SELECTED_COLOR = "#2d6cdf"    # a checked (active) button, as the annotator

# Large Home/Settings buttons: same size as before, gray with white text.
BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 700;
    padding: 10px 18px;
    min-height: 42px;
    border-radius: 10px;
    border: 1px solid #4A5566;
    color: #F5F7FA;
    background: #313B48;
}
QPushButton:hover {
    background: #3A4656;
}
QPushButton:disabled {
    color: #98A3B0;
    background: #262E38;
    border: 1px solid #3A4352;
}
"""

# Stop buttons: the same size as BUTTON_STYLE, red text on black.
CANCEL_BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 700;
    padding: 10px 18px;
    min-height: 42px;
    border-radius: 10px;
    border: 1px solid #4A5566;
    color: #FF5555;
    background: #000000;
}
QPushButton:hover {
    color: #FF7777;
    background: #171717;
}
QPushButton:disabled {
    color: #A66B6B;
    background: #222222;
    border: 1px solid #777777;
}
"""

# The ✕ that closes the picture in the editor.
DANGER_BUTTON_STYLE = """
QPushButton {
    font-size: 18px;
    font-weight: 900;
    min-width: 38px;
    min-height: 34px;
    border-radius: 8px;
    border: 1px solid #A0A0A0;
    color: white;
    background: #111111;
}
QPushButton:hover {
    background: #C62828;
    border: 1px solid #C62828;
}
"""


def apply_button_style(button: QPushButton):
    button.setStyleSheet(BUTTON_STYLE)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


def apply_fixed_dark_theme(app: QApplication) -> None:
    """Dark palette for the whole app, set BEFORE any window is created.

    Fusion is Qt's cross-platform style, so no PC's Windows theme changes the
    widgets. QPalette colors every widget without its own stylesheet; the QSS
    below sets the page backgrounds, the blue checked buttons and popups.
    """
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(APP_DARK_BG))
    palette.setColor(QPalette.WindowText, QColor(APP_TEXT_COLOR))
    palette.setColor(QPalette.Base, QColor("#222A35"))
    palette.setColor(QPalette.AlternateBase, QColor("#2B3543"))
    palette.setColor(QPalette.Text, QColor(APP_TEXT_COLOR))
    palette.setColor(QPalette.Button, QColor("#313B48"))
    palette.setColor(QPalette.ButtonText, QColor(APP_TEXT_COLOR))
    palette.setColor(QPalette.Highlight, QColor("#387CE8"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipBase, QColor("#252D37"))
    palette.setColor(QPalette.ToolTipText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Link, QColor("#8BC0FF"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#98A3B0"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#98A3B0"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#98A3B0"))
    app.setPalette(palette)
    # Only the named root/page widgets get an explicit background: styling
    # every QWidget/QFrame would also repaint the picture canvases.
    app.setStyleSheet(f"""
        QWidget#AniBgRmvMainWindow,
        QWidget#AniBgRmvHomePage,
        QWidget#AniBgRmvSinglePage,
        QWidget#AniBgRmvBatchPage,
        QWidget#AniBgRmvVideoPage,
        QWidget#AniBgRmvComputePage {{
            background-color: {APP_DARK_BG};
            color: {APP_TEXT_COLOR};
        }}
        QPushButton:checked {{
            background-color: {SELECTED_COLOR};
            color: white;
        }}
        QComboBox {{
            color: {APP_TEXT_COLOR};
            background-color: #252E3B;
            selection-color: #FFFFFF;
            selection-background-color: #387CE8;
        }}
        QComboBox QAbstractItemView {{
            color: {APP_TEXT_COLOR};
            background-color: #252E3B;
            selection-color: #FFFFFF;
            selection-background-color: #387CE8;
        }}
        QToolTip {{
            color: #FFFFFF;
            background-color: #252D37;
            border: 1px solid #657287;
        }}
    """)
