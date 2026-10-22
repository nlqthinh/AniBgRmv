from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QPushButton, QSizePolicy

BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 700;
    padding: 10px 18px;
    min-height: 42px;
    border-radius: 10px;
    border: 1px solid #B8B8B8;
    color: #000000;
    background: #F6F6F6;
}
QPushButton:hover {
    color: #000000;
    background: #ECECEC;
}
QPushButton:disabled {
    color: #777777;
    background: #EEEEEE;
    border: 1px solid #DDDDDD;
}
"""

PRIMARY_BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 900;
    padding: 10px 22px;
    min-height: 44px;
    border-radius: 10px;
    border: 1px solid #159447;
    color: #000000;
    background: #35D06F;
}
QPushButton:hover {
    color: #000000;
    background: #22C55E;
}
QPushButton:disabled {
    color: #777777;
    background: #BDEECC;
    border: 1px solid #BDEECC;
}
"""

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

CANCEL_BUTTON_STYLE = """
QPushButton {
    font-size: 15px;
    font-weight: 900;
    padding: 9px 22px;
    min-height: 40px;
    border-radius: 10px;
    border: 1px solid #000000;
    color: #FF2B2B;
    background: #000000;
}
QPushButton:hover {
    color: #FF4D4D;
    background: #171717;
}
QPushButton:disabled {
    color: #B54B4B;
    background: #222222;
    border: 1px solid #222222;
}
"""



EDIT_BLACK_BUTTON_STYLE = """
QPushButton {
    font-size: 14px;
    font-weight: 900;
    padding: 5px 14px;
    min-height: 30px;
    border-radius: 9px;
    border: 1px solid #FFFFFF;
    color: #FFFFFF;
    background: #000000;
}
QPushButton:hover {
    background: #151515;
}
QPushButton:checked {
    border: 2px solid #FFFFFF;
    background: #222222;
}
QPushButton:disabled {
    color: #999999;
    background: #222222;
    border: 1px solid #777777;
}
"""

EDIT_ERASE_BUTTON_STYLE = EDIT_BLACK_BUTTON_STYLE.replace("color: #FFFFFF;", "color: #FF3030;")
EDIT_RESTORE_BUTTON_STYLE = EDIT_BLACK_BUTTON_STYLE.replace("color: #FFFFFF;", "color: #20D060;")

DEBUG_BUTTON_STYLE = """
QPushButton {
    font-size: 14px;
    font-weight: 700;
    padding: 10px 14px;
    min-height: 42px;
    border-radius: 10px;
    border: 1px solid #B8B8B8;
    color: #444444;
    background: #F1F1F1;
}
QPushButton:checked {
    border: 1px solid #178A42;
    color: #06451F;
    background: #C4F2D1;
}
QPushButton:hover { border-color: #777777; }
QPushButton:disabled { color: #888888; background: #EEEEEE; }
"""


def apply_button_style(button: QPushButton, primary: bool = False):
    button.setStyleSheet(PRIMARY_BUTTON_STYLE if primary else BUTTON_STYLE)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


# =========================
# SMALL HELPERS
# =========================

# M49 — theme cố định, không phụ thuộc Windows light/dark mode.
# Giữ canvas upload/Edit màu sáng để xem ảnh rõ; phần giao diện bên ngoài tối.
APP_DARK_BG = "#171B22"
APP_TEXT_COLOR = "#F5F7FA"
APP_MUTED_TEXT_COLOR = "#D6DEE8"


def apply_fixed_dark_theme(app: QApplication) -> None:
    """Áp dụng palette tối cho toàn bộ app TRƯỚC khi khởi tạo MainWindow.

    Fusion là Qt style đa nền tảng và tránh Qt/Windows tự đổi màu widget
    theo theme của từng PC. QPalette xử lý widget không có stylesheet riêng;
    QSS bên dưới cố định nền trang và popup QComboBox.
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
    # Only the named root/page widgets receive an explicit dark background.
    # Do NOT style every QWidget/QFrame globally: the image canvases have
    # their own intentionally light backgrounds and must stay untouched.
    app.setStyleSheet("""
        QWidget#AniBgRmvMainWindow,
        QWidget#AniBgRmvHomePage,
        QWidget#AniBgRmvSinglePage,
        QWidget#AniBgRmvBatchPage,
        QWidget#AniBgRmvComputePage {
            background-color: #171B22;
            color: #F5F7FA;
        }
        QComboBox {
            color: #F5F7FA;
            background-color: #252E3B;
            selection-color: #FFFFFF;
            selection-background-color: #387CE8;
        }
        QComboBox QAbstractItemView {
            color: #F5F7FA;
            background-color: #252E3B;
            selection-color: #FFFFFF;
            selection-background-color: #387CE8;
        }
        QToolTip {
            color: #FFFFFF;
            background-color: #252D37;
            border: 1px solid #657287;
        }
    """)

