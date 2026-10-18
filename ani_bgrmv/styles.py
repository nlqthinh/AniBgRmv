from PySide6.QtWidgets import QPushButton, QSizePolicy

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
