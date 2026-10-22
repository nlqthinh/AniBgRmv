import sys

from PySide6.QtWidgets import QApplication

from ani_bgrmv.main_window import MainWindow
from ani_bgrmv.styles import apply_fixed_dark_theme


def main() -> int:
    app = QApplication(sys.argv)
    apply_fixed_dark_theme(app)  # M49: always dark on Windows light/dark PCs
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
