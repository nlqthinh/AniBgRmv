import sys


def main() -> int:
    # The M78 packaging self-test must run before importing Qt or MainWindow,
    # so it tests CUDA in the actual *frozen* process with no GUI dependencies.
    if '--video-selftest' in sys.argv[1:]:
        from ani_bgrmv.frozen_video_selftest import main as video_selftest
        return video_selftest()

    if '--cpu-selftest' in sys.argv[1:]:
        from ani_bgrmv.frozen_cpu_selftest import main as cpu_selftest
        return cpu_selftest()

    if '--cuda-selftest' in sys.argv[1:]:
        from ani_bgrmv.frozen_cuda_selftest import main as cuda_selftest
        return cuda_selftest()

    from PySide6.QtWidgets import QApplication
    from ani_bgrmv.main_window import MainWindow
    from ani_bgrmv.styles import apply_fixed_dark_theme

    app = QApplication(sys.argv)
    apply_fixed_dark_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
