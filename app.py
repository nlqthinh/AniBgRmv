import sys


def main() -> int:
    # Packaging self-tests must run before importing Qt or MainWindow,
    # so they test CUDA in the actual *frozen* process with no GUI dependencies.
    if '--video-selftest' in sys.argv[1:]:
        from ani_bgrmv.frozen_video_selftest import main as video_selftest
        return video_selftest()

    if '--cpu-selftest' in sys.argv[1:]:
        from ani_bgrmv.frozen_cpu_selftest import main as cpu_selftest
        return cpu_selftest()

    if '--cuda-selftest' in sys.argv[1:]:
        from ani_bgrmv.frozen_cuda_selftest import main as cuda_selftest
        return cuda_selftest()

    # Instance-annotation command line tools (bước 0), also inside the EXE.
    if '--annotation-tool' in sys.argv[1:]:
        from ani_bgrmv.annotation.cli import main as annotation_tool
        args = sys.argv[1:]
        return annotation_tool(args[args.index('--annotation-tool') + 1:])

    # Instance-annotation app for the dataset annotator (separate window).
    if '--annotator' in sys.argv[1:]:
        from ani_bgrmv.annotation.annotator_window import main as annotator
        return annotator()

    from PySide6.QtWidgets import QApplication, QMessageBox
    from ani_bgrmv.app_config import APP_BASE_DIR
    from ani_bgrmv.main_window import MainWindow
    from ani_bgrmv.release_models import missing_model_files, missing_models_message
    from ani_bgrmv.styles import apply_fixed_dark_theme

    app = QApplication(sys.argv)
    apply_fixed_dark_theme(app)
    missing = missing_model_files(APP_BASE_DIR)
    if missing:  # e.g. only the App archive of the GPU release was extracted
        QMessageBox.critical(None, 'Thiếu model', missing_models_message(missing))
        return 1
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
