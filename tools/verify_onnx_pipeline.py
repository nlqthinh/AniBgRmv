from pathlib import Path
import argparse
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from bg_engine import BgRemoveEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path ảnh test")
    parser.add_argument("--f2", default="models/f3_unet_mit_b1_768.onnx")
    parser.add_argument("--m6", default="models/m9_vitmatte_best_768.onnx")
    parser.add_argument("--out", default="outputs/verify_onnx")
    parser.add_argument("--no_debug", action="store_true")
    args = parser.parse_args()

    engine = BgRemoveEngine(
        f2a1_model_path=args.f2,
        m6_model_path=args.m6,
        size=768,
    )
    prediction = engine.process_for_preview(args.image)
    saved = engine.save_prediction(
        prediction,
        args.out,
        save_debug=not args.no_debug,
    )

    print("Runtime:", engine.runtime_info)
    print("Kết quả:")
    for k, v in saved.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
