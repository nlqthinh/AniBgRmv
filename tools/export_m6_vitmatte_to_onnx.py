from pathlib import Path
import argparse

import torch
import torch.nn as nn
from transformers import VitMatteForImageMatting

class ViTMatteExportWrapper(nn.Module):
    def __init__(self, model: VitMatteForImageMatting):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values, return_dict=True)
        return out.alphas


def clean_state_dict(state: dict):
    clean = {}
    for k, v in state.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt",
        default="models/m6_vitmatte_tune_best.pt",
        help="Path tới M6 ViTMatte .pt",
    )
    parser.add_argument(
        "--base_model",
        default="hustvl/vitmatte-small-distinctions-646",
        help="HF base model dùng khi fine-tune M6",
    )
    parser.add_argument(
        "--out",
        default="models/m6_vitmatte_m6_768.onnx",
        help="Path ONNX output",
    )
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Bật strict=True nếu checkpoint chắc chắn khớp 100% với HF model.",
    )
    args = parser.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Processor call is not used during export, but loading it here helps confirm
    # the base model package is available and compatible.
    # Không cần AutoImageProcessor khi export ONNX.
    # _ = AutoImageProcessor.from_pretrained(args.base_model)
    model = VitMatteForImageMatting.from_pretrained(args.base_model)

    ckpt = torch.load(pt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError("Checkpoint format không hợp lệ.")

    state = clean_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=args.strict)

    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("Một số missing keys đầu tiên:", missing[:10])
    if len(unexpected) > 0:
        print("Một số unexpected keys đầu tiên:", unexpected[:10])

    model.eval()
    wrapper = ViTMatteExportWrapper(model)
    wrapper.eval()

    # ViTMatte input is 4 channels: RGB image normalized + trimap rescaled.
    dummy = torch.randn(1, 4, args.size, args.size, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["pixel_values"],
        output_names=["alpha"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Export xong M6 ViTMatte ONNX: {out_path}")


if __name__ == "__main__":
    main()
