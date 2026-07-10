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
    parser.add_argument("--pt", default="models/m6_vitmatte_tune_best.pt")
    parser.add_argument("--base_model", default="hustvl/vitmatte-small-distinctions-646")
    parser.add_argument("--out", default="models/m6_vitmatte_m6_768.onnx")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--external_data", action="store_true")
    args = parser.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    dummy = torch.randn(1, 4, args.size, args.size, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["pixel_values"],
        output_names=["alpha"],
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=args.external_data,
    )

    print(f"Export xong M6 ViTMatte ONNX: {out_path}")


if __name__ == "__main__":
    main()
