from pathlib import Path
import argparse

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class F2A1ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, export_prob: bool = True):
        super().__init__()
        self.model = model
        self.export_prob = export_prob

    def forward(self, x):
        y = self.model(x)
        if self.export_prob:
            y = torch.sigmoid(y)
        return y


def build_f2a1_model():
    return smp.Unet(
        encoder_name="mit_b1",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_state(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format không hợp lệ.")

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    return clean_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pth",
        default="models/F2_A1_mild_aug_from_FP3_768_unet_mit_b1_3k5_best.pth",
        help="Path tới F2_A1 .pth",
    )
    parser.add_argument(
        "--out",
        default="models/f2a1_unet_mit_b1_768.onnx",
        help="Path ONNX output",
    )
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    pth_path = Path(args.pth)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_f2a1_model()
    model.load_state_dict(load_state(pth_path), strict=True)
    model.eval()

    wrapper = F2A1ExportWrapper(model, export_prob=True)
    wrapper.eval()

    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["alpha"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Export xong F2_A1 ONNX: {out_path}")


if __name__ == "__main__":
    main()
