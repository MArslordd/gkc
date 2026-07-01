from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import build_model
from .utils import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobilenetv3_defactify_all.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--output", default="outputs/aigc_detector.onnx")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy, dummy, dummy),
        out_path,
        input_names=["rgb", "srm", "fft"],
        output_names=["logits"],
        dynamic_axes={"rgb": {0: "batch"}, "srm": {0: "batch"}, "fft": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out_path}")


if __name__ == "__main__":
    main()
