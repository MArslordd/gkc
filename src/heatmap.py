from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .infer import prepare_image
from .model import build_model
from .utils import get_device, load_config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobilenetv3_defactify_all.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--patch", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--output", default="outputs/visualizations/heatmap.jpg")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {args.image}")
    h, w = image.shape[:2]
    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    temp_dir = Path(cfg["output"]["dir"]) / "_patch_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "patch.jpg"

    for y in range(0, max(1, h - args.patch + 1), args.stride):
        for x in range(0, max(1, w - args.patch + 1), args.stride):
            patch = image[y : y + args.patch, x : x + args.patch]
            if patch.shape[0] < args.patch or patch.shape[1] < args.patch:
                continue
            cv2.imwrite(str(temp_file), patch)
            rgb, srm, fft = prepare_image(str(temp_file), cfg["data"]["image_size"], device)
            score = torch.softmax(model(rgb, srm, fft), dim=1)[0, 1].item()
            score_map[y : y + args.patch, x : x + args.patch] += score
            count_map[y : y + args.patch, x : x + args.patch] += 1

    count_map[count_map == 0] = 1
    score_map /= count_map
    heat = cv2.applyColorMap((score_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
