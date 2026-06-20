from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.infer import prepare_image
from src.model import build_model
from src.utils import get_device, load_config


def patch_positions(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def make_heatmap(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    image_path: Path,
    output_path: Path,
    patch: int,
    stride: int,
    temp_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, w = image.shape[:2]
    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in patch_positions(h, patch, stride):
        for x in patch_positions(w, patch, stride):
            crop = image[y : min(y + patch, h), x : min(x + patch, w)]
            if crop.shape[0] != patch or crop.shape[1] != patch:
                crop = cv2.resize(crop, (patch, patch), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(temp_path), crop)
            rgb, srm, fft = prepare_image(str(temp_path), cfg["data"]["image_size"], device)
            score = torch.softmax(model(rgb, srm, fft), dim=1)[0, 1].item()
            score_map[y : min(y + patch, h), x : min(x + patch, w)] += score
            count_map[y : min(y + patch, h), x : min(x + patch, w)] += 1

    count_map[count_map == 0] = 1
    score_map /= count_map
    heat = cv2.applyColorMap((np.clip(score_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate patch heatmaps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_image.png")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--patch", default=224, type=int)
    parser.add_argument("--stride", default=112, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_paths = sorted(args.input_dir.glob(args.pattern))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images matched {args.input_dir / args.pattern}")

    temp_dir = args.output_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "patch.jpg"

    for image_path in tqdm(image_paths, desc="heatmaps"):
        stem = image_path.name.replace("_image.png", "").replace("_image.jpg", "")
        output_path = args.output_dir / f"{stem}_heatmap.jpg"
        make_heatmap(model, cfg, device, image_path, output_path, args.patch, args.stride, temp_path)

    print(f"saved {len(image_paths)} heatmaps to {args.output_dir}")


if __name__ == "__main__":
    main()