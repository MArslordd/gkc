from __future__ import annotations

import argparse

import cv2
import torch
from PIL import Image

from .datasets import build_transforms
from .fft_features import compute_fft_spectrum
from .model import build_model
from .srm import compute_srm_residual
from .utils import get_device, load_config


def prepare_image(path: str, image_size: int, device: torch.device):
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tfm = build_transforms(image_size, train=False)
    rgb = tfm(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
    srm = tfm(Image.fromarray(compute_srm_residual(image_rgb))).unsqueeze(0).to(device)
    fft = tfm(Image.fromarray(compute_fft_spectrum(image_rgb))).unsqueeze(0).to(device)
    return rgb, srm, fft


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobilenetv3_multibranch.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rgb, srm, fft = prepare_image(args.image, cfg["data"]["image_size"], device)
    prob_fake = torch.softmax(model(rgb, srm, fft), dim=1)[0, 1].item()
    label = "Fake" if prob_fake >= 0.5 else "Real"
    print({"image": args.image, "prediction": label, "fake_score": round(prob_fake, 6)})


if __name__ == "__main__":
    main()
