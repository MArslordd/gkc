from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def save_mask_overlay(image: Image.Image, mask: Image.Image, output_path: Path) -> None:
    image_rgb = np.array(image.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))
    binary = (mask_arr > 0).astype(np.uint8) * 255
    binary = cv2.resize(binary, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = image_rgb.copy()
    color = np.zeros_like(overlay)
    color[:, :, 0] = binary
    blended = cv2.addWeighted(overlay, 0.65, color, 0.35, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small DiffSeg30k patch demo subset.")
    parser.add_argument("--target", default=Path("data/patch_demo_diffseg30k"), type=Path)
    parser.add_argument("--count", default=100, type=int)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--skip", default=0, type=int)
    parser.add_argument("--max-scan", default=5000, type=int, help="Maximum streamed rows to scan.")
    return parser.parse_args()


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Please install datasets first: pip install datasets") from exc

    args = parse_args()
    target_root = args.target
    images_dir = target_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Chaos2629/Diffseg30k split={args.split} with streaming=True")
    print("The first sample can take a while because Hugging Face is resolving remote files.")
    dataset = load_dataset("Chaos2629/Diffseg30k", split=args.split, streaming=True)
    metadata = []
    saved = 0
    started_at = time.time()

    for index, sample in enumerate(tqdm(dataset, total=args.max_scan, desc="streaming samples")):
        if index >= args.max_scan:
            break
        if index < args.skip:
            continue
        if saved >= args.count:
            break

        image = sample["image"].convert("RGB")
        mask = sample["mask"].convert("L")
        stem = f"diffseg30k_{saved:03d}"

        image_path = images_dir / f"{stem}_image.png"
        mask_path = images_dir / f"{stem}_mask.png"
        overlay_path = images_dir / f"{stem}_mask_overlay.jpg"

        image.save(image_path)
        mask.save(mask_path)
        save_mask_overlay(image, mask, overlay_path)

        mask_values = sorted(int(value) for value in np.unique(np.array(mask)))
        metadata.append(
            {
                "index": index,
                "image": str(image_path),
                "mask": str(mask_path),
                "mask_overlay": str(overlay_path),
                "mask_values": mask_values,
            }
        )
        saved += 1
        if saved == 1:
            elapsed = time.time() - started_at
            print(f"First sample saved after {elapsed:.1f}s")

    (target_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {saved} samples to {target_root}")


if __name__ == "__main__":
    main()
