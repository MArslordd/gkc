from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def patch_positions(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test_seen"


def save_patch(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a patch-level dataset from DiffSeg30k image/mask pairs.")
    parser.add_argument("--source", default=Path("data/patch_demo_diffseg30k/images"), type=Path)
    parser.add_argument("--target", default=Path("data/diffseg30k_patch"), type=Path)
    parser.add_argument("--patch", default=224, type=int)
    parser.add_argument("--stride", default=112, type=int)
    parser.add_argument("--fake-threshold", default=0.30, type=float, help="Mask coverage >= threshold => fake patch.")
    parser.add_argument("--real-threshold", default=0.02, type=float, help="Mask coverage <= threshold => real patch.")
    parser.add_argument("--train-ratio", default=0.70, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--max-per-class-split", default=None, type=int)
    parser.add_argument("--limit-images", default=None, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.source.glob("*_image.*"))
    if args.limit_images is not None:
        image_paths = image_paths[: args.limit_images]
    if not image_paths:
        raise FileNotFoundError(f"No '*_image.*' files found under {args.source}")

    counts: dict[str, dict[str, int]] = {
        "train": {"real": 0, "fake": 0, "ignored": 0},
        "val": {"real": 0, "fake": 0, "ignored": 0},
        "test_seen": {"real": 0, "fake": 0, "ignored": 0},
    }
    metadata = []

    for image_index, image_path in enumerate(tqdm(image_paths, desc="images")):
        stem = image_path.name.split("_image.")[0]
        mask_path = args.source / f"{stem}_mask.png"
        if not mask_path.exists():
            counts[split_name(image_index, len(image_paths), args.train_ratio, args.val_ratio)]["ignored"] += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        h, w = image.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_binary = mask > 0
        split = split_name(image_index, len(image_paths), args.train_ratio, args.val_ratio)

        for y in patch_positions(h, args.patch, args.stride):
            for x in patch_positions(w, args.patch, args.stride):
                crop = image[y : min(y + args.patch, h), x : min(x + args.patch, w)]
                crop_mask = mask_binary[y : min(y + args.patch, h), x : min(x + args.patch, w)]
                if crop.shape[0] != args.patch or crop.shape[1] != args.patch:
                    crop = cv2.resize(crop, (args.patch, args.patch), interpolation=cv2.INTER_LINEAR)
                coverage = float(crop_mask.mean()) if crop_mask.size else 0.0

                if coverage >= args.fake_threshold:
                    label = "fake"
                elif coverage <= args.real_threshold:
                    label = "real"
                else:
                    counts[split]["ignored"] += 1
                    continue

                if args.max_per_class_split is not None and counts[split][label] >= args.max_per_class_split:
                    continue

                index = counts[split][label]
                output_path = args.target / split / label / f"{stem}_{y:04d}_{x:04d}_{index:06d}.jpg"
                save_patch(crop, output_path)
                counts[split][label] += 1
                metadata.append(
                    {
                        "source_image": str(image_path),
                        "source_mask": str(mask_path),
                        "output": str(output_path),
                        "split": split,
                        "label": label,
                        "x": x,
                        "y": y,
                        "mask_coverage": coverage,
                    }
                )

    args.target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.source),
        "target": str(args.target),
        "patch": args.patch,
        "stride": args.stride,
        "fake_threshold": args.fake_threshold,
        "real_threshold": args.real_threshold,
        "counts": counts,
        "num_records": len(metadata),
    }
    (args.target / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.target / "patch_records.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
