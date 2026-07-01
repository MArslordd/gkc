from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def filter_fake_images(paths: list[Path], prefixes: list[str] | None) -> list[Path]:
    if not prefixes:
        return paths
    lowered = tuple(prefix.lower() for prefix in prefixes)
    return [path for path in paths if path.name.lower().startswith(lowered)]


def read_square(path: Path, tile_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    image = image[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)


def patch_positions(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    last = length - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


def split_plan(count_train: int, count_val: int, count_test: int) -> list[tuple[str, int]]:
    return [("train", count_train), ("val", count_val), ("test_seen", count_test)]


def build_mosaic(
    real_paths: list[Path],
    fake_paths: list[Path],
    rng: random.Random,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str | int]]:
    real_choices = [rng.choice(real_paths) for _ in range(3)]
    fake_choice = rng.choice(fake_paths)
    fake_slot = rng.randrange(4)

    canvas = np.zeros((tile_size * 2, tile_size * 2, 3), dtype=np.uint8)
    mask = np.zeros((tile_size * 2, tile_size * 2), dtype=np.uint8)
    real_index = 0
    placements: dict[str, str | int] = {"fake_slot": fake_slot, "fake": str(fake_choice)}

    for slot in range(4):
        y = 0 if slot < 2 else tile_size
        x = 0 if slot % 2 == 0 else tile_size
        if slot == fake_slot:
            tile = read_square(fake_choice, tile_size)
            mask[y : y + tile_size, x : x + tile_size] = 255
        else:
            path = real_choices[real_index]
            tile = read_square(path, tile_size)
            placements[f"real_{real_index}"] = str(path)
            real_index += 1
        canvas[y : y + tile_size, x : x + tile_size] = tile

    return canvas, mask, placements


def save_mask_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.45
    overlay = (image.astype(np.float32) * (1 - alpha) + red.astype(np.float32) * alpha).astype(np.uint8)
    cv2.imwrite(str(output_path), overlay)


def export_patches(
    image: np.ndarray,
    mask: np.ndarray,
    target_root: Path,
    split: str,
    stem: str,
    patch: int,
    stride: int,
    fake_threshold: float,
    real_threshold: float,
    counters: dict[str, dict[str, int]],
    records: list[dict[str, str | int | float]],
) -> None:
    h, w = image.shape[:2]
    mask_binary = mask > 0
    for y in patch_positions(h, patch, stride):
        for x in patch_positions(w, patch, stride):
            crop = image[y : min(y + patch, h), x : min(x + patch, w)]
            crop_mask = mask_binary[y : min(y + patch, h), x : min(x + patch, w)]
            if crop.shape[0] != patch or crop.shape[1] != patch:
                crop = cv2.resize(crop, (patch, patch), interpolation=cv2.INTER_LINEAR)
            coverage = float(crop_mask.mean()) if crop_mask.size else 0.0
            if coverage >= fake_threshold:
                label = "fake"
            elif coverage <= real_threshold:
                label = "real"
            else:
                counters[split]["ignored"] += 1
                continue

            index = counters[split][label]
            output_path = target_root / split / label / f"{stem}_{y:04d}_{x:04d}_{index:06d}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), crop)
            counters[split][label] += 1
            records.append(
                {
                    "mosaic": stem,
                    "output": str(output_path),
                    "split": split,
                    "label": label,
                    "x": x,
                    "y": y,
                    "mask_coverage": coverage,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create local-AIGC mosaic images and a patch-level real/fake training set."
    )
    parser.add_argument("--source", default=Path("data/defactify_all"), type=Path)
    parser.add_argument("--target", default=Path("data/mosaic_patch"), type=Path)
    parser.add_argument("--tile-size", default=224, type=int)
    parser.add_argument("--patch", default=224, type=int)
    parser.add_argument("--stride", default=224, type=int)
    parser.add_argument("--train-count", default=2000, type=int)
    parser.add_argument("--val-count", default=400, type=int)
    parser.add_argument("--test-count", default=200, type=int)
    parser.add_argument("--fake-threshold", default=0.60, type=float)
    parser.add_argument("--real-threshold", default=0.05, type=float)
    parser.add_argument("--fake-prefixes", default="", help="Optional comma-separated fake filename prefixes, e.g. sd21,sdxl.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--demo-limit", default=100, type=int, help="Number of test mosaics copied to demo_images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    prefixes = [item.strip() for item in args.fake_prefixes.split(",") if item.strip()]

    real_by_split = {split: list_images(args.source / split / "real") for split, _ in split_plan(args.train_count, args.val_count, args.test_count)}
    fake_by_split = {
        split: filter_fake_images(list_images(args.source / split / "fake"), prefixes)
        for split, _ in split_plan(args.train_count, args.val_count, args.test_count)
    }
    for split in real_by_split:
        if not real_by_split[split]:
            raise FileNotFoundError(f"No real images found for split '{split}'")
        if not fake_by_split[split]:
            raise FileNotFoundError(f"No fake images found for split '{split}' after prefix filtering")

    counts: dict[str, dict[str, int]] = {
        "train": {"real": 0, "fake": 0, "ignored": 0, "mosaics": 0},
        "val": {"real": 0, "fake": 0, "ignored": 0, "mosaics": 0},
        "test_seen": {"real": 0, "fake": 0, "ignored": 0, "mosaics": 0},
    }
    records: list[dict[str, str | int | float]] = []
    mosaics: list[dict[str, str | int]] = []

    for split, count in split_plan(args.train_count, args.val_count, args.test_count):
        for idx in tqdm(range(count), desc=f"{split} mosaics"):
            stem = f"{split}_{idx:06d}"
            image, mask, placements = build_mosaic(real_by_split[split], fake_by_split[split], rng, args.tile_size)

            mosaic_dir = args.target / "mosaics" / split
            image_path = mosaic_dir / f"{stem}_image.jpg"
            mask_path = mosaic_dir / f"{stem}_mask.png"
            overlay_path = mosaic_dir / f"{stem}_mask_overlay.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), image)
            cv2.imwrite(str(mask_path), mask)
            save_mask_overlay(image, mask, overlay_path)

            if split == "test_seen" and idx < args.demo_limit:
                demo_dir = args.target / "demo_images"
                demo_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(demo_dir / f"{stem}_image.jpg"), image)
                cv2.imwrite(str(demo_dir / f"{stem}_mask.png"), mask)
                save_mask_overlay(image, mask, demo_dir / f"{stem}_mask_overlay.jpg")

            export_patches(
                image=image,
                mask=mask,
                target_root=args.target,
                split=split,
                stem=stem,
                patch=args.patch,
                stride=args.stride,
                fake_threshold=args.fake_threshold,
                real_threshold=args.real_threshold,
                counters=counts,
                records=records,
            )
            counts[split]["mosaics"] += 1
            mosaics.append({"split": split, "stem": stem, "image": str(image_path), "mask": str(mask_path), **placements})

    manifest = {
        "source": str(args.source),
        "target": str(args.target),
        "tile_size": args.tile_size,
        "patch": args.patch,
        "stride": args.stride,
        "fake_threshold": args.fake_threshold,
        "real_threshold": args.real_threshold,
        "fake_prefixes": prefixes,
        "seed": args.seed,
        "counts": counts,
        "num_patch_records": len(records),
        "num_mosaics": len(mosaics),
    }
    args.target.mkdir(parents=True, exist_ok=True)
    (args.target / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.target / "patch_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.target / "mosaic_records.json").write_text(json.dumps(mosaics, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
