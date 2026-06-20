from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def find_class_dir(root: Path, split: str, class_name: str) -> Path:
    candidates = [
        root / split / class_name,
        root / split / class_name.upper(),
        root / split / class_name.lower(),
        root / class_name,
        root / class_name.upper(),
        root / class_name.lower(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {class_name} folder for split '{split}' under {root}")


def copy_images(images: list[Path], target_dir: Path, prefix: str) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(images):
        dst = target_dir / f"{prefix}_{index:06d}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
    return len(images)


def split_train_val(images: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    val_count = int(len(images) * val_ratio)
    return images[val_count:], images[:val_count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CIFAKE data for this project.")
    parser.add_argument("--source", required=True, type=Path, help="Raw CIFAKE root folder.")
    parser.add_argument("--target", default=Path("data/cifake"), type=Path, help="Output dataset root.")
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--limit-train-per-class", default=None, type=int)
    parser.add_argument("--limit-test-per-class", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_root = args.source.resolve()
    target_root = args.target.resolve()

    train_real = list_images(find_class_dir(source_root, "train", "REAL"))
    train_fake = list_images(find_class_dir(source_root, "train", "FAKE"))
    test_real = list_images(find_class_dir(source_root, "test", "REAL"))
    test_fake = list_images(find_class_dir(source_root, "test", "FAKE"))

    rng.shuffle(train_real)
    rng.shuffle(train_fake)
    rng.shuffle(test_real)
    rng.shuffle(test_fake)

    if args.limit_train_per_class is not None:
        train_real = train_real[: args.limit_train_per_class]
        train_fake = train_fake[: args.limit_train_per_class]
    if args.limit_test_per_class is not None:
        test_real = test_real[: args.limit_test_per_class]
        test_fake = test_fake[: args.limit_test_per_class]

    train_real, val_real = split_train_val(train_real, args.val_ratio)
    train_fake, val_fake = split_train_val(train_fake, args.val_ratio)

    counts = {
        "train_real": copy_images(train_real, target_root / "train" / "real", "real"),
        "train_fake": copy_images(train_fake, target_root / "train" / "fake", "fake"),
        "val_real": copy_images(val_real, target_root / "val" / "real", "real"),
        "val_fake": copy_images(val_fake, target_root / "val" / "fake", "fake"),
        "test_seen_real": copy_images(test_real, target_root / "test_seen" / "real", "real"),
        "test_seen_fake": copy_images(test_fake, target_root / "test_seen" / "fake", "fake"),
        "test_unseen_real": copy_images(test_real, target_root / "test_unseen" / "real", "real"),
        "test_unseen_fake": copy_images(test_fake, target_root / "test_unseen" / "fake", "fake"),
    }

    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print("Note: CIFAKE has no cross-generator unseen split; test_unseen duplicates test_seen for pipeline compatibility.")


if __name__ == "__main__":
    main()
