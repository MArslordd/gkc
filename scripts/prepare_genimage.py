from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_GENERATORS = ["Stable Diffusion V1.4", "Stable Diffusion V1.5", "BigGAN"]
UNSEEN_GENERATORS = ["Midjourney", "ADM", "GLIDE", "Wukong", "VQDM"]


def parse_generator_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    generators = [item.strip() for item in value.split(",") if item.strip()]
    if not generators:
        raise ValueError("Generator list cannot be empty.")
    return generators


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def find_generator_dir(source_root: Path, generator_name: str) -> Path:
    target = normalize_name(generator_name)
    candidates = [path for path in source_root.iterdir() if path.is_dir()]
    for path in candidates:
        if normalize_name(path.name) == target:
            return path
    for path in candidates:
        current = normalize_name(path.name)
        if target in current or current in target:
            return path
    available = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Generator '{generator_name}' not found under {source_root}. Available: {available}")


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(images: list[Path], start: int, limit: int | None) -> list[Path]:
    if limit is None:
        return images[start:]
    return images[start : start + limit]


def select_images_by_budget(images: list[Path], start: int, max_bytes: int) -> tuple[list[Path], int, int]:
    selected: list[Path] = []
    used_bytes = 0
    index = start
    while index < len(images):
        image = images[index]
        size = image.stat().st_size
        if selected and used_bytes + size > max_bytes:
            break
        if not selected and size > max_bytes:
            break
        selected.append(image)
        used_bytes += size
        index += 1
    return selected, used_bytes, index


def copy_or_link(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def export_images(
    images: list[Path],
    target_dir: Path,
    generator: str,
    label: str,
    mode: str,
    dry_run: bool,
) -> int:
    safe_generator = normalize_name(generator)
    for index, src in enumerate(images):
        dst_name = f"{safe_generator}_{label}_{index:06d}{src.suffix.lower()}"
        copy_or_link(src, target_dir / dst_name, mode=mode, dry_run=dry_run)
    return len(images)


def prepare_split(
    source_root: Path,
    target_root: Path,
    split: str,
    generators: list[str],
    source_split: str,
    start: int,
    limit_per_class: int | None,
    seed: int,
    mode: str,
    dry_run: bool,
) -> dict[str, int]:
    counts = {"real": 0, "fake": 0, "bytes": 0}
    rng = random.Random(seed)

    for generator in generators:
        gen_dir = find_generator_dir(source_root, generator)
        real_images = list_images(gen_dir / source_split / "nature")
        fake_images = list_images(gen_dir / source_split / "ai")
        rng.shuffle(real_images)
        rng.shuffle(fake_images)

        real_selected = select_images(real_images, start=start, limit=limit_per_class)
        fake_selected = select_images(fake_images, start=start, limit=limit_per_class)
        pair_count = min(len(real_selected), len(fake_selected))
        real_selected = real_selected[:pair_count]
        fake_selected = fake_selected[:pair_count]

        counts["real"] += export_images(
            real_selected,
            target_root / split / "real",
            generator,
            "real",
            mode,
            dry_run,
        )
        counts["bytes"] += sum(path.stat().st_size for path in real_selected)
        counts["fake"] += export_images(
            fake_selected,
            target_root / split / "fake",
            generator,
            "fake",
            mode,
            dry_run,
        )
        counts["bytes"] += sum(path.stat().st_size for path in fake_selected)

    return counts


def prepare_split_by_budget(
    source_root: Path,
    target_root: Path,
    split: str,
    generators: list[str],
    source_split: str,
    budget_bytes: int,
    cursors: dict[tuple[str, str, str], int],
    seed: int,
    mode: str,
    dry_run: bool,
) -> dict[str, int]:
    counts = {"real": 0, "fake": 0, "bytes": 0}
    rng = random.Random(seed)
    per_bucket_budget = budget_bytes // max(1, len(generators) * 2)

    for generator in generators:
        gen_dir = find_generator_dir(source_root, generator)
        label_jobs = [
            ("real", "nature", target_root / split / "real"),
            ("fake", "ai", target_root / split / "fake"),
        ]

        for label, source_label, target_dir in label_jobs:
            key = (generator, source_split, source_label)
            images = list_images(gen_dir / source_split / source_label)
            rng.shuffle(images)
            start = cursors.get(key, 0)
            selected, used_bytes, next_start = select_images_by_budget(images, start, per_bucket_budget)
            cursors[key] = next_start

            counts[label] += export_images(
                selected,
                target_dir,
                generator,
                label,
                mode,
                dry_run,
            )
            counts["bytes"] += used_bytes

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GenImage data for this project.")
    parser.add_argument("--source", required=True, type=Path, help="Raw GenImage root folder.")
    parser.add_argument("--target", default=Path("data/genimage_imagenet"), type=Path, help="Output dataset root.")
    parser.add_argument("--max-total-gb", default=None, type=float, help="Approximate total copied data budget.")
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--test-seen-ratio", default=0.1, type=float)
    parser.add_argument("--test-unseen-ratio", default=0.1, type=float)
    parser.add_argument("--train-generators", default=None, help="Comma-separated seen generators.")
    parser.add_argument("--unseen-generators", default=None, help="Comma-separated unseen generators.")
    parser.add_argument("--train-per-generator", default=10000, type=int)
    parser.add_argument("--val-per-generator", default=2000, type=int)
    parser.add_argument("--test-seen-per-generator", default=2000, type=int)
    parser.add_argument("--test-unseen-per-generator", default=2000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    target_root = args.target.resolve()
    train_generators = parse_generator_list(args.train_generators, TRAIN_GENERATORS)
    unseen_generators = parse_generator_list(args.unseen_generators, UNSEEN_GENERATORS)
    summary: dict[str, object] = {
        "source": str(source_root),
        "target": str(target_root),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "train_generators": train_generators,
        "unseen_generators": unseen_generators,
        "splits": {},
    }

    split_jobs = [
        (
            "train",
            train_generators,
            "train",
            0,
            args.train_per_generator,
        ),
        (
            "val",
            train_generators,
            "val",
            0,
            args.val_per_generator,
        ),
        (
            "test_seen",
            train_generators,
            "val",
            args.val_per_generator,
            args.test_seen_per_generator,
        ),
        (
            "test_unseen",
            unseen_generators,
            "val",
            0,
            args.test_unseen_per_generator,
        ),
    ]

    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    print(f"Mode: {args.mode}")
    if args.dry_run:
        print("Dry run: no files will be written")

    if args.max_total_gb is None:
        for split, generators, source_split, start, limit in split_jobs:
            counts = prepare_split(
                source_root=source_root,
                target_root=target_root,
                split=split,
                generators=generators,
                source_split=source_split,
                start=start,
                limit_per_class=limit,
                seed=args.seed,
                mode=args.mode,
                dry_run=args.dry_run,
            )
            summary["splits"][split] = counts
            print(f"{split}: real={counts['real']} fake={counts['fake']} bytes={counts['bytes']}")
    else:
        ratios = {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test_seen": args.test_seen_ratio,
            "test_unseen": args.test_unseen_ratio,
        }
        ratio_sum = sum(ratios.values())
        if ratio_sum <= 0:
            raise ValueError("Split ratios must sum to a positive number.")

        total_budget = int(args.max_total_gb * 1024**3)
        summary["max_total_gb"] = args.max_total_gb
        summary["split_ratios"] = ratios
        cursors: dict[tuple[str, str, str], int] = {}

        for split, generators, source_split, _start, _limit in split_jobs:
            split_budget = int(total_budget * ratios[split] / ratio_sum)
            counts = prepare_split_by_budget(
                source_root=source_root,
                target_root=target_root,
                split=split,
                generators=generators,
                source_split=source_split,
                budget_bytes=split_budget,
                cursors=cursors,
                seed=args.seed,
                mode=args.mode,
                dry_run=args.dry_run,
            )
            counts["budget_bytes"] = split_budget
            summary["splits"][split] = counts
            used_gb = counts["bytes"] / 1024**3
            budget_gb = split_budget / 1024**3
            print(
                f"{split}: real={counts['real']} fake={counts['fake']} "
                f"used={used_gb:.2f}GB budget={budget_gb:.2f}GB"
            )

    total_bytes = sum(split["bytes"] for split in summary["splits"].values())
    summary["total_bytes"] = total_bytes
    summary["total_gb"] = total_bytes / 1024**3
    print(f"total: {summary['total_gb']:.2f}GB")

    if not args.dry_run:
        target_root.mkdir(parents=True, exist_ok=True)
        manifest_path = target_root / "prepare_manifest.json"
        manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
