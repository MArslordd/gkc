from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


SOURCE_NAMES = {
    0: "real",
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "midjourney",
}

ALL_FAKE_SOURCES = {1, 2, 3, 4, 5}


def parse_int_list(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def find_parquet_files(source: Path) -> dict[str, list[str]]:
    files = sorted(source.rglob("*.parquet"))
    data_files: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for path in files:
        name = path.as_posix().lower()
        if "train" in name:
            data_files["train"].append(str(path))
        elif "validation" in name or "/val" in name or "val-" in name:
            data_files["validation"].append(str(path))
        elif "test" in name:
            data_files["test"].append(str(path))
    return {split: paths for split, paths in data_files.items() if paths}


def get_value(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    available = ", ".join(row.keys())
    raise KeyError(f"None of {names} found. Available columns: {available}")


def get_label_a(row: dict[str, Any]) -> int:
    return int(get_value(row, ["Label_A", "label_a", "label", "Label"]))


def get_label_b(row: dict[str, Any]) -> int:
    return int(get_value(row, ["Label_B", "label_b", "source", "Source"]))


def get_image(row: dict[str, Any]) -> Image.Image:
    image = get_value(row, ["Image", "image", "img"])
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and "bytes" in image:
        from io import BytesIO

        return Image.open(BytesIO(image["bytes"]))
    raise TypeError(f"Unsupported image value type: {type(image)!r}")


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def should_keep_row(label_a: int, label_b: int, allowed_fake_sources: set[int] | None) -> bool:
    if label_a == 0:
        return True
    return allowed_fake_sources is None or label_b in allowed_fake_sources


def export_split_streaming(
    dataset,
    target_root: Path,
    output_split: str,
    allowed_fake_sources: set[int] | None,
    limit_per_class: int | None,
    max_scan: int | None,
) -> dict[str, int]:
    counts = {"real": 0, "fake": 0, "scanned": 0, "skipped": 0}
    progress = tqdm(dataset, desc=output_split, unit="row")

    for row in progress:
        if max_scan is not None and counts["scanned"] >= max_scan:
            break
        counts["scanned"] += 1

        label_a = get_label_a(row)
        label_b = get_label_b(row)
        if not should_keep_row(label_a, label_b, allowed_fake_sources):
            counts["skipped"] += 1
            continue

        label_name = "real" if label_a == 0 else "fake"
        if limit_per_class is not None and counts[label_name] >= limit_per_class:
            other = "fake" if label_name == "real" else "real"
            if counts[other] >= limit_per_class:
                break
            continue

        source = SOURCE_NAMES.get(label_b, f"source{label_b}")
        output_path = (
            target_root
            / output_split
            / label_name
            / f"{source}_{counts[label_name]:06d}.jpg"
        )
        save_image(get_image(row), output_path)
        counts[label_name] += 1
        progress.set_postfix(real=counts["real"], fake=counts["fake"], skipped=counts["skipped"])

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Defactify Image Dataset for this project.")
    parser.add_argument("--source", default="data/defactify_raw", help="Downloaded HF dataset folder or HF dataset id.")
    parser.add_argument("--target", default=Path("data/defactify_seen_unseen"), type=Path)
    parser.add_argument("--seen-sources", default="1,2,3", help="Label_B ids used for train/val/test_seen fake.")
    parser.add_argument("--unseen-sources", default="4,5", help="Label_B ids used for test_unseen fake.")
    parser.add_argument(
        "--all-generators",
        action="store_true",
        help="Use all fake sources for train/val/test_seen/test_unseen. This disables seen/unseen filtering.",
    )
    parser.add_argument("--limit-train-per-class", default=20000, type=int)
    parser.add_argument("--limit-val-per-class", default=4000, type=int)
    parser.add_argument("--limit-test-per-class", default=4000, type=int)
    parser.add_argument("--max-scan-per-split", default=None, type=int)
    return parser.parse_args()


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Please install datasets first: pip install datasets") from exc

    args = parse_args()
    source_path = Path(args.source)
    source = str(source_path.resolve()) if source_path.exists() else args.source
    target_root = args.target
    seen_sources = parse_int_list(args.seen_sources)
    unseen_sources = parse_int_list(args.unseen_sources)
    if args.all_generators:
        seen_sources = set(ALL_FAKE_SOURCES)
        unseen_sources = set(ALL_FAKE_SOURCES)

    print(f"Source: {source}")
    print(f"Target: {target_root.resolve()}")
    print(f"Seen fake sources: {[SOURCE_NAMES.get(i, i) for i in sorted(seen_sources)]}")
    print(f"Unseen fake sources: {[SOURCE_NAMES.get(i, i) for i in sorted(unseen_sources)]}")

    if source_path.exists():
        data_files = find_parquet_files(source_path)
        if data_files:
            print("Loading local parquet files:")
            for split, files in data_files.items():
                print(f"  {split}: {len(files)} files")
            ds = load_dataset("parquet", data_files=data_files, streaming=True)
        else:
            ds = load_dataset(source, streaming=True)
    else:
        ds = load_dataset(source, streaming=True)

    summary = {
        "source": source,
        "target": str(target_root.resolve()),
        "seen_sources": sorted(seen_sources),
        "unseen_sources": sorted(unseen_sources),
        "all_generators": args.all_generators,
        "limits": {
            "train_per_class": args.limit_train_per_class,
            "val_per_class": args.limit_val_per_class,
            "test_per_class": args.limit_test_per_class,
            "max_scan_per_split": args.max_scan_per_split,
        },
        "splits": {},
    }

    if args.all_generators:
        split_plan = [
            ("train", "train", ALL_FAKE_SOURCES, args.limit_train_per_class),
            ("validation", "val", ALL_FAKE_SOURCES, args.limit_val_per_class),
            ("test", "test_seen", ALL_FAKE_SOURCES, args.limit_test_per_class),
        ]
    else:
        split_plan = [
            ("train", "train", seen_sources, args.limit_train_per_class),
            ("validation", "val", seen_sources, args.limit_val_per_class),
            ("test", "test_seen", seen_sources, args.limit_test_per_class),
            ("test", "test_unseen", unseen_sources, args.limit_test_per_class),
        ]
    for source_split, output_split, allowed_sources, limit in split_plan:
        if source_split not in ds:
            raise KeyError(f"Missing split '{source_split}'. Available splits: {list(ds.keys())}")
        summary["splits"][output_split] = export_split_streaming(
            dataset=ds[source_split],
            target_root=target_root,
            output_split=output_split,
            allowed_fake_sources=allowed_sources,
            limit_per_class=limit,
            max_scan=args.max_scan_per_split,
        )

    target_root.mkdir(parents=True, exist_ok=True)
    manifest_path = target_root / "prepare_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
