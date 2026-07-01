from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def quadrant_slices(height: int, width: int) -> list[tuple[int, slice, slice]]:
    mid_y = height // 2
    mid_x = width // 2
    return [
        (0, slice(0, mid_y), slice(0, mid_x)),
        (1, slice(0, mid_y), slice(mid_x, width)),
        (2, slice(mid_y, height), slice(0, mid_x)),
        (3, slice(mid_y, height), slice(mid_x, width)),
    ]


def score_heatmap(image_bgr: np.ndarray, mode: str) -> np.ndarray:
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    if mode == "red":
        return r
    if mode == "red_minus_blue":
        return r - b
    if mode == "jet":
        return r + 0.5 * g - b
    raise ValueError(f"Unsupported score mode: {mode}")


def load_score_map(heatmap_path: Path, mode: str) -> tuple[np.ndarray, str]:
    score_path = heatmap_path.with_name(heatmap_path.name.replace("_heatmap.jpg", "_score.npy"))
    if score_path.exists():
        return np.load(score_path).astype(np.float32), str(score_path)

    score_png_path = heatmap_path.with_name(heatmap_path.name.replace("_heatmap.jpg", "_score.png"))
    if score_png_path.exists():
        score_png = cv2.imread(str(score_png_path), cv2.IMREAD_GRAYSCALE)
        if score_png is None:
            raise ValueError(f"Failed to read score map: {score_png_path}")
        return score_png.astype(np.float32) / 255.0, str(score_png_path)

    heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_COLOR)
    if heatmap is None:
        raise ValueError(f"Failed to read heatmap: {heatmap_path}")
    return score_heatmap(heatmap, mode), str(heatmap_path)


def quadrant_means(score: np.ndarray) -> list[float]:
    h, w = score.shape[:2]
    values = []
    for _, ys, xs in quadrant_slices(h, w):
        values.append(float(score[ys, xs].mean()))
    return values


def mask_quadrant(mask: np.ndarray) -> tuple[int, list[float]]:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = mask > 0
    h, w = binary.shape[:2]
    coverages = []
    for _, ys, xs in quadrant_slices(h, w):
        coverages.append(float(binary[ys, xs].mean()))
    return int(np.argmax(coverages)), coverages


def stem_from_heatmap(path: Path) -> str:
    name = path.name
    if name.endswith("_heatmap.jpg"):
        return name[: -len("_heatmap.jpg")]
    if name.endswith("_heatmap.png"):
        return name[: -len("_heatmap.png")]
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mosaic heatmaps with generated fake-region masks.")
    parser.add_argument("--demo-dir", default=Path("data/mosaic_patch/demo_images"), type=Path)
    parser.add_argument("--heatmap-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_heatmap.jpg")
    parser.add_argument("--score-mode", default="jet", choices=["jet", "red", "red_minus_blue"])
    parser.add_argument("--output-json", default=None, type=Path)
    parser.add_argument("--output-csv", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heatmap_paths = sorted(args.heatmap_dir.glob(args.pattern))
    if not heatmap_paths:
        raise FileNotFoundError(f"No heatmaps matched: {args.heatmap_dir / args.pattern}")

    records = []
    for heatmap_path in heatmap_paths:
        stem = stem_from_heatmap(heatmap_path)
        mask_path = args.demo_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            records.append({"stem": stem, "heatmap": str(heatmap_path), "missing_mask": True, "correct": False})
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        score_map, score_source = load_score_map(heatmap_path, args.score_mode)
        if score_map.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (score_map.shape[1], score_map.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_quad, gt_coverages = mask_quadrant(mask)
        heat_scores = quadrant_means(score_map)
        pred_quad = int(np.argmax(heat_scores))
        correct = pred_quad == gt_quad

        records.append(
            {
                "stem": stem,
                "heatmap": str(heatmap_path),
                "score_source": score_source,
                "mask": str(mask_path),
                "gt_quadrant": gt_quad,
                "pred_quadrant": pred_quad,
                "correct": correct,
                "gt_coverages": gt_coverages,
                "heat_scores": heat_scores,
            }
        )

    valid = [record for record in records if not record.get("missing_mask")]
    correct_count = sum(1 for record in valid if record["correct"])
    accuracy = correct_count / len(valid) if valid else 0.0
    summary = {
        "demo_dir": str(args.demo_dir),
        "heatmap_dir": str(args.heatmap_dir),
        "pattern": args.pattern,
        "score_mode": args.score_mode,
        "total": len(records),
        "valid": len(valid),
        "correct": correct_count,
        "accuracy": accuracy,
        "records": records,
    }

    if args.output_json is None:
        args.output_json = args.heatmap_dir / "mosaic_heatmap_eval.json"
    if args.output_csv is None:
        args.output_csv = args.heatmap_dir / "mosaic_heatmap_eval.csv"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stem",
                "gt_quadrant",
                "pred_quadrant",
                "correct",
                "gt_q0",
                "gt_q1",
                "gt_q2",
                "gt_q3",
                "heat_q0",
                "heat_q1",
                "heat_q2",
                "heat_q3",
            ]
        )
        for record in records:
            if record.get("missing_mask"):
                writer.writerow([record["stem"], "missing_mask", "", False, "", "", "", "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    record["stem"],
                    record["gt_quadrant"],
                    record["pred_quadrant"],
                    record["correct"],
                    *[f"{value:.6f}" for value in record["gt_coverages"]],
                    *[f"{value:.6f}" for value in record["heat_scores"]],
                ]
            )

    print(f"valid: {len(valid)} / {len(records)}")
    print(f"correct: {correct_count}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"saved {args.output_json}")
    print(f"saved {args.output_csv}")


if __name__ == "__main__":
    main()
