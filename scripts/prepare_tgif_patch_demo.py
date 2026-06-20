from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.infer import prepare_image
from src.model import build_model
from src.utils import get_device, load_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SUBSETS = [
    "SD2-sp",
    "PS-sp",
    "flux1schnell-sp",
    "flux1dev-sp",
    "flux1filldev-sp",
    "sd2-sp",
]


@dataclass
class DemoSample:
    image_path: str
    mask_path: str
    subset: str
    split: str
    score: float | None
    inside_mean: float | None
    outside_mean: float | None
    heatmap_auc: float | None


def normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def find_dir(root: Path, candidates: list[str]) -> Path | None:
    if not root.exists():
        return None
    wanted = {normalize_name(name) for name in candidates}
    for path in root.rglob("*"):
        if path.is_dir() and normalize_name(path.name) in wanted:
            return path
    return None


def list_images(folder: Path) -> list[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_coco_id(path: Path) -> str:
    return path.name.split("_mask_")[0]


def parse_mask_token(path: Path) -> str | None:
    name = path.name
    marker = "_mask_"
    if marker not in name:
        return None
    prefix, tail = name.split(marker, 1)
    parts = tail.split(".png", 1)
    if not parts:
        return None
    return f"{prefix}_mask_{parts[0]}.png"


def build_mask_index(source_root: Path, split: str) -> dict[str, list[Path]]:
    mask_roots = [
        find_dir(source_root, ["masks"]),
        find_dir(source_root, ["masks-flux", "masks_flux"]),
        find_dir(source_root, ["masks-sd2", "masks_sd2"]),
        find_dir(source_root, ["masks-sdxl", "masks_sdxl"]),
    ]
    mask_index: dict[str, list[Path]] = {}
    for mask_root in [root for root in mask_roots if root is not None]:
        split_dir = find_dir(mask_root, [split, split.lower(), split.capitalize()])
        search_root = split_dir if split_dir is not None else mask_root
        for mask in list_images(search_root):
            coco_id = parse_coco_id(mask)
            mask_index.setdefault(coco_id, []).append(mask)
    return mask_index


def match_mask(image_path: Path, mask_index: dict[str, list[Path]]) -> Path | None:
    coco_id = parse_coco_id(image_path)
    candidates = mask_index.get(coco_id, [])
    if not candidates:
        return None

    token = parse_mask_token(image_path)
    if token is not None:
        normalized_token = normalize_name(token)
        for candidate in candidates:
            if normalized_token in normalize_name(candidate.name):
                return candidate

    non_ps = [path for path in candidates if "ps_mask" not in path.name]
    return non_ps[0] if non_ps else candidates[0]


def find_subset_split_dir(source_root: Path, subset: str, split: str) -> Path | None:
    subset_dir = find_dir(source_root, [subset])
    if subset_dir is None:
        return None
    split_dir = find_dir(subset_dir, [split, split.lower(), split.capitalize()])
    return split_dir if split_dir is not None else subset_dir


def compute_score_map(
    image_path: Path,
    cfg: dict,
    model: torch.nn.Module,
    device: torch.device,
    patch: int,
    stride: int,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = image.shape[:2]
    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    y_positions = list(range(0, max(1, h - patch + 1), stride))
    x_positions = list(range(0, max(1, w - patch + 1), stride))
    if y_positions[-1] != max(0, h - patch):
        y_positions.append(max(0, h - patch))
    if x_positions[-1] != max(0, w - patch):
        x_positions.append(max(0, w - patch))

    with tempfile.TemporaryDirectory() as tmp:
        temp_file = Path(tmp) / "patch.jpg"
        for y in y_positions:
            for x in x_positions:
                crop = image[y : y + patch, x : x + patch]
                if crop.shape[0] != patch or crop.shape[1] != patch:
                    crop = cv2.resize(crop, (patch, patch), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(temp_file), crop)
                rgb, srm, fft = prepare_image(str(temp_file), cfg["data"]["image_size"], device)
                fake_score = torch.softmax(model(rgb, srm, fft), dim=1)[0, 1].item()
                score_map[y : min(y + patch, h), x : min(x + patch, w)] += fake_score
                count_map[y : min(y + patch, h), x : min(x + patch, w)] += 1

    count_map[count_map == 0] = 1
    return score_map / count_map


def score_against_mask(score_map: np.ndarray, mask_path: Path) -> tuple[float, float, float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    mask = cv2.resize(mask, (score_map.shape[1], score_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 127
    if mask_bool.sum() == 0 or (~mask_bool).sum() == 0:
        return 0.0, 0.0, 0.0

    inside_mean = float(score_map[mask_bool].mean())
    outside_mean = float(score_map[~mask_bool].mean())
    try:
        auc = float(roc_auc_score(mask_bool.astype(np.uint8).ravel(), score_map.ravel()))
    except ValueError:
        auc = 0.0
    return inside_mean, outside_mean, auc


def save_visuals(image_path: Path, mask_path: Path, target_stem: Path, score_map: np.ndarray | None) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    target_stem.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_stem.with_name(target_stem.name + "_image" + image_path.suffix.lower()))
    shutil.copy2(mask_path, target_stem.with_name(target_stem.name + "_mask.png"))

    mask_color = np.zeros_like(image)
    mask_color[:, :, 2] = mask
    mask_overlay = cv2.addWeighted(image, 0.65, mask_color, 0.35, 0)
    cv2.imwrite(str(target_stem.with_name(target_stem.name + "_mask_overlay.jpg")), mask_overlay)

    if score_map is not None:
        heat = cv2.applyColorMap((np.clip(score_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
        cv2.imwrite(str(target_stem.with_name(target_stem.name + "_heatmap.jpg")), heat_overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 100 local AIGC patch demo samples from TGIF/TGIF2.")
    parser.add_argument("--source", required=True, type=Path, help="Raw TGIF/TGIF2 root folder.")
    parser.add_argument("--target", default=Path("data/patch_demo_tgif"), type=Path)
    parser.add_argument("--count", default=100, type=int)
    parser.add_argument("--split", default="testing")
    parser.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--candidate-multiplier", default=5, type=int)
    parser.add_argument("--patch", default=224, type=int)
    parser.add_argument("--stride", default=112, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_root = args.source.resolve()
    target_root = args.target.resolve()
    subsets = [item.strip() for item in args.subsets.split(",") if item.strip()]

    mask_index = build_mask_index(source_root, args.split)
    candidates: list[tuple[Path, Path, str]] = []
    for subset in subsets:
        subset_dir = find_subset_split_dir(source_root, subset, args.split)
        if subset_dir is None:
            continue
        for image_path in list_images(subset_dir):
            mask_path = match_mask(image_path, mask_index)
            if mask_path is not None:
                candidates.append((image_path, mask_path, subset))

    if not candidates:
        raise FileNotFoundError(
            f"No TGIF samples found under {source_root}. Check --split, --subsets, and downloaded mask folders."
        )

    rng.shuffle(candidates)
    max_candidates = min(len(candidates), max(args.count, args.count * args.candidate_multiplier))
    candidates = candidates[:max_candidates]

    cfg = None
    model = None
    device = None
    use_model = args.config is not None and args.checkpoint is not None
    if use_model:
        cfg = load_config(args.config)
        device = get_device(cfg.get("device", "cuda"))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = build_model(cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    samples: list[tuple[DemoSample, Path, Path, np.ndarray | None]] = []
    with torch.no_grad():
        for image_path, mask_path, subset in candidates:
            score_map = None
            inside_mean = None
            outside_mean = None
            heatmap_auc = None
            rank_score = None
            if use_model and cfg is not None and model is not None and device is not None:
                score_map = compute_score_map(image_path, cfg, model, device, args.patch, args.stride)
                inside_mean, outside_mean, heatmap_auc = score_against_mask(score_map, mask_path)
                rank_score = heatmap_auc + max(0.0, inside_mean - outside_mean)

            sample = DemoSample(
                image_path=str(image_path),
                mask_path=str(mask_path),
                subset=subset,
                split=args.split,
                score=rank_score,
                inside_mean=inside_mean,
                outside_mean=outside_mean,
                heatmap_auc=heatmap_auc,
            )
            samples.append((sample, image_path, mask_path, score_map))

    if use_model:
        samples.sort(key=lambda item: item[0].score if item[0].score is not None else -1, reverse=True)
    selected = samples[: args.count]

    images_dir = target_root / "images"
    metadata = []
    for index, (sample, image_path, mask_path, score_map) in enumerate(selected):
        stem = images_dir / f"patch_demo_{index:03d}"
        save_visuals(image_path, mask_path, stem, score_map)
        metadata.append(asdict(sample))

    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"candidates: {len(candidates)}")
    print(f"selected: {len(selected)}")
    print(f"saved: {target_root}")


if __name__ == "__main__":
    main()
