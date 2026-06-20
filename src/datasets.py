from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .fft_features import compute_fft_spectrum
from .srm import compute_srm_residual


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transforms(image_size: int, train: bool) -> Callable[[Image.Image], torch.Tensor]:
    aug = []
    if train:
        aug.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04, hue=0.01),
            ]
        )
    else:
        aug.extend(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
            ]
        )
    aug.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(aug)


class MultiBranchImageDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, class_names: list[str]) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.class_names = class_names
        self.transform = build_transforms(image_size, train=split == "train")
        self.samples = self._find_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.split_dir}. Expected folders like {self.split_dir / class_names[0]}."
            )

    def _find_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(self.class_names):
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, label = self.samples[index]
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        rgb = Image.fromarray(image_rgb)
        srm = Image.fromarray(compute_srm_residual(image_rgb))
        fft = Image.fromarray(compute_fft_spectrum(image_rgb))

        return {
            "rgb": self.transform(rgb),
            "srm": self.transform(srm),
            "fft": self.transform(fft),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "srm": torch.stack([item["srm"] for item in batch]),
        "fft": torch.stack([item["fft"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "path": [str(item["path"]) for item in batch],
    }
