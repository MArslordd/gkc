from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _make_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    weights = None
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    feature_dim = net.classifier[0].in_features
    backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten(1))
    return backbone, feature_dim


class MultiBranchAIGCDetector(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        branch_mode: str = "full",
        hidden_dim: int = 512,
        dropout: float = 0.25,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.branch_mode = branch_mode
        self.rgb_branch, dim = _make_backbone(backbone, pretrained)

        if branch_mode == "rgb":
            branches = 1
            self.srm_branch = None
            self.fft_branch = None
        elif branch_mode == "full":
            branches = 3
            self.srm_branch, _ = _make_backbone(backbone, pretrained)
            self.fft_branch, _ = _make_backbone(backbone, pretrained)
        else:
            raise ValueError("branch_mode must be 'rgb' or 'full'")

        fused_dim = dim * branches
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, rgb: torch.Tensor, srm: torch.Tensor | None = None, fft: torch.Tensor | None = None) -> torch.Tensor:
        features = [self.rgb_branch(rgb)]
        if self.branch_mode == "full":
            if srm is None or fft is None:
                raise ValueError("srm and fft inputs are required when branch_mode='full'")
            features.append(self.srm_branch(srm))
            features.append(self.fft_branch(fft))
        return self.classifier(torch.cat(features, dim=1))


def build_model(config: dict) -> MultiBranchAIGCDetector:
    model_cfg = config["model"]
    return MultiBranchAIGCDetector(
        backbone=model_cfg.get("backbone", "mobilenet_v3_small"),
        pretrained=model_cfg.get("pretrained", True),
        branch_mode=model_cfg.get("branch_mode", "full"),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        dropout=model_cfg.get("dropout", 0.25),
        num_classes=len(config["data"].get("class_names", ["real", "fake"])),
    )
