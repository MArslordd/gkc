from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import MultiBranchImageDataset, collate_batch
from .metrics import binary_metrics
from .model import build_model
from .utils import get_device, load_config, save_json


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobilenetv3_defactify_all.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--split", default="test_seen")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_cfg = cfg["data"]
    ds = MultiBranchImageDataset(data_cfg["root"], args.split, data_cfg["image_size"], data_cfg["class_names"])
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    labels: list[int] = []
    probs_fake: list[float] = []
    rows = []
    for batch in tqdm(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        srm = batch["srm"].to(device, non_blocking=True)
        fft = batch["fft"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        logits = model(rgb, srm, fft)
        prob = torch.softmax(logits, dim=1)[:, 1]
        labels.extend(target.cpu().tolist())
        probs_fake.extend(prob.cpu().tolist())
        rows.extend({"path": p, "label": int(y), "fake_score": float(s)} for p, y, s in zip(batch["path"], target.cpu(), prob.cpu()))

    metrics = binary_metrics(labels, probs_fake)
    out = {"split": args.split, "metrics": metrics, "predictions": rows}
    out_path = Path(cfg["output"]["dir"]) / f"eval_{args.split}.json"
    save_json(out, out_path)
    print(metrics)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
