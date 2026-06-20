from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import MultiBranchImageDataset, collate_batch
from .metrics import binary_metrics
from .model import build_model
from .utils import describe_cuda, ensure_dir, get_device, load_config, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=True):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    labels: list[int] = []
    probs_fake: list[float] = []

    for batch in tqdm(loader, leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        srm = batch["srm"].to(device, non_blocking=True)
        fft = batch["fft"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(rgb, srm, fft)
                loss = criterion(logits, target)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.item()) * target.size(0)
        prob = torch.softmax(logits.detach(), dim=1)[:, 1]
        labels.extend(target.detach().cpu().tolist())
        probs_fake.extend(prob.cpu().tolist())

    metrics = binary_metrics(labels, probs_fake)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mobilenetv3_multibranch.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "cuda"))
    print(describe_cuda())

    out_dir = ensure_dir(cfg["output"]["dir"])
    ckpt_path = out_dir / cfg["output"]["checkpoint_name"]

    data_cfg = cfg["data"]
    train_ds = MultiBranchImageDataset(data_cfg["root"], "train", data_cfg["image_size"], data_cfg["class_names"])
    val_ds = MultiBranchImageDataset(data_cfg["root"], "val", data_cfg["image_size"], data_cfg["class_names"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True) and device.type == "cuda")

    best_f1 = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, cfg["train"].get("amp", True))
        val_metrics = run_epoch(model, val_loader, criterion, device, amp=cfg["train"].get("amp", True))
        scheduler.step()

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(f"train: {train_metrics}")
        print(f"val:   {val_metrics}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            stale_epochs = 0
            torch.save({"config": cfg, "model": model.state_dict(), "metrics": val_metrics}, ckpt_path)
            print(f"saved best checkpoint to {ckpt_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= cfg["train"].get("early_stop_patience", 6):
                print("early stopping")
                break

    save_json({"history": history, "best_f1": best_f1}, Path(out_dir) / "train_history.json")


if __name__ == "__main__":
    main()
