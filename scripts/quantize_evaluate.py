from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import MultiBranchImageDataset, collate_batch
from src.metrics import binary_metrics
from src.model import build_model
from src.utils import load_config, save_json


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    probs_fake: list[float] = []

    for batch in tqdm(loader, leave=False):
        rgb = batch["rgb"].to(device)
        srm = batch["srm"].to(device)
        fft = batch["fft"].to(device)
        target = batch["label"].to(device)
        logits = model(rgb, srm, fft)
        prob = torch.softmax(logits, dim=1)[:, 1]
        labels.extend(target.cpu().tolist())
        probs_fake.extend(prob.cpu().tolist())

    return binary_metrics(labels, probs_fake)


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamically quantize and evaluate the full-image detector.")
    parser.add_argument("--config", default="configs/mobilenetv3_defactify_all.yaml")
    parser.add_argument("--checkpoint", default="outputs_defactify_all/best_model.pt")
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--limit-samples", default=None, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir or Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    fp32_model = build_model(cfg).to(device)
    fp32_model.load_state_dict(checkpoint["model"])
    fp32_model.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.eval()

    data_cfg = cfg["data"]
    dataset = MultiBranchImageDataset(data_cfg["root"], args.split, data_cfg["image_size"], data_cfg["class_names"])
    if args.limit_samples is not None:
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else data_cfg["num_workers"],
        pin_memory=False,
        collate_fn=collate_batch,
    )

    print("Evaluating FP32 model on CPU...")
    fp32_metrics = evaluate_model(fp32_model, loader, device)
    print(fp32_metrics)

    print("Evaluating dynamic INT8 model on CPU...")
    int8_metrics = evaluate_model(quantized_model, loader, device)
    print(int8_metrics)

    quantized_path = output_dir / "best_model_dynamic_int8_linear.pt"
    torch.save(
        {
            "config": cfg,
            "quantization": {
                "type": "torch_dynamic",
                "dtype": "qint8",
                "modules": ["Linear"],
                "note": "MobileNetV3 convolution branches remain FP32; classifier Linear layers are INT8 dynamic quantized.",
            },
            "model": quantized_model.state_dict(),
            "metrics": int8_metrics,
        },
        quantized_path,
    )

    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "quantized_checkpoint": str(quantized_path),
        "quantization": {
            "type": "torch_dynamic",
            "dtype": "qint8",
            "modules": ["Linear"],
        },
        "fp32_metrics": fp32_metrics,
        "int8_metrics": int8_metrics,
        "metric_delta_int8_minus_fp32": {
            key: int8_metrics[key] - fp32_metrics[key]
            for key in fp32_metrics
            if key in int8_metrics
        },
        "checkpoint_size_mb": {
            "fp32_source": model_size_mb(Path(args.checkpoint)),
            "int8_dynamic": model_size_mb(quantized_path),
        },
    }
    report_path = output_dir / f"quant_eval_{args.split}.json"
    save_json(report, report_path)
    print(f"saved quantized checkpoint: {quantized_path}")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
