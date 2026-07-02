# Lightweight AIGC Image Detection

This project keeps two active experiment lines:

1. Full-image baseline on Defactify with all available generators.
2. Mosaic patch demonstration, where each 2x2 image contains three real tiles and one fake tile.

The mosaic route is the recommended patch innovation path because the fake region has a clean known mask and can be shown clearly with sliding-window heatmaps.

## Environment

```bash
pip install -r requirements.txt
python scripts/check_cuda.py
```

## Prepare Defactify

```bash
hf download \
  Rajarshi-Roy-research/Defactify_Image_Dataset \
  --repo-type dataset \
  --local-dir data/defactify_raw

python scripts/prepare_defactify.py \
  --source data/defactify_raw \
  --target data/defactify_all \
  --all-generators
```

## Full-Image Baseline

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --config configs/mobilenetv3_defactify_all.yaml

CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --config configs/mobilenetv3_defactify_all.yaml \
  --split test_seen \
  --checkpoint outputs_defactify_all/best_model.pt
```

## Quantize Full-Image Baseline

This command applies PyTorch dynamic INT8 quantization to the non-patch full-image baseline and evaluates FP32 vs INT8 on the same split:

```bash
python scripts/quantize_evaluate.py \
  --config configs/mobilenetv3_defactify_all.yaml \
  --checkpoint outputs_defactify_all/best_model.pt \
  --split test_seen \
  --output-dir outputs_defactify_all
```

## Mosaic Patch Demo

Create 2x2 local-AIGC mosaics and patch-level training samples:

```bash
python scripts/prepare_mosaic_patch_dataset.py \
  --source data/defactify_all \
  --target data/mosaic_patch \
  --tile-size 224 \
  --patch 224 \
  --stride 224 \
  --train-count 2000 \
  --val-count 400 \
  --test-count 200
```

Train the patch classifier:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --config configs/mobilenetv3_mosaic_patch.yaml
```

Evaluate it:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --config configs/mobilenetv3_mosaic_patch.yaml \
  --split test_seen \
  --checkpoint outputs_mosaic_patch/best_model.pt
```

Generate patch heatmaps on demo mosaics:

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.batch_heatmap \
  --config configs/mobilenetv3_mosaic_patch.yaml \
  --checkpoint outputs_mosaic_patch/best_model.pt \
  --input-dir data/mosaic_patch/demo_images \
  --output-dir outputs_mosaic_patch/visualizations \
  --pattern "*_image.jpg" \
  --limit 20 \
  --patch 224 \
  --stride 112
```

Compare heatmaps with `data/mosaic_patch/demo_images/*_mask_overlay.jpg`.

## Active Files

```text
configs/mobilenetv3_defactify_all.yaml    Full-image baseline config
configs/mobilenetv3_mosaic_patch.yaml     Mosaic patch config
scripts/prepare_defactify.py              Defactify exporter
scripts/prepare_mosaic_patch_dataset.py   Mosaic and patch dataset builder
scripts/batch_heatmap.py                  Batch sliding-window heatmaps
src/                                    Training, evaluation, inference, model code
docs/MOSAIC_PATCH_WORKFLOW.md             Detailed patch workflow
```
