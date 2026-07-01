# Mosaic Patch Workflow

This workflow is designed for the local-AIGC demonstration: one image is built from four tiles, with three real tiles and one fake tile. The fake tile is the ground-truth local AIGC region.

## Data Construction

Input data should follow the existing ImageFolder layout:

```text
data/defactify_all/
  train/real
  train/fake
  val/real
  val/fake
  test_seen/real
  test_seen/fake
```

Generate mosaic images, masks, and patch-level classification samples:

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

The output contains:

```text
data/mosaic_patch/
  train/real
  train/fake
  val/real
  val/fake
  test_seen/real
  test_seen/fake
  mosaics/train
  mosaics/val
  mosaics/test_seen
  demo_images
```

## Training

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --config configs/mobilenetv3_mosaic_patch.yaml
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --config configs/mobilenetv3_mosaic_patch.yaml \
  --split test_seen \
  --checkpoint outputs_mosaic_patch/best_model.pt
```

## Heatmap Demo

Use the patch classifier on full mosaic images:

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

For the cleanest demonstration, compare each generated heatmap with the matching `*_mask_overlay.jpg` in `data/mosaic_patch/demo_images`.
