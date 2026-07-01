# Experiment Roadmap

## Stage 1: Full-Image Baseline

Prepare Defactify with all generators:

```bash
python scripts/prepare_defactify.py \
  --source data/defactify_raw \
  --target data/defactify_all \
  --all-generators
```

Train:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --config configs/mobilenetv3_defactify_all.yaml
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --config configs/mobilenetv3_defactify_all.yaml \
  --split test_seen \
  --checkpoint outputs_defactify_all/best_model.pt
```

## Stage 2: Mosaic Patch Dataset

Create 2x2 images with three real tiles and one fake tile:

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

## Stage 3: Patch Classifier

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --config configs/mobilenetv3_mosaic_patch.yaml

CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --config configs/mobilenetv3_mosaic_patch.yaml \
  --split test_seen \
  --checkpoint outputs_mosaic_patch/best_model.pt
```

## Stage 4: Patch Heatmap Demo

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

Use `data/mosaic_patch/demo_images/*_mask_overlay.jpg` as the ground-truth visual reference.
