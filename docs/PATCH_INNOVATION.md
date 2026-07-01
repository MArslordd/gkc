# Patch Innovation

The current patch idea is not to rely on cross-generator generalization. Instead, we construct images where only a local region is fake.

## Method

Each demo image is a 2x2 mosaic:

```text
real | fake
real | real
```

The fake tile position is randomized. A binary mask is saved for every mosaic, so the demonstration has a clear ground truth.

## Why This Works Better For The Demo

The previous open-set generator setting made the classifier fail on unseen generators. In the mosaic setting, the model is trained and visualized on patch-level local real/fake evidence. The goal becomes:

```text
Given a local patch, decide whether that patch is AIGC.
```

This directly supports the claim: if only part of an image is AIGC, sliding-window patch detection can localize that region.

## Commands

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

CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --config configs/mobilenetv3_mosaic_patch.yaml

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

## Success Criteria

- Heatmaps should be strongest on the fake tile.
- Real tiles should remain relatively cool.
- The generated heatmap should visually align with `*_mask_overlay.jpg`.
