# Project Summary

This project implements a lightweight AIGC image detection baseline and a controlled local-AIGC patch demonstration.

## Active Baseline

The full-image baseline uses Defactify with all available fake generators merged into the training and evaluation data.

```text
Dataset: Defactify all-generators
Task: real/fake image classification
Backbone: MobileNetV3-Small
Input: RGB + SRM residual + FFT spectrum
Output: real/fake probability
Config: configs/mobilenetv3_defactify_all.yaml
```

This baseline is kept as the main comparison point for whole-image detection.

## Patch Innovation

The patch demonstration uses synthetic local-AIGC mosaics. Each image is a 2x2 grid made from three real image tiles and one fake image tile.

```text
real + real + real + fake
-> 2x2 mosaic image
-> exact fake-region mask
-> patch-level real/fake samples
-> sliding-window heatmap
```

This avoids the unstable open-generator setting where a classifier trained on some generators fails on unseen ones. The demonstration target is clearer: given a local patch, decide whether that patch is fake.

## Active Workflow

```text
data/defactify_raw
-> scripts/prepare_defactify.py
-> data/defactify_all
-> full-image baseline
-> scripts/prepare_mosaic_patch_dataset.py
-> data/mosaic_patch
-> patch classifier
-> heatmap visualization
```

## Success Criteria

- The full-image model provides the baseline real/fake result.
- The mosaic patch model should highlight the fake tile in `data/mosaic_patch/demo_images`.
- The generated heatmap should align with the saved `*_mask_overlay.jpg`.

## Active Outputs

```text
outputs_defactify_all
outputs_mosaic_patch
```
