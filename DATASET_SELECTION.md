# Dataset Selection

The project now uses Defactify as the single active data source.

## Why Defactify

Defactify can be downloaded with one Hugging Face command, contains real and generated images, and covers multiple fake sources. This makes it suitable for:

- a full-image baseline trained with all generators;
- a controlled local-AIGC mosaic demo;
- repeatable experiments on the server without Baidu Netdisk or manual archive stitching.

## Download

```bash
hf download \
  Rajarshi-Roy-research/Defactify_Image_Dataset \
  --repo-type dataset \
  --local-dir data/defactify_raw
```

## Export

```bash
python scripts/prepare_defactify.py \
  --source data/defactify_raw \
  --target data/defactify_all \
  --all-generators
```

## Active Derived Datasets

```text
data/defactify_all       Full-image real/fake training data
data/mosaic_patch        2x2 local-AIGC mosaics and patch-level data
```

Older CIFAKE, GenImage, TGIF, and DiffSeg30k paths are no longer part of the active workflow.
