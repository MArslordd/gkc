# Project Structure

```text
gkc/
  configs/
    mobilenetv3_defactify_all.yaml
    mobilenetv3_mosaic_patch.yaml
  docs/
    MOSAIC_PATCH_WORKFLOW.md
    EXPERIMENT_ROADMAP.md
    PATCH_INNOVATION.md
  scripts/
    check_cuda.py
    prepare_defactify.py
    prepare_mosaic_patch_dataset.py
    batch_heatmap.py
  src/
    datasets.py
    train.py
    evaluate.py
    infer.py
    heatmap.py
    export_onnx.py
    model.py
    srm.py
    fft_features.py
    metrics.py
    utils.py
  reports/
```

## Current Main Route

1. Download Defactify.
2. Export `data/defactify_all` with all generators.
3. Train the full-image baseline with `configs/mobilenetv3_defactify_all.yaml`.
4. Build local-AIGC 2x2 mosaics with `scripts/prepare_mosaic_patch_dataset.py`.
5. Train the patch classifier with `configs/mobilenetv3_mosaic_patch.yaml`.
6. Generate heatmaps on `data/mosaic_patch/demo_images`.

## Server Data To Keep

```text
data/defactify_raw
data/defactify_all
data/mosaic_patch
outputs_defactify_all
outputs_mosaic_patch
```

## Server Data That Can Be Deleted

```text
data/cifake_raw
data/cifake
data/GenImage_raw
data/genimage_10gb
data/genimage_glide
data/patch_demo_diffseg30k
data/diffseg30k_patch
data/TGIF_raw
outputs_cifake
outputs_defactify
outputs_diffseg30k_patch
outputs_genimage_10gb
```
