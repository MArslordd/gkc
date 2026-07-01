# Patch 局部检测方案

## 核心问题

整图 AIGC 检测模型只学习 `image -> real/fake`。如果直接把它滑窗应用到局部 patch 上，可能得到热力图，但不能保证热力图一定对齐真实 AIGC 区域。

原因有两个：

1. 整图训练标签没有告诉模型“哪一块是 fake”。
2. 局部 AIGC 图像来自 inpainting/editing 模型，和整图 text-to-image fake 的痕迹不同。

因此，Patch 部分要稳定成功，需要从 DiffSeg30k 的 mask 构造 patch-level 训练数据。

## 展示数据：DiffSeg30k

DiffSeg30k 是 localized AIGC detection 数据集，提供：

```text
edited image
pixel-level mask
mask overlay
```

准备 100 张展示图：

```bash
python scripts/prepare_diffseg30k_patch_demo.py \
  --target data/patch_demo_diffseg30k \
  --count 100 \
  --split validation
```

输出：

```text
data/patch_demo_diffseg30k/images/
  diffseg30k_000_image.png
  diffseg30k_000_mask.png
  diffseg30k_000_mask_overlay.jpg
```

## 基础 Heatmap 展示

使用整图模型进行滑窗可视化：

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.batch_heatmap \
  --config configs/mobilenetv3_defactify_all.yaml \
  --checkpoint outputs_defactify_all/best_model.pt \
  --input-dir data/patch_demo_diffseg30k/images \
  --output-dir outputs_defactify_all/visualizations/patch_demo \
  --pattern "*_image.png" \
  --limit 20 \
  --patch 224 \
  --stride 112
```

这一版适合作为初步可视化。如果效果不稳定，这是预期现象，因为模型没有接受 patch-level 监督。

## 保证 Patch 成功的训练方法

从 DiffSeg30k mask 切出 patch：

```bash
python scripts/prepare_diffseg30k_patch_dataset.py \
  --source data/patch_demo_diffseg30k/images \
  --target data/diffseg30k_patch \
  --patch 224 \
  --stride 112 \
  --fake-threshold 0.30 \
  --real-threshold 0.02
```

规则：

```text
mask 覆盖率 >= 30% -> fake patch
mask 覆盖率 <= 2%  -> real patch
中间区域忽略
```

训练 patch-level full 模型：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/mobilenetv3_diffseg30k_patch.yaml
```

生成 patch-level heatmap：

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.batch_heatmap \
  --config configs/mobilenetv3_diffseg30k_patch.yaml \
  --checkpoint outputs_diffseg30k_patch/best_model.pt \
  --input-dir data/patch_demo_diffseg30k/images \
  --output-dir outputs_diffseg30k_patch/visualizations/patch_demo \
  --pattern "*_image.png" \
  --limit 20 \
  --patch 224 \
  --stride 112
```

这条路线能显著提高 Patch 成功率，因为训练目标和展示目标一致。

## 报告表述

可以这样描述：

```text
为提高局部 AIGC 编辑区域的可解释性，本文在整图检测器基础上进一步构造 patch-level 训练任务。根据 DiffSeg30k 提供的像素级编辑 mask，将 mask 覆盖率较高的 patch 标记为 fake，将几乎不含编辑区域的 patch 标记为 real，忽略边界过渡 patch。训练后的 patch-level 检测器通过滑窗方式对整图逐 patch 预测 fake score，并将重叠区域分数聚合为热力图。相比直接使用整图分类模型滑窗，该方法引入了局部监督，因此更适合定位局部 AIGC 编辑区域。
```

## 展示建议

每个样例展示三列：

```text
原图 | Ground-truth mask overlay | Patch fake-score heatmap
```

如果需要更细热力图：

```bash
--patch 128 --stride 64
```

如果需要更快：

```bash
--patch 224 --stride 224
```
