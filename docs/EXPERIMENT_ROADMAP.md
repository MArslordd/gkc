# 实验路线

## 1. 快速 Baseline：CIFAKE

CIFAKE 用于验证训练链路是否可用。它的原始分辨率是 `32x32`，结果不作为最终泛化结论。

```bash
python scripts/prepare_cifake.py --source data/cifake_raw --target data/cifake
python -m src.train --config configs/mobilenetv3_cifake.yaml
python -m src.evaluate --config configs/mobilenetv3_cifake.yaml --split test_seen --checkpoint outputs_cifake/best_model.pt
```

已有 baseline：

```text
Accuracy  = 98.30%
Precision = 97.44%
Recall    = 99.21%
F1-score  = 98.32%
AUC       = 99.66%
```

## 2. 主实验：Defactify 全生成器训练

Defactify / MS COCOAI 覆盖 SD2.1、SDXL、SD3、DALL-E 3、Midjourney 等生成器，下载和整理比 GenImage 更稳定。当前推荐把全部生成器都纳入训练，以提升 Patch 展示模型对不同生成痕迹的覆盖。

```bash
python scripts/prepare_defactify.py \
  --source data/defactify_raw \
  --target data/defactify_all \
  --all-generators
```

训练 full 模型：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/mobilenetv3_defactify_all.yaml
```

评估：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.evaluate \
  --config configs/mobilenetv3_defactify_all.yaml \
  --split test_seen \
  --checkpoint outputs_defactify_all/best_model.pt
```

说明：在 `defactify_all` 中，`test_seen` 表示全生成器测试集，不再表示 seen/unseen 划分。

## 3. 泛化对照：Defactify Seen/Unseen

如果需要说明跨生成器泛化困难，可以保留 seen/unseen 实验：

```bash
python scripts/prepare_defactify.py \
  --source data/defactify_raw \
  --target data/defactify_seen_unseen
```

默认划分：

```text
train/val/test_seen: SD2.1, SDXL, SD3
test_unseen: DALL-E 3, Midjourney
```

这个实验已经显示：即便使用 full 模型，未见生成器上也可能显著失效。这个结果可以作为“整图分类模型不等价于局部泛化定位模型”的论据。

## 4. Patch 展示：DiffSeg30k

准备 100 张局部 AIGC 展示图：

```bash
python scripts/prepare_diffseg30k_patch_demo.py \
  --target data/patch_demo_diffseg30k \
  --count 100 \
  --split validation
```

批量生成 heatmap：

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

展示时使用三张图：

```text
diffseg30k_000_image.png          原图
diffseg30k_000_mask_overlay.jpg   真实局部编辑区域
diffseg30k_000_heatmap.jpg        模型 Patch 热力图
```

## 5. 保证 Patch 成功：Patch-Level 训练

只用整图 real/fake 模型直接滑窗，不能保证定位效果。要让 Patch 部分稳定成功，应使用 DiffSeg30k mask 构造 patch-level 数据：

```bash
python scripts/prepare_diffseg30k_patch_dataset.py \
  --source data/patch_demo_diffseg30k/images \
  --target data/diffseg30k_patch \
  --patch 224 \
  --stride 112 \
  --fake-threshold 0.30 \
  --real-threshold 0.02
```

训练 Patch 分类器：

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/mobilenetv3_diffseg30k_patch.yaml
```

再生成 heatmap：

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

这一版才是 Patch 创新点的可靠实验路线。
