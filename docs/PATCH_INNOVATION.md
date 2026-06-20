# Patch 切割创新点

## 动机

普通 AIGC 检测模型通常只输出整图级别的 `real/fake` 结果。这种方式适合作为二分类指标，但解释性较弱：当模型判断一张图为 fake 时，用户不知道模型依据来自图像的哪一部分。

Patch 切割模块的目标是把整图检测扩展成局部区域分析，让模型输出一张 fake score 热力图，用来观察哪些区域更可疑。

## 基本方法

当前 `src/heatmap.py` 已实现滑动窗口式 Patch 检测：

```text
input image
-> crop patches with patch size and stride
-> run model on each patch
-> get fake_score for each patch
-> accumulate patch scores on original image canvas
-> average overlapping regions
-> generate heatmap overlay
```

默认参数：

```text
patch size: 224
stride: 112
```

这意味着相邻 patch 有 50% 重叠，可以减少边界处的评分突变。

## 创新点表述

可以在报告里把它描述为：

```text
本文在整图 AIGC 二分类检测的基础上，引入 Patch-level fake score aggregation 方法。该方法通过滑动窗口将输入图像切分为多个局部 patch，并对每个 patch 独立预测 fake score，再将重叠区域的预测分数进行空间聚合，生成局部 AIGC 可疑区域热力图。相比仅输出整图分类结果，该方法能够提供更细粒度的可解释性，有助于观察模型关注的生成痕迹区域，并为后续局部篡改检测和人机审核提供依据。
```

## 后续可以增强的方向

### 1. 多尺度 Patch

目前只使用固定 patch 尺寸。后续可以同时使用多种尺度：

```text
128x128
224x224
384x384
```

小 patch 更关注纹理、边缘和局部噪声；大 patch 更关注结构和语义一致性。最终将多尺度 score map 融合。

### 2. Top-K Patch 聚合

整图判断不一定使用所有 patch 的平均值，可以使用最可疑的 Top-K patch：

```text
image_fake_score = mean(top_k(patch_fake_scores))
```

这对局部生成、局部篡改或拼接图像更敏感。

### 3. Patch 分数统计特征

除平均分外，还可以记录：

```text
max score
mean score
score variance
high-score patch ratio
```

这些统计量可以作为一个轻量级二阶段分类器的输入。

### 4. 与 SRM / FFT 结合

Patch 输入仍然可以走 RGB + SRM + FFT 三分支结构：

```text
patch_rgb -> RGB branch
patch_srm -> SRM branch
patch_fft -> FFT branch
```

这样可以观察局部区域的空间纹理、高频残差和频域异常是否一致。

### 5. 可解释性输出

最终展示可以包含：

```text
original image
fake score heatmap
overlay image
top suspicious patches
```

这部分适合放在报告的可视化分析章节。

## 当前命令

```powershell
python -m src.heatmap `
  --config configs/mobilenetv3_genimage_10gb.yaml `
  --checkpoint outputs_genimage_10gb/best_model.pt `
  --image path\to\image.jpg `
  --patch 224 `
  --stride 112 `
  --output outputs_genimage_10gb/visualizations/heatmap.jpg
```

如果显存或速度压力较大，可以增大 stride：

```powershell
--stride 224
```

如果希望热力图更细，可以减小 stride：

```powershell
--stride 56
```

## Patch 展示数据准备

Patch 展示需要局部 AIGC 图像和对应 mask。推荐使用 TGIF/TGIF2 这类 Text-Guided Inpainting Forgery 数据集，因为它们包含：

```text
manipulated image
mask
original image
```

准备 100 张展示样本：

```powershell
python scripts/prepare_tgif_patch_demo.py `
  --source D:\datasets\TGIF `
  --target data\patch_demo_tgif `
  --count 100 `
  --split testing
```

如果已经有训练好的 GenImage 模型，可以让脚本自动筛选热力图和 mask 匹配更好的样本：

```powershell
python scripts/prepare_tgif_patch_demo.py `
  --source D:\datasets\TGIF `
  --target data\patch_demo_tgif `
  --count 100 `
  --split testing `
  --config configs\mobilenetv3_genimage_10gb.yaml `
  --checkpoint outputs_genimage_10gb\best_model.pt `
  --patch 224 `
  --stride 112
```

输出目录：

```text
data/patch_demo_tgif/
  images/
    patch_demo_000_image.png
    patch_demo_000_mask.png
    patch_demo_000_mask_overlay.jpg
    patch_demo_000_heatmap.jpg
  metadata.json
```

其中：

- `*_image`: 局部 AIGC 编辑后的图像
- `*_mask`: 局部 AIGC 区域标注
- `*_mask_overlay`: mask 叠加到原图上的展示图
- `*_heatmap`: 模型 patch fake score 热力图

如果当前只有 CIFAKE 模型，不建议用它来筛选 TGIF 展示图，因为 CIFAKE 是 `32x32` 数据训练出的整图 baseline，对高分辨率局部编辑图的泛化解释性有限。更合适的流程是先训练 GenImage 10GB baseline，再用这个脚本筛 Patch 展示样本。

## 更新的轻量展示数据：DiffSeg30k

如果不想使用 TGIF，推荐使用 DiffSeg30k。它是面向 localized AIGC detection 的扩散编辑数据集，提供 edited image 和 pixel-level mask，适合只抽取 100 张做 Patch 展示。

安装依赖：

```powershell
pip install datasets
```

流式抽取 100 张，不需要下载完整数据集：

```powershell
python scripts/prepare_diffseg30k_patch_demo.py --target data/patch_demo_diffseg30k --count 100 --split validation
```

输出：

```text
data/patch_demo_diffseg30k/
  images/
    diffseg30k_000_image.png
    diffseg30k_000_mask.png
    diffseg30k_000_mask_overlay.jpg
  metadata.json
```

如果想换一批展示图，可以加 `--skip`：

```powershell
python scripts/prepare_diffseg30k_patch_demo.py --target data/patch_demo_diffseg30k_b --count 100 --split validation --skip 100
```
