# 数据集选择

当前项目不再以 GenImage 作为主线。原因是 GenImage 下载和整理成本较高，不适合当前迭代节奏。最新主线采用：

```text
Defactify / MS COCOAI: 整图 real/fake 训练
DiffSeg30k: 局部 AIGC Patch 展示与 patch-level 训练
CIFAKE: 快速 baseline
```

## 1. Defactify / MS COCOAI

用途：主训练集。

覆盖生成器：

```text
SD2.1
SDXL
SD3
DALL-E 3
Midjourney
```

下载：

```bash
hf download \
  Rajarshi-Roy-research/Defactify_Image_Dataset \
  --repo-type dataset \
  --local-dir data/defactify_raw
```

推荐整理方式：全部生成器进入训练。

```bash
python scripts/prepare_defactify.py \
  --source data/defactify_raw \
  --target data/defactify_all \
  --all-generators
```

输出：

```text
data/defactify_all/
  train/real
  train/fake
  val/real
  val/fake
  test_seen/real
  test_seen/fake
```

说明：在 `defactify_all` 中，`test_seen` 实际表示全生成器测试集，不再表示 seen/unseen。

## 2. DiffSeg30k

用途：Patch 局部 AIGC 展示和 patch-level 训练。

DiffSeg30k 提供 edited image 和 pixel-level mask，适合验证局部 AIGC 检测。

准备 100 张展示图：

```bash
python scripts/prepare_diffseg30k_patch_demo.py \
  --target data/patch_demo_diffseg30k \
  --count 100 \
  --split validation
```

构造 patch-level 训练集：

```bash
python scripts/prepare_diffseg30k_patch_dataset.py \
  --source data/patch_demo_diffseg30k/images \
  --target data/diffseg30k_patch \
  --patch 224 \
  --stride 112 \
  --fake-threshold 0.30 \
  --real-threshold 0.02
```

输出：

```text
data/diffseg30k_patch/
  train/real
  train/fake
  val/real
  val/fake
  test_seen/real
  test_seen/fake
```

## 3. CIFAKE

用途：快速 baseline，仅用于验证工程链路。

注意：CIFAKE 原始图像是 `32x32`，不适合作为最终主实验数据。

```bash
kaggle datasets download \
  -d birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p data/cifake_raw \
  --unzip

python scripts/prepare_cifake.py \
  --source data/cifake_raw \
  --target data/cifake
```

## 4. 不再推荐

```text
GenImage: 数据质量好，但下载和整理成本高，当前不作为主线。
TGIF/TGIF2: 可做局部编辑实验，但数据相对旧，当前使用 DiffSeg30k 替代。
```
