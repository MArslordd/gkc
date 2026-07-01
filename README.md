# 轻量化 AIGC 图像检测系统

本项目面向 AIGC 图像检测与局部可解释性展示，包含整图 `real/fake` 二分类、RGB/SRM/FFT 三分支特征融合、ONNX 导出，以及基于 DiffSeg30k 的 Patch 级局部 AIGC 热力图。

当前推荐主线：

```text
Defactify 全生成器训练
-> MobileNetV3-Small RGB + SRM + FFT
-> DiffSeg30k Patch 展示
-> DiffSeg30k patch-level 训练保证局部定位效果
```

## 文档入口

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md): 项目总结
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md): 文件夹结构
- [BASELINE_SUMMARY.md](BASELINE_SUMMARY.md): CIFAKE baseline
- [DATASET_SELECTION.md](DATASET_SELECTION.md): 数据集说明
- [docs/EXPERIMENT_ROADMAP.md](docs/EXPERIMENT_ROADMAP.md): 实验路线
- [docs/PATCH_INNOVATION.md](docs/PATCH_INNOVATION.md): Patch 局部检测方案

## 环境

```bash
pip install -r requirements.txt
python scripts/check_cuda.py
```

## Defactify 全生成器训练

下载 Defactify：

```bash
hf download \
  Rajarshi-Roy-research/Defactify_Image_Dataset \
  --repo-type dataset \
  --local-dir data/defactify_raw
```

整理全部生成器：

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

## DiffSeg30k Patch 展示

准备 100 张展示图：

```bash
python scripts/prepare_diffseg30k_patch_demo.py \
  --target data/patch_demo_diffseg30k \
  --count 100 \
  --split validation
```

用整图模型生成 heatmap：

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

## 保证 Patch 效果的训练

直接用整图模型滑窗不一定能定位局部 AIGC。要提高成功率，先用 mask 切 patch：

```bash
python scripts/prepare_diffseg30k_patch_dataset.py \
  --source data/patch_demo_diffseg30k/images \
  --target data/diffseg30k_patch \
  --patch 224 \
  --stride 112 \
  --fake-threshold 0.30 \
  --real-threshold 0.02
```

训练 patch-level 模型：

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

## 主要代码

```text
src/datasets.py       数据加载，生成 RGB/SRM/FFT 三路输入
src/model.py          MobileNetV3 单分支/三分支模型
src/train.py          训练
src/evaluate.py       评估
src/heatmap.py        单图 Patch heatmap
scripts/batch_heatmap.py 批量 Patch heatmap
```
