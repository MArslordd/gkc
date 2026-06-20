# 实验路线

## 阶段 1：CIFAKE Baseline

目标：验证训练、评估、导出和推理链路完整可用。

配置：

```text
configs/mobilenetv3_cifake.yaml
```

已完成结果：

```text
Dataset: CIFAKE
Model: MobileNetV3-Small
Input: RGB only
Accuracy: 98.30%
Precision: 97.44%
Recall: 99.21%
F1-score: 98.32%
AUC: 99.66%
```

这个阶段作为工程 baseline，而不是最终泛化结论。原因是 CIFAKE 原始图像为 `32x32`，真实场景复杂度较低。

## 阶段 2：GenImage 10GB RGB Baseline

目标：在更真实、更高分辨率、更多生成器来源的数据上建立主 baseline。

数据准备：

```powershell
python scripts/prepare_genimage.py --source D:\datasets\GenImage --target data/genimage_10gb --max-total-gb 10 --dry-run
python scripts/prepare_genimage.py --source D:\datasets\GenImage --target data/genimage_10gb --max-total-gb 10
```

训练：

```powershell
python -m src.train --config configs/mobilenetv3_genimage_10gb.yaml
```

评估：

```powershell
python -m src.evaluate --config configs/mobilenetv3_genimage_10gb.yaml --split test_seen --checkpoint outputs_genimage_10gb/best_model.pt
python -m src.evaluate --config configs/mobilenetv3_genimage_10gb.yaml --split test_unseen --checkpoint outputs_genimage_10gb/best_model.pt
```

需要记录：

```text
test_seen: Accuracy / Precision / Recall / F1 / AUC
test_unseen: Accuracy / Precision / Recall / F1 / AUC
```

## 阶段 3：RGB + SRM + FFT 多分支模型

目标：验证频域和高频残差特征是否提升检测效果，尤其是未见生成器泛化能力。

做法：

1. 复制 `configs/mobilenetv3_genimage_10gb.yaml`。
2. 将 `branch_mode` 改为 `full`。
3. 输出目录改成 `outputs_genimage_10gb_full`。

核心对比表：

```text
RGB only
RGB + SRM + FFT
```

重点看 `test_unseen` 上的 F1 和 AUC。如果 full 模型在 test_unseen 上更好，就能说明频域/高频分支对泛化有帮助。

## 阶段 4：Patch 切割与局部可解释性

目标：从整图真假分类，扩展到局部区域可疑性分析。

命令示例：

```powershell
python -m src.heatmap `
  --config configs/mobilenetv3_genimage_10gb.yaml `
  --checkpoint outputs_genimage_10gb/best_model.pt `
  --image path\to\test_image.jpg `
  --patch 224 `
  --stride 112 `
  --output outputs_genimage_10gb/visualizations/heatmap.jpg
```

输出热力图用于展示模型认为哪些局部区域更像 AIGC 生成区域。

## 阶段 5：鲁棒性测试

目标：模拟真实传播场景里的图像退化。

建议扰动：

```text
JPEG quality: 95 / 75 / 50
Resize: 0.5x / 0.75x / 1.25x
Gaussian blur
Center crop / random crop
```

建议输出表格：

```text
Clean
JPEG-95
JPEG-75
JPEG-50
Resize-0.5x
Blur
Crop
```

每一行记录 Accuracy / F1 / AUC。
