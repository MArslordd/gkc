# 轻量化 AIGC 图像检测系统

这是一个面向端侧部署的 AIGC 图像二分类检测框架。项目采用 **RGB 空间域 + SRM 高频残差 + FFT 频域分析** 的三分支结构，主干默认使用 `MobileNetV3-Small`，适合在 RTX 4060 上训练和实验。

## 0. 项目文档入口

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md): 项目整体目录和模块说明
- [BASELINE_SUMMARY.md](BASELINE_SUMMARY.md): CIFAKE RGB baseline 结果总结
- [DATASET_SELECTION.md](DATASET_SELECTION.md): CIFAKE / GenImage 数据选择和整理命令
- [docs/EXPERIMENT_ROADMAP.md](docs/EXPERIMENT_ROADMAP.md): 后续实验路线
- [docs/PATCH_INNOVATION.md](docs/PATCH_INNOVATION.md): Patch 切割创新点说明

## 1. 环境准备

建议使用 Python 3.10 或 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

检查 GPU：

```powershell
python scripts/check_cuda.py
```

如果能看到 `cuda available: True` 和 RTX 4060 名称，就说明 PyTorch 可以调用 GPU。

## 2. 数据集选择

本项目第一版固定使用 **GenImage** 做主数据集。GenImage 自带真实图像和生成图像配对，真实图像来自 ImageNet，生成图像覆盖 Midjourney、Stable Diffusion V1.4、Stable Diffusion V1.5、ADM、GLIDE、Wukong、VQDM、BigGAN 等生成器。因此第一版不再混用 COCO/Flickr30k 作为训练 real，避免模型学到“数据源差异”而不是“真假差异”。

最终选择如下：

- `real`: GenImage 每个子集里的 `nature` / real ImageNet 图像
- `fake`: GenImage 每个生成器子集里的 `ai` / generated 图像
- `train`: Stable Diffusion V1.4 + Stable Diffusion V1.5 + BigGAN
- `val`: 与 `train` 相同生成器，但使用 GenImage 官方 test 或保留验证图片
- `test_seen`: Stable Diffusion V1.4 + Stable Diffusion V1.5 + BigGAN 的未参与训练图片
- `test_unseen`: Midjourney + ADM + GLIDE + Wukong + VQDM

这样划分的含义：

- `test_seen` 测同类生成器上的检测能力
- `test_unseen` 测跨生成器泛化能力
- real 和 fake 都来自 GenImage 的 ImageNet 类别体系，语义类别更平衡

如果 RTX 4060 本地训练压力太大，先做一个轻量子集：

- 每个训练生成器取 `fake` 10000 张
- 每个训练生成器匹配 `real` 10000 张
- `val` 每类 2000 张
- `test_seen` 每类 2000 张
- `test_unseen` 每个未见生成器各取 fake 2000 张，并匹配 real 2000 张

注意：划分时不要让同一路径、同一张图片或高度相似样本同时进入训练集和测试集，否则泛化指标会虚高。

如果 GenImage 已经下载并解压到 `D:\datasets\GenImage`，可以用脚本整理成当前项目需要的目录：

```powershell
python scripts/prepare_genimage.py --source D:\datasets\GenImage --dry-run
python scripts/prepare_genimage.py --source D:\datasets\GenImage
```

## 3. 数据目录格式

把下载好的开源数据集整理成下面结构：

```text
data/genimage_imagenet/
  train/
    real/
      xxx.jpg
    fake/
      yyy.jpg
  val/
    real/
    fake/
  test_seen/
    real/
    fake/
  test_unseen/
    real/
    fake/
```

类别名固定为：

- `real`: 真实图像
- `fake`: AIGC 生成图像

## 4. RTX 4060 推荐训练配置

默认配置文件在：

```text
configs/mobilenetv3_multibranch.yaml
```

当前默认值：

```yaml
image_size: 224
batch_size: 32
epochs: 20
amp: true
backbone: mobilenet_v3_small
branch_mode: full
```

RTX 4060 通常可以跑：

- `batch_size: 32`: 推荐起步
- `batch_size: 48` 或 `64`: 显存足够时可以尝试
- 如果显存不足，改成 `batch_size: 16`

如果只是先跑通流程，可以把：

```yaml
model:
  branch_mode: rgb
```

这样只训练 RGB 分支，速度更快。完整实验再切回：

```yaml
model:
  branch_mode: full
```

## 5. 训练

```powershell
python -m src.train --config configs/mobilenetv3_multibranch.yaml
```

训练完成后会生成：

```text
outputs/best_model.pt
outputs/train_history.json
```

`best_model.pt` 按验证集 F1-score 保存。

## 6. 评估

评估训练时见过生成器分布：

```powershell
python -m src.evaluate --split test_seen --checkpoint outputs/best_model.pt
```

评估未见生成器泛化能力：

```powershell
python -m src.evaluate --split test_unseen --checkpoint outputs/best_model.pt
```

输出文件：

```text
outputs/eval_test_seen.json
outputs/eval_test_unseen.json
```

指标包括：

- Accuracy
- Precision
- Recall
- F1-score
- AUC

## 7. 单图推理

```powershell
python -m src.infer --image path\to\image.jpg --checkpoint outputs/best_model.pt
```

输出示例：

```text
{'image': 'demo.jpg', 'prediction': 'Fake', 'fake_score': 0.873421}
```

## 8. Patch 热力图

用于观察局部 AIGC 可疑区域：

```powershell
python -m src.heatmap --image path\to\image.jpg --checkpoint outputs/best_model.pt
```

输出：

```text
outputs/visualizations/heatmap.jpg
```

## 9. ONNX 导出

```powershell
python -m src.export_onnx --checkpoint outputs/best_model.pt --output outputs/aigc_detector.onnx
```

后续可以用 ONNX Runtime、TensorRT 或移动端推理框架做部署测试。

## 10. 建议实验顺序

1. `branch_mode: rgb` 跑通 baseline
2. `branch_mode: full` 训练 RGB + SRM + FFT 三分支模型
3. 做消融实验：RGB、RGB+SRM、RGB+FFT、RGB+SRM+FFT
4. 做跨模型泛化：`test_unseen`
5. 做鲁棒性测试：JPEG 压缩、Resize、Blur
6. 导出 ONNX，测试端侧推理速度
7. 输出 Patch 热力图做可视化展示

## 11. 当前框架说明

主要文件：

```text
src/datasets.py       数据加载与三路输入生成
src/srm.py            SRM 高频残差特征
src/fft_features.py   FFT 频谱特征
src/model.py          MobileNetV3 多分支检测模型
src/train.py          训练脚本
src/evaluate.py       评估脚本
src/infer.py          单图推理
src/export_onnx.py    ONNX 导出
src/heatmap.py        Patch 热力图
```

这个版本的目标是先建立完整工程闭环：能训练、能评估、能推理、能可视化、能导出部署模型。后续可以继续加入量化、更多 backbone、鲁棒性自动评测和数据集整理脚本。
