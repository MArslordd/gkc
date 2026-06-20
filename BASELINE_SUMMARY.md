# Baseline 总结

本项目首先构建了一个轻量化 AIGC 图像检测 baseline，用于完成真实图像与 AI 生成图像的二分类任务。baseline 采用 CIFAKE 数据集进行训练和测试，该数据集包含真实图像与 AI 生成图像两类样本，原始分辨率为 `32x32`。实验中所有图像统一 resize 到 `224x224`，以适配 ImageNet 预训练的 MobileNetV3 输入尺寸。

模型结构采用 **MobileNetV3-Small + MLP 分类头**。输入为 RGB 图像，主干网络提取视觉特征后，经过全连接分类器输出 `real / fake` 二分类结果。该 baseline 没有引入额外的频域特征或高频残差特征，主要用于衡量普通 RGB 空间域检测方法在轻量模型上的基础性能。

## 训练配置

```text
Dataset: CIFAKE
Input: RGB image
Image size: 224x224
Backbone: MobileNetV3-Small
Pretrained: ImageNet
Classifier: Linear + ReLU + Dropout + Linear
Loss: CrossEntropyLoss
Optimizer: AdamW
Batch size: 64
Epochs: 10
Mixed precision: enabled
```

## 测试结果

```text
Accuracy  = 98.30%
Precision = 97.44%
Recall    = 99.21%
F1-score  = 98.32%
AUC       = 99.66%
```

## 混淆矩阵

```text
TP = 9921
TN = 9739
FP = 261
FN = 79
Total = 20000
```

从结果可以看出，该 baseline 在 CIFAKE 测试集上取得了较高的整体性能，尤其是 fake 类别的召回率较高，说明模型能够较好地识别 AI 生成图像。与此同时，模型存在一定数量的 false positive，即部分真实图像被误判为 fake，说明该模型倾向于更保守地检测生成图像。

需要注意的是，CIFAKE 图像分辨率较低，且数据分布相对简单，因此该结果更适合作为训练流程和模型结构的初始 baseline。后续实验应进一步引入更高分辨率、更复杂生成器来源的数据集，例如 GenImage 子集，并与 RGB + SRM + FFT 多分支模型进行对比，以验证频域和高频残差信息是否能够提升跨生成器泛化能力。
