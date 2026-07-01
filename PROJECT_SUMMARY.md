# 项目总结

本项目实现了一个轻量化 AIGC 图像检测系统，目标是在较小模型和较低部署成本下完成真实图像与 AI 生成图像的二分类检测，并进一步探索局部 AIGC 编辑区域的 Patch 级可解释性。

## 当前方法

模型采用 MobileNetV3-Small 作为轻量主干，并支持两种输入模式：

```text
RGB only
RGB + SRM + FFT
```

其中：

- RGB 分支学习常规空间域视觉特征。
- SRM 分支强调高频残差和局部纹理异常。
- FFT 分支引入频域特征，辅助捕捉生成图像的频谱差异。

三分支模型会将不同分支的特征拼接后输入 MLP 分类头，输出 `real/fake`。

## 数据集

当前项目保留三类数据用途：

```text
CIFAKE: 快速 baseline，验证工程链路。
Defactify / MS COCOAI: 主训练数据，覆盖 SD2.1、SDXL、SD3、DALL-E 3、Midjourney。
DiffSeg30k: 局部 AIGC Patch 展示与 patch-level 训练数据。
```

GenImage 因下载和整理成本较高，已不作为当前主线。

## 已得到的经验

1. CIFAKE 上的 RGB baseline 可以达到较高指标，但 CIFAKE 分辨率低，不能代表真实场景。
2. Defactify 的 seen/unseen 划分显示，整图检测器对未见生成器可能严重失效。
3. 直接用整图模型做滑窗 heatmap，不能保证定位局部 AIGC 区域。
4. Patch 部分要稳定成功，需要使用 DiffSeg30k mask 构造 patch-level 训练集。

## Patch 成功的关键

Patch 展示的可靠路线是：

```text
DiffSeg30k image + mask
-> 根据 mask 覆盖率切分 real/fake patch
-> 训练 patch-level RGB/SRM/FFT 检测器
-> 滑窗预测每个 patch 的 fake score
-> 聚合成 heatmap
```

标签规则：

```text
mask coverage >= 30% -> fake patch
mask coverage <= 2%  -> real patch
其他边界 patch 忽略
```

这样训练目标与展示目标一致，能够显著提高热力图与真实 mask 的一致性。

## 最终推荐实验表

```text
CIFAKE RGB baseline
Defactify all-generators RGB+SRM+FFT
DiffSeg30k patch-level RGB+SRM+FFT
```

报告中应区分：

- 整图检测指标：Accuracy / Precision / Recall / F1 / AUC。
- Patch 展示效果：原图、mask overlay、heatmap 的可视化对比。

如果后续需要定量评价 Patch 定位，可以进一步计算 heatmap 与 mask 的 pixel-level AUC、IoU 或 Dice。
