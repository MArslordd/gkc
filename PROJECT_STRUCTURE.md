# 项目整体框架

本项目面向轻量化 AIGC 图像检测，目标是构建一个可以训练、评估、部署和可视化解释的二分类检测系统。项目分为数据准备、模型训练、模型评估、端侧导出、Patch 级定位分析五个部分。

## 目录结构

```text
gongkechuang/
  configs/
    mobilenetv3_cifake.yaml              # CIFAKE RGB baseline 配置
    mobilenetv3_genimage_10gb.yaml       # GenImage 10GB RGB baseline 配置
    mobilenetv3_multibranch.yaml         # RGB + SRM + FFT 三分支配置

  data/
    cifake/                              # CIFAKE 整理后的训练数据
    cifake_raw/                          # CIFAKE 原始下载数据
    genimage_10gb/                       # GenImage 10GB 子集，整理后生成

  docs/
    EXPERIMENT_ROADMAP.md                # 实验路线和阶段安排
    PATCH_INNOVATION.md                  # Patch 切割创新点说明

  outputs/
    checkpoints/                         # 通用 checkpoint 输出目录
    metrics/                             # 通用指标输出目录
    visualizations/                      # 通用可视化输出目录

  outputs_cifake/
    best_model.pt                        # CIFAKE baseline 最优权重
    train_history.json                   # CIFAKE baseline 训练日志
    eval_test_seen.json                  # CIFAKE baseline 测试结果
    aigc_detector_cifake.onnx            # CIFAKE ONNX 模型结构
    aigc_detector_cifake.onnx.data       # CIFAKE ONNX 外置权重

  outputs_genimage_10gb/
    checkpoints/                         # GenImage 实验权重
    metrics/                             # GenImage 实验指标
    visualizations/                      # GenImage 可视化结果

  reports/
    figures/                             # 报告图片、曲线、热力图
    tables/                              # 实验表格

  scripts/
    check_cuda.py                        # 检查 CUDA 和 GPU
    prepare_cifake.py                    # 整理 CIFAKE
    prepare_genimage.py                  # 整理 GenImage，支持约 10GB 抽样
    prepare_tgif_patch_demo.py           # 从 TGIF/TGIF2 准备 Patch 展示样本

  src/
    datasets.py                          # 数据加载和 RGB/SRM/FFT 输入生成
    evaluate.py                          # 测试集评估
    export_onnx.py                       # ONNX 导出
    fft_features.py                      # FFT 频域图生成
    heatmap.py                           # Patch 滑窗热力图
    infer.py                             # 单图推理
    metrics.py                           # Accuracy/Precision/Recall/F1/AUC
    model.py                             # MobileNetV3 单分支/多分支模型
    srm.py                               # SRM 高频残差图生成
    train.py                             # 训练入口
    utils.py                             # 配置、设备、JSON 工具

  BASELINE_SUMMARY.md                    # CIFAKE baseline 总结
  DATASET_SELECTION.md                   # 数据集选择和整理说明
  README.md                              # 项目说明和运行命令
  PROJECT_STRUCTURE.md                   # 当前文件
  requirements.txt                       # Python 依赖
```

## 模型路线

第一阶段是 RGB-only baseline：

```text
RGB image -> MobileNetV3-Small -> MLP classifier -> real/fake
```

第二阶段是多分支检测器：

```text
RGB image -> MobileNetV3 branch -> feature_rgb
SRM image -> MobileNetV3 branch -> feature_srm
FFT image -> MobileNetV3 branch -> feature_fft

concat(feature_rgb, feature_srm, feature_fft)
-> MLP classifier
-> real/fake
```

第三阶段是 Patch 级定位分析：

```text
input image
-> sliding window patch crop
-> each patch gets fake_score
-> aggregate fake_score map
-> overlay heatmap on original image
```

## 推荐实验顺序

1. CIFAKE RGB baseline：确认训练链路、评估链路和 ONNX 导出链路。
2. GenImage 10GB RGB baseline：在更高分辨率、多生成器数据上建立新 baseline。
3. GenImage 10GB RGB + SRM + FFT：验证频域和高频残差信息是否提升泛化能力。
4. `test_seen` / `test_unseen` 对比：衡量同生成器和未见生成器上的性能差异。
5. Patch 热力图：输出局部 fake score，可用于结果解释和展示。
6. 鲁棒性测试：JPEG 压缩、resize、blur、crop 等扰动下评估模型稳定性。
