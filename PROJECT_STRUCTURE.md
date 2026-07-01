# 项目结构

本项目是一套轻量化 AIGC 图像检测工程，当前主线是：

```text
Defactify 全生成器整图训练
-> MobileNetV3 RGB/SRM/FFT 三分支检测
-> DiffSeg30k 局部 AIGC Patch 展示与 patch-level 训练
```

## 目录

```text
gongkechuang/
  configs/
    mobilenetv3_cifake.yaml              # CIFAKE 快速 baseline
    mobilenetv3_defactify.yaml           # Defactify seen/unseen 泛化实验
    mobilenetv3_defactify_all.yaml       # Defactify 全生成器 full 模型
    mobilenetv3_diffseg30k_patch.yaml    # DiffSeg30k patch-level 模型
    mobilenetv3_multibranch.yaml         # 通用三分支模板

  docs/
    EXPERIMENT_ROADMAP.md                # 推荐实验流程
    PATCH_INNOVATION.md                  # Patch 方法、训练和展示说明

  reports/
    figures/                             # 报告图片
    tables/                              # 实验表格

  scripts/
    check_cuda.py                        # 检查 CUDA/GPU
    prepare_cifake.py                    # 整理 CIFAKE
    prepare_defactify.py                 # 整理 Defactify
    prepare_diffseg30k_patch_demo.py     # 下载/整理 100 张 Patch 展示图
    prepare_diffseg30k_patch_dataset.py  # 根据 mask 切 patch 训练集
    batch_heatmap.py                     # 批量生成 Patch heatmap

  src/
    datasets.py                          # 读取 real/fake 目录并生成 RGB/SRM/FFT 输入
    model.py                             # MobileNetV3 单分支/三分支模型
    train.py                             # 训练入口
    evaluate.py                          # 评估入口
    infer.py                             # 单图推理
    heatmap.py                           # 单图 Patch heatmap
    export_onnx.py                       # ONNX 导出
    srm.py                               # SRM 高频残差
    fft_features.py                      # FFT 频域特征
    metrics.py                           # Accuracy/Precision/Recall/F1/AUC
    utils.py                             # 配置、设备、JSON 工具

  BASELINE_SUMMARY.md                    # CIFAKE baseline 结果
  DATASET_SELECTION.md                   # 数据集选择说明
  PROJECT_SUMMARY.md                     # 项目总结
  PROJECT_STRUCTURE.md                   # 当前文件
  README.md                              # 快速开始
  requirements.txt                       # Python 依赖
```

## 不再维护的内容

以下内容已从主线删除：

- TGIF 相关脚本：数据较旧，且当前展示数据已切换到 DiffSeg30k。
- GenImage 命令行下载/整理脚本：下载成本高，当前主训练集切换为 Defactify。
- `tools/` 下载工具和 `outputs_*` 训练产物：不属于源码，应由 `.gitignore` 忽略。

## 当前推荐主线

1. 用 `prepare_defactify.py --all-generators` 整理全生成器整图数据。
2. 用 `mobilenetv3_defactify_all.yaml` 训练 full 模型。
3. 用 `prepare_diffseg30k_patch_demo.py` 准备局部 AIGC 展示图。
4. 用 `batch_heatmap.py` 批量生成展示热力图。
5. 若需要保证 Patch 效果，用 `prepare_diffseg30k_patch_dataset.py` 构造 patch-level 数据，再训练 `mobilenetv3_diffseg30k_patch.yaml`。
