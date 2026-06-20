# 数据集选择方案

## 结论

第一版训练和评估都以 GenImage 为主：

- fake 数据：GenImage 的 AI-generated 图像
- real 数据：GenImage 自带的 ImageNet real / nature 图像

不建议第一版用 COCO、Flickr30k 或 LAION 作为训练 real，因为这些真实图像和 GenImage fake 的来源、压缩、分辨率、内容分布可能不同，模型容易学到数据集偏差。GenImage 自带 real 与 fake 在 ImageNet 1000 类体系下配对，更适合作为二分类检测的 MVP 数据。

## 具体划分

```text
data/genimage_imagenet/
  train/
    real/
    fake/
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

推荐生成器划分：

| Split | Fake 生成器 | Real 来源 | 目的 |
| --- | --- | --- | --- |
| train | SD V1.4, SD V1.5, BigGAN | GenImage nature/ImageNet | 训练基础检测器 |
| val | SD V1.4, SD V1.5, BigGAN | GenImage nature/ImageNet | 选择 checkpoint 和调参 |
| test_seen | SD V1.4, SD V1.5, BigGAN | GenImage nature/ImageNet | 测同分布效果 |
| test_unseen | Midjourney, ADM, GLIDE, Wukong, VQDM | GenImage nature/ImageNet | 测跨生成器泛化 |

## RTX 4060 轻量子集

如果先跑通流程，不需要一次性使用完整 GenImage：

- train: 每个已见生成器 fake 10000 张，并匹配 real 10000 张
- val: real 2000 张，fake 2000 张
- test_seen: real 2000 张，fake 2000 张
- test_unseen: 每个未见生成器 fake 2000 张，并匹配 real 2000 张

所有 split 内保持 `real:fake = 1:1`。如果某个生成器数量不足，宁可少取，也不要破坏类别平衡。

## 整理命令

假设下载并解压后的 GenImage 原始目录是：

```text
D:\datasets\GenImage
```

其中包含 `Midjourney`、`Stable Diffusion V1.4`、`Stable Diffusion V1.5`、`ADM`、`GLIDE`、`Wukong`、`VQDM`、`BigGAN` 等子目录，每个子目录下有 `train/ai`、`train/nature`、`val/ai`、`val/nature`。

先预览将要整理的数量：

```powershell
python scripts/prepare_genimage.py --source D:\datasets\GenImage --dry-run
```

确认没问题后执行整理：

```powershell
python scripts/prepare_genimage.py --source D:\datasets\GenImage
```

如果希望节省磁盘空间，可以尝试硬链接模式。硬链接失败时脚本会自动退回复制：

```powershell
python scripts/prepare_genimage.py --source D:\datasets\GenImage --mode hardlink
```

## 约 10GB 的 GenImage 训练子集

如果只希望整理约 10GB 数据用于训练，可以使用容量预算模式。默认划分比例为：

```text
train: 70%
val: 10%
test_seen: 10%
test_unseen: 10%
```

默认生成器划分仍然是：

```text
train / val / test_seen: Stable Diffusion V1.4, Stable Diffusion V1.5, BigGAN
test_unseen: Midjourney, ADM, GLIDE, Wukong, VQDM
```

先预览：

```powershell
python scripts/prepare_genimage.py --source D:\datasets\GenImage --target data/genimage_10gb --max-total-gb 10 --dry-run
```

正式整理：

```powershell
python scripts/prepare_genimage.py --source D:\datasets\GenImage --target data/genimage_10gb --max-total-gb 10
```

如果只下载了部分 GenImage 子集，可以手动指定生成器。例如只用 SD V1.4、SD V1.5 做训练，用 ADM、GLIDE 做未见生成器测试：

```powershell
python scripts/prepare_genimage.py `
  --source D:\datasets\GenImage `
  --target data/genimage_10gb `
  --max-total-gb 10 `
  --train-generators "Stable Diffusion V1.4,Stable Diffusion V1.5" `
  --unseen-generators "ADM,GLIDE"
```

整理完成后会生成：

```text
data/genimage_10gb/prepare_manifest.json
```

里面记录每个 split 实际整理了多少图片、占用了多少 GB。

## 后续可选扩展

完成 GenImage 主实验后，可以再加入一个外部真实图像测试集，例如 COCO 2017 validation 或 OpenImages 小子集，只用于额外测试，不用于第一版训练。这样可以观察真实图像域迁移时的误报率。

## 更轻量的一键下载替代方案

如果 GenImage 下载太麻烦，可以先用 CIFAKE 跑通 MVP。CIFAKE 比 GenImage 小很多，Kaggle CLI 可以一条命令下载。

下载：

```powershell
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/cifake_raw --unzip
```

整理成项目目录：

```powershell
python scripts/prepare_cifake.py --source data/cifake_raw --target data/cifake
```

然后把配置文件里的数据根目录改为：

```yaml
data:
  root: data/cifake
```

CIFAKE 的图像分辨率较低，适合先验证代码链路和消融实验，不建议作为最终报告里唯一的数据集。
