# DLForPytorch

> **本项目已停止维护 (Archived)**
>
> 该仓库于 2026 年 8 月正式归档，不再接受新的功能开发、bug 修复或 Pull Request。
> 代码保留仅供学习参考。

---

## 项目简介

DLForPytorch 是一个基于 PyTorch 的深度学习框架，专注于 **3D 医学图像分割** 任务，同时涵盖 2D 图像分类、分割和图像风格迁移。项目最初由 [FGBraKing](https://github.com/FGBraKing) 开发，主要用于前列腺 MRI/TRUS 图像分割、脑肿瘤图像分割（BraTS）等医学影像分析场景。

- **框架**: PyTorch 1.8
- **核心任务**: 3D 医学图像分割
- **数据格式**: NIfTI (.nii/.nii.gz), HDF5 (.h5), NumPy (.npy)
- **分布式支持**: DDP, DataParallel, Horovod, APEX

---

## 项目结构

```
DLForPytorch/
├── train.py / test.py / predict.py      # 主入口脚本
├── configs/                             # 配置系统 (YAML + argparse)
│   ├── default_config.py                # 默认配置
│   ├── simple_options.py                # 参数解析
│   └── defaults/                        # 实验配置文件
├── data/                                # 数据模块
│   ├── dataloads/                       # 数据集类 (TRUS, PROMISE12, BraTS 等)
│   ├── transforms/                      # 数据增强 (15+ 种操作)
│   ├── pre_process/                     # 数据预处理
│   └── utils_data.py                    # 数据加载工具
├── models/                              # 模型模块
│   ├── networks/                        # 封装网络模型 (Unet3dModel, Vnet3dModel, CycleGAN 等)
│   ├── modules/segmentation/three_d/    # 20+ 种 3D 分割网络 (UNet, VNet, UNETR, DenseVoxelNet 等)
│   ├── modules/classification/two_d/    # 50+ 种 2D 分类网络 (timm 风格)
│   ├── loss/                            # 15+ 种损失函数 (Dice, Focal, Tversky, Combo 等)
│   ├── optim/                           # 15 种优化器
│   └── scheduler/                       # 8 种学习率调度策略
├── scripts/                             # Shell 启动脚本
└── pytorch_grad_cam/                    # Grad-CAM 可视化
```

---

## 技术特点

- **完整的 3D 医学图像分割流水线**: 数据加载 → 预处理 → 增强 → 训练 → 测试 → 评估
- **丰富的网络架构**: 20+ 种 3D 分割网络、50+ 种 2D 分类网络
- **多种分布式训练策略**: DDP、DataParallel、Horovod、APEX，支持单节点和多节点
- **灵活的数据增强**: 三段式增强流水线，支持 15+ 种操作
- **丰富的损失函数库**: 15+ 种分割损失函数
- **三级配置系统**: 默认值 → YAML 文件 → 命令行参数
- **滑动窗口推理**: 支持大体积图像的滑动窗口预测和拼接
- **测试时增强**: 支持多轴翻转的测试时数据增强

---

## 环境要求 (原始)

```
Python 3.8+
PyTorch 1.8
CUDA 11.x
```

详见 [requirements.txt](./requirements.txt)。

---

## 替代方案推荐

由于本项目技术栈较为陈旧（PyTorch 1.8, 2021 年），以下更成熟的方案值得考虑：

### 医学图像分割

| 方案 | 说明 | 链接 |
|------|------|------|
| **MONAI** | 医学影像分析官方框架，开箱即用 20+ 3D 网络、滑动窗口、数据增强、损失函数 | [https://monai.io](https://monai.io) |
| **nnUNet** | 自配置 3D 医学分割框架，在多个挑战赛中取得 SOTA | [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| **nnUNetv2** | nnUNet 的升级版，更灵活的配置和更优的性能 | [https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2](https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2) |

### 通用图像分类

| 方案 | 说明 | 链接 |
|------|------|------|
| **timm** | PyTorch 图像模型库，500+ 预训练模型 | [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| **torchvision** | PyTorch 官方视觉库 | [https://pytorch.org/vision/](https://pytorch.org/vision/) |

### 图像风格迁移

| 方案 | 说明 | 链接 |
|------|------|------|
| **CycleGAN/Pix2Pix 官方实现** | 原作者维护的 PyTorch 版本 | [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |

---

## 许可证

本项目代码仅供学习参考，未明确指定开源许可证。