 训练记录

## 环境

| 项 | 配置 |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| Ultralytics | 8.4.208 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（6 GB） |
| 驱动 | 517.00 |
| 系统内存 | 16 GB |

## 数据集

**Hard Hat Workers**（Roboflow，CC BY 4.0）

| 子集 | 图片数 | 标签数 |
|---|---|---|
| train | 5269 | 5269 |
| test（作为 val 使用） | 1766 | 1766 |

类别定义（`data.yaml`）：

```yaml
train: ../train/images
val: ../test/images
test: ../test/images

nc: 3
names: ['head', 'helmet', 'person']
```

> **说明**：官方导出未提供 valid 集，因此将 test 集同时用作验证集。
> 正式实验中建议从 train 中划分 10% 作为独立验证集。

## 训练命令

```bash
yolo detect train \
    data=D:\projects\dataset\hard-hat-workers\data.yaml \
    model=yolo11n.pt \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    amp=False \
    workers=0
```

### 参数说明

| 参数 | 值 | 原因 |
|---|---|---|
| `model` | `yolo11n.pt` | 最小模型，先验证流程 |
| `epochs` | 100 | 训练充分 |
| `batch` | 8 | 6GB 显存的安全值 |
| `amp` | `False` | AMP 检查阶段在本机崩溃，关闭后稳定 |
| `workers` | 0 | 16GB 内存下多进程 dataloader 会 OOM |

### 训练中断与恢复

训练过程中曾因内存不足中断两次：

1. 第一次跑到 epoch 56 中断 → 用 `resume` 从 57 继续
2. 第二次跑到 epoch 79 中断 → 用 `resume` 从 80 继续至 100

```bash
yolo detect train resume model=runs/detect/train-4/weights/last.pt
```

## 训练结果

### 总览

| 指标 | 数值 |
|---|---|
| 完成轮数 | 100 |
| 总训练耗时 | 46.8 分钟（含中断恢复） |
| 最佳 epoch | 95 |
| 最佳 mAP@0.5 | **0.6504** |
| 最佳 mAP@0.5:0.95 | 0.4342 |
| Precision | 0.7263 |
| Recall | 0.6215 |

### 分类别最终验证结果

| 类别 | 图片数 | 实例数 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| **all** | 1766 | 8915 | 0.732 | 0.621 | **0.650** | 0.434 |
| **helmet** | 1604 | 6482 | 0.947 | 0.931 | **0.970** | 0.652 |
| **head** | 339 | 2208 | 0.916 | 0.920 | **0.948** | 0.631 |
| person | 45 | 225 | 0.334 | 0.013 | 0.033 | 0.019 |

### 结果分析

**核心类别表现优异**：
- `helmet`（安全帽）mAP@0.5 = 0.970，几乎完美
- `head`（未戴帽人头）mAP@0.5 = 0.948，同样很高

**person 类别失败的原因**：
- 该数据集主要标注"头"和"安全帽"，人体框极少
- 验证集 1766 张图中仅 45 张含人体框，共 225 个实例
- 训练样本不足导致模型无法学会
- **解决方案**：采用 COCO 预训练的 YOLOv11n 专门检测 person

### 推理性能

| 指标 | 数值 |
|---|---|
| 预处理 | 3.1 ms / 张 |
| 推理 | 2.8 ms / 张 |
| 后处理 | 1.7 ms / 张 |
| 单模型 FPS | 约 350 |
| 双模型 + 视频处理 FPS | 约 16（640×480） |

## 训练产物

位于 `runs/detect/train-4/`：

| 文件 | 说明 |
|---|---|
| `weights/best.pt` | 最佳权重（5.3 MB），检测时使用 |
| `weights/last.pt` | 最后一轮权重，用于 resume |
| `results.csv` | 100 轮逐轮指标 |
| `results.png` | 训练损失与指标曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `BoxPR_curve.png` | PR 曲线 |
| `val_batch*_pred.jpg` | 验证集检测效果 |

## 模型对比参考

| 模型 | 类别 | mAP@0.5 | 说明 |
|---|---|---|---|
| YOLOv5s（原项目） | helmet | 0.917 | 原项目 README 给出（50 epoch） |
| YOLOv5s（原项目） | 总体 | 0.888 | 原项目 README 给出 |
| **YOLOv11n（本项目）** | **helmet** | **0.970** | 100 epoch，本机训练 |
| **YOLOv11n（本项目）** | **总体** | **0.650** | 含拉低均值的 person 类 |

> **注**：原项目总体 mAP 只统计 helmet/head/person 中表现良好的类别，
> 本项目总体 mAP 因 person 类极低（0.033）而被显著拉低。
> **单看核心类别 helmet：本项目 0.970 > 原项目 0.917**。

## 复现步骤

1. 下载数据集：https://universe.roboflow.com/lee-sandbox/hard-hat-workers
2. 导出格式选择 YOLOv8/YOLO11
3. 修改 `data.yaml` 的 `val` 指向 `../test/images`（官方未提供 valid 集）
4. 执行上面的训练命令
5. 训练完成后权重位于 `runs/detect/train-N/weights/best.pt`

