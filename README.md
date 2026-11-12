# smart-site-safety-monitor

## 基于 YOLOv11 的智慧工地 PPE 与危险区域安全监测系统

> 面向 EHS（环境、健康与安全）场景的计算机视觉辅助监测系统。
> 基于开源项目 [Smart_Construction](https://github.com/PeterH0323/Smart_Construction) 复现，
> 完成模型升级、业务逻辑改造与 EHS 闭环模块开发。

---

## 项目背景

施工现场的高处坠落、物体打击、机械伤害等事故中，**未佩戴劳动防护用品（PPE）**与**违规进入危险区域**是两类高频直接原因。传统做法依赖安全员现场巡检与人工翻查监控录像，存在覆盖不全、响应滞后、取证困难等问题。

本项目实现一套 AI 辅助监测原型：

- **自动识别未佩戴安全帽人员**，将"目标检测"升级为"PPE 合规判定"
- **视频流实时检测危险区域闯入**，支持多边形禁区标定
- **违规自动截图归档**，并导出 EHS 隐患台账（Excel）

---

## 技术方案

### 双模型协同

| 模型 | 来源 | 职责 | 关键指标 |
|---|---|---|---|
| **PPE 检测模型** | 自训练 YOLOv11n | 识别 `head`（未戴帽人头）/ `helmet`（安全帽） | helmet mAP@0.5 = **0.970** |
| **人体检测模型** | COCO 预训练 YOLOv11n | 识别 `person`，用于危险区域判定 | COCO 官方权重 |

> **为什么用双模型**：自建数据集中 `person` 标注不足（验证集仅 45 张图 / 225 个目标，召回率 0.013），
> 因此采用 COCO 预训练的通用人体检测器补充，各司其职。

### PPE 合规判定逻辑

```
对每个检测到的 head 框：
    ↳ 计算它与所有 helmet 框的 IoU
    ↳ 最大 IoU >= 0.25  →  佩戴安全帽（合规）
    ↳ 最大 IoU <  0.25  →  未佩戴安全帽（违规，报警）
```

### 危险区域判定逻辑

```
对每个检测到的 person 框：
    ↳ 取人体框底边中心点（脚部位置）作为代表点
    ↳ cv2.pointPolygonTest 判断点是否落在多边形禁区内
    ↳ 在区内 → 违规闯入，报警 + 截图
```

---

## 训练结果

### 训练配置

| 项 | 值 |
|---|---|
| 模型 | YOLOv11n（2,590,425 参数） |
| 数据集 | [Hard Hat Workers](https://universe.roboflow.com/lee-sandbox/hard-hat-workers) |
| 训练集 / 验证集 | 5269 张 / 1766 张 |
| 类别 | `head` / `helmet` / `person` |
| Epochs | 100 |
| 图像尺寸 | 640 × 640 |
| Batch size | 8 |
| 训练环境 | RTX 3060 Laptop (8GB) / CUDA 11.8 |
| 训练耗时 | **46.8 分钟** |

### 关键指标（最佳 epoch 95）

| 指标 | 数值 |
|---|---|
| **mAP@0.5（总体）** | **0.650** |
| mAP@0.5:0.95 | 0.434 |
| Precision | 0.726 |
| Recall | 0.622 |

### 分类别表现（最终验证）

| 类别 | mAP@0.5 | Precision | Recall | 说明 |
|---|---|---|---|---|
| **helmet（安全帽）** | **0.970** | 0.947 | 0.931 | 核心类别，表现优异 |
| **head（未戴帽人头）** | **0.948** | 0.916 | 0.920 | 核心类别，表现优异 |
| person（人体） | 0.033 | 0.334 | 0.013 | **标注样本不足**，改用 COCO 模型 |

### 推理性能

| 指标 | 数值 |
|---|---|
| 推理速度 | **2.8 ms/张**（GPU，batch=1） |
| 处理帧率 | 约 **16 FPS**（双模型 + 640×480 视频） |

> 完整训练曲线见 `runs/detect/train-4/results.png`，
> 指标数据见 `runs/detect/train-4/results.csv`。

---

## 功能模块

| 文件 | 功能 |
|---|---|
| `ehs_detect.py` | 主检测脚本：PPE 判定 + 危险区域 + 声音报警 + 截图 |
| `roi_tool.py` | 禁区标定工具：鼠标点选多边形，保存为 json |
| `export_ledger.py` | EHS 违规台账导出：统计汇总 + Excel |

### 主检测脚本用法

```bash
# 视频检测（自动加载同名禁区 json）
python ehs_detect.py --source test_videos/demo.mp4 --show --save-video

# 摄像头实时检测 + 手动指定禁区
python ehs_detect.py --source 0 --roi "300,150,700,150,700,450,300,450" --show

# 单张图片
python ehs_detect.py --source area_dangerous/1.jpg
```

运行中按键：

| 按键 | 功能 |
|---|---|
| `空格` | 暂停 / 继续 |
| `S` | 手动截图 |
| `Q` | 退出 |

### 禁区标定工具

```bash
python roi_tool.py --video test_videos/demo.mp4
# 鼠标左键加点 → 右键撤销 → Enter 保存 json → R 重画 → Q 退出
```

保存的 json 与原项目 `Smart_Construction` 格式完全兼容，位于 `area_dangerous/area_labels/`。

### 违规台账导出

```bash
python export_ledger.py
# 输出: output/EHS违规台账.xlsx（5 个工作表）
```

台账包含：按类型汇总、按日期汇总、按区域汇总、日期类型交叉表、违规明细。

---

## 声音报警设计

按违规类型用不同音效区分，便于现场人员听声辨位：

| 违规类型 | 声音文件 | 播放次数 | 听感 |
|---|---|---|---|
| **PPE 违规**（未戴安全帽） | `Windows Ding.wav` | 1 次 | 🔔 短促"叮"一声 |
| **危险区域闯入** | `Alarm05.wav` | 5 次 | 🚨 连续"嘟——嘟——嘟" |
| **两者同时** | `Alarm05.wav` | 8 次 | 🚨🚨 持续急促警报 |

可通过 `--alarm-cooldown` 调整报警最小间隔（默认 2.0 秒），`--no-alarm` 关闭声音。

---

## 与原开源项目的差异

| 维度 | 原项目 Smart_Construction | 本项目 |
|---|---|---|
| **检测模型** | YOLOv5 v2.x（2020） | **YOLOv11n**（Ultralytics 统一框架） |
| **训练框架** | 自维护 YOLOv5 代码库 | Ultralytics 官方 API |
| **数据集** | SHWD | **Hard Hat Workers**（Roboflow） |
| **训练记录** | 仅给出最终 mAP | **完整 100 轮训练曲线与逐轮指标** |
| **PPE 合规判定** | ❌ 仅输出检测框 | ✅ **IoU 匹配判定"有头无帽"违规** |
| **危险区域** | 单张图片 + 静态 json | **视频流逐帧实时 + 多边形 ROI** |
| **人体检测** | 使用自训练 person 类（效果差） | **COCO 预训练模型协同** |
| **声音报警** | ❌ 无 | ✅ 分级报警，可区分违规类型 |
| **违规记录** | 仅保存检测结果图 | ✅ **违规截图独立归档** |
| **EHS 台账** | ❌ 无 | ✅ **Excel 台账自动导出** |
| **交互控制** | 仅退出 | 暂停 / 截图 / 退出 |

---

## 目录结构

```
smart-site-safety-monitor/
├── ehs_detect.py              # 主检测脚本
├── roi_tool.py                # 禁区标定工具
├── export_ledger.py           # 台账导出脚本
├── requirements.txt           # 依赖清单
├── README.md
├── docs/
│   ├── training_log.md        # 完整训练过程与指标
│   └── usage.md               # 环境配置与运行指南
├── demo/
│   ├── demo.mp4               # 演示视频
│   └── screenshots/           # 效果截图
├── weights/
│   └── best.pt                # PPE 训练权重（5.3 MB）
├── runs/detect/train-4/       # 训练产物（曲线、混淆矩阵、results.csv）
├── area_dangerous/
│   ├── area_labels/           # 禁区 json 标注
│   └── *.jpg                  # 示例图片
└── test_videos/               # 测试视频（不含在仓库中，需自行准备）
```

---

## 环境配置

```bash
# Python 3.10
python -m venv sc_env
sc_env\Scripts\activate

# PyTorch（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 依赖
pip install ultralytics pandas openpyxl
```

硬件：NVIDIA GPU（本项目在 RTX 3060 Laptop 6GB 上完成训练与测试）

---

## 已知限制

| 限制 | 说明 |
|---|---|
| **危险区域需人工标定** | 系统判断"人是否越界"，不自动识别"哪里危险"；符合电子围栏的实际部署方式 |
| **危险源类型单一** | 目前仅覆盖 PPE 与区域闯入；基坑、高处作业、烟火等需扩展模型 |
| **person 类别依赖外部模型** | 自建数据集中该类别标注不足，采用 COCO 预训练模型 |
| **图片拼接视频的局限** | 演示视频由独立图片合成，禁区坐标无法对应真实场景的空间关系 |
| **不替代人工巡检** | 定位为辅助工具，报警仍需人工核实与处置 |

---

## 数据集来源

- **Hard Hat Workers**：https://universe.roboflow.com/lee-sandbox/hard-hat-workers（CC BY 4.0）
- **COCO 预训练权重**：Ultralytics 官方 `yolo11n.pt`

> 数据集未包含在本仓库中，请按上述链接自行下载。

---

## 参考与致谢

- [PeterH0323/Smart_Construction](https://github.com/PeterH0323/Smart_Construction) —— 原开源项目，提供禁区标注与画框等基础实现思路
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) —— YOLOv11 训练与推理框架
- [njvisionpower/Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) —— SHWD 安全帽数据集

