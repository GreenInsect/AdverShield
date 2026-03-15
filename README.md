# AdverShield 🛡️

> **对抗补丁防御系统** — 基于 SAC 算法的实时对抗样本检测与净化平台，支持摄像头实时流与 Carla 自动驾驶仿真双模式。

---

## 项目简介

AdverShield 是一个面向对抗样本攻防研究的可视化演示平台。系统可以：

- 实时接收摄像头视频流或 Carla 仿真摄像头帧
- 向检测目标（行人）动态叠加对抗补丁（Adversarial Patch）
- 使用 **SAC（Shape-Completion Adversarial Cleaner）** 算法检测并净化对抗补丁
- 使用 YOLO 系列模型进行行人目标检测
- 在 Carla 仿真场景中演示：**贴补丁的行人无法被识别 → 车辆不停 → 启用 SAC 净化 → 识别到行人 → 车辆自动刹车**

---

## 演示效果

```
自动驾驶 + 补丁叠加 + SAC 关闭
  → YOLO 被欺骗，检测不到人 → 车辆继续前进 ✅

自动驾驶 + 补丁叠加 + SAC 开启
  → SAC 去除补丁 → YOLO 检测到人 → 车辆自动紧急刹车 🛑
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         浏览器前端                               │
│   WebRTC 摄像头 / Carla 帧显示 | 控制面板 | 实时检测结果         │
└────────────────────────┬────────────────────────────────────────┘
                         │ Socket.IO / WebSocket
┌────────────────────────▼────────────────────────────────────────┐
│                    main.py（Python 3.10）                        │
│   FastAPI + Socket.IO | YOLO 推理 | SAC 净化 | 补丁叠加          │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP + WebSocket（端口 7100）
┌────────────────────────▼────────────────────────────────────────┐
│                carla_server.py（Python 3.7）                     │
│   Carla 场景管理 | 摄像头帧推送 | 车辆控制 | 紧急刹车接口         │
└────────────────────────┬────────────────────────────────────────┘
                         │ Carla Client API（端口 2000）
┌────────────────────────▼────────────────────────────────────────┐
│              Carla 仿真器（Docker，端口 2000）                    │
│   Town10HD_Opt 地图 | Tesla Model3 | 前方 50m/100m 静止行人       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目文件结构

```
AdverShield/
├── main.py                    # 主服务：FastAPI + Socket.IO，YOLO/SAC 推理
├── carla_server.py            # Carla 独立服务（Python 3.7 环境运行）
├── carla_manager.py           # Carla HTTP/WS 客户端（Python 3.10 调用）
├── patch_detector.py          # SAC PatchDetector 模型定义
├── yolov2_detect.py           # YOLOv2 推理封装
├── load_data.py               # PatchTransformer / PatchApplier / InriaDataset
├── cfg/
│   └── yolov2.cfg             # YOLOv2 网络配置
├── models/
│   ├── patch_processor.pth    # SAC 模型权重（需自行提供）
│   └── yolo2.weights          # YOLOv2 权重（需自行提供）
├── frontend/
│   ├── index.html             # 前端主页面
│   └── static/
│       ├── css/style.css      # 暗色赛博朋克 HUD 风格
│       ├── js/app.js          # 前端逻辑：WebRTC、Socket.IO、Carla 控制
│       └── images/overlays/   # 用户上传的对抗补丁图片目录
└── requirements.txt           # Python 3.10 依赖
```

---

## 环境依赖

### Python 环境说明

本项目需要**两个独立的 Python 环境**，因为 Carla 客户端库仅支持 Python 3.7，而推理框架需要 Python 3.10+。

| 环境 | Python 版本 | 用途 |
|------|-------------|------|
| `AdverShield`（主环境） | 3.10+ | main.py、YOLO、SAC、FastAPI |
| `carla_37`（Carla 环境） | 3.7 | carla_server.py、Carla 客户端 |

### Python 3.10 依赖（主环境）

```bash
pip install -r requirements.txt
```

或者

```
conda env create -f AdverShield_env.yaml
```

### Python 3.7 依赖（Carla 环境）

```bash
conda create -f carla_37_env.yaml
```

### Carla 仿真器

Carla 仿真器本体需单独下载（约 12GB）：

```
https://github.com/carla-simulator/carla/releases/tag/0.9.13
```

本项目使用 Docker 运行 Carla：

```bash
docker pull carlasim/carla:0.9.13
```

---

## 模型权重

以下权重文件需自行准备，放置于 `models/` 目录：

| 文件 | 说明 |
|------|------|
| `models/yolo2.weights` | YOLOv2 预训练权重 |

YOLO v8/v5 系列权重（`yolov8n.pt` 等）会在首次运行时由 ultralytics 自动下载。

---

## 快速开始

### 第一步：启动 Carla 仿真器

```bash
docker run --privileged --gpus all \
  --net=host \
  -it carlasim/carla:0.9.13 \
  ./CarlaUE4.sh -opengl -RenderOffScreen
```

### 第二步：启动 Carla 服务（Python 3.7 环境）

```bash
conda activate carla_37
cd /path/to/AdverShield
python carla_server.py
```

启动成功后监听 `http://0.0.0.0:7100`，并自动连接仿真器、生成场景（车辆 + 前方 50m/100m 各一个行人）。

### 第三步：启动主服务（Python 3.10 环境）

```bash
conda activate AdverShield
cd /path/to/AdverShield
python main.py
```

服务启动后监听 `http://0.0.0.0:8000`。

### 第四步：打开浏览器

访问 `http://localhost:8000`



---

## 功能使用说明

### 摄像头模式

1. Header 处选择 **摄像头** 模式
2. 点击 **◉ 开始检测** 授权摄像头
3. 在右侧面板上传对抗补丁图片，开启**图片叠加**
4. 开启 **SAC 净化**，观察净化前后的检测差异

### Carla 仿真模式

1. Header 处切换到 **Carla仿真** 模式
2. 在右侧 Carla 控制面板填写服务器地址（默认 `127.0.0.1`），点击**连接仿真器**
3. 连接成功后画面自动显示车载摄像头视角
4. 点击**自动前进**，车辆开始向前行驶
5. 开启**图片叠加**将对抗补丁贴到行人身上
6. 此时 YOLO 被欺骗，检测不到行人，车辆不停
7. 开启 **SAC净化**，净化后 YOLO 识别到行人，车辆自动紧急刹车

### 手动驾驶控制

| 操作 | 键盘 | 按钮 |
|------|------|------|
| 前进 | `W` 或 `↑` | ▲ |
| 刹车/后退 | `S` 或 `↓` | ▼ / ■ |
| 左转 | `A` 或 `←` | ◀ |
| 右转 | `D` 或 `→` | ▶ |

---

## 支持的检测模型

| 模型 | 说明 |
|------|------|
| YOLOv8n/s/m/l/x | Ultralytics YOLOv8 系列 |
| YOLOv5n/s/m | Ultralytics YOLOv5 系列 |
| YOLOv2 | 自定义 YOLOv2 实现 |

---

## 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 2000 | Carla 仿真器 | Carla 原生通信端口，固定不变 |
| 7100 | carla_server.py | Carla 桥接服务，供 main.py 调用 |
| 8000 | main.py | 主服务，浏览器访问入口（HTTP） |


## 技术栈

| 层次 | 技术 |
|------|------|
| 前端 | HTML5 / CSS3 / JavaScript / Socket.IO / WebRTC |
| 后端 | FastAPI / Python-SocketIO / Uvicorn / aiohttp |
| 推理 | PyTorch / Ultralytics YOLO / OpenCV |
| 仿真 | Carla 0.9.13 / Docker |
| 防御算法 | SAC（Shape-Completion Adversarial Cleaner） |
