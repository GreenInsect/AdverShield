# AdverShield · 对抗样本防御系统

> 实时摄像头 → YOLO目标检测 → SAC对抗补丁净化 · WebRTC + Socket.IO + FastAPI

---

## 系统架构

```
浏览器 (WebRTC摄像头采集)
    │  Socket.IO binary frame push
    ▼
FastAPI 后端 (uvicorn)
    ├─ YOLO 推理 (ultralytics)
    ├─ SAC 补丁净化 (patch_detector.py)
    └─ 人体图片叠加
    │  Socket.IO processed frame
    ▼
浏览器 Canvas 实时展示
```

## 目录结构

```
.
├── cfg
│   └── yolov2.cfg
├── cfg.py
├── darknet.py
├── data
│   ├── coco.names
│   └── voc.names
├── frontend
│   ├── index.html
│   └── static
│       ├── css
│       │   └── style.css
│       ├── images
│       │   └── overlays
│       │       ├── class_detection.png
│       │       ├── hat_red.png
│       │       ├── mask_blue.png
│       │       ├── patch11.jpg
│       │       ├── patch_green.png
│       │       └── star_gold.png
│       └── js
│           └── app.js
├── load_data.py
├── main.py
├── median_pool.py
├── models
│   ├── patch_processor.pth
│   └── yolo2.weights
├── patch_detector.py
├── README.md
├── region_loss.py
├── requirements.txt
├── start.sh
├── unet
│   ├── __init__.py
│   ├── unet_model.py
│   └── unet_parts.py
├── utils.py
├── yolov2_detect.py
```

## 快速启动

### 1. 环境要求
- Ubuntu 20.04 / 22.04
- Python 3.10+
- CUDA 11.8+ (可选，有GPU时自动使用)

### 2. 安装并启动
```bash
chmod +x start.sh
./start.sh
```

### 3. 访问系统
打开浏览器访问: **http://服务器IP:8000**

---


---

## 添加自定义覆盖图片

将 PNG/JPG 图片放入 `frontend/static/images/overlays/` 目录，
或在网页界面右侧面板点击"上传自定义图片"。

推荐使用带透明通道（RGBA）的 PNG 图片以获得最佳叠加效果。

---

## 主要功能

| 功能 | 说明 |
|------|------|
| 实时摄像头检测 | WebRTC采集，Socket.IO传输 |
| YOLO模型选择 | YOLOv8 n/s/m/l/x + YOLOv5 |
| 置信度调节 | 0.1 ~ 0.95 实时调整 |
| SAC补丁净化 | 可开关，净化后检测 |
| 人体图片叠加 | 支持头部/胸部/全身三种位置 |
| 自定义上传图片 | PNG/JPG/GIF/WebP |
| 实时统计HUD | FPS、延迟、检测数量 |
| 截图保存 | 一键保存处理后画面 |

---

## 端口与部署

默认监听 `0.0.0.0:8000`，如需修改：

```bash
uvicorn main:socket_app --host 0.0.0.0 --port 8080 --workers 1
```

**注意**：Socket.IO 模式下 workers 必须为 1，否则使用 Redis adapter。
