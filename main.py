"""
AdverShield - Adversarial Patch Defense System
FastAPI + Socket.IO Backend
"""

import asyncio
import base64
import io
import json
import logging
import os, sys
import time
from typing import Optional

import cv2
import numpy as np
import socketio
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from yolov2_detect import YOLO as YOLO2
from patch_detector import PatchDetector
from load_data import PatchTransformer, PatchApplier, InriaDataset

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger("AdverShield")

# ─── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

ckpt_path = "models/patch_processor.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[150, 100, 75, 50, 25], n_patch=1)
SAC_processor.unet.load_state_dict(torch.load(ckpt_path, map_location=device))
SAC_processor.unet.to(device)
SAC_processor.unet.eval()

transformer = PatchTransformer().cuda()
applier = PatchApplier().cuda()

# ─── Model Registry ────────────────────────────────────────────────────────────
YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    "yolov5n": "yolov5nu.pt",
    "yolov5s": "yolov5su.pt",
    "yolov5m": "yolov5mu.pt",
    "yolov2": ""
}

loaded_models: dict[str, YOLO] = {}
current_model_name: str = "yolov8n"

yolov2_model = YOLO2("cfg/yolov2.cfg", "models/yolo2.weights")

def get_model(model_name: str) -> YOLO:
    if model_name not in loaded_models:
        if model_name == "yolov2":
            return yolov2_model
        logger.info(f"Loading model: {model_name}")
        model_path = YOLO_MODELS.get(model_name, f"{model_name}.pt")
        loaded_models[model_name] = YOLO(model_path)
    return loaded_models[model_name]


# ─── Overlay Images ────────────────────────────────────────────────────────────
OVERLAY_DIR = os.path.join(os.path.dirname(__file__), "frontend", "static", "images", "overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)

overlay_cache: dict[str, np.ndarray] = {}


def load_overlay(name: str) -> Optional[np.ndarray]:
    if name in overlay_cache:
        return overlay_cache[name]
    path = os.path.join(OVERLAY_DIR, name)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        overlay_cache[name] = img
    return img


# ─── Adversarial Patch Defense ────────────────────────────────────────────────
def detect_and_remove_patch(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    输入: (1, 3, H, W) float32 [0,1]
    输出: (1, 3, H, W) float32 [0,1]  —— 与输入保持相同 shape
    """
    with torch.no_grad():
        x_processed, _, _ = SAC_processor(image_tensor, bpda=True, shape_completion=False)
        return x_processed  # (1, 3, H, W)


def purify_patch_bgr(patch_bgr: np.ndarray) -> np.ndarray:
    """
    对任意尺寸的 BGR uint8 图像块做 SAC 净化，返回同尺寸 BGR uint8。
    原始脚本的逻辑：
        image_0  = cv2.cvtColor(patch_bgr, BGR->RGB)
        image    = np.stack([image_0], axis=0).astype(float32) / 255.0   # (1,H,W,3)
        image    = image.transpose(0,3,1,2)                               # (1,3,H,W)
        x_processed, _, _ = SAC_processor(image, bpda=True, ...)
        image_sac = np.asarray(x_processed[0].cpu().detach())            # (3,H,W)
        image_sac = image_sac.transpose(1,2,0)                           # (H,W,3)
        image_sac = np.clip(image_sac, 0, 1)
    """
    # 预处理
    image_0 = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    image = np.stack([image_0], axis=0).astype(np.float32) / 255.0  # (1,H,W,3)
    image = image.transpose(0, 3, 1, 2)                               # (1,3,H,W)
    image = torch.tensor(image).to(device)

    # SAC 推理
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # start_time = time.perf_counter()

    x_processed = detect_and_remove_patch(image)  

    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()

    # end_time = time.perf_counter()

    # duration = end_time - start_time
    # print(f"处理耗时: {duration:.4f} 秒 (约 {1/duration:.2f} FPS)")                         

    # 后处理
    image_sac = np.asarray(x_processed[0].cpu().detach())            # (3,H,W)
    image_sac = image_sac.transpose(1, 2, 0)                         # (H,W,3)
    image_sac = np.clip(image_sac, 0, 1)

    # 转回 BGR uint8 供 OpenCV 使用
    return cv2.cvtColor((image_sac * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def apply_overlay_to_person_pos(frame: np.ndarray, box, overlay_img: np.ndarray,
                             position: str = "head") -> np.ndarray:
    """将覆盖图片贴到检测到的人体上，返回合成后的整帧（不做净化）。"""
    if overlay_img is None:
        return frame
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    person_w = x2 - x1
    person_h = y2 - y1

    if position == "head":
        target_w = int(person_w * 0.7)
        target_h = int(person_h * 0.3)
        tx = x1 + (person_w - target_w) // 2
        ty = y1
    elif position == "chest":
        target_w = int(person_w * 0.6)
        target_h = int(person_h * 0.25)      
        tx = x1 + (person_w - target_w) // 2
        ty = y1 + int(person_h * 0.4)
    elif position == "full":
        target_w = person_w
        target_h = person_h
        tx = x1
        ty = y1
    else:
        target_w = int(person_w * 0.5)
        target_h = int(person_h * 0.2)
        tx = x1 + (person_w - target_w) // 2
        ty = y1

    if target_w <= 0 or target_h <= 0:
        return frame

    ov_resized = cv2.resize(overlay_img, (target_w, target_h))

    # 处理 alpha 通道
    if ov_resized.ndim == 3 and ov_resized.shape[2] == 4:
        alpha = ov_resized[:, :, 3:4] / 255.0
        ov_bgr = ov_resized[:, :, :3]
    else:
        alpha = np.ones((target_h, target_w, 1), dtype=np.float32)
        ov_bgr = ov_resized

    # 计算有效区域
    fy1, fy2 = max(ty, 0), min(ty + target_h, frame.shape[0])
    fx1, fx2 = max(tx, 0), min(tx + target_w, frame.shape[1])
    oy1 = fy1 - ty
    oy2 = oy1 + (fy2 - fy1)
    ox1 = fx1 - tx
    ox2 = ox1 + (fx2 - fx1)

    if fy2 <= fy1 or fx2 <= fx1:
        return frame

    roi     = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    ov_crop = ov_bgr[oy1:oy2, ox1:ox2].astype(np.float32)
    al_crop = alpha[oy1:oy2, ox1:ox2]
    blended = ov_crop * al_crop + roi * (1 - al_crop)
    frame[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)
    return frame


def apply_overlay_to_person_auto(frame: np.ndarray, box, adv_patch_tensor, 
                                transformer, applier) -> np.ndarray:
    if adv_patch_tensor is None:
        return frame

    # 1. 准备图像：NumPy (H,W,3) BGR -> Tensor (1,3,H,W) RGB
    fh, fw = frame.shape[:2] # 此时 fh=720, fw=1280
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).cuda()

    # 2. 准备标签 (归一化坐标)
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) / fw, (y2 - y1) / fh
    bcx, bcy = (x1 + x2) / 2 / fw, (y1 + y2) / 2 / fh
    
    lab_batch = torch.zeros((1, 14, 5)).cuda()
    lab_batch[0, 0] = torch.tensor([0, bcx, bcy, bw, bh])
    if lab_batch.size(1) > 1:
        lab_batch[0, 1:] = 1

    # 3. 调用黑盒逻辑
    with torch.no_grad():
        # 这里传入 fw (1280)，transformer 会输出 (1, 14, 3, 1280, 1280) 的正方形
        adv_batch_t = transformer(adv_patch_tensor, lab_batch, fw, do_rotate=True, rand_loc=False)
        
        if adv_batch_t.shape[-2] != fh or adv_batch_t.shape[-1] != fw:
            # 展平 batch 维度以适配 F.interpolate
            s = adv_batch_t.shape # (Batch, Max_Lab, C, H, W)
            adv_batch_t = adv_batch_t.view(s[0] * s[1], s[2], s[3], s[4])
            
            # 强制调整为视频帧的真实尺寸 (720, 1280)
            adv_batch_t = torch.nn.functional.interpolate(adv_batch_t, size=(fh, fw), mode='nearest')
            
            # 还原维度
            adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], fh, fw)
        # ------------------------------

        patched_img_tensor = applier(img_tensor, adv_batch_t)

    # 4. 转回 OpenCV 格式
    patched_img = patched_img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    patched_img = np.clip(patched_img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(patched_img, cv2.COLOR_RGB2BGR)

# ─── Socket.IO ────────────────────────────────────────────────────────────────
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)

# Per-session state
session_state: dict[str, dict] = {}


@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    session_state[sid] = {
        "model": "yolov8n",
        "purify": False,
        "overlay": None,
        "overlay_position": "head",
        "show_overlay": False,
        "show_boxes": True,
        "confidence": 0.4,
        "frame_count": 0,
    }
    await sio.emit("connected", {"sid": sid, "device": str(device)}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    session_state.pop(sid, None)


@sio.event
async def update_settings(sid, data):
    if sid in session_state:
        session_state[sid].update(data)
        logger.debug(f"Settings updated for {sid}: {data}")


@sio.event
async def video_frame(sid, data):
    """接收前端推送的视频帧，处理后推回"""
    if sid not in session_state:
        return
    state = session_state[sid]
    t0 = time.time()

    try:
        # ── 解码 base64 帧 ──
        if isinstance(data, str):
            img_bytes = base64.b64decode(data.split(",")[-1] if "," in data else data)
        else:
            img_bytes = bytes(data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        purified = state.get("purify", False)
        model = get_model(state["model"])
        show_overlay = state.get("show_overlay") and state.get("overlay")

        if show_overlay:
                    # 步骤1：初步检测定位人体 bbox
            pre_results = model(frame, conf=state.get("confidence", 0.4), classes=[0], verbose=False)
            overlay_img = load_overlay(state["overlay"]) # 这是 NumPy BGR 格式

            if overlay_img is not None:
                # --- 核心修改：预处理补丁图，将 NumPy 转换为 Tensor ---
                # 1. BGR 转 RGB 
                overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                # 2. 转为 Tensor 并归一化到 [0, 1]，形状变为 (3, H, W)
                adv_patch_tensor = transforms.ToTensor()(overlay_rgb).cuda()
                # ---------------------------------------------------

                for r in pre_results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # 步骤2：使用已转换好的 adv_patch_tensor 调用自动贴图函数
                            # 注意：传入的是转换后的 Tensor 而非原始 overlay_img
                            frame = apply_overlay_to_person_auto(
                                frame, 
                                [x1, y1, x2, y2], 
                                adv_patch_tensor, 
                                transformer, 
                                applier
                            )

        # 步骤3：对完整合成帧（含覆盖图）做 SAC 净化
        if purified:
            frame = purify_patch_bgr(frame)

        # 步骤4：对净化后的合成帧做最终 YOLO 检测
        results = model(frame, conf=state.get("confidence", 0.4), classes=[0], verbose=False)

        # ── 绘制检测框 ──
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append({"bbox": [x1, y1, x2, y2], "conf": round(conf, 3), "label": label})

                if state.get("show_boxes", True):
                    color = (0, 255, 120) if purified else (0, 200, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label_text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 4, int(y1)), color, -1)
                    cv2.putText(frame, label_text, (int(x1) + 2, int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)

        # ── 状态角标 ──
        status_lines = [
            f"Model: {state['model']}",
            f"Persons: {len(detections)}",
            f"Purify: {'ON' if purified else 'OFF'}",
            f"FPS: {1/(time.time()-t0+1e-6):.1f}",
        ]
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (10, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ── 编码并推回 ──
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        out_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

        state["frame_count"] += 1
        await sio.emit("processed_frame", {
            "frame": out_b64,
            "detections": detections,
            "purified": purified,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "frame_count": state["frame_count"],
        }, to=sid)

    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)


# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="AdverShield", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "frontend", "static")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/models")
async def list_models():
    return {"models": list(YOLO_MODELS.keys()), "current": current_model_name}


@app.get("/api/overlays")
async def list_overlays():
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    files = [f for f in os.listdir(OVERLAY_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))]
    return {"overlays": files}


@app.post("/api/overlays/upload")
async def upload_overlay(file: UploadFile):
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, "不支持的文件格式")
    dest = os.path.join(OVERLAY_DIR, file.filename)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    overlay_cache.pop(file.filename, None)
    return {"status": "ok", "filename": file.filename}


@app.delete("/api/overlays/{filename}")
async def delete_overlay(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(OVERLAY_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    os.remove(path)
    overlay_cache.pop(safe_name, None)
    return {"status": "ok", "filename": safe_name}


@app.get("/api/status")
async def status():
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(loaded_models.keys()),
        "active_sessions": len(session_state),
    }


# ─── Mount Socket.IO ───────────────────────────────────────────────────────────
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    try:
        get_model("yolov8n")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")

    uvicorn.run(
        "main:socket_app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,
    )