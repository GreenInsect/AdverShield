"""
AdverShield - Adversarial Patch Defense System
FastAPI + Socket.IO Backend  —— 支持 WebRTC 摄像头 & Carla 仿真双模式
"""

import asyncio
import base64
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from ultralytics import YOLO
from yolov2_detect import YOLO as YOLO2
from patch_detector import PatchDetector
from load_data import PatchTransformer, PatchApplier
from carla_manager import CarlaManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AdverShield")

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
applier     = PatchApplier().cuda()

carla_mgr = CarlaManager()

YOLO_MODELS = {
    "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt", "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt", "yolov5n": "yolov5nu.pt",
    "yolov5s": "yolov5su.pt", "yolov5m": "yolov5mu.pt",
    "yolov2":  ""
}
loaded_models: dict = {}
yolov2_model = YOLO2("cfg/yolov2.cfg", "models/yolo2.weights")

def get_model(name):
    if name == "yolov2": return yolov2_model
    if name not in loaded_models:
        loaded_models[name] = YOLO(YOLO_MODELS.get(name, f"{name}.pt"))
    return loaded_models[name]

OVERLAY_DIR = os.path.join(os.path.dirname(__file__), "frontend", "static", "images", "overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)
overlay_cache: dict = {}

def load_overlay(name):
    if name in overlay_cache: return overlay_cache[name]
    path = os.path.join(OVERLAY_DIR, name)
    if not os.path.exists(path): return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None: overlay_cache[name] = img
    return img

def detect_and_remove_patch(tensor):
    with torch.no_grad():
        x_processed, _, _ = SAC_processor(tensor, bpda=True, shape_completion=False)
        return x_processed

def purify_patch_bgr(bgr):
    img0  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = np.stack([img0], axis=0).astype(np.float32) / 255.0
    image = torch.tensor(image.transpose(0, 3, 1, 2)).to(device)
    x     = detect_and_remove_patch(image)
    sac   = np.asarray(x[0].cpu().detach()).transpose(1, 2, 0)
    sac   = np.clip(sac, 0, 1)
    return cv2.cvtColor((sac * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def apply_overlay_pos(frame, box, overlay_img, position="head"):
    if overlay_img is None: return frame
    x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    pw,ph = x2-x1, y2-y1
    if   position=="head":  tw,th,tx,ty = int(pw*.7),int(ph*.3), x1+(pw-int(pw*.7))//2, y1
    elif position=="chest": tw,th,tx,ty = int(pw*.6),int(ph*.25),x1+(pw-int(pw*.6))//2, y1+int(ph*.4)
    elif position=="full":  tw,th,tx,ty = pw,ph,x1,y1
    else:                   tw,th,tx,ty = int(pw*.5),int(ph*.2), x1+(pw-int(pw*.5))//2, y1
    if tw<=0 or th<=0: return frame
    ovr = cv2.resize(overlay_img, (tw, th))
    if ovr.ndim==3 and ovr.shape[2]==4:
        alpha = ovr[:,:,3:4]/255.0; ov_bgr = ovr[:,:,:3]
    else:
        alpha = np.ones((th,tw,1),dtype=np.float32); ov_bgr = ovr
    fy1,fy2 = max(ty,0),min(ty+th,frame.shape[0])
    fx1,fx2 = max(tx,0),min(tx+tw,frame.shape[1])
    oy1,ox1 = fy1-ty, fx1-tx
    oy2,ox2 = oy1+(fy2-fy1), ox1+(fx2-fx1)
    if fy2<=fy1 or fx2<=fx1: return frame
    roi = frame[fy1:fy2,fx1:fx2].astype(np.float32)
    blended = ov_bgr[oy1:oy2,ox1:ox2].astype(np.float32)*alpha[oy1:oy2,ox1:ox2] + roi*(1-alpha[oy1:oy2,ox1:ox2])
    frame[fy1:fy2,fx1:fx2] = blended.astype(np.uint8)
    return frame

def apply_overlay_auto(frame, box, adv_patch_tensor):
    if adv_patch_tensor is None: return frame
    fh,fw = frame.shape[:2]
    img_t = transforms.ToTensor()(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
    x1,y1,x2,y2 = box
    lab = torch.zeros((1,14,5)).cuda()
    lab[0,0] = torch.tensor([0,(x1+x2)/2/fw,(y1+y2)/2/fh,(x2-x1)/fw,(y2-y1)/fh])
    if lab.size(1)>1: lab[0,1:]=1
    with torch.no_grad():
        ab = transformer(adv_patch_tensor, lab, fw, do_rotate=True, rand_loc=False)
        if ab.shape[-2]!=fh or ab.shape[-1]!=fw:
            s=ab.shape; ab=ab.view(s[0]*s[1],s[2],s[3],s[4])
            ab=torch.nn.functional.interpolate(ab,size=(fh,fw),mode='nearest')
            ab=ab.view(s[0],s[1],s[2],fh,fw)
        out = applier(img_t, ab)
    p = out.squeeze(0).cpu().numpy().transpose(1,2,0)
    return cv2.cvtColor(np.clip(p*255,0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def process_frame(frame, state):
    purified     = state.get("purify", False)
    model        = get_model(state["model"])
    show_overlay = state.get("show_overlay") and state.get("overlay")

    if show_overlay:
        pre = model(frame, conf=state.get("confidence",0.4), classes=[0], verbose=False)
        ov  = load_overlay(state["overlay"])
        if ov is not None:
            ov_t = transforms.ToTensor()(cv2.cvtColor(ov[:,:,:3] if ov.shape[2]==4 else ov,
                                                       cv2.COLOR_BGR2RGB)).cuda()
            for r in pre:
                for box in r.boxes:
                    if int(box.cls[0])==0:
                        frame = apply_overlay_auto(frame, box.xyxy[0].tolist(), ov_t)

    if purified:
        frame = purify_patch_bgr(frame)

    results    = model(frame, conf=state.get("confidence",0.4), classes=[0], verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            detections.append({"bbox":[x1,y1,x2,y2],"conf":round(conf,3),"label":label})
            if state.get("show_boxes",True):
                color = (0,255,120) if purified else (0,200,255)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
                lt = f"{label} {conf:.2f}"
                (tw,th),_ = cv2.getTextSize(lt,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
                cv2.rectangle(frame,(int(x1),int(y1)-th-8),(int(x1)+tw+4,int(y1)),color,-1)
                cv2.putText(frame,lt,(int(x1)+2,int(y1)-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(10,10,10),1)
    return frame, detections, purified

def draw_status(frame, state, detections, purified, t0):
    lines = [f"Model:{state['model']}", f"Persons:{len(detections)}",
             f"Purify:{'ON' if purified else 'OFF'}", f"FPS:{1/(time.time()-t0+1e-6):.1f}"]
    for i,l in enumerate(lines):
        cv2.putText(frame,l,(10,22+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,l,(10,22+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

# ─── Socket.IO ────────────────────────────────────────────────────────────────
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*",
                            max_http_buffer_size=10*1024*1024,
                            ping_timeout=120,    # 延长超时，避免 Carla 推帧时心跳失败
                            ping_interval=60)
session_state: dict = {}
carla_tasks:   dict = {}

@sio.event
async def connect(sid, environ):
    session_state[sid] = {
        "model":"yolov8n","purify":False,"overlay":None,"overlay_position":"head",
        "show_overlay":False,"show_boxes":True,"confidence":0.4,"frame_count":0,"source":"webcam"
    }
    await sio.emit("connected", {"sid":sid,"device":str(device)}, to=sid)

@sio.event
async def disconnect(sid):
    _stop_carla_task(sid)
    session_state.pop(sid, None)

@sio.event
async def update_settings(sid, data):
    if sid in session_state: session_state[sid].update(data)

@sio.event
async def video_frame(sid, data):
    if sid not in session_state: return
    state = session_state[sid]
    t0    = time.time()
    try:
        if isinstance(data, str):
            img_bytes = base64.b64decode(data.split(",")[-1] if "," in data else data)
        else:
            img_bytes = bytes(data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: return
        frame, detections, purified = process_frame(frame, state)
        draw_status(frame, state, detections, purified, t0)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        state["frame_count"] += 1
        await sio.emit("processed_frame", {
            "frame":      "data:image/jpeg;base64,"+base64.b64encode(buf.tobytes()).decode(),
            "detections": detections, "purified": purified,
            "latency_ms": round((time.time()-t0)*1000,1), "frame_count": state["frame_count"],
        }, to=sid)
    except Exception as e:
        logger.error(f"video_frame error: {e}", exc_info=True)

@sio.event
async def carla_connect(sid, data):
    """连接 carla_server.py（独立的 Python 3.7 进程，监听 :7100）"""
    if not carla_mgr._connected:
        ok, msg = await carla_mgr.connect()
        if not ok:
            await sio.emit("carla_status", {"ok": False, "msg": f"无法连接 carla_server: {msg}"}, to=sid)
            return
    await sio.emit("carla_status", {"ok": True, "msg": "Carla 已连接，场景就绪"}, to=sid)
    _stop_carla_task(sid)
    carla_tasks[sid] = asyncio.create_task(_carla_push_loop(sid))


@sio.event
async def carla_disconnect(sid, data=None):
    _stop_carla_task(sid)
    await carla_mgr.disconnect()
    await sio.emit("carla_status", {"ok": False, "msg": "Carla 已断开"}, to=sid)


@sio.event
async def carla_control(sid, data):
    if not carla_mgr._connected: return
    t = data.get("type")
    if   t == "auto_drive": await carla_mgr.set_auto_drive(data.get("enabled", False))
    elif t == "key":        await carla_mgr.set_key(data.get("key", ""), data.get("pressed", False))


async def _carla_push_loop(sid):
    loop = asyncio.get_event_loop()
    logger.info(f"[Carla] push loop started sid={sid}")
    frame_count_local  = 0
    last_brake_state   = False   # 上一帧的刹车状态，避免重复发送

    while sid in session_state:
        state = session_state.get(sid)
        if not state: break
        t0 = time.time()
        try:
            frame = await carla_mgr.wait_for_frame(timeout=2.0)
            if frame is None:
                logger.warning("[Carla] wait_for_frame 超时")
                continue
            frame_count_local += 1
            if frame_count_local <= 3:
                logger.info(f"[Carla] 收到第 {frame_count_local} 帧 shape={frame.shape}")

            frame, detections, purified = await loop.run_in_executor(None, process_frame, frame, state)
            draw_status(frame, state, detections, purified, t0)

            # ── 自动驾驶防撞逻辑 ──────────────────────────────────────────────
            # 只在自动驾驶模式下生效；手动模式下不干预
            vinfo = await carla_mgr.get_vehicle_info()
            is_auto = vinfo.get("auto_drive", False) if vinfo else False

            if is_auto:
                person_detected = len(detections) > 0
                # 只在状态变化时发送，避免每帧都发 HTTP 请求
                if person_detected != last_brake_state:
                    await carla_mgr.send_emergency_brake(person_detected)
                    last_brake_state = person_detected
                    logger.info(f"[AutoBrake] {'🛑 检测到人，刹车' if person_detected else '▶ 无人，恢复前进'}")
            else:
                # 切换回手动时重置刹车状态
                if last_brake_state:
                    await carla_mgr.send_emergency_brake(False)
                    last_brake_state = False

            # ── 画面叠加车辆状态 ──────────────────────────────────────────────
            if vinfo:
                spd     = f"SPD: {vinfo.get('speed_kmh', 0):.1f} km/h"
                e_brake = vinfo.get("emergency_brake", False)
                mod     = ("🛑 BRAKING" if e_brake else "AUTO") if is_auto else "MANUAL"
                mod_col = (0,0,255) if e_brake else ((0,255,136) if is_auto else (0,200,255))
                cx = frame.shape[1] - 200
                for txt, y, col in [(spd, 28, (255,200,0)), (mod, 52, mod_col)]:
                    cv2.putText(frame,txt,(cx,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),3,cv2.LINE_AA)
                    cv2.putText(frame,txt,(cx,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,1,cv2.LINE_AA)

                # 检测到人时在画面顶部显示红色警告横幅
                if e_brake and state.get("show_boxes", True):
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 180), -1)
                    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
                    cv2.putText(frame, "⚠  PERSON DETECTED — AUTO BRAKING",
                                (w//2 - 230, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), 2, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            state["frame_count"] += 1
            await sio.emit("processed_frame", {
                "frame":           "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                "detections":      detections,
                "purified":        purified,
                "latency_ms":      round((time.time()-t0)*1000, 1),
                "frame_count":     state["frame_count"],
                "vehicle":         vinfo,
                "emergency_brake": last_brake_state,
            }, to=sid)
        except asyncio.CancelledError: break
        except Exception as e:
            logger.error(f"Carla push error: {e}", exc_info=True)
            await asyncio.sleep(0.05)

def _stop_carla_task(sid):
    task = carla_tasks.pop(sid, None)
    if task and not task.done(): task.cancel()

# ─── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="AdverShield", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATIC_DIR   = os.path.join(os.path.dirname(__file__), "frontend", "static")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root(): return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/api/models")
async def list_models(): return {"models": list(YOLO_MODELS.keys())}

@app.get("/api/overlays")
async def list_overlays():
    return {"overlays": [f for f in os.listdir(OVERLAY_DIR)
                         if f.lower().endswith((".png",".jpg",".jpeg",".gif",".webp"))]}

@app.post("/api/overlays/upload")
async def upload_overlay(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".png",".jpg",".jpeg",".gif",".webp"}: raise HTTPException(400,"不支持的格式")
    content = await file.read()
    with open(os.path.join(OVERLAY_DIR, file.filename), "wb") as f: f.write(content)
    overlay_cache.pop(file.filename, None)
    return {"status":"ok","filename":file.filename}

@app.delete("/api/overlays/{filename}")
async def delete_overlay(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(OVERLAY_DIR, safe)
    if not os.path.exists(path): raise HTTPException(404,"文件不存在")
    os.remove(path); overlay_cache.pop(safe, None)
    return {"status":"ok","filename":safe}

@app.get("/api/status")
async def status():
    return {"device":str(device),"cuda_available":torch.cuda.is_available(),
            "loaded_models":list(loaded_models.keys()),
            "active_sessions":len(session_state),"carla_connected":carla_mgr._connected}

@app.get("/api/carla/vehicle")
async def vehicle_info(): return carla_mgr.get_vehicle_info()

socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    try: get_model("yolov8n")
    except Exception as e: logger.warning(f"Pre-load failed: {e}")
    uvicorn.run("main:socket_app", host="0.0.0.0", port=8000, reload=False, workers=1,
                ws_ping_interval=None,   # 禁用 websockets keepalive ping，避免高负载下 AssertionError
                ws_ping_timeout=None)