"""
carla_server.py — Carla 独立进程（Python 3.7 环境运行）
异步模式，无需手动 tick。
新增：
  - 在车前方 50m / 100m 精确放置静止行人
  - POST /emergency_brake  接受 main.py 的紧急刹车/恢复指令
"""

import asyncio
import base64
import logging
import math
import queue
import threading
import time

import carla
import cv2
import numpy as np
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("CarlaServer")

# ── 配置 ──────────────────────────────────────────────────────────────────────
CARLA_HOST    = "127.0.0.1"
CARLA_PORT    = 2000
SERVE_HOST    = "0.0.0.0"
SERVE_PORT    = 7100
CAM_W, CAM_H  = 1280, 720
PED_DISTANCES = [50, 100]   # 车前方放人的距离（米）

# ── 全局状态 ──────────────────────────────────────────────────────────────────
client       = None
world        = None
vehicle      = None
camera       = None
peds         = []           # 场景中所有行人

frame_queue  = queue.Queue(maxsize=4)
ws_clients   = set()

auto_drive       = False    # 是否自动前进
emergency_brake  = False    # main.py 检测到人时置 True，让控制循环刹车
keys         = {"throttle": False, "brake": False, "left": False, "right": False}
ctrl_running = False

_frame_event = threading.Event()
_latest_bgr  = None
_frame_total = 0


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _forward_location(transform, distance):
    """沿 transform 的朝向前进 distance 米，返回新的 carla.Location。"""
    yaw_rad = math.radians(transform.rotation.yaw)
    loc = transform.location
    return carla.Location(
        x = loc.x + distance * math.cos(yaw_rad),
        y = loc.y + distance * math.sin(yaw_rad),
        z = loc.z
    )


def _spawn_ped_at(bp_lib, location, ped_bp=None):
    """在指定位置生成一个静止行人，返回 actor 或 None。"""
    if ped_bp is None:
        ped_bps = bp_lib.filter("walker.pedestrian.*")
        ped_bp  = ped_bps[0]

    # 将位置投影到导航网格，确保落地
    nav_loc = world.get_random_location_from_navigation()
    # 用指定 x,y 但保留导航网格的 z
    spawn_tf = carla.Transform(
        carla.Location(x=location.x, y=location.y, z=location.z + 0.5),
        carla.Rotation()
    )
    ped = world.try_spawn_actor(ped_bp, spawn_tf)
    if ped is None:
        # 稍微偏移再试
        for dz in [1.0, 2.0]:
            spawn_tf.location.z = location.z + dz
            ped = world.try_spawn_actor(ped_bp, spawn_tf)
            if ped: break
    return ped


# ── Carla 初始化 ──────────────────────────────────────────────────────────────
def init_carla():
    global client, world, vehicle, camera, peds, ctrl_running

    logger.info(f"连接 Carla {CARLA_HOST}:{CARLA_PORT} ...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(15.0)
    world  = client.get_world()
    logger.info(f"已连接，地图: {world.get_map().name}")

    # 强制异步模式
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        time.sleep(0.5)
    logger.info("仿真模式: 异步")

    bp_lib    = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()
    if not spawn_pts:
        raise RuntimeError("地图没有 spawn points")

    # ── 车辆 ──────────────────────────────────────────────────────────────────
    candidates = bp_lib.filter("vehicle.tesla.model3") or bp_lib.filter("vehicle.*")
    vbp = candidates[0]
    vbp.set_attribute("role_name", "hero")
    vehicle = None
    for sp in spawn_pts[:10]:
        vehicle = world.try_spawn_actor(vbp, sp)
        if vehicle:
            logger.info(f"车辆 spawned: {vehicle.type_id}  位置=({sp.location.x:.1f},{sp.location.y:.1f})")
            break
    if vehicle is None:
        raise RuntimeError("车辆 spawn 失败")

    # ── 摄像头 ────────────────────────────────────────────────────────────────
    cbp = bp_lib.find("sensor.camera.rgb")
    cbp.set_attribute("image_size_x", str(CAM_W))
    cbp.set_attribute("image_size_y", str(CAM_H))
    cbp.set_attribute("fov", "90")
    cbp.set_attribute("sensor_tick", "0.05")
    camera = world.spawn_actor(
        cbp,
        carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=-5)),
        attach_to=vehicle
    )
    camera.listen(_on_frame)
    logger.info("摄像头已挂载")

    # ── 在车前方精确放置行人 ──────────────────────────────────────────────────
    time.sleep(0.3)   # 等车稳定后再取 transform
    vehicle_tf = vehicle.get_transform()
    ped_bps    = bp_lib.filter("walker.pedestrian.*")
    logger.info(f"车辆朝向 yaw={vehicle_tf.rotation.yaw:.1f}°，"
                f"在前方 {PED_DISTANCES} m 处放置行人")

    for i, dist in enumerate(PED_DISTANCES):
        loc = _forward_location(vehicle_tf, dist)
        ped_bp = ped_bps[i % len(ped_bps)]
        ped = _spawn_ped_at(bp_lib, loc, ped_bp)
        if ped:
            peds.append(ped)
            logger.info(f"行人 {i+1} spawned: 距离={dist}m  "
                        f"位置=({loc.x:.1f},{loc.y:.1f})  id={ped.id}")
        else:
            logger.warning(f"行人 {i+1} (距离{dist}m) spawn 失败，尝试备用位置")
            # 备用：在行人所在 lane 上方再试
            for doff in [2, 4, -2, -4]:
                loc2 = _forward_location(vehicle_tf, dist + doff)
                ped = _spawn_ped_at(bp_lib, loc2, ped_bp)
                if ped:
                    peds.append(ped)
                    logger.info(f"  备用位置成功：偏移={doff}m  id={ped.id}")
                    break
            else:
                logger.error(f"  行人 {i+1} 所有备用位置均失败")

    logger.info(f"场景就绪：车辆 1 辆，行人 {len(peds)} 个")

    # ── 启动控制线程 ──────────────────────────────────────────────────────────
    ctrl_running = True
    threading.Thread(target=_ctrl_loop, daemon=True).start()
    logger.info("车辆控制线程已启动")


# ── 摄像头回调 ────────────────────────────────────────────────────────────────
def _on_frame(image):
    global _latest_bgr, _frame_total
    try:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        _latest_bgr = bgr
        _frame_event.set()
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        try: frame_queue.put_nowait(bgr)
        except: pass
        _frame_total += 1
        if _frame_total <= 3 or _frame_total % 200 == 0:
            logger.info(f"[Camera] 第 {_frame_total} 帧  frame_id={image.frame}")
    except Exception as e:
        logger.error(f"_on_frame 异常: {e}", exc_info=True)


# ── 车辆控制循环 ──────────────────────────────────────────────────────────────
def _ctrl_loop():
    """
    优先级：emergency_brake > auto_drive > 手动按键
    main.py 检测到人时发 emergency_brake=True → 刹车
    检测不到人时发 emergency_brake=False → 恢复油门
    """
    logger.info("控制循环启动")
    tick = 0
    while ctrl_running and vehicle is not None:
        try:
            ctrl = carla.VehicleControl()
            ctrl.hand_brake = False
            ctrl.reverse    = False

            if emergency_brake:
                # YOLO 检测到人 → 紧急刹车
                ctrl.throttle = 0.0
                ctrl.brake    = 1.0
                ctrl.steer    = 0.0
            elif auto_drive:
                # 自动前进
                ctrl.throttle = 0.6
                ctrl.brake    = 0.0
                ctrl.steer    = 0.0
            else:
                # 手动控制
                ctrl.throttle = 0.6  if keys["throttle"] else 0.0
                ctrl.brake    = 1.0  if keys["brake"]    else 0.0
                ctrl.steer    = (-0.5 if keys["left"] else 0.5 if keys["right"] else 0.0)

            vehicle.apply_control(ctrl)
            tick += 1
            if tick <= 3 or tick % 200 == 0:
                v   = vehicle.get_velocity()
                spd = 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5
                logger.info(f"[ctrl] tick={tick} auto={auto_drive} "
                            f"brake={emergency_brake} throttle={ctrl.throttle:.1f} "
                            f"speed={spd:.1f}km/h")
        except Exception as e:
            logger.error(f"控制循环异常: {e}", exc_info=True)
        time.sleep(0.05)
    logger.info("控制循环退出")


# ── 车辆信息 ──────────────────────────────────────────────────────────────────
def get_vehicle_info():
    if not vehicle:
        return {}
    try:
        v   = vehicle.get_velocity()
        t   = vehicle.get_transform()
        spd = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        return {
            "speed_kmh":      round(spd, 1),
            "location":       {"x": round(t.location.x, 1), "y": round(t.location.y, 1)},
            "rotation":       {"yaw": round(t.rotation.yaw, 1)},
            "auto_drive":     auto_drive,
            "emergency_brake": emergency_brake,
        }
    except Exception as e:
        logger.warning(f"get_vehicle_info 异常: {e}")
        return {}


# ── 清理 ──────────────────────────────────────────────────────────────────────
def destroy():
    global ctrl_running
    ctrl_running = False
    for a in ([camera, vehicle] if camera and vehicle else []) + peds:
        try: a.destroy()
        except: pass
    if world:
        try:
            s = world.get_settings()
            s.synchronous_mode    = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
        except: pass
    logger.info("场景已销毁")


# ── HTTP / WS 路由 ────────────────────────────────────────────────────────────
async def handle_init(request):
    if vehicle is not None:
        return web.json_response({"ok": True, "msg": "已就绪"})
    try:
        await asyncio.get_event_loop().run_in_executor(None, init_carla)
        return web.json_response({"ok": True, "msg": "场景初始化成功"})
    except Exception as e:
        logger.error(f"init_carla 失败: {e}", exc_info=True)
        return web.json_response({"ok": False, "msg": str(e)}, status=500)


async def handle_control(request):
    global auto_drive
    data = await request.json()
    t    = data.get("type")
    if t == "auto_drive":
        auto_drive = bool(data.get("enabled", False))
        logger.info(f"自动驾驶: {auto_drive}")
    elif t == "key":
        k = data.get("key", "")
        if k in keys:
            keys[k] = bool(data.get("pressed", False))
    return web.json_response({"ok": True})


async def handle_emergency_brake(request):
    """
    POST /emergency_brake
    body: {"brake": true/false}
    由 main.py 在 YOLO 检测到/未检测到行人时调用
    """
    global emergency_brake
    data = await request.json()
    emergency_brake = bool(data.get("brake", False))
    logger.info(f"[EmergencyBrake] {'🛑 刹车' if emergency_brake else '▶ 恢复前进'}")
    return web.json_response({"ok": True, "emergency_brake": emergency_brake})


async def handle_vehicle_info(request):
    info = await asyncio.get_event_loop().run_in_executor(None, get_vehicle_info)
    return web.json_response(info)


async def handle_status(request):
    return web.json_response({
        "ok":               True,
        "vehicle_spawned":  vehicle is not None,
        "ped_count":        len(peds),
        "ws_clients":       len(ws_clients),
        "frame_total":      _frame_total,
        "has_frame":        _latest_bgr is not None,
        "auto_drive":       auto_drive,
        "emergency_brake":  emergency_brake,
    })


async def handle_ws_frames(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    ws_clients.add(ws)
    logger.info(f"WS 客户端连接，共 {len(ws_clients)} 个")
    try:
        async for _ in ws:
            pass
    except Exception as e:
        logger.warning(f"WS 异常: {e}")
    finally:
        ws_clients.discard(ws)
        logger.info(f"WS 客户端断开，剩余 {len(ws_clients)} 个")
    return ws


def _encode_frame(bgr):
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


async def frame_broadcaster():
    global ws_clients
    loop     = asyncio.get_event_loop()
    sent     = 0
    last_log = time.time()
    logger.info("[broadcaster] 启动，等待摄像头回调...")
    while True:
        try:
            triggered = await loop.run_in_executor(
                None, lambda: _frame_event.wait(timeout=0.1)
            )
            if not triggered:
                await asyncio.sleep(0)
                continue
            _frame_event.clear()
            bgr = _latest_bgr
            if bgr is None or not ws_clients:
                continue
            b64  = await loop.run_in_executor(None, _encode_frame, bgr)
            dead = set()
            for ws in list(ws_clients):
                try:
                    await ws.send_str(b64)
                    sent += 1
                except Exception:
                    dead.add(ws)
            ws_clients -= dead
            if time.time() - last_log > 5:
                logger.info(f"[broadcaster] 已广播 {sent} 帧，WS={len(ws_clients)}")
                last_log = time.time()
        except Exception as e:
            logger.error(f"[broadcaster] 异常: {e}")
            await asyncio.sleep(0.05)


async def on_startup(app):
    asyncio.ensure_future(frame_broadcaster())


async def on_cleanup(app):
    destroy()


def main():
    app = web.Application()
    app.router.add_post("/init",             handle_init)
    app.router.add_post("/control",          handle_control)
    app.router.add_post("/emergency_brake",  handle_emergency_brake)
    app.router.add_get( "/vehicle_info",     handle_vehicle_info)
    app.router.add_get( "/ws/frames",        handle_ws_frames)
    app.router.add_get( "/status",           handle_status)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    logger.info(f"Carla Server 启动，监听 {SERVE_HOST}:{SERVE_PORT}")
    web.run_app(app, host=SERVE_HOST, port=SERVE_PORT, access_log=None)


if __name__ == "__main__":
    main()