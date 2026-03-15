"""
carla_manager.py — 运行在 Python 3.10 主进程中
通过 HTTP + WebSocket 与 carla_server.py (Python 3.7) 通信
不直接 import carla
"""

import asyncio
import base64
import logging
from typing import Callable, Optional

import aiohttp
import cv2
import numpy as np

logger = logging.getLogger("AdverShield.CarlaClient")

CARLA_SERVER = "http://127.0.0.1:7100"
CARLA_WS     = "ws://127.0.0.1:7100/ws/frames"


class CarlaManager:
    def __init__(self):
        self._connected          = False
        self._frame_callback: Optional[Callable] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_event: Optional[asyncio.Event] = None  # 懒惰初始化，必须在事件循环内创建
        self._ws_task: Optional[asyncio.Task]    = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._vehicle_info: dict = {}
        self._auto_drive = False

    def _get_event(self) -> asyncio.Event:
        """在事件循环内懒惰创建 Event，避免绑定到错误的循环。"""
        if self._new_frame_event is None:
            self._new_frame_event = asyncio.Event()
        return self._new_frame_event

    # ── 连接 ──────────────────────────────────────────────────────────────────
    async def connect(self) -> tuple[bool, str]:
        try:
            logger.info(f"[CarlaClient] 尝试连接 {CARLA_SERVER}/init ...")
            self._session = aiohttp.ClientSession()
            async with self._session.post(
                f"{CARLA_SERVER}/init",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                data = await r.json()
                logger.info(f"[CarlaClient] /init 响应: {data}")
                if not data.get("ok"):
                    return False, data.get("msg", "初始化失败")
            self._connected = True
            # 重置 event（确保在当前事件循环中创建）
            self._new_frame_event = asyncio.Event()
            self._ws_task = asyncio.create_task(self._ws_recv_loop())
            logger.info(f"[CarlaClient] WS 任务已创建，目标: {CARLA_WS}")
            return True, data.get("msg", "已连接")
        except Exception as e:
            logger.error(f"[CarlaClient] connect 失败: {e}")
            return False, str(e)

    async def disconnect(self):
        self._connected = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
        if self._session:
            await self._session.close()
            self._session = None
        self._latest_frame = None
        logger.info("CarlaManager disconnected")

    # ── WebSocket 帧接收 ───────────────────────────────────────────────────────
    async def _ws_recv_loop(self):
        logger.info(f"[CarlaClient] WS recv loop 启动，目标={CARLA_WS}")
        frame_count = 0
        while self._connected:
            try:
                logger.info(f"[CarlaClient] 正在建立 WS 连接 {CARLA_WS} ...")
                async with aiohttp.ClientSession() as sess:
                    async with sess.ws_connect(
                        CARLA_WS,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as ws:
                        logger.info("[CarlaClient] ✓ WS 已连接，等待帧...")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                b64 = msg.data
                                img_bytes = base64.b64decode(
                                    b64.split(",")[-1] if "," in b64 else b64
                                )
                                arr   = np.frombuffer(img_bytes, np.uint8)
                                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    self._latest_frame = frame
                                    self._get_event().set()
                                    frame_count += 1
                                    if frame_count <= 3:
                                        logger.info(f"[CarlaClient] ✓ 收到第 {frame_count} 帧 shape={frame.shape}")
                                    if self._frame_callback:
                                        self._frame_callback(frame)
                                else:
                                    logger.warning("[CarlaClient] imdecode 返回 None，帧数据损坏")
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR
                            ):
                                logger.warning(f"[CarlaClient] WS msg type={msg.type}，断开重连...")
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._connected:
                    logger.warning(f"[CarlaClient] WS 连接异常 ({type(e).__name__}: {e})，1秒后重试")
                    await asyncio.sleep(1.0)
        logger.info("[CarlaClient] WS recv loop 结束")

    # ── 等待并获取最新帧（阻塞直到有新帧，最多等 timeout 秒）────────────────
    async def wait_for_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            await asyncio.wait_for(self._get_event().wait(), timeout=timeout)
            self._get_event().clear()
            return self._latest_frame
        except asyncio.TimeoutError:
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """同步获取当前缓存帧（不等待）。"""
        return self._latest_frame

    def set_frame_callback(self, cb: Callable):
        self._frame_callback = cb

    # ── 车辆控制 ──────────────────────────────────────────────────────────────
    async def send_control(self, data: dict):
        if not self._connected or not self._session:
            return
        try:
            async with self._session.post(
                f"{CARLA_SERVER}/control", json=data,
                timeout=aiohttp.ClientTimeout(total=2)
            ) as r:
                await r.json()
        except Exception as e:
            logger.debug(f"send_control error: {e}")

    async def set_auto_drive(self, enabled: bool):
        self._auto_drive = enabled
        await self.send_control({"type": "auto_drive", "enabled": enabled})

    async def set_key(self, key: str, pressed: bool):
        await self.send_control({"type": "key", "key": key, "pressed": pressed})

    async def send_emergency_brake(self, brake: bool):
        """通知 carla_server 刹车（True）或恢复前进（False）。"""
        if not self._connected or not self._session:
            return
        try:
            async with self._session.post(
                f"{CARLA_SERVER}/emergency_brake",
                json={"brake": brake},
                timeout=aiohttp.ClientTimeout(total=2)
            ) as r:
                await r.json()
        except Exception as e:
            logger.debug(f"send_emergency_brake error: {e}")

    # ── 车辆状态 ──────────────────────────────────────────────────────────────
    async def get_vehicle_info(self) -> dict:
        if not self._connected or not self._session:
            return {}
        try:
            async with self._session.get(
                f"{CARLA_SERVER}/vehicle_info",
                timeout=aiohttp.ClientTimeout(total=2)
            ) as r:
                info = await r.json()
                self._vehicle_info = info
                return info
        except Exception:
            return self._vehicle_info  # 返回上次缓存值