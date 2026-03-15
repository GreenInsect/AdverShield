'use strict';

// ─── Config ───────────────────────────────────────────────────────────────────
const CONFIG = {
  SERVER_URL:       window.location.origin,
  FRAME_INTERVAL_MS: 80,
  JPEG_QUALITY:     0.75,
  VIDEO_CONSTRAINTS: { width:{ideal:1280}, height:{ideal:720}, facingMode:'user' },
};

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  socket: null, stream: null, frameTimer: null,
  isRunning: false, source: 'webcam',           // 'webcam' | 'carla'
  carlaConnected: false, autoDrive: false,
  selectedModel: 'yolov8n', selectedOverlay: null,
  overlayEnabled: false, purifyEnabled: false,
  showBoxes: true, confidence: 0.4, overlayPosition: 'head',
  fpsHistory: [], lastFrameTime: 0,
};

// ─── DOM Helpers ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── Log ─────────────────────────────────────────────────────────────────────
function log(msg, level='info') {
  const c = $('logConsole');
  const d = document.createElement('div');
  d.className = `log-entry log-${level}`;
  const ts = new Date().toLocaleTimeString('zh-CN',{hour12:false});
  d.innerHTML = `<span class="log-ts">[${ts}]</span>${msg}`;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
  while (c.children.length > 60) c.removeChild(c.firstChild);
}

// ─── Socket ───────────────────────────────────────────────────────────────────
function initSocket() {
  log('正在连接服务器...');
  state.socket = io(CONFIG.SERVER_URL, { transports:['websocket'], reconnectionAttempts:10, reconnectionDelay:1500 });
  state.socket.on('connect',       () => { log('Socket.IO 连接成功','ok'); setStatus('connected','已连接'); });
  state.socket.on('disconnect',    () => { log('连接断开','warn'); setStatus('error','连接断开'); if(state.isRunning) stopStream(); });
  state.socket.on('connect_error', e  => { log(`连接错误: ${e.message}`,'error'); setStatus('error','连接失败'); });
  state.socket.on('connected',     d  => { log(`设备: ${d.device}`,'ok'); $('deviceChip').textContent=`DEVICE: ${d.device.toUpperCase()}`; loadModels(); loadOverlays(); });
  state.socket.on('processed_frame', handleProcessedFrame);
  state.socket.on('carla_status',    handleCarlaStatus);
}

function setStatus(type, label) {
  $('statusDot').className = `status-dot ${type}`;
  $('statusLabel').textContent = label;
}

// ─── Source Switch ────────────────────────────────────────────────────────────
function switchSource(src) {
  state.source = src;
  $('srcWebcam').classList.toggle('active', src==='webcam');
  $('srcCarla').classList.toggle('active',  src==='carla');
  $('carlaPanel').style.display = src==='carla' ? '' : 'none';
  $('carlaHud').style.display   = src==='carla' ? '' : 'none';
  $('hudSource').textContent    = src==='carla' ? 'CARLA' : 'WEBCAM';
  // 更新启动引导文字
  if (src==='carla') {
    $('startTitle').textContent = '启动 Carla 仿真';
    $('startSub').textContent   = '请先在右侧面板连接仿真器';
  } else {
    $('startTitle').textContent = '启动摄像头';
    $('startSub').textContent   = '点击开始实时对抗补丁检测';
  }
  if (state.isRunning) stopStream();
  log(`图像源切换: ${src==='carla'?'Carla仿真':'用户摄像头'}`,'info');
}

// ─── Models ───────────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const d = await (await fetch('/api/models')).json();
    const grid = $('modelGrid'); grid.innerHTML='';
    d.models.forEach(name => {
      const btn = document.createElement('button');
      btn.className = 'model-btn' + (name===state.selectedModel?' active':'');
      btn.textContent = name; btn.dataset.model = name;
      btn.addEventListener('click', () => selectModel(name, btn));
      grid.appendChild(btn);
    });
    log(`已加载 ${d.models.length} 个模型配置`,'ok');
  } catch(e) { log('模型列表加载失败','error'); }
}

function selectModel(name, btn) {
  document.querySelectorAll('.model-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active','loading');
  state.selectedModel = name;
  $('modelBadge').textContent = name;
  $('hudModel').textContent   = name.toUpperCase();
  emitSettings({model:name});
  setTimeout(()=>btn.classList.remove('loading'), 800);
  log(`切换模型: ${name}`,'info');
}

// ─── Overlays ─────────────────────────────────────────────────────────────────
async function loadOverlays() {
  try {
    const d = await (await fetch('/api/overlays')).json();
    renderOverlayGrid(d.overlays);
  } catch(e) { log('覆盖图片加载失败','warn'); }
}

function renderOverlayGrid(overlays) {
  const grid = $('overlayGrid'); grid.innerHTML='';
  const none = document.createElement('div');
  none.className='ov-item ov-none selected'; none.innerHTML='<span>无</span>';
  none.addEventListener('click',()=>selectOverlay(null,none));
  grid.appendChild(none);
  overlays.forEach(f=>{
    const item=document.createElement('div'); item.className='ov-item'; item.dataset.filename=f;
    item.innerHTML=`<img src="/static/images/overlays/${encodeURIComponent(f)}" alt="${f}"><div class="ov-label">${f}</div><button class="ov-delete-btn" title="删除">✕</button>`;
    item.querySelector('img').addEventListener('click',()=>selectOverlay(f,item));
    item.querySelector('.ov-label').addEventListener('click',()=>selectOverlay(f,item));
    item.querySelector('.ov-delete-btn').addEventListener('click',e=>{e.stopPropagation();deleteOverlay(f,item);});
    grid.appendChild(item);
  });
}

function selectOverlay(filename, itemEl) {
  document.querySelectorAll('.ov-item').forEach(i=>i.classList.remove('selected'));
  itemEl.classList.add('selected');
  state.selectedOverlay = filename;
  emitSettings({overlay:filename});
  log(`选择覆盖图: ${filename||'无'}`,'info');
}

async function deleteOverlay(filename, itemEl) {
  if (!confirm(`确定删除「${filename}」？`)) return;
  try {
    const res = await fetch(`/api/overlays/${encodeURIComponent(filename)}`,{method:'DELETE'});
    if (res.ok) {
      if(state.selectedOverlay===filename){ state.selectedOverlay=null; emitSettings({overlay:null}); document.querySelector('.ov-none')?.classList.add('selected'); }
      itemEl.remove(); log(`已删除: ${filename}`,'warn');
    }
  } catch(e) { log(`删除失败: ${e.message}`,'error'); }
}

// ─── Webcam Stream ────────────────────────────────────────────────────────────
function getCompatibleGetUserMedia() {
  if (navigator.mediaDevices?.getUserMedia) return c=>navigator.mediaDevices.getUserMedia(c);
  const gum = navigator.getUserMedia||navigator.webkitGetUserMedia||navigator.mozGetUserMedia;
  if (gum) return c=>new Promise((res,rej)=>gum.call(navigator,c,res,rej));
  return null;
}


function stopStream() {
  state.isRunning = false;
  clearInterval(state.frameTimer);
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  $("startOverlay").style.display = "flex";
  $("stopBtn").disabled = true; $("captureBtn").disabled = true;
  if (state.source === "carla") {
    log("帧流显示已暂停（Carla 仍在运行，可重新点击开始）", "warn");
  } else {
    log("已停止", "warn");
  }
}

function startSendingFrames() {
  const video=$('localVideo');
  const cap=document.createElement('canvas'); cap.width=video.videoWidth; cap.height=video.videoHeight;
  const ctx=cap.getContext('2d');
  state.frameTimer=setInterval(()=>{
    if(!state.isRunning||!state.socket?.connected) return;
    if(video.readyState<2) return;
    ctx.drawImage(video,0,0);
    state.socket.emit('video_frame', cap.toDataURL('image/jpeg',CONFIG.JPEG_QUALITY));
  }, CONFIG.FRAME_INTERVAL_MS);
}

// ─── Carla Stream ─────────────────────────────────────────────────────────────
function startCarlaStream() {
  // 预先把 canvas 设为已知尺寸，避免首帧画不出来
  const canvas = $('outputCanvas');
  if (canvas.width < 16)  canvas.width  = 1280;
  if (canvas.height < 16) canvas.height = 720;

  state.isRunning = true;
  $('startOverlay').style.display = 'none';
  $('stopBtn').disabled    = false;
  $('captureBtn').disabled = false;
  log('Carla 帧流已启动，等待后端推帧...', 'ok');
}

async function startStream() {
  // Carla 模式：点"开始检测"等同于先连接再启动
  if (state.source === 'carla') {
    if (!state.carlaConnected) {
      // 还没连接，自动触发连接流程
      await carlaConnect();
    } else {
      // 已连接，直接启动帧流
      startCarlaStream();
    }
    return;
  }

  // ── 摄像头模式（原有逻辑）──────────────────────────────────────────────
  const isSecure = location.protocol === 'https:' ||
    ['localhost', '127.0.0.1', '::1'].includes(location.hostname);
  if (!isSecure) { showHttpsModal(); return; }
  const gum = getCompatibleGetUserMedia();
  if (!gum) { alert('浏览器不支持摄像头，请使用最新版 Chrome/Firefox'); return; }
  try {
    log('正在申请摄像头权限...');
    state.stream = await gum({ video: CONFIG.VIDEO_CONSTRAINTS, audio: false });
    const video = $('localVideo');
    video.srcObject = state.stream;
    await video.play();
    await new Promise(res => {
      if (video.videoWidth > 0) res();
      else video.addEventListener('loadedmetadata', res, { once: true });
    });
    const canvas = $('outputCanvas');
    canvas.width  = video.videoWidth  || 1280;
    canvas.height = video.videoHeight || 720;
    state.isRunning = true;
    $('startOverlay').style.display = 'none';
    $('stopBtn').disabled    = false;
    $('captureBtn').disabled = false;
    log(`摄像头启动 · ${video.videoWidth}×${video.videoHeight}`, 'ok');
    startSendingFrames();
  } catch (err) {
    const hints = {
      NotAllowedError:  '请在浏览器允许摄像头权限',
      NotFoundError:    '未找到摄像头设备',
      NotReadableError: '摄像头被其他程序占用',
    };
    log(`摄像头错误 [${err.name}]: ${hints[err.name] || err.message}`, 'error');
    alert(`无法访问摄像头\n${hints[err.name] || err.message}`);
  }
}

async function carlaConnect() {
  if (!state.socket?.connected) { log('Socket 未连接', 'error'); return; }
  const host = $('carlaHost').value || '127.0.0.1';
  const port = parseInt($('carlaPort').value) || 2000;
  log(`正在连接 Carla Server ${host}:${port} ...`, 'info');
  $('carlaConnBtn').disabled = true;
  $('startBtn') && ($('startBtn').disabled = true);
  state.socket.emit('carla_connect', { host, port });
}

function carlaDisconnect() {
  state.socket?.emit('carla_disconnect', {});
  state.carlaConnected = false;
  state.isRunning      = false;
  $('startOverlay').style.display = 'flex';
  $('stopBtn').disabled     = true;
  $('captureBtn').disabled  = true;
  $('carlaBadge').textContent  = '未连接';
  $('carlaConnBtn').disabled   = false;
  $('carlaDiscBtn').disabled   = true;
  if ($('startBtn')) $('startBtn').disabled = false;
  log('Carla 已断开', 'warn');
}

function handleCarlaStatus(data) {
  if ($('startBtn')) $('startBtn').disabled = false;
  if (data.ok) {
    state.carlaConnected = true;
    $('carlaBadge').textContent  = '已连接';
    $('carlaConnBtn').disabled   = true;
    $('carlaDiscBtn').disabled   = false;
    log(data.msg, 'ok');
    // 连接成功后自动启动帧流显示
    startCarlaStream();
  } else {
    state.carlaConnected = false;
    $('carlaBadge').textContent = '未连接';
    $('carlaConnBtn').disabled  = false;
    $('carlaDiscBtn').disabled  = true;
    log(`Carla 连接失败: ${data.msg}`, 'error');
  }
}

// ─── Drive Mode ───────────────────────────────────────────────────────────────
function setDriveMode(auto) {
  state.autoDrive = auto;
  $('btnManual').classList.toggle('active',!auto);
  $('btnAuto').classList.toggle('active', auto);
  // D-pad 透明度
  $('dpad').style.opacity = auto ? '0.35' : '1';
  $('dpad').style.pointerEvents = auto ? 'none' : '';
  state.socket?.emit('carla_control',{type:'auto_drive',enabled:auto});
  log(`驾驶模式: ${auto?'自动前进':'人工控制'}`,'info');
}

// ─── D-Pad / Keyboard ─────────────────────────────────────────────────────────
function keyDown(key) {
  if (state.autoDrive) return;
  const btnMap={throttle:'dpadUp',brake:'dpadBrake',left:'dpadLeft',right:'dpadRight'};
  $(btnMap[key])?.classList.add('pressed');
  state.socket?.emit('carla_control',{type:'key',key,pressed:true});
}
function keyUp(key) {
  const btnMap={throttle:'dpadUp',brake:'dpadBrake',left:'dpadLeft',right:'dpadRight'};
  $(btnMap[key])?.classList.remove('pressed');
  state.socket?.emit('carla_control',{type:'key',key,pressed:false});
}

const KEY_MAP={'w':'throttle','arrowup':'throttle','s':'brake','arrowdown':'brake','a':'left','arrowleft':'left','d':'right','arrowright':'right'};
document.addEventListener('keydown', e => {
  if (state.source!=='carla'||state.autoDrive) return;
  const k=KEY_MAP[e.key.toLowerCase()];
  if(k){ e.preventDefault(); keyDown(k); }
});
document.addEventListener('keyup', e => {
  if (state.source!=='carla') return;
  const k=KEY_MAP[e.key.toLowerCase()];
  if(k){ e.preventDefault(); keyUp(k); }
});

// ─── Frame Handler ────────────────────────────────────────────────────────────
function handleProcessedFrame(data) {
  // 收到第一帧时确保 overlay 已隐藏
  if (!state.isRunning) {
    state.isRunning = true;
    $('startOverlay').style.display = 'none';
    $('stopBtn').disabled    = false;
    $('captureBtn').disabled = false;
    log('收到第一帧，画面已启动', 'ok');
  }
  const canvas = $('outputCanvas'), ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    // 先 resize 再 draw，避免 resize 清空画布
    if (canvas.width !== img.width || canvas.height !== img.height) {
      canvas.width  = img.width;
      canvas.height = img.height;
    }
    ctx.drawImage(img, 0, 0);
  };
  img.src = data.frame;

  const now=performance.now();
  if(state.lastFrameTime>0){ state.fpsHistory.push(1000/(now-state.lastFrameTime)); if(state.fpsHistory.length>20)state.fpsHistory.shift(); }
  state.lastFrameTime=now;
  const fps=state.fpsHistory.length?(state.fpsHistory.reduce((a,b)=>a+b,0)/state.fpsHistory.length).toFixed(1):'—';

  $('statPersons').textContent=data.detections.length;
  $('statFrames').textContent=data.frame_count;
  $('statFps').textContent=fps;
  $('statLatency').textContent=data.latency_ms;
  $('fpsChip').textContent=`FPS: ${fps}`;
  $('latencyChip').textContent=`LAT: ${data.latency_ms}ms`;
  $('hudPersonCount').textContent=`${data.detections.length} PERSONS`;

  // 更新 Carla 车辆状态
  if (data.vehicle && Object.keys(data.vehicle).length) {
    const v=data.vehicle;
    $('vSpeed').textContent=v.speed_kmh??'—';
    $('vYaw').textContent=v.rotation?.yaw??'—';
    $('carlaSpeed').textContent=`SPD: ${v.speed_kmh??'—'} km/h`;
    $('carlaMode').textContent=v.auto_drive?'AUTO':'MANUAL';
  }

  renderDetections(data.detections);
}

function renderDetections(dets) {
  const list=$('detectionList');
  if(!dets.length){ list.innerHTML='<div class="det-empty">暂无检测目标</div>'; return; }
  list.innerHTML=dets.map((d,i)=>`
    <div class="det-item">
      <span class="det-idx">${String(i+1).padStart(2,'0')}</span>
      <span class="det-label">${d.label}</span>
      <span class="det-conf">${(d.conf*100).toFixed(0)}%</span>
    </div>`).join('');
}

// ─── Settings ─────────────────────────────────────────────────────────────────
function emitSettings(partial) { state.socket?.connected && state.socket.emit('update_settings',partial); }

function updatePurifyStatus(active) {
  const ind=$('purifyStatus').querySelector('.purify-indicator');
  const txt=ind.querySelector('.pi-text');
  ind.classList.toggle('inactive',!active); ind.classList.toggle('active',active);
  txt.textContent=active?'SAC净化模块运行中':'净化模块待机中';
}

// ─── HTTPS Modal ──────────────────────────────────────────────────────────────
function showHttpsModal() {
  $('httpsModal')?.remove();
  const m=document.createElement('div'); m.id='httpsModal';
  m.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9999;display:flex;align-items:center;justify-content:center;font-family:"Share Tech Mono",monospace;';
  m.innerHTML=`<div style="background:#0b1016;border:1px solid #ff3b6b;border-radius:8px;padding:32px;max-width:480px;width:90%;color:#c8d8e8;">
    <div style="color:#ff3b6b;font-size:18px;font-weight:700;margin-bottom:16px;">⚠ 需要 HTTPS 访问</div>
    <p style="font-size:12px;line-height:1.8;color:#aaa;">摄像头需要 HTTPS 或 localhost。<br>
    ① 用 <code>http://localhost:8000</code> 访问<br>
    ② 运行 <code>./start_https.sh</code> 启用 HTTPS</p>
    <div style="margin-top:16px;display:flex;gap:8px;">
      <button onclick="window.location.href='http://localhost:'+location.port+location.pathname"
        style="padding:7px 14px;background:#00e5ff22;border:1px solid #00e5ff;border-radius:4px;color:#00e5ff;cursor:pointer;font-family:inherit;">
        切换 localhost</button>
      <button onclick="document.getElementById('httpsModal').remove()"
        style="padding:7px 14px;background:#fff1;border:1px solid #444;border-radius:4px;color:#aaa;cursor:pointer;font-family:inherit;">
        关闭</button>
    </div></div>`;
  document.body.appendChild(m);
}

// ─── Wire Controls ────────────────────────────────────────────────────────────
function wireControls() {
  $('startBtn').addEventListener('click', startStream);
  $('stopBtn').addEventListener('click', stopStream);
  $('captureBtn').addEventListener('click', ()=>{
    const a=document.createElement('a'); a.download=`advershield_${Date.now()}.jpg`;
    a.href=$('outputCanvas').toDataURL('image/jpeg',0.92); a.click();
    log('截图已保存','ok');
  });
  $('fullscreenBtn').addEventListener('click', ()=>{
    const s=document.querySelector('.center-stage');
    document.fullscreenElement ? document.exitFullscreen() : s.requestFullscreen();
  });
  $('confSlider').addEventListener('input', e=>{
    const v=parseFloat(e.target.value).toFixed(2);
    $('confVal').textContent=v; state.confidence=parseFloat(v);
    emitSettings({confidence:state.confidence});
  });
  $('showBoxes').addEventListener('change', e=>{ state.showBoxes=e.target.checked; emitSettings({show_boxes:state.showBoxes}); });
  $('enablePurify').addEventListener('change', e=>{
    state.purifyEnabled=e.target.checked; emitSettings({purify:state.purifyEnabled});
    updatePurifyStatus(state.purifyEnabled);
    $('hudPurify').textContent=`PURIFY: ${state.purifyEnabled?'ON':'OFF'}`;
    log(`补丁净化: ${state.purifyEnabled?'启用':'禁用'}`,state.purifyEnabled?'ok':'warn');
  });
  $('enableOverlay').addEventListener('change', e=>{ state.overlayEnabled=e.target.checked; emitSettings({show_overlay:state.overlayEnabled}); });
  $('overlayPosition').addEventListener('change', e=>{ state.overlayPosition=e.target.value; emitSettings({overlay_position:state.overlayPosition}); });
  $('overlayUpload').addEventListener('change', async e=>{
    const file=e.target.files[0]; if(!file) return;
    const fd=new FormData(); fd.append('file',file);
    try {
      const res=await fetch('/api/overlays/upload',{method:'POST',body:fd});
      if(res.ok){ log(`上传成功: ${file.name}`,'ok'); await loadOverlays(); }
      else log('上传失败','error');
    } catch { log('上传失败','error'); }
  });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  log('AdverShield 系统初始化...');
  wireControls();
  initSocket();
  log('就绪','ok');
}
document.addEventListener('DOMContentLoaded', init);