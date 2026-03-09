/**
 * AdverShield · Frontend Application
 * WebRTC Camera → Socket.IO Stream → Backend YOLO + SAC → Canvas Display
 */

'use strict';

// ─── Configuration ───────────────────────────────────────────────────────────
const CONFIG = {
  SERVER_URL: window.location.origin,
  FRAME_INTERVAL_MS: 80,      // ~12 FPS send rate
  JPEG_QUALITY: 0.75,
  VIDEO_CONSTRAINTS: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user',
  },
};

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  socket: null,
  stream: null,
  frameTimer: null,
  isRunning: false,
  selectedModel: 'yolov8n',
  selectedOverlay: null,
  overlayEnabled: false,
  purifyEnabled: false,
  showBoxes: true,
  confidence: 0.4,
  overlayPosition: 'head',
  frameCount: 0,
  lastFrameTime: 0,
  fpsHistory: [],
};

// ─── DOM Refs ─────────────────────────────────────────────────────────────────
const dom = {
  localVideo:      () => document.getElementById('localVideo'),
  outputCanvas:    () => document.getElementById('outputCanvas'),
  startOverlay:    () => document.getElementById('startOverlay'),
  startBtn:        () => document.getElementById('startBtn'),
  stopBtn:         () => document.getElementById('stopBtn'),
  captureBtn:      () => document.getElementById('captureBtn'),
  fullscreenBtn:   () => document.getElementById('fullscreenBtn'),
  statusDot:       () => document.getElementById('statusDot'),
  statusLabel:     () => document.getElementById('statusLabel'),
  deviceChip:      () => document.getElementById('deviceChip'),
  fpsChip:         () => document.getElementById('fpsChip'),
  latencyChip:     () => document.getElementById('latencyChip'),
  modelGrid:       () => document.getElementById('modelGrid'),
  modelBadge:      () => document.getElementById('modelBadge'),
  confSlider:      () => document.getElementById('confSlider'),
  confVal:         () => document.getElementById('confVal'),
  showBoxes:       () => document.getElementById('showBoxes'),
  enablePurify:    () => document.getElementById('enablePurify'),
  enableOverlay:   () => document.getElementById('enableOverlay'),
  overlayControls: () => document.getElementById('overlayControls'),
  overlayGrid:     () => document.getElementById('overlayGrid'),
  overlayPosition: () => document.getElementById('overlayPosition'),
  overlayUpload:   () => document.getElementById('overlayUpload'),
  logConsole:      () => document.getElementById('logConsole'),
  purifyStatus:    () => document.getElementById('purifyStatus'),
  detectionList:   () => document.getElementById('detectionList'),
  statPersons:     () => document.getElementById('statPersons'),
  statFrames:      () => document.getElementById('statFrames'),
  statFps:         () => document.getElementById('statFps'),
  statLatency:     () => document.getElementById('statLatency'),
  hudModel:        () => document.getElementById('hudModel'),
  hudPurify:       () => document.getElementById('hudPurify'),
  hudPersonCount:  () => document.getElementById('hudPersonCount'),
};

// ─── Logging ─────────────────────────────────────────────────────────────────
function log(msg, level = 'info') {
  const console_ = dom.logConsole();
  const entry = document.createElement('div');
  entry.className = `log-entry log-${level}`;
  const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  entry.innerHTML = `<span class="log-ts">[${ts}]</span>${msg}`;
  console_.appendChild(entry);
  console_.scrollTop = console_.scrollHeight;
  // Keep max 60 entries
  while (console_.children.length > 60) {
    console_.removeChild(console_.firstChild);
  }
}

// ─── Connection ───────────────────────────────────────────────────────────────
function initSocket() {
  log('正在连接服务器...');
  state.socket = io(CONFIG.SERVER_URL, {
    transports: ['websocket'],
    reconnectionAttempts: 10,
    reconnectionDelay: 1500,
  });

  state.socket.on('connect', () => {
    log('Socket.IO 连接成功', 'ok');
    setStatus('connected', '已连接');
  });

  state.socket.on('disconnect', () => {
    log('连接断开', 'warn');
    setStatus('error', '连接断开');
    if (state.isRunning) stopStream();
  });

  state.socket.on('connect_error', (err) => {
    log(`连接错误: ${err.message}`, 'error');
    setStatus('error', '连接失败');
  });

  state.socket.on('connected', (data) => {
    log(`服务器确认 · 设备: ${data.device}`, 'ok');
    dom.deviceChip().textContent = `DEVICE: ${data.device.toUpperCase()}`;
    loadModels();
    loadOverlays();
  });

  state.socket.on('processed_frame', handleProcessedFrame);
}

// ─── Status ───────────────────────────────────────────────────────────────────
function setStatus(type, label) {
  const dot = dom.statusDot();
  dot.className = `status-dot ${type}`;
  dom.statusLabel().textContent = label;
}

// ─── Models ───────────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    renderModelGrid(data.models);
    log(`已加载 ${data.models.length} 个模型配置`, 'ok');
  } catch (e) {
    log('模型列表加载失败', 'error');
  }
}

function renderModelGrid(models) {
  const grid = dom.modelGrid();
  grid.innerHTML = '';
  models.forEach(name => {
    const btn = document.createElement('button');
    btn.className = 'model-btn' + (name === state.selectedModel ? ' active' : '');
    btn.textContent = name;
    btn.dataset.model = name;
    btn.addEventListener('click', () => selectModel(name, btn));
    grid.appendChild(btn);
  });
}

function selectModel(name, btn) {
  // Update UI
  document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active', 'loading');
  
  state.selectedModel = name;
  dom.modelBadge().textContent = name;
  dom.hudModel().textContent = name.toUpperCase();
  
  // Emit settings
  emitSettings({ model: name });
  
  setTimeout(() => btn.classList.remove('loading'), 800);
  log(`切换模型: ${name}`, 'info');
}

// ─── Overlays ─────────────────────────────────────────────────────────────────
async function loadOverlays() {
  try {
    const res = await fetch('/api/overlays');
    const data = await res.json();
    renderOverlayGrid(data.overlays);
  } catch (e) {
    log('覆盖图片加载失败', 'warn');
  }
}

function renderOverlayGrid(overlays) {
  const grid = dom.overlayGrid();
  grid.innerHTML = '';

  // None option
  const noneItem = document.createElement('div');
  noneItem.className = 'ov-item ov-none selected';
  noneItem.innerHTML = '<span>无</span>';
  noneItem.addEventListener('click', () => selectOverlay(null, noneItem));
  grid.appendChild(noneItem);

  overlays.forEach(filename => {
    const item = document.createElement('div');
    item.className = 'ov-item';
    item.dataset.filename = filename;
    item.innerHTML = `
      <img src="/static/images/overlays/${encodeURIComponent(filename)}" alt="${filename}">
      <div class="ov-label">${filename}</div>
      <button class="ov-delete-btn" title="删除">✕</button>
    `;

    // 点击图片区域 = 选中
    item.querySelector('img').addEventListener('click', () => selectOverlay(filename, item));
    item.querySelector('.ov-label').addEventListener('click', () => selectOverlay(filename, item));

    // 点击删除按钮
    item.querySelector('.ov-delete-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteOverlay(filename, item);
    });

    grid.appendChild(item);
  });
}

async function deleteOverlay(filename, itemEl) {
  if (!confirm(`确定删除「${filename}」？`)) return;
  try {
    const res = await fetch(`/api/overlays/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    if (res.ok) {
      // 若删除的是当前选中项，重置为"无"
      if (state.selectedOverlay === filename) {
        state.selectedOverlay = null;
        emitSettings({ overlay: null });
        document.querySelector('.ov-none')?.classList.add('selected');
      }
      itemEl.remove();
      log(`已删除覆盖图: ${filename}`, 'warn');
    } else {
      const err = await res.json().catch(() => ({}));
      log(`删除失败: ${err.detail || res.status}`, 'error');
    }
  } catch (e) {
    log(`删除请求异常: ${e.message}`, 'error');
  }
}

function selectOverlay(filename, itemEl) {
  document.querySelectorAll('.ov-item').forEach(i => i.classList.remove('selected'));
  itemEl.classList.add('selected');
  state.selectedOverlay = filename;
  emitSettings({ overlay: filename });
  log(`选择覆盖图: ${filename || '无'}`, 'info');
}

// ─── Camera & Stream ──────────────────────────────────────────────────────────

/**
 * 兼容性 getUserMedia：
 *  - 标准 API (HTTPS / localhost)
 *  - 旧版前缀 API (部分老浏览器)
 *  - HTTP + 非localhost 时给出明确提示
 */
function getCompatibleGetUserMedia() {
  // 标准 API
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    return (constraints) => navigator.mediaDevices.getUserMedia(constraints);
  }
  // 旧版前缀 API
  const legacyGUM = navigator.getUserMedia
    || navigator.webkitGetUserMedia
    || navigator.mozGetUserMedia
    || navigator.msGetUserMedia;

  if (legacyGUM) {
    return (constraints) => new Promise((resolve, reject) => {
      legacyGUM.call(navigator, constraints, resolve, reject);
    });
  }
  return null;
}

async function startStream() {
  // ── 检查是否处于安全上下文 ──
  const isSecure = location.protocol === 'https:'
    || location.hostname === 'localhost'
    || location.hostname === '127.0.0.1'
    || location.hostname === '::1';

  if (!isSecure) {
    const msg = `摄像头需要 HTTPS 访问。\n\n` +
      `当前地址: ${location.href}\n\n` +
      `解决方案（任选一种）：\n` +
      `① 用 localhost 访问: http://localhost:8000\n` +
      `② 启用 HTTPS（运行 start_https.sh）\n` +
      `③ Chrome 将此地址加入白名单:\n` +
      `   chrome://flags/#unsafely-treat-insecure-origin-as-secure`;
    log('⚠ 非安全上下文，摄像头不可用', 'error');
    log('请使用 HTTPS 或 localhost 访问', 'warn');
    showHttpsModal(msg);
    return;
  }

  const getUserMedia = getCompatibleGetUserMedia();
  if (!getUserMedia) {
    log('浏览器不支持摄像头API，请升级Chrome/Firefox', 'error');
    alert('您的浏览器不支持摄像头，请使用最新版 Chrome 或 Firefox。');
    return;
  }

  try {
    log('正在申请摄像头权限...');
    state.stream = await getUserMedia({
      video: CONFIG.VIDEO_CONSTRAINTS,
      audio: false,
    });

    const video = dom.localVideo();
    video.srcObject = state.stream;
    await video.play();

    // 等待视频元数据加载以获取真实分辨率
    await new Promise((resolve) => {
      if (video.videoWidth > 0) { resolve(); return; }
      video.addEventListener('loadedmetadata', resolve, { once: true });
    });

    const canvas = dom.outputCanvas();
    canvas.width  = video.videoWidth  || 1280;
    canvas.height = video.videoHeight || 720;

    state.isRunning = true;
    dom.startOverlay().style.display = 'none';
    dom.stopBtn().disabled   = false;
    dom.captureBtn().disabled = false;

    log(`摄像头启动 · ${video.videoWidth}×${video.videoHeight}`, 'ok');
    startSendingFrames();

  } catch (err) {
    let hint = '';
    if (err.name === 'NotAllowedError')  hint = '用户拒绝了摄像头权限，请在浏览器地址栏允许摄像头。';
    if (err.name === 'NotFoundError')    hint = '未找到摄像头设备。';
    if (err.name === 'NotReadableError') hint = '摄像头被其他程序占用。';
    log(`摄像头错误 [${err.name}]: ${hint || err.message}`, 'error');
    alert(`无法访问摄像头\n\n原因: ${hint || err.message}`);
  }
}

// ── HTTPS 提示弹窗 ──────────────────────────────────────────────────────────
function showHttpsModal(msg) {
  // 移除旧弹窗
  document.getElementById('httpsModal')?.remove();

  const modal = document.createElement('div');
  modal.id = 'httpsModal';
  modal.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:9999;
    display:flex;align-items:center;justify-content:center;
    font-family:'Share Tech Mono',monospace;
  `;
  modal.innerHTML = `
    <div style="
      background:#0b1016;border:1px solid #ff3b6b;border-radius:8px;
      padding:32px;max-width:500px;width:90%;color:#c8d8e8;
    ">
      <div style="color:#ff3b6b;font-size:18px;font-weight:700;margin-bottom:16px;letter-spacing:.1em;">
        ⚠ 需要 HTTPS 访问
      </div>
      <pre style="font-size:12px;line-height:1.8;white-space:pre-wrap;color:#aaa;">${msg}</pre>
      <div style="margin-top:20px;display:flex;gap:10px;flex-wrap:wrap;">
        <button onclick="window.location.href='http://localhost:'+location.port+location.pathname"
          style="padding:8px 16px;background:#00e5ff22;border:1px solid #00e5ff;
                 border-radius:4px;color:#00e5ff;cursor:pointer;font-family:inherit;">
          切换到 localhost
        </button>
        <button onclick="document.getElementById('httpsModal').remove()"
          style="padding:8px 16px;background:#ffffff11;border:1px solid #444;
                 border-radius:4px;color:#aaa;cursor:pointer;font-family:inherit;">
          关闭
        </button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
}

function stopStream() {
  state.isRunning = false;
  clearInterval(state.frameTimer);
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  dom.startOverlay().style.display = 'flex';
  dom.stopBtn().disabled = true;
  dom.captureBtn().disabled = true;
  log('摄像头已停止', 'warn');
}

// ─── Frame Sending ────────────────────────────────────────────────────────────
function startSendingFrames() {
  const video = dom.localVideo();
  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  const ctx = captureCanvas.getContext('2d');

  state.frameTimer = setInterval(() => {
    if (!state.isRunning || !state.socket?.connected) return;
    if (video.readyState < 2) return;

    ctx.drawImage(video, 0, 0);
    const dataUrl = captureCanvas.toDataURL('image/jpeg', CONFIG.JPEG_QUALITY);
    state.socket.emit('video_frame', dataUrl);
  }, CONFIG.FRAME_INTERVAL_MS);
}

// ─── Processed Frame Handler ──────────────────────────────────────────────────
function handleProcessedFrame(data) {
  // Draw returned frame
  const canvas = dom.outputCanvas();
  const ctx = canvas.getContext('2d');
  const img = new Image();
  img.onload = () => {
    if (canvas.width !== img.width) canvas.width = img.width;
    if (canvas.height !== img.height) canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
  };
  img.src = data.frame;

  // Update stats
  const now = performance.now();
  if (state.lastFrameTime > 0) {
    const dt = now - state.lastFrameTime;
    state.fpsHistory.push(1000 / dt);
    if (state.fpsHistory.length > 20) state.fpsHistory.shift();
  }
  state.lastFrameTime = now;
  state.frameCount = data.frame_count;

  const fps = state.fpsHistory.length
    ? (state.fpsHistory.reduce((a,b) => a+b, 0) / state.fpsHistory.length).toFixed(1)
    : '—';

  dom.statPersons().textContent = data.detections.length;
  dom.statFrames().textContent = data.frame_count;
  dom.statFps().textContent = fps;
  dom.statLatency().textContent = data.latency_ms;
  dom.fpsChip().textContent = `FPS: ${fps}`;
  dom.latencyChip().textContent = `LAT: ${data.latency_ms}ms`;
  dom.hudPersonCount().textContent = `${data.detections.length} PERSONS`;

  // Update detection list
  renderDetections(data.detections);
}

function renderDetections(detections) {
  const list = dom.detectionList();
  if (!detections.length) {
    list.innerHTML = '<div class="det-empty">暂无检测目标</div>';
    return;
  }
  list.innerHTML = detections.map((d, i) => `
    <div class="det-item">
      <span class="det-idx">${String(i+1).padStart(2,'0')}</span>
      <span class="det-label">${d.label}</span>
      <span class="det-conf">${(d.conf * 100).toFixed(0)}%</span>
    </div>
  `).join('');
}

// ─── Settings Emission ────────────────────────────────────────────────────────
function emitSettings(partial) {
  if (!state.socket?.connected) return;
  state.socket.emit('update_settings', partial);
}

// ─── Controls Wiring ─────────────────────────────────────────────────────────
function wireControls() {
  // Start / Stop
  dom.startBtn().addEventListener('click', startStream);
  dom.stopBtn().addEventListener('click', stopStream);

  // Capture screenshot
  dom.captureBtn().addEventListener('click', () => {
    const canvas = dom.outputCanvas();
    const a = document.createElement('a');
    a.download = `advershield_${Date.now()}.jpg`;
    a.href = canvas.toDataURL('image/jpeg', 0.92);
    a.click();
    log('截图已保存', 'ok');
  });

  // Fullscreen
  dom.fullscreenBtn().addEventListener('click', () => {
    const stage = document.querySelector('.center-stage');
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      stage.requestFullscreen();
    }
  });

  // Confidence slider
  dom.confSlider().addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    dom.confVal().textContent = val;
    state.confidence = parseFloat(val);
    emitSettings({ confidence: state.confidence });
  });

  // Show boxes toggle
  dom.showBoxes().addEventListener('change', (e) => {
    state.showBoxes = e.target.checked;
    emitSettings({ show_boxes: state.showBoxes });
  });

  // Purify toggle
  dom.enablePurify().addEventListener('change', (e) => {
    state.purifyEnabled = e.target.checked;
    emitSettings({ purify: state.purifyEnabled });
    updatePurifyStatus(state.purifyEnabled);
    dom.hudPurify().textContent = `PURIFY: ${state.purifyEnabled ? 'ON' : 'OFF'}`;
    log(`补丁净化: ${state.purifyEnabled ? '启用' : '禁用'}`, state.purifyEnabled ? 'ok' : 'warn');
  });

  // Overlay toggle
  dom.enableOverlay().addEventListener('change', (e) => {
    state.overlayEnabled = e.target.checked;
    emitSettings({ show_overlay: state.overlayEnabled });
    log(`图片叠加: ${state.overlayEnabled ? '启用' : '禁用'}`, 'info');
  });

  // Overlay position
  dom.overlayPosition().addEventListener('change', (e) => {
    state.overlayPosition = e.target.value;
    emitSettings({ overlay_position: state.overlayPosition });
  });

  // File upload for custom overlays
  dom.overlayUpload().addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    log(`上传覆盖图: ${file.name}...`, 'info');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/overlays/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        log(`上传成功: ${file.name}`, 'ok');
        await loadOverlays();
      } else {
        log('上传失败', 'error');
      }
    } catch {
      // Fallback: use local object URL for display (no server upload)
      log('上传接口不可用，使用本地预览', 'warn');
      const url = URL.createObjectURL(file);
      const grid = dom.overlayGrid();
      const item = document.createElement('div');
      item.className = 'ov-item';
      item.innerHTML = `<img src="${url}" alt="${file.name}"><div class="ov-label">${file.name}</div>`;
      item.addEventListener('click', () => {
        // Register with backend via a data URL approach won't work fully without server
        // but we can show selection
        selectOverlay(file.name, item);
      });
      grid.appendChild(item);
    }
  });
}

// ─── Purify Status UI ─────────────────────────────────────────────────────────
function updatePurifyStatus(active) {
  const indicator = dom.purifyStatus().querySelector('.purify-indicator');
  const text = indicator.querySelector('.pi-text');
  if (active) {
    indicator.classList.remove('inactive');
    indicator.classList.add('active');
    text.textContent = 'SAC净化模块运行中';
  } else {
    indicator.classList.add('inactive');
    indicator.classList.remove('active');
    text.textContent = '净化模块待机中';
  }
}

// ─── API Status Check ─────────────────────────────────────────────────────────
async function checkApiStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    log(`系统状态 · CUDA: ${data.cuda_available ? '可用' : '不可用'} · 设备: ${data.device}`, 'ok');
  } catch {
    log('API状态检查失败', 'warn');
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  log('AdverShield 系统初始化...', 'info');
  wireControls();
  initSocket();
  await checkApiStatus();
  log('就绪 · 请点击"开始检测"', 'ok');
}

document.addEventListener('DOMContentLoaded', init);