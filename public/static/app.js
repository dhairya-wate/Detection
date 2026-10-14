/**
 * ecovision — minimal text & white theme waste detection
 */

(() => {
  // ── State ──────────────────────────────────────────────────────────────────
  const state = {
    isStreaming: false,
    isProcessing: false,
    selectedDeviceId: '',
    stream: null,
    detectIntervalTimer: null,
    targetFps: 6,
    confThreshold: 0.25,
    soundAlertsEnabled: true,
    lastDetections: [],
    rulesMap: {},
    lastFrameTimestamp: performance.now(),
    fpsHistory: [],
    audioCtx: null,
  };

  // ── DOM Elements ───────────────────────────────────────────────────────────
  const video = document.getElementById('cameraVideo');
  const canvas = document.getElementById('overlayCanvas');
  const ctx = canvas.getContext('2d');
  const videoContainer = document.getElementById('videoContainer');
  const standbyOverlay = document.getElementById('standbyOverlay');
  const btnToggleCamera = document.getElementById('btnToggleCamera');
  const btnCameraText = document.getElementById('btnCameraText');
  const cameraSelect = document.getElementById('cameraSelect');
  const confSlider = document.getElementById('confSlider');
  const confValueDisplay = document.getElementById('confValueDisplay');
  const soundToggle = document.getElementById('soundToggle');
  const btnSnapshot = document.getElementById('btnSnapshot');
  const btnFullscreen = document.getElementById('btnFullscreen');
  const fpsDisplay = document.getElementById('fpsDisplay');
  const latencyDisplay = document.getElementById('latencyDisplay');
  const detectionsCountBadge = document.getElementById('detectionsCountBadge');
  const detectionDrawer = document.getElementById('detectionDrawer');
  const drawerCount = document.getElementById('drawerCount');
  const detectionsList = document.getElementById('detectionsList');
  const tacoGuideModal = document.getElementById('tacoGuideModal');
  const btnOpenGuide = document.getElementById('btnOpenGuide');
  const btnCloseGuide = document.getElementById('btnCloseGuide');
  const rulesTableBody = document.getElementById('rulesTableBody');
  const tabCamera = document.getElementById('tabCamera');
  const tabUpload = document.getElementById('tabUpload');
  const uploadStage = document.getElementById('uploadStage');
  const fileInput = document.getElementById('fileInput');
  const uploadedImagePreview = document.getElementById('uploadedImagePreview');
  const uploadCanvas = document.getElementById('uploadOverlayCanvas');
  const uploadCtx = uploadCanvas ? uploadCanvas.getContext('2d') : null;
  const dropzoneText = document.getElementById('dropzoneText');
  const serverOfflineBanner = document.getElementById('serverOfflineBanner');

  // Off-screen canvas for capturing video frames
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });

  // ── Audio Feedback Synthesizer ─────────────────────────────────────────────
  function playDetectionBeep() {
    if (!state.soundAlertsEnabled) return;
    try {
      if (!state.audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        state.audioCtx = new AudioContext();
      }
      if (state.audioCtx.state === 'suspended') {
        state.audioCtx.resume();
      }
      const osc = state.audioCtx.createOscillator();
      const gain = state.audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(659.25, state.audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(880, state.audioCtx.currentTime + 0.08);
      gain.gain.setValueAtTime(0.04, state.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, state.audioCtx.currentTime + 0.12);
      osc.connect(gain);
      gain.connect(state.audioCtx.destination);
      osc.start();
      osc.stop(state.audioCtx.currentTime + 0.12);
    } catch {
      // Audio policy catch
    }
  }

  // ── Rules & Model Status ───────────────────────────────────────────────────
  async function loadRulesAndStatus() {
    try {
      const res = await fetch('/rules');
      if (res.ok) {
        const data = await res.json();
        state.rulesMap = data.rules || {};
        renderRulesTable(data.rules || {});
      }
    } catch (err) {
      console.warn('Could not fetch rules:', err);
    }

    try {
      const healthRes = await fetch('/health');
      if (healthRes.ok) {
        if (serverOfflineBanner) serverOfflineBanner.style.display = 'none';
      } else {
        if (serverOfflineBanner) serverOfflineBanner.style.display = 'block';
      }
    } catch {
      if (serverOfflineBanner) serverOfflineBanner.style.display = 'block';
    }
  }

  function renderRulesTable(rules) {
    if (!rulesTableBody) return;
    rulesTableBody.innerHTML = '';
    for (const [mat, info] of Object.entries(rules)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 500;">${mat}</td>
        <td><span class="${info.recyclable ? 'status-yes' : 'status-no'}">${info.recyclable ? 'yes' : 'no'}</span></td>
        <td><span class="${info.biodegradable ? 'status-yes' : 'status-no'}">${info.biodegradable ? 'yes' : 'no'}</span></td>
        <td style="color: #444; font-family: var(--font-mono);">${info.bin || 'general'}</td>
      `;
      rulesTableBody.appendChild(tr);
    }
  }

  // ── Camera Enumeration & Control ───────────────────────────────────────────
  async function enumerateCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      if (!cameraSelect) return;
      cameraSelect.innerHTML = '';

      if (videoDevices.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.text = 'default camera';
        cameraSelect.appendChild(opt);
        return;
      }

      videoDevices.forEach((device, idx) => {
        const opt = document.createElement('option');
        opt.value = device.deviceId;
        opt.text = device.label || `camera ${idx + 1}`;
        cameraSelect.appendChild(opt);
      });

      if (!state.selectedDeviceId && videoDevices.length > 0) {
        state.selectedDeviceId = videoDevices[0].deviceId;
      }
    } catch (err) {
      console.warn('Enumerate devices failed:', err);
    }
  }

  async function startCamera() {
    if (state.stream) {
      stopCamera();
    }

    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'environment',
      },
      audio: false,
    };

    if (state.selectedDeviceId) {
      constraints.video.deviceId = { exact: state.selectedDeviceId };
    }

    try {
      state.stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = state.stream;
      await video.play();

      state.isStreaming = true;
      if (standbyOverlay) standbyOverlay.classList.add('hidden');
      if (btnCameraText) btnCameraText.innerText = '[ stop camera ]';
      btnToggleCamera.classList.add('stop');

      await enumerateCameras();
      startDetectionLoop();
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Unable to access camera: ' + err.message);
    }
  }

  function stopCamera() {
    state.isStreaming = false;
    if (state.detectIntervalTimer) {
      clearInterval(state.detectIntervalTimer);
      state.detectIntervalTimer = null;
    }
    if (state.stream) {
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
    }
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (standbyOverlay) standbyOverlay.classList.remove('hidden');
    if (btnCameraText) btnCameraText.innerText = '[ start camera ]';
    btnToggleCamera.classList.remove('stop');
    updateDetectionsList([]);
  }

  // ── Detection Loop ─────────────────────────────────────────────────────────
  function startDetectionLoop() {
    if (state.detectIntervalTimer) {
      clearInterval(state.detectIntervalTimer);
    }

    const intervalMs = Math.round(1000 / state.targetFps);
    state.detectIntervalTimer = setInterval(async () => {
      if (!state.isStreaming || state.isProcessing || video.readyState < 2) return;
      await captureAndDetectFrame();
    }, intervalMs);
  }

  async function captureAndDetectFrame() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || video.readyState < 2) return;

    if (captureCanvas.width !== vw || captureCanvas.height !== vh) {
      captureCanvas.width = vw;
      captureCanvas.height = vh;
    }
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw;
      canvas.height = vh;
    }

    state.isProcessing = true;
    const t0 = performance.now();

    captureCtx.drawImage(video, 0, 0, vw, vh);
    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);

    try {
      const res = await fetch('/predict_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          conf_thres: state.confThreshold,
          iou_thres: 0.45,
        }),
      });

      if (res.ok) {
        if (serverOfflineBanner) serverOfflineBanner.style.display = 'none';
        const data = await res.json();
        const latency = Math.round(performance.now() - t0);

        if (latencyDisplay) latencyDisplay.innerText = `${data.inference_ms || latency} ms`;
        updateFpsCounter();

        if (data.count > 0 && state.lastDetections.length === 0) {
          playDetectionBeep();
        }

        state.lastDetections = data.detections || [];
        renderSegmentation(ctx, canvas, data.detections || []);
        updateDetectionsList(data.detections || []);
      } else {
        if (serverOfflineBanner) serverOfflineBanner.style.display = 'block';
      }
    } catch (err) {
      if (serverOfflineBanner) serverOfflineBanner.style.display = 'block';
      console.warn('Frame detection network error:', err);
    } finally {
      state.isProcessing = false;
    }
  }

  function updateFpsCounter() {
    const now = performance.now();
    const delta = now - state.lastFrameTimestamp;
    state.lastFrameTimestamp = now;
    if (delta > 0 && fpsDisplay) {
      const currentFps = 1000 / delta;
      state.fpsHistory.push(currentFps);
      if (state.fpsHistory.length > 5) state.fpsHistory.shift();
      const avgFps = (state.fpsHistory.reduce((a, b) => a + b, 0) / state.fpsHistory.length).toFixed(1);
      fpsDisplay.innerText = `${avgFps} fps`;
    }
  }

  // ── Overlay Renderer (Minimal Black & White Bounding Box & Contour) ────────
  function renderSegmentation(targetCtx, targetCanvas, detections) {
    targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
    if (!detections || detections.length === 0) return;

    detections.forEach(det => {
      const { bbox, material, confidence, biodegradable, recyclable, polygon } = det;
      const x = bbox.x1;
      const y = bbox.y1;
      const w = bbox.x2 - bbox.x1;
      const h = bbox.y2 - bbox.y1;

      // 1. Segmentation Mask (Subtle outline & translucent shade)
      if (polygon && Array.isArray(polygon) && polygon.length >= 3) {
        targetCtx.beginPath();
        targetCtx.moveTo(polygon[0][0], polygon[0][1]);
        for (let i = 1; i < polygon.length; i++) {
          targetCtx.lineTo(polygon[i][0], polygon[i][1]);
        }
        targetCtx.closePath();

        targetCtx.fillStyle = 'rgba(0, 0, 0, 0.08)';
        targetCtx.fill();

        targetCtx.lineWidth = 1;
        targetCtx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        targetCtx.stroke();
      }

      // 2. High-Contrast Bounding Box
      targetCtx.lineWidth = 1.5;
      targetCtx.strokeStyle = '#111111';
      targetCtx.strokeRect(x, y, w, h);

      // Corner Accents (Target Brackets)
      const cornerLen = Math.min(14, Math.max(5, Math.floor(w / 5)), Math.max(5, Math.floor(h / 5)));
      if (w > 12 && h > 12) {
        targetCtx.lineWidth = 2.5;
        targetCtx.strokeStyle = '#111111';

        // Top-Left
        targetCtx.beginPath();
        targetCtx.moveTo(x, y + cornerLen);
        targetCtx.lineTo(x, y);
        targetCtx.lineTo(x + cornerLen, y);
        targetCtx.stroke();

        // Top-Right
        targetCtx.beginPath();
        targetCtx.moveTo(x + w - cornerLen, y);
        targetCtx.lineTo(x + w, y);
        targetCtx.lineTo(x + w, y + cornerLen);
        targetCtx.stroke();

        // Bottom-Left
        targetCtx.beginPath();
        targetCtx.moveTo(x, y + h - cornerLen);
        targetCtx.lineTo(x, y + h);
        targetCtx.lineTo(x + cornerLen, y + h);
        targetCtx.stroke();

        // Bottom-Right
        targetCtx.beginPath();
        targetCtx.moveTo(x + w - cornerLen, y + h);
        targetCtx.lineTo(x + w, y + h);
        targetCtx.lineTo(x + w, y + h - cornerLen);
        targetCtx.stroke();
      }

      // 3. Crisp Text Label Tag
      const labelText = `${material.toLowerCase()} · ${Math.round(confidence * 100)}%`;
      targetCtx.font = '500 12px "JetBrains Mono", monospace';
      const textWidth = targetCtx.measureText(labelText).width;
      const pillWidth = textWidth + 14;
      const pillHeight = 20;
      const pillY = Math.max(y - pillHeight - 2, 2);

      // Solid white background with black border
      targetCtx.fillStyle = '#ffffff';
      targetCtx.fillRect(x, pillY, pillWidth, pillHeight);
      targetCtx.lineWidth = 1;
      targetCtx.strokeStyle = '#111111';
      targetCtx.strokeRect(x, pillY, pillWidth, pillHeight);

      // Label text
      targetCtx.fillStyle = '#111111';
      targetCtx.fillText(labelText, x + 7, pillY + 14);

      // 4. Compact Status Line inside box
      if (h > 40) {
        const statusText = `${recyclable ? 'rec' : 'landfill'} | ${biodegradable ? 'bio' : 'non-bio'}`;
        targetCtx.font = '400 10px "JetBrains Mono", monospace';
        const stWidth = targetCtx.measureText(statusText).width + 8;
        const tagY = y + h - 16;

        targetCtx.fillStyle = '#ffffff';
        targetCtx.fillRect(x + 2, tagY, stWidth, 14);
        targetCtx.strokeStyle = '#e5e5e5';
        targetCtx.strokeRect(x + 2, tagY, stWidth, 14);

        targetCtx.fillStyle = recyclable ? '#15803d' : '#475569';
        targetCtx.fillText(statusText, x + 6, tagY + 11);
      }
    });
  }

  // ── Detections Readout (Pure Minimal Text List) ─────────────────────────────
  function updateDetectionsList(detections) {
    if (detectionsCountBadge) {
      detectionsCountBadge.innerText = `${detections.length} detected`;
    }

    if (!detectionDrawer) return;

    if (!detections || detections.length === 0) {
      detectionDrawer.style.display = 'none';
      if (drawerCount) drawerCount.innerText = '0';
      if (detectionsList) detectionsList.innerHTML = '';
      return;
    }

    detectionDrawer.style.display = 'block';
    if (drawerCount) drawerCount.innerText = detections.length;

    if (!detectionsList) return;
    detectionsList.innerHTML = '';

    detections.forEach(det => {
      const row = document.createElement('div');
      row.className = 'text-det-row';

      row.innerHTML = `
        <div class="det-line-primary">
          <span class="det-name">${det.material}</span>
          <span class="det-conf">${Math.round(det.confidence * 100)}% conf</span>
        </div>
        <div class="det-line-secondary">
          <span>recyclable: ${det.recyclable ? 'yes' : 'no'}</span>
          <span>·</span>
          <span>bin: ${det.bin || 'general'}</span>
          <span>·</span>
          <span>bio: ${det.biodegradable ? 'yes' : 'no'}</span>
        </div>
      `;
      detectionsList.appendChild(row);
    });
  }

  // ── Snapshot Capture ───────────────────────────────────────────────────────
  function captureSnapshot() {
    if (!state.isStreaming) {
      alert('Start the camera before capturing a snapshot.');
      return;
    }

    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = video.videoWidth;
    snapCanvas.height = video.videoHeight;
    const sCtx = snapCanvas.getContext('2d');

    sCtx.drawImage(video, 0, 0, snapCanvas.width, snapCanvas.height);
    sCtx.drawImage(canvas, 0, 0, snapCanvas.width, snapCanvas.height);

    const link = document.createElement('a');
    link.download = `waste_snapshot_${Date.now()}.png`;
    link.href = snapCanvas.toDataURL('image/png');
    link.click();
  }

  // ── Image Upload Mode ──────────────────────────────────────────────────────
  if (tabCamera) {
    tabCamera.addEventListener('click', () => {
      tabCamera.classList.add('active');
      tabCamera.setAttribute('aria-selected', 'true');
      tabUpload.classList.remove('active');
      tabUpload.setAttribute('aria-selected', 'false');
      if (videoContainer) videoContainer.style.display = 'flex';
      if (uploadStage) uploadStage.classList.remove('active');
    });
  }

  if (tabUpload) {
    tabUpload.addEventListener('click', () => {
      tabUpload.classList.add('active');
      tabUpload.setAttribute('aria-selected', 'true');
      tabCamera.classList.remove('active');
      tabCamera.setAttribute('aria-selected', 'false');
      if (videoContainer) videoContainer.style.display = 'none';
      if (uploadStage) uploadStage.classList.add('active');
      if (state.isStreaming) stopCamera();
    });
  }

  if (uploadStage && fileInput) {
    uploadStage.addEventListener('click', () => fileInput.click());

    uploadStage.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadStage.style.borderColor = '#111111';
    });

    uploadStage.addEventListener('dragleave', () => {
      uploadStage.style.borderColor = '';
    });

    uploadStage.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadStage.style.borderColor = '';
      if (e.dataTransfer.files.length > 0) {
        handleUploadedFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleUploadedFile(e.target.files[0]);
      }
    });
  }

  async function handleUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (jpeg, png, webp).');
      return;
    }

    const reader = new FileReader();
    reader.onload = async (evt) => {
      if (!uploadedImagePreview) return;
      uploadedImagePreview.src = evt.target.result;
      uploadedImagePreview.style.display = 'block';
      if (dropzoneText) dropzoneText.style.display = 'none';

      uploadedImagePreview.onload = async () => {
        if (!uploadCanvas || !uploadCtx) return;
        uploadCanvas.width = uploadedImagePreview.naturalWidth;
        uploadCanvas.height = uploadedImagePreview.naturalHeight;

        const formData = new FormData();
        formData.append('file', file);

        try {
          const res = await fetch('/predict', {
            method: 'POST',
            body: formData,
          });

          if (res.ok) {
            const data = await res.json();
            if (latencyDisplay) latencyDisplay.innerText = `${data.inference_ms} ms`;
            renderSegmentation(uploadCtx, uploadCanvas, data.detections || []);
            updateDetectionsList(data.detections || []);
          }
        } catch (err) {
          console.error('Upload predict failed:', err);
        }
      };
    };
    reader.readAsDataURL(file);
  }

  // ── Event Listeners ────────────────────────────────────────────────────────
  if (btnToggleCamera) {
    btnToggleCamera.addEventListener('click', () => {
      if (state.isStreaming) {
        stopCamera();
      } else {
        startCamera();
      }
    });
  }

  if (cameraSelect) {
    cameraSelect.addEventListener('change', (e) => {
      state.selectedDeviceId = e.target.value;
      if (state.isStreaming) {
        startCamera();
      }
    });
  }

  if (confSlider) {
    confSlider.addEventListener('input', (e) => {
      state.confThreshold = parseFloat(e.target.value);
      if (confValueDisplay) {
        confValueDisplay.innerText = `${Math.round(state.confThreshold * 100)}%`;
      }
    });
  }

  if (soundToggle) {
    soundToggle.addEventListener('change', (e) => {
      state.soundAlertsEnabled = e.target.checked;
    });
  }

  if (btnSnapshot) {
    btnSnapshot.addEventListener('click', captureSnapshot);
  }

  if (btnFullscreen) {
    btnFullscreen.addEventListener('click', () => {
      const container = document.getElementById('cameraViewport') || videoContainer;
      if (!document.fullscreenElement) {
        container.requestFullscreen().catch(err => console.warn(err.message));
      } else {
        document.exitFullscreen();
      }
    });
  }

  if (btnOpenGuide && tacoGuideModal) {
    btnOpenGuide.addEventListener('click', () => {
      tacoGuideModal.classList.add('open');
    });
  }

  if (btnCloseGuide && tacoGuideModal) {
    btnCloseGuide.addEventListener('click', () => {
      tacoGuideModal.classList.remove('open');
    });
  }

  if (tacoGuideModal) {
    tacoGuideModal.addEventListener('click', (e) => {
      if (e.target === tacoGuideModal) {
        tacoGuideModal.classList.remove('open');
      }
    });
  }

  // ── Initialize ─────────────────────────────────────────────────────────────
  async function init() {
    await loadRulesAndStatus();
    await enumerateCameras();
  }

  init();
})();
