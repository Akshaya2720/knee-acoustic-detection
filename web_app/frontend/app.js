/* ════════════════════════════════════════════════
   KneeGuard AI — app.js
   Handles: upload, waveform drawing, API call,
            result display, playback controls

   KEY FIX: All audio formats (MP3, OGG, FLAC, WAV)
   are decoded in-browser via Web Audio API and
   re-encoded as uncompressed PCM WAV before being
   sent to the backend — no ffmpeg required.
════════════════════════════════════════════════ */

const API_BASE = 'http://localhost:8000';

// ─── DOM refs ───
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const waveformSection = document.getElementById('waveformSection');
const waveformCanvas = document.getElementById('waveformCanvas');
const fileNameEl = document.getElementById('fileName');
const fileDurationEl = document.getElementById('fileDuration');
const clearFileBtn = document.getElementById('clearFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const playBtn = document.getElementById('playBtn');
const playIcon = document.getElementById('playIcon');
const pauseIcon = document.getElementById('pauseIcon');
const progressBar = document.getElementById('progressBar');
const timeLabel = document.getElementById('timeLabel');

const uploadCard = document.getElementById('uploadCard');
const processingCard = document.getElementById('processingCard');
const resultCard = document.getElementById('resultCard');
const errorToast = document.getElementById('errorToast');
const errorMsg = document.getElementById('errorMsg');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

// ─── State ───
let selectedFile = null;
let decodedBuffer = null;   // AudioBuffer — decoded once, reused for waveform + upload
let audioElement = null;
let audioCtx = null;

// ─── Health check ───
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const data = await res.json();
      if (data.model_ready) {
        statusDot.className = 'status-dot online';
        statusText.textContent = 'Model Ready';
      } else {
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Model Not Loaded';
      }
    } else { setOffline(); }
  } catch { setOffline(); }
}
function setOffline() {
  statusDot.className = 'status-dot offline';
  statusText.textContent = 'Backend Offline';
}
checkHealth();
setInterval(checkHealth, 10000);

// ─── Drag & Drop ───
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});
dropZone.addEventListener('click', e => { if (e.target !== browseBtn) fileInput.click(); });
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });
clearFileBtn.addEventListener('click', clearFile);

// ─── Load & Display File ───
async function loadFile(file) {
  selectedFile = file;
  decodedBuffer = null;
  fileNameEl.textContent = file.name;
  analyzeBtn.disabled = true;    // disabled until decode succeeds
  waveformSection.style.display = 'block';

  // Playback element
  const url = URL.createObjectURL(file);
  audioElement = new Audio(url);
  audioElement.addEventListener('timeupdate', onTimeUpdate);
  audioElement.addEventListener('ended', onAudioEnded);
  audioElement.addEventListener('loadedmetadata', () => {
    fileDurationEl.textContent = formatTime(audioElement.duration);
  });

  // Decode audio → draw waveform + cache buffer for WAV conversion on upload
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    const arrayBuf = await file.arrayBuffer();
    decodedBuffer = await audioCtx.decodeAudioData(arrayBuf);

    drawWaveformFromBuffer(decodedBuffer);
    analyzeBtn.disabled = false;
  } catch (err) {
    // Draw a decorative fallback waveform but still allow submission
    drawFallbackWave(waveformCanvas.getContext('2d'));
    analyzeBtn.disabled = false;
    console.warn('Audio decode warning:', err);
  }
}

function clearFile() {
  selectedFile = null; decodedBuffer = null;
  fileInput.value = '';
  if (audioElement) { audioElement.pause(); audioElement = null; }
  waveformSection.style.display = 'none';
  analyzeBtn.disabled = true;
  progressBar.style.width = '0%';
  timeLabel.textContent = '0:00';
  playIcon.style.display = 'block';
  pauseIcon.style.display = 'none';
  showCard('upload');
}

// ─── Waveform Drawing ───
function drawWaveformFromBuffer(audioBuffer) {
  const cvs = waveformCanvas;
  const ctx = cvs.getContext('2d');
  const data = audioBuffer.getChannelData(0);
  const W = cvs.offsetWidth || 760;
  const H = 80;
  cvs.width = W; cvs.height = H;

  const step = Math.ceil(data.length / W);
  const amp = H / 2;
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#00e5ff');
  grad.addColorStop(0.5, '#7c3aed');
  grad.addColorStop(1, '#06d6a0');

  ctx.clearRect(0, 0, W, H);
  ctx.beginPath();
  ctx.strokeStyle = grad;
  ctx.lineWidth = 1.5;

  for (let i = 0; i < W; i++) {
    let min = 1, max = -1;
    for (let j = 0; j < step; j++) {
      const s = data[i * step + j] || 0;
      if (s < min) min = s;
      if (s > max) max = s;
    }
    ctx.moveTo(i, amp + min * amp * 0.95);
    ctx.lineTo(i, amp + max * amp * 0.95);
  }
  ctx.stroke();
}

function drawFallbackWave(ctx) {
  const W = 760, H = 80;
  waveformCanvas.width = W; waveformCanvas.height = H;
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#00e5ff'); grad.addColorStop(1, '#7c3aed');
  ctx.strokeStyle = grad; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < W; i++) {
    const y = H / 2 + Math.sin(i * 0.08) * 20 + (Math.random() - 0.5) * 8;
    i === 0 ? ctx.moveTo(i, y) : ctx.lineTo(i, y);
  }
  ctx.stroke();
}

// ─────────────────────────────────────────────────────────
// WAV ENCODER — converts an AudioBuffer to a 16-bit PCM WAV
// Blob. This runs entirely in the browser — no server-side
// format conversion or ffmpeg required.
// ─────────────────────────────────────────────────────────
function audioBufferToWavBlob(audioBuffer) {
  const SR = 22050;   // Downsample to SR used during training
  const numCh = 1;       // Mono
  const numSamples = Math.round(audioBuffer.duration * SR);

  // Get channel data and resample to SR if needed
  const rawData = audioBuffer.getChannelData(0);
  const outData = new Int16Array(numSamples);

  // Mix down to mono + resample
  const ratio = rawData.length / numSamples;
  let maxAbs = 0;
  for (let i = 0; i < numSamples; i++) {
    const srcIdx = Math.min(Math.floor(i * ratio), rawData.length - 1);
    maxAbs = Math.max(maxAbs, Math.abs(rawData[srcIdx]));
  }
  const scale = maxAbs > 0 ? 32767 / maxAbs : 32767;
  for (let i = 0; i < numSamples; i++) {
    const srcIdx = Math.min(Math.floor(i * ratio), rawData.length - 1);
    outData[i] = Math.round(rawData[srcIdx] * scale);
  }

  // Build WAV header
  const byteRate = SR * numCh * 2;
  const blockAlign = numCh * 2;
  const dataSize = numSamples * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  function writeStr(off, str) { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); }
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);           // PCM chunk size
  view.setUint16(20, 1, true);           // PCM format
  view.setUint16(22, numCh, true);
  view.setUint32(24, SR, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);           // 16-bit
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);

  // Write PCM samples
  const byteView = new Uint8Array(buf, 44);
  for (let i = 0; i < numSamples; i++) {
    const val = outData[i];
    byteView[i * 2] = val & 0xff;
    byteView[i * 2 + 1] = (val >> 8) & 0xff;
  }

  return new Blob([buf], { type: 'audio/wav' });
}


// ─── Playback ───
playBtn.addEventListener('click', () => {
  if (!audioElement) return;
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
  if (audioElement.paused) {
    audioElement.play();
    playIcon.style.display = 'none';
    pauseIcon.style.display = 'block';
  } else {
    audioElement.pause();
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
  }
});

function onTimeUpdate() {
  if (!audioElement) return;
  const pct = (audioElement.currentTime / audioElement.duration) * 100;
  progressBar.style.width = `${pct}%`;
  timeLabel.textContent = formatTime(audioElement.currentTime);
}
function onAudioEnded() {
  playIcon.style.display = 'block';
  pauseIcon.style.display = 'none';
  progressBar.style.width = '0%';
  timeLabel.textContent = '0:00';
}
function formatTime(s) {
  if (!s || isNaN(s)) return '0:00';
  return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

// ─── Analysis ───
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;
  if (audioElement) { audioElement.pause(); playIcon.style.display = 'block'; pauseIcon.style.display = 'none'; }

  showCard('processing');
  animateSteps();

  try {
    // ── Step 1: Ensure audio is decoded ──
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    let buffer = decodedBuffer;
    if (!buffer) {
      const ab = await selectedFile.arrayBuffer();
      buffer = await audioCtx.decodeAudioData(ab);
      decodedBuffer = buffer;
    }

    // ── Step 2: Convert to 22050 Hz 16-bit PCM WAV ──
    const wavBlob = audioBufferToWavBlob(buffer);
    const wavFile = new File([wavBlob], 'signal.wav', { type: 'audio/wav' });

    // ── Step 3: POST to backend ──
    const formData = new FormData();
    formData.append('file', wavFile);

    const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    showResult(data);

  } catch (err) {
    showCard('upload');
    showError(err.message || 'Could not reach backend. Is the server running?');
  }
}

function animateSteps() {
  const steps = ['step1', 'step2', 'step3', 'step4'];
  const delays = [0, 700, 1400, 2100];
  steps.forEach((id, i) => {
    const el = document.getElementById(id);
    el.className = 'proc-step';
    setTimeout(() => {
      if (i > 0) document.getElementById(steps[i - 1]).className = 'proc-step done';
      el.className = 'proc-step active';
    }, delays[i]);
  });
}

// ─── Result Display ───
function showResult(data) {
  // If the signal doesn't look like a joint recording, show uncertain card
  if (data.is_uncertain) {
    showUncertain(data);
    return;
  }

  const isNormal = data.label === 'Normal';
  resultCard.className = `card result-card ${isNormal ? 'result-normal' : 'result-abnormal'}`;

  document.getElementById('resultEmoji').textContent = isNormal ? '✅' : '⚠️';
  document.getElementById('resultLabel').textContent = data.label;
  const badge = document.getElementById('resultBadge');
  badge.textContent = data.label;
  badge.className = `result-badge ${isNormal ? 'badge-normal' : 'badge-abnormal'}`;

  const confPct = Math.round(data.confidence * 100);
  document.getElementById('gaugePct').textContent = `${confPct}%`;
  setTimeout(() => { document.getElementById('gaugeFill').style.width = `${confPct}%`; }, 100);

  const pNorm = Math.round(data.prob_normal * 100);
  const pAbnorm = Math.round(data.prob_abnormal * 100);
  document.getElementById('probNormalVal').textContent = `${pNorm}%`;
  document.getElementById('probAbnormalVal').textContent = `${pAbnorm}%`;
  setTimeout(() => {
    document.getElementById('probNormal').style.width = `${pNorm}%`;
    document.getElementById('probAbnormal').style.width = `${pAbnorm}%`;
  }, 150);

  document.getElementById('chipModel').textContent = data.model_name || '—';
  document.getElementById('chipAcc').textContent = data.model_accuracy
    ? `${(data.model_accuracy * 100).toFixed(1)}%` : '—';

  const interpText = isNormal
    ? 'The acoustic signal shows characteristics consistent with a <strong>normal knee joint</strong>. No significant abnormal patterns were detected.'
    : 'The acoustic signal shows characteristics <strong>indicative of an abnormal knee condition</strong>. We recommend further clinical evaluation.';
  document.getElementById('interpText').innerHTML = interpText;

  showCard('result');
}

// ─── Uncertain / Invalid Signal Display ───
function showUncertain(data) {
  document.getElementById('uncertainMsg').textContent =
    data.warning || 'The uploaded audio does not appear to be a knee joint recording.';

  const pNorm = Math.round(data.prob_normal * 100);
  const pAbnorm = Math.round(data.prob_abnormal * 100);
  document.getElementById('uProbNormalVal').textContent = `${pNorm}%`;
  document.getElementById('uProbAbnormalVal').textContent = `${pAbnorm}%`;
  setTimeout(() => {
    document.getElementById('uProbNormal').style.width = `${pNorm}%`;
    document.getElementById('uProbAbnormal').style.width = `${pAbnorm}%`;
  }, 150);

  showCard('uncertain');
}

// ─── Card Visibility ───
function showCard(name) {
  uploadCard.style.display = (name === 'upload') ? 'block' : 'none';
  processingCard.style.display = (name === 'processing') ? 'block' : 'none';
  document.getElementById('uncertainCard').style.display = (name === 'uncertain') ? 'block' : 'none';
  resultCard.style.display = (name === 'result') ? 'block' : 'none';

  if (name === 'result') {
    document.getElementById('gaugeFill').style.width = '0%';
    document.getElementById('probNormal').style.width = '0%';
    document.getElementById('probAbnormal').style.width = '0%';
  }
  if (name === 'uncertain') {
    document.getElementById('uProbNormal').style.width = '0%';
    document.getElementById('uProbAbnormal').style.width = '0%';
  }
}

// ─── Analyze Another ───
document.getElementById('analyzeAnotherBtn').addEventListener('click', () => {
  clearFile(); showCard('upload');
});
document.getElementById('uncertainBackBtn').addEventListener('click', () => {
  clearFile(); showCard('upload');
});

// ─── Error Toast ───
function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.style.display = 'flex';
  setTimeout(() => { errorToast.style.display = 'none'; }, 7000);
}
