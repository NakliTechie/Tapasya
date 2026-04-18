'use strict';

// ============================================================
// TAPASYA — main thread
// Owns: UI state, worker lifecycle, chart rendering, DOM updates
// Never touches model weights directly.
// ============================================================

// ---- State ----
const S = {
  mode: null,          // 'guided' | 'free'
  corpus: '',
  corpusLocked: false,
  worker: null,
  stages: {
    1: { status: 'idle', step: 0, steps: 500, history: [], samples: [], unlocked: true,  cfg: {} },
    2: { status: 'idle', step: 0, steps: 2000, history: [], samples: [], unlocked: false, cfg: {} },
    3: { status: 'idle', step: 0, steps: 2000, history: [], samples: [], unlocked: false, cfg: {} }
  },
  activeStage: 1,
  compareMode: false,
  compareUnlocked: false,
  workerReady: false,
};

// ---- Worker ----
function spawnWorker() {
  if (S.worker) S.worker.terminate();
  const blob = new Blob([`importScripts(location.origin + '/worker.js');`], { type: 'application/javascript' });
  // Use a direct URL instead — worker.js is same-origin
  S.worker = new Worker('worker.js');
  S.worker.addEventListener('message', onWorkerMessage);
  S.worker.addEventListener('error', e => {
    console.error('Worker error:', e.message);
    showBanner('Worker error: ' + e.message, 'error');
  });
}

function workerPost(msg) {
  if (!S.worker) spawnWorker();
  S.worker.postMessage(msg);
}

// ---- Worker → Main ----
function onWorkerMessage(e) {
  const m = e.data;
  switch (m.type) {
    case 'ready':
      S.stages[m.stage].status = 'ready';
      updateStageUI(m.stage);
      onStageReady(m.stage);
      break;

    case 'checkpoint_found':
      offerResume(m.stage, m.step);
      break;

    case 'progress':
      onProgress(m);
      break;

    case 'sample':
      onSample(m);
      break;

    case 'checkpoint_saved':
      flashStatus(m.stage, `checkpoint saved @ step ${m.step}`);
      break;

    case 'paused':
      S.stages[m.stage].status = 'paused';
      S.stages[m.stage].step   = m.step;
      updateStageUI(m.stage);
      break;

    case 'done':
      onStageDone(m);
      break;

    case 'generated':
      onGenerated(m);
      break;

    case 'unlock':
      if (m.unlocks === 'compare') {
        S.compareUnlocked = true;
        updateCompareButton();
      } else {
        S.stages[m.unlocks].unlocked = true;
        updateStageUI(m.unlocks);
        if (S.mode === 'guided') showUnlockBanner(m.unlocks);
      }
      break;

    case 'reset_done':
      onResetDone(m.stage);
      break;

    case 'status':
      flashStatus(m.stage, m.message);
      break;

    case 'gpu_ready':
      console.log('[tapasya] WebGPU backend active');
      showBanner('WebGPU active — training will be accelerated.', 'ok');
      break;

    case 'error':
      console.error('[tapasya worker]', m.message);
      showBanner('Error: ' + m.message, 'error');
      if (m.stage) { S.stages[m.stage].status = 'error'; updateStageUI(m.stage); }
      break;
  }
}

// ---- Corpus ----
function lockCorpus() {
  S.corpusLocked = true;
  $('corpus-input').setAttribute('disabled', '');
  $('corpus-lock-msg').textContent = 'Corpus locked — reset all stages to change it.';
}

// ---- Init stage ----
function sendInit(stage) {
  const st  = S.stages[stage];
  const cfg = { ...st.cfg, steps: st.steps };
  workerPost({ type: 'init', stage, config: cfg, corpus: S.corpus, seed: 42 });
}

// ---- Controls ----
function startStage(stage) {
  if (!S.corpus.trim()) { showBanner('Paste a corpus first.', 'warn'); return; }
  if (!S.corpusLocked) lockCorpus();

  const st = S.stages[stage];
  if (st.status === 'idle' || st.status === 'error') sendInit(stage);

  // Worker will post 'ready', then we call start
  st._pendingStart = true;
  st.status = 'init';
  updateStageUI(stage);
}

// Called when worker posts 'ready'
function onStageReady(stage) {
  const st = S.stages[stage];
  if (st._pendingStart) {
    delete st._pendingStart;
    workerPost({ type: 'start', stage });
    st.status = 'running';
    updateStageUI(stage);
  }
}

function pauseStage(stage) {
  workerPost({ type: 'pause', stage });
  S.stages[stage].status = 'pausing';
  updateStageUI(stage);
}

function resumeStage(stage) {
  workerPost({ type: 'start', stage });
  S.stages[stage].status = 'running';
  updateStageUI(stage);
}

function resetStage(stage) {
  workerPost({ type: 'reset', stage });
  S.stages[stage].status = 'resetting';
  updateStageUI(stage);
}

function downloadStage(stage) {
  const st = S.stages[stage];
  if (!st.modelJson) { showBanner('Stage not complete yet.', 'warn'); return; }
  const blob = new Blob([JSON.stringify(st.modelJson)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `tapasya-stage-${stage}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---- Progress ----
function onProgress(m) {
  const st = S.stages[m.stage];
  st.step  = m.step;
  st.status = 'running';
  st.history.push({ step: m.step, train: m.train_loss, val: m.val_loss });
  if (st.history.length > 500) st.history.shift();
  drawChart(m.stage);
  updateStatus(m.stage, m.step, m.train_loss, m.val_loss, m.wall_ms);
}

function onSample(m) {
  const st = S.stages[m.stage];
  st.samples.unshift({ step: m.step, text: m.text });
  if (st.samples.length > 20) st.samples.pop();
  renderSamples(m.stage);
}

function onStageDone(m) {
  const st = S.stages[m.stage];
  st.status   = 'done';
  st.step     = m.step;
  st.modelJson = m.model_json;
  updateStageUI(m.stage);
  showBanner(`Stage ${m.stage} complete.`, 'ok');
}

function onGenerated(m) {
  const col = $(`compare-col-${m.stage}`);
  if (col) col.querySelector('.compare-output').textContent = m.text;
}

function onResetDone(stage) {
  const st = S.stages[stage];
  st.status = 'idle';
  st.step   = 0;
  st.history = [];
  st.samples = [];
  st.modelJson = null;
  delete st._pendingStart;
  updateStageUI(stage);
  drawChart(stage);
  renderSamples(stage);

  if (st._pendingFreshStart) {
    delete st._pendingFreshStart;
    sendInit(stage);
    st._pendingStart = true;
    st.status = 'init';
    updateStageUI(stage);
    return;
  }

  // Unlock corpus if all stages are idle
  const allIdle = [1, 2, 3].every(i => ['idle', 'error'].includes(S.stages[i].status));
  if (allIdle) {
    S.corpusLocked = false;
    $('corpus-input').removeAttribute('disabled');
    $('corpus-lock-msg').textContent = '';
  }
}

// ---- Resume offer ----
function offerResume(stage, step) {
  const st = S.stages[stage];
  const msg = `Stage ${stage} checkpoint found at step ${step}. Resume?`;
  // Simple banner with resume / start-fresh buttons
  const bar = document.createElement('div');
  bar.className = 'resume-bar';
  bar.innerHTML = `<span>${msg}</span>
    <button onclick="doResume(${stage},this)">Resume</button>
    <button onclick="doFresh(${stage},this)">Start fresh</button>`;
  $(`stage-${stage}-header`).after(bar);
}

window.doResume = function(stage, btn) {
  btn.closest('.resume-bar').remove();
  workerPost({ type: 'resume', stage });
  S.stages[stage]._pendingStart = true;
  S.stages[stage].status = 'init';
  updateStageUI(stage);
};

window.doFresh = function(stage, btn) {
  btn.closest('.resume-bar').remove();
  // Reset deletes the OPFS checkpoint; onResetDone will then call sendInit
  workerPost({ type: 'reset', stage });
  S.stages[stage]._pendingFreshStart = true;
  S.stages[stage].status = 'resetting';
  updateStageUI(stage);
};

// ---- Unlocks ----
function showUnlockBanner(stage) {
  showBanner(`Stage ${stage} unlocked!`, 'ok');
  const tab = $(`tab-${stage}`);
  if (tab) tab.classList.add('unlocked-flash');
  setTimeout(() => tab && tab.classList.remove('unlocked-flash'), 2000);
}

// ---- Compare mode ----
function toggleCompare() {
  if (!S.compareUnlocked) return;
  S.compareMode = !S.compareMode;
  $('compare-panel').style.display = S.compareMode ? 'flex' : 'none';
  $('stages-panel').style.display  = S.compareMode ? 'none' : 'block';
  if (S.compareMode) runCompare();
}

function runCompare() {
  const seedText = $('compare-seed').value || '';
  [1, 2, 3].forEach(stage => {
    const col = $(`compare-col-${stage}`);
    if (!col) return;
    col.querySelector('.compare-output').textContent = '…generating…';
    if (S.stages[stage].modelJson || S.stages[stage].status === 'done' || S.stages[stage].step > 0) {
      workerPost({ type: 'generate', stage, seed_text: seedText, max_tokens: 200 });
    } else {
      col.querySelector('.compare-output').textContent = '(not trained)';
    }
  });
}

function updateCompareButton() {
  const btn = $('compare-btn');
  if (!btn) return;
  btn.disabled = !S.compareUnlocked;
  btn.title    = S.compareUnlocked ? '' : 'Complete at least two stages to compare';
}

// ---- Stage tab switching ----
function switchStage(stage) {
  if (S.mode === 'guided' && !S.stages[stage].unlocked) return;
  S.activeStage = stage;
  [1, 2, 3].forEach(i => {
    $(`tab-${i}`)?.classList.toggle('active', i === stage);
    $(`stage-panel-${i}`)?.style.setProperty('display', i === stage ? 'block' : 'none');
  });
}

// ---- Chart (Canvas 2D) ----
const chartCtx = {};
function initChart(stage) {
  const canvas = $(`chart-${stage}`);
  if (!canvas) return;
  chartCtx[stage] = canvas.getContext('2d');
  drawChart(stage);
}

function drawChart(stage) {
  const ctx = chartCtx[stage];
  if (!ctx) return;
  const hist = S.stages[stage].history;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (hist.length < 2) {
    ctx.fillStyle = '#555';
    ctx.font = '13px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Loss chart — starts when training begins', W / 2, H / 2);
    return;
  }

  const vals = hist.flatMap(p => [p.train, isFinite(p.val) ? p.val : p.train]).filter(isFinite);
  const maxV = Math.max(...vals) * 1.05;
  const minV = Math.max(0, Math.min(...vals) * 0.95);
  const pad  = { l: 44, r: 12, t: 10, b: 28 };
  const cW   = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  const xOf = i => pad.l + (i / (hist.length - 1)) * cW;
  const yOf = v => pad.t + (1 - (v - minV) / (maxV - minV + 1e-10)) * cH;

  // Grid
  ctx.strokeStyle = '#2a2a2a'; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad.t + g * cH / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cW, y); ctx.stroke();
    const v = maxV - g * (maxV - minV) / 4;
    ctx.fillStyle = '#555'; ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), pad.l - 4, y + 4);
  }

  // Curves
  function drawLine(key, color) {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    let started = false;
    for (let i = 0; i < hist.length; i++) {
      const v = hist[i][key];
      if (!isFinite(v)) continue;
      if (!started) { ctx.moveTo(xOf(i), yOf(v)); started = true; }
      else            ctx.lineTo(xOf(i), yOf(v));
    }
    ctx.stroke();
  }
  drawLine('train', '#4e9af1');
  drawLine('val',   '#f1a84e');

  // X axis labels
  ctx.fillStyle = '#555'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  const ticks = [0, Math.floor(hist.length / 2), hist.length - 1];
  ticks.forEach(i => {
    if (i < hist.length) ctx.fillText(hist[i].step, xOf(i), H - pad.b + 14);
  });

  // Legend
  ctx.font = '11px monospace'; ctx.textAlign = 'left';
  ctx.fillStyle = '#4e9af1'; ctx.fillText('— train', pad.l, pad.t + 14);
  ctx.fillStyle = '#f1a84e'; ctx.fillText('— val',   pad.l + 60, pad.t + 14);
}

// ---- Status line ----
function updateStatus(stage, step, trainLoss, valLoss, wallMs) {
  const el = $(`status-${stage}`);
  if (!el) return;
  const steps = S.stages[stage].steps;
  const eta   = wallMs > 0 ? Math.round((steps - step) * (wallMs / step) / 1000) : 0;
  el.textContent = `Step ${step}/${steps} · train ${trainLoss.toFixed(4)} · val ${isFinite(valLoss) ? valLoss.toFixed(4) : 'n/a'} · ETA ~${eta}s`;
}

function flashStatus(stage, msg) {
  const el = $(`status-${stage}`);
  if (!el) return;
  const old = el.textContent;
  el.textContent = msg;
  setTimeout(() => { if (el.textContent === msg) el.textContent = old; }, 1500);
}

// ---- Samples pane ----
function renderSamples(stage) {
  const el = $(`samples-${stage}`);
  if (!el) return;
  const st = S.stages[stage];
  el.innerHTML = st.samples.map(s =>
    `<div class="sample-entry"><span class="sample-step">step ${s.step}</span><pre class="sample-text">${escHtml(s.text)}</pre></div>`
  ).join('');
}

// ---- Stage UI update (buttons, labels) ----
function updateStageUI(stage) {
  const st  = S.stages[stage];
  const tab = $(`tab-${stage}`);

  // Tab lock state (guided mode)
  if (tab) {
    tab.classList.toggle('locked', S.mode === 'guided' && !st.unlocked);
    tab.classList.toggle('active', stage === S.activeStage);
  }

  const panel = $(`stage-panel-${stage}`);
  if (!panel) return;

  // Status badge
  const badge = panel.querySelector('.stage-status-badge');
  if (badge) {
    const labels = { idle: 'idle', init: 'initialising…', ready: 'ready',
                     running: 'training', pausing: 'pausing…', paused: 'paused',
                     done: 'complete', resetting: 'resetting…', error: 'error' };
    badge.textContent = labels[st.status] || st.status;
    badge.className   = `stage-status-badge badge-${st.status}`;
  }

  // Buttons
  const startBtn   = panel.querySelector('.btn-start');
  const pauseBtn   = panel.querySelector('.btn-pause');
  const resetBtn   = panel.querySelector('.btn-reset');
  const dlBtn      = panel.querySelector('.btn-download');

  if (startBtn)  startBtn.disabled  = ['running', 'init', 'pausing', 'resetting'].includes(st.status);
  if (pauseBtn)  pauseBtn.disabled  = !['running'].includes(st.status);
  if (resetBtn)  resetBtn.disabled  = ['resetting', 'init'].includes(st.status);
  if (dlBtn)     dlBtn.disabled     = st.status !== 'done';

  // Settings (lock in guided mode)
  const settings = panel.querySelector('.settings-panel');
  if (settings) {
    const locked = S.mode === 'guided' || ['running', 'init', 'pausing'].includes(st.status);
    settings.querySelectorAll('input').forEach(i => { i.disabled = locked; });
  }

  // Guided-mode unlock overlay
  const overlay = panel.querySelector('.locked-overlay');
  if (overlay) overlay.style.display = (S.mode === 'guided' && !st.unlocked) ? 'flex' : 'none';

  // Ready → start if pending
  if (st.status === 'ready') onStageReady(stage);
}

// ---- Mode selection ----
function selectMode(mode) {
  S.mode = mode;
  $('landing').style.display    = 'none';
  $('app-shell').style.display  = 'block';

  if (mode === 'guided') {
    $('stages-label').textContent = 'Guided tour';
    S.stages[2].unlocked = false;
    S.stages[3].unlocked = false;
  } else {
    $('stages-label').textContent = 'Free mode';
    S.stages[1].unlocked = true;
    S.stages[2].unlocked = true;
    S.stages[3].unlocked = true;
  }

  [1, 2, 3].forEach(i => { initChart(i); updateStageUI(i); });
  switchStage(1);
  spawnWorker();
  saveModeToOPFS(mode);
}

// ---- OPFS mode persistence ----
async function saveModeToOPFS(mode) {
  try {
    const root = await navigator.storage.getDirectory();
    const dir  = await root.getDirectoryHandle('tapasya', { create: true });
    const fh   = await dir.getFileHandle('prefs.json', { create: true });
    const w    = await fh.createWritable();
    await w.write(JSON.stringify({ mode }));
    await w.close();
  } catch {}
}

async function loadModeFromOPFS() {
  try {
    const root = await navigator.storage.getDirectory();
    const dir  = await root.getDirectoryHandle('tapasya', { create: false });
    const fh   = await dir.getFileHandle('prefs.json');
    const f    = await fh.getFile();
    return (JSON.parse(await f.text())).mode || null;
  } catch { return null; }
}

// ---- Helpers ----
function $(id) { return document.getElementById(id); }

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showBanner(msg, type = 'info') {
  const b = $('toast');
  if (!b) return;
  b.textContent = msg;
  b.className   = `toast toast-${type} show`;
  clearTimeout(b._t);
  b._t = setTimeout(() => b.classList.remove('show'), 3000);
}

// ---- Settings reading (free mode) ----
function readSettings(stage) {
  const panel = $(`stage-panel-${stage}`);
  if (!panel) return {};
  const get = name => {
    const el = panel.querySelector(`[data-cfg="${name}"]`);
    return el ? parseFloat(el.value) || undefined : undefined;
  };
  return {
    steps:      get('steps')      || S.stages[stage].steps,
    pop_size:   get('pop_size'),
    sigma:      get('sigma'),
    lr:         get('lr'),
    batch_size: get('batch_size'),
    seed:       get('seed'),
  };
}

// ---- Boot ----
async function boot() {
  const savedMode = await loadModeFromOPFS();
  if (savedMode) {
    selectMode(savedMode);
  }
  // Landing stays visible until user picks a mode
  $('btn-guided').addEventListener('click', () => selectMode('guided'));
  $('btn-free').addEventListener('click',   () => selectMode('free'));

  $('corpus-input').addEventListener('input', e => {
    S.corpus = e.target.value;
    $('corpus-char-count').textContent = S.corpus.length.toLocaleString() + ' chars';
  });

  $('compare-btn').addEventListener('click', toggleCompare);
  $('compare-regen').addEventListener('click', runCompare);
  $('compare-back').addEventListener('click', toggleCompare);

  updateCompareButton();
}

document.addEventListener('DOMContentLoaded', boot);
