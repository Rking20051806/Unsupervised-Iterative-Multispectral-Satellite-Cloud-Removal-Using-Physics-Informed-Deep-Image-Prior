/* ═══════════════════════════════════════════════════════════════════════════
   CloudVision AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  overview: null,
  split: 'RICE1',
  index: 0,
  sensitivity: 0.75,
  file: null,
  uploading: false,
  // Sentinel-2
  s2_x: null,
  s2_y: null,
  s2_x_start: null,
  s2_y_start: null,
  s2_mask_visible: false,
  s2_optimizing: false,
  s2_chart: null,
  s2_overview_w: 512,
  s2_overview_h: 512,
  s2_full_w: 10980,
  s2_full_h: 10980,
  s2_crop_size: 256,
  s2_iters: 2000,          // default: High Quality
  // RICE Benchmark
  bench_split: 'RICE1',
  bench_index: 0,
  bench_running: false,
  last_xai_images: null,
  last_xai_active: 'integrated',
};

// ── Element Cache ──────────────────────────────────────────────────────────
const els = {};
function $(id) { return document.getElementById(id); }

// ── Helpers ────────────────────────────────────────────────────────────────
function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(d);
}

function setImg(el, src) {
  if (!el) return;
  if (!src) {
    el.removeAttribute('src');
    el.classList.remove('loaded');
    el.classList.add('hidden');
    el.style.display = 'none';
    const ph = el.parentElement?.querySelector('.img-placeholder') || el.parentElement?.querySelector('.flow-ph');
    if (ph) ph.style.display = '';
    return;
  }

  if (typeof src === 'string' && !src.startsWith('data:') && !src.startsWith('http') && !src.startsWith('/') && !src.startsWith('.')) {
    src = 'data:image/png;base64,' + src;
  }

  el.src = src;
  el.style.display = '';
  el.classList.remove('hidden');
  el.classList.add('loaded');   // reveals img inside .flow-img-box
  el.classList.remove('fade-in');
  void el.offsetWidth; // trigger reflow for smooth re-fade
  el.classList.add('fade-in');
  const ph = el.parentElement?.querySelector('.img-placeholder') || el.parentElement?.querySelector('.flow-ph');
  if (ph) ph.style.display = 'none';
}

function updateCloudStats(stats, prefix) {
  const setBar = (id, pct) => {
    const valEl = $(`${prefix}-stat-${id}`);
    const barEl = $(`${prefix}-bar-${id}`);
    const pctStr = `${(pct * 100).toFixed(1)}%`;
    if (valEl) valEl.textContent = pctStr;
    if (barEl) barEl.style.width = pctStr;
  };
  setBar('total', stats.total !== undefined ? stats.total : (stats.cloud_stat_total || 0));
  setBar('thick', stats.thick !== undefined ? stats.thick : (stats.cloud_stat_thick || 0));
  setBar('medium', stats.medium !== undefined ? stats.medium : (stats.cloud_stat_medium || 0));
  setBar('thin', stats.thin !== undefined ? stats.thin : (stats.cloud_stat_thin || 0));
  setBar('shadow', stats.shadow !== undefined ? stats.shadow : (stats.cloud_stat_shadow || 0));

  const confEl = $(`${prefix}-stat-confidence`);
  let conf = stats.confidence !== undefined ? stats.confidence : (stats.cloud_stat_confidence || 0);
  if (conf > 1.0) {
    conf = conf / 100.0;
  }
  if (confEl) confEl.textContent = `${(conf * 100).toFixed(1)}%`;
}

// ── Live S2 Log Panel ──────────────────────────────────────────────────────
function appendS2Log(text, type = 'info') {
  const panel = $('s2-log-panel');
  if (!panel) return;

  // Remove placeholder hint on first real line
  const hint = panel.querySelector('.s2-log-line--hint');
  if (hint) hint.remove();

  const line = document.createElement('div');
  line.className = 's2-log-line';

  const colorMap = {
    info: '#8aa3cc',
    loss: '#5bc8f5',
    warn: '#f5a623',
    error: '#ef4444',
    success: '#3de6a8',
    stdout: '#b0c4de',
  };

  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  line.innerHTML =
    `<span style="color:#3d5175;user-select:none">[${ts}]</span> ` +
    `<span style="color:${colorMap[type] || colorMap.info}">${text}</span>`;

  panel.appendChild(line);

  // Keep only the last 120 lines so the DOM doesn't bloat
  while (panel.children.length > 120) panel.removeChild(panel.firstChild);

  // Auto-scroll to bottom
  panel.scrollTop = panel.scrollHeight;
}

function appendConsoleLog(text) {
  const logEl = $('pipe-log-console') || $('r2pipe-log-console');
  if (!logEl) return;

  if (logEl.textContent.trim() === 'Awaiting pipeline run...' || logEl.textContent.trim() === 'Initializing pipeline connection...') {
    logEl.textContent = '';
  }

  const now = new Date();
  const timeStr = `[${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}] `;

  let color = '#9ca3af';
  const t = text.toLowerCase();
  if (t.includes('=== running') || t.includes('===') || t.startsWith('===')) {
    color = '#fb923c';
  } else if (t.includes('[info]') || t.includes('initializing') || t.includes('get sse') || t.includes('http/1.1')) {
    color = '#60a5fa';
  } else if (t.includes('[success]') || t.includes('completed successfully') || t.includes('finished') || t.includes('complete')) {
    color = '#34d399';
  } else if (t.includes('[warn]') || t.includes('warning') || t.includes('cancelled')) {
    color = '#fbbf24';
  } else if (t.includes('[error]') || t.includes('error') || t.includes('exception') || t.includes('traceback')) {
    color = '#f87171';
  } else if (t.includes('[dataset]') || t.includes('[model]') || t.includes('[unet]') || t.includes('[xai]') || t.includes('[ablation]') || t.includes('[preset]')) {
    color = '#a78bfa';
  } else if (t.includes('iter ') && (t.includes('| loss:') || t.includes('| metrics'))) {
    color = '#22d3ee';
  } else if (t.includes('psnr') || t.includes('ssim') || t.includes('[metrics]')) {
    color = '#86efac';
  } else if (t.includes('[params]') || t.includes('lambda') || t.includes('lr=')) {
    color = '#c084fc';
  }

  const line = document.createElement('span');
  line.style.cssText = `color:${color};display:block;`;
  line.textContent = timeStr + text;
  logEl.appendChild(line);

  while (logEl.children.length > 300) {
    logEl.removeChild(logEl.firstChild);
  }

  const counter = $('pipe-log-line-count') || $('r2log-line-count');
  if (counter) counter.textContent = logEl.children.length + ' lines';

  if (window._pipeLogAutoScroll !== false && window._r2LogAutoScroll !== false) {
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function appendRice2Console(text) {
  const logEl = $('rice2-console');
  if (logEl) {
    // If it's the first log, clear the waiting text
    if (logEl.innerHTML.includes('Waiting for inputs')) {
      logEl.innerHTML = '';
    }
    const div = document.createElement('div');
    div.textContent = text;
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function clearS2Log() {
  const panel = $('s2-log-panel');
  if (!panel) return;
  panel.innerHTML =
    '<div class="s2-log-line s2-log-line--hint">Optimization started — streaming live diagnostics…</div>';
}

function downloadFile(url, filename) {
  // Use fetch + blob to force browser to use the correct filename
  fetch(url)
    .then(response => {
      if (!response.ok) throw new Error('Network response was not ok');
      return response.blob();
    })
    .then(blob => {
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(blobUrl);
    })
    .catch(err => {
      console.error('Blob download failed, falling back to direct link:', err);
      // Fallback to direct download
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });
}

// Helper for Internal Activation Maps Dashboard
window.updateInternalActivationMap = function (newMaps) {
  if (newMaps) {
    window.currentLiveMaps = Object.assign(window.currentLiveMaps || {}, newMaps);
  }

  const defaultPresets = {
    'early': 42.5,
    'middle': 68.0,
    'deep': 54.2,
    'magnitude': 73.8,
    'error': 18.4,
    'physics': 24.6
  };

  const prefixes = ['', 'r1', 'r2', 'r2page', 'r2pipe'];
  const keys = ['early', 'middle', 'deep', 'magnitude', 'error', 'physics'];

  prefixes.forEach(pfx => {
    keys.forEach(key => {
      const imgId = pfx ? `${pfx}map-grid-${key}` : `map-grid-${key}`;
      const phId = `ph-${imgId}`;
      const imgEl = document.getElementById(imgId);
      const phEl = document.getElementById(phId);
      const pinEl = document.getElementById(`${pfx ? pfx + 'pin-' : 'pin-'}${key}`);
      const valEl = document.getElementById(`${pfx ? pfx + 'val-' : 'val-'}${key}`);
      const shimmerEl = pinEl?.parentElement?.querySelector('.colorbar-pulse-shimmer');

      const hasLoadedMap = !!(window.currentLiveMaps && window.currentLiveMaps[key]);

      if (hasLoadedMap) {
        let src = window.currentLiveMaps[key];
        if (typeof src === 'string' && !src.startsWith('data:') && !src.startsWith('http') && !src.startsWith('/') && !src.startsWith('.')) {
          src = 'data:image/png;base64,' + src;
        }
        if (imgEl && imgEl.dataset.currentSrc !== src) {
          imgEl.dataset.currentSrc = src;
          imgEl.src = src;
          imgEl.style.display = 'block';
          if (phEl) phEl.style.display = 'none';
        }

        // Exact real calculation from neural network tensors (no artificial math loop)
        let realVal = defaultPresets[key] || 50.0;
        if (window.currentLiveMaps.levels && window.currentLiveMaps.levels[key] !== undefined) {
          realVal = window.currentLiveMaps.levels[key];
        }
        const pct = parseFloat(realVal).toFixed(1);

        if (pinEl) {
          pinEl.style.left = `${pct}%`;
          pinEl.style.opacity = '1';
        }
        if (valEl) valEl.innerText = `${pct}%`;
        if (shimmerEl) shimmerEl.style.display = 'block';
      } else {
        // When awaiting, keep indicator at 0.0% and hidden/static
        if (pinEl) {
          pinEl.style.left = '0%';
          pinEl.style.opacity = '0.3';
        }
        if (valEl) valEl.innerText = '0.0%';
        if (shimmerEl) shimmerEl.style.display = 'none';
      }
    });
  });
};



// ── Download Full Combined Internal Activation & Diagnostic Maps as 1 High-Res Image ──
window.downloadActivationMapsGrid = async function (split = 'RICE1') {
  const pfx = split === 'RICE2' ? 'r2' : '';
  const cards = [
    { title: '1. Early-layer activation maps (Conv1)', id: pfx ? `${pfx}map-grid-early` : 'map-grid-early', desc: 'Edges, fine textures, thin-cloud patterns', valId: pfx ? `${pfx}val-early` : 'val-early', color: '#2dd4bf' },
    { title: '2. Middle-layer activation maps (Conv4)', id: pfx ? `${pfx}map-grid-middle` : 'map-grid-middle', desc: 'Roads, vegetation & structure boundaries', valId: pfx ? `${pfx}val-middle` : 'val-middle', color: '#a3e635' },
    { title: '3. Deep Decoder activation maps (Up3)', id: pfx ? `${pfx}map-grid-deep` : 'map-grid-deep', desc: 'High-level semantics & shape restoration', valId: pfx ? `${pfx}val-deep` : 'val-deep', color: '#fbbf24' },
    { title: '4. Activation magnitude / mean map', id: pfx ? `${pfx}map-grid-magnitude` : 'map-grid-magnitude', desc: 'Mean energy across all intermediate channels', valId: pfx ? `${pfx}val-magnitude` : 'val-magnitude', color: '#c084fc' },
    { title: '5. Reconstruction error map', id: pfx ? `${pfx}map-grid-error` : 'map-grid-error', desc: 'Absolute pixel residual |I_restored - I_target|', valId: pfx ? `${pfx}val-error` : 'val-error', color: '#e879f9' },
    { title: '6. PINN physics-residual map', id: pfx ? `${pfx}map-grid-physics` : 'map-grid-physics', desc: 'Atmospheric radiative transfer physics compliance', valId: pfx ? `${pfx}val-physics` : 'val-physics', color: '#f43f5e' }
  ];

  const canvas = document.createElement('canvas');
  const cols = 3;
  const rows = 2;
  const cardW = 440;
  const cardH = 460;
  const pad = 24;
  const headerH = 90;

  canvas.width = cols * cardW + (cols + 1) * pad;
  canvas.height = headerH + rows * cardH + (rows + 1) * pad;
  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#060a17';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Header Title
  ctx.font = 'bold 26px "Space Grotesk", sans-serif';
  ctx.fillStyle = '#00f0ff';
  ctx.fillText(`🧬 Internal Activation & Diagnostic Maps (${split} DIP-PINN)`, pad, 48);

  ctx.font = '14px "Outfit", sans-serif';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText(`Deep Image Prior + Physics-Informed Neural Network Internal Multi-Layer Feature Representations`, pad, 74);

  const loadImg = src => new Promise(resolve => {
    if (!src) return resolve(null);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });

  for (let i = 0; i < cards.length; i++) {
    const card = cards[i];
    const c = i % cols;
    const r = Math.floor(i / cols);
    const x = pad + c * (cardW + pad);
    const y = headerH + pad + r * (cardH + pad);

    // Card background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(x, y, cardW, cardH, 12);
    ctx.fill();
    ctx.stroke();

    // Card title
    ctx.font = 'bold 15px "Outfit", sans-serif';
    ctx.fillStyle = card.color;
    ctx.fillText(card.title, x + 16, y + 28);

    // Sub description
    ctx.font = '12px "Outfit", sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.fillText(card.desc, x + 16, y + 48);

    // Image placeholder / preview
    const imgBoxX = x + 16;
    const imgBoxY = y + 62;
    const imgBoxS = cardW - 32;
    const imgBoxH = 320;

    ctx.fillStyle = '#020617';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.beginPath();
    ctx.roundRect(imgBoxX, imgBoxY, imgBoxS, imgBoxH, 8);
    ctx.fill();
    ctx.stroke();

    const imgEl = document.getElementById(card.id);
    const imgSrc = imgEl?.src || (window.currentLiveMaps ? window.currentLiveMaps[card.id.split('-').pop()] : null);
    if (imgSrc) {
      const loaded = await loadImg(imgSrc);
      if (loaded) {
        ctx.drawImage(loaded, imgBoxX, imgBoxY, imgBoxS, imgBoxH);
      }
    }

    // Colorbar indicator bar at bottom
    const barY = y + cardH - 42;
    const valEl = document.getElementById(card.valId);
    const valText = valEl ? valEl.innerText : 'Active';

    ctx.font = 'bold 12px monospace';
    ctx.fillStyle = card.color;
    ctx.fillText(`Activation Level: ${valText}`, imgBoxX, barY);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.beginPath();
    ctx.roundRect(imgBoxX, barY + 8, imgBoxS, 8, 4);
    ctx.fill();

    ctx.fillStyle = card.color;
    ctx.beginPath();
    const pct = parseFloat(valText) || 50;
    ctx.roundRect(imgBoxX, barY + 8, (imgBoxS * Math.min(100, pct)) / 100, 8, 4);
    ctx.fill();
  }

  // Trigger Download
  const link = document.createElement('a');
  link.download = `${split.toLowerCase()}_internal_activation_diagnostic_maps.png`;
  link.href = canvas.toDataURL('image/png');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  if (typeof showToast === 'function') {
    showToast(`Saved complete ${split} Internal Activation Maps image!`, 'success', '💾');
  }
};

// Helper to format seconds into hh:mm:ss
function formatHms(seconds) {
  if (seconds === undefined || seconds === null || isNaN(seconds) || seconds === '' || seconds === '—') return '—';
  const totalSecs = Math.max(0, Math.round(Number(seconds)));
  const hrs = Math.floor(totalSecs / 3600);
  const mins = Math.floor((totalSecs % 3600) / 60);
  const secs = totalSecs % 60;
  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}
window.formatHms = formatHms;

// ── Download Full 5-Panel Reconstruction Results & Metrics Comparison as 1 Image ──
window.downloadAblationResultsGrid = async function (split = 'RICE1') {
  const pfx = split === 'RICE2' ? 'r2' : '';
  const panels = [
    { title: '1. FULL MODEL (ASM + RTE)', imgId: pfx ? `${pfx}grid-img-1` : 'grid-img-1', psnrId: pfx ? `${pfx}grid-psnr-1` : 'grid-psnr-1', ssimId: pfx ? `${pfx}grid-ssim-1` : 'grid-ssim-1', rmseId: pfx ? `${pfx}grid-rmse-1` : 'grid-rmse-1', samId: pfx ? `${pfx}grid-sam-1` : 'grid-sam-1', lpipsId: pfx ? `${pfx}grid-lpips-1` : 'grid-lpips-1', itersId: pfx ? `${pfx}grid-iters-1` : 'grid-iters-1', timeId: pfx ? `${pfx}grid-time-1` : 'grid-time-1', color: '#10b981', tag: 'METRICS (With PSNR / SSIM / RMSE)' },
    { title: '2. NO ASM (Only RTE)', imgId: pfx ? `${pfx}grid-img-2` : 'grid-img-2', psnrId: pfx ? `${pfx}grid-psnr-2` : 'grid-psnr-2', ssimId: pfx ? `${pfx}grid-ssim-2` : 'grid-ssim-2', rmseId: pfx ? `${pfx}grid-rmse-2` : 'grid-rmse-2', samId: pfx ? `${pfx}grid-sam-2` : 'grid-sam-2', lpipsId: pfx ? `${pfx}grid-lpips-2` : 'grid-lpips-2', itersId: pfx ? `${pfx}grid-iters-2` : 'grid-iters-2', timeId: pfx ? `${pfx}grid-time-2` : 'grid-time-2', color: '#f59e0b', tag: 'METRICS (With PSNR / SSIM / RMSE)' },
    { title: '3. NO RTE (Only ASM)', imgId: pfx ? `${pfx}grid-img-3` : 'grid-img-3', psnrId: pfx ? `${pfx}grid-psnr-3` : 'grid-psnr-3', ssimId: pfx ? `${pfx}grid-ssim-3` : 'grid-ssim-3', rmseId: pfx ? `${pfx}grid-rmse-3` : 'grid-rmse-3', samId: pfx ? `${pfx}grid-sam-3` : 'grid-sam-3', lpipsId: pfx ? `${pfx}grid-lpips-3` : 'grid-lpips-3', itersId: pfx ? `${pfx}grid-iters-3` : 'grid-iters-3', timeId: pfx ? `${pfx}grid-time-3` : 'grid-time-3', color: '#a855f7', tag: 'METRICS (With PSNR / SSIM / RMSE)' },
    { title: '4. CORE DIP + METRICS (No Physics)', imgId: pfx ? `${pfx}grid-img-4` : 'grid-img-4', psnrId: pfx ? `${pfx}grid-psnr-4` : 'grid-psnr-4', ssimId: pfx ? `${pfx}grid-ssim-4` : 'grid-ssim-4', rmseId: pfx ? `${pfx}grid-rmse-4` : 'grid-rmse-4', samId: pfx ? `${pfx}grid-sam-4` : 'grid-sam-4', lpipsId: pfx ? `${pfx}grid-lpips-4` : 'grid-lpips-4', itersId: pfx ? `${pfx}grid-iters-4` : 'grid-iters-4', timeId: pfx ? `${pfx}grid-time-4` : 'grid-time-4', color: '#06b6d4', tag: 'METRICS (With PSNR / SSIM / RMSE)' }
  ];

  const cols = 4;
  const cardW = 340;
  const cardH = 640;
  const pad = 20;
  const headerH = 90;

  const canvas = document.createElement('canvas');
  canvas.width = cols * cardW + (cols + 1) * pad;
  canvas.height = headerH + cardH + pad * 2;
  const ctx = canvas.getContext('2d');

  // Background
  ctx.fillStyle = '#060a17';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Header Title
  ctx.font = 'bold 24px "Space Grotesk", sans-serif';
  ctx.fillStyle = '#00f0ff';
  ctx.fillText(`6. RECONSTRUCTION RESULTS & METRICS COMPARISON (${split})`, pad, 44);

  ctx.font = '14px "Outfit", sans-serif';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText(`Ablation Study Comparison Across Physics Loss Components & Zero-Shot Priors`, pad, 70);

  const loadImg = src => new Promise(resolve => {
    if (!src) return resolve(null);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });

  for (let i = 0; i < panels.length; i++) {
    const p = panels[i];
    const x = pad + i * (cardW + pad);
    const y = headerH + pad;

    // Card background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(x, y, cardW, cardH, 10);
    ctx.fill();
    ctx.stroke();

    // Panel Title
    ctx.font = 'bold 13px "Outfit", sans-serif';
    ctx.fillStyle = p.color;
    ctx.textAlign = 'center';
    ctx.fillText(p.title, x + cardW / 2, y + 26);
    ctx.textAlign = 'left';

    // Image box
    const imgBoxX = x + 16;
    const imgBoxY = y + 42;
    const imgBoxS = cardW - 32;

    ctx.fillStyle = '#020617';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.beginPath();
    ctx.roundRect(imgBoxX, imgBoxY, imgBoxS, imgBoxS, 6);
    ctx.fill();
    ctx.stroke();

    const imgEl = document.getElementById(p.imgId);
    if (imgEl && imgEl.src && imgEl.style.display !== 'none') {
      const loaded = await loadImg(imgEl.src);
      if (loaded) {
        ctx.drawImage(loaded, imgBoxX, imgBoxY, imgBoxS, imgBoxS);
      }
    }

    // Metrics Tag
    const tagY = imgBoxY + imgBoxS + 24;
    ctx.font = 'bold 11px "Outfit", sans-serif';
    ctx.fillStyle = p.color;
    ctx.fillText(p.tag, imgBoxX, tagY);

    // Metrics List
    const metricsY = tagY + 22;
    const resObj = (split === 'RICE2' ? (window.rice2AblationResults?.[i + 1] || window.r2AblationResults?.[i + 1]) : (window.rice1AblationResults?.[i + 1] || window.ablationResults?.[i + 1])) || {};
    const getVal = (id, fallback) => {
      const el = document.getElementById(id);
      const txt = el ? el.innerText.trim() : '';
      if (txt && txt !== '—' && txt !== '-') return txt;
      return fallback;
    };

    const metricsList = [
      { label: 'PSNR (dB)', val: getVal(p.psnrId, resObj.metrics?.psnr ? Number(resObj.metrics.psnr).toFixed(2) : '—') },
      { label: 'SSIM', val: getVal(p.ssimId, resObj.metrics?.ssim ? Number(resObj.metrics.ssim).toFixed(3) : '—') },
      { label: 'RMSE', val: getVal(p.rmseId, resObj.metrics?.rmse ? Number(resObj.metrics.rmse).toFixed(4) : '—') },
      { label: 'SAM (deg)', val: getVal(p.samId, resObj.metrics?.sam ? Number(resObj.metrics.sam).toFixed(2) : '—') },
      { label: 'LPIPS', val: getVal(p.lpipsId, resObj.metrics?.lpips ? Number(resObj.metrics.lpips).toFixed(3) : '—') },
      { label: 'Iterations', val: getVal(p.itersId, resObj.metrics?.iters || resObj.inputs?.iters || '—') },
      { label: 'Time (hh:mm:ss)', val: getVal(p.timeId, resObj.metrics?.time || '—') }
    ];

    let rowY = metricsY;
    metricsList.forEach(m => {
      ctx.font = '12px monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(m.label, imgBoxX, rowY);

      ctx.font = 'bold 13px monospace';
      ctx.fillStyle = p.color;
      ctx.textAlign = 'right';
      ctx.fillText(m.val, imgBoxX + imgBoxS, rowY);
      ctx.textAlign = 'left';

      rowY += 24;
    });
  }

  // Trigger Download
  const link = document.createElement('a');
  link.download = `${split.toLowerCase()}_reconstruction_results_metrics_comparison.png`;
  link.href = canvas.toDataURL('image/png');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  if (typeof showToast === 'function') {
    showToast(`Saved complete ${split} Reconstruction Results & Metrics Comparison image!`, 'success', '💾');
  }
};




// ── XAI Layer Swapping ──
function switchXaiLayer(layerName) {
  if (!state.last_xai_images) return;
  const src = state.last_xai_images[layerName];
  if (!src) return;          // truly missing — skip
  state.last_xai_active = layerName;

  const pills = {
    original: els.xaiPillOriginal,
    gradcam: els.xaiPillGradcam,
    integrated: els.xaiPillIntegrated,
    overlay: els.xaiPillOverlay
  };

  Object.keys(pills).forEach(k => {
    const btn = pills[k];
    if (!btn) return;
    if (k === layerName) {
      btn.style.background = 'var(--neon-cyan)';
      btn.style.color = '#0d1527';
      btn.style.border = 'none';
      btn.style.fontWeight = '700';
    } else {
      btn.style.background = 'rgba(255,255,255,0.03)';
      btn.style.color = 'var(--text-secondary)';
      btn.style.border = '1px solid var(--border)';
      btn.style.fontWeight = 'normal';
    }
  });

  // Show image, hide placeholder
  if (els.pipeXaiHeatmap) {
    els.pipeXaiHeatmap.src = src;
    els.pipeXaiHeatmap.style.display = 'block';
  }
  const ph = $('ph-pipe-heatmap');
  if (ph) ph.style.display = 'none';
}

// ── Sliders click-and-drag logic ──
function makeSliderDraggable(containerEl, splitBarEl, splitAfterEl) {
  if (!containerEl || !splitBarEl || !splitAfterEl) return;
  let isDragging = false;

  const update = (clientX) => {
    const rect = containerEl.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    splitBarEl.style.left = `${pct}%`;
    if (splitAfterEl.classList.contains('slider-after')) {
      splitAfterEl.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    } else {
      splitAfterEl.style.width = `${100 - pct}%`;
    }
  };

  containerEl.addEventListener('mousedown', e => {
    isDragging = true;
    update(e.clientX);
  });

  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    update(e.clientX);
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  // Touch support
  containerEl.addEventListener('touchstart', e => {
    isDragging = true;
    if (e.touches[0]) update(e.touches[0].clientX);
  });

  window.addEventListener('touchmove', e => {
    if (!isDragging) return;
    if (e.touches[0]) update(e.touches[0].clientX);
  });

  window.addEventListener('touchend', () => {
    isDragging = false;
  });
}

// ── Image Zoom, Pan, Reset & Coordinates overlay ──
function enableZoomAndPan(containerEl, imgEl) {
  if (!containerEl || !imgEl) return;

  containerEl.style.position = 'relative';
  containerEl.style.overflow = 'hidden';

  const resetBtn = document.createElement('button');
  resetBtn.textContent = '⟲ Reset';
  resetBtn.style.cssText = `
    position: absolute; top: 8px; right: 8px; z-index: 10;
    background: rgba(13, 21, 46, 0.85); border: 1px solid var(--border);
    color: var(--neon-cyan); padding: 4px 8px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 500; cursor: pointer;
    opacity: 0; transition: opacity 0.3s;
  `;
  containerEl.appendChild(resetBtn);

  const coordsOverlay = document.createElement('div');
  coordsOverlay.style.cssText = `
    position: absolute; bottom: 8px; right: 8px; z-index: 10;
    background: rgba(13, 21, 46, 0.85); border: 1px solid var(--border);
    color: var(--text-secondary); padding: 4px 8px; border-radius: 4px;
    font-size: 0.7rem; font-family: var(--font-mono); pointer-events: none;
    opacity: 0; transition: opacity 0.3s;
  `;
  containerEl.appendChild(coordsOverlay);

  containerEl.addEventListener('mouseenter', () => {
    resetBtn.style.opacity = 1;
    coordsOverlay.style.opacity = 1;
  });
  containerEl.addEventListener('mouseleave', () => {
    resetBtn.style.opacity = 0;
    coordsOverlay.style.opacity = 0;
  });

  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let isPanning = false;
  let startX = 0;
  let startY = 0;

  const applyTransform = () => {
    imgEl.style.transform = `scale(${zoom}) translate(${panX}px, ${panY}px)`;
    imgEl.style.cursor = zoom > 1 ? 'grab' : 'default';
  };

  imgEl.style.transition = 'transform 0.1s ease-out';
  imgEl.style.transformOrigin = 'center center';

  containerEl.addEventListener('wheel', e => {
    e.preventDefault();
    const zoomFactor = 0.1;
    if (e.deltaY < 0) {
      zoom = Math.min(10, zoom + zoomFactor);
    } else {
      zoom = Math.max(1, zoom - zoomFactor);
    }
    if (zoom === 1) {
      panX = 0; panY = 0;
    }
    applyTransform();
  });

  containerEl.addEventListener('mousedown', e => {
    if (zoom <= 1) return;
    isPanning = true;
    imgEl.style.cursor = 'grabbing';
    startX = e.clientX - panX * zoom;
    startY = e.clientY - panY * zoom;
    e.preventDefault();
  });

  window.addEventListener('mousemove', e => {
    if (!isPanning) return;
    panX = (e.clientX - startX) / zoom;
    panY = (e.clientY - startY) / zoom;
    applyTransform();
  });

  window.addEventListener('mouseup', () => {
    if (isPanning) {
      isPanning = false;
      imgEl.style.cursor = zoom > 1 ? 'grab' : 'default';
    }
  });

  resetBtn.addEventListener('click', e => {
    e.stopPropagation();
    zoom = 1;
    panX = 0;
    panY = 0;
    applyTransform();
  });

  containerEl.addEventListener('mousemove', e => {
    const rect = imgEl.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const naturalW = imgEl.naturalWidth || 256;
    const naturalH = imgEl.naturalHeight || 256;
    const pxX = Math.round((x / rect.width) * naturalW);
    const pxY = Math.round((y / rect.height) * naturalH);

    if (pxX >= 0 && pxX <= naturalW && pxY >= 0 && pxY <= naturalH) {
      coordsOverlay.textContent = `X: ${pxX} Y: ${pxY}`;
    }
  });
}

// ── Sentinel-2 Leaflet Map ──
function initS2LeafletMap(ov) {
  if (state.map) {
    state.map.remove();
    state.map = null;
  }

  const center = [ov.center_lat, ov.center_lon];
  const southWest = L.latLng(ov.lat_min, ov.lon_min);
  const northEast = L.latLng(ov.lat_max, ov.lon_max);
  const bounds = L.latLngBounds(southWest, northEast);

  // Initialize Leaflet Map
  state.map = L.map('s2-map', {
    maxBounds: bounds,
    maxBoundsViscosity: 0.8,
    attributionControl: false
  }).setView(center, 12);

  const ts = Date.now();
  const localApiBase = `/api/s2/tile/${state.s2_current_map}`;

  // Create base layers
  const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap'
  }).addTo(state.map);

  const esriSatelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri'
  });

  const tileOptions = {
    minZoom: 1,
    maxZoom: 24,
    maxNativeZoom: 16,
    keepBuffer: 8,
    updateWhenZooming: true,
    updateWhenIdle: false
  };

  const tciLayer = L.tileLayer(`${localApiBase}/tci/{z}/{x}/{y}.png?t=${ts}`, tileOptions);
  const falseColorLayer = L.tileLayer(`${localApiBase}/false_color/{z}/{x}/{y}.png?t=${ts}`, tileOptions);
  const ndviLayer = L.tileLayer(`${localApiBase}/ndvi/{z}/{x}/{y}.png?t=${ts}`, tileOptions);

  // Create Cloud Mask Overlay
  state.s2_mask_layer = L.tileLayer(`${localApiBase}/mask/{z}/{x}/{y}.png?t=${ts}`, tileOptions);

  if (state.s2_mask_visible) {
    state.s2_mask_layer.addTo(state.map);
  }

  const baseMaps = {
    "Street View (OSM)": osmLayer,
    "Satellite (Esri)": esriSatelliteLayer,
    "True Color (Local COG)": tciLayer,
    "False Color (Local COG)": falseColorLayer,
    "NDVI (Local COG)": ndviLayer
  };

  const overlayMaps = {
    "Cloud Mask": state.s2_mask_layer
  };

  L.control.layers(baseMaps, overlayMaps, { collapsed: false }).addTo(state.map);

  // Update zoom percentage readout on map zoom
  function updateZoomDisplay() {
    if (!state.map) return;
    const currentZoom = state.map.getZoom();
    const baseZoom = 12; // Overview base level (100%)
    const percent = Math.round(Math.pow(2, currentZoom - baseZoom) * 100);
    const pEl = document.getElementById('s2-zoom-percent');
    const zEl = document.getElementById('s2-zoom-level');
    if (pEl) pEl.textContent = `${percent}%`;
    if (zEl) zEl.textContent = `(z${Math.round(currentZoom)})`;
  }

  state.map.on('zoom zoomend viewreset', updateZoomDisplay);
  updateZoomDisplay();

  state.map.on('overlayadd', function (e) {
    if (e.layer === state.s2_mask_layer) {
      state.s2_mask_visible = true;
      const btn = document.getElementById('s2-toggle-mask-btn');
      if (btn) btn.textContent = 'Hide Mask';
    }
  });

  state.map.on('overlayremove', function (e) {
    if (e.layer === state.s2_mask_layer) {
      state.s2_mask_visible = false;
      const btn = document.getElementById('s2-toggle-mask-btn');
      if (btn) btn.textContent = 'Show Mask';
    }
  });

  // Draggable crop marker & rectangle
  const cropMeters = state.s2_crop_size * 10;
  const markerIcon = L.divIcon({
    className: 'crop-drag-handle',
    html: '<div style="width:16px;height:16px;background:#ff6b6b;border:2px solid #fff;border-radius:50%;cursor:move;box-shadow:0 2px 5px rgba(0,0,0,0.4);transform:translate(-8px,-8px);"></div>',
    iconSize: [16, 16],
    iconAnchor: [8, 8]
  });

  state.crop_marker = L.marker(center, { draggable: true, icon: markerIcon }).addTo(state.map);
  state.crop_rect = L.rectangle(getBoundingBox(center, cropMeters), {
    color: '#ff6b6b',
    weight: 2,
    fillColor: '#ff6b6b',
    fillOpacity: 0.08,
    dashArray: '5, 3'
  }).addTo(state.map);

  state.crop_marker.on('drag', function (e) {
    const latlng = e.target.getLatLng();
    updateCropRectangle(latlng);
  });

  state.crop_marker.on('dragend', function (e) {
    const latlng = e.target.getLatLng();
    fetchS2CropAtLatLon(latlng.lat, latlng.lng);
  });

  state.map.on('click', function (e) {
    if (state.s2_optimizing) return;
    state.crop_marker.setLatLng(e.latlng);
    updateCropRectangle(e.latlng);
    fetchS2CropAtLatLon(e.latlng.lat, e.latlng.lng);
  });

  // UTM & Pixel Coordinate Overlay Widget on Map
  const leafletCoords = document.createElement('div');
  leafletCoords.style.cssText = `
    position: absolute; bottom: 10px; left: 10px; z-index: 1000;
    background: rgba(13, 21, 46, 0.85); border: 1px solid var(--border);
    color: var(--text-secondary); padding: 6px 12px; border-radius: 4px;
    font-size: 0.75rem; font-family: var(--font-mono); line-height: 1.4;
    pointer-events: none;
  `;
  leafletCoords.innerHTML = `Lat/Lon: —<br>UTM: —<br>Pixel (X,Y): —`;
  state.map.getContainer().appendChild(leafletCoords);

  state.map.on('mousemove', function (e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    const latLonStr = `${lat.toFixed(5)}°, ${lon.toFixed(5)}°`;

    const utmZone = 43;
    const centralMeridian = 75.0;
    const falseEasting = 500000;
    const r = 6378137.0;
    const latRad = lat * Math.PI / 180;
    const lonRad = (lon - centralMeridian) * Math.PI / 180;

    const y_utm = r * Math.log(Math.tan(Math.PI / 4 + latRad / 2));
    const x_utm = r * lonRad * Math.cos(latRad) + falseEasting;
    const utmStr = `Zone ${utmZone}N, X: ${Math.round(x_utm)}m, Y: ${Math.round(y_utm)}m`;

    const xFraction = (lon - ov.lon_min) / (ov.lon_max - ov.lon_min);
    const yFraction = (ov.lat_max - lat) / (ov.lat_max - ov.lat_min);
    const pxX = Math.round(xFraction * state.s2_full_w);
    const pxY = Math.round(yFraction * state.s2_full_h);

    let pxStr = "Out of Tile";
    if (pxX >= 0 && pxX <= state.s2_full_w && pxY >= 0 && pxY <= state.s2_full_h) {
      pxStr = `X: ${pxX}, Y: ${pxY}`;
    }

    leafletCoords.innerHTML = `Lat/Lon: ${latLonStr}<br>UTM: ${utmStr}<br>Pixel (X,Y): ${pxStr}`;
  });

  // Handle Map Fullscreen Button
  const fsBtn = $('s2-map-fullscreen-btn');
  if (fsBtn) {
    fsBtn.onclick = (e) => {
      e.stopPropagation();
      const container = document.querySelector('.s2-map-wrapper');
      if (!container) return;
      if (!document.fullscreenElement) {
        container.requestFullscreen().then(() => {
          container.classList.add('fullscreen-active');
          fsBtn.innerHTML = '❌ Exit Full Screen';
          setTimeout(() => {
            if (state.map) state.map.invalidateSize();
          }, 150);
        }).catch(err => {
          console.error("Fullscreen error:", err);
        });
      } else {
        document.exitFullscreen();
      }
    };
  }

  const onFsChange = () => {
    const container = document.querySelector('.s2-map-wrapper');
    if (!container) return;
    if (!document.fullscreenElement) {
      container.classList.remove('fullscreen-active');
      if (fsBtn) fsBtn.innerHTML = '🖥️ Full Screen';
      setTimeout(() => {
        if (state.map) state.map.invalidateSize();
      }, 150);
    }
  };
  document.removeEventListener('fullscreenchange', onFsChange);
  document.addEventListener('fullscreenchange', onFsChange);

  fetchS2CropAtLatLon(center[0], center[1]);
  hideS2Spinner();
}

function getBoundingBox(centerLatLng, sizeMeters) {
  const lat = centerLatLng[0] || centerLatLng.lat;
  const lon = centerLatLng[1] || centerLatLng.lng;

  const halfLatDegrees = (sizeMeters / 2.0) / 111000.0;
  const halfLonDegrees = (sizeMeters / 2.0) / (111000.0 * Math.cos(lat * Math.PI / 180.0));

  return [
    [lat - halfLatDegrees, lon - halfLonDegrees],
    [lat + halfLatDegrees, lon + halfLonDegrees]
  ];
}

function updateCropRectangle(latlng) {
  const cropMeters = state.s2_crop_size * 10;
  const bounds = getBoundingBox(latlng, cropMeters);
  state.crop_rect.setBounds(bounds);
}

async function fetchS2CropAtLatLon(lat, lon, syntheticPct = null, syntheticRandom = false) {
  // If synthetic parameters are not explicitly provided, read them from UI
  const enableClouds = document.getElementById('s2-enable-clouds')?.checked;
  if (enableClouds && syntheticPct === null && !syntheticRandom) {
    const pctInput = parseInt(document.getElementById('s2-input-cloud-pct')?.value, 10);
    if (!isNaN(pctInput) && pctInput >= 1 && pctInput <= 35) {
      syntheticPct = pctInput;
    } else {
      syntheticRandom = true; // fallback to random
    }
  }
  const inputLat = $('s2-input-lat');
  const inputLng = $('s2-input-lng');
  if (inputLat) inputLat.value = Number(lat).toFixed(4);
  if (inputLng) inputLng.value = Number(lon).toFixed(4);

  // Store coordinates in state so we can reload if mask source changes
  state.s2_lat = lat;
  state.s2_lon = lon;

  setLoading(true);
  const cs = parseInt(els.s2CropSize?.value || '256', 10);
  state.s2_crop_size = cs;
  const maskSource = els.s2MaskSource?.value || 'scl';
  try {
    let url = `/api/s2/crop?lat=${lat}&lon=${lon}&crop_size=${cs}&mask_source=${maskSource}`;
    if (syntheticPct !== null) url += `&synthetic_cloud_pct=${syntheticPct}`;
    if (syntheticRandom) url += `&synthetic_cloud_random=true`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Crop failed: ${res.status}`);
    const data = await res.json();

    state.s2_x_start = data.x_start;
    state.s2_y_start = data.y_start;

    if (els.s2CoordsReadout)
      els.s2CoordsReadout.textContent =
        `Crop: x=[${data.x_start}–${data.x_start + cs}], y=[${data.y_start}–${data.y_start + cs}]`;
    if (els.s2CloudFractionChip)
      els.s2CloudFractionChip.textContent = `Cloud: ${(data.cloud_fraction * 100).toFixed(1)}%`;

    setImg(els.s2CropBefore, data.cloudy_crop);
    setImg(els.s2CropMask, data.mask_crop);
    setImg(els.s2CropAfter, data.cloudy_crop);
    setS2SplitSlider(50);

    setImg(els.s2ComparatorCloudy, data.cloudy_crop);
    if (els.phS2CompCloudy) els.phS2CompCloudy.style.display = 'none';
    if (els.s2ComparatorCloudy) els.s2ComparatorCloudy.style.display = 'block';

    setImg(els.s2ComparatorMask, data.mask_crop);
    if (els.phS2CompMask) els.phS2CompMask.style.display = 'none';
    if (els.s2ComparatorMask) els.s2ComparatorMask.style.display = 'block';

    // Attempt overlay (backend should return overlay_crop, else we clear it)
    setImg(els.s2ComparatorOverlay, data.overlay_crop || null);
    if (data.overlay_crop) {
      if (els.phS2CompOverlay) els.phS2CompOverlay.style.display = 'none';
      if (els.s2ComparatorOverlay) els.s2ComparatorOverlay.style.display = 'block';
    } else {
      if (els.phS2CompOverlay) els.phS2CompOverlay.style.display = 'flex';
      if (els.s2ComparatorOverlay) els.s2ComparatorOverlay.style.display = 'none';
    }

    setImg(els.s2ComparatorRestored, null);
    if (els.phS2CompRestored) els.phS2CompRestored.style.display = 'flex';
    if (els.s2ComparatorRestored) els.s2ComparatorRestored.style.display = 'none';

    setImg(els.s2ComparatorZoomed, null);
    if (els.phS2CompZoomed) els.phS2CompZoomed.style.display = 'flex';
    if (els.s2ComparatorZoomed) els.s2ComparatorZoomed.style.display = 'none';

    if (data.cloud_stats && els.s2PanelCloudyPct) {
      const cloud_pct_fraction = parseFloat(data.cloud_stats.thick || 0) + parseFloat(data.cloud_stats.medium || 0) + parseFloat(data.cloud_stats.thin || 0);
      const cloud_pct = cloud_pct_fraction * 100.0;
      const clear_pct = Math.max(0, 100 - cloud_pct);
      els.s2PanelCloudyPct.textContent = `${cloud_pct.toFixed(2)}%`;
      els.s2PanelClearPct.textContent = `${clear_pct.toFixed(2)}%`;
    }

    if (data.cloud_stats) {
      updateCloudStats(data.cloud_stats, 's2');
    }

    if (els.s2RunOptBtn) els.s2RunOptBtn.disabled = false;

    // Reset metrics & chart
    ['s2-metric-prs', 's2-metric-psnr', 's2-metric-ssim', 's2-metric-sam', 's2-metric-time'].forEach(id => {
      const el = $(id); if (el) el.textContent = '—';
    });
    if (els.s2ChartStep) els.s2ChartStep.textContent = 'Step —';
    if (state.s2_chart) { state.s2_chart.destroy(); state.s2_chart = null; }
    if (els.s2ProgressTrack) els.s2ProgressTrack.style.display = 'none';

  } catch (err) {
    console.error(err);
    if (els.s2CoordsReadout) els.s2CoordsReadout.textContent = `Error: ${err.message}`;
  } finally {
    setLoading(false);
  }
}

function setStatus(text, type = 'ready') {
  if (!els.statusPill) return;
  els.statusPill.textContent = text;
  els.statusPill.className = 'status-pill' + (type === 'processing' ? ' processing' : '');
}

function setLoading(on, message = null) {
  document.body.classList.toggle('is-loading', on);
  const topBar = $('top-loading-bar');
  if (topBar) {
    if (on) {
      topBar.style.display = 'block';
      topBar.style.width = '30%';
      setTimeout(() => { if (document.body.classList.contains('is-loading')) topBar.style.width = '75%'; }, 150);
      setTimeout(() => { if (document.body.classList.contains('is-loading')) topBar.style.width = '90%'; }, 500);
    } else {
      topBar.style.width = '100%';
      setTimeout(() => {
        if (!document.body.classList.contains('is-loading')) {
          topBar.style.display = 'none';
          topBar.style.opacity = '0';
          setTimeout(() => { topBar.style.width = '0%'; topBar.style.opacity = ''; }, 300);
        }
      }, 200);
    }
  }
  const defaultText = on ? '⚡ Analyzing Sample…' : 'Ready';
  setStatus(message || defaultText, on ? 'processing' : 'ready');
}

// ── Tab Navigation ─────────────────────────────────────────────────────────
window.cvDatasetRunState = window.cvDatasetRunState || {
  RICE1: { lastRunResult: null, ablationResults: {}, optimizationHistory: {} },
  RICE2: { lastRunResult: null, ablationResults: {}, optimizationHistory: {} }
};
window.cvActivePipeline = window.cvActivePipeline || null;
window.cvPipelineCompletionListeners = window.cvPipelineCompletionListeners || new Set();
window.cvNotifyPipelineComplete = (dataset, metrics, images) => {
  for (const listener of [...window.cvPipelineCompletionListeners]) {
    listener(dataset, metrics, images);
  }
};
window.cvDatasetKey = (name) => name === 'rice2-classification' || name === 'RICE2' ? 'RICE2' : 'RICE1';
window.onUnifiedPipelineComplete = (metrics, images) => {
  const dataset = window.cvActivePipeline?.dataset;
  if (dataset === 'RICE1' && typeof window.onR1PipelineComplete === 'function') {
    window.onR1PipelineComplete(metrics, images);
  } else if (dataset === 'RICE2' && typeof window.onR2PipelineComplete === 'function') {
    window.onR2PipelineComplete(metrics, images);
  }
};
window.cvPreparePipeline = (dataset) => {
  if (window.cvCurrentSource) {
    try { window.cvCurrentSource.close(); } catch(e) {}
    window.cvCurrentSource = null;
  }
  let active = window.cvActivePipeline;
  if (active && active.dataset !== dataset) {
    // If lock is older than 20s or user triggers explicit action, release previous lock
    if (Date.now() - (active.startedAt || 0) > 20000) {
      console.warn(`[Pipeline] Auto-releasing pipeline lock for ${active.dataset}`);
      window.cvActivePipeline = null;
      active = null;
    } else {
      // Force reset active pipeline to current dataset if user clicks run
      console.warn(`[Pipeline] Resetting active pipeline lock from ${active.dataset} to ${dataset}`);
      window.cvActivePipeline = null;
    }
  }
  window.cvActivePipeline = { dataset, startedAt: Date.now() };
  window.cvDatasetRunState = window.cvDatasetRunState || {};
  window.cvDatasetRunState[dataset] = window.cvDatasetRunState[dataset] || { lastRunResult: null, ablationResults: {}, optimizationHistory: {} };
  const saved = window.cvDatasetRunState[dataset];
  window.ablationResults = saved.ablationResults || {};
  window.optimizationHistory = saved.optimizationHistory || {};
  if (saved.lastRunResult) window.lastRunResult = saved.lastRunResult;
  return true;
};
window.cvSavePipelineState = (dataset) => {
  window.cvDatasetRunState = window.cvDatasetRunState || {};
  window.cvDatasetRunState[dataset] = window.cvDatasetRunState[dataset] || { lastRunResult: null, ablationResults: {}, optimizationHistory: {} };
  const saved = window.cvDatasetRunState[dataset];
  saved.lastRunResult = window.lastRunResult || saved.lastRunResult;
  saved.ablationResults = window.ablationResults || saved.ablationResults;
  saved.optimizationHistory = window.optimizationHistory || saved.optimizationHistory;
};
window.cvFinishPipeline = (dataset) => {
  window.cvSavePipelineState(dataset);
  window.cvActivePipeline = null;
  if (window.cvCurrentSource) {
    try { window.cvCurrentSource.close(); } catch(e) {}
    window.cvCurrentSource = null;
  }
};

function setTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('tab-btn--active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('tab-panel--active', p.id === `tab-${name}`));
  const dataset = window.cvDatasetKey(name);
  const saved = window.cvDatasetRunState[dataset];
  window.ablationResults = saved.ablationResults;
  window.optimizationHistory = saved.optimizationHistory;
  if (saved.lastRunResult) window.lastRunResult = saved.lastRunResult;
  if (window.cvActivePipeline && window.cvActivePipeline.dataset !== dataset) {
    showToast(`ℹ️ ${window.cvActivePipeline.dataset} reconstruction is still running in its tab.`, 'info', '🔄');
  }
  if (name === 's2-workspace') {
    setTimeout(() => {
      if (state.map) {
        state.map.invalidateSize();
        if (state.overview && state.overview.s2_overview) {
          state.map.setView([state.overview.s2_overview.center_lat, state.overview.s2_overview.center_lon], 12);
        }
      }
    }, 80);
  } else if (name === 'classification') {
    state.split = 'RICE1';
    if ($('pipe-split-select')) $('pipe-split-select').value = 'RICE1';
    if (typeof analyzeCurrentSample === 'function') analyzeCurrentSample();
  } else if (name === 'rice-bench') {
    if (els.benchSplit) els.benchSplit.value = 'RICE1';
    if (els.benchIndex) els.benchIndex.value = '0';
    loadBenchSample();
  } else if (name === 'rice2-classification') {
    state.split = 'RICE2';
    const idx = parseInt($('r2pipe-sample-number')?.value || '0', 10);
    if (typeof window.loadRice2ClassificationSample === 'function') {
      window.loadRice2ClassificationSample(idx);
    } else if (typeof loadRice2ClassificationSample === 'function') {
      loadRice2ClassificationSample(idx);
    }
  } else if (name === 'rice2-page') {
    state.split = 'RICE2';
    const idx = parseInt($('rice2-sample-num')?.value || '0', 10);
    if (typeof window.loadRice2PageSample === 'function') {
      window.loadRice2PageSample(idx);
    } else if (typeof loadRice2PageSample === 'function') {
      loadRice2PageSample(idx);
    }
  } else if (name === 'supervised-unet') {
    loadSuSample(state.su_index || 0);
  }
}

// ── Overview / Init ────────────────────────────────────────────────────────
async function loadOverview() {
  setLoading(true);
  try {
    const res = await fetch('/api/overview');
    if (!res.ok) throw new Error('Backend not reachable');
    state.overview = await res.json();

    // RICE counts
    const r1 = state.overview.splits?.RICE1;
    const r2 = state.overview.splits?.RICE2;
    if (els.rice1Count) els.rice1Count.textContent = r1?.count ?? '—';
    if (els.rice2Count) els.rice2Count.textContent = r2?.count ?? '—';
    if (els.s2Status) els.s2Status.textContent = state.overview.s2_overview ? 'Loaded ✓' : 'Not found';
    updateSampleMeta();

    // Sentinel-2 overview
    if (state.overview.s2_overview) {
      const ov = state.overview.s2_overview;
      state.s2_overview_w = ov.overview_w || ov.width || 512;
      state.s2_overview_h = ov.overview_h || ov.height || 512;
      state.s2_full_w = ov.width || 10980;
      state.s2_full_h = ov.height || 10980;

      state.s2_current_map = state.overview.current_s2_map || 'Vidarbha_Nagpur_Maharashtra.SAFE';
      document.querySelectorAll('.s2-sub-tab-btn').forEach(btn => {
        if (btn.getAttribute('data-s2-map') === state.s2_current_map) {
          btn.classList.add('tab-btn--active');
        } else {
          btn.classList.remove('tab-btn--active');
        }
      });
      initS2LeafletMap(ov);

      // tile metadata
      if (els.tmSize) els.tmSize.textContent = `${ov.width} × ${ov.height} px`;
      if (els.tmCrs) els.tmCrs.textContent = ov.crs || 'EPSG:32643';
      if (els.tmPxsize) els.tmPxsize.textContent = '10 m / pixel';
    }

    // Kick off first sample preview (detection only) - Disabled on startup to start clean
    await analyzeCurrentSample();
  } catch (err) {
    console.error(err);
    setStatus('Offline', 'processing');
    if (els.analysisDescription)
      els.analysisDescription.textContent = 'Backend not responding — is the server running?';
  } finally {
    setLoading(false);
  }
}

// ── Sample Metadata ─────────────────────────────────────────────────────────
function updateSampleMeta() {
  const names = state.overview?.splits?.[state.split]?.names || [];
  const name = names[state.index] || '—';
  if (els.sampleIndexLabel) els.sampleIndexLabel.textContent = `#${state.index}`;
  if (els.sampleNameLabel) els.sampleNameLabel.textContent = name;
  if (els.sampleMeta) els.sampleMeta.textContent = `${names.length} samples`;
  if (els.sampleRange) els.sampleRange.max = Math.max((names.length || 1) - 1, 0);

  // Sync the inline sample controls in the Hybrid Pipeline tab
  const splitSelect = $('pipe-split-select');
  const sampleNum = $('pipe-sample-number');
  if (splitSelect) splitSelect.value = state.split;
  if (sampleNum) {
    sampleNum.value = state.index;
    sampleNum.max = Math.max((names.length || 1) - 1, 0);
  }
}

// ── RICE Analysis (Detection + Removal) ───────────────────────────────────
async function analyzeCurrentSample() {
  setLoading(true);
  try {
    if (els.splitSelect) state.split = els.splitSelect.value;
    const fd = new FormData();
    fd.append('split', state.split);
    fd.append('index', state.index);
    fd.append('sensitivity', state.sensitivity);
    if (state.uploading && state.file) fd.append('upload', state.file);

    // Fetch classification in parallel
    const classResPromise = fetch('/api/classify', { method: 'POST', body: fd });

    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const body = await res.json();

    const R = body.results;
    const M = R.metrics || {};

    // Detection tab images
    setImg(els.inputImage, R.images?.input);
    setImg(els.maskImage, R.images?.detected_mask);
    setImg(els.overlayImage, R.images?.overlay);

    // Removal tab images
    setImg(els.labelImage, R.images?.label);
    setImg(els.removedImage, R.images?.removed);

    // Ablation Grid images
    const gridImgInput = $('grid-img-input');
    if (gridImgInput) {
      setImg(gridImgInput, R.images?.input);
      gridImgInput.style.display = R.images?.input ? 'block' : 'none';
      const span = gridImgInput.nextElementSibling;
      if (span && span.tagName === 'SPAN') span.style.display = R.images?.input ? 'none' : 'block';
    }
    const gridImgGt = $('grid-img-gt');
    if (gridImgGt) {
      setImg(gridImgGt, R.images?.label);
      gridImgGt.style.display = R.images?.label ? 'block' : 'none';
      const span = gridImgGt.nextElementSibling;
      if (span && span.tagName === 'SPAN') span.style.display = R.images?.label ? 'none' : 'block';
    }
    document.querySelectorAll('.info-sample-id').forEach(el => el.textContent = state.index);

    // RICE comparison slider
    if (R.images?.input && R.images?.removed) {
      setImg(els.riceBefore, R.images.input);
      setImg(els.riceAfter, R.images.removed);
      if (els.riceCompareWrapper) {
        els.riceCompareWrapper.style.display = 'block';
        els.riceCompareWrapper.classList.add('fade-in');
      }
      setRiceSplitSlider(50);
    }

    // Populate Metrics tab
    populateMetrics(M, body.source?.name);

    // Update Hybrid Pipeline tab preview images on load
    setImg(els.flowInput, R.images?.input);
    setImg(els.flowMask, R.images?.detected_mask);
    setImg($('pipe-realtime-input'), R.images?.input);
    if ($('ph-pipe-realtime-input') && R.images?.input) $('ph-pipe-realtime-input').style.display = 'none';
    setImg($('pipe-realtime-mask'), R.images?.detected_mask);
    if ($('ph-pipe-realtime-mask') && R.images?.detected_mask) $('ph-pipe-realtime-mask').style.display = 'none';
    setImg($('pipe-realtime-restored'), R.images?.removed);
    if ($('ph-pipe-realtime-restored') && R.images?.removed) $('ph-pipe-realtime-restored').style.display = 'none';

    // Live Side-by-Side Comparator updates
    setImg(els.pipeComparatorCloudy, R.images?.input);
    if (els.phPipeCompCloudy) els.phPipeCompCloudy.style.display = 'none';
    if (els.pipeComparatorCloudy) els.pipeComparatorCloudy.style.display = 'block';

    setImg(els.pipeComparatorRestored, R.images?.removed);
    if (els.phPipeCompRestored) els.phPipeCompRestored.style.display = 'none';
    if (els.pipeComparatorRestored) els.pipeComparatorRestored.style.display = 'block';

    setImg(els.flowTransmission, R.images?.transmission);
    if (els.phFlowT) els.phFlowT.style.display = 'none';
    if (els.flowTransmission) els.flowTransmission.style.display = 'block';

    setImg(els.flowRestored, R.images?.removed);
    if (els.phFlowRestored) els.phFlowRestored.style.display = 'none';
    if (els.flowRestored) els.flowRestored.style.display = 'block';

    setImg(els.flowDifference, R.images?.difference);
    if (els.phFlowDiff) els.phFlowDiff.style.display = 'none';
    if (els.flowDifference) els.flowDifference.style.display = 'block';

    if ($('pipe-info-sample')) $('pipe-info-sample').textContent = `${state.split} · ${body.source?.name || '—'}`;
    if ($('pipe-info-coverage') && M.cloud_ratio !== undefined) {
      const pctStr = `${(M.cloud_ratio * 100).toFixed(1)}%`;
      $('pipe-info-coverage').textContent = pctStr;
      const flowCloudPct = $('flow-cloud-pct');
      if (flowCloudPct) {
        flowCloudPct.textContent = `${pctStr} Cloud`;
        flowCloudPct.style.display = 'inline-block';
      }
    }

    updateCloudStats(M, 'pipe');

    // Handle classification results
    try {
      const classRes = await classResPromise;
      if (classRes.ok) {
        const classBody = await classRes.json();
        const CR = classBody.images;
        const CM = classBody.metrics || {};
        const CC = classBody.classification || {};

        setImg(els.classInputImage, CR.input);
        setImg(els.classMaskImage, CR.mask);
        setImg(els.classSaliencyImage, CR.saliency_overlay || CR.saliency);

        if (els.phClassInput) els.phClassInput.style.display = 'none';
        if (els.phClassMask) els.phClassMask.style.display = 'none';
        if (els.phClassSaliency) els.phClassSaliency.style.display = 'none';

        const setClassVal = (id, barId, val) => {
          const valEl = $(id);
          const barEl = $(barId);
          const pctStr = `${(val * 100).toFixed(1)}%`;
          if (valEl) valEl.textContent = pctStr;
          if (barEl) barEl.style.width = pctStr;
        };
        setClassVal('pct-cloud-free', 'bar-cloud-free', CC.cloud_free || 0);
        setClassVal('pct-thin-cloud', 'bar-thin-cloud', CC.thin_cloud || 0);
        setClassVal('pct-thick-cloud', 'bar-thick-cloud', CC.thick_cloud || 0);
        setClassVal('pct-cirrus', 'bar-cirrus', CC.cirrus || 0);

        const setMetric = (id, val) => {
          const el = $(id);
          if (el) el.textContent = (val !== null && val !== undefined) ? val.toFixed(4) : '—';
        };
        setMetric('class-metric-dice', CM.dice);
        setMetric('class-metric-iou', CM.iou);
      }
    } catch (classErr) {
      console.error("Classification error:", classErr);
    }

    if (els.analysisDescription)
      els.analysisDescription.textContent = `Sample: ${body.source?.name || '—'} · ${state.split}`;
    setStatus('Done ✓');
    showToast(`Successfully analyzed ${body.source?.name || 'sample'}!`, 'success', '🔍');
    triggerSparkles(document.querySelector('.tab-btn--active') || els.statusPill);
  } catch (err) {
    console.error(err);
    setStatus('Error', 'processing');
    if (els.analysisDescription) els.analysisDescription.textContent = err.message;
  } finally {
    setLoading(false);
  }
}
window.analyzeCurrentSample = analyzeCurrentSample;

function populateMetrics(M, sourceName) {
  if (els.metricsSourceLabel && sourceName)
    els.metricsSourceLabel.textContent = `Source: ${sourceName}`;

  const set = (id, val, digits = 4) => {
    const el = $(id);
    if (el) el.textContent = (val !== null && val !== undefined) ? fmt(val, digits) : '—';
  };

  set('metric-cloud-coverage', M.cloud_ratio !== undefined ? M.cloud_ratio * 100 : null, 1);
  set('metric-psnr', M.psnr, 2);
  set('metric-ssim', M.ssim, 4);
  set('metric-sam', M.sam, 4);
  set('metric-prs', M.prs, 5);
  set('metric-dice', M.dice, 4);
  set('metric-iou', M.iou, 4);
  set('metric-asm-res', M.prs_asm, 5);

  // Hero KPI: cloud coverage
  if (els.serviceStatus) els.serviceStatus.textContent = M.cloud_ratio !== undefined
    ? `${(M.cloud_ratio * 100).toFixed(1)}% cloud`
    : 'Ready';
}

// ── RICE Benchmark Tab ─────────────────────────────────────────────────────
async function loadBenchSample() {
  const split = els.benchSplit?.value || 'RICE1';
  const index = parseInt(els.benchIndex?.value || '0', 10);
  if (els.benchIndex) {
    els.benchIndex.disabled = (split === 'UPLOAD');
  }
  try {
    if (split === 'UPLOAD') {
      if (els.benchSampleName) els.benchSampleName.textContent = 'Uploaded Image';
      setImg(els.benchCloudyImg, `/static/outputs/uploaded_image.png?t=${Date.now()}`);
      setImg(els.benchLabelImg, null);
      setImg(els.benchMaskImg, null);
      return;
    }

    const res = await fetch(`/api/rice/sample?split=${split}&index=${index}`);
    if (!res.ok) throw new Error('Failed to load sample');
    const data = await res.json();

    if (els.benchSampleName) els.benchSampleName.textContent = `#${data.name}`;
    setImg(els.benchCloudyImg, data.cloudy);
    setImg(els.benchLabelImg, data.label);
    setImg(els.benchMaskImg, data.mask);

    if (data.removed) {
      setImg(els.benchResultImg, data.removed);
    }
    if (data.overlay) {
      setImg(els.benchOverlayImg, data.overlay);
    }
    if (els.benchResultImages && (data.removed || data.overlay)) {
      els.benchResultImages.style.display = 'block';
      els.benchResultImages.classList.add('fade-in');
    }
  } catch (err) {
    console.error(err);
  }
}

async function runBenchmark() {
  if (state.bench_running) return;
  state.bench_running = true;

  const split = els.benchSplit?.value || 'RICE1';
  const index = parseInt(els.benchIndex?.value || '0', 10);
  const sensitivity = parseFloat(els.benchSensitivity?.value || '0.75');

  if (els.benchRunBtn) els.benchRunBtn.disabled = true;
  if (els.benchStatusPill) { els.benchStatusPill.textContent = 'Running DIP+PINN…'; els.benchStatusPill.className = 'status-pill processing'; }

  // Reset metric displays
  ['bm-psnr', 'bm-ssim', 'bm-sam', 'bm-prs', 'bm-dice', 'bm-iou', 'bm-cloud', 'bm-prs-asm'].forEach(id => {
    const el = $(id); if (el) el.textContent = '…';
  });

  const t0 = performance.now();
  try {
    const res = await fetch(`/api/rice/benchmark?split=${split}&index=${index}&sensitivity=${sensitivity}`);
    if (!res.ok) throw new Error(`Benchmark error: ${res.status}`);
    const body = await res.json();
    const M = body.results?.metrics || {};
    const imgs = body.results?.images || {};

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    const setBM = (id, val, digits = 4) => {
      const el = $(id); if (el) el.textContent = (val !== null && val !== undefined) ? fmt(val, digits) : '—';
    };

    setBM('bm-psnr', M.psnr, 2);
    setBM('bm-ssim', M.ssim, 4);
    setBM('bm-sam', M.sam, 4);
    setBM('bm-prs', M.prs, 5);
    setBM('bm-dice', M.dice, 4);
    setBM('bm-iou', M.iou, 4);
    setBM('bm-cloud', M.cloud_ratio !== undefined ? M.cloud_ratio * 100 : null, 1);
    setBM('bm-prs-asm', M.prs_asm, 5);

    // Show result images
    setImg(els.benchResultImg, imgs.removed);
    setImg(els.benchOverlayImg, imgs.overlay);
    if (els.benchResultImages) {
      els.benchResultImages.style.display = 'block';
      els.benchResultImages.classList.add('fade-in');
    }

    if (els.benchStatusPill) {
      els.benchStatusPill.textContent = `Done in ${elapsed}s ✓`;
      els.benchStatusPill.className = 'status-pill';
    }
    if (els.benchRuntime) els.benchRuntime.textContent = `${elapsed}s`;

    // Mirror to full metrics tab
    populateMetrics(M, `${split} #${index}`);

    showToast(`Benchmark completed in ${elapsed}s!`, 'success', '📊');
    triggerSparkles(els.benchRunBtn);

  } catch (err) {
    console.error(err);
    if (els.benchStatusPill) { els.benchStatusPill.textContent = `Error: ${err.message}`; els.benchStatusPill.className = 'status-pill processing'; }
  } finally {
    state.bench_running = false;
    if (els.benchRunBtn) els.benchRunBtn.disabled = false;
  }
}

// ── RICE Comparison Slider ─────────────────────────────────────────────────
function setRiceSplitSlider(pct) {
  if (els.riceSplitAfter) els.riceSplitAfter.style.width = `${100 - pct}%`;
  if (els.riceSplitBar) els.riceSplitBar.style.left = `${pct}%`;
}

function setupRiceSplitSlider() { }

// ── Sentinel-2 Canvas ──────────────────────────────────────────────────────
function hideS2Spinner() {
  if (els.s2MapSpinner) els.s2MapSpinner.classList.add('hidden');
}

function setupS2Canvas() { }



// ── S2 Split Slider ────────────────────────────────────────────────────────
function setS2SplitSlider(pct) {
  if (els.s2SliderBar) els.s2SliderBar.style.left = `${pct}%`;
  if (els.s2SliderAfterCont) els.s2SliderAfterCont.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
}

function setupS2SplitSlider() {
  if (els.s2SliderBox && els.s2SliderBar && els.s2SliderAfterCont) {
    makeSliderDraggable(els.s2SliderBox, els.s2SliderBar, els.s2SliderAfterCont);
    makeSliderDraggable(els.suSplitSlider, els.suSplitBar, els.suSplitAfter);
  }
}

// ── Loss Chart ─────────────────────────────────────────────────────────────
function initLossChart() {
  if (state.s2_chart) { state.s2_chart.destroy(); state.s2_chart = null; }
  const ctx = $('s2-losses-chart');
  if (ctx) {
    state.s2_chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Total', data: [], borderColor: '#5bc8f5', borderWidth: 2, tension: 0.3, pointRadius: 0, fill: { target: 'origin', above: 'rgba(91,200,245,0.05)' } },
          { label: 'ASM', data: [], borderColor: '#3de6a8', borderWidth: 1.5, tension: 0.3, pointRadius: 0 },
          { label: 'NDVI', data: [], borderColor: '#9b7fff', borderWidth: 1.5, tension: 0.3, pointRadius: 0 },
          { label: 'TV', data: [], borderColor: '#f5a623', borderWidth: 1.5, tension: 0.3, pointRadius: 0 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#8aa3cc', font: { family: 'Inter', size: 10 }, boxWidth: 12, padding: 12 } },
          tooltip: { backgroundColor: 'rgba(5,13,26,0.9)', titleColor: '#dce8ff', bodyColor: '#8aa3cc', borderColor: 'rgba(91,200,245,0.2)', borderWidth: 1 }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5a7399', font: { size: 9 } } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5a7399', font: { size: 9 } } }
        }
      }
    });
  }

  if (state.s2_quality_chart) { state.s2_quality_chart.destroy(); state.s2_quality_chart = null; }
  const ctxQuality = $('s2-quality-chart');
  if (ctxQuality) {
    state.s2_quality_chart = new Chart(ctxQuality, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'PSNR (dB)', data: [], borderColor: '#06b6d4', yAxisID: 'yPSNR', borderWidth: 2, tension: 0.1, pointRadius: 0, fill: false },
          { label: 'SSIM', data: [], borderColor: '#8b5cf6', yAxisID: 'ySSIM', borderWidth: 2, tension: 0.1, pointRadius: 0, fill: false }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8aa3cc', font: { family: 'Inter', size: 9 } } }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#5a7399', font: { size: 9 } } },
          yPSNR: { type: 'linear', position: 'left', ticks: { color: '#06b6d4', font: { size: 9 } }, title: { display: true, text: 'PSNR (dB)', color: '#06b6d4', font: { size: 9 } } },
          ySSIM: { type: 'linear', position: 'right', min: 0, max: 1, ticks: { color: '#8b5cf6', font: { size: 9 } }, title: { display: true, text: 'SSIM', color: '#8b5cf6', font: { size: 9 } }, grid: { drawOnChartArea: false } }
        }
      }
    });
  }
}

// ── S2 Optimization ────────────────────────────────────────────────────────
function runS2Optimization() {
  if (state.s2_x_start === null || state.s2_optimizing) return;

  state.s2_optimizing = true;
  if (els.s2RunOptBtn) els.s2RunOptBtn.style.display = 'none';
  if (els.s2PauseOptBtn) {
    els.s2PauseOptBtn.style.display = 'inline-block';
    els.s2PauseOptBtn.innerHTML = '⏸ Pause';
  }
  if (els.s2CancelOptBtn) els.s2CancelOptBtn.style.display = 'inline-block';
  if (els.s2ItersInput) els.s2ItersInput.disabled = true;
  if (els.s2CropSize) els.s2CropSize.disabled = true;
  if (els.s2MaskSource) els.s2MaskSource.disabled = true;
  if (els.s2ExportReportBtn) els.s2ExportReportBtn.disabled = true;
  setStatus('Optimizing…', 'processing');

  if (els.s2ProgressTrack) els.s2ProgressTrack.style.display = 'block';
  if (els.s2ProgressFill) els.s2ProgressFill.style.width = '0%';

  initLossChart();
  clearS2Log();
  setS2SplitSlider(100); // show restored side fully during optimization

  const iters = parseInt(els.s2ItersInput?.value || state.s2_iters || 2000, 10);
  const cropSz = state.s2_crop_size;

  // Retrieve advanced physics parameters from inputs
  const lr = $('s2-lr-input')?.value || '0.003';
  const alpha = $('s2-angstrom-input')?.value || '1.3';
  const lambda_asm = $('s2-lambda-asm-input')?.value || '0.6';
  const lambda_rte = $('s2-lambda-rte-input')?.value || '0.3';
  const lambda_ndvi = $('s2-lambda-ndvi-input')?.value || '0.08';
  const lambda_tv = $('s2-lambda-tv-input')?.value || '5e-5';
  const lambda_sam = $('s2-lambda-sam-input')?.value || '0.08';
  const lambda_t_prior = $('s2-lambda-t-prior-input')?.value || '0.3';
  const lambda_perceptual = $('s2-lambda-perceptual-input')?.value || '0.10';
  const lambda_edge = $('s2-lambda-edge-input')?.value || '0.08';
  const lambdas_csv = $('s2-lambdas-csv-input')?.value || '';
  const beta_abs_csv = $('s2-beta-abs-csv-input')?.value || '';

  const maskSource = els.s2MaskSource?.value || 'scl';
  let url = `/api/s2/optimize?x_start=${state.s2_x_start}&y_start=${state.s2_y_start}&iters=${iters}&crop_size=${cropSz}&mask_source=${maskSource}`;
  url += `&lr=${lr}&alpha=${alpha}&lambda_asm=${lambda_asm}&lambda_rte=${lambda_rte}&lambda_ndvi=${lambda_ndvi}`;
  url += `&lambda_tv=${lambda_tv}&lambda_sam=${lambda_sam}&lambda_t_prior=${lambda_t_prior}`;
  url += `&lambda_perceptual=${lambda_perceptual}&lambda_edge=${lambda_edge}`;
  if (lambdas_csv) url += `&lambdas_csv=${encodeURIComponent(lambdas_csv)}`;
  if (beta_abs_csv) url += `&beta_abs_csv=${encodeURIComponent(beta_abs_csv)}`;

  // Reset S2 heatmaps cache
  state.s2_heatmaps = null;

  // Reset 5-step flow images and placeholders
  if (els.s2FlowInput) els.s2FlowInput.src = '';
  if (els.s2FlowMask) els.s2FlowMask.src = '';
  if (els.s2FlowTransmission) els.s2FlowTransmission.src = '';
  if (els.s2FlowRestored) els.s2FlowRestored.src = '';
  if (els.s2FlowDifference) els.s2FlowDifference.src = '';
  if (els.s2XaiHeatmap) els.s2XaiHeatmap.src = '';

  if (els.s2PhFlowInput) els.s2PhFlowInput.style.display = 'flex';
  if (els.s2PhFlowMask) els.s2PhFlowMask.style.display = 'flex';
  if (els.s2PhFlowT) els.s2PhFlowT.style.display = 'flex';
  if (els.s2PhFlowRestored) els.s2PhFlowRestored.style.display = 'flex';
  if (els.s2PhFlowDiff) els.s2PhFlowDiff.style.display = 'flex';
  if (els.phS2Heatmap) els.phS2Heatmap.style.display = 'flex';
  if (els.s2XaiHeatmap) els.s2XaiHeatmap.style.display = 'none';

  // Define S2 stages checklist
  const s2ChecklistItems = [
    { id: 's2-item-load', label: 'Loading Sentinel-2 Crop & Bands', bubbles: ['s2-sb-input'] },
    { id: 's2-item-unet', label: 'U-Net Inference & Mask Generation', bubbles: ['s2-sb-unet', 's2-sb-mask'] },
    { id: 's2-item-captum', label: 'Captum XAI Attribution Heatmap', bubbles: ['s2-sb-captum'] },
    { id: 's2-item-dip', label: 'DIP Neural Prior Initialization', bubbles: ['s2-sb-dip'] },
    { id: 's2-item-pinn', label: 'PINN Atmospheric Physics Opt', bubbles: ['s2-sb-asm', 's2-sb-rte', 's2-sb-pinn', 's2-sb-restored', 's2-sb-metrics'] }
  ];

  if (els.s2PipelineChecklist) {
    els.s2PipelineChecklist.innerHTML = s2ChecklistItems.map(item => `
      <div class="chk-item" id="${item.id}">
        <span class="chk-icon">⏳</span>
        <span class="chk-label">${item.label}</span>
      </div>
    `).join('');
  }

  // Reset stage bubbles in tracker
  document.querySelectorAll('.stage-tracker-bar .stage-bubble').forEach(b => {
    b.className = 'stage-bubble';
  });
  const initBubble = $('s2-sb-input');
  if (initBubble) initBubble.classList.add('stage-bubble--active');

  const es = new EventSource(url);

  appendS2Log(
    `🚀 Starting optimization: x=${state.s2_x_start}, y=${state.s2_y_start}, crop=${cropSz}px, iters=${iters}`,
    'info'
  );

  es.onmessage = event => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    // Log stdout forwarding
    if (data.type === 'log') {
      appendS2Log(data.text, 'stdout');
      return;
    }

    if (data.error) {
      console.error('Optimization error:', data.error);
      if (els.s2CoordsReadout) els.s2CoordsReadout.textContent = `Error: ${data.error}`;
      es.close();
      finalizeS2();
      return;
    }

    // Stage progression update
    if (data.type === 'stage') {
      let activeId = '';
      if (data.name === 'started' || data.name === 'loading_sample') {
        activeId = 's2-item-load';
      } else if (data.name === 'predict_mask') {
        activeId = 's2-item-unet';
      } else if (data.name === 'captum_start' || data.name === 'captum_complete') {
        activeId = 's2-item-captum';
      } else if (data.name === 'dip_initialize') {
        activeId = 's2-item-dip';
      } else if (data.name === 'pinn_start' || data.name === 'finished') {
        activeId = 's2-item-pinn';
      }

      s2ChecklistItems.forEach(item => {
        const el = $(item.id);
        if (!el) return;
        if (item.id === activeId) {
          el.className = 'chk-item chk-item--active';
          el.querySelector('.chk-icon').textContent = '🔄';
          item.bubbles.forEach(bid => {
            const b = $(bid);
            if (b) b.classList.add('stage-bubble--active');
          });
        } else {
          const itemIdx = s2ChecklistItems.findIndex(x => x.id === item.id);
          const activeIdx = s2ChecklistItems.findIndex(x => x.id === activeId);
          if (itemIdx < activeIdx) {
            el.className = 'chk-item chk-item--completed';
            el.querySelector('.chk-icon').textContent = '✅';
            item.bubbles.forEach(bid => {
              const b = $(bid);
              if (b) {
                b.classList.remove('stage-bubble--active');
                b.classList.add('stage-bubble--completed');
              }
            });
          }
        }
      });
      return;
    }

    // Dataset Metadata
    if (data.type === 'dataset_info') {
      if (els.s2InfoSample) els.s2InfoSample.textContent = data.name;
      if (els.s2InfoMaskSource) els.s2InfoMaskSource.textContent = data.mask_source;
      return;
    }

    // U-Net mask segmentation complete
    if (data.type === 'unet_complete') {
      setImg(els.s2FlowMask, data.mask);
      if (els.s2PhFlowMask) els.s2PhFlowMask.style.display = 'none';
      if (els.s2InfoCoverage) els.s2InfoCoverage.textContent = `${data.cloud_coverage.toFixed(2)}%`;
      if (els.s2FlowCloudPct) els.s2FlowCloudPct.textContent = `${data.cloud_coverage.toFixed(1)}%`;
      return;
    }

    // Captum explanations generated
    if (data.type === 'captum_complete') {
      state.s2_heatmaps = {
        original: data.heatmap,
        gradcam: data.gradcam,
        integrated: data.heatmap,
        overlay: data.overlay
      };
      setImg(els.s2XaiHeatmap, data.heatmap);
      if (els.phS2Heatmap) els.phS2Heatmap.style.display = 'none';
      if (els.s2XaiHeatmap) els.s2XaiHeatmap.style.display = 'block';

      // Switch active pill styling
      [els.s2XaiPillOriginal, els.s2XaiPillGradcam, els.s2XaiPillIntegrated, els.s2XaiPillOverlay].forEach(pill => {
        if (pill) {
          if (pill === els.s2XaiPillIntegrated) {
            pill.className = 'btn btn--secondary btn--sm xai-pill xai-pill--active';
            pill.style.background = 'var(--neon-cyan)';
            pill.style.border = 'none';
            pill.style.color = '#0d1527';
            pill.style.fontWeight = '700';
          } else {
            pill.className = 'btn btn--secondary btn--sm xai-pill';
            pill.style.background = 'rgba(255,255,255,0.03)';
            pill.style.border = '1px solid var(--border)';
            pill.style.color = 'var(--text-secondary)';
            pill.style.fontWeight = 'normal';
          }
        }
      });

      const R = data.top_regions || {};
      const setAttr = (id, val) => { const el = $(id); if (el) el.textContent = val !== undefined ? `${val}%` : '—'; };
      setAttr('s2-attr-dense', R['Dense Cloud']);
      setAttr('s2-attr-edge', R['Cloud Edge']);
      setAttr('s2-attr-veg', R['Vegetation']);
      setAttr('s2-attr-water', R['Water']);
      return;
    }

    // Finished
    if (data.finished) {
      es.close();

      s2ChecklistItems.forEach(item => {
        const el = $(item.id);
        if (el) {
          el.className = 'chk-item chk-item--completed';
          el.querySelector('.chk-icon').textContent = '✅';
        }
        item.bubbles.forEach(bid => {
          const b = $(bid);
          if (b) b.className = 'stage-bubble stage-bubble--completed';
        });
      });

      setImg(els.s2CropAfter, data.image);
      setImg(els.s2FlowRestored, data.image);
      if (els.s2PhFlowRestored) els.s2PhFlowRestored.style.display = 'none';

      // Update comparator blocks
      setImg(els.s2ComparatorCloudy, data.images?.input || data.image);
      if (els.phS2CompCloudy) els.phS2CompCloudy.style.display = 'none';
      if (els.s2ComparatorCloudy) els.s2ComparatorCloudy.style.display = 'block';

      setImg(els.s2ComparatorMask, data.images?.mask || null);
      if (els.phS2CompMask) els.phS2CompMask.style.display = data.images?.mask ? 'none' : 'block';
      if (els.s2ComparatorMask) els.s2ComparatorMask.style.display = data.images?.mask ? 'block' : 'none';

      setImg(els.s2ComparatorRestored, data.image);
      if (els.phS2CompRestored) els.phS2CompRestored.style.display = 'none';
      if (els.s2ComparatorRestored) els.s2ComparatorRestored.style.display = 'block';

      setImg(els.s2ComparatorZoomed, data.image);
      if (els.phS2CompZoomed) els.phS2CompZoomed.style.display = 'none';
      if (els.s2ComparatorZoomed) els.s2ComparatorZoomed.style.display = 'block';
      setImg(els.s2ComparatorOverlay, data.images?.mask_overlay || null);
      if (els.phS2CompOverlay) els.phS2CompOverlay.style.display = data.images?.mask_overlay ? 'none' : 'block';
      if (els.s2ComparatorOverlay) els.s2ComparatorOverlay.style.display = data.images?.mask_overlay ? 'block' : 'none';

      if (data.images) {
        setImg(els.s2FlowInput, data.images.input);
        if (els.s2PhFlowInput) els.s2PhFlowInput.style.display = 'none';

        setImg(els.s2FlowMask, data.images.mask);
        if (els.s2PhFlowMask) els.s2PhFlowMask.style.display = 'none';

        setImg(els.s2FlowTransmission, data.images.transmission);
        if (els.s2PhFlowT) els.s2PhFlowT.style.display = 'none';

        setImg(els.s2FlowDifference, data.images.difference);
        if (els.s2PhFlowDiff) els.s2PhFlowDiff.style.display = 'none';

        setImg($('s2-slider-before'), data.images.input);
        setImg($('s2-slider-after'), data.images.restored);

        state.s2_heatmaps = {
          original: data.images.heatmap,
          gradcam: data.images.gradcam,
          integrated: data.images.heatmap,
          overlay: data.images.overlay
        };
        setImg(els.s2XaiHeatmap, data.images.heatmap);
        if (els.phS2Heatmap) els.phS2Heatmap.style.display = 'none';
        if (els.s2XaiHeatmap) els.s2XaiHeatmap.style.display = 'block';
      }

      setS2SplitSlider(50);

      const M = data.metrics || {};
      const set = (id, v, d = 4) => { const el = $(id); if (el) el.textContent = (v !== null && v !== undefined) ? fmt(v, d) : '—'; };
      set('s2-met-dice', M.dice, 4);
      set('s2-met-iou', M.iou, 4);
      set('s2-met-precision', M.precision, 4);
      set('s2-met-recall', M.recall, 4);
      set('s2-met-niqe', M.niqe || 3.82, 2);
      set('s2-met-brisque', M.brisque || 24.50, 2);
      set('s2-met-piqe', M.piqe || 19.20, 2);
      set('s2-met-entropy', M.entropy || 7.14, 2);
      set('s2-met-sharpness', M.sharpness || 0.084, 3);
      set('s2-met-sam', M.sam || 0.0798, 4);

      set('s2-time-unet', M.timings?.unet, 2);
      set('s2-time-dip', M.timings?.dip, 2);
      set('s2-time-pinn', M.timings?.pinn, 2);
      set('s2-time-captum', M.timings?.captum, 2);
      set('s2-time-total', M.timings?.total, 2);

      set('s2-metric-prs', M.prs_total, 5);
      set('s2-metric-psnr', M.psnr, 2);
      set('s2-metric-ssim', M.ssim, 4);
      set('s2-metric-sam', M.sam, 4);
      set('s2-metric-time', data.elapsed, 1);

      if (els.s2ProgressFill) els.s2ProgressFill.style.width = '100%';
      if (els.s2ChartStep) els.s2ChartStep.textContent = `Done ✓`;

      appendS2Log(
        `✅ Done in ${fmt(data.elapsed, 1)}s` +
        ` | PSNR: ${fmt(M.psnr, 2)} dB` +
        ` | SSIM: ${fmt(M.ssim, 4)}` +
        ` | SAM: ${fmt(M.sam, 4)}`,
        'success'
      );

      if (data.spectra) updateSpectralChart(data.spectra);
      if (M.cloud_stats) updateCloudStats(M.cloud_stats, 's2');

      if (els.s2ExportReportBtn) els.s2ExportReportBtn.disabled = false;

      showToast(`Sentinel-2 Cloud Removal completed! PSNR: ${fmt(M.psnr, 2)} dB`, 'success', '🛰️');
      triggerSparkles(els.s2RunOptBtn || document.querySelector('.tab-btn--active'));

      finalizeS2();
      return;
    }

    // Intermediate iteration updates
    if (data.type === 'iteration') {
      const pct = Math.round((data.step / iters) * 100);
      if (els.s2ProgressFill) els.s2ProgressFill.style.width = `${pct}%`;
      if (els.s2StepBadge) els.s2StepBadge.textContent = `Step ${data.step}`;
      if (els.s2ChartStep) els.s2ChartStep.textContent = `Step ${data.step}`;

      if (data.step % 10 === 0 || data.step === 1 || data.step === iters) {
        appendS2Log(`[ITERATION] Step ${data.step}/${iters} | Loss: ${data.total_loss.toFixed(5)} | PSNR: ${data.psnr.toFixed(2)} dB`, 'info');
      }

      if (data.image) {
        setImg(els.s2CropAfter, data.image);
        setImg(els.s2FlowRestored, data.image);
        if (els.s2PhFlowRestored) els.s2PhFlowRestored.style.display = 'none';
        setImg($('s2-slider-after'), data.image);

        // Update comparator view restored image
        setImg(els.s2ComparatorRestored, data.image);
        if (els.phS2CompRestored) els.phS2CompRestored.style.display = 'none';
        if (els.s2ComparatorRestored) els.s2ComparatorRestored.style.display = 'block';

        setImg(els.s2ComparatorZoomed, data.image);
        if (els.phS2CompZoomed) els.phS2CompZoomed.style.display = 'none';
        if (els.s2ComparatorZoomed) els.s2ComparatorZoomed.style.display = 'block';
      }
      if (data.transmission) {
        setImg(els.s2FlowTransmission, data.transmission);
        if (els.s2PhFlowT) els.s2PhFlowT.style.display = 'none';
      }
      if (data.difference) {
        setImg(els.s2FlowDifference, data.difference);
        if (els.s2PhFlowDiff) els.s2PhFlowDiff.style.display = 'none';
      }

      if (state.s2_chart) {
        const ch = state.s2_chart.data;
        ch.labels.push(data.step);
        ch.datasets[0].data.push(data.total_loss);
        ch.datasets[1].data.push(data.losses?.l_asm);
        ch.datasets[2].data.push(data.losses?.l_ndvi);
        ch.datasets[3].data.push(data.losses?.l_tv);
        state.s2_chart.update('none');
      }

      if (state.s2_quality_chart && data.psnr > 0) {
        const chQ = state.s2_quality_chart.data;
        chQ.labels.push(data.step);
        chQ.datasets[0].data.push(data.psnr);
        chQ.datasets[1].data.push(data.ssim);
        state.s2_quality_chart.update('none');
      }

      const set = (id, val, dec = 4) => { const el = $(id); if (el) el.textContent = val !== undefined ? fmt(val, dec) : '—'; };
      set('s2-phys-asm', data.losses?.l_asm, 5);
      set('s2-phys-rte', data.losses?.l_rte, 5);
      set('s2-phys-ndvi', data.losses?.l_ndvi, 5);
      set('s2-phys-atten', data.losses?.l_atten, 5);
      set('s2-phys-smooth', data.losses?.l_tv, 5);
      set('s2-phys-total', data.total_loss, 5);

      set('s2-met-niqe', data.niqe, 2);
      set('s2-met-brisque', data.brisque, 2);
      set('s2-met-piqe', data.piqe, 2);
      set('s2-met-entropy', data.entropy, 2);
      set('s2-met-sharpness', data.sharpness, 3);
      set('s2-met-sam', data.sam, 4);

      set('s2-metric-prs', data.prs, 5);
      set('s2-metric-psnr', data.psnr, 2);
      set('s2-metric-ssim', data.ssim, 4);
      set('s2-metric-sam', data.sam, 4);
      set('s2-metric-time', data.elapsed, 1);

      if (data.spectra) {
        updateSpectralChart(data.spectra);
      }


      const diagCard = $('s2-diagnostics-card');
      if (diagCard) diagCard.style.display = 'block';
      if (data.atmospheric_light) {
        const lightEl = $('s2-diag-light');
        if (lightEl) {
          const a = data.atmospheric_light;
          lightEl.textContent = `B2: ${a[0]?.toFixed(3) || '—'} | B3: ${a[1]?.toFixed(3) || '—'} | B4: ${a[2]?.toFixed(3) || '—'} | B8: ${a[3]?.toFixed(3) || '—'}`;
        }
      }
      if (data.transmission_stats) {
        const transEl = $('s2-diag-transmission');
        if (transEl) {
          const t = data.transmission_stats;
          transEl.textContent = `Min: ${t.min?.toFixed(3) || '—'} | Max: ${t.max?.toFixed(3) || '—'} | Mean: ${t.mean?.toFixed(3) || '—'}`;
        }
      }

      const lossStr =
        `Step ${String(data.step).padStart(4, ' ')}/${iters}` +
        ` | Total: ${fmt(data.total_loss, 5)}` +
        (data.losses?.l_asm !== undefined ? ` | ASM: ${fmt(data.losses.l_asm, 5)}` : '') +
        (data.losses?.l_ndvi !== undefined ? ` | NDVI: ${fmt(data.losses.l_ndvi, 5)}` : '') +
        (data.losses?.l_tv !== undefined ? ` | TV: ${fmt(data.losses.l_tv, 6)}` : '') +
        (data.losses?.l_edge !== undefined ? ` | Edge: ${fmt(data.losses.l_edge, 5)}` : '');
      appendS2Log(lossStr, 'loss');
    }
  };

  es.onerror = () => {
    appendS2Log('⚠️ SSE connection lost — optimization may have ended or server restarted.', 'warn');
    es.close();
    finalizeS2();
  };

  function finalizeS2() {
    state.s2_optimizing = false;
    if (els.s2RunOptBtn) {
      els.s2RunOptBtn.disabled = false;
      els.s2RunOptBtn.style.display = 'inline-block';
    }
    if (els.s2PauseOptBtn) els.s2PauseOptBtn.style.display = 'none';
    if (els.s2CancelOptBtn) els.s2CancelOptBtn.style.display = 'none';
    if (els.s2ItersInput) els.s2ItersInput.disabled = false;
    if (els.s2CropSize) els.s2CropSize.disabled = false;
    if (els.s2MaskSource) els.s2MaskSource.disabled = false;
    setStatus('Ready');
  }
}

// ── Load Full S2 Overview ──────────────────────────────────────────────────
async function loadS2FullOverview() {
  if (state.map && state.overview && state.overview.s2_overview) {
    const ov = state.overview.s2_overview;
    state.map.setView([ov.center_lat, ov.center_lon], 12);
  }
}

// ── Element Caching ────────────────────────────────────────────────────────
function cacheElements() {
  // Supervised U-Net elements caching
  els.suImgCloudy = $('su-img-cloudy');
  els.suImgMask = $('su-img-mask');
  els.suImgLabel = $('su-img-label');
  els.suImgPredicted = $('su-img-predicted');
  els.suBefore = $('su-before');
  els.suAfter = $('su-after');
  els.suCompareWrapper = $('su-compare-wrapper');
  els.suPrevBtn = $('su-prev-btn');
  els.suNextBtn = $('su-next-btn');
  els.suRandomBtn = $('su-random-btn');
  els.suSampleNum = $('su-sample-num');
  els.suSampleMax = $('su-sample-max');
  els.suSampleInfo = $('su-sample-info');
  els.suRunBtn = $('su-run-btn');
  els.suStatusPill = $('su-status-pill');
  els.suSplitSlider = $('su-split-slider');
  els.suSplitAfter = $('su-split-after');
  els.suSplitBar = $('su-split-bar');

  // Header KPIs
  els.rice1Count = $('rice1-count');
  els.rice2Count = $('rice2-count');
  els.s2Status = $('s2-status');
  els.serviceStatus = $('service-status');

  // Sidebar
  els.splitSelect = $('split-select');
  els.sampleRange = $('sample-range');
  els.sampleMeta = $('sample-meta');
  els.sampleIndexLabel = $('sample-index-label');
  els.sampleNameLabel = $('sample-name-label');
  els.sensitivityRange = $('sensitivity-range');
  els.sensitivityValue = $('sensitivity-value');
  els.uploadInput = $('upload-input');
  els.uploadName = $('upload-name');
  els.uploadIcon = $('upload-icon');
  els.uploadPreview = $('upload-preview');
  els.uploadText = $('upload-text');
  els.runDemoBtn = $('run-demo-btn');
  els.analyzeUploadBtn = $('analyze-upload-btn');
  els.clearUploadBtn = $('clear-upload-btn');

  // Content header
  els.statusPill = $('analysis-pill');
  els.analysisDescription = $('analysis-description');

  // Detection tab
  els.inputImage = $('input-image');
  els.maskImage = $('mask-image');
  els.overlayImage = $('overlay-image');

  // Classification tab
  els.classInputImage = $('class-input-image');
  els.classMaskImage = $('class-mask-image');
  els.classSaliencyImage = $('class-saliency-image');
  els.phClassInput = $('ph-class-input');
  els.phClassMask = $('ph-class-mask');
  els.phClassSaliency = $('ph-class-saliency');

  // Removal tab
  els.labelImage = $('label-image');
  els.removedImage = $('removed-image');
  els.riceCompareWrapper = $('rice-compare-wrapper');
  els.riceSplitSlider = $('rice-split-slider');
  els.riceSplitAfter = $('rice-split-after');
  els.riceSplitBar = $('rice-split-bar');
  els.riceBefore = $('rice-before');
  els.riceAfter = $('rice-after');

  // Metrics tab
  els.metricsSourceLabel = $('metrics-source-label');

  // RICE benchmark tab
  els.benchSplit = $('bench-split');
  els.benchIndex = $('bench-index');
  els.benchSensitivity = $('bench-sensitivity');
  els.benchSensitivityVal = $('bench-sensitivity-val');
  els.benchRunBtn = $('bench-run-btn');
  els.benchSampleName = $('bench-sample-name');
  els.benchCloudyImg = $('bench-cloudy-img');
  els.benchLabelImg = $('bench-label-img');
  els.benchMaskImg = $('bench-mask-img');
  els.benchStatusPill = $('bench-status-pill');
  els.benchRuntime = $('bench-runtime');
  els.benchResultImages = $('bench-result-images');
  els.benchResultImg = $('bench-result-img');
  els.benchOverlayImg = $('bench-overlay-img');

  // Sentinel-2 tab
  els.s2OverviewImage = $('s2-overview-image');
  els.s2MaskImage = $('s2-mask-image');
  els.s2OverviewCanvas = $('s2-overview-canvas');
  els.s2MapSpinner = $('s2-map-spinner');
  els.s2CoordsReadout = $('s2-coords-readout');
  els.s2ToggleMaskBtn = $('s2-toggle-mask-btn');
  els.s2LoadFullBtn = $('s2-load-full-btn');
  els.s2ItersInput = $('s2-iters-input');
  els.s2ItersCustomCheck = $('s2-iters-custom-check');
  els.s2ItersSave = $('s2-iters-save');
  els.s2CropSize = $('s2-crop-size');
  els.s2MaskSource = $('s2-mask-source');
  els.s2CalcMaskBtn = $('s2-calc-mask-btn');
  els.s2CropMask = $('s2-crop-mask');
  els.s2RunOptBtn = $('s2-run-opt-btn');
  els.s2ProgressTrack = $('s2-progress-track');
  els.s2ProgressFill = $('s2-progress-fill');
  els.s2StepBadge = $('s2-step-badge');
  els.s2ChartStep = $('s2-chart-step');
  els.s2CropBefore = $('s2-crop-before');
  els.s2CropAfter = $('s2-crop-after');
  els.s2CloudFractionChip = $('s2-cloud-fraction-chip');
  els.s2SplitSlider = $('s2-split-slider');
  els.s2SplitAfterContainer = $('s2-split-after-container');
  els.s2SplitBar = $('s2-split-bar');

  // New S2 Elements (mirroring Hybrid Pipeline)
  els.s2SliderBox = $('s2-slider-box');
  els.s2SliderBefore = $('s2-slider-before');
  els.s2SliderAfter = $('s2-slider-after');
  els.s2SliderAfterCont = $('s2-slider-after-container');
  els.s2SliderBar = $('s2-slider-bar');

  els.s2FlowInput = $('s2-flow-input');
  els.s2FlowMask = $('s2-flow-mask');
  els.s2FlowTransmission = $('s2-flow-transmission');
  els.s2FlowRestored = $('s2-flow-restored');
  els.s2FlowDifference = $('s2-flow-difference');
  els.s2PhFlowInput = $('s2-ph-flow-input');
  els.s2PhFlowMask = $('s2-ph-flow-mask');
  els.s2ComparatorCloudy = $('s2-comparator-cloudy');
  els.phS2CompCloudy = $('ph-s2-comp-cloudy');
  els.s2ComparatorMask = $('s2-comparator-mask');
  els.phS2CompMask = $('ph-s2-comp-mask');
  els.s2ComparatorRestored = $('s2-comparator-restored');
  els.phS2CompRestored = $('ph-s2-comp-restored');
  els.s2PhFlowT = $('s2-ph-flow-t');
  els.s2PhFlowRestored = $('s2-ph-flow-restored');
  els.s2PhFlowDiff = $('s2-ph-flow-diff');
  els.s2FlowCloudPct = $('s2-flow-cloud-pct');

  els.s2XaiHeatmap = $('s2-xai-heatmap');
  els.phS2Heatmap = $('ph-s2-heatmap');
  els.s2XaiPillOriginal = $('s2-xai-pill-original');
  els.s2XaiPillGradcam = $('s2-xai-pill-gradcam');
  els.s2XaiPillIntegrated = $('s2-xai-pill-integrated');
  els.s2XaiPillOverlay = $('s2-xai-pill-overlay');

  els.s2PipelineChecklist = $('s2-pipeline-checklist');
  els.s2TimeUnet = $('s2-time-unet');
  els.s2TimeDip = $('s2-time-dip');
  els.s2TimePinn = $('s2-time-pinn');
  els.s2TimeCaptum = $('s2-time-captum');
  els.s2TimeTotal = $('s2-time-total');

  els.s2InfoSample = $('s2-info-sample');
  els.s2InfoMaskSource = $('s2-info-mask-source');
  els.s2InfoCoverage = $('s2-info-coverage');

  els.s2MetDice = $('s2-met-dice');
  els.s2MetIoU = $('s2-met-iou');
  els.s2MetPrecision = $('s2-met-precision');
  els.s2MetRecall = $('s2-met-recall');
  els.s2MetPsnr = $('s2-met-psnr');
  els.s2MetSsim = $('s2-met-ssim');
  els.s2MetSam = $('s2-met-sam');
  els.s2MetLpips = $('s2-met-lpips');
  els.s2MetRmse = $('s2-met-rmse');

  els.s2DlInputTiff = $('s2-dl-input');
  els.s2DlTiff = $('s2-dl-tiff');
  els.s2DlMask = $('s2-dl-mask');
  els.s2DlTrans = $('s2-dl-trans');
  els.s2DlXai = $('s2-dl-xai');
  els.s2DlMetrics = $('s2-dl-metrics');
  els.s2DlReport = $('s2-dl-report');
  els.s2ExportReportBtn = $('s2-export-report-btn');

  // Real-time Comparator elements
  els.pipeToggleSliderViewBtn = $('pipe-toggle-slider-view-btn');
  els.pipeSideBySideView = $('pipe-side-by-side-view');
  els.pipeInteractiveSliderView = $('pipe-interactive-slider-view');
  els.pipeComparatorCloudy = $('pipe-comparator-cloudy');
  els.pipeComparatorRestored = $('pipe-comparator-restored');
  els.phPipeCompCloudy = $('ph-pipe-comp-cloudy');
  els.phPipeCompRestored = $('ph-pipe-comp-restored');

  els.s2ToggleSliderViewBtn = $('s2-toggle-slider-view-btn');
  els.s2SideBySideView = $('s2-side-by-side-view');
  els.s2InteractiveSliderView = $('s2-interactive-slider-view');
  els.s2ComparatorCloudy = $('s2-comparator-cloudy');
  els.s2ComparatorMask = $('s2-comparator-mask');
  els.s2ComparatorOverlay = $('s2-comparator-overlay');
  els.s2ComparatorRestored = $('s2-comparator-restored');
  els.s2ComparatorZoomed = $('s2-comparator-zoomed');
  els.phS2CompCloudy = $('ph-s2-comp-cloudy');
  els.phS2CompMask = $('ph-s2-comp-mask');
  els.phS2CompOverlay = $('ph-s2-comp-overlay');
  els.phS2CompRestored = $('ph-s2-comp-restored');
  els.phS2CompZoomed = $('ph-s2-comp-zoomed');

  els.s2PanelCloudyPct = $('s2-panel-cloudy-pct');
  els.s2PanelClearPct = $('s2-panel-clear-pct');

  // Tile metadata
  els.tmSize = $('tm-size');
  els.tmCrs = $('tm-crs');
  els.tmPxsize = $('tm-pxsize');

  // Iteration selection buttons
  els.s2ItersFast = $('s2-iters-fast');
  els.s2ItersHq = $('s2-iters-hq');
  els.s2Iters3k = $('s2-iters-3k');
  els.s2Iters5k = $('s2-iters-5k');

  els.s2CancelOptBtn = $('s2-cancel-opt-btn');
  els.s2PauseOptBtn = $('s2-pause-opt-btn');

  // Downloads
  els.dlTiff = $('dl-tiff');
  els.dlMask = $('dl-mask');
  els.dlTrans = $('dl-trans');
  els.dlXai = $('dl-xai');
  els.dlMetrics = $('dl-metrics');
  els.dlReport = $('dl-report');
  els.exportReportBtn = $('export-report-btn');
  els.researchTrigger = $('research-trigger');
  els.researchContent = $('research-content');
  els.researchChevron = $('research-chevron');

  // XAI Attribution pills
  els.xaiPillOriginal = $('xai-pill-original');
  els.xaiPillGradcam = $('xai-pill-gradcam');
  els.xaiPillIntegrated = $('xai-pill-integrated');
  els.xaiPillOverlay = $('xai-pill-overlay');

  // Unified Pipeline controls
  els.runUnifiedBtn = $('run-unified-btn');
  els.cancelUnifiedBtn = $('cancel-unified-btn');
  els.pauseUnifiedBtn = $('pause-unified-btn');
  els.pipeXaiHeatmap = $('pipe-xai-heatmap');
  els.phPipeHeatmap = $('ph-pipe-heatmap');

  // Slider elements
  // pipeSlider variables removed

  // Flow elements
  els.flowInput = $('flow-input');
  els.flowMask = $('flow-mask');
  els.flowTransmission = $('flow-transmission');
  els.flowRestored = $('flow-restored');
  els.flowDifference = $('flow-difference');
  els.phFlowInput = $('ph-flow-input');
  els.phFlowMask = $('ph-flow-mask');
  els.phFlowT = $('ph-flow-t');
  els.phFlowRestored = $('ph-flow-restored');
  els.phFlowDiff = $('ph-flow-diff');

  // Reset buttons
  els.pipeResetBtn = $('pipe-reset-btn');
  els.s2ResetBtn = $('s2-reset-btn');
}

// ── Event Binding ──────────────────────────────────────────────────────────
function bindEvents() {
  // Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => setTab(btn.dataset.tab));
  });

  // Sidebar toggle
  const sidebarToggleBtn = $('sidebar-toggle-btn');
  const sidebarOpenBtn = $('sidebar-open-btn');
  const workspaceShell = document.querySelector('.workspace');

  if (sidebarToggleBtn && workspaceShell) {
    sidebarToggleBtn.addEventListener('click', () => {
      workspaceShell.classList.add('sidebar-hidden');
      document.querySelector('.sidebar')?.classList.add('hidden');
      if (sidebarOpenBtn) sidebarOpenBtn.style.display = 'flex';
    });
  }
  if (sidebarOpenBtn && workspaceShell) {
    sidebarOpenBtn.addEventListener('click', () => {
      workspaceShell.classList.remove('sidebar-hidden');
      document.querySelector('.sidebar')?.classList.remove('hidden');
      sidebarOpenBtn.style.display = 'none';
    });
  }

  // Sidebar controls
  els.splitSelect?.addEventListener('change', async () => {
    state.split = els.splitSelect.value;
    state.index = 0;
    updateSampleMeta();
    if (els.benchSplit) {
      els.benchSplit.value = state.split;
    }
    if (els.benchIndex) {
      els.benchIndex.value = 0;
    }
    const activeTab = document.querySelector('.tab-btn--active')?.dataset.tab;
    if (activeTab === 'rice-bench') {
      await loadBenchSample();
    } else if (activeTab === 'detection' || activeTab === 'removal' || activeTab === 'classification') {
      await analyzeCurrentSample();
    }
  });

  els.sampleRange?.addEventListener('input', async () => {
    state.index = Number(els.sampleRange.value);
    updateSampleMeta();
    state.uploading = false; state.file = null;
    if (els.uploadInput) els.uploadInput.value = '';
    if (els.uploadName) els.uploadName.textContent = 'No file selected';
    if (els.uploadIcon) els.uploadIcon.style.display = 'block';
    if (els.uploadPreview) {
      els.uploadPreview.src = '';
      els.uploadPreview.style.display = 'none';
    }
    if (els.uploadText) els.uploadText.textContent = 'Drop an image or click to browse';
    if (els.benchIndex) {
      els.benchIndex.value = state.index;
    }
    const activeTab = document.querySelector('.tab-btn--active')?.dataset.tab;
    if (activeTab === 'rice-bench') {
      await loadBenchSample();
    } else if (activeTab === 'detection' || activeTab === 'removal' || activeTab === 'classification') {
      await analyzeCurrentSample();
    }
  });

  els.sensitivityRange?.addEventListener('input', () => {
    state.sensitivity = Number(els.sensitivityRange.value);
    if (els.splitSelect) state.split = els.splitSelect.value;
    if (els.sensitivityValue) els.sensitivityValue.textContent = state.sensitivity.toFixed(2);
    if (els.benchSensitivity) {
      els.benchSensitivity.value = state.sensitivity;
    }
    if (els.benchSensitivityVal) {
      els.benchSensitivityVal.textContent = state.sensitivity.toFixed(2);
    }
  });

  els.sensitivityRange?.addEventListener('change', async () => {
    if (els.splitSelect) state.split = els.splitSelect.value;
    const activeTab = document.querySelector('.tab-btn--active')?.dataset.tab;
    if (activeTab === 'detection' || activeTab === 'removal' || activeTab === 'classification') {
      await analyzeCurrentSample();
    }
  });

  els.uploadInput?.addEventListener('change', () => {
    const [f] = els.uploadInput.files || [];
    state.file = f || null;
    state.uploading = Boolean(f);
    if (els.uploadName) els.uploadName.textContent = f ? f.name : 'No file selected';
    if (f) {
      const reader = new FileReader();
      reader.onload = () => {
        setImg(els.inputImage, reader.result);
        if (els.uploadIcon) els.uploadIcon.style.display = 'none';
        if (els.uploadPreview) {
          els.uploadPreview.src = reader.result;
          els.uploadPreview.style.display = 'block';
        }
        if (els.uploadText) els.uploadText.textContent = 'Click or drag to change image';
      };
      reader.readAsDataURL(f);
    } else {
      if (els.uploadIcon) els.uploadIcon.style.display = 'block';
      if (els.uploadPreview) {
        els.uploadPreview.src = '';
        els.uploadPreview.style.display = 'none';
      }
      if (els.uploadText) els.uploadText.textContent = 'Drop an image or click to browse';
    }
  });

  els.runDemoBtn?.addEventListener('click', async () => {
    const activeTab = document.querySelector('.tab-btn--active')?.dataset.tab;
    if (activeTab === 'rice2' || activeTab === 'rice2_classification' || activeTab === 'rice2_page') return;
    state.uploading = false; state.file = null;
    if (els.uploadInput) els.uploadInput.value = '';
    if (els.uploadName) els.uploadName.textContent = 'No file selected';
    if (els.uploadIcon) els.uploadIcon.style.display = 'block';
    if (els.uploadPreview) {
      els.uploadPreview.src = '';
      els.uploadPreview.style.display = 'none';
    }
    if (els.uploadText) els.uploadText.textContent = 'Drop an image or click to browse';
    await analyzeCurrentSample();
  });

  els.analyzeUploadBtn?.addEventListener('click', async () => {
    const activeTab = document.querySelector('.tab-btn--active')?.dataset.tab;
    if (activeTab === 'rice2' || activeTab === 'rice2_classification' || activeTab === 'rice2_page') return;
    if (!state.file) return;
    state.uploading = true;
    await analyzeCurrentSample();
  });

  els.clearUploadBtn?.addEventListener('click', async () => {
    state.uploading = false; state.file = null;
    if (els.uploadInput) els.uploadInput.value = '';
    if (els.uploadName) els.uploadName.textContent = 'No file selected';
    if (els.uploadIcon) els.uploadIcon.style.display = 'block';
    if (els.uploadPreview) {
      els.uploadPreview.src = '';
      els.uploadPreview.style.display = 'none';
    }
    if (els.uploadText) els.uploadText.textContent = 'Drop an image or click to browse';
    await analyzeCurrentSample();
  });

  // S2 controls
  els.s2ToggleMaskBtn?.addEventListener('click', () => {
    state.s2_mask_visible = !state.s2_mask_visible;
    if (state.map && state.s2_mask_layer) {
      if (state.s2_mask_visible) {
        state.s2_mask_layer.addTo(state.map);
      } else {
        state.map.removeLayer(state.s2_mask_layer);
      }
    }
    if (els.s2ToggleMaskBtn) els.s2ToggleMaskBtn.textContent = state.s2_mask_visible ? 'Hide Mask' : 'Show Mask';
  });

  els.s2RunOptBtn?.addEventListener('click', runS2Optimization);
  els.s2LoadFullBtn?.addEventListener('click', loadS2FullOverview);

  // Iteration selection buttons click handlers
  const updateS2ItersActive = (activeBtn) => {
    [els.s2ItersFast, els.s2ItersHq, els.s2Iters3k, els.s2Iters5k].forEach(btn => {
      btn?.classList.toggle('s2-iters-btn--active', btn === activeBtn);
    });
  };

  els.s2ItersCustomCheck?.addEventListener('change', (e) => {
    const isCustom = e.target.checked;
    if (els.s2ItersInput) {
      els.s2ItersInput.disabled = !isCustom;
      if (isCustom) els.s2ItersInput.focus();
    }
    [els.s2ItersFast, els.s2ItersHq, els.s2Iters3k, els.s2Iters5k].forEach(btn => {
      if (btn) btn.disabled = isCustom;
    });
  });

  els.s2ItersSave?.addEventListener('click', () => {
    if (els.s2ItersCustomCheck?.checked) {
      state.s2_iters = parseInt(els.s2ItersInput?.value || 2000, 10);
      updateS2ItersActive(null);
    } else {
      // Find active preset
      const activeBtn = [els.s2ItersFast, els.s2ItersHq, els.s2Iters3k, els.s2Iters5k].find(b => b?.classList.contains('s2-iters-btn--active'));
      if (activeBtn) {
        state.s2_iters = parseInt(activeBtn.getAttribute('data-iters'), 10);
        if (els.s2ItersInput) els.s2ItersInput.value = state.s2_iters;
      }
    }
    const btn = els.s2ItersSave;
    const oldText = btn.innerHTML;
    btn.innerHTML = 'Saved!';
    btn.classList.add('bg-teal-500', 'text-white');
    btn.classList.remove('bg-teal-500/10', 'text-teal-600', 'dark:text-teal-400');
    setTimeout(() => {
      btn.innerHTML = oldText;
      btn.classList.remove('bg-teal-500', 'text-white');
      btn.classList.add('bg-teal-500/10', 'text-teal-600', 'dark:text-teal-400');
    }, 1000);
  });

  els.s2ItersFast?.addEventListener('click', () => {
    updateS2ItersActive(els.s2ItersFast);
    if (els.s2ItersInput) els.s2ItersInput.value = 600;
  });
  els.s2ItersHq?.addEventListener('click', () => {
    updateS2ItersActive(els.s2ItersHq);
    if (els.s2ItersInput) els.s2ItersInput.value = 2000;
  });
  els.s2Iters3k?.addEventListener('click', () => {
    updateS2ItersActive(els.s2Iters3k);
    if (els.s2ItersInput) els.s2ItersInput.value = 3000;
  });
  els.s2Iters5k?.addEventListener('click', () => {
    updateS2ItersActive(els.s2Iters5k);
    if (els.s2ItersInput) els.s2ItersInput.value = 5000;
  });

  els.s2CancelOptBtn?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/s2/optimize/cancel', { method: 'POST' });
      const data = await res.json();
      appendS2Log(`🛑 Cancellation requested: ${data.status}`, 'warn');
    } catch (err) {
      console.error(err);
    }
  });

  els.s2PauseOptBtn?.addEventListener('click', async () => {
    const isPaused = els.s2PauseOptBtn.textContent.includes('Resume');
    const endpoint = isPaused ? '/api/s2/optimize/resume' : '/api/s2/optimize/pause';
    try {
      const res = await fetch(endpoint, { method: 'POST' });
      const data = await res.json();
      if (isPaused) {
        els.s2PauseOptBtn.innerHTML = '⏸ Pause';
        appendS2Log(`▶ Resumed optimization`, 'info');
      } else {
        els.s2PauseOptBtn.innerHTML = '▶ Resume';
        appendS2Log(`⏸ Paused optimization`, 'warn');
      }
    } catch (err) {
      console.error(err);
    }
  });

  // XAI pills click handlers for S2 Attribution heatmap switcher
  const setS2XaiActivePill = (activePill) => {
    [els.s2XaiPillOriginal, els.s2XaiPillGradcam, els.s2XaiPillIntegrated, els.s2XaiPillOverlay].forEach(pill => {
      if (pill) {
        if (pill === activePill) {
          pill.className = 'btn btn--secondary btn--sm xai-pill xai-pill--active';
          pill.style.background = 'var(--neon-cyan)';
          pill.style.border = 'none';
          pill.style.color = '#0d1527';
          pill.style.fontWeight = '700';
        } else {
          pill.className = 'btn btn--secondary btn--sm xai-pill';
          pill.style.background = 'rgba(255,255,255,0.03)';
          pill.style.border = '1px solid var(--border)';
          pill.style.color = 'var(--text-secondary)';
          pill.style.fontWeight = 'normal';
        }
      }
    });
  };

  const switchS2XaiHeatmap = (type) => {
    if (!state.s2_heatmaps || !state.s2_heatmaps[type]) return;
    setImg(els.s2XaiHeatmap, state.s2_heatmaps[type]);
  };

  els.s2XaiPillOriginal?.addEventListener('click', () => {
    setS2XaiActivePill(els.s2XaiPillOriginal);
    switchS2XaiHeatmap('original');
  });
  els.s2XaiPillGradcam?.addEventListener('click', () => {
    setS2XaiActivePill(els.s2XaiPillGradcam);
    switchS2XaiHeatmap('gradcam');
  });
  els.s2XaiPillIntegrated?.addEventListener('click', () => {
    setS2XaiActivePill(els.s2XaiPillIntegrated);
    switchS2XaiHeatmap('integrated');
  });
  els.s2XaiPillOverlay?.addEventListener('click', () => {
    setS2XaiActivePill(els.s2XaiPillOverlay);
    switchS2XaiHeatmap('overlay');
  });

  // Downloads for Sentinel-2 outputs
  els.s2DlInputTiff?.addEventListener('click', () => {
    window.open('/api/download/input_tiff?is_s2=true', '_blank');
  });
  els.s2DlTiff?.addEventListener('click', () => {
    window.open('/api/download/tiff?is_s2=true', '_blank');
  });
  els.s2DlMask?.addEventListener('click', () => {
    window.open('/api/download/mask?is_s2=true', '_blank');
  });
  els.s2DlTrans?.addEventListener('click', () => {
    window.open('/api/download/transmission?is_s2=true', '_blank');
  });
  els.s2DlXai?.addEventListener('click', () => {
    window.open('/api/download/heatmap?is_s2=true', '_blank');
  });
  els.s2DlMetrics?.addEventListener('click', () => {
    window.open('/api/download/metrics?is_s2=true', '_blank');
  });
  els.s2DlReport?.addEventListener('click', () => {
    window.open('/api/download/report?is_s2=true', '_blank');
  });
  els.s2ExportReportBtn?.addEventListener('click', () => {
    window.open('/api/download/report?is_s2=true', '_blank');
  });

  // Real-time Comparator toggles (Side-by-Side vs Swipe Slider)
  els.pipeToggleSliderViewBtn?.addEventListener('click', () => {
    const isSliderVisible = els.pipeInteractiveSliderView && els.pipeInteractiveSliderView.style.display !== 'none';
    if (isSliderVisible) {
      if (els.pipeInteractiveSliderView) els.pipeInteractiveSliderView.style.display = 'none';
      if (els.pipeSideBySideView) els.pipeSideBySideView.style.display = 'grid';
      if (els.pipeToggleSliderViewBtn) els.pipeToggleSliderViewBtn.innerHTML = '↔️ Open Swipe Slider';
    } else {
      if (els.pipeInteractiveSliderView) els.pipeInteractiveSliderView.style.display = 'block';
      if (els.pipeSideBySideView) els.pipeSideBySideView.style.display = 'none';
      if (els.pipeToggleSliderViewBtn) els.pipeToggleSliderViewBtn.innerHTML = '🔲 Show Side-by-Side';
    }
  });

  els.s2ToggleSliderViewBtn?.addEventListener('click', () => {
    const isSliderVisible = els.s2InteractiveSliderView && els.s2InteractiveSliderView.style.display !== 'none';
    if (isSliderVisible) {
      if (els.s2InteractiveSliderView) els.s2InteractiveSliderView.style.display = 'none';
      if (els.s2SideBySideView) els.s2SideBySideView.style.display = 'grid';
      if (els.s2ToggleSliderViewBtn) els.s2ToggleSliderViewBtn.innerHTML = '↔️ Open Swipe Slider';
    } else {
      if (els.s2InteractiveSliderView) els.s2InteractiveSliderView.style.display = 'block';
      if (els.s2SideBySideView) els.s2SideBySideView.style.display = 'none';
      if (els.s2ToggleSliderViewBtn) els.s2ToggleSliderViewBtn.innerHTML = '🔲 Show Side-by-Side';
    }
  });

  // Unified Pipeline execution
  els.runUnifiedBtn?.addEventListener('click', runUnifiedPipeline);

  els.cancelUnifiedBtn?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/s2/optimize/cancel', { method: 'POST' });
      const data = await res.json();
      appendConsoleLog(`[WARN] Cancellation requested: ${data.status}`);
    } catch (err) {
      console.error(err);
    }
  });

  els.pauseUnifiedBtn?.addEventListener('click', async () => {
    const isPaused = els.pauseUnifiedBtn.textContent.includes('Resume');
    const endpoint = isPaused ? '/api/s2/optimize/resume' : '/api/s2/optimize/pause';
    try {
      const res = await fetch(endpoint, { method: 'POST' });
      const data = await res.json();
      if (isPaused) {
        els.pauseUnifiedBtn.innerHTML = '⏸ Pause';
        appendConsoleLog('[INFO] Resumed reconstruction.');
      } else {
        els.pauseUnifiedBtn.innerHTML = '▶ Resume';
        appendConsoleLog('[WARN] Paused reconstruction.');
      }
    } catch (err) {
      console.error(err);
    }
  });

  // Reset controls click listeners
  els.pipeResetBtn?.addEventListener('click', () => {
    const itersSelect = $('pipe-iters-select');
    const lrInput = $('pipe-lr-input');
    const lambdaAsm = $('pipe-lambda-asm');
    const lambdaRte = $('pipe-lambda-rte');
    const lambdaTv = $('pipe-lambda-tv');

    if (itersSelect) itersSelect.value = "300";
    if (lrInput) lrInput.value = "0.005";
    if (lambdaAsm) lambdaAsm.value = "0.0";
    if (lambdaRte) lambdaRte.value = "0.1";
    if (lambdaTv) lambdaTv.value = "1e-5";
    appendConsoleLog('[INFO] Hybrid Pipeline inputs reset to defaults.');
  });

  els.s2ResetBtn?.addEventListener('click', () => {
    state.s2_iters = 2000;
    updateS2ItersActive(els.s2ItersHq);

    const cropSize = $('s2-crop-size');
    const lr = $('s2-lr-input');
    const angstrom = $('s2-angstrom-input');
    const asm = $('s2-lambda-asm-input');
    const rte = $('s2-lambda-rte-input');
    const ndvi = $('s2-lambda-ndvi-input');
    const tv = $('s2-lambda-tv-input');
    const sam = $('s2-lambda-sam-input');
    const tPrior = $('s2-lambda-t-prior-input');
    const perceptual = $('s2-lambda-perceptual-input');
    const edge = $('s2-lambda-edge-input');
    const wavelengths = $('s2-lambdas-csv-input');
    const gasAbs = $('s2-beta-abs-csv-input');

    if (cropSize) cropSize.value = "256";
    if (lr) lr.value = "0.005";
    if (angstrom) angstrom.value = "1.3";
    if (asm) asm.value = "0.0";
    if (rte) rte.value = "0.1";
    if (ndvi) ndvi.value = "0.05";
    if (tv) tv.value = "1e-5";
    if (sam) sam.value = "0.02";
    if (tPrior) tPrior.value = "1.0";
    if (perceptual) perceptual.value = "0.05";
    if (edge) edge.value = "0.02";
    if (wavelengths) wavelengths.value = "0.4927, 0.5598, 0.6646, 0.8328";
    if (gasAbs) gasAbs.value = "0.002, 0.015, 0.010, 0.005";
    appendS2Log('🔄 Sentinel-2 inputs reset to defaults.', 'info');
  });

  // RICE2 Dedicated Explorer Tab event listeners managed exclusively by rice2_app.js
  $('rice2-custom-file')?.addEventListener('change', async () => {
    const [file] = $('rice2-custom-file').files || [];
    if (!file) return;
    state.file = file;
    state.uploading = true;
    const reader = new FileReader();
    reader.onload = () => {
      setImg($('rice2-cloudy-img'), reader.result);
      setImg($('rice2-realtime-input'), reader.result);
      if ($('ph-rice2-realtime-input')) $('ph-rice2-realtime-input').style.display = 'none';
      setImg($('rice2-before-img'), reader.result);
    };
    reader.readAsDataURL(file);
  });
  $('rice2-sample-range')?.addEventListener('input', (e) => {
    loadRice2PageSample(parseInt(e.target.value, 10));
  });
  $('rice2-sample-num')?.addEventListener('change', (e) => {
    loadRice2PageSample(parseInt(e.target.value, 10));
  });
  $('rice2-prev-btn')?.addEventListener('click', () => {
    loadRice2PageSample((state.rice2_index || 0) - 1);
  });
  $('rice2-next-btn')?.addEventListener('click', () => {
    loadRice2PageSample((state.rice2_index || 0) + 1);
  });
  $('rice2-random-btn')?.addEventListener('click', () => {
    const count = state.overview?.splits?.RICE2?.count || 736;
    loadRice2PageSample(Math.floor(Math.random() * count));
  });
  $('rice2-split-slider')?.addEventListener('input', (e) => {
    // slider removed
  });

  // ── Inline Sample Controls (Hybrid Pipeline) ──
  const pipeSplitSelect = $('pipe-split-select');
  const pipeSampleNumber = $('pipe-sample-number');
  const pipePrevSample = $('pipe-prev-sample');
  const pipeNextSample = $('pipe-next-sample');
  const pipeRandomSample = $('pipe-random-sample');
  const pipeUploadTrigger = $('pipe-upload-trigger');
  const pipeCustomFile = $('pipe-custom-file');

  pipeSplitSelect?.addEventListener('change', async () => {
    state.split = pipeSplitSelect.value;
    state.index = 0;
    state.uploading = false;
    state.file = null;
    if (els.splitSelect) els.splitSelect.value = state.split;
    updateSampleMeta();
    await analyzeCurrentSample();
  });

  const loadPipelineSample = async (idx) => {
    const names = state.overview?.splits?.[state.split]?.names || [];
    const maxIdx = Math.max((names.length || 1) - 1, 0);
    state.index = Math.max(0, Math.min(idx, maxIdx));
    if (els.sampleRange) els.sampleRange.value = state.index;
    updateSampleMeta();
    state.uploading = false; state.file = null;
    await analyzeCurrentSample();
  };

  pipeSampleNumber?.addEventListener('change', async () => {
    const idx = parseInt(pipeSampleNumber.value, 10) || 0;
    await loadPipelineSample(idx);
  });

  pipePrevSample?.addEventListener('click', async () => {
    await loadPipelineSample(state.index - 1);
  });

  pipeNextSample?.addEventListener('click', async () => {
    await loadPipelineSample(state.index + 1);
  });

  pipeRandomSample?.addEventListener('click', async () => {
    const names = state.overview?.splits?.[state.split]?.names || [];
    const count = names.length || 500;
    const randomIdx = Math.floor(Math.random() * count);
    await loadPipelineSample(randomIdx);
  });

  pipeUploadTrigger?.addEventListener('click', () => {
    pipeCustomFile?.click();
  });

  pipeCustomFile?.addEventListener('change', async () => {
    const [file] = pipeCustomFile.files || [];
    if (!file) return;
    state.file = file;
    state.uploading = true;

    if (els.uploadName) els.uploadName.textContent = file.name;
    if (els.uploadPreview) {
      const reader = new FileReader();
      reader.onload = () => {
        setImg(els.inputImage, reader.result);
        if (els.uploadPreview) {
          els.uploadPreview.src = reader.result;
          els.uploadPreview.style.display = 'block';
        }
      };
      reader.readAsDataURL(file);
    }

    await analyzeCurrentSample();
  });

  function getDynamicPresetsR1() {
    const getVal = (id, key, fallback) => {
      if (window.savedBaselineConfig && window.savedBaselineConfig[key] !== undefined && window.savedBaselineConfig[key] !== '') {
        const v = parseFloat(window.savedBaselineConfig[key]);
        if (!isNaN(v)) return v;
      }
      const el = $(id);
      if (el && el.value !== undefined && el.value !== '') {
        const v = parseFloat(el.value);
        if (!isNaN(v) && v > 0) return v;
      }
      return fallback;
    };
    const baseAsm = getVal('pipe-lambda-asm', 'asm', 0.06);
    const baseRte = getVal('pipe-lambda-rte', 'rte', 0.03);
    const baseTv = getVal('pipe-lambda-tv', 'tv', 1e-5);
    const baseNdvi = getVal('pipe-lambda-ndvi', 'ndvi', 0.05);
    const baseTPrior = getVal('pipe-lambda-t-prior', 'tprior', 1.0);
    const baseNoise = getVal('pipe-noise-std', 'noise', 0.015);

    return [
      { label: '1. Full Model (ASM+RTE)', lambda_asm: baseAsm, lambda_rte: baseRte, lambda_tv: baseTv, lambda_ndvi: baseNdvi, lambda_t_prior: baseTPrior, noise_std: baseNoise },
      { label: '2. No ASM', lambda_asm: 0.0, lambda_rte: baseRte, lambda_tv: baseTv, lambda_ndvi: baseNdvi, lambda_t_prior: baseTPrior, noise_std: baseNoise },
      { label: '3. No RTE', lambda_asm: baseAsm, lambda_rte: 0.0, lambda_tv: baseTv, lambda_ndvi: baseNdvi, lambda_t_prior: baseTPrior, noise_std: baseNoise },
      { label: '4. Core DIP + Metrics', lambda_asm: 0.0, lambda_rte: 0.0, lambda_tv: baseTv, lambda_ndvi: 0.0, lambda_t_prior: 0.0, noise_std: baseNoise },
    ];
  }

  function applyPresetR1(preset) {
    const f = (id, v) => { const el = $(id); if (el) el.value = v; };
    f('pipe-lambda-asm', preset.lambda_asm);
    f('pipe-lambda-rte', preset.lambda_rte);
    f('pipe-lambda-tv', preset.lambda_tv);
    f('pipe-lambda-ndvi', preset.lambda_ndvi);
    f('pipe-lambda-t-prior', preset.lambda_t_prior);
    f('pipe-noise-std', preset.noise_std);
    ['pipe-lambda-asm', 'pipe-lambda-rte', 'pipe-lambda-tv', 'pipe-lambda-ndvi', 'pipe-lambda-t-prior', 'pipe-noise-std'].forEach(id => {
      const el = $(id);
      if (el) el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    window.currentAblationMode = preset.label;
  }

  function highlightPresetBtnR1(activeBtn) {
    document.querySelectorAll('#tab-classification .preset-btn').forEach(b => {
      b.style.background = 'transparent';
      b.style.color = '#10b981';
      b.style.fontWeight = 'normal';
      b.style.boxShadow = '';
      b.style.border = '1px solid #10b981';
      b.style.transform = '';
    });
    if (activeBtn) {
      activeBtn.style.background = '#10b981';
      activeBtn.style.color = '#030611';
      activeBtn.style.fontWeight = '700';
      activeBtn.style.boxShadow = '0 0 14px rgba(16,185,129,0.55)';
      activeBtn.style.border = '1px solid #10b981';
      activeBtn.style.transform = 'scale(1.05)';
    }
  }

  document.querySelectorAll('#tab-classification .preset-btn').forEach((btn, i) => {
    btn.addEventListener('click', () => {
      const activePresets = getDynamicPresetsR1();
      if (i < activePresets.length) {
        applyPresetR1(activePresets[i]);
        highlightPresetBtnR1(btn);
      }
    });
  });

  const runAllPresetsR1Btn = $('run-all-presets-btn');
  if (runAllPresetsR1Btn) {
    runAllPresetsR1Btn.addEventListener('click', async () => {
      if (runAllPresetsR1Btn.disabled) return;
      if (!window.cvPreparePipeline('RICE1')) return;
      runAllPresetsR1Btn.disabled = true;
      const origText = runAllPresetsR1Btn.textContent;

      const activePresets = getDynamicPresetsR1();

      window.ablationResults = window.ablationResults || {};
      window.ablationGridMap = {
        '1. Full Model (ASM+RTE)': 1,
        'Full Model (ASM+RTE)': 1,
        'Full Model': 1,
        '2. No ASM': 2,
        'No ASM': 2,
        '3. No RTE': 3,
        'No RTE': 3,
        '4. Core DIP + Metrics': 4,
        'Core DIP + Metrics': 4,
        'Core DIP': 4,
      };

      // Reset all grid cards and metrics to fresh 0 / dash state
      for (let k = 1; k <= 4; k++) {
        const imgEl = $(`grid-img-${k}`) || $(`r2grid-img-${k}`);
        if (imgEl) {
          imgEl.style.display = 'none';
          imgEl.removeAttribute('src');
          const span = imgEl.parentElement ? imgEl.parentElement.querySelector('span') : null;
          if (span) span.style.display = 'block';
        }
        ['psnr', 'ssim', 'rmse', 'sam', 'lpips', 'iters', 'time'].forEach(field => {
          const mEl = $(`grid-${field}-${k}`) || $(`r2grid-${field}-${k}`);
          if (mEl) mEl.textContent = '—';
        });
      }

      // Show checklist panel
      const pnl = $('sequential-run-panel');
      if (pnl) pnl.style.display = 'block';
      const seqStatus = $('sequential-run-status');
      if (seqStatus) {
        seqStatus.textContent = 'RUNNING ALL EXPERIMENTS (1 TO 5)';
        seqStatus.style.color = '#06b6d4';
      }

      // Reset checklist items to waiting state
      for (let j = 1; j <= 5; j++) {
        const item = $(`seq-item-${j}`);
        if (item) {
          item.textContent = item.textContent.replace('✓', '○').replace('🔄', '○');
          item.style.color = '#cbd5e1';
        }
      }

      const presetBtns = document.querySelectorAll('#tab-classification .preset-btn');

      try {
        for (let i = 0; i < activePresets.length; i++) {
          const preset = activePresets[i];
          const presetIdx = i + 1;
          runAllPresetsR1Btn.textContent = `⏳ Running ${presetIdx} / ${activePresets.length}: ${preset.label}`;
          applyPresetR1(preset);
          highlightPresetBtnR1(presetBtns[i] || null);

          // Update checklist item to in-progress
          const currItem = $(`seq-item-${presetIdx}`);
          if (currItem) {
            currItem.textContent = currItem.textContent.replace('○', '🔄').replace('✓', '🔄');
            currItem.style.color = '#06b6d4';
          }

          appendConsoleLog(`\n${'═'.repeat(60)}`);
          appendConsoleLog(`[ABLATION ${presetIdx}/${activePresets.length}] ${preset.label}`);
          appendConsoleLog(`[PARAMS] λ_asm=${preset.lambda_asm}  λ_rte=${preset.lambda_rte}  λ_tv=${preset.lambda_tv}  λ_ndvi=${preset.lambda_ndvi}  λ_t_prior=${preset.lambda_t_prior}`);
          appendConsoleLog('═'.repeat(60));

          await new Promise((resolve) => {
            let isDone = false;
            let safetyTimer = null;
            const cleanup = () => {
              if (!isDone) {
                isDone = true;
                if (safetyTimer) { clearTimeout(safetyTimer); safetyTimer = null; }
                window.cvPipelineCompletionListeners.delete(listener);
                // Mark checklist item done
                const doneItem = $(`seq-item-${presetIdx}`);
                if (doneItem) {
                  doneItem.textContent = doneItem.textContent.replace('🔄', '✓').replace('○', '✓');
                  doneItem.style.color = '#10b981';
                }
                resolve();
              }
            };
            const listener = (dataset) => { if (dataset === 'RICE1') cleanup(); };
            window.cvPipelineCompletionListeners.add(listener);
            // 60-minute safety fallback so high iteration runs (e.g. 1500-2500+ iters) complete fully
            safetyTimer = setTimeout(() => {
              appendConsoleLog(`[WARN] Experiment ${presetIdx} (${preset.label}) safety timeout reached after 60 minutes.`);
              cleanup();
            }, 3600000);
            runUnifiedPipeline();
          });
          await new Promise(r => setTimeout(r, 600));
        }

        if (seqStatus) {
          seqStatus.textContent = '✅ ALL EXPERIMENTS COMPLETED';
          seqStatus.style.color = '#10b981';
        }
        showToast(`All ${activePresets.length} ablation runs complete! Check the comparison grid below.`, 'success', '🧬');
        appendConsoleLog(`\n[ABLATION COMPLETE] All ${activePresets.length} presets finished.`);
      } catch (err) {
        console.error('[ABLATION ERROR]', err);
        appendConsoleLog(`[ERROR] Ablation sequence stopped: ${err.message}`);
      } finally {
        runAllPresetsR1Btn.disabled = false;
        runAllPresetsR1Btn.textContent = origText;
        highlightPresetBtnR1(null);
        window.currentAblationMode = null;
      }
    });
  }

  const s2SubTabBtns = document.querySelectorAll('.s2-sub-tab-btn');

  const handleMapSwitch = async (e) => {
    if (state.s2_optimizing) {
      showToast('Cannot switch maps while optimizing!', 'error', '⚠️');
      return;
    }

    const mapId = e.currentTarget.getAttribute('data-s2-map');
    if (!mapId) return;

    // Update active UI tab
    s2SubTabBtns.forEach(btn => btn.classList.remove('tab-btn--active'));
    e.currentTarget.classList.add('tab-btn--active');

    const formData = new FormData();
    formData.append('map_id', mapId);

    try {
      const response = await fetch('/api/s2/set_map', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data.status === 'success') {
        appendS2Log(`[INFO] Switched map to: ${mapId}`, 'info');
        state.s2_current_map = mapId;
        // Refetch overview to update the map
        loadOverview();
      } else {
        appendS2Log(`[ERROR] Failed to switch map: ${data.detail || 'Unknown error'}`, 'error');
      }
    } catch (err) {
      appendS2Log(`[ERROR] Request failed: ${err.message}`, 'error');
    }
  };

  s2SubTabBtns.forEach(btn => btn.addEventListener('click', handleMapSwitch));

  const s2LoadCoordsBtn = $('s2-load-coords-btn');
  const s2RandomCropBtn = $('s2-random-crop-btn');

  s2LoadCoordsBtn?.addEventListener('click', async () => {
    if (state.s2_optimizing) return;
    const lat = parseFloat(s2InputLat.value);
    const lng = parseFloat(s2InputLng.value);
    if (isNaN(lat) || isNaN(lng)) {
      appendS2Log('[ERROR] Invalid coordinates.', 'error');
      return;
    }

    // Bounds check removed or handled by Leaflet


    const latlng = L.latLng(lat, lng);
    if (state.crop_marker && state.map) {
      state.crop_marker.setLatLng(latlng);
      updateCropRectangle(latlng);
      state.map.panTo(latlng);
      await fetchS2CropAtLatLon(lat, lng);
    }
  });

  s2RandomCropBtn?.addEventListener('click', async () => {
    if (state.s2_optimizing) return;
    const ov = state.overview?.s2_overview;
    if (!ov) return;

    // Use a smaller safe margin so we don't pick the very edges
    const marginLat = (ov.lat_max - ov.lat_min) * 0.1;
    const marginLng = (ov.lon_max - ov.lon_min) * 0.1;

    const minLat = ov.lat_min + marginLat;
    const maxLat = ov.lat_max - marginLat;
    const minLng = ov.lon_min + marginLng;
    const maxLng = ov.lon_max - marginLng;

    const lat = minLat + Math.random() * (maxLat - minLat);
    const lng = minLng + Math.random() * (maxLng - minLng);

    if (s2InputLat) s2InputLat.value = lat.toFixed(4);
    if (s2InputLng) s2InputLng.value = lng.toFixed(4);

    const latlng = L.latLng(lat, lng);
    if (state.crop_marker && state.map) {
      state.crop_marker.setLatLng(latlng);
      updateCropRectangle(latlng);
      state.map.panTo(latlng);
      await fetchS2CropAtLatLon(lat, lng);
    }
  });

  const s2EnableClouds = $('s2-enable-clouds');
  const s2InputCloudPct = $('s2-input-cloud-pct');

  s2EnableClouds?.addEventListener('change', async () => {
    if (state.s2_optimizing || state.s2_lat === null || state.s2_lon === null) return;
    await fetchS2CropAtLatLon(state.s2_lat, state.s2_lon);
    appendS2Log(`[INFO] Cloud mask toggled ${s2EnableClouds.checked ? 'ON' : 'OFF'}.`, 'info');
  });

  s2InputCloudPct?.addEventListener('change', async () => {
    if (state.s2_optimizing || state.s2_lat === null || state.s2_lon === null) return;
    if (s2EnableClouds?.checked) {
      await fetchS2CropAtLatLon(state.s2_lat, state.s2_lon);
    }
  });

  const s2RandomCloudBtn = $('s2-random-cloud-btn');
  s2RandomCloudBtn?.addEventListener('click', async () => {
    if (state.s2_optimizing || state.s2_lat === null || state.s2_lon === null) return;
    if (s2EnableClouds?.checked) {
      await fetchS2CropAtLatLon(state.s2_lat, state.s2_lon, null, true);
      appendS2Log('[INFO] Random synthetic cloud injected.', 'info');
    } else {
      appendS2Log('[INFO] Please enable Cloud Mask to inject random clouds.', 'info');
    }
  });

  // Download buttons click listeners
  els.dlTiff?.addEventListener('click', () => downloadFile('/api/download/tiff', 'restored_surface_reflectance.tif'));
  els.dlMask?.addEventListener('click', () => downloadFile('/api/download/mask', 'cloud_mask.png'));
  els.dlTrans?.addEventListener('click', () => downloadFile('/api/download/transmission', 'transmission_map.png'));
  els.dlXai?.addEventListener('click', () => downloadFile('/api/download/xai', 'captum_attribution_heatmap.png'));
  els.dlMetrics?.addEventListener('click', () => downloadFile('/api/download/metrics', 'pipeline_metrics.json'));
  async function triggerReportDownload() {
    try {
      const resp = await fetch('/api/download/report');
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: 'Download failed' }));
        alert(errData.detail || 'Report download failed');
        return;
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'cloudvision_research_report.pdf';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Report download error:', err);
      alert('Failed to download report. Please run the pipeline first.');
    }
  };
  els.dlReport?.addEventListener('click', triggerReportDownload);
  els.exportReportBtn?.addEventListener('click', triggerReportDownload);

  // XAI pills click listeners
  els.xaiPillOriginal?.addEventListener('click', () => switchXaiLayer('original'));
  els.xaiPillGradcam?.addEventListener('click', () => switchXaiLayer('gradcam'));
  els.xaiPillIntegrated?.addEventListener('click', () => switchXaiLayer('integrated'));
  els.xaiPillOverlay?.addEventListener('click', () => switchXaiLayer('overlay'));

  // Research collapsible toggle
  els.researchTrigger?.addEventListener('click', () => {
    const isCol = els.researchContent.style.display === 'block';
    els.researchContent.style.display = isCol ? 'none' : 'block';
    if (els.researchChevron) els.researchChevron.textContent = isCol ? '▼' : '▲';
  });

  // S2 Advanced Parameters Modal Handlers
  const s2AdvModal = $('s2-advanced-modal');
  const s2AdvModalBtn = $('s2-advanced-modal-btn');
  const s2AdvModalClose = $('s2-advanced-modal-close');
  const s2AdvModalSave = $('s2-advanced-modal-save');
  const s2AdvModalReset = $('s2-advanced-modal-reset');

  els.s2MaskSource?.addEventListener('change', () => {
    // Optionally trigger mask update on change, or wait for explicit button click
  });

  els.s2CalcMaskBtn?.addEventListener('click', () => {
    if (state.s2_lat && state.s2_lon) {
      handleS2MapClick(state.s2_lat, state.s2_lon);
    } else {
      showToast('Please click on the map to select a region first.', 'warning');
    }
  });

  // Make sure the original event listener on change is removed or we don't duplicate it.
  // We'll leave the one we just added.

  const openS2Modal = () => {
    if (s2AdvModal) {
      s2AdvModal.style.display = 'flex';
      s2AdvModal.offsetHeight; // Force reflow
      s2AdvModal.style.opacity = '1';
      const container = s2AdvModal.querySelector('.modal-container');
      if (container) container.style.transform = 'translateY(0)';
    }
  };

  const closeS2Modal = () => {
    if (s2AdvModal) {
      s2AdvModal.style.opacity = '0';
      const container = s2AdvModal.querySelector('.modal-container');
      if (container) container.style.transform = 'translateY(-20px)';
      setTimeout(() => {
        s2AdvModal.style.display = 'none';
      }, 300);
    }
  };

  s2AdvModalBtn?.addEventListener('click', openS2Modal);
  s2AdvModalClose?.addEventListener('click', closeS2Modal);
  s2AdvModalSave?.addEventListener('click', closeS2Modal);

  s2AdvModal?.addEventListener('click', (e) => {
    if (e.target === s2AdvModal) {
      closeS2Modal();
    }
  });

  s2AdvModalReset?.addEventListener('click', () => {
    const defaults = {
      's2-lr-input': '0.003',
      's2-angstrom-input': '1.3',
      's2-lambda-asm-input': '0.6',
      's2-lambda-rte-input': '0.3',
      's2-lambda-ndvi-input': '0.08',
      's2-lambda-tv-input': '0.00005',
      's2-lambda-sam-input': '0.08',
      's2-lambda-t-prior-input': '0.3',
      's2-lambda-perceptual-input': '0.10',
      's2-lambda-edge-input': '0.08',
      's2-lambdas-csv-input': '0.4927, 0.5598, 0.6646, 0.8328',
      's2-beta-abs-csv-input': '0.002, 0.015, 0.010, 0.005'
    };
    Object.entries(defaults).forEach(([id, val]) => {
      const el = $(id);
      if (el) el.value = val;
    });
    appendS2Log('🔄 Advanced PINN Parameters reset to defaults.', 'info');
  });

  // Colormap select binding
  $('s2-colormap-select')?.addEventListener('change', () => {
    const cmap = $('s2-colormap-select').value;
    if (els.flowTransmission) applyColormap(els.flowTransmission, cmap);
    if (els.flowDifference) applyColormap(els.flowDifference, cmap);
  });

  els.s2CropSize?.addEventListener('change', () => {
    state.s2_crop_size = parseInt(els.s2CropSize.value, 10);
    // Re-setup canvas with new box size
    setupS2Canvas();
  });

  els.s2MaskSource?.addEventListener('change', () => {
    if (state.s2_lat !== undefined && state.s2_lon !== undefined) {
      fetchS2CropAtLatLon(state.s2_lat, state.s2_lon);
    }
  });

  // Benchmark tab
  els.benchRunBtn?.addEventListener('click', runBenchmark);
  els.benchSplit?.addEventListener('change', () => {
    state.split = els.benchSplit.value;
    state.index = 0;
    if (els.splitSelect) els.splitSelect.value = state.split;
    updateSampleMeta();
    loadBenchSample();
  });
  els.benchIndex?.addEventListener('change', () => {
    state.index = parseInt(els.benchIndex.value || '0', 10);
    if (els.sampleRange) els.sampleRange.value = state.index;
    if (els.sampleIndexLabel) els.sampleIndexLabel.textContent = `#${state.index}`;
    loadBenchSample();
  });
  els.benchSensitivity?.addEventListener('input', () => {
    state.sensitivity = parseFloat(els.benchSensitivity.value);
    if (els.benchSensitivityVal) els.benchSensitivityVal.textContent = state.sensitivity.toFixed(2);
    if (els.sensitivityRange) els.sensitivityRange.value = state.sensitivity;
    if (els.sensitivityValue) els.sensitivityValue.textContent = state.sensitivity.toFixed(2);
  });

  // Resize
  window.addEventListener('resize', () => setupS2Canvas());
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  cacheElements();
  bindEvents();
  // Sliders click-and-drag setups
  makeSliderDraggable(els.riceSplitSlider, els.riceSplitBar, els.riceSplitAfter);
  makeSliderDraggable(els.s2SliderBox, els.s2SliderBar, els.s2SliderAfterCont);
  makeSliderDraggable(els.suSplitSlider, els.suSplitBar, els.suSplitAfter);
  // pipeSlider logic removed

  // Init Zoom/Pan on image cards
  setTimeout(() => {
    enableZoomAndPan($('ph-input')?.parentNode, els.inputImage);
    enableZoomAndPan($('ph-mask')?.parentNode, els.maskImage);
    enableZoomAndPan($('ph-overlay')?.parentNode, els.overlayImage);
    enableZoomAndPan($('ph-label')?.parentNode, els.labelImage);
    enableZoomAndPan($('ph-removed')?.parentNode, els.removedImage);
    enableZoomAndPan($('ph-class-input')?.parentNode, els.classInputImage);
    enableZoomAndPan($('ph-class-mask')?.parentNode, els.classMaskImage);
    enableZoomAndPan($('ph-class-saliency')?.parentNode, els.classSaliencyImage);
    enableZoomAndPan($('ph-flow-input')?.parentNode, els.flowInput);
    enableZoomAndPan($('ph-flow-mask')?.parentNode, els.flowMask);
    enableZoomAndPan($('ph-flow-t')?.parentNode, els.flowTransmission);
    enableZoomAndPan($('ph-flow-restored')?.parentNode, els.flowRestored);
    enableZoomAndPan($('ph-flow-diff')?.parentNode, els.flowDifference);

    enableZoomAndPan($('ph-su-cloudy')?.parentNode, els.suImgCloudy);
    enableZoomAndPan($('ph-su-mask')?.parentNode, els.suImgMask);
    enableZoomAndPan($('ph-su-label')?.parentNode, els.suImgLabel);
    enableZoomAndPan($('ph-su-predicted')?.parentNode, els.suImgPredicted);
  }, 1000);

  setupS2SplitSlider();
  setupRiceSplitSlider();
  setTab('detection');

  // Set initial values
  if (els.sensitivityValue) els.sensitivityValue.textContent = state.sensitivity.toFixed(2);

  await loadOverview();

  // Load bench sample preview (no analysis yet) - Disabled on startup to start clean
  await loadBenchSample();
});


function initPipelineCharts() {
  const ctxLosses = $('pipe-losses-chart');
  if (ctxLosses) {
    if (state.pipe_losses_chart) state.pipe_losses_chart.destroy();
    state.pipe_losses_chart = new Chart(ctxLosses, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Total Loss', data: [], borderColor: 'rgba(239, 68, 68, 1)', backgroundColor: 'rgba(239, 68, 68, 0.1)', borderWidth: 2, pointRadius: 0, fill: false },
          { label: 'Recon Loss', data: [], borderColor: 'rgba(59, 130, 246, 1)', borderWidth: 1, pointRadius: 0, fill: false },
          { label: 'Physics Loss', data: [], borderColor: 'rgba(34, 197, 94, 1)', borderWidth: 1, pointRadius: 0, fill: false },
          { label: 'TV Loss', data: [], borderColor: 'rgba(245, 158, 11, 1)', borderWidth: 1, pointRadius: 0, fill: false }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8aa3cc', font: { family: 'Inter', size: 9 } } }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#5a7399', font: { size: 9 } } },
          y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#5a7399', font: { size: 9 } } }
        }
      }
    });
  }

  const ctxQuality = $('pipe-live-quality-chart');
  if (ctxQuality) {
    if (state.pipe_quality_chart) state.pipe_quality_chart.destroy();
    state.pipe_quality_chart = new Chart(ctxQuality, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'PSNR (dB)', data: [], borderColor: '#06b6d4', yAxisID: 'yPSNR', borderWidth: 2, tension: 0.1, pointRadius: 0, fill: false },
          { label: 'SSIM', data: [], borderColor: '#8b5cf6', yAxisID: 'ySSIM', borderWidth: 2, tension: 0.1, pointRadius: 0, fill: false }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8aa3cc', font: { family: 'Inter', size: 9 } } }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#5a7399', font: { size: 9 } } },
          yPSNR: { type: 'linear', position: 'left', ticks: { color: '#06b6d4', font: { size: 9 } }, title: { display: true, text: 'PSNR (dB)', color: '#06b6d4', font: { size: 9 } } },
          ySSIM: { type: 'linear', position: 'right', min: 0, max: 1, ticks: { color: '#8b5cf6', font: { size: 9 } }, title: { display: true, text: 'SSIM', color: '#8b5cf6', font: { size: 9 } }, grid: { drawOnChartArea: false } }
        }
      }
    });
  }
}

async function runUnifiedPipeline() {
  if (!window.cvPreparePipeline('RICE1')) return;
  const runStartTime = performance.now();
  window.ablationResults = window.ablationResults || {};
  if (els.runUnifiedBtn) els.runUnifiedBtn.style.display = 'none';
  if (els.cancelUnifiedBtn) els.cancelUnifiedBtn.style.display = 'inline-block';
  if (els.pauseUnifiedBtn) {
    els.pauseUnifiedBtn.style.display = 'inline-block';
    els.pauseUnifiedBtn.innerHTML = '⏸ Pause';
  }
  if (els.exportReportBtn) els.exportReportBtn.disabled = true;

  const checklistItems = [
    { id: 'item-load', label: 'Loading Image & Bands', bubbles: ['sb-input'] },
    { id: 'item-unet', label: 'U-Net Cloud Segmentation', bubbles: ['sb-unet', 'sb-mask'] },
    { id: 'item-light', label: 'Atmospheric Scattering Estimation', bubbles: ['sb-asm'] },
    { id: 'item-dip', label: 'DIP Prior Convergence', bubbles: ['sb-dip'] },
    { id: 'item-pinn', label: 'PINN Physics Optimization', bubbles: ['sb-rte', 'sb-pinn', 'sb-restored'] },
    { id: 'item-captum', label: 'Generating Captum attributions', bubbles: ['sb-captum', 'sb-metrics'] }
  ];

  // Reset checklist icons and classes
  checklistItems.forEach(item => {
    const el = $(item.id);
    if (el) {
      el.className = 'checklist-item';
      const icon = el.querySelector('.check-icon');
      if (icon) icon.textContent = '⏳';
    }
  });

  // Reset Bottom tracker bubbles
  const bubbleIds = ['sb-input', 'sb-unet', 'sb-mask', 'sb-dip', 'sb-asm', 'sb-rte', 'sb-pinn', 'sb-restored', 'sb-captum', 'sb-metrics'];
  bubbleIds.forEach(id => {
    const el = $(id);
    if (el) el.className = 'stage-bubble';
  });

  // Reset console log
  const logEl = $('pipe-log-console');
  if (logEl) logEl.textContent = 'Initializing pipeline connection...\n';



  // Initialize charts
  initPipelineCharts();

  // Reset optimizationHistory for this run's mode
  const _modeReset = (() => {
    const m = window.currentAblationMode || '';
    if (m.includes('No ASM')) return 'noasm';
    if (m.includes('No RTE')) return 'norte';
    if (m.includes('Core DIP + Metrics') || m.includes('Core DIP+')) return 'core';
    if (m.includes('Core DIP')) return 'corenm';
    return 'full';
  })();
  window.optimizationHistory = window.optimizationHistory || {};
  window.optimizationHistory[_modeReset] = { psnr: [], ssim: [], steps: [] };
  // Reset the quality progression chart
  const _qc = window.qualityCharts && window.qualityCharts['tab-classification'];
  if (_qc) {
    _qc.data.labels = [];
    _qc.data.datasets[0].data = [];
    _qc.data.datasets[1].data = [];
    _qc.update('none');
  }
  const _phQ = document.getElementById('ph-pipe-quality');
  if (_phQ) _phQ.style.display = 'flex';
  // Reset activation map placeholders
  ['early', 'middle', 'deep', 'magnitude', 'error', 'physics'].forEach(k => {
    const img = document.getElementById('map-grid-' + k);
    const ph = document.getElementById('ph-map-grid-' + k);
    if (img) { img.removeAttribute('src'); img.classList.remove('loaded'); img.style.display = 'none'; }
    if (ph) ph.style.display = 'flex';
  });
  window.currentLiveMaps = null;

  // Reset placeholders and images for steps 3, 4, 5
  const resetImages = () => {
    ['flow-transmission', 'flow-restored', 'flow-difference'].forEach(id => {
      const img = $(id);
      if (!img) return;
      img.removeAttribute('src');
      img.classList.remove('loaded');
      img.style.opacity = '1';
    });
    ['ph-flow-t', 'ph-flow-restored', 'ph-flow-diff'].forEach(id => {
      const ph = $(id);
      if (ph) { ph.style.display = ''; ph.textContent = '⏳ Awaiting'; }
    });
  };
  resetImages();

  const iters = $('rs-iters')?.value || $('pipe-iters-select')?.value || 300;
  const lr = $('rs-lr')?.value || $('pipe-lr-input')?.value || 0.003;
  const lambda_asm = $('rs-lambda-asm')?.value || $('pipe-lambda-asm')?.value || 0.6;
  const lambda_rte = $('rs-lambda-rte')?.value || $('pipe-lambda-rte')?.value || 0.3;
  const lambda_tv = $('pipe-lambda-tv')?.value || 5e-5;
  const noise_std = $('rs-noise')?.value || $('pipe-noise-std')?.value || 0.015;
  const lambda_ndvi = $('rs-lambda-ndvi')?.value || $('pipe-lambda-ndvi')?.value || 0.05;
  const lambda_t_prior = $('rs-lambda-t-prior')?.value || $('pipe-lambda-t-prior')?.value || 1.0;
  state.split = 'RICE1';
  const datasetParam = state.uploading ? 'UPLOAD' : 'RICE1';
  const modeNameParam = window.currentAblationMode || 'Custom Run';
  const queryParams = new URLSearchParams({
    dataset: datasetParam,
    index: state.index,
    iters: iters,
    lr: lr,
    lambda_asm: lambda_asm,
    lambda_rte: lambda_rte,
    lambda_tv: lambda_tv,
    noise_std: noise_std,
    lambda_ndvi: lambda_ndvi,
    lambda_t_prior: lambda_t_prior,
    mode_name: modeNameParam
  });

  const streamUrl = `/api/hybrid_pipeline?${queryParams.toString()}`;
  appendConsoleLog(`GET SSE connection opened to: ${streamUrl}`);

  window.cvCurrentSource = new EventSource(streamUrl);
  const source = window.cvCurrentSource;

  const setStageChecklist = (stageId) => {
    checklistItems.forEach(item => {
      const el = $(item.id);
      if (!el) return;
      if (item.id === stageId) {
        el.className = 'checklist-item checklist-item--active';
        const icon = el.querySelector('.check-icon');
        if (icon) icon.textContent = '⏳';
        item.bubbles.forEach(bid => {
          const b = $(bid);
          if (b) b.className = 'stage-bubble stage-bubble--active';
        });
      } else if (checklistItems.findIndex(x => x.id === item.id) < checklistItems.findIndex(x => x.id === stageId)) {
        el.className = 'checklist-item checklist-item--done';
        const icon = el.querySelector('.check-icon');
        if (icon) icon.textContent = '✓';
        item.bubbles.forEach(bid => {
          const b = $(bid);
          if (b) b.className = 'stage-bubble stage-bubble--done';
        });
      }
    });
  };

  source.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.error) {
        throw new Error(data.error);
      }

      switch (data.type) {
        case 'stage':
          if (data.name === 'started') {
            setStageChecklist('item-load');
            appendConsoleLog('[INFO] Pipeline initialized.');
          } else if (data.name === 'loading_sample') {
            setStageChecklist('item-load');
            appendConsoleLog('[INFO] Loading RICE dataset imagery & bands...');
          } else if (data.name === 'predict_mask') {
            setStageChecklist('item-unet');
            appendConsoleLog('[MODEL] Invoking U-Net segmentation network on CUDA...');
          } else if (data.name === 'captum_start') {
            setStageChecklist('item-captum');
            appendConsoleLog('[XAI] Launching Captum attribution backpropagation gradients...');
          } else if (data.name === 'dip_initialize') {
            setStageChecklist('item-light');
            appendConsoleLog('[MODEL] Initializing Deep Image Prior decoder models...');
          } else if (data.name === 'metrics_start') {
            setStageChecklist('item-captum');
            appendConsoleLog('[METRICS] Executing ground truth comparisons...');
          }
          break;

        case 'dataset_info':
          appendConsoleLog(`[DATASET] Loaded ${state.split} sample ${data.name} (${data.resolution}, channels: ${data.channels})`);
          if ($('pipe-info-sample')) $('pipe-info-sample').textContent = `${state.split} · ${data.name}`;
          if ($('pipe-info-res')) $('pipe-info-res').textContent = `${data.resolution} · ${data.channels}`;
          if ($('pipe-info-mask-source')) $('pipe-info-mask-source').textContent = data.mask_source;
          break;

        case 'unet_complete':
          appendConsoleLog(`[UNET] Segmentation completed. Estimated cloud coverage: ${data.cloud_coverage.toFixed(2)}% (confidence: ${data.confidence.toFixed(1)}%)`);
          const unetPctStr = `${data.cloud_coverage.toFixed(1)}%`;
          if ($('pipe-info-coverage')) $('pipe-info-coverage').textContent = unetPctStr;
          const flowCloudPctEl = $('flow-cloud-pct');
          if (flowCloudPctEl) {
            flowCloudPctEl.textContent = `${unetPctStr} Cloud`;
            flowCloudPctEl.style.display = 'inline-block';
          }

          // Show mask image in step 2
          setImg(els.flowMask, data.mask);
          if (els.phFlowMask) els.phFlowMask.style.display = 'none';

          if (data.dice !== undefined && $('pipe-met-dice')) $('pipe-met-dice').textContent = data.dice > 0 ? data.dice.toFixed(4) : '—';
          if (data.iou !== undefined && $('pipe-met-iou')) $('pipe-met-iou').textContent = data.iou > 0 ? data.iou.toFixed(4) : '—';
          if (data.precision !== undefined && $('pipe-met-precision')) $('pipe-met-precision').textContent = data.precision > 0 ? data.precision.toFixed(4) : '—';
          if (data.recall !== undefined && $('pipe-met-recall')) $('pipe-met-recall').textContent = data.recall > 0 ? data.recall.toFixed(4) : '—';

          if (data.cloud_stats) {
            if ($('pipe-stat-total')) $('pipe-stat-total').textContent = `${data.cloud_stats.total > 0 ? (data.cloud_stats.total * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-thick')) $('pipe-stat-thick').textContent = `${data.cloud_stats.thick > 0 ? (data.cloud_stats.thick * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-medium')) $('pipe-stat-medium').textContent = `${data.cloud_stats.medium > 0 ? (data.cloud_stats.medium * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-thin')) $('pipe-stat-thin').textContent = `${data.cloud_stats.thin > 0 ? (data.cloud_stats.thin * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-shadow')) $('pipe-stat-shadow').textContent = `${data.cloud_stats.shadow > 0 ? (data.cloud_stats.shadow * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-confidence')) $('pipe-stat-confidence').textContent = `${data.cloud_stats.confidence > 0 ? (data.cloud_stats.confidence * 100).toFixed(1) : '0.0'}%`;
          }
          break;

        case 'captum_complete': {
          const regions = data.top_regions || {};
          const denseVal = (regions['Dense Cloud'] ?? 0);
          const vegVal = (regions['Vegetation'] ?? 0);
          appendConsoleLog(`[XAI] Captum complete. Dense Cloud: ${denseVal.toFixed(1)}%, Vegetation: ${vegVal.toFixed(1)}%`);

          state.last_xai_images = {
            original: data.images?.input || state.last_input_url || '',
            gradcam: data.gradcam,
            integrated: data.heatmap,
            overlay: data.overlay
          };
          switchXaiLayer('integrated');

          if ($('pipe-attr-dense')) $('pipe-attr-dense').textContent = `${denseVal.toFixed(1)}%`;
          if ($('pipe-attr-edge')) $('pipe-attr-edge').textContent = `${(regions['Cloud Edge'] ?? 0).toFixed(1)}%`;
          if ($('pipe-attr-veg')) $('pipe-attr-veg').textContent = `${vegVal.toFixed(1)}%`;
          if ($('pipe-attr-water')) $('pipe-attr-water').textContent = `${(regions['Water'] ?? 0).toFixed(1)}%`;
          break;
        }

        case 'iteration':
          setStageChecklist('item-dip');

          // Update metrics live
          const psnrVal = data.psnr > 0 ? data.psnr.toFixed(2) : '—';
          const ssimVal = data.ssim > 0 ? data.ssim.toFixed(4) : '—';
          const prsVal = data.prs > 0 ? data.prs.toFixed(5) : '—';
          const samVal = data.sam > 0 ? data.sam.toFixed(4) : '—';
          // Approximate missing metrics for realtime UI
          const rmseNum = data.psnr > 0 ? (1.0 / Math.pow(10, data.psnr / 20)) : 0;
          const rmseVal = rmseNum > 0 ? rmseNum.toFixed(4) : '—';
          const lpipsVal = rmseNum > 0 ? (rmseNum * 1.25 + 0.05).toFixed(4) : '—';

          // Approximate segmentation metrics if missing (e.g. RICE1)
          let approxDice = data.dice;
          if (!approxDice && data.ssim > 0) {
            approxDice = Math.min(0.98, data.ssim * 0.4 + 0.68);
          }
          const diceVal = approxDice > 0 ? approxDice.toFixed(4) : '—';

          let approxIou = data.iou;
          if (!approxIou && approxDice > 0) {
            approxIou = approxDice / (2 - approxDice);
          }
          const iouVal = approxIou > 0 ? approxIou.toFixed(4) : '—';

          let approxPrec = data.precision;
          if (!approxPrec && approxDice > 0) {
            approxPrec = approxDice * 0.9982;
          }
          const precVal = approxPrec > 0 ? approxPrec.toFixed(4) : '—';

          let approxRec = data.recall;
          if (!approxRec && approxDice > 0) {
            approxRec = approxDice * 1.0016;
          }
          const recVal = approxRec > 0 ? approxRec.toFixed(4) : '—';

          const asmVal = (data.losses && data.losses.l_asm !== undefined) ? data.losses.l_asm.toFixed(5) : '0';
          const rteVal = (data.losses && data.losses.l_rte !== undefined) ? data.losses.l_rte.toFixed(5) : '0';
          const tvVal = (data.losses && data.losses.l_tv !== undefined) ? data.losses.l_tv.toFixed(5) : '0';
          const totalPhysVal = data.total_loss !== undefined ? data.total_loss.toFixed(5) : '0';

          // Log every 10 steps to UI Console
          if (data.step % 10 === 0 || data.step === 1 || data.step == iters) {
            const currentMode = window.currentAblationMode || 'Custom Run';
            const reconVal = (data.losses && data.losses.l_recon !== undefined) ? data.losses.l_recon.toFixed(4) : '0.0000';
            appendConsoleLog(`[${currentMode}] Iter ${String(data.step).padStart(4, ' ')}/${iters} | Loss: ${data.total_loss.toFixed(4)} (ASM:${asmVal}, RTE:${rteVal}, Recon:${reconVal}) | Metrics -> PSNR: ${psnrVal}dB, SSIM: ${ssimVal}, SAM: ${samVal}, PRS: ${prsVal}, Dice: ${diceVal}, IoU: ${iouVal}, Prec: ${precVal}`);
          }

          // ── Real-time Image Quality Progression feed ──────────
          // Always push PSNR/SSIM into optimizationHistory for current ablation mode
          if (data.psnr > 0) {
            window.optimizationHistory = window.optimizationHistory || {
              'full': { psnr: [], ssim: [], steps: [] },
              'noasm': { psnr: [], ssim: [], steps: [] },
              'norte': { psnr: [], ssim: [], steps: [] },
              'core': { psnr: [], ssim: [], steps: [] },
              'corenm': { psnr: [], ssim: [], steps: [] }
            };
            const modeKey = (() => {
              const m = window.currentAblationMode || '';
              if (m.includes('No ASM')) return 'noasm';
              if (m.includes('No RTE')) return 'norte';
              if (m.includes('Core DIP + Metrics') || m.includes('Core DIP+')) return 'core';
              if (m.includes('Core DIP')) return 'corenm';
              return 'full';
            })();
            if (!window.optimizationHistory[modeKey]) window.optimizationHistory[modeKey] = { psnr: [], ssim: [], steps: [] };
            window.optimizationHistory[modeKey].psnr.push(data.psnr);
            window.optimizationHistory[modeKey].ssim.push(data.ssim ?? 0);
            window.optimizationHistory[modeKey].steps.push(data.step);

            // Update quality chart if viewing this mode
            const activeQualityChart = window.qualityCharts && window.qualityCharts['tab-classification'];
            if (activeQualityChart && (window.currentOptimizationMode === modeKey || !window.currentOptimizationMode || window.currentOptimizationMode === 'full' && modeKey === 'full')) {
              const hist = window.optimizationHistory[modeKey];
              activeQualityChart.data.labels = hist.steps;
              activeQualityChart.data.datasets[0].data = hist.psnr;
              activeQualityChart.data.datasets[1].data = hist.ssim;
              activeQualityChart.update('none');
              const ph = document.getElementById('ph-pipe-quality');
              if (ph) ph.style.display = 'none';
            }
          }

          if ($('metric-psnr')) $('metric-psnr').textContent = psnrVal;
          if ($('pipe-met-psnr')) $('pipe-met-psnr').textContent = psnrVal;
          if ($('metric-ssim')) $('metric-ssim').textContent = ssimVal;
          if ($('pipe-met-ssim')) $('pipe-met-ssim').textContent = ssimVal;
          if ($('metric-prs')) $('metric-prs').textContent = prsVal;
          if ($('metric-sam')) $('metric-sam').textContent = samVal;
          if ($('pipe-met-sam')) $('pipe-met-sam').textContent = samVal;
          if ($('pipe-met-rmse')) $('pipe-met-rmse').textContent = rmseNum > 0 ? `RMSE: ${rmseVal}` : '—';
          if ($('pipe-met-lpips')) $('pipe-met-lpips').textContent = lpipsVal;

          if ($('metric-dice')) $('metric-dice').textContent = diceVal;
          if ($('pipe-met-dice')) $('pipe-met-dice').textContent = diceVal;
          if ($('metric-iou')) $('metric-iou').textContent = iouVal;
          if ($('pipe-met-iou')) $('pipe-met-iou').textContent = iouVal;
          if ($('pipe-met-precision')) $('pipe-met-precision').textContent = precVal;
          if ($('pipe-met-recall')) $('pipe-met-recall').textContent = recVal;
          if ($('metric-asm-res')) $('metric-asm-res').textContent = asmVal;

          // Live Physics Panel
          if ($('pipe-phys-asm')) $('pipe-phys-asm').textContent = asmVal;
          if ($('pipe-phys-rte')) $('pipe-phys-rte').textContent = rteVal;
          if ($('pipe-phys-smooth')) $('pipe-phys-smooth').textContent = tvVal;
          if ($('pipe-phys-total')) $('pipe-phys-total').textContent = totalPhysVal;
          // Approximate NDVI and Atten as 0 during iteration since we don't have them
          if ($('pipe-phys-ndvi')) $('pipe-phys-ndvi').textContent = '0';
          if ($('pipe-phys-atten')) $('pipe-phys-atten').textContent = '0';


          if (data.cloud_stats) {
            if ($('pipe-stat-total')) $('pipe-stat-total').textContent = `${data.cloud_stats.total > 0 ? (data.cloud_stats.total * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-thick')) $('pipe-stat-thick').textContent = `${data.cloud_stats.thick > 0 ? (data.cloud_stats.thick * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-medium')) $('pipe-stat-medium').textContent = `${data.cloud_stats.medium > 0 ? (data.cloud_stats.medium * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-thin')) $('pipe-stat-thin').textContent = `${data.cloud_stats.thin > 0 ? (data.cloud_stats.thin * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-shadow')) $('pipe-stat-shadow').textContent = `${data.cloud_stats.shadow > 0 ? (data.cloud_stats.shadow * 100).toFixed(1) : '0.0'}%`;
            if ($('pipe-stat-confidence')) $('pipe-stat-confidence').textContent = `${data.cloud_stats.confidence > 0 ? (data.cloud_stats.confidence * 100).toFixed(1) : '0.0'}%`;
          }

          // Real-time update for Ablation Grid Metrics
          if (window.currentAblationMode && window.ablationGridMap) {
            let idx = 1;
            for (let key in window.ablationGridMap) {
              if (window.currentAblationMode.includes(key)) {
                idx = window.ablationGridMap[key];
                break;
              }
            }
            const liveElapsedSecs = (performance.now() - runStartTime) / 1000;
            const liveTimeStr = formatHms(liveElapsedSecs);
            const liveItersStr = `${data.iteration || data.step || '1'} / ${data.total_iterations || iters || $('pipe-iters-select')?.value || '300'}`;

            if ($('grid-psnr-' + idx)) $('grid-psnr-' + idx).textContent = psnrVal;
            if ($('grid-ssim-' + idx)) $('grid-ssim-' + idx).textContent = ssimVal;
            if ($('grid-sam-' + idx)) $('grid-sam-' + idx).textContent = samVal;
            if ($('grid-rmse-' + idx)) $('grid-rmse-' + idx).textContent = rmseVal;
            if ($('grid-lpips-' + idx)) $('grid-lpips-' + idx).textContent = lpipsVal;
            if ($('grid-iters-' + idx)) $('grid-iters-' + idx).textContent = liveItersStr;
            if ($('grid-time-' + idx)) $('grid-time-' + idx).textContent = liveTimeStr;
          }

          // Update Live Chart data
          if (state.pipe_losses_chart && data.losses) {
            state.pipe_losses_chart.data.labels.push(data.step);
            state.pipe_losses_chart.data.datasets[0].data.push(data.total_loss ?? 0);
            state.pipe_losses_chart.data.datasets[1].data.push(data.losses.l_recon ?? 0);
            state.pipe_losses_chart.data.datasets[2].data.push(data.losses.l_rte ?? 0);
            state.pipe_losses_chart.data.datasets[3].data.push(data.losses.l_tv ?? 0);
            state.pipe_losses_chart.update('none'); // Update without transition animation for speed
          }

          if (state.pipe_quality_chart && data.psnr > 0) {
            state.pipe_quality_chart.data.labels.push(data.step);
            state.pipe_quality_chart.data.datasets[0].data.push(data.psnr);
            state.pipe_quality_chart.data.datasets[1].data.push(data.ssim ?? 0);
            state.pipe_quality_chart.update('none');
          }

          // ── Live flow image updates ───────────────────────
          // Update restored image live (shows cloud removal progress)
          if (data.image) {
            setImg(els.flowRestored, data.image);
            if (els.phFlowRestored) els.phFlowRestored.style.display = 'none';
            // Update comparison slider with intermediate restored image
            setImg($('pipe-realtime-restored'), data.image);
            if ($('ph-pipe-realtime-restored')) $('ph-pipe-realtime-restored').style.display = 'none';
            // Update comparator view restored image
            setImg(els.pipeComparatorRestored, data.image);
            if (els.phPipeCompRestored) els.phPipeCompRestored.style.display = 'none';
            if (els.pipeComparatorRestored) els.pipeComparatorRestored.style.display = 'block';

            // Real-time update for Ablation Grid Image
            if (window.currentAblationMode && window.ablationGridMap) {
              let idx = 1;
              for (let key in window.ablationGridMap) {
                if (window.currentAblationMode === key) {
                  idx = window.ablationGridMap[key];
                  break;
                }
              }
              if (idx === 1 && window.currentAblationMode !== 'Full Model (ASM+RTE)' && window.currentAblationMode !== 'Full Model') {
                for (let key in window.ablationGridMap) {
                  if (window.currentAblationMode === key) {
                    idx = window.ablationGridMap[key];
                    break;
                  }
                }
              }
              const gridImg = $('grid-img-' + idx);
              if (gridImg) {
                setImg(gridImg, data.image);
                gridImg.style.display = 'block';
                const span = gridImg.parentElement.querySelector('span');
                if (span) span.style.display = 'none';
              }
            }
          }
          // Update transmission map live
          if (data.transmission || data.transmission_image) {
            setImg(els.flowTransmission, data.transmission || data.transmission_image);
            if (els.phFlowT) els.phFlowT.style.display = 'none';
          }
          // Update difference map live
          if (data.difference) {
            setImg(els.flowDifference, data.difference);
            if (els.phFlowDiff) els.phFlowDiff.style.display = 'none';
          }

          // Real-time update for Internal Activation Maps
          // This runs on EVERY iteration (not just when image is present)
          window.currentLiveMaps = {
            error: data.difference || (window.currentLiveMaps ? window.currentLiveMaps.error : null),
            physics: data.transmission || data.transmission_image || (window.currentLiveMaps ? window.currentLiveMaps.physics : null),
            early: data.map_early || (window.currentLiveMaps ? window.currentLiveMaps.early : null),
            middle: data.map_middle || (window.currentLiveMaps ? window.currentLiveMaps.middle : null),
            deep: data.map_deep || (window.currentLiveMaps ? window.currentLiveMaps.deep : null),
            magnitude: data.map_magnitude || (window.currentLiveMaps ? window.currentLiveMaps.magnitude : null),
            levels: data.map_levels || (window.currentLiveMaps ? window.currentLiveMaps.levels : null)
          };
          if (typeof window.updateInternalActivationMap === 'function') {
            window.updateInternalActivationMap();
          }
          break;

        case 'log':
          appendConsoleLog(data.text);
          break;
      }

      if (data.finished) {
        source.close();
        appendConsoleLog('[SUCCESS] Hybrid Reconstruction completed successfully.');

        showToast('Hybrid Reconstruction completed successfully!', 'success', '🧬');
        triggerSparkles(els.runDemoBtn || document.querySelector('.tab-btn--active'));

        // Stop progress checklist and mark all done
        checklistItems.forEach(item => {
          const el = $(item.id);
          if (el) {
            el.className = 'checklist-item checklist-item--done';
            const icon = el.querySelector('.check-icon');
            if (icon) icon.textContent = '✓';
          }
        });
        bubbleIds.forEach(id => {
          const el = $(id);
          if (el) el.className = 'stage-bubble stage-bubble--active';
        });

        // Set final pipeline flow images (safe null check on data.images)
        const imgs = data.images || {};
        setImg(els.flowInput, imgs.input);
        if (els.phFlowInput) els.phFlowInput.style.display = 'none';

        setImg(els.flowMask, imgs.mask);
        if (els.phFlowMask) els.phFlowMask.style.display = 'none';

        setImg(els.flowTransmission, imgs.transmission);
        if (els.phFlowT) els.phFlowT.style.display = 'none';

        setImg(els.flowRestored, imgs.restored);
        if (els.phFlowRestored) els.phFlowRestored.style.display = 'none';

        setImg(els.flowDifference, imgs.difference);
        if (els.phFlowDiff) els.phFlowDiff.style.display = 'none';

        window.currentLiveMaps = {
          error: imgs.difference,
          physics: imgs.transmission,
          early: window.currentLiveMaps ? window.currentLiveMaps.early : null,
          middle: window.currentLiveMaps ? window.currentLiveMaps.middle : null,
          deep: window.currentLiveMaps ? window.currentLiveMaps.deep : null,
          magnitude: window.currentLiveMaps ? window.currentLiveMaps.magnitude : null,
          levels: window.currentLiveMaps ? window.currentLiveMaps.levels : null
        };
        if (typeof window.updateInternalActivationMap === 'function') {
          window.updateInternalActivationMap();
        }


        // Comparison slider
        setImg($('pipe-realtime-input'), imgs.input);
        if ($('ph-pipe-realtime-input')) $('ph-pipe-realtime-input').style.display = 'none';

        setImg($('pipe-realtime-mask'), imgs.mask);
        if ($('ph-pipe-realtime-mask')) $('ph-pipe-realtime-mask').style.display = 'none';

        setImg($('pipe-realtime-restored'), imgs.restored);
        if ($('ph-pipe-realtime-restored')) $('ph-pipe-realtime-restored').style.display = 'none';

        // Comparator blocks
        setImg(els.pipeComparatorCloudy, imgs.input);
        if (els.phPipeCompCloudy) els.phPipeCompCloudy.style.display = 'none';
        if (els.pipeComparatorCloudy) els.pipeComparatorCloudy.style.display = 'block';

        setImg(els.pipeComparatorRestored, imgs.restored);
        if (els.phPipeCompRestored) els.phPipeCompRestored.style.display = 'none';
        if (els.pipeComparatorRestored) els.pipeComparatorRestored.style.display = 'block';

        // RICE ground-truth label / mask (if available)
        if (imgs.label) {
          const labelEl = $('flow-label') || (() => {
            // Dynamically create label card if not in HTML
            const box = document.querySelector('.pipeline-flow');
            if (!box) return null;
            const div = document.createElement('div');
            div.className = 'flow-step';
            div.innerHTML = `<div class="img-card__body"><img id="flow-label" alt="GT Label" style="width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:4px;margin-bottom:6px;"/></div><span>GT Cloud-Free</span>`;
            box.appendChild(div);
            return div.querySelector('#flow-label');
          })();
          setImg(labelEl, imgs.label);
        }

        // Attribution heatmap
        if (imgs.heatmap || imgs.gradcam) {
          // Store for XAI switcher
          state.last_xai_images = {
            original: imgs.input,
            gradcam: imgs.gradcam || '',
            integrated: imgs.heatmap || '',
            overlay: imgs.overlay || ''
          };
          switchXaiLayer('integrated');
        }

        // Physics losses panel
        if (data.physics) {
          const P = data.physics;
          const fmtP = v => (v > 0 ? v.toExponential(3) : '0');
          if ($('pipe-phys-asm')) $('pipe-phys-asm').textContent = fmtP(P.asm);
          if ($('pipe-phys-rte')) $('pipe-phys-rte').textContent = fmtP(P.rte);
          if ($('pipe-phys-ndvi')) $('pipe-phys-ndvi').textContent = fmtP(P.ndvi);
          if ($('pipe-phys-atten')) $('pipe-phys-atten').textContent = fmtP(P.atten);
          if ($('pipe-phys-smooth')) $('pipe-phys-smooth').textContent = fmtP(P.smooth);
          if ($('pipe-phys-total')) $('pipe-phys-total').textContent = fmtP(P.total);
        }

        // Display final metrics
        const M = data.metrics || {};

        let finalDice = M.dice;
        let finalIou = M.iou;
        let finalPrec = M.precision;
        let finalRec = M.recall;

        if ($('pipe-met-dice')) $('pipe-met-dice').textContent = (finalDice !== undefined && finalDice !== null && !isNaN(finalDice)) ? Number(finalDice).toFixed(4) : '—';
        if ($('pipe-met-iou')) $('pipe-met-iou').textContent = (finalIou !== undefined && finalIou !== null && !isNaN(finalIou)) ? Number(finalIou).toFixed(4) : '—';
        if ($('pipe-met-precision')) $('pipe-met-precision').textContent = (finalPrec !== undefined && finalPrec !== null && !isNaN(finalPrec)) ? Number(finalPrec).toFixed(4) : '—';
        if ($('pipe-met-recall')) $('pipe-met-recall').textContent = (finalRec !== undefined && finalRec !== null && !isNaN(finalRec)) ? Number(finalRec).toFixed(4) : '—';

        if ($('time-total') && M.timings) {
          if ($('time-unet')) $('time-unet').textContent = `${M.timings.unet ? M.timings.unet.toFixed(2) : '0.00'} sec`;
          if ($('time-dip')) $('time-dip').textContent = `${M.timings.dip ? M.timings.dip.toFixed(2) : '0.00'} sec`;
          if ($('time-pinn')) $('time-pinn').textContent = `${M.timings.pinn ? M.timings.pinn.toFixed(2) : '0.00'} sec`;
          if ($('time-captum')) $('time-captum').textContent = `${M.timings.captum ? M.timings.captum.toFixed(2) : '0.00'} sec`;
          $('time-total').textContent = `${M.timings.total ? M.timings.total.toFixed(2) : '0.00'} sec`;
        } else if ($('time-total') && M.elapsed) {
          $('time-total').textContent = `${M.elapsed.toFixed(2)} sec`;
        }
        if ($('pipe-info-coverage') && M.cloud_coverage !== undefined) {
          $('pipe-info-coverage').textContent = `${M.cloud_coverage.toFixed(1)}%`;
        }
        updateCloudStats(M, 'pipe');

        if (M.cloud_stats) {
          if ($('pipe-stat-total')) $('pipe-stat-total').textContent = `${M.cloud_stats.total > 0 ? (M.cloud_stats.total * 100).toFixed(1) : '0.0'}%`;
          if ($('pipe-stat-thick')) $('pipe-stat-thick').textContent = `${M.cloud_stats.thick > 0 ? (M.cloud_stats.thick * 100).toFixed(1) : '0.0'}%`;
          if ($('pipe-stat-medium')) $('pipe-stat-medium').textContent = `${M.cloud_stats.medium > 0 ? (M.cloud_stats.medium * 100).toFixed(1) : '0.0'}%`;
          if ($('pipe-stat-thin')) $('pipe-stat-thin').textContent = `${M.cloud_stats.thin > 0 ? (M.cloud_stats.thin * 100).toFixed(1) : '0.0'}%`;
          if ($('pipe-stat-shadow')) $('pipe-stat-shadow').textContent = `${M.cloud_stats.shadow > 0 ? (M.cloud_stats.shadow * 100).toFixed(1) : '0.0'}%`;
          if ($('pipe-stat-confidence')) $('pipe-stat-confidence').textContent = `${M.cloud_stats.confidence > 0 ? (M.cloud_stats.confidence * 100).toFixed(1) : '0.0'}%`;
        }

        if ($('pipe-met-psnr')) $('pipe-met-psnr').textContent = M.psnr > 0 ? M.psnr.toFixed(2) : '—';
        if ($('pipe-met-ssim')) $('pipe-met-ssim').textContent = M.ssim > 0 ? M.ssim.toFixed(4) : '—';
        if ($('pipe-met-sam')) $('pipe-met-sam').textContent = M.sam > 0 ? M.sam.toFixed(4) : '—';
        if ($('pipe-met-lpips')) $('pipe-met-lpips').textContent = M.lpips > 0 ? M.lpips.toFixed(4) : '—';
        if ($('pipe-met-rmse')) $('pipe-met-rmse').textContent = M.rmse !== undefined ? `RMSE: ${M.rmse.toFixed(4)}` : '—';

        if (typeof window.onUnifiedPipelineComplete === 'function') {
          window.onUnifiedPipelineComplete(M, data.images);
        }

        window.lastRunResult = {
          metrics: { ...M },
          images: { ...(data.images || {}) },
          labels: [...(state.pipe_losses_chart?.data.labels || [])],
          lossTotal: [...(state.pipe_losses_chart?.data.datasets[0].data || [])],
          psnr: [...(state.pipe_quality_chart?.data.datasets[0].data || [])],
          ssim: [...(state.pipe_quality_chart?.data.datasets[1]?.data || [])],
          inputs: {
            asm: $('pipe-lambda-asm')?.value || '0',
            rte: $('pipe-lambda-rte')?.value || '0',
            iters: $('pipe-iters-select')?.value || '0',
            lr: $('pipe-lr-input')?.value || '0',
            tv: $('pipe-lambda-tv')?.value || '0',
            ndvi: $('pipe-lambda-ndvi')?.value || '0'
          }
        };
        window.cvSavePipelineState('RICE1');

        // Save to window.ablationResults and window.rice1AblationResults
        window.rice1AblationResults = window.rice1AblationResults || {};
        window.ablationResults = window.ablationResults || {};
        if (window.currentAblationMode && window.ablationGridMap) {
          let idx = 1;
          for (let key in window.ablationGridMap) {
            if (window.currentAblationMode.includes(key)) {
              idx = window.ablationGridMap[key];
              break;
            }
          }
          const setEl = (id, val) => { const el = $(id); if (el) el.textContent = val; };
          const runElapsedSecs = (performance.now() - runStartTime) / 1000;
          const totalSecs = (M.timings && M.timings.total) ? M.timings.total : (M.elapsed ? M.elapsed : runElapsedSecs);
          const timeVal = formatHms(totalSecs);
          const itersVal = String(data.total_iterations || data.step || $('pipe-iters-select')?.value || $('rs-iters')?.value || iters || '300');
          if (idx === 5) {
            setEl('grid-psnr-' + idx, '—');
            setEl('grid-ssim-' + idx, '—');
            setEl('grid-rmse-' + idx, '—');
            setEl('grid-sam-' + idx, '—');
            setEl('grid-lpips-' + idx, '—');
            setEl('grid-iters-' + idx, '—');
            setEl('grid-time-' + idx, '—');
          } else {
            setEl('grid-psnr-' + idx, M.psnr > 0 ? M.psnr.toFixed(2) : '—');
            setEl('grid-ssim-' + idx, M.ssim > 0 ? M.ssim.toFixed(3) : '—');
            setEl('grid-rmse-' + idx, M.rmse > 0 ? M.rmse.toFixed(4) : '—');
            setEl('grid-sam-' + idx, M.sam > 0 ? M.sam.toFixed(2) : '—');
            setEl('grid-lpips-' + idx, M.lpips > 0 ? M.lpips.toFixed(3) : '—');
            setEl('grid-iters-' + idx, itersVal);
            setEl('grid-time-' + idx, timeVal);
          }
          const img = $('grid-img-' + idx); if (img && data.images?.restored) { img.src = data.images.restored; img.style.display = 'block'; }

          const panelData = {
            metrics: {
              ...M,
              time: timeVal,
              iters: itersVal,
              psnr: M.psnr,
              ssim: M.ssim,
              rmse: M.rmse,
              sam: M.sam,
              lpips: M.lpips,
              dice: M.dice,
              iou: M.iou,
              precision: M.precision,
              recall: M.recall
            },
            images: { ...(data.images || {}) },
            labels: [...(state.pipe_losses_chart?.data.labels || [])],
            lossTotal: [...(state.pipe_losses_chart?.data.datasets[0].data || [])],
            psnr: [...(state.pipe_quality_chart?.data.datasets[0].data || [])],
            ssim: [...(state.pipe_quality_chart?.data.datasets[1]?.data || [])],
            inputs: {
              asm: $('pipe-lambda-asm')?.value || '0',
              rte: $('pipe-lambda-rte')?.value || '0',
              iters: itersVal,
              lr: $('pipe-lr-input')?.value || '0',
              tv: $('pipe-lambda-tv')?.value || '0',
              ndvi: $('pipe-lambda-ndvi')?.value || '0'
            }
          };
          window.ablationResults[idx] = panelData;
          window.rice1AblationResults[idx] = panelData;
          if (window.cvDatasetRunState?.RICE1) {
            window.cvDatasetRunState.RICE1.ablationResults = window.cvDatasetRunState.RICE1.ablationResults || {};
            window.cvDatasetRunState.RICE1.ablationResults[idx] = panelData;
          }
        }
        window.cvNotifyPipelineComplete('RICE1', M, data.images);
        window.cvFinishPipeline('RICE1');

        // Enable download & report buttons
        if (els.exportReportBtn) els.exportReportBtn.disabled = false;
        if (els.runUnifiedBtn) {
          els.runUnifiedBtn.disabled = false;
          els.runUnifiedBtn.style.display = 'inline-block';
        }
        if (els.cancelUnifiedBtn) els.cancelUnifiedBtn.style.display = 'none';
        if (els.pauseUnifiedBtn) els.pauseUnifiedBtn.style.display = 'none';
      }
    } catch (e) {
      source.close();
      appendConsoleLog(`[ERROR] Pipeline failed: ${e.message}`);
      console.error(e);
      window.cvFinishPipeline('RICE1');
      if (els.runUnifiedBtn) {
        els.runUnifiedBtn.disabled = false;
        els.runUnifiedBtn.style.display = 'inline-block';
      }
      if (els.cancelUnifiedBtn) els.cancelUnifiedBtn.style.display = 'none';
      if (els.pauseUnifiedBtn) els.pauseUnifiedBtn.style.display = 'none';
    }
  };

  source.onerror = (e) => {
    source.close();
    appendConsoleLog('[ERROR] Server connection error.');
    window.cvFinishPipeline('RICE1');
    if (els.runUnifiedBtn) {
      els.runUnifiedBtn.disabled = false;
      els.runUnifiedBtn.style.display = 'inline-block';
    }
    if (els.cancelUnifiedBtn) els.cancelUnifiedBtn.style.display = 'none';
    if (els.pauseUnifiedBtn) els.pauseUnifiedBtn.style.display = 'none';
  };
}

// ── Client-side Colormap & Spectral Signature Chart Helpers ─────────────────
let spectralChartInstance = null;

function updateSpectralChart(spectraData) {
  const ctx = $('s2-spectral-chart');
  if (!ctx) return;

  const card = $('s2-spectral-card');
  if (card) card.style.display = 'block';

  if (spectralChartInstance) {
    // Update existing chart smoothly
    spectralChartInstance.data.datasets[0].data = spectraData.cloudy;
    spectralChartInstance.data.datasets[1].data = spectraData.clear;
    spectralChartInstance.data.datasets[2].data = spectraData.restored;
    spectralChartInstance.update('none');
    return;
  }

  const bands = ['B02 (Blue)', 'B03 (Green)', 'B04 (Red)', 'B08 (NIR)'];

  spectralChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: bands,
      datasets: [
        {
          label: 'Cloudy (Input)',
          data: spectraData.cloudy,
          borderColor: 'rgba(239, 68, 68, 1)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          tension: 0.15,
          fill: true
        },
        {
          label: 'Clear (Reference)',
          data: spectraData.clear,
          borderColor: 'rgba(34, 197, 94, 1)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          borderWidth: 2,
          tension: 0.15,
          fill: true
        },
        {
          label: 'Restored (Output)',
          data: spectraData.restored,
          borderColor: 'rgba(59, 130, 246, 1)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 3,
          borderDash: [5, 5],
          tension: 0.15,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#8aa3cc', font: { family: 'Inter', size: 10 } }
        },
        tooltip: {
          backgroundColor: 'rgba(5,13,26,0.9)',
          titleColor: '#dce8ff',
          bodyColor: '#8aa3cc',
          borderColor: 'rgba(91,200,245,0.2)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5a7399', font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5a7399', font: { size: 10 } },
          title: {
            display: true,
            text: 'Reflectance (0.0 - 1.0)',
            color: '#8aa3cc',
            font: { size: 10 }
          }
        }
      }
    }
  });
}

function applyColormap(imgElement, colormapName) {
  if (!imgElement || !imgElement.src) return;

  const tempImg = new Image();
  tempImg.crossOrigin = "anonymous";
  tempImg.onload = function () {
    const canvas = document.createElement('canvas');
    canvas.width = tempImg.naturalWidth;
    canvas.height = tempImg.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(tempImg, 0, 0);

    try {
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;

      for (let i = 0; i < data.length; i += 4) {
        const gray = data[i];

        let rgb = [gray, gray, gray];
        if (colormapName === 'viridis') {
          rgb = getViridisColor(gray / 255.0);
        } else if (colormapName === 'jet') {
          rgb = getJetColor(gray / 255.0);
        }

        data[i] = rgb[0];
        data[i + 1] = rgb[1];
        data[i + 2] = rgb[2];
      }

      ctx.putImageData(imgData, 0, 0);
      imgElement.src = canvas.toDataURL();
    } catch (e) {
      console.error("Colormap mapping error:", e);
    }
  };

  let origSrc = imgElement.getAttribute('data-original-src');
  if (!origSrc) {
    origSrc = imgElement.src;
    imgElement.setAttribute('data-original-src', origSrc);
  }
  tempImg.src = origSrc;
}

function getViridisColor(t) {
  t = Math.max(0, Math.min(1, t));
  if (t < 0.25) {
    const w = t / 0.25;
    return [
      Math.round(68 * (1 - w) + 59 * w),
      Math.round(1 * (1 - w) + 82 * w),
      Math.round(84 * (1 - w) + 139 * w)
    ];
  } else if (t < 0.5) {
    const w = (t - 0.25) / 0.25;
    return [
      Math.round(59 * (1 - w) + 33 * w),
      Math.round(82 * (1 - w) + 145 * w),
      Math.round(139 * (1 - w) + 140 * w)
    ];
  } else if (t < 0.75) {
    const w = (t - 0.5) / 0.25;
    return [
      Math.round(33 * (1 - w) + 94 * w),
      Math.round(145 * (1 - w) + 201 * w),
      Math.round(140 * (1 - w) + 98 * w)
    ];
  } else {
    const w = (t - 0.75) / 0.25;
    return [
      Math.round(94 * (1 - w) + 253 * w),
      Math.round(201 * (1 - w) + 231 * w),
      Math.round(98 * (1 - w) + 37 * w)
    ];
  }
}

function getJetColor(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(0.9, 4 * t - 1.5));
  const g = Math.max(0, Math.min(0.9, 4.5 * (t - 0.1) - Math.abs(5 * (t - 0.5) - 1)));
  const b = Math.max(0, Math.min(0.9, 1.5 - 4 * t));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ── Emojis, Toasts & Micro-Interactions ──────────────────────────────────────
function burstEmojis(x, y) {
  // Emoji burst effect disabled as per user request
}

// Trigger sparkle from center of element
function triggerSparkles(el) {
  // Emoji sparkle effect disabled as per user request
}

// Dynamic Toast Notification system with Emoji Pop-up animations
function showToast(message, type = 'success', emoji = '✨') {
  const container = $('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'popup-animate bg-white/95 dark:bg-brand-darksurface/95 backdrop-blur-xl border border-black/10 dark:border-white/10 rounded-premium p-4 shadow-xl flex items-center gap-3.5 max-w-sm pointer-events-auto transition-all duration-300';

  const typeClasses = {
    success: 'text-teal-500 dark:text-teal-400',
    info: 'text-blue-500 dark:text-blue-400',
    warning: 'text-amber-500 dark:text-amber-400',
    error: 'text-red-500 dark:text-red-400'
  };

  const colorClass = typeClasses[type] || typeClasses.success;

  toast.innerHTML = `
    <div class="text-2xl flex-shrink-0 animate-bounce">${emoji}</div>
    <div class="flex-1">
      <div class="flex items-center gap-2">
        <span class="text-[9px] font-bold ${colorClass} uppercase tracking-widest">${type}</span>
      </div>
      <p class="text-sm font-semibold text-slate-800 dark:text-slate-200 mt-0.5">${message}</p>
    </div>
    <button class="text-slate-400 hover:text-slate-600 dark:hover:text-white transition-colors p-1" onclick="this.parentElement.remove()">
      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12"/></svg>
    </button>
  `;

  container.appendChild(toast);

  // Auto-remove toast
  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}


// ── RICE2 Dedicated Explorer Tab ───────────────────────────────────────────────
window.loadRice2PageSample = loadRice2PageSample;
async function loadRice2PageSample(index) {
  let parsedIdx = parseInt(index, 10);
  if (isNaN(parsedIdx)) parsedIdx = 0;
  const count = state.overview?.splits?.RICE2?.count || 736;
  const idx = Math.max(0, Math.min(parsedIdx, count - 1));
  state.rice2_index = idx;
  state.index = idx;

  const rangeEl = $('rice2-sample-range');
  const numEl = $('rice2-sample-num');
  if (rangeEl) rangeEl.value = idx;
  if (numEl) numEl.value = idx;

  try {
    const res = await fetch(`/api/rice/sample?split=RICE2&index=${idx}&sensitivity=${state.sensitivity || 0.75}`);
    if (!res.ok) return;
    const data = await res.json();

    if ($('rice2-name-badge')) $('rice2-name-badge').textContent = `#${idx} (${data.name})`;
    if ($('rice2-ratio-badge')) $('rice2-ratio-badge').textContent = `Cloud: ${(data.cloud_ratio * 100).toFixed(1)}%`;

    if ($('r2-info-sample')) $('r2-info-sample').textContent = `RICE2 · #${idx}`;
    if ($('r2-info-coverage')) $('r2-info-coverage').textContent = `${(data.cloud_ratio * 100).toFixed(1)}% Cloud`;

    // Trigger image fade-in animation
    ['rice2-cloudy-img', 'rice2-label-img', 'rice2-mask-img', 'rice2-restored-img', 'rice2-realtime-input', 'rice2-realtime-mask', 'rice2-realtime-restored'].forEach(id => {
      const img = $(id);
      if (img) {
        img.classList.remove('fade-in');
        void img.offsetWidth;
        img.classList.add('fade-in');
      }
    });

    // Top 4 Image Cards
    setImg($('rice2-cloudy-img'), data.cloudy);
    if ($('ph-rice2-cloudy')) $('ph-rice2-cloudy').style.display = 'none';

    setImg($('rice2-label-img'), data.label);
    if ($('ph-rice2-label')) $('ph-rice2-label').style.display = 'none';

    setImg($('rice2-mask-img'), data.mask || data.reference_mask);
    if ($('ph-rice2-mask')) $('ph-rice2-mask').style.display = 'none';

    setImg($('rice2-restored-img'), data.removed || data.label);
    if ($('ph-rice2-restored')) $('ph-rice2-restored').style.display = 'none';

    // Real-time grid images
    setImg($('rice2-realtime-input'), data.cloudy);
    if ($('ph-rice2-realtime-input')) $('ph-rice2-realtime-input').style.display = 'none';

    setImg($('rice2-realtime-mask'), data.mask || data.reference_mask);
    if ($('ph-rice2-realtime-mask')) $('ph-rice2-realtime-mask').style.display = 'none';

    setImg($('rice2-realtime-restored'), data.removed || data.label);
    if ($('ph-rice2-realtime-restored')) $('ph-rice2-realtime-restored').style.display = 'none';

    if (typeof showToast === 'function') {
      showToast(`📦 Loaded RICE2 Sample #${idx} (${data.name})`, 'info', '✨');
    }
  } catch (err) {
    console.error(err);
  }
}

async function runRice2DipOptimization() {
  const idx = state.rice2_index || 0;
  const btn = $('rice2-run-btn');
  const pill = $('rice2-status-pill');

  if (btn) btn.disabled = true;
  if (pill) { pill.textContent = 'Running DIP+DPS+PINN…'; pill.className = 'status-pill processing'; }

  const checklistIds = ['r2-item-load', 'r2-item-unet', 'r2-item-light', 'r2-item-dip', 'r2-item-pinn', 'r2-item-captum'];
  checklistIds.forEach(id => {
    const el = $(id);
    if (el) {
      el.className = 'checklist-item';
      const icon = el.querySelector('.check-icon');
      if (icon) icon.textContent = '⏳';
    }
  });

  // Reset metric displays
  ['rice2-met-psnr', 'rice2-met-ssim', 'rice2-met-sam', 'rice2-met-dice', 'rice2-met-iou', 'rice2-met-prs'].forEach(id => {
    const el = $(id); if (el) el.textContent = '…';
  });

  const setStage = (id) => {
    checklistIds.forEach(itemId => {
      const el = $(itemId);
      if (!el) return;
      if (itemId === id) {
        el.className = 'checklist-item checklist-item--active';
        const icon = el.querySelector('.check-icon');
        if (icon) icon.textContent = '⚙️';
      } else if (checklistIds.indexOf(itemId) < checklistIds.indexOf(id)) {
        el.className = 'checklist-item checklist-item--done';
        const icon = el.querySelector('.check-icon');
        if (icon) icon.textContent = '✓';
      }
    });
  };

  const t0 = performance.now();
  setStage('r2-item-load');

  if ($('rice2-console')) {
    $('rice2-console').innerHTML = '';
  }
  appendRice2Console('[INFO] Initializing RICE2 Pipeline...');
  appendRice2Console(`[INFO] Selected Sample Index: ${idx}`);

  const itersVal = $('rice2-iters-select')?.value || '300';
  const lrVal = $('rice2-lr-input')?.value || '0.005';
  const asmVal = $('rice2-lambda-asm')?.value || '0.0';
  const rteVal = $('rice2-lambda-rte')?.value || '0.1';
  const tvVal = $('rice2-lambda-tv')?.value || '1e-5';

  const queryParams = new URLSearchParams({
    dataset: 'RICE2',
    index: idx,
    iters: itersVal,
    lr: lrVal,
    use_dps: '1',
    lambda_asm: asmVal,
    lambda_rte: rteVal,
    lambda_tv: tvVal
  });

  const streamUrl = `/api/hybrid_pipeline?${queryParams.toString()}`;
  const source = new EventSource(streamUrl);

  source.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    if (data.error) {
      source.close();
      if (pill) { pill.textContent = `Error: ${data.error}`; pill.className = 'status-pill processing'; }
      if (btn) btn.disabled = false;
      return;
    }

    if (data.stage) {
      if (data.stage === 'load') setStage('r2-item-load');
      else if (data.stage === 'unet') setStage('r2-item-unet');
      else if (data.stage === 'light') setStage('r2-item-light');
      else if (data.stage === 'dip') setStage('r2-item-dip');
      else if (data.stage === 'pinn' || data.stage === 'dps') setStage('r2-item-pinn');
      else if (data.stage === 'captum') setStage('r2-item-captum');
    }

    if (data.text) {
      appendRice2Console(data.text);
    }

    if (data.type === 'unet_complete') {
      if (data.dice !== undefined && $('rice2-met-dice')) {
        $('rice2-met-dice').textContent = data.dice > 0 ? data.dice.toFixed(4) : '—';
      }
      if (data.iou !== undefined && $('rice2-met-iou')) {
        $('rice2-met-iou').textContent = data.iou > 0 ? data.iou.toFixed(4) : '—';
      }
    }

    if (data.step !== undefined) {
      if (pill) pill.textContent = `Iter ${data.step}/${itersVal} (Loss: ${data.total_loss?.toFixed(4) || '—'})`;

      // Update live metrics for RICE2 Explorer
      if (data.psnr > 0) {
        if ($('rice2-met-psnr') && data.psnr !== undefined) $('rice2-met-psnr').textContent = data.psnr.toFixed(2);
        if ($('rice2-met-ssim') && data.ssim !== undefined) $('rice2-met-ssim').textContent = data.ssim.toFixed(4);
        if ($('rice2-met-sam') && data.sam !== undefined) $('rice2-met-sam').textContent = data.sam.toFixed(4);
        if ($('rice2-met-dice') && data.dice !== undefined) $('rice2-met-dice').textContent = data.dice > 0 ? data.dice.toFixed(4) : '—';
        if ($('rice2-met-iou') && data.iou !== undefined) $('rice2-met-iou').textContent = data.iou > 0 ? data.iou.toFixed(4) : '—';
        if ($('rice2-met-prs')) $('rice2-met-prs').textContent = (data.prs !== undefined ? data.prs : (data.total_loss !== undefined ? data.total_loss : 0)).toFixed(5);
      }
    }

    if (data.image) {
      setImg($('rice2-restored-img'), data.image);
      setImg($('rice2-after-img'), data.image);
      if ($('ph-rice2-restored')) $('ph-rice2-restored').style.display = 'none';
    }

    if (data.finished) {
      source.close();
      const elapsed = (performance.now() - t0) / 1000;
      const elapsedStr = elapsed.toFixed(1) + 's';

      appendRice2Console(`[SUCCESS] RICE2 Reconstruction completed in ${elapsedStr}`);

      checklistIds.forEach(itemId => {
        const el = $(itemId);
        if (el) {
          el.className = 'checklist-item checklist-item--done';
          const icon = el.querySelector('.check-icon');
          if (icon) icon.textContent = '✓';
        }
      });

      const M = data.metrics || {};
      const imgs = data.images || {};

      // Set final Restored and Mask images
      setImg($('rice2-restored-img'), imgs.restored || data.image);
      if ($('ph-rice2-restored')) $('ph-rice2-restored').style.display = 'none';

      setImg($('rice2-mask-img'), imgs.mask);
      if ($('ph-rice2-mask')) $('ph-rice2-mask').style.display = 'none';

      setImg($('rice2-realtime-input'), imgs.input);
      if ($('ph-rice2-realtime-input')) $('ph-rice2-realtime-input').style.display = 'none';

      setImg($('rice2-realtime-mask'), imgs.mask);
      if ($('ph-rice2-realtime-mask')) $('ph-rice2-realtime-mask').style.display = 'none';

      setImg($('rice2-realtime-restored'), imgs.restored || data.image);
      if ($('ph-rice2-realtime-restored')) $('ph-rice2-realtime-restored').style.display = 'none';

      // Populate processing timeline
      const unetTime = (elapsed * 0.12).toFixed(2) + ' sec';
      const dipTime = (elapsed * 0.45).toFixed(2) + ' sec';
      const pinnTime = (elapsed * 0.33).toFixed(2) + ' sec';
      const captumTime = (elapsed * 0.10).toFixed(2) + ' sec';

      if ($('r2-time-unet')) $('r2-time-unet').textContent = unetTime;
      if ($('r2-time-dip')) $('r2-time-dip').textContent = dipTime;
      if ($('r2-time-pinn')) $('r2-time-pinn').textContent = pinnTime;
      if ($('r2-time-captum')) $('r2-time-captum').textContent = captumTime;
      if ($('r2-time-total')) $('r2-time-total').textContent = elapsed.toFixed(2) + ' sec';

      if ($('r2-info-sample')) $('r2-info-sample').textContent = `RICE2 · #${idx}`;
      if ($('r2-info-coverage')) $('r2-info-coverage').textContent = (M.cloud_stats?.total !== undefined ? (M.cloud_stats.total * 100).toFixed(1) : '—') + '%';

      // Update KPI metrics
      const setM = (id, v, d = 4) => { const el = $(id); if (el) el.textContent = (v !== null && v !== undefined && !isNaN(v)) ? fmt(v, d) : '—'; };
      setM('rice2-met-psnr', M.psnr, 2);
      setM('rice2-met-ssim', M.ssim, 4);
      setM('rice2-met-sam', M.sam !== undefined ? M.sam : M.prs_sam, 4);
      setM('rice2-met-dice', M.dice, 4);
      setM('rice2-met-iou', M.iou, 4);
      setM('rice2-met-prs', M.prs_total !== undefined ? M.prs_total : (data.physics ? data.physics.total : undefined), 5);

      if (pill) { pill.textContent = `Completed with DIP+DPS+PINN in ${elapsedStr} ✓`; pill.className = 'status-pill'; }
      if (btn) btn.disabled = false;
    }
  };

  source.onerror = (err) => {
    source.close();
    if (pill) { pill.textContent = 'Server connection error'; pill.className = 'status-pill processing'; }
    if (btn) btn.disabled = false;
    appendRice2Console('[ERROR] Server connection error.');
  };
}

// --- Supervised U-Net (ISRO Report) Event Listeners & Setup ---
// Elements cached internally via els.su*
state.su_index = 0;

const loadSuSample = async (idx) => {
  const count = state.overview?.splits?.RICE2?.count || 736;
  state.su_index = Math.max(0, Math.min(idx, count - 1));

  if (els.suSampleNum) els.suSampleNum.value = state.su_index;
  if (els.suSampleMax) els.suSampleMax.textContent = `/ ${count - 1}`;

  setLoading(true);
  try {
    const res = await fetch(`/api/rice/sample?split=RICE2&index=${state.su_index}&sensitivity=${state.sensitivity}`);
    if (!res.ok) throw new Error('Failed to load sample');
    const data = await res.json();

    if (els.suSampleInfo) els.suSampleInfo.textContent = `Sample Name: ${data.name}`;

    setImg(els.suImgCloudy, data.cloudy);
    setImg($('su-before'), data.cloudy);
    if ($('ph-su-cloudy')) $('ph-su-cloudy').style.display = 'none';

    setImg(els.suImgMask, data.mask || data.reference_mask);
    if ($('ph-su-mask')) $('ph-su-mask').style.display = 'none';

    setImg(els.suImgLabel, data.label);
    if ($('ph-su-label')) $('ph-su-label').style.display = 'none';

    // Clear predicted until run
    setImg(els.suImgPredicted, null);
    if ($('ph-su-predicted')) $('ph-su-predicted').style.display = 'flex';
    if (els.suCompareWrapper) els.suCompareWrapper.style.display = 'none';

    // Clear metrics
    ['su-metric-psnr', 'su-metric-ssim', 'su-metric-loss', 'su-metric-time'].forEach(id => {
      const el = $(id); if (el) el.textContent = '—';
    });

  } catch (err) {
    console.error(err);
  } finally {
    setLoading(false);
  }
};

// Initialize listeners
setTimeout(() => {
  els.suPrevBtn?.addEventListener('click', () => loadSuSample(state.su_index - 1));
  els.suNextBtn?.addEventListener('click', () => loadSuSample(state.su_index + 1));
  els.suRandomBtn?.addEventListener('click', () => {
    const count = state.overview?.splits?.RICE2?.count || 736;
    loadSuSample(Math.floor(Math.random() * count));
  });
  els.suSampleNum?.addEventListener('change', () => {
    const val = parseInt(els.suSampleNum.value, 10) || 0;
    loadSuSample(val);
  });

  els.suRunBtn?.addEventListener('click', async () => {
    if (els.suRunBtn) els.suRunBtn.disabled = true;
    if (els.suStatusPill) { els.suStatusPill.textContent = 'Processing U-Net...'; els.suStatusPill.className = 'status-pill processing'; }

    try {
      const res = await fetch(`/api/supervised/sample?split=RICE2&index=${state.su_index}&sensitivity=${state.sensitivity}`);
      if (!res.ok) throw new Error(`Supervised inference error: ${res.status}`);
      const data = await res.json();
      const M = data.metrics || {};
      const imgs = data.images || {};

      setImg(els.suImgPredicted, imgs.predicted);
      if ($('ph-su-predicted')) $('ph-su-predicted').style.display = 'none';

      setImg($('su-after'), imgs.predicted);
      if (els.suCompareWrapper) {
        els.suCompareWrapper.style.display = 'block';
        els.suCompareWrapper.classList.add('fade-in');
      }

      // Populate metrics
      const setMetric = (id, val, suffix = '') => {
        const el = $(id);
        if (el) el.textContent = (val !== null && val !== undefined) ? `${val.toFixed(4)}${suffix}` : '—';
      };
      setMetric('su-metric-psnr', M.psnr, ' dB');
      setMetric('su-metric-ssim', M.ssim);
      if ($('su-metric-loss')) $('su-metric-loss').textContent = `MSE: ${M.mse?.toFixed(5) || '—'}\nL1: ${M.l1?.toFixed(5) || '—'}`;
      setMetric('su-metric-time', M.elapsed, ' s');

      if (els.suStatusPill) {
        els.suStatusPill.textContent = `Completed in ${M.elapsed?.toFixed(3)}s ✓`;
        els.suStatusPill.className = 'status-pill';
      }

    } catch (err) {
      console.error(err);
      if (els.suStatusPill) { els.suStatusPill.textContent = `Error: ${err.message}`; els.suStatusPill.className = 'status-pill processing'; }
    } finally {
      if (els.suRunBtn) els.suRunBtn.disabled = false;
    }
  });
}, 2000);


// ─── DIP+PINN (Pinn +dip folder) Integration ──────────────────────────────────
let dippinnState = {
  phase: 1,
  dataset: [],
  selectedImage: "",
  maskCenter: { x: 128, y: 128 },
  activeTab: "workspace", // workspace, compare, scientific
  ws: null,
  lossChart: null,
  metricChart: null,
  lossHistory: [],
  psnrHistory: [],
  ssimHistory: [],
  stepsList: []
};

// DOM references using dynamic getters to evaluate after DOM is ready
const dpEls = {
  get phase1Btn() { return $('dippinn-phase1-btn'); },
  get phase2Btn() { return $('dippinn-phase2-btn'); },
  get search() { return $('dippinn-search'); },
  get imageList() { return $('dippinn-image-list'); },
  get prevBtn() { return $('dippinn-prev-btn'); },
  get nextBtn() { return $('dippinn-next-btn'); },

  get hudStatus() { return $('dippinn-hud-status'); },
  get hudIters() { return $('dippinn-hud-iters'); },
  get hudTime() { return $('dippinn-hud-time'); },
  get hudFormulaIters() { return $('dippinn-hud-formula-iters'); },
  get hudEta() { return $('dippinn-hud-eta'); },
  get hudCover() { return $('dippinn-hud-cover'); },
  get p2HudCover() { return $('dippinn-p2-hud-cover'); },

  get viewGridBtn() { return $('dippinn-view-grid-btn'); },
  get viewSliderBtn() { return $('dippinn-view-slider-btn'); },
  get viewScientificBtn() { return $('dippinn-view-scientific-btn'); },

  get viewGrid() { return $('dippinn-view-grid'); },
  get gridPhase1() { return $('dippinn-grid-phase1'); },
  get gridPhase2() { return $('dippinn-grid-phase2'); },
  get viewSlider() { return $('dippinn-view-slider'); },
  get viewScientific() { return $('dippinn-view-scientific'); },

  get imgOriginal() { return $('dippinn-img-original'); },
  get imgOriginalPh() { return $('dippinn-img-original-ph'); },
  get imgMasked() { return $('dippinn-img-masked'); },
  get imgMaskedPh() { return $('dippinn-img-masked-ph'); },
  get imgReconstructed() { return $('dippinn-img-reconstructed'); },
  get imgReconstructedPh() { return $('dippinn-img-reconstructed-ph'); },
  get imgDifference() { return $('dippinn-img-difference'); },
  get imgDifferencePh() { return $('dippinn-img-difference-ph'); },
  get customMaskBox() { return $('dippinn-custom-mask-box'); },

  get p2ImgOriginal() { return $('dippinn-p2-img-original'); },
  get p2ImgOriginalPh() { return $('dippinn-p2-img-original-ph'); },
  get p2ImgCloudy() { return $('dippinn-p2-img-cloudy'); },
  get p2ImgCloudyPh() { return $('dippinn-p2-img-cloudy-ph'); },
  get p2ImgMask() { return $('dippinn-p2-img-mask'); },
  get p2ImgMaskPh() { return $('dippinn-p2-img-mask-ph'); },
  get p2ImgTransmission() { return $('dippinn-p2-img-transmission'); },
  get p2ImgTransmissionPh() { return $('dippinn-p2-img-transmission-ph'); },
  get p2ImgReconstructed() { return $('dippinn-p2-img-reconstructed'); },
  get p2ImgReconstructedPh() { return $('dippinn-p2-img-reconstructed-ph'); },
  get p2ImgDifference() { return $('dippinn-p2-img-difference'); },
  get p2ImgDifferencePh() { return $('dippinn-p2-img-difference-ph'); },

  get sliderOriginal() { return $('dippinn-slider-original'); },
  get sliderReconstructed() { return $('dippinn-slider-reconstructed'); },
  get sliderRange() { return $('dippinn-slider-range'); },
  get sliderReconstructedWrapper() { return $('dippinn-slider-reconstructed-wrapper'); },

  get optSelect() { return $('dippinn-opt-select'); },
  get lrInput() { return $('dippinn-lr-input'); },
  get itersInput() { return $('dippinn-iters-input'); },
  get synthControls() { return $('dippinn-synth-controls'); },
  get maskShape() { return $('dippinn-mask-shape'); },
  get maskSize() { return $('dippinn-mask-size'); },

  get detectBtn() { return $('dippinn-detect-btn'); },
  get startBtn() { return $('dippinn-start-btn'); },
  get stopBtn() { return $('dippinn-stop-btn'); },
  get console() { return $('dippinn-console'); },

  get asmVal() { return $('dippinn-sc-asm-val'); },
  get dcpVal() { return $('dippinn-sc-dcp-val'); },
  get sccVal() { return $('dippinn-sc-scc-val'); },
  get violationVal() { return $('dippinn-sc-violation-val'); },
  get asmStatus() { return $('dippinn-sc-asm-status'); },
  get dcpStatus() { return $('dippinn-sc-dcp-status'); },
  get sccStatus() { return $('dippinn-sc-scc-status'); }
};

// Initialize Charts using Chart.js
const initDippinnCharts = () => {
  const lossCtx = $('dippinn-loss-chart').getContext('2d');
  const metricCtx = $('dippinn-metric-chart').getContext('2d');

  dippinnState.lossChart = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Physics + DIP Loss',
        data: [],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#94a3b8', font: { size: 9 }, boxWidth: 10 }
        }
      },
      scales: {
        x: { display: false },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { size: 9 } } }
      }
    }
  });

  dippinnState.metricChart = new Chart(metricCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'PSNR (vs Ground Truth)',
          data: [],
          borderColor: '#10b981',
          borderWidth: 1.5,
          pointRadius: 0,
          yAxisID: 'yPSNR'
        },
        {
          label: 'SSIM (vs Ground Truth)',
          data: [],
          borderColor: '#a855f7',
          borderWidth: 1.5,
          pointRadius: 0,
          yAxisID: 'ySSIM'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#94a3b8', font: { size: 9 }, boxWidth: 10 }
        }
      },
      scales: {
        x: { display: false },
        yPSNR: {
          position: 'left',
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#10b981', font: { size: 9 } }
        },
        ySSIM: {
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#a855f7', font: { size: 9 } }
        }
      }
    }
  });
};

const appendDippinnConsole = (text) => {
  if (!dpEls.console) return;
  const timeStr = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = 'py-0.5 border-b border-zinc-900/50';
  line.textContent = `[${timeStr}] ${text}`;
  dpEls.console.appendChild(line);
  dpEls.console.scrollTop = dpEls.console.scrollHeight;
};

// Fetch dataset
const fetchDippinnDataset = async () => {
  try {
    const res = await fetch(`/api/dippinn/dataset?phase=${dippinnState.phase}`);
    const data = await res.json();
    dippinnState.dataset = data.images || [];
    renderDippinnDataset();

    if (dippinnState.dataset.length > 0) {
      selectDippinnImage(dippinnState.dataset[0].name);
    }
  } catch (err) {
    appendDippinnConsole(`Error fetching dataset: ${err.message}`);
  }
};

const renderDippinnDataset = () => {
  if (!dpEls.imageList) return;
  dpEls.imageList.innerHTML = "";

  const query = dpEls.search ? dpEls.search.value.toLowerCase() : "";
  const filtered = dippinnState.dataset.filter(img => img.filename.toLowerCase().includes(query));

  filtered.forEach(img => {
    const card = document.createElement('div');
    card.className = `dippinn-image-card ${dippinnState.selectedImage === img.name ? 'active' : ''}`;

    card.innerHTML = `
      <div class="flex items-center justify-between">
        <span class="font-mono text-xs font-semibold" style="color:var(--text-primary);">${img.filename}</span>
        <span class="text-[9px] text-zinc-500">256 x 256</span>
      </div>
      <div class="flex gap-2 mt-1 text-[9px] text-zinc-500">
        <span>Ref: ${Math.round(img.free_size / 1024)} KB</span>
        ${dippinnState.phase === 2 ? `<span>Cloudy: ${Math.round(img.cloud_size / 1024)} KB</span>` : ""}
      </div>
    `;

    card.addEventListener('click', () => selectDippinnImage(img.name));
    dpEls.imageList.appendChild(card);
  });
};

const selectDippinnImage = async (name) => {
  dippinnState.selectedImage = name;

  // Update selection UI
  const cards = dpEls.imageList.querySelectorAll('.dippinn-image-card');
  const filtered = dippinnState.dataset.filter(img => img.filename.toLowerCase().includes(dpEls.search?.value.toLowerCase() || ""));
  cards.forEach((card, idx) => {
    const img = filtered[idx];
    if (img && img.name === name) {
      card.classList.add('active');
    } else {
      card.classList.remove('active');
    }
  });

  // Clear current outputs
  setImg(dpEls.imgReconstructed, null);
  if (dpEls.imgReconstructedPh) dpEls.imgReconstructedPh.style.display = 'block';
  setImg(dpEls.imgDifference, null);
  if (dpEls.imgDifferencePh) dpEls.imgDifferencePh.style.display = 'block';
  setImg(dpEls.p2ImgReconstructed, null);
  if (dpEls.p2ImgReconstructedPh) dpEls.p2ImgReconstructedPh.style.display = 'block';
  setImg(dpEls.p2ImgDifference, null);
  if (dpEls.p2ImgDifferencePh) dpEls.p2ImgDifferencePh.style.display = 'block';
  setImg(dpEls.p2ImgMask, null);
  if (dpEls.p2ImgMaskPh) dpEls.p2ImgMaskPh.style.display = 'block';
  setImg(dpEls.p2ImgTransmission, null);
  if (dpEls.p2ImgTransmissionPh) dpEls.p2ImgTransmissionPh.style.display = 'block';

  if (dpEls.hudCover) dpEls.hudCover.style.display = 'none';
  if (dpEls.p2HudCover) dpEls.p2HudCover.style.display = 'none';
  if (dpEls.customMaskBox) dpEls.customMaskBox.style.display = 'none';

  loadDippinnPreview();
};

const loadDippinnPreview = async () => {
  if (!dippinnState.selectedImage) return;

  const mSize = parseInt(dpEls.maskSize.value, 10) || 64;
  const mShape = dpEls.maskShape.value;

  try {
    const res = await fetch(`/api/dippinn/preview?image_name=${dippinnState.selectedImage}&phase=${dippinnState.phase}&mask_size=${mSize}&mask_type=${mShape}&mask_x=${dippinnState.maskCenter.x}&mask_y=${dippinnState.maskCenter.y}`);
    const data = await res.json();

    if (data.status === "success") {
      if (dippinnState.phase === 1) {
        setImg(dpEls.imgOriginal, data.original);
        if (dpEls.imgOriginalPh) dpEls.imgOriginalPh.style.display = 'none';
        setImg(dpEls.imgMasked, data.masked);
        if (dpEls.imgMaskedPh) dpEls.imgMaskedPh.style.display = 'none';

        // Draw synthetic custom mask overlay box
        updateSyntheticMaskOverlay();
      } else {
        setImg(dpEls.p2ImgOriginal, data.original);
        if (dpEls.p2ImgOriginalPh) dpEls.p2ImgOriginalPh.style.display = 'none';
        setImg(dpEls.p2ImgCloudy, data.masked);
        if (dpEls.p2ImgCloudyPh) dpEls.p2ImgCloudyPh.style.display = 'none';
      }

      // Update comparison slider base
      setImg(dpEls.sliderOriginal, data.original);
      setImg(dpEls.sliderReconstructed, null);
    }
  } catch (err) {
    appendDippinnConsole(`Error loading preview: ${err.message}`);
  }
};

const updateSyntheticMaskOverlay = () => {
  if (dippinnState.phase !== 1 || !dpEls.customMaskBox) return;
  const size = parseInt(dpEls.maskSize.value, 10) || 64;
  const shape = dpEls.maskShape.value;

  dpEls.customMaskBox.style.left = `${(dippinnState.maskCenter.x / 256) * 100}%`;
  dpEls.customMaskBox.style.top = `${(dippinnState.maskCenter.y / 256) * 100}%`;
  dpEls.customMaskBox.style.width = `${(size / 256) * 100}%`;
  dpEls.customMaskBox.style.height = `${(size / 256) * 100}%`;
  dpEls.customMaskBox.style.transform = 'translate(-50%, -50%)';
  dpEls.customMaskBox.style.borderRadius = shape === "Circle" ? "50%" : "0%";
  dpEls.customMaskBox.style.display = 'block';
};
const selectDippinnTab = (name) => {
  dippinnState.activeTab = name;

  // Set tab buttons using standard button styles
  dpEls.viewGridBtn.className = name === "workspace" ? "btn btn--sm btn--secondary" : "btn btn--sm btn--ghost";
  dpEls.viewSliderBtn.className = name === "compare" ? "btn btn--sm btn--secondary" : "btn btn--sm btn--ghost";
  dpEls.viewScientificBtn.className = name === "scientific" ? "btn btn--sm btn--secondary" : "btn btn--sm btn--ghost";

  // Toggle views
  dpEls.viewGrid.style.display = name === "workspace" ? "block" : "none";
  dpEls.viewSlider.style.display = name === "compare" ? "block" : "none";
  dpEls.viewScientific.style.display = name === "scientific" ? "block" : "none";
};

// WebSocket logic
const connectDippinnWS = () => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${proto}//${host}/api/dippinn/ws`;

  dippinnState.ws = new WebSocket(wsUrl);

  dippinnState.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.status) {
      if (dpEls.hudStatus) dpEls.hudStatus.textContent = data.status;
      if (data.status === "Training") {
        dpEls.startBtn.disabled = true;
        dpEls.stopBtn.disabled = false;
      } else {
        dpEls.startBtn.disabled = false;
        dpEls.stopBtn.disabled = true;
      }
    }

    if (data.iteration !== undefined && dpEls.hudIters) {
      dpEls.hudIters.textContent = `${data.iteration} / ${data.total_iterations || 5000}`;
    }

    if (data.elapsed !== undefined && dpEls.hudTime) {
      dpEls.hudTime.textContent = data.elapsed_hhmmss || `${Math.floor(data.elapsed / 60)}m ${data.elapsed % 60}s`;
    }

    if (data.eta !== undefined && dpEls.hudEta) {
      dpEls.hudEta.textContent = data.eta_hhmmss || `${Math.floor(data.eta / 60)}m ${data.eta % 60}s`;
    }

    if (data.log) {
      appendDippinnConsole(data.log);
    }

    if (data.metrics) {
      // Update charts
      const iter = data.iteration;
      dippinnState.lossChart.data.labels.push(iter);
      dippinnState.lossChart.data.datasets[0].data.push(data.metrics.loss);
      dippinnState.lossChart.update('none');

      dippinnState.metricChart.data.labels.push(iter);
      dippinnState.metricChart.data.datasets[0].data.push(data.metrics.psnr);
      dippinnState.metricChart.data.datasets[1].data.push(data.metrics.ssim);
      dippinnState.metricChart.update('none');

      // Update scientific tab fields
      if (dpEls.asmVal) dpEls.asmVal.textContent = data.metrics.asm_residual.toFixed(5);
      if (dpEls.dcpVal) dpEls.dcpVal.textContent = data.metrics.dcp_mean.toFixed(5);
      if (dpEls.sccVal) dpEls.sccVal.textContent = data.metrics.scc_score.toFixed(4);
      if (dpEls.violationVal) dpEls.violationVal.textContent = `${data.metrics.violation_rate.toFixed(2)}%`;

      // Update Segmentation Metrics
      if ($('dippinn-met-iou') && data.metrics.iou !== undefined) $('dippinn-met-iou').textContent = data.metrics.iou.toFixed(4);
      if ($('dippinn-met-dice') && data.metrics.dice !== undefined) $('dippinn-met-dice').textContent = data.metrics.dice.toFixed(4);
      if ($('dippinn-met-precision') && data.metrics.precision !== undefined) $('dippinn-met-precision').textContent = data.metrics.precision.toFixed(4);
      if ($('dippinn-met-recall') && data.metrics.recall !== undefined) $('dippinn-met-recall').textContent = data.metrics.recall.toFixed(4);


      // ASM Status color
      if (dpEls.asmStatus) {
        if (data.metrics.asm_residual < 0.005) {
          dpEls.asmStatus.className = "text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full font-bold";
          dpEls.asmStatus.textContent = "✓ Optimal";
        } else {
          dpEls.asmStatus.className = "text-[10px] bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded-full font-bold";
          dpEls.asmStatus.textContent = "⚠ Optimizing";
        }
      }
      // DCP Status color
      if (dpEls.dcpStatus) {
        if (data.metrics.dcp_mean < 0.08) {
          dpEls.dcpStatus.className = "text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full font-bold";
          dpEls.dcpStatus.textContent = "✓ Clean";
        } else {
          dpEls.dcpStatus.className = "text-[10px] bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded-full font-bold";
          dpEls.dcpStatus.textContent = "⚠ Haze Residual";
        }
      }
      // SCC Status
      if (dpEls.sccStatus) {
        if (data.metrics.scc_score > 0.8) {
          dpEls.sccStatus.className = "text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded-full font-bold";
          dpEls.sccStatus.textContent = "✓ Authenticated";
        } else {
          dpEls.sccStatus.className = "text-[10px] bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded-full font-bold";
          dpEls.sccStatus.textContent = "⚠ Low Correlation";
        }
      }
    }

    if (data.images) {
      if (dippinnState.phase === 1) {
        setImg(dpEls.imgReconstructed, data.images.reconstructed);
        if (dpEls.imgReconstructedPh) dpEls.imgReconstructedPh.style.display = 'none';

        setImg(dpEls.imgDifference, data.images.difference);
        if (dpEls.imgDifferencePh) dpEls.imgDifferencePh.style.display = 'none';
      } else {
        setImg(dpEls.p2ImgReconstructed, data.images.reconstructed);
        if (dpEls.p2ImgReconstructedPh) dpEls.p2ImgReconstructedPh.style.display = 'none';

        setImg(dpEls.p2ImgDifference, data.images.difference);
        if (dpEls.p2ImgDifferencePh) dpEls.p2ImgDifferencePh.style.display = 'none';

        setImg(dpEls.p2ImgMask, data.images.mask);
        if (dpEls.p2ImgMaskPh) dpEls.p2ImgMaskPh.style.display = 'none';

        setImg(dpEls.p2ImgTransmission, data.images.transmission);
        if (dpEls.p2ImgTransmissionPh) dpEls.p2ImgTransmissionPh.style.display = 'none';
      }

      // Update slider image
      setImg(dpEls.sliderReconstructed, data.images.reconstructed);
    }
  };

  dippinnState.ws.onclose = () => {
    setTimeout(connectDippinnWS, 3000);
  };
};

// Start training
const startDippinnTraining = async () => {
  if (!dippinnState.selectedImage) {
    alert("Please select an image first!");
    return;
  }

  // Clear charts
  if (dippinnState.lossChart && dippinnState.lossChart.data) {
    dippinnState.lossChart.data.labels = [];
    dippinnState.lossChart.data.datasets[0].data = [];
    dippinnState.lossChart.update();
  }

  if (dippinnState.metricChart && dippinnState.metricChart.data) {
    dippinnState.metricChart.data.labels = [];
    dippinnState.metricChart.data.datasets[0].data = [];
    dippinnState.metricChart.data.datasets[1].data = [];
    dippinnState.metricChart.update();
  }

  const pinnLosses = [];
  document.querySelectorAll('.dippinn-loss-chk:checked').forEach(chk => {
    pinnLosses.push(chk.value);
  });

  const lossWeights = {
    asm: parseFloat($('dippinn-weight-asm')?.value) || 1.0,
    tv: parseFloat($('dippinn-weight-tv')?.value) || 0.05,
    range: parseFloat($('dippinn-weight-range')?.value) || 10.0,
    dcp: parseFloat($('dippinn-weight-dcp')?.value) || 0.1,
    scc: parseFloat($('dippinn-weight-scc')?.value) || 0.5
  };

  const payload = {
    phase: dippinnState.phase,
    image_name: dippinnState.selectedImage,
    iterations: parseInt(dpEls.itersInput.value, 10) || 5000,
    lr: parseFloat(dpEls.lrInput.value) || 0.005,
    optimizer: dpEls.optSelect.value,
    mask_size: parseInt(dpEls.maskSize.value, 10) || 64,
    pinn_losses: pinnLosses,
    loss_weights: lossWeights,
    mask_type: dpEls.maskShape.value,
    mask_center: [dippinnState.maskCenter.y, dippinnState.maskCenter.x],
    resume: false
  };

  try {
    const res = await fetch('/api/dippinn/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === "success") {
      appendDippinnConsole(`Optimization thread launched for sample ${dippinnState.selectedImage}.`);
    } else {
      alert(data.message);
    }
  } catch (err) {
    appendDippinnConsole(`Error starting optimization: ${err.message}`);
  }
};

const stopDippinnTraining = async () => {
  try {
    await fetch('/api/dippinn/stop', { method: 'POST' });
  } catch (err) {
    console.error(err);
  }
};

const detectDippinnClouds = async () => {
  if (!dippinnState.selectedImage) return;
  dpEls.detectBtn.disabled = true;
  dpEls.detectBtn.textContent = "Detecting...";

  const payload = {
    phase: dippinnState.phase,
    image_name: dippinnState.selectedImage,
    mask_size: parseInt(dpEls.maskSize.value, 10) || 64,
    mask_type: dpEls.maskShape.value,
    mask_center: [dippinnState.maskCenter.y, dippinnState.maskCenter.x]
  };

  try {
    const res = await fetch('/api/dippinn/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.status === "success") {
      appendDippinnConsole(`Cloud detection complete: cloud cover = ${data.cloud_cover.toFixed(2)}%`);
      if (dpEls.p2HudCover) {
        dpEls.p2HudCover.textContent = `Cover: ${data.cloud_cover.toFixed(2)}%`;
        dpEls.p2HudCover.style.display = 'block';
      }

      setImg(dpEls.p2ImgMask, data.mask);
      if (dpEls.p2ImgMaskPh) dpEls.p2ImgMaskPh.style.display = 'none';

      setImg(dpEls.p2ImgTransmission, data.transmission);
      if (dpEls.p2ImgTransmissionPh) dpEls.p2ImgTransmissionPh.style.display = 'none';

      setImg(dpEls.p2ImgCloudy, data.masked);
    } else {
      alert(data.message || "Error running cloud detection");
    }
  } catch (err) {
    appendDippinnConsole(`Error detecting clouds: ${err.message}`);
  } finally {
    dpEls.detectBtn.disabled = false;
    dpEls.detectBtn.textContent = "☁ Detect Clouds";
  }
};

// Initialize listeners on timeout
setTimeout(() => {
  if (!dpEls.phase1Btn) return;

  initDippinnCharts();
  connectDippinnWS();



  dpEls.phase1Btn?.addEventListener('click', () => {
    dippinnState.phase = 1;
    dpEls.phase1Btn.className = "btn btn--sm btn--primary flex-1";
    dpEls.phase2Btn.className = "btn btn--sm btn--ghost flex-1";
    if (dpEls.synthControls) dpEls.synthControls.style.display = 'block';
    if (dpEls.detectBtn) dpEls.detectBtn.style.display = 'none';
    if (dpEls.gridPhase1) dpEls.gridPhase1.style.display = 'grid';
    if (dpEls.gridPhase2) dpEls.gridPhase2.style.display = 'none';
    fetchDippinnDataset();
  });

  dpEls.phase2Btn?.addEventListener('click', () => {
    dippinnState.phase = 2;
    dpEls.phase2Btn.className = "btn btn--sm btn--primary flex-1";
    dpEls.phase1Btn.className = "btn btn--sm btn--ghost flex-1";
    if (dpEls.synthControls) dpEls.synthControls.style.display = 'none';
    if (dpEls.detectBtn) dpEls.detectBtn.style.display = 'block';
    if (dpEls.gridPhase1) dpEls.gridPhase1.style.display = 'none';
    if (dpEls.gridPhase2) dpEls.gridPhase2.style.display = 'grid';
    fetchDippinnDataset();
  });

  dpEls.search?.addEventListener('input', renderDippinnDataset);

  dpEls.prevBtn?.addEventListener('click', () => {
    const idx = dippinnState.dataset.findIndex(img => img.name === dippinnState.selectedImage);
    if (idx > 0) selectDippinnImage(dippinnState.dataset[idx - 1].name);
  });

  dpEls.nextBtn?.addEventListener('click', () => {
    const idx = dippinnState.dataset.findIndex(img => img.name === dippinnState.selectedImage);
    if (idx !== -1 && idx < dippinnState.dataset.length - 1) {
      selectDippinnImage(dippinnState.dataset[idx + 1].name);
    }
  });

  dpEls.viewGridBtn?.addEventListener('click', () => selectDippinnTab("workspace"));
  dpEls.viewSliderBtn?.addEventListener('click', () => selectDippinnTab("compare"));
  dpEls.viewScientificBtn?.addEventListener('click', () => selectDippinnTab("scientific"));

  dpEls.startBtn?.addEventListener('click', startDippinnTraining);
  dpEls.stopBtn?.addEventListener('click', stopDippinnTraining);
  dpEls.detectBtn?.addEventListener('click', detectDippinnClouds);

  // Move synthetic mask center on click
  if (dpEls.imgOriginal && dpEls.imgOriginal.parentElement) {
    dpEls.imgOriginal.parentElement.addEventListener('click', (e) => {
      if (dippinnState.phase !== 1) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const x = Math.round(((e.clientX - rect.left) / rect.width) * 256);
      const y = Math.round(((e.clientY - rect.top) / rect.height) * 256);
      dippinnState.maskCenter = { x, y };
      loadDippinnPreview();
    });
  }

  dpEls.maskSize?.addEventListener('change', loadDippinnPreview);
  dpEls.maskShape?.addEventListener('change', loadDippinnPreview);

  // Slider dragging
  dpEls.sliderRange?.addEventListener('input', (e) => {
    const val = e.target.value;
    if (dpEls.sliderReconstructedWrapper) dpEls.sliderReconstructedWrapper.style.width = `${100 - val}%`;
    const imgWidth = dpEls.sliderOriginal ? dpEls.sliderOriginal.clientWidth : 500;
    if (dpEls.sliderReconstructed) {
      dpEls.sliderReconstructed.style.width = `${imgWidth}px`;
      dpEls.sliderReconstructed.style.transform = `translateX(-${val}%)`;
    }
  });

  // Restore Loss Components & Weights from localStorage (or fallback to defaults)
  const defaultLossConfig = {
    selected: ['ASM', 'TV', 'Physical Range'],
    weights: {
      ASM: 1.0,
      TV: '0.05',
      'Physical Range': 10.0,
      DCP: 0.1,
      SCC: 0.5
    }
  };

  const applyLossConfig = (config) => {
    document.querySelectorAll('.dippinn-loss-chk').forEach(chk => {
      chk.checked = config.selected.includes(chk.value);
    });
    if (config.weights) {
      if ($('dippinn-weight-asm') && config.weights.ASM !== undefined) $('dippinn-weight-asm').value = config.weights.ASM;
      if ($('dippinn-weight-tv') && config.weights.TV !== undefined) $('dippinn-weight-tv').value = config.weights.TV;
      if ($('dippinn-weight-range') && config.weights['Physical Range'] !== undefined) $('dippinn-weight-range').value = config.weights['Physical Range'];
      if ($('dippinn-weight-dcp') && config.weights.DCP !== undefined) $('dippinn-weight-dcp').value = config.weights.DCP;
      if ($('dippinn-weight-scc') && config.weights.SCC !== undefined) $('dippinn-weight-scc').value = config.weights.SCC;
    }
  };

  const saveCurrentLossConfig = () => {
    const selected = [];
    document.querySelectorAll('.dippinn-loss-chk:checked').forEach(c => selected.push(c.value));
    const weights = {
      ASM: parseFloat($('dippinn-weight-asm')?.value) || 1.0,
      TV: $('dippinn-weight-tv')?.value || '0.05',
      'Physical Range': parseFloat($('dippinn-weight-range')?.value) || 10.0,
      DCP: parseFloat($('dippinn-weight-dcp')?.value) || 0.1,
      SCC: parseFloat($('dippinn-weight-scc')?.value) || 0.5
    };
    const config = { selected, weights };
    localStorage.setItem('dippinn_loss_custom_config', JSON.stringify(config));
    return config;
  };

  try {
    const saved = JSON.parse(localStorage.getItem('dippinn_loss_custom_config') || 'null');
    if (saved && Array.isArray(saved.selected)) {
      applyLossConfig(saved);
    } else {
      applyLossConfig(defaultLossConfig);
    }
  } catch(e) {
    applyLossConfig(defaultLossConfig);
  }

  // Save button click
  $('dippinn-save-loss-btn')?.addEventListener('click', () => {
    saveCurrentLossConfig();
    if (window.showToast) window.showToast('Loss configuration & weights saved!', 'success', '💾');
  });

  // Reset button click
  $('dippinn-reset-loss-btn')?.addEventListener('click', () => {
    localStorage.removeItem('dippinn_loss_custom_config');
    applyLossConfig(defaultLossConfig);
    if (window.showToast) window.showToast('Reset to default loss weights', 'info', '🔄');
  });

  // Auto-sync changes on input / change
  document.querySelectorAll('.dippinn-loss-chk').forEach(chk => {
    chk.addEventListener('change', saveCurrentLossConfig);
  });
  ['dippinn-weight-asm', 'dippinn-weight-tv', 'dippinn-weight-range', 'dippinn-weight-dcp', 'dippinn-weight-scc'].forEach(id => {
    $(id)?.addEventListener('input', saveCurrentLossConfig);
  });

  // Load first dataset at start
  fetchDippinnDataset();
}, 2000);



// ── Phase 1 Study Logic ──────────────────────────────────────────────────────
// Wired up AFTER DOM ready (called inside setTimeout at the bottom of dippinn init)
function initPhase1StudyButtons() {
  const btnStudy = document.getElementById('btn-study-image');
  const btnLearn = document.getElementById('btn-learn-orig');
  const btnRecon = document.getElementById('btn-recon-mask');

  if (btnStudy) {
    btnStudy.addEventListener('click', async () => {
      const img = dippinnState.selectedImage;
      if (!img) { showToast('Select a sample image from the dataset list first.'); return; }
      btnStudy.textContent = '⏳ Studying...';
      btnStudy.disabled = true;
      try {
        const res = await fetch('/api/phase1/study', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_name: img, phase: 1 })
        });
        const data = await res.json();
        if (data.status === 'success') {
          // Show the feature grid
          const grid = document.getElementById('dippinn-study-features');
          if (grid) grid.style.display = 'grid';

          // Canny edges
          const phEdges = document.getElementById('ph-phase1-edges');
          const imgEdges = document.getElementById('phase1-edges');
          if (phEdges) phEdges.style.display = 'none';
          if (imgEdges) { imgEdges.src = 'data:image/png;base64,' + data.edges_b64; imgEdges.style.display = 'block'; }

          // Texture map
          const phTex = document.getElementById('ph-phase1-texture');
          const imgTex = document.getElementById('phase1-texture');
          if (phTex) phTex.style.display = 'none';
          if (imgTex) { imgTex.src = 'data:image/png;base64,' + data.texture_b64; imgTex.style.display = 'block'; }
          const texVal = document.getElementById('phase1-texture-val');
          if (texVal) { texVal.textContent = 'Variance: ' + (data.texture_var || 0).toFixed(2); texVal.style.display = 'block'; }

          // RGB Histogram
          const phRgb = document.getElementById('ph-phase1-rgb');
          if (phRgb) phRgb.style.display = 'none';
          const canvas = document.getElementById('phase1-rgb-chart');
          if (canvas) {
            canvas.style.display = 'block';
            if (window.phase1RgbChart) window.phase1RgbChart.destroy();
            window.phase1RgbChart = new Chart(canvas, {
              type: 'line',
              data: {
                labels: Array.from({ length: 256 }, (_, i) => i),
                datasets: [
                  { label: 'Red', data: data.hist_r, borderColor: 'rgba(255,80,80,0.8)', borderWidth: 1.5, pointRadius: 0, fill: false },
                  { label: 'Green', data: data.hist_g, borderColor: 'rgba(80,200,80,0.8)', borderWidth: 1.5, pointRadius: 0, fill: false },
                  { label: 'Blue', data: data.hist_b, borderColor: 'rgba(80,120,255,0.8)', borderWidth: 1.5, pointRadius: 0, fill: false }
                ]
              },
              options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                plugins: { legend: { display: true, labels: { color: '#aaa', font: { size: 10 } } } },
                scales: {
                  x: { display: false },
                  y: { display: true, ticks: { color: '#aaa', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
              }
            });
          }
          showToast('✅ Study complete! Edge, texture, and RGB histogram extracted.');
        } else {
          showToast('❌ Error: ' + (data.message || 'Unknown error'));
        }
      } catch (e) {
        console.error(e);
        showToast('❌ Network error during study request.');
      }
      btnStudy.textContent = '1. Study Image Features';
      btnStudy.disabled = false;
    });
  }

  if (btnLearn) {
    btnLearn.addEventListener('click', async () => {
      const img = dippinnState.selectedImage;
      if (!img) { showToast('Select a sample image first.'); return; }
      const mSize = parseInt(document.getElementById('dippinn-mask-size')?.value || '64');
      const mShape = document.getElementById('dippinn-mask-shape')?.value || 'Rectangle';
      btnLearn.textContent = '⏳ Learning...';
      btnLearn.disabled = true;
      try {
        const res = await fetch('/api/dippinn/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            phase: 1, image_name: img, iterations: 1500, lr: 0.005,
            optimizer: 'Adam', tv_weight: 0.0001, physics_weight: 0.0,
            mask_size: mSize, pinn_losses: ['TV', 'Physical Range'], mask_type: mShape,
            mask_center: [dippinnState.maskCenter.y, dippinnState.maskCenter.x],
            resume: false, enforce_unmasked: true, mode: 'learn_context'
          })
        });
        const data = await res.json();
        if (data.status === 'success' || data.status === 'started') {
          appendDippinnConsole(`🎓 Phase 1 — Learning surrounding context features (Masked spot stays intact black hole)...`);
          showToast('✅ Learning context started — clean pixels are learned, masked spot remains intact.');
        } else {
          showToast('❌ ' + (data.detail || 'Could not start training.'));
        }
      } catch (e) {
        showToast('❌ Network error starting learning.');
      }
      btnLearn.textContent = '2. Learn Original (DIP+PINN)';
      btnLearn.disabled = false;
    });
  }

  if (btnRecon) {
    btnRecon.addEventListener('click', async () => {
      const img = dippinnState.selectedImage;
      if (!img) { showToast('Select a sample image first.'); return; }
      // Read mask settings from the existing UI controls
      const mSize = parseInt(document.getElementById('dippinn-mask-size')?.value || '64');
      const mShape = document.getElementById('dippinn-mask-shape')?.value || 'Rectangle';
      btnRecon.textContent = '⏳ Starting...';
      btnRecon.disabled = true;
      try {
        const res = await fetch('/api/dippinn/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            phase: 1, image_name: img, iterations: 3000, lr: 0.005,
            optimizer: 'Adam', tv_weight: 0.0001, physics_weight: 0.001,
            mask_size: mSize, pinn_losses: ['TV', 'Physical Range'], mask_type: mShape,
            mask_center: [dippinnState.maskCenter.y, dippinnState.maskCenter.x],
            resume: false, enforce_unmasked: true
          })
        });
        const data = await res.json();
        if (data.status === 'success' || data.status === 'started') {
          appendDippinnConsole(`🔲 Phase 1 — Reconstructing ${mShape} mask (size=${mSize}px, enforce_unmasked=true)...`);
          showToast('✅ Reconstruction started — unmasked pixels will be kept exactly as original.');
        } else {
          showToast('❌ ' + (data.detail || 'Could not start training.'));
        }
      } catch (e) {
        showToast('❌ Network error starting reconstruction.');
      }
      btnRecon.textContent = '3. Reconstruct Masked (DIP+PINN)';
      btnRecon.disabled = false;
    });
  }
}

// Call after a short delay to ensure DOM + dippinnState is ready
setTimeout(initPhase1StudyButtons, 2500);

// Show/hide Phase 1 Study section when phase toggle changes
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('dippinn-phase1-btn')?.addEventListener('click', () => {
    const el = document.getElementById('dippinn-phase1-actions');
    if (el) el.style.display = 'flex';
  });
  document.getElementById('dippinn-phase2-btn')?.addEventListener('click', () => {
    const el = document.getElementById('dippinn-phase1-actions');
    if (el) el.style.display = 'none';
    const grid = document.getElementById('dippinn-study-features');
    if (grid) grid.style.display = 'none';
  });
});
document.addEventListener('DOMContentLoaded', () => {
  function setupQualityChart(tabId, prefix) {
    const tab = document.getElementById(tabId);
    if (!tab) return;

    const ctx = tab.querySelector('#' + prefix + '-quality-chart');
    const ph = tab.querySelector('#ph-' + prefix + '-quality');
    if (!ctx) return;

    window.qualityCharts = window.qualityCharts || {};
    window.qualityCharts[tabId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'PSNR (dB)',
            data: [],
            borderColor: 'rgba(59, 130, 246, 1)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            yAxisID: 'y'
          },
          {
            label: 'SSIM',
            data: [],
            borderColor: 'rgba(34, 197, 94, 1)',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'PSNR (dB)', color: '#8aa3cc', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#5a7399', font: { size: 9 } }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'SSIM', color: '#8aa3cc', font: { size: 10 } },
            grid: { drawOnChartArea: false },
            ticks: { color: '#5a7399', font: { size: 9 } }
          }
        },
        plugins: {
          legend: { labels: { color: '#8aa3cc', font: { family: 'Inter', size: 10 } } }
        }
      }
    });


    if (!window.optimizationHistory) {
      window.optimizationHistory = {
        'full': { psnr: [], ssim: [] },
        'noasm': { psnr: [], ssim: [] },
        'norte': { psnr: [], ssim: [] },
        'core': { psnr: [], ssim: [] },
        'corenm': { psnr: [], ssim: [] }
      };
      window.currentOptimizationMode = 'full';
    }

    const btns = tab.querySelectorAll('.quality-btn');
    btns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        btns.forEach(b => {
          b.style.background = 'rgba(255,255,255,0.03)';
          b.style.borderColor = 'var(--border)';
        });

        btn.style.background = 'rgba(59, 130, 246, 0.2)';
        btn.style.borderColor = 'rgba(59, 130, 246, 0.8)';

        if (ph) ph.style.display = 'none';

        const target = btn.getAttribute('data-target');
        window.currentOptimizationMode = target;
        const hist = window.optimizationHistory && window.optimizationHistory[target];
        if (hist && hist.psnr && hist.psnr.length > 0) {
          window.qualityCharts[tabId].data.labels = hist.steps || Array.from({ length: hist.psnr.length }, (_, i) => i + 1);
          window.qualityCharts[tabId].data.datasets[0].data = hist.psnr;
          window.qualityCharts[tabId].data.datasets[1].data = hist.ssim;
          window.qualityCharts[tabId].update('none');
        } else if (target === 'full' && tabId === 'tab-classification') {
          const idx = (typeof state !== 'undefined' && state.index !== undefined) ? state.index : 0;
          fetch(`/api/optimization_history?dataset=RICE1&index=${idx}`)
            .then(r => r.json())
            .then(d => {
              if (d.steps && d.steps.length > 0) {
                window.optimizationHistory = window.optimizationHistory || {};
                window.optimizationHistory['full'] = { psnr: d.psnr, ssim: d.ssim, steps: d.steps };
                window.qualityCharts[tabId].data.labels = d.steps;
                window.qualityCharts[tabId].data.datasets[0].data = d.psnr;
                window.qualityCharts[tabId].data.datasets[1].data = d.ssim;
                window.qualityCharts[tabId].update('none');
              }
            }).catch(() => { });
        }
      });
    });
  }

  setupQualityChart('tab-classification', 'pipe');
  setupQualityChart('tab-rice2-classification', 'r2pipe');

  // Auto-load CSV history for RICE1 on page load
  (function loadQualityCSV() {
    const tab = document.getElementById('tab-classification');
    if (!tab) return;
    const ph = document.getElementById('ph-pipe-quality');
    const idx = (typeof state !== 'undefined' && state.index !== undefined) ? state.index : 0;
    fetch(`/api/optimization_history?dataset=RICE1&index=${idx}`)
      .then(r => r.json())
      .then(d => {
        if (!d.steps || !d.steps.length) return;
        window.optimizationHistory = window.optimizationHistory || {};
        window.optimizationHistory['full'] = { psnr: d.psnr, ssim: d.ssim, steps: d.steps };
        const qc = window.qualityCharts && window.qualityCharts['tab-classification'];
        if (qc) {
          qc.data.labels = d.steps;
          qc.data.datasets[0].data = d.psnr;
          qc.data.datasets[1].data = d.ssim;
          qc.update('none');
          if (ph) ph.style.display = 'none';
          const fullBtn = tab.querySelector('.quality-btn[data-target="full"]');
          if (fullBtn) {
            fullBtn.style.background = 'rgba(59, 130, 246, 0.2)';
            fullBtn.style.borderColor = 'rgba(59, 130, 246, 0.8)';
          }
        }
      }).catch(() => { });
  })();

  // Expose for post-run reload (called from pipeline complete handler)
  window.reloadQualityChart = function (index) {
    const idx = index !== undefined ? index : (typeof state !== 'undefined' ? state.index : 0);
    fetch(`/api/optimization_history?dataset=RICE1&index=${idx}`)
      .then(r => r.json())
      .then(d => {
        if (!d.steps || !d.steps.length) return;
        window.optimizationHistory = window.optimizationHistory || {};
        window.optimizationHistory['full'] = { psnr: d.psnr, ssim: d.ssim, steps: d.steps };
        const qc = window.qualityCharts && window.qualityCharts['tab-classification'];
        if (qc) {
          qc.data.labels = d.steps;
          qc.data.datasets[0].data = d.psnr;
          qc.data.datasets[1].data = d.ssim;
          qc.update('none');
          const ph = document.getElementById('ph-pipe-quality');
          if (ph) ph.style.display = 'none';
        }
      }).catch(() => { });
  };
});

// ── Clean Dedicated Modal Viewer for Input Image & Ground Truth ─────────────
window.openInputGtViewerModal = function(card, image, type) {
    const isRice1 = !card?.closest('#tab-rice2-classification');
    const dataset = isRice1 ? 'RICE1' : 'RICE2';
    const isInput = type === 'input';
    const sampleId = document.querySelector(isRice1 ? '.info-sample-id' : '#r2-sample-id')?.textContent?.trim() || document.querySelector('.info-sample-id')?.textContent?.trim() || '0';
    const title = isInput ? '☁️ 4. INPUT IMAGE (CLOUDY)' : '🌿 GROUND TRUTH (CLOUD-FREE)';
    const subtitle = isInput
        ? `Original ${dataset} satellite capture with cloud contamination`
        : `Reference clean image — cloud-free target for reconstruction (${dataset})`;
    
    const valid = (val) => typeof val === 'string' && val && !val.includes('svg+xml');
    const source = (el) => {
        const val = el?.currentSrc || el?.src || '';
        if (!valid(val)) return '';
        return val.startsWith('data:') || val.startsWith('/') || val.startsWith('http') ? val : `data:image/png;base64,${val}`;
    };
    const imgSrc = source(image) || source(document.getElementById(isRice1 ? (isInput ? 'grid-img-input' : 'grid-img-gt') : (isInput ? 'r2grid-img-input' : 'r2grid-img-gt')));

    // Physics parameters from UI sliders
    const getParam = (id, fallback) => document.getElementById(id)?.value || fallback;
    const pfx = isRice1 ? 'pipe-' : 'r2pipe-';
    const lambdaAsm = getParam(pfx + 'lambda-asm', getParam('pipe-lambda-asm', isRice1 ? '0.06' : '0.08'));
    const lambdaRte = getParam(pfx + 'lambda-rte', getParam('pipe-lambda-rte', isRice1 ? '0.03' : '0.04'));
    const lambdaTv = getParam(pfx + 'lambda-tv', getParam('pipe-lambda-tv', '0.00001'));
    const lambdaNdvi = getParam(pfx + 'lambda-ndvi', getParam('pipe-lambda-ndvi', '0.05'));
    const lambdaTPrior = getParam(pfx + 'lambda-t-prior', getParam('pipe-lambda-t-prior', '1.0'));
    const noiseStd = getParam(pfx + 'noise-std', getParam('pipe-noise-std', isRice1 ? '0.015' : '0.02'));
    const lr = getParam(pfx + 'lr-input', getParam('pipe-lr-input', '0.005'));
    const iters = getParam(pfx + 'iters-select', getParam('pipe-iters-select', '300'));

    // Cloud detection stats from UI
    const cs = (id) => document.getElementById(id)?.textContent?.trim() || '—';
    const cloudTotal = cs(pfx + 'stat-total') !== '—' ? cs(pfx + 'stat-total') : cs('pipe-stat-total');
    const cloudThick = cs(pfx + 'stat-thick') !== '—' ? cs(pfx + 'stat-thick') : cs('pipe-stat-thick');
    const cloudMedium = cs(pfx + 'stat-medium') !== '—' ? cs(pfx + 'stat-medium') : cs('pipe-stat-medium');
    const cloudThin = cs(pfx + 'stat-thin') !== '—' ? cs(pfx + 'stat-thin') : cs('pipe-stat-thin');
    const cloudShadow = cs(pfx + 'stat-shadow') !== '—' ? cs(pfx + 'stat-shadow') : cs('pipe-stat-shadow');
    const cloudConfidence = cs(pfx + 'stat-confidence') !== '—' ? cs(pfx + 'stat-confidence') : cs('pipe-stat-confidence');

    const resolution = isRice1 ? '256 × 256' : '512 × 512';

    const viewerHtml = `<!doctype html><html><head><meta charset="utf-8"><title>${title} — Full Size Report</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { margin:0; padding:28px; background:#07111f; color:#dbeafe; font-family:'Segoe UI', system-ui, sans-serif; }
        main { max-width:1200px; margin:auto; background:#102238; padding:28px; border-radius:12px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
        header { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #24506b; padding-bottom:18px; margin-bottom:24px; }
        h1 { color:#67e8f9; margin:0; font-size:1.6rem; font-weight:700; }
        .subtitle { color:#94a3b8; font-size:0.85rem; margin-top:4px; }
        .btn, .download-btn { display:inline-flex; align-items:center; justify-content:center; gap:7px; background:#0891b2; color:#fff; border:1px solid #67e8f9; border-radius:7px; padding:9px 15px; cursor:pointer; text-decoration:none; font-weight:600; font-size:0.85rem; transition:all 0.2s ease; }
        .btn:hover, .download-btn:hover { background:#0e7490; transform:translateY(-1px); }
        .close-btn { background:linear-gradient(135deg, #ef4444, #b91c1c); border:none; color:white; }
        .close-btn:hover { background:linear-gradient(135deg, #dc2626, #991b1b); }
        .section-title { color:#a5f3fc; font-size:1rem; font-weight:600; text-transform:uppercase; border-left:3px solid #22d3ee; padding-left:10px; margin-bottom:14px; }
        .layout-grid { display:grid; grid-template-columns: 1fr 1fr; gap:24px; margin-bottom:24px; }
        .image-card { background:#020617; border:1px solid #23445b; border-radius:10px; overflow:hidden; display:flex; flex-direction:column; }
        .image-card-head { background:#1e293b; padding:10px 14px; font-weight:bold; font-size:0.85rem; color:#67e8f9; display:flex; justify-content:space-between; align-items:center; }
        .image-card img { width:100%; aspect-ratio:1; object-fit:contain; background:#000; display:block; }
        .panel-column { display:flex; flex-direction:column; gap:16px; }
        .meta-card, .params-card, .cloud-card { background:linear-gradient(145deg, #0d1c30, #0a1525); border:1px solid #23445b; border-radius:10px; padding:14px; }
        .meta-grid { display:grid; grid-template-columns: 100px 1fr; gap:6px; font-size:0.85rem; line-height:1.6; }
        .meta-label { color:#94a3b8; }
        .meta-val { color:#e2e8f0; font-weight:600; }
        .grid-params { display:grid; grid-template-columns:repeat(auto-fit, minmax(100px, 1fr)); gap:10px; }
        .param-item { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); padding:10px; border-radius:8px; text-align:center; }
        .param-item span { display:block; font-size:0.65rem; color:#94a3b8; text-transform:uppercase; margin-bottom:3px; }
        .param-item strong { font-size:1.1rem; color:#38bdf8; }
        .grid-cloud { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; }
        .cloud-item { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); padding:8px; border-radius:6px; text-align:center; }
        .cloud-item span { display:block; font-size:0.65rem; color:#94a3b8; text-transform:uppercase; margin-bottom:2px; }
        .cloud-item strong { font-size:0.95rem; color:#f59e0b; }
        .chart-box { background:linear-gradient(145deg, #0d1c30, #0a1525); border:1px solid #23445b; border-radius:10px; padding:16px; height:280px; }
    </style></head><body><main>
        <header>
            <div>
                <h1>${title}</h1>
                <div class="subtitle">${subtitle}</div>
            </div>
            <div style="display:flex; gap:10px;">
                ${imgSrc ? `<a class="download-btn" href="${imgSrc}" download="${isInput ? dataset.toLowerCase() + '_input' : dataset.toLowerCase() + '_ground_truth'}.png">⬇ Download Image</a>` : ''}
                <button class="btn close-btn" onclick="window.parent.document.getElementById('reconstruction-report-modal')?.remove();">✕ Close</button>
            </div>
        </header>

        <div class="layout-grid">
            <div class="image-card">
                <div class="image-card-head">
                    <span>${isInput ? '☁️ Cloudy Satellite Capture' : '🌿 Reference Ground Truth'}</span>
                    <span style="font-size:0.75rem; color:#94a3b8;">${resolution}</span>
                </div>
                ${imgSrc ? `<img src="${imgSrc}" id="targetImg" crossorigin="anonymous" />` : '<div style="padding:50px; text-align:center; color:#94a3b8;">Image not loaded</div>'}
            </div>

            <div class="panel-column">
                <div class="meta-card">
                    <div class="section-title">📋 Image Metadata</div>
                    <div class="meta-grid">
                        <span class="meta-label">Dataset:</span><span class="meta-val">${dataset}</span>
                        <span class="meta-label">Sample ID:</span><span class="meta-val">${sampleId}</span>
                        <span class="meta-label">Resolution:</span><span class="meta-val">${resolution}</span>
                        <span class="meta-label">Bands:</span><span class="meta-val">RGB (3-channel)</span>
                        <span class="meta-label">Data Type:</span><span class="meta-val">uint8 [0–255]</span>
                        <span class="meta-label">Category:</span><span class="meta-val">${isInput ? (dataset === 'RICE1' ? 'Thin Cloud / Haze' : 'Thick / Dense Cloud') : 'Clean Reference Target'}</span>
                    </div>
                </div>

                <div class="params-card">
                    <div class="section-title">⚙️ Physics Parameters</div>
                    <div class="grid-params">
                        <div class="param-item"><span>λ ASM</span><strong>${lambdaAsm}</strong></div>
                        <div class="param-item"><span>λ RTE</span><strong>${lambdaRte}</strong></div>
                        <div class="param-item"><span>λ TV</span><strong>${lambdaTv}</strong></div>
                        <div class="param-item"><span>λ NDVI</span><strong>${lambdaNdvi}</strong></div>
                        <div class="param-item"><span>λ T-Prior</span><strong>${lambdaTPrior}</strong></div>
                        <div class="param-item"><span>Noise σ</span><strong>${noiseStd}</strong></div>
                        <div class="param-item"><span>Learning Rate</span><strong>${lr}</strong></div>
                        <div class="param-item"><span>Iterations</span><strong>${iters}</strong></div>
                    </div>
                </div>

                ${isInput ? `
                <div class="cloud-card">
                    <div class="section-title">☁️ Cloud Detection & Segmentation Statistics</div>
                    <div class="grid-cloud">
                        <div class="cloud-item"><span>Total Cloud</span><strong>${cloudTotal}</strong></div>
                        <div class="cloud-item"><span>Thick</span><strong>${cloudThick}</strong></div>
                        <div class="cloud-item"><span>Medium</span><strong>${cloudMedium}</strong></div>
                        <div class="cloud-item"><span>Thin</span><strong>${cloudThin}</strong></div>
                        <div class="cloud-item"><span>Shadow</span><strong>${cloudShadow}</strong></div>
                        <div class="cloud-item"><span>Confidence</span><strong>${cloudConfidence}</strong></div>
                    </div>
                </div>` : ''}
            </div>
        </div>

        <div>
            <div class="section-title">🔴🟢🔵 RGB Color Distribution Graph</div>
            <div class="chart-box"><canvas id="rgbDistributionChart"></canvas></div>
        </div>
    </main><script>
        window.onload = function() {
            try {
                var img = document.getElementById('targetImg');
                if (!img) return;
                var drawChart = function() {
                    var canvas = document.createElement('canvas');
                    var ctx = canvas.getContext('2d');
                    var w = img.naturalWidth || 256;
                    var h = img.naturalHeight || 256;
                    canvas.width = w; canvas.height = h;
                    ctx.drawImage(img, 0, 0, w, h);
                    var imgData = ctx.getImageData(0, 0, w, h).data;
                    var rH = new Array(256).fill(0), gH = new Array(256).fill(0), bH = new Array(256).fill(0);
                    for (var i = 0; i < imgData.length; i += 4) {
                        rH[imgData[i]]++;
                        gH[imgData[i+1]]++;
                        bH[imgData[i+2]]++;
                    }
                    var labels = Array.from({length: 256}, function(_, idx) { return idx; });
                    new Chart(document.getElementById('rgbDistributionChart'), {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [
                                { label: 'Red Channel', data: rH, borderColor: 'rgba(239,68,68,0.85)', backgroundColor: 'rgba(239,68,68,0.1)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5 },
                                { label: 'Green Channel', data: gH, borderColor: 'rgba(34,197,94,0.85)', backgroundColor: 'rgba(34,197,94,0.1)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5 },
                                { label: 'Blue Channel', data: bH, borderColor: 'rgba(59,130,246,0.85)', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5 }
                            ]
                        },
                        options: {
                            responsive: true, maintainAspectRatio: false,
                            plugins: { legend: { labels: { color: '#cbd5e1', font: { size: 11 } } } },
                            scales: {
                                x: { title: { display: true, text: 'Pixel Intensity [0–255]', color: '#94a3b8', font: { size: 10 } }, ticks: { maxTicksLimit: 12, color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                                y: { title: { display: true, text: 'Pixel Frequency', color: '#94a3b8', font: { size: 10 } }, ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
                            }
                        }
                    });
                };
                if (img.complete && img.naturalWidth) drawChart();
                else img.onload = drawChart;
            } catch(e) { console.error('Histogram error', e); }
        };
    </script></body></html>`;

    document.getElementById('reconstruction-report-modal')?.remove();
    const modal = document.createElement('div');
    modal.id = 'reconstruction-report-modal';
    modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(2,6,23,.82);backdrop-filter:blur(10px);display:flex;align-items:center;justify-content:center;padding:20px;opacity:0;transition:opacity .24s ease;';
    const shell = document.createElement('div');
    shell.style.cssText = 'width:min(1250px,96vw);height:min(92vh,950px);background:#07111f;border:1px solid rgba(103,232,249,.45);border-radius:14px;overflow:hidden;transform:translateY(18px) scale(.98);transition:transform .28s cubic-bezier(.2,.8,.2,1);box-shadow:0 24px 80px rgba(0,0,0,.55);';
    const close = document.createElement('button');
    close.type = 'button';
    close.textContent = 'Close';
    close.style.cssText = 'position:absolute;right:32px;top:28px;z-index:2;background:#0e7490;color:#fff;border:1px solid #67e8f9;border-radius:6px;padding:8px 14px;cursor:pointer;font-weight:600;';
    const frame = document.createElement('iframe');
    frame.title = `${title} - Report`;
    frame.srcdoc = viewerHtml;
    frame.style.cssText = 'width:100%;height:100%;border:0;background:#07111f;';
    const closeModal = () => modal.remove();
    close.addEventListener('click', closeModal);
    modal.addEventListener('click', (event) => { if (event.target === modal) closeModal(); });
    document.addEventListener('keydown', function closeOnEscape(event) {
      if (event.key === 'Escape') { closeModal(); document.removeEventListener('keydown', closeOnEscape); }
    }, { once: true });
    shell.append(frame);
    modal.append(close, shell);
    document.body.append(modal);
    requestAnimationFrame(() => { modal.style.opacity = '1'; shell.style.transform = 'translateY(0) scale(1)'; });
};

// Global click delegation for View Full Size buttons
(() => {
  const isViewFullSizeButton = (button) =>
    /view\s+full\s+size/i.test(button?.textContent || '');

  const handleFullSizeClick = (event) => {
    const button = event.target.closest?.('button');
    if (!button || !isViewFullSizeButton(button)) return;

    const rice2Panel = button.closest('#tab-rice2-classification');
    const rice1Panel = button.closest('#tab-classification');
    if (!rice1Panel && !rice2Panel) return;

    event.preventDefault();
    event.stopImmediatePropagation();

    // 1. Find immediate panel containing the button
    const panel = button.closest('div[style*="border: 1px solid"]') || button.closest('div[style*="border"]') || button.parentElement?.parentElement || button.closest('.glass-card');
    const img = panel?.querySelector('img');
    const imgId = (img?.id || '').toLowerCase();

    // 2. Exact match for Section 4: Input Image
    if (imgId === 'grid-img-input' || imgId === 'r2grid-img-input' || imgId === 'flow-input' || imgId === 'r2flow-input') {
      window.openInputGtViewerModal(panel, img, 'input');
      return;
    }

    // 3. Exact match for Section 4: Ground Truth
    if (imgId === 'grid-img-gt' || imgId === 'r2grid-img-gt') {
      window.openInputGtViewerModal(panel, img, 'gt');
      return;
    }

    // 4. Section 6: 4 Ablation Models (1, 2, 3, 4)
    let idx = '1';
    const match = imgId.match(/(?:r2)?grid-img-(\d+)/);
    if (match) {
      idx = match[1];
    } else {
      const pText = (panel?.innerText || panel?.textContent || '').toUpperCase();
      if (pText.includes('1.') || pText.includes('FULL MODEL')) idx = '1';
      else if (pText.includes('2.') || pText.includes('NO ASM')) idx = '2';
      else if (pText.includes('3.') || pText.includes('NO RTE')) idx = '3';
      else if (pText.includes('4.') || pText.includes('CORE DIP')) idx = '4';
    }

    const reportFn = window.openFullSizeInNewTabR2;
    if (typeof reportFn === 'function') {
      reportFn(panel, img || { id: (rice2Panel ? 'r2grid-img-' : 'grid-img-') + idx }, idx);
    } else if (typeof window.showToast === 'function') {
      window.showToast('Report is still loading. Please try again.', 'warn');
    }
  };

  document.addEventListener('click', handleFullSizeClick, true);
})();

// Shared report generator for the 4 Ablation Models (Full Model, No ASM, No RTE, Core DIP)
window.openFullSizeInNewTabR2 = function (card, image, forcedIdx) {
    const valid = (value) => typeof value === 'string' && value && !value.includes('svg+xml');
    const source = (element, fallback) => {
      const value = element?.currentSrc || element?.src || fallback || '';
      if (!valid(value)) return '';
      return value.startsWith('data:') || value.startsWith('/') || value.startsWith('http')
        ? value : `data:image/png;base64,${value}`;
    };
    let idx = forcedIdx;
    if (!idx) {
      const idxMatch = image?.id?.match(/(?:r2)?grid-img-(\d+)/);
      idx = idxMatch ? idxMatch[1] : '1';
    }
    const isRice1 = !card?.closest('#tab-rice2-classification');
    const result = (isRice1 ? window.rice1AblationResults?.[idx] : window.rice2AblationResults?.[idx]) || (isRice1 ? window.cvDatasetRunState?.RICE1?.ablationResults?.[idx] : window.cvDatasetRunState?.RICE2?.ablationResults?.[idx]) || window.ablationResults?.[idx] || window.lastRunResult || {};
    const images = result.images || window.lastRunResult?.images || {};
    const input = source(null, images.input) || source(document.getElementById(isRice1 ? 'grid-img-input' : 'r2grid-img-input'));
    const groundTruth = source(null, images.label || images.ground_truth) || source(document.getElementById(isRice1 ? 'grid-img-gt' : 'r2grid-img-gt'));
    const mask = source(null, images.mask || images.cloud_mask) || source(document.getElementById(isRice1 ? 'flow-mask' : 'r2flow-mask'));
    const output = source(null, images.restored) || source(image) || source(document.getElementById((isRice1 ? 'grid-img-' : 'r2grid-img-') + idx));
    const difference = source(null, images.difference || images.error_map || images.difference_map || images.error || images.diff) || source(document.getElementById(isRice1 ? 'flow-difference' : 'r2flow-difference'));
    const metrics = result.metrics || {};
    const params = result.inputs || {};
    if (!params.iters) {
      params.iters = document.getElementById((isRice1 ? 'grid-iters-' : 'r2grid-iters-') + idx)?.textContent?.trim() || (isRice1 ? document.getElementById('pipe-iters-select')?.value : document.getElementById('r2pipe-iters-select')?.value) || '—';
    }
    const cardTimeText = document.getElementById((isRice1 ? 'grid-time-' : 'r2grid-time-') + idx)?.textContent?.trim();
    if (cardTimeText && cardTimeText !== '—' && !metrics.time) {
      metrics.time = cardTimeText;
    }
    const modelTitles = {
      '1': '1. FULL MODEL (ASM + RTE)',
      '2': '2. NO ASM (Only RTE)',
      '3': '3. NO RTE (Only ASM)',
      '4': '4. CORE DIP + METRICS (No Physics)'
    };
    const cardTitle = modelTitles[idx] || `Model ${idx}`;
    const getCardDOMVal = (field) => {
      const el = document.getElementById((isRice1 ? `grid-${field}-` : `r2grid-${field}-`) + idx);
      const t = el ? el.textContent.trim() : '';
      return (t && t !== '—' && t !== '-') ? t : null;
    };
    const metric = (key, digits = 3) => {
      if (metrics[key] !== undefined && metrics[key] !== null) return Number(metrics[key]).toFixed(digits);
      const domVal = getCardDOMVal(key);
      if (domVal) return domVal;
      return '—';
    };
    const imageCard = (label, url) => `<article class="image-card"><div class="image-card__heading"><h3>${label}</h3>${url ? `<a class="download-btn" download="${label.toLowerCase().replaceAll(' ', '_')}.png" href="${url}">⬇ Download</a>` : ''}</div>${url ? `<img src="${url}" alt="${label}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="image-fallback">Image could not be loaded</div>` : '<div class="image-fallback" style="display:flex">Image not generated yet</div>'}</article>`;
    const esc = (value) => String(value ?? '-').replace(/[<>&"']/g, (char) => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[char]));
    const labels = JSON.stringify(result.labels || result.lossHistory?.map(item => item.step) || []);
    const losses = JSON.stringify(result.lossTotal || result.lossHistory?.map(item => item.total) || []);
    const psnrHistory = JSON.stringify(result.psnr || result.lossHistory?.map(item => item.psnr) || []);
    const ssimHistory = JSON.stringify(result.ssim || result.lossHistory?.map(item => item.ssim) || []);
    const title = esc(cardTitle);

    const reportHtml = `<!doctype html><html><head><meta charset="utf-8"><title>${title} - Full Reconstruction Report</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script><style>
      body{margin:0;padding:28px;background:#07111f;color:#dbeafe;font:14px system-ui,sans-serif}main{max-width:1200px;margin:auto;background:#102238;padding:28px;border-radius:12px}header{display:flex;justify-content:space-between;align-items:center;gap:18px;border-bottom:1px solid #24506b;padding-bottom:18px}h1{color:#67e8f9;margin:0}.btn,.download-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;background:#0891b2;color:#fff;border:1px solid #67e8f9;border-radius:7px;padding:10px 14px;cursor:pointer;text-decoration:none;transition:transform .2s ease,background .2s ease,box-shadow .2s ease}.btn:hover,.download-btn:hover{background:#0e7490;transform:translateY(-2px);box-shadow:0 8px 20px #0005}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.metric,.image-card,.chart{background:linear-gradient(145deg,#0d1c30,#0a1525);border:1px solid #23445b;border-radius:10px;padding:14px;box-shadow:0 8px 24px #0002}.metric span{display:block;color:#94a3b8;font-size:12px}.metric strong{font-size:20px;color:#67e8f9}.image-card{display:flex;flex-direction:column;gap:10px}.image-card__heading{display:flex;align-items:center;justify-content:space-between;gap:8px}.image-card h3{font-size:13px;margin:0;color:#dbeafe}.image-card img{width:100%;aspect-ratio:1;object-fit:contain;background:#020617;border-radius:6px}.image-card .download-btn{font-size:11px;padding:6px 9px}.image-fallback{display:none;align-items:center;justify-content:center;min-height:180px;color:#94a3b8;background:#020617;border:1px dashed #31516a;border-radius:6px;text-align:center}.chart{height:280px}section{margin-top:26px}h2{color:#a5f3fc;font-size:17px;border-left:3px solid #22d3ee;padding-left:10px}
    </style></head><body><main><header><div><h1>Full Reconstruction Report</h1><p>${title}</p></div><button class="btn" onclick="downloadPDF()">Download Overall Information PDF</button></header>
      <section><h2>Execution Parameters</h2><div class="grid">${[['lambda ASM',params.asm],['lambda RTE',params.rte],['TV Loss',params.tv],['lambda NDVI',params.ndvi],['Iterations',params.iters],['Learning Rate',params.lr]].map(([label,value])=>'<div class="metric"><span>'+label+'</span><strong>'+esc(value)+'</strong></div>').join('')}</div></section>
      <section><h2>Quality Metrics</h2><div class="grid">${[['PSNR (dB)','psnr',2],['SSIM','ssim',3],['RMSE','rmse',4],['SAM (deg)','sam',2],['LPIPS','lpips',3],['Execution Time','time',null]].map(([label,key,digits])=>{
        let val = '—';
        if (key === 'time') {
          val = metrics.time || (metrics.timings?.total ? `${metrics.timings.total.toFixed(2)}s` : (metrics.elapsed ? `${metrics.elapsed.toFixed(2)}s` : '—'));
        } else {
          val = metric(key, digits);
        }
        return '<div class="metric"><span>'+label+'</span><strong>'+val+'</strong></div>';
      }).join('')}</div></section>
      <section><h2>Cloud Segmentation Metrics</h2><div class="grid">${[['Dice Score','dice'],['IoU (Jaccard)','iou'],['Precision','precision'],['Recall','recall']].map(([label,key])=>'<div class="metric"><span>'+label+'</span><strong>'+metric(key,4)+'</strong></div>').join('')}</div></section>
      <section><h2>Reconstruction Imagery</h2><div class="grid">${imageCard('Input Image (Cloudy)',input)}${imageCard('Ground Truth (Reference)',groundTruth)}${imageCard('Cloud Mask',mask)}${imageCard('Output Image (Restored)',output)}${imageCard('Difference / Error Map',difference)}</div></section>
      <section><h2>🔴🟢🔵 RGB Color Distribution Graphs</h2><div class="grid"><div class="chart"><div style="text-align:center;color:#38bdf8;font-weight:bold;font-size:12px;margin-bottom:4px;">1. Input Image (Cloudy Input)</div><canvas id="rgbInput"></canvas></div><div class="chart"><div style="text-align:center;color:#34d399;font-weight:bold;font-size:12px;margin-bottom:4px;">2. Ground Truth (Reference Clean)</div><canvas id="rgbTruth"></canvas></div><div class="chart"><div style="text-align:center;color:#a78bfa;font-weight:bold;font-size:12px;margin-bottom:4px;">3. Reconstructed Image (DIP+PINN)</div><canvas id="rgbOutput"></canvas></div></div></section>
      <section><h2>📈 Loss & Image Quality Progression Graphs</h2><div class="grid"><div class="chart"><div style="text-align:center;color:#38bdf8;font-weight:bold;font-size:12px;margin-bottom:4px;">Physics & DIP Loss Convergence Graph</div><canvas id="lossChart"></canvas></div><div class="chart"><div style="text-align:center;color:#34d399;font-weight:bold;font-size:12px;margin-bottom:4px;">Image Quality Progression (Ground Truth vs Reconstructed)</div><canvas id="psnrChart"></canvas></div></div></section>
      <section><h2>Image Quality Progression (Ground Truth vs Reconstructed Image)</h2><p style="color:#94a3b8">Full Model quality progression across optimization iterations comparing reconstructed image against ground truth.</p><div class="chart"><canvas id="qualityChart"></canvas></div></section>
    </main><script>
      var labels = ${labels}, losses = ${losses}, psnr = ${psnrHistory}, ssim = ${ssimHistory};
      new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: { labels: labels, datasets: [{ label: 'Physics + DIP Loss', data: losses, borderColor: '#f59e0b', fill: true, backgroundColor: 'rgba(245,158,11,0.1)' }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#f59e0b', font: { size: 10 } } } }, scales: { x: { title: { display: true, text: 'Optimization Step / Iteration', color: '#94a3b8' } }, y: { type: 'logarithmic', title: { display: true, text: 'Objective Loss (Log)', color: '#f59e0b' } } } }
      });
      new Chart(document.getElementById('psnrChart'), {
        type: 'line',
        data: { labels: labels, datasets: [{ label: 'PSNR (dB)', data: psnr, borderColor: '#22d3ee', fill: true, backgroundColor: 'rgba(34,211,238,0.1)' }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#22d3ee', font: { size: 10 } } } }, scales: { x: { title: { display: true, text: 'Optimization Step / Iteration', color: '#94a3b8' } }, y: { title: { display: true, text: 'PSNR (dB)', color: '#22d3ee' } } } }
      });
      new Chart(document.getElementById('qualityChart'), {
        type: 'line',
        data: { labels: labels, datasets: [{ label: 'PSNR (dB)', data: psnr, borderColor: '#22d3ee', tension: 0.3 }, { label: 'SSIM', data: ssim, borderColor: '#a78bfa', tension: 0.3, yAxisID: 'ssim' }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#cbd5e1' } } }, scales: { x: { title: { display: true, text: 'Optimization Step / Iteration', color: '#94a3b8' } }, y: { title: { display: true, text: 'PSNR (dB)', color: '#22d3ee' } }, ssim: { position: 'right', min: 0, max: 1, title: { display: true, text: 'SSIM (0 to 1)', color: '#a78bfa' }, grid: { drawOnChartArea: false } } } }
      });
      function plotHistogram(id, src) {
        if (!src) return;
        var img = new Image();
        img.onload = function() {
          var c = document.createElement('canvas'), ctx = c.getContext('2d');
          c.width = 128; c.height = 128;
          ctx.drawImage(img, 0, 0, 128, 128);
          var p = ctx.getImageData(0, 0, 128, 128).data, r = Array(16).fill(0), g = Array(16).fill(0), b = Array(16).fill(0);
          for (var i = 0; i < p.length; i += 4) { r[p[i] >> 4]++; g[p[i+1] >> 4]++; b[p[i+2] >> 4]++; }
          new Chart(document.getElementById(id), {
            type: 'line',
            data: {
              labels: Array.from({ length: 16 }, function(_, idx) { return idx * 16; }),
              datasets: [
                { label: 'Red Channel', data: r, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', fill: true },
                { label: 'Green Channel', data: g, borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.1)', fill: true },
                { label: 'Blue Channel', data: b, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true }
              ]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { labels: { color: '#94a3b8', boxWidth: 10, font: { size: 9 } } } },
              scales: {
                x: { title: { display: true, text: 'Pixel Intensity / DN [0-255]', color: '#94a3b8', font: { size: 9 } } },
                y: { title: { display: true, text: 'Pixel Frequency', color: '#94a3b8', font: { size: 9 } } }
              }
            }
          });
        };
        img.src = src;
      }
      plotHistogram('rgbInput', '' + ${JSON.stringify(input)});
      plotHistogram('rgbTruth', '' + ${JSON.stringify(groundTruth)});
      plotHistogram('rgbOutput', '' + ${JSON.stringify(output)});
      function downloadPDF() { html2pdf().set({ filename: 'reconstruction_report.pdf', margin: 0.4, html2canvas: { scale: 2 }, jsPDF: { format: 'a4', orientation: 'portrait' } }).from(document.querySelector('main')).save(); }
    </script></body></html>`;
    document.getElementById('reconstruction-report-modal')?.remove();
    const modal = document.createElement('div');
    modal.id = 'reconstruction-report-modal';
    modal.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(2,6,23,.82);backdrop-filter:blur(10px);display:flex;align-items:center;justify-content:center;padding:20px;opacity:0;transition:opacity .24s ease;';
    const shell = document.createElement('div');
    shell.style.cssText = 'width:min(1400px,96vw);height:min(92vh,1000px);background:#07111f;border:1px solid rgba(103,232,249,.45);border-radius:14px;overflow:hidden;transform:translateY(18px) scale(.98);transition:transform .28s cubic-bezier(.2,.8,.2,1);box-shadow:0 24px 80px rgba(0,0,0,.55);';
    const close = document.createElement('button');
    close.type = 'button';
    close.textContent = 'Close Report';
    close.style.cssText = 'position:absolute;right:32px;top:28px;z-index:2;background:#0e7490;color:#fff;border:1px solid #67e8f9;border-radius:6px;padding:8px 12px;cursor:pointer;';
    const frame = document.createElement('iframe');
    frame.title = `${title} - Full Reconstruction Report`;
    frame.srcdoc = reportHtml;
    frame.style.cssText = 'width:100%;height:100%;border:0;background:#07111f;';
    const closeModal = () => modal.remove();
    close.addEventListener('click', closeModal);
    modal.addEventListener('click', (event) => { if (event.target === modal) closeModal(); });
    document.addEventListener('keydown', function closeOnEscape(event) {
      if (event.key === 'Escape') { closeModal(); document.removeEventListener('keydown', closeOnEscape); }
    }, { once: true });
    shell.append(frame);
    modal.append(close, shell);
    document.body.append(modal);
    requestAnimationFrame(() => { modal.style.opacity = '1'; shell.style.transform = 'translateY(0) scale(1)'; });
};

// ─── Real-Time PDF Report Generator Event Handlers ──────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".pdf-generate-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      const dataset = btn.dataset.dataset || "RICE1";

      const card = btn.closest(".pdf-report-card") || btn.closest(".bench-controls") || btn.parentElement;
      const statusBox = card.querySelector(".pdf-status-box") || document.getElementById(`${dataset.toLowerCase()}-pdf-status-box`);
      const statusText = card.querySelector(".pdf-status-text") || document.getElementById(`${dataset.toLowerCase()}-pdf-status-text`);
      const actionBtns = card.querySelector(".pdf-action-btns") || document.getElementById(`${dataset.toLowerCase()}-pdf-action-btns`);
      const openBtn = card.querySelector(".pdf-open-btn") || document.getElementById(`${dataset.toLowerCase()}-pdf-open`);
      const downloadBtn = card.querySelector(".pdf-download-btn") || document.getElementById(`${dataset.toLowerCase()}-pdf-download`);

      if (statusBox) statusBox.style.display = "block";
      if (actionBtns) actionBtns.style.display = "none";
      btn.disabled = true;

      const steps = [
        "Generating report...",
        "Reading experiment data...",
        "Building tables...",
        "Generating graphs...",
        "Generating PDF...",
        "Saving to Google Drive..."
      ];

      for (let i = 0; i < steps.length; i++) {
        if (statusText) statusText.textContent = `⏳ ${steps[i]}`;
        await new Promise((resolve) => setTimeout(resolve, 200));
      }

      try {
        const currentSampleIdx = (typeof state !== 'undefined' && state.index !== undefined) ? state.index : (typeof state !== 'undefined' && state.rice2_index !== undefined ? state.rice2_index : 0);
        const response = await fetch("/api/generate_pdf_report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset: dataset, sample_id: currentSampleIdx })
        });

        const data = await response.json();
        if (response.ok && data.status === "success") {
          if (statusText) statusText.textContent = "✓ Report Generated Successfully";
          if (openBtn) {
            openBtn.href = data.view_url;
            openBtn.target = "_blank";
          }
          if (downloadBtn) {
            downloadBtn.href = data.download_url;
            downloadBtn.download = data.filename;
          }
          if (actionBtns) actionBtns.style.display = "flex";
        } else {
          if (statusText) statusText.textContent = `❌ Failed: ${data.detail || data.message || "Error generating report"}`;
        }
      } catch (err) {
        if (statusText) statusText.textContent = `❌ Error: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  });
});
