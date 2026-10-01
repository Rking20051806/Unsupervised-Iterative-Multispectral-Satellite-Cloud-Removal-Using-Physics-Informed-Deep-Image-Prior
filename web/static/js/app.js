/* ═══════════════════════════════════════════════════════════════════════════
   CloudVision AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  overview:       null,
  split:          'RICE1',
  index:          0,
  sensitivity:    0.75,
  file:           null,
  uploading:      false,
  // Sentinel-2
  s2_x:           null,
  s2_y:           null,
  s2_x_start:     null,
  s2_y_start:     null,
  s2_mask_visible:true,
  s2_optimizing:  false,
  s2_chart:       null,
  s2_overview_w:  512,
  s2_overview_h:  512,
  s2_full_w:      10980,
  s2_full_h:      10980,
  s2_crop_size:   256,
  // RICE Benchmark
  bench_split:    'RICE1',
  bench_index:    0,
  bench_running:  false,
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
  if (!src) { el.removeAttribute('src'); el.style.opacity = 0.2; return; }
  el.src = src;
  el.style.opacity = 1;
}

function setStatus(text, type = 'ready') {
  if (!els.statusPill) return;
  els.statusPill.textContent = text;
  els.statusPill.className = 'status-pill' + (type === 'processing' ? ' processing' : '');
}

function setLoading(on) {
  document.body.classList.toggle('is-loading', on);
  setStatus(on ? 'Processing…' : 'Ready', on ? 'processing' : 'ready');
}

// ── Tab Navigation ─────────────────────────────────────────────────────────
function setTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('tab-btn--active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('tab-panel--active', p.id === `tab-${name}`));
  if (name === 's2-workspace') setTimeout(setupS2Canvas, 80);
}

// ── Overview / Init ────────────────────────────────────────────────────────
async function loadOverview() {
  setLoading(true);
  try {
    const res  = await fetch('/api/overview');
    if (!res.ok) throw new Error('Backend not reachable');
    state.overview = await res.json();

    // RICE counts
    const r1 = state.overview.splits?.RICE1;
    const r2 = state.overview.splits?.RICE2;
    if (els.rice1Count) els.rice1Count.textContent = r1?.count ?? '—';
    if (els.rice2Count) els.rice2Count.textContent = r2?.count ?? '—';
    if (els.s2Status)   els.s2Status.textContent   = state.overview.s2_overview ? 'Loaded ✓' : 'Not found';
    updateSampleMeta();

    // Sentinel-2 overview
    if (state.overview.s2_overview) {
      const ov = state.overview.s2_overview;
      state.s2_overview_w = ov.overview_w || ov.width  || 512;
      state.s2_overview_h = ov.overview_h || ov.height || 512;
      state.s2_full_w     = ov.width  || 10980;
      state.s2_full_h     = ov.height || 10980;

      if (els.s2OverviewImage) {
        els.s2OverviewImage.src = ov.tci_url;
        els.s2OverviewImage.onload = () => {
          setupS2Canvas();
          hideS2Spinner();
        };
        if (els.s2OverviewImage.complete) { setupS2Canvas(); hideS2Spinner(); }
      }
      if (els.s2MaskImage) els.s2MaskImage.src = ov.mask_url;

      // tile metadata
      if (els.tmSize)   els.tmSize.textContent   = `${ov.width} × ${ov.height} px`;
      if (els.tmCrs)    els.tmCrs.textContent    = ov.crs || 'EPSG:32643';
      if (els.tmPxsize) els.tmPxsize.textContent = '10 m / pixel';
    }

    // Kick off first sample preview (detection only)
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
  const names  = state.overview?.splits?.[state.split]?.names || [];
  const name   = names[state.index] || '—';
  if (els.sampleIndexLabel) els.sampleIndexLabel.textContent = `#${state.index}`;
  if (els.sampleNameLabel)  els.sampleNameLabel.textContent  = name;
  if (els.sampleMeta)       els.sampleMeta.textContent       = `${names.length} samples`;
  if (els.sampleRange)      els.sampleRange.max              = Math.max((names.length || 1) - 1, 0);
}

// ── RICE Analysis (Detection + Removal) ───────────────────────────────────
async function analyzeCurrentSample() {
  setLoading(true);
  try {
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
    setImg(els.inputImage,   R.images?.input);
    setImg(els.maskImage,    R.images?.detected_mask);
    setImg(els.overlayImage, R.images?.overlay);

    // Removal tab images
    setImg(els.labelImage,   R.images?.label);
    setImg(els.removedImage, R.images?.removed);

    // RICE comparison slider
    if (R.images?.input && R.images?.removed) {
      setImg(els.riceBefore, R.images.input);
      setImg(els.riceAfter,  R.images.removed);
      if (els.riceCompareWrapper) {
        els.riceCompareWrapper.style.display = 'block';
        els.riceCompareWrapper.classList.add('fade-in');
      }
      setRiceSplitSlider(50);
    }

    // Populate Metrics tab
    populateMetrics(M, body.source?.name);

    // Handle classification results
    try {
      const classRes = await classResPromise;
      if (classRes.ok) {
        const classBody = await classRes.json();
        const CR = classBody.images;
        const CM = classBody.metrics || {};
        const CC = classBody.classification || {};

        setImg(els.classInputImage,    CR.input);
        setImg(els.classMaskImage,     CR.mask);
        setImg(els.classSaliencyImage, CR.saliency_overlay || CR.saliency);

        if (els.phClassInput) els.phClassInput.style.display = 'none';
        if (els.phClassMask)  els.phClassMask.style.display = 'none';
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
  } catch (err) {
    console.error(err);
    setStatus('Error', 'processing');
    if (els.analysisDescription) els.analysisDescription.textContent = err.message;
  } finally {
    setLoading(false);
  }
}

function populateMetrics(M, sourceName) {
  if (els.metricsSourceLabel && sourceName)
    els.metricsSourceLabel.textContent = `Source: ${sourceName}`;

  const set = (id, val, digits = 4) => {
    const el = $(id);
    if (el) el.textContent = (val !== null && val !== undefined) ? fmt(val, digits) : '—';
  };

  set('metric-cloud-coverage', M.cloud_ratio !== undefined ? M.cloud_ratio * 100 : null, 1);
  set('metric-psnr',  M.psnr,  2);
  set('metric-ssim',  M.ssim,  4);
  set('metric-sam',   M.sam,   4);
  set('metric-prs',   M.prs,   5);
  set('metric-dice',  M.dice,  4);
  set('metric-iou',   M.iou,   4);
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
  try {
    const res  = await fetch(`/api/rice/sample?split=${split}&index=${index}`);
    if (!res.ok) throw new Error('Failed to load sample');
    const data = await res.json();

    if (els.benchSampleName) els.benchSampleName.textContent = `#${data.name}`;
    setImg(els.benchCloudyImg, data.cloudy);
    setImg(els.benchLabelImg,  data.label);
    setImg(els.benchMaskImg,   data.mask);
  } catch (err) {
    console.error(err);
  }
}

async function runBenchmark() {
  if (state.bench_running) return;
  state.bench_running = true;

  const split       = els.benchSplit?.value || 'RICE1';
  const index       = parseInt(els.benchIndex?.value || '0', 10);
  const sensitivity = parseFloat(els.benchSensitivity?.value || '0.75');

  if (els.benchRunBtn)       els.benchRunBtn.disabled = true;
  if (els.benchStatusPill) { els.benchStatusPill.textContent = 'Running DIP+PINN…'; els.benchStatusPill.className = 'status-pill processing'; }

  // Reset metric displays
  ['bm-psnr','bm-ssim','bm-sam','bm-prs','bm-dice','bm-iou','bm-cloud','bm-prs-asm'].forEach(id => {
    const el = $(id); if (el) el.textContent = '…';
  });

  const t0 = performance.now();
  try {
    const res  = await fetch(`/api/rice/benchmark?split=${split}&index=${index}&sensitivity=${sensitivity}`);
    if (!res.ok) throw new Error(`Benchmark error: ${res.status}`);
    const body = await res.json();
    const M    = body.results?.metrics || {};
    const imgs = body.results?.images  || {};

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

    const setBM = (id, val, digits = 4) => {
      const el = $(id); if (el) el.textContent = (val !== null && val !== undefined) ? fmt(val, digits) : '—';
    };

    setBM('bm-psnr',    M.psnr,      2);
    setBM('bm-ssim',    M.ssim,      4);
    setBM('bm-sam',     M.sam,       4);
    setBM('bm-prs',     M.prs,       5);
    setBM('bm-dice',    M.dice,      4);
    setBM('bm-iou',     M.iou,       4);
    setBM('bm-cloud',   M.cloud_ratio !== undefined ? M.cloud_ratio * 100 : null, 1);
    setBM('bm-prs-asm', M.prs_asm,   5);

    // Show result images
    setImg(els.benchResultImg,  imgs.removed);
    setImg(els.benchOverlayImg, imgs.overlay);
    if (els.benchResultImages) {
      els.benchResultImages.style.display = 'block';
      els.benchResultImages.classList.add('fade-in');
    }

    if (els.benchStatusPill) {
      els.benchStatusPill.textContent = `Done in ${elapsed}s ✓`;
      els.benchStatusPill.className   = 'status-pill';
    }
    if (els.benchRuntime) els.benchRuntime.textContent = `${elapsed}s`;

    // Mirror to full metrics tab
    populateMetrics(M, `${split} #${index}`);

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
  if (els.riceSplitBar)   els.riceSplitBar.style.left    = `${pct}%`;
}

function setupRiceSplitSlider() {
  const slider = els.riceSplitSlider;
  if (!slider) return;
  function update(clientX) {
    const rect = slider.getBoundingClientRect();
    const pct  = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setRiceSplitSlider(pct);
  }
  slider.addEventListener('mousemove', e => update(e.clientX));
  slider.addEventListener('touchmove', e => { if (e.touches[0]) update(e.touches[0].clientX); });
}

// ── Sentinel-2 Canvas ──────────────────────────────────────────────────────
function hideS2Spinner() {
  if (els.s2MapSpinner) els.s2MapSpinner.classList.add('hidden');
}

function setupS2Canvas() {
  const canvas = els.s2OverviewCanvas;
  const img    = els.s2OverviewImage;
  if (!canvas || !img) return;

  canvas.width  = img.clientWidth  || 512;
  canvas.height = img.clientHeight || 512;

  const ctx      = canvas.getContext('2d');
  const cropSize = state.s2_crop_size;
  const canvasBox = cropSize * (canvas.width / state.s2_full_w);

  function drawCanvas(mx = null, my = null) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Locked selection
    if (state.s2_x !== null) {
      ctx.save();
      ctx.strokeStyle = '#ff6b6b';
      ctx.lineWidth   = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(state.s2_x - canvasBox/2, state.s2_y - canvasBox/2, canvasBox, canvasBox);
      ctx.fillStyle = 'rgba(255,107,107,0.08)';
      ctx.fillRect(state.s2_x - canvasBox/2, state.s2_y - canvasBox/2, canvasBox, canvasBox);
      ctx.restore();
    }

    // Hover
    if (mx !== null) {
      ctx.save();
      ctx.strokeStyle = '#5bc8f5';
      ctx.lineWidth   = 2;
      ctx.setLineDash([]);
      ctx.strokeRect(mx - canvasBox/2, my - canvasBox/2, canvasBox, canvasBox);
      ctx.fillStyle = 'rgba(91,200,245,0.12)';
      ctx.fillRect(mx - canvasBox/2, my - canvasBox/2, canvasBox, canvasBox);
      // Crosshair
      ctx.strokeStyle = 'rgba(91,200,245,0.5)';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(mx, 0); ctx.lineTo(mx, canvas.height);
      ctx.moveTo(0, my); ctx.lineTo(canvas.width, my);
      ctx.stroke();
      ctx.restore();
    }
  }

  canvas.addEventListener('mousemove', e => {
    if (state.s2_optimizing) return;
    drawCanvas(e.offsetX, e.offsetY);
  });
  canvas.addEventListener('mouseleave', () => { if (!state.s2_optimizing) drawCanvas(); });
  canvas.addEventListener('click', async e => {
    if (state.s2_optimizing) return;
    state.s2_x = e.offsetX;
    state.s2_y = e.offsetY;
    drawCanvas(e.offsetX, e.offsetY);
    await fetchS2Crop(e.offsetX, e.offsetY);
  });

  drawCanvas();
}

async function fetchS2Crop(cx, cy) {
  setLoading(true);
  const cs = parseInt(els.s2CropSize?.value || '256', 10);
  state.s2_crop_size = cs;
  try {
    const url = `/api/s2/crop?x=${cx}&y=${cy}&overview_w=${state.s2_overview_w}&overview_h=${state.s2_overview_h}&crop_size=${cs}`;
    const res  = await fetch(url);
    if (!res.ok) throw new Error(`Crop failed: ${res.status}`);
    const data = await res.json();

    state.s2_x_start = data.x_start;
    state.s2_y_start = data.y_start;

    if (els.s2CoordsReadout)
      els.s2CoordsReadout.textContent =
        `Crop: x=[${data.x_start}–${data.x_start+cs}], y=[${data.y_start}–${data.y_start+cs}]`;
    if (els.s2CloudFractionChip)
      els.s2CloudFractionChip.textContent = `Cloud: ${(data.cloud_fraction*100).toFixed(1)}%`;

    setImg(els.s2CropBefore, data.cloudy_crop);
    setImg(els.s2CropAfter,  data.cloudy_crop);
    setS2SplitSlider(50);

    if (els.s2RunOptBtn) els.s2RunOptBtn.disabled = false;

    // Reset metrics & chart
    ['s2-metric-prs','s2-metric-psnr','s2-metric-ssim','s2-metric-sam','s2-metric-time'].forEach(id => {
      const el = $(id); if (el) el.textContent = '—';
    });
    if (els.s2ChartStep) els.s2ChartStep.textContent = 'Step —';
    if (state.s2_chart)  { state.s2_chart.destroy(); state.s2_chart = null; }
    if (els.s2ProgressTrack) els.s2ProgressTrack.style.display = 'none';

  } catch (err) {
    console.error(err);
    if (els.s2CoordsReadout) els.s2CoordsReadout.textContent = `Error: ${err.message}`;
  } finally {
    setLoading(false);
  }
}

// ── S2 Split Slider ────────────────────────────────────────────────────────
function setS2SplitSlider(pct) {
  if (els.s2SplitAfterContainer) els.s2SplitAfterContainer.style.width = `${100 - pct}%`;
  if (els.s2SplitBar)            els.s2SplitBar.style.left              = `${pct}%`;
}

function setupS2SplitSlider() {
  const slider = els.s2SplitSlider;
  if (!slider) return;
  function update(clientX) {
    const rect = slider.getBoundingClientRect();
    const pct  = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setS2SplitSlider(pct);
  }
  slider.addEventListener('mousemove', e => update(e.clientX));
  slider.addEventListener('touchmove', e => { if (e.touches[0]) update(e.touches[0].clientX); });
}

// ── Loss Chart ─────────────────────────────────────────────────────────────
function initLossChart() {
  if (state.s2_chart) { state.s2_chart.destroy(); state.s2_chart = null; }
  const ctx = $('s2-losses-chart');
  if (!ctx) return;
  state.s2_chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Total',  data: [], borderColor: '#5bc8f5', borderWidth: 2, tension: 0.3, pointRadius: 0, fill: { target: 'origin', above: 'rgba(91,200,245,0.05)' } },
        { label: 'ASM',    data: [], borderColor: '#3de6a8', borderWidth: 1.5, tension: 0.3, pointRadius: 0 },
        { label: 'NDVI',   data: [], borderColor: '#9b7fff', borderWidth: 1.5, tension: 0.3, pointRadius: 0 },
        { label: 'TV',     data: [], borderColor: '#f5a623', borderWidth: 1.5, tension: 0.3, pointRadius: 0 },
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

// ── S2 Optimization ────────────────────────────────────────────────────────
function runS2Optimization() {
  if (state.s2_x_start === null || state.s2_optimizing) return;

  state.s2_optimizing = true;
  if (els.s2RunOptBtn)   els.s2RunOptBtn.disabled   = true;
  if (els.s2ItersSelect) els.s2ItersSelect.disabled = true;
  if (els.s2CropSize)    els.s2CropSize.disabled     = true;
  setStatus('Optimizing…', 'processing');

  if (els.s2ProgressTrack) els.s2ProgressTrack.style.display = 'block';
  if (els.s2ProgressFill)  els.s2ProgressFill.style.width    = '0%';

  initLossChart();
  setS2SplitSlider(100); // show restored side fully during optimization

  const iters   = parseInt(els.s2ItersSelect?.value || '100', 10);
  const cropSz  = state.s2_crop_size;
  const es      = new EventSource(
    `/api/s2/optimize?x_start=${state.s2_x_start}&y_start=${state.s2_y_start}&iters=${iters}&crop_size=${cropSz}`
  );

  es.onmessage = event => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    if (data.error) {
      console.error('Optimization error:', data.error);
      if (els.s2CoordsReadout) els.s2CoordsReadout.textContent = `Error: ${data.error}`;
      es.close();
      finalizeS2();
      return;
    }

    if (data.finished) {
      es.close();
      setImg(els.s2CropAfter, data.image);
      setS2SplitSlider(50);

      const M = data.metrics || {};
      const set = (id, v, d = 4) => { const el = $(id); if (el) el.textContent = (v !== null && v !== undefined) ? fmt(v, d) : '—'; };
      set('s2-metric-prs',  M.prs_total, 5);
      set('s2-metric-psnr', M.psnr,      2);
      set('s2-metric-ssim', M.ssim,      4);
      set('s2-metric-sam',  M.prs_sam,   4);
      set('s2-metric-time', data.elapsed, 1);

      if (els.s2ProgressFill) els.s2ProgressFill.style.width = '100%';
      if (els.s2ChartStep)    els.s2ChartStep.textContent    = `Done ✓`;

      finalizeS2();
      return;
    }

    // Intermediate update
    if (data.image)  setImg(els.s2CropAfter, data.image);
    if (data.step !== undefined) {
      const pct = Math.round((data.step / iters) * 100);
      if (els.s2ProgressFill) els.s2ProgressFill.style.width = `${pct}%`;
      if (els.s2StepBadge)    els.s2StepBadge.textContent    = `Step ${data.step}`;
      if (els.s2ChartStep)    els.s2ChartStep.textContent    = `Step ${data.step}`;
    }

    // Chart update
    if (state.s2_chart && data.step !== undefined) {
      const ch = state.s2_chart.data;
      ch.labels.push(data.step);
      ch.datasets[0].data.push(data.total_loss);
      ch.datasets[1].data.push(data.losses?.l_asm);
      ch.datasets[2].data.push(data.losses?.l_ndvi);
      ch.datasets[3].data.push(data.losses?.l_tv);
      state.s2_chart.update('none');
    }
  };

  es.onerror = () => { es.close(); finalizeS2(); };

  function finalizeS2() {
    state.s2_optimizing = false;
    if (els.s2RunOptBtn)   els.s2RunOptBtn.disabled   = false;
    if (els.s2ItersSelect) els.s2ItersSelect.disabled  = false;
    if (els.s2CropSize)    els.s2CropSize.disabled      = false;
    setStatus('Ready');
  }
}

// ── Load Full S2 Overview ──────────────────────────────────────────────────
async function loadS2FullOverview() {
  if (els.s2LoadFullBtn) els.s2LoadFullBtn.disabled = true;
  if (els.s2MapSpinner)  els.s2MapSpinner.classList.remove('hidden');
  try {
    const res  = await fetch('/api/s2/overview_full?max_dim=1024');
    if (!res.ok) throw new Error('Failed to load full overview');
    const data = await res.json();

    state.s2_overview_w = data.thumb_w;
    state.s2_overview_h = data.thumb_h;
    state.s2_full_w     = data.full_w;
    state.s2_full_h     = data.full_h;

    if (els.s2OverviewImage) {
      els.s2OverviewImage.src    = data.tci_url;
      els.s2OverviewImage.onload = () => { setupS2Canvas(); hideS2Spinner(); };
    }
    if (els.s2MaskImage) els.s2MaskImage.src = data.mask_url;

    if (els.tmSize)   els.tmSize.textContent   = `${data.full_w} × ${data.full_h} px`;
    if (els.tmCrs)    els.tmCrs.textContent    = data.crs || 'EPSG:32643';
    if (els.tmPxsize) els.tmPxsize.textContent = `${data.pixel_size || 10} m / pixel`;

    if (els.s2LoadFullBtn) els.s2LoadFullBtn.textContent = '1024px ✓';
  } catch (err) {
    console.error(err);
    hideS2Spinner();
    if (els.s2LoadFullBtn) { els.s2LoadFullBtn.disabled = false; els.s2LoadFullBtn.textContent = 'Retry'; }
  }
}

// ── Element Caching ────────────────────────────────────────────────────────
function cacheElements() {
  // Header KPIs
  els.rice1Count     = $('rice1-count');
  els.rice2Count     = $('rice2-count');
  els.s2Status       = $('s2-status');
  els.serviceStatus  = $('service-status');

  // Sidebar
  els.splitSelect        = $('split-select');
  els.sampleRange        = $('sample-range');
  els.sampleMeta         = $('sample-meta');
  els.sampleIndexLabel   = $('sample-index-label');
  els.sampleNameLabel    = $('sample-name-label');
  els.sensitivityRange   = $('sensitivity-range');
  els.sensitivityValue   = $('sensitivity-value');
  els.uploadInput        = $('upload-input');
  els.uploadName         = $('upload-name');
  els.runDemoBtn         = $('run-demo-btn');
  els.analyzeUploadBtn   = $('analyze-upload-btn');
  els.clearUploadBtn     = $('clear-upload-btn');

  // Content header
  els.statusPill         = $('analysis-pill');
  els.analysisDescription= $('analysis-description');

  // Detection tab
  els.inputImage    = $('input-image');
  els.maskImage     = $('mask-image');
  els.overlayImage  = $('overlay-image');

  // Classification tab
  els.classInputImage    = $('class-input-image');
  els.classMaskImage     = $('class-mask-image');
  els.classSaliencyImage = $('class-saliency-image');
  els.phClassInput       = $('ph-class-input');
  els.phClassMask        = $('ph-class-mask');
  els.phClassSaliency    = $('ph-class-saliency');

  // Removal tab
  els.labelImage          = $('label-image');
  els.removedImage        = $('removed-image');
  els.riceCompareWrapper  = $('rice-compare-wrapper');
  els.riceSplitSlider     = $('rice-split-slider');
  els.riceSplitAfter      = $('rice-split-after');
  els.riceSplitBar        = $('rice-split-bar');
  els.riceBefore          = $('rice-before');
  els.riceAfter           = $('rice-after');

  // Metrics tab
  els.metricsSourceLabel  = $('metrics-source-label');

  // RICE benchmark tab
  els.benchSplit          = $('bench-split');
  els.benchIndex          = $('bench-index');
  els.benchSensitivity    = $('bench-sensitivity');
  els.benchSensitivityVal = $('bench-sensitivity-val');
  els.benchRunBtn         = $('bench-run-btn');
  els.benchSampleName     = $('bench-sample-name');
  els.benchCloudyImg      = $('bench-cloudy-img');
  els.benchLabelImg       = $('bench-label-img');
  els.benchMaskImg        = $('bench-mask-img');
  els.benchStatusPill     = $('bench-status-pill');
  els.benchRuntime        = $('bench-runtime');
  els.benchResultImages   = $('bench-result-images');
  els.benchResultImg      = $('bench-result-img');
  els.benchOverlayImg     = $('bench-overlay-img');

  // Sentinel-2 tab
  els.s2OverviewImage        = $('s2-overview-image');
  els.s2MaskImage            = $('s2-mask-image');
  els.s2OverviewCanvas       = $('s2-overview-canvas');
  els.s2MapSpinner           = $('s2-map-spinner');
  els.s2CoordsReadout        = $('s2-coords-readout');
  els.s2ToggleMaskBtn        = $('s2-toggle-mask-btn');
  els.s2LoadFullBtn          = $('s2-load-full-btn');
  els.s2ItersSelect          = $('s2-iters-select');
  els.s2CropSize             = $('s2-crop-size');
  els.s2RunOptBtn            = $('s2-run-opt-btn');
  els.s2ProgressTrack        = $('s2-progress-track');
  els.s2ProgressFill         = $('s2-progress-fill');
  els.s2StepBadge            = $('s2-step-badge');
  els.s2ChartStep            = $('s2-chart-step');
  els.s2CropBefore           = $('s2-crop-before');
  els.s2CropAfter            = $('s2-crop-after');
  els.s2CloudFractionChip    = $('s2-cloud-fraction-chip');
  els.s2SplitSlider          = $('s2-split-slider');
  els.s2SplitAfterContainer  = $('s2-split-after-container');
  els.s2SplitBar             = $('s2-split-bar');

  // Tile metadata
  els.tmSize   = $('tm-size');
  els.tmCrs    = $('tm-crs');
  els.tmPxsize = $('tm-pxsize');
}

// ── Event Binding ──────────────────────────────────────────────────────────
function bindEvents() {
  // Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => setTab(btn.dataset.tab));
  });

  // Sidebar controls
  els.splitSelect?.addEventListener('change', async () => {
    state.split = els.splitSelect.value;
    state.index = 0;
    updateSampleMeta();
    await analyzeCurrentSample();
  });

  els.sampleRange?.addEventListener('input', async () => {
    state.index = Number(els.sampleRange.value);
    updateSampleMeta();
    state.uploading = false; state.file = null;
    if (els.uploadInput) els.uploadInput.value = '';
    if (els.uploadName)  els.uploadName.textContent = 'No file selected';
    await analyzeCurrentSample();
  });

  els.sensitivityRange?.addEventListener('input', () => {
    state.sensitivity = Number(els.sensitivityRange.value);
    if (els.sensitivityValue) els.sensitivityValue.textContent = state.sensitivity.toFixed(2);
  });

  els.uploadInput?.addEventListener('change', () => {
    const [f] = els.uploadInput.files || [];
    state.file     = f || null;
    state.uploading = Boolean(f);
    if (els.uploadName) els.uploadName.textContent = f ? f.name : 'No file selected';
    if (f) {
      const reader = new FileReader();
      reader.onload = () => setImg(els.inputImage, reader.result);
      reader.readAsDataURL(f);
    }
  });

  els.runDemoBtn?.addEventListener('click', async () => {
    state.uploading = false; state.file = null;
    if (els.uploadInput) els.uploadInput.value = '';
    if (els.uploadName)  els.uploadName.textContent = 'No file selected';
    await analyzeCurrentSample();
  });

  els.analyzeUploadBtn?.addEventListener('click', async () => {
    if (!state.file) return;
    state.uploading = true;
    await analyzeCurrentSample();
  });

  els.clearUploadBtn?.addEventListener('click', async () => {
    state.uploading = false; state.file = null;
    if (els.uploadInput) els.uploadInput.value = '';
    if (els.uploadName)  els.uploadName.textContent = 'No file selected';
    await analyzeCurrentSample();
  });

  // S2 controls
  els.s2ToggleMaskBtn?.addEventListener('click', () => {
    state.s2_mask_visible = !state.s2_mask_visible;
    if (els.s2MaskImage) els.s2MaskImage.style.display = state.s2_mask_visible ? '' : 'none';
    if (els.s2ToggleMaskBtn) els.s2ToggleMaskBtn.textContent = state.s2_mask_visible ? 'Hide Mask' : 'Show Mask';
  });

  els.s2RunOptBtn?.addEventListener('click', runS2Optimization);
  els.s2LoadFullBtn?.addEventListener('click', loadS2FullOverview);

  els.s2CropSize?.addEventListener('change', () => {
    state.s2_crop_size = parseInt(els.s2CropSize.value, 10);
    // Re-setup canvas with new box size
    setupS2Canvas();
  });

  // Benchmark tab
  els.benchRunBtn?.addEventListener('click', runBenchmark);
  els.benchSplit?.addEventListener('change', loadBenchSample);
  els.benchIndex?.addEventListener('change', loadBenchSample);
  els.benchSensitivity?.addEventListener('input', () => {
    if (els.benchSensitivityVal)
      els.benchSensitivityVal.textContent = parseFloat(els.benchSensitivity.value).toFixed(2);
  });

  // Resize
  window.addEventListener('resize', () => setupS2Canvas());
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  cacheElements();
  bindEvents();
  setupS2SplitSlider();
  setupRiceSplitSlider();
  setTab('detection');

  // Set initial values
  if (els.sensitivityValue) els.sensitivityValue.textContent = state.sensitivity.toFixed(2);

  await loadOverview();

  // Load bench sample preview (no analysis yet)
  await loadBenchSample();
});
