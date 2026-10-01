document.addEventListener('DOMContentLoaded', () => {
    // --- UI Elements ---
    const els = {
        mapContainer: document.getElementById('s2-map'),
        mapOverlay: document.getElementById('s2-map-overlay'),
        btnToggleMask: document.getElementById('s2-btn-toggle-mask'),
        
        previewInput: document.getElementById('s2-preview-input'),
        previewOutput: document.getElementById('s2-preview-output'),
        slider: document.getElementById('s2-compare-slider'),
        sliderHandle: document.getElementById('s2-slider-handle'),
        previewEmpty: document.getElementById('s2-preview-empty'),
        previewStatus: document.getElementById('s2-preview-status'),

        inputLat: document.getElementById('s2-input-lat'),
        inputLon: document.getElementById('s2-input-lon'),
        inputSize: document.getElementById('s2-input-size'),
        inputIters: document.getElementById('s2-input-iters'),
        inputMask: document.getElementById('s2-input-mask'),

        btnRandom: document.getElementById('s2-btn-random'),
        btnStart: document.getElementById('s2-btn-start'),
        btnPause: document.getElementById('s2-btn-pause'),
        btnStop: document.getElementById('s2-btn-stop'),
        activeControls: document.getElementById('s2-active-controls'),

        logContainer: document.getElementById('s2-log-container'),
        btnClearLog: document.getElementById('s2-btn-clear-log'),
        
        lossChart: document.getElementById('s2-loss-chart'),
        metricChart: document.getElementById('s2-metric-chart'),
        
        mapSelectors: document.querySelectorAll('.s2-sub-tab-btn'),
        layerRadios: document.querySelectorAll('input[name="s2_layer"]'),
        layerMaskCheck: document.getElementById('s2_layer_mask'),
        
        readoutLatLon: document.getElementById('s2-readout-latlon'),
        readoutUtm: document.getElementById('s2-readout-utm'),
        readoutPixel: document.getElementById('s2-readout-pixel'),
        cropX: document.getElementById('s2-crop-x'),
        cropY: document.getElementById('s2-crop-y'),
        
        presetBtns: document.querySelectorAll('.s2-preset-btn'),
        itersText: document.getElementById('s2-iters-text'),
        
        btnFullscreen: document.getElementById('s2-map-fs-btn'),
        mapWrapper: document.getElementById('s2-map-wrapper')
    };

    if (!els.mapContainer) return;

    // --- State ---
    const state = {
        map: null,
        tileLayer: null,
        maskLayer: null,
        currentMapId: 'Vidarbha_Nagpur_Maharashtra.SAFE',
        currentLayer: 'tci',
        marker: null,
        lossChartInst: null,
        metricChartInst: null,
        isOptimizing: false,
        controller: null,
        bounds: null
    };

    // --- Logger ---
    const log = (msg, type = 'info') => {
        if (!els.logContainer) return;
        const div = document.createElement('div');
        const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
        
        if (type === 'error') div.className = 'text-red-400';
        else if (type === 'success') div.className = 'text-green-400';
        else if (type === 'warn') div.className = 'text-yellow-400';
        else div.className = 'text-gray-300';
        
        div.textContent = `[${timestamp}] ${msg}`;
        els.logContainer.appendChild(div);
        els.logContainer.scrollTop = els.logContainer.scrollHeight;
    };

    els.btnClearLog?.addEventListener('click', () => {
        if (els.logContainer) els.logContainer.innerHTML = '';
        log('Logs cleared.', 'info');
    });

    // --- Map Initialization ---
    const initMap = async () => {
        els.mapOverlay.style.display = 'flex';
        try {
            const res = await fetch('/api/s2/maps');
            const data = await res.json();
            if (data.current) {
                state.currentMapId = data.current;
            }

            state.map = L.map('s2-map', {
                center: [21.1458, 79.0882],
                zoom: 10,
                zoomControl: false // Move zoom to bottom right or keep default but offset
            });
            L.control.zoom({ position: 'topleft' }).addTo(state.map);

            loadTileLayer();

            // Map Click (Update Inputs & Crop Text)
            state.map.on('click', async (e) => {
                if (state.isOptimizing) return;
                const lat = e.latlng.lat;
                const lng = e.latlng.lng;
                els.inputLat.value = lat.toFixed(6);
                els.inputLon.value = lng.toFixed(6);
                
                if (state.marker) state.map.removeLayer(state.marker);
                state.marker = L.marker([lat, lng]).addTo(state.map);
                log(`Selected coordinates: Lat ${lat.toFixed(4)}, Lon ${lng.toFixed(4)}`);
                
                // Fetch crop coordinates implicitly
                try {
                    const size = els.inputSize.value;
                    const cropUrl = `/api/s2/crop?lat=${lat}&lon=${lng}&crop_size=${size}&mask_source=scl`;
                    const cropRes = await fetch(cropUrl);
                    if (cropRes.ok) {
                        const cropData = await cropRes.json();
                        els.cropX.textContent = `${cropData.x_start}-${cropData.x_start + parseInt(size)}`;
                        els.cropY.textContent = `${cropData.y_start}-${cropData.y_start + parseInt(size)}`;
                    }
                } catch(e){}
            });
            
            // Map MouseMove (Update Readouts)
            state.map.on('mousemove', (e) => {
                if(!els.readoutLatLon) return;
                const lat = e.latlng.lat.toFixed(5);
                const lng = e.latlng.lng.toFixed(5);
                els.readoutLatLon.textContent = `${lat}°, ${lng}°`;
                
                // Simple UTM Approximation for UI
                const zone = Math.floor((e.latlng.lng + 180) / 6) + 1;
                els.readoutUtm.textContent = `Zone ${zone}N`;
                
                // Approximate Pixel for UI (Requires actual bounds info to be accurate)
                // For now, we will just show Lat/Lon and UTM to fill the layout appropriately
                els.readoutPixel.textContent = `Lat: ${lat}, Lon: ${lng}`;
            });

            // Map Selector Buttons
            els.mapSelectors.forEach(btn => {
                const mapId = btn.getAttribute('data-s2-map');
                if(mapId === state.currentMapId) {
                    btn.classList.add('border-green-900', 'bg-green-900/20', 'text-green-400');
                    btn.classList.remove('border-gray-700', 'bg-[#1a1d24]', 'text-gray-300');
                }
                
                btn.addEventListener('click', async (e) => {
                    if (state.isOptimizing) return;
                    
                    els.mapSelectors.forEach(b => {
                        b.classList.remove('border-green-900', 'bg-green-900/20', 'text-green-400');
                        b.classList.add('border-gray-700', 'bg-[#1a1d24]', 'text-gray-300');
                    });
                    
                    const tBtn = e.currentTarget;
                    tBtn.classList.remove('border-gray-700', 'bg-[#1a1d24]', 'text-gray-300');
                    tBtn.classList.add('border-green-900', 'bg-green-900/20', 'text-green-400');

                    els.mapOverlay.style.display = 'flex';
                    const formData = new FormData();
                    formData.append('map_id', mapId);
                    
                    try {
                        const setRes = await fetch('/api/s2/set_map', { method: 'POST', body: formData });
                        if (setRes.ok) {
                            state.currentMapId = mapId;
                            loadTileLayer();
                            log(`Switched to map: ${mapId}`, 'success');
                        }
                    } catch (err) {
                        log(`Failed to switch map: ${err.message}`, 'error');
                    }
                    els.mapOverlay.style.display = 'none';
                });
            });

            // Layer Switchers (Radio)
            els.layerRadios.forEach(radio => {
                radio.addEventListener('change', (e) => {
                    state.currentLayer = e.target.value;
                    loadTileLayer();
                });
            });

            // Layer Mask (Checkbox)
            els.layerMaskCheck?.addEventListener('change', (e) => {
                if (e.target.checked) {
                    state.maskLayer = L.tileLayer(`/api/s2/tile/${state.currentMapId}/SCL/{z}/{x}/{y}.png`, {
                        maxZoom: 18, opacity: 0.6
                    }).addTo(state.map);
                } else {
                    if (state.maskLayer) {
                        state.map.removeLayer(state.maskLayer);
                        state.maskLayer = null;
                    }
                }
            });

            // Bottom Right Show Mask Button (Syncs with checkbox)
            els.btnToggleMask?.addEventListener('click', () => {
                if(els.layerMaskCheck) {
                    els.layerMaskCheck.checked = !els.layerMaskCheck.checked;
                    els.layerMaskCheck.dispatchEvent(new Event('change'));
                }
            });
            
            // Full Screen Button
            els.btnFullscreen?.addEventListener('click', () => {
                if (!document.fullscreenElement) {
                    els.mapWrapper.requestFullscreen().catch(err => {
                        log(`Error attempting to enable fullscreen: ${err.message}`, 'error');
                    });
                } else {
                    document.exitFullscreen();
                }
            });

            // Resize map on tab switch
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    if (btn.getAttribute('data-tab') === 's2-workspace') {
                        setTimeout(() => {
                            if (state.map) state.map.invalidateSize();
                        }, 100);
                    }
                });
            });

        } catch (error) {
            log(`Map init failed: ${error.message}`, 'error');
        }
        els.mapOverlay.style.display = 'none';
    };

    const loadTileLayer = () => {
        if (state.tileLayer) {
            state.map.removeLayer(state.tileLayer);
        }
        
        let url = '';
        if (state.currentLayer === 'osm') {
            url = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
        } else if (state.currentLayer === 'esri') {
            url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
        } else {
            // Local COG layers: tci, false_color, ndvi
            url = `/api/s2/tile/${state.currentMapId}/${state.currentLayer}/{z}/{x}/{y}.png`;
        }

        state.tileLayer = L.tileLayer(url, { maxZoom: 18 }).addTo(state.map);
        if (state.maskLayer) {
            state.maskLayer.bringToFront(); // Ensure mask is always on top
        }
    };
    
    // --- Optimization Settings Logic ---
    els.presetBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const t = e.currentTarget;
            
            // Reset all
            els.presetBtns.forEach(b => {
                b.classList.remove('active', 'bg-gradient-to-r', 'from-teal-900/40', 'to-blue-900/40', 'border-teal-500/50', 'text-white', 'shadow-[0_0_15px_rgba(20,184,166,0.15)]');
                b.classList.add('bg-[#1a1d24]', 'border-gray-700', 'text-gray-300');
            });
            
            // Set active
            t.classList.remove('bg-[#1a1d24]', 'border-gray-700', 'text-gray-300');
            t.classList.add('active', 'bg-gradient-to-r', 'from-teal-900/40', 'to-blue-900/40', 'border-teal-500/50', 'text-white', 'shadow-[0_0_15px_rgba(20,184,166,0.15)]');
            
            // Update hidden input and title
            const iters = t.getAttribute('data-iters');
            const label = t.getAttribute('data-label');
            els.inputIters.value = iters;
            
            let min = (iters / 250).toFixed(0);
            let max = (iters / 125).toFixed(0);
            els.itersText.textContent = `${iters} ITERS = ${min}-${max} MIN`;
        });
    });

    // --- Charts Initialization ---
    const initCharts = () => {
        if (window.Chart) {
            Chart.defaults.color = '#9ca3af';
            Chart.defaults.font.family = 'monospace';

            const ctxLoss = els.lossChart?.getContext('2d');
            if (ctxLoss) {
                state.lossChartInst = new Chart(ctxLoss, {
                    type: 'line',
                    data: { labels: [], datasets: [
                        { label: 'Total Loss', data: [], borderColor: '#3b82f6', tension: 0.2, pointRadius: 0, borderWidth: 2 },
                        { label: 'ASM Loss', data: [], borderColor: '#10b981', tension: 0.2, pointRadius: 0, borderWidth: 2 }
                    ]},
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true } } }
                });
            }

            const ctxMetric = els.metricChart?.getContext('2d');
            if (ctxMetric) {
                state.metricChartInst = new Chart(ctxMetric, {
                    type: 'line',
                    data: { labels: [], datasets: [
                        { label: 'PSNR', data: [], borderColor: '#f59e0b', tension: 0.2, pointRadius: 0, borderWidth: 2 },
                        { label: 'SSIM (x100)', data: [], borderColor: '#8b5cf6', tension: 0.2, pointRadius: 0, borderWidth: 2 }
                    ]},
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true } } }
                });
            }
        }
    };

    const updateCharts = (step, loss, l_asm, psnr, ssim) => {
        if (state.lossChartInst) {
            state.lossChartInst.data.labels.push(step);
            state.lossChartInst.data.datasets[0].data.push(loss);
            state.lossChartInst.data.datasets[1].data.push(l_asm);
            if (state.lossChartInst.data.labels.length > 50) {
                state.lossChartInst.data.labels.shift();
                state.lossChartInst.data.datasets.forEach(d => d.data.shift());
            }
            state.lossChartInst.update('none');
        }
        if (state.metricChartInst) {
            state.metricChartInst.data.labels.push(step);
            state.metricChartInst.data.datasets[0].data.push(psnr);
            state.metricChartInst.data.datasets[1].data.push(ssim * 100);
            if (state.metricChartInst.data.labels.length > 50) {
                state.metricChartInst.data.labels.shift();
                state.metricChartInst.data.datasets.forEach(d => d.data.shift());
            }
            state.metricChartInst.update('none');
        }
    };

    const initSlider = () => {
        if (!els.slider) return;
        els.slider.addEventListener('input', (e) => {
            const val = e.target.value;
            els.previewOutput.style.clipPath = `polygon(${val}% 0, 100% 0, 100% 100%, ${val}% 100%)`;
            els.sliderHandle.style.left = `${val}%`;
        });
    };

    els.btnRandom?.addEventListener('click', async () => {
        if (state.isOptimizing) return;
        const bounds = state.map.getBounds();
        const lat = bounds.getSouth() + Math.random() * (bounds.getNorth() - bounds.getSouth());
        const lon = bounds.getWest() + Math.random() * (bounds.getEast() - bounds.getWest());
        els.inputLat.value = lat.toFixed(6);
        els.inputLon.value = lon.toFixed(6);
        if (state.marker) state.map.removeLayer(state.marker);
        state.marker = L.marker([lat, lon]).addTo(state.map);
        state.map.panTo([lat, lon]);
        log(`Randomly selected Lat ${lat.toFixed(4)}, Lon ${lon.toFixed(4)}`);
        
        try {
            const size = els.inputSize.value;
            const cropUrl = `/api/s2/crop?lat=${lat}&lon=${lon}&crop_size=${size}&mask_source=scl`;
            const cropRes = await fetch(cropUrl);
            if (cropRes.ok) {
                const cropData = await cropRes.json();
                els.cropX.textContent = `${cropData.x_start}-${cropData.x_start + parseInt(size)}`;
                els.cropY.textContent = `${cropData.y_start}-${cropData.y_start + parseInt(size)}`;
            }
        } catch(e){}
    });

    els.btnStart?.addEventListener('click', async () => {
        if (state.isOptimizing) return;
        const lat = els.inputLat.value;
        const lon = els.inputLon.value;
        const size = els.inputSize.value;
        const iters = els.inputIters.value || 2000;
        const maskSource = els.inputMask.value;

        if (!lat || !lon) {
            log('Please select a point on the map first.', 'warn');
            return;
        }

        log('Fetching crop coordinates...', 'info');
        
        try {
            const cropUrl = `/api/s2/crop?lat=${lat}&lon=${lon}&crop_size=${size}&mask_source=${maskSource}`;
            const cropRes = await fetch(cropUrl);
            if (!cropRes.ok) throw new Error('Failed to get crop. Out of bounds?');
            const cropData = await cropRes.json();
            
            const startX = cropData.x_start;
            const startY = cropData.y_start;
            
            els.cropX.textContent = `${startX}-${startX + parseInt(size)}`;
            els.cropY.textContent = `${startY}-${startY + parseInt(size)}`;
            
            els.previewEmpty.style.display = 'none';
            els.previewInput.style.display = 'block';
            els.previewInput.src = cropData.cloudy_url || cropData.tci_url || '';
            els.previewOutput.style.display = 'block';
            els.slider.style.display = 'block';
            els.sliderHandle.style.display = 'block';

            log(`Crop acquired at (${startX}, ${startY}). Starting optimization...`, 'success');
            startOptimization(startX, startY, size, iters, maskSource);
            
        } catch (e) {
            log(e.message, 'error');
        }
    });

    const startOptimization = async (x, y, size, iters, maskSource) => {
        state.isOptimizing = true;
        els.btnStart.classList.add('opacity-50', 'pointer-events-none');
        els.activeControls.classList.remove('hidden');
        els.previewStatus.textContent = 'Optimizing...';
        els.previewStatus.className = 'text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full font-medium animate-pulse';

        if(state.lossChartInst) { state.lossChartInst.data.labels=[]; state.lossChartInst.data.datasets.forEach(d=>d.data=[]); state.lossChartInst.update(); }
        if(state.metricChartInst) { state.metricChartInst.data.labels=[]; state.metricChartInst.data.datasets.forEach(d=>d.data=[]); state.metricChartInst.update(); }

        state.controller = new AbortController();
        const url = `/api/s2/optimize?x_start=${x}&y_start=${y}&crop_size=${size}&iters=${iters}&mask_source=${maskSource}`;

        try {
            const response = await fetch(url, { signal: state.controller.signal });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(l => l.trim() !== '');
                
                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.type === 'progress') {
                            if (data.step % 10 === 0) {
                                log(`Step ${data.step}/${iters} | Loss: ${data.loss.toFixed(4)} | PSNR: ${data.metrics.psnr.toFixed(2)}`);
                            }
                            updateCharts(data.step, data.loss, data.metrics.l_asm, data.metrics.psnr, data.metrics.ssim);
                            
                            if (data.image) {
                                els.previewOutput.src = data.image;
                            }
                        } else if (data.type === 'error') {
                            log(`Error: ${data.message}`, 'error');
                        } else if (data.type === 'done') {
                            log(`Optimization completed successfully.`, 'success');
                        }
                    } catch(e) {
                        // ignore malformed JSON chunks
                    }
                }
            }
        } catch (e) {
            if (e.name === 'AbortError') log('Optimization stopped by user.', 'warn');
            else log(`Optimization error: ${e.message}`, 'error');
        } finally {
            state.isOptimizing = false;
            els.btnStart.classList.remove('opacity-50', 'pointer-events-none');
            els.activeControls.classList.add('hidden');
            els.previewStatus.textContent = 'Completed';
            els.previewStatus.className = 'text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded-full font-medium';
        }
    };

    els.btnStop?.addEventListener('click', async () => {
        if (state.controller) state.controller.abort();
        try {
            await fetch('/api/s2/optimize/cancel', { method: 'POST' });
        } catch(e) {}
    });

    els.btnPause?.addEventListener('click', async () => {
        const isPaused = els.btnPause.textContent.includes('Resume');
        if (isPaused) {
            await fetch('/api/s2/optimize/resume', { method: 'POST' });
            els.btnPause.innerHTML = '<i data-lucide="pause" class="w-4 h-4 mr-2"></i> Pause';
            log('Optimization resumed.', 'info');
        } else {
            await fetch('/api/s2/optimize/pause', { method: 'POST' });
            els.btnPause.innerHTML = '<i data-lucide="play" class="w-4 h-4 mr-2"></i> Resume';
            log('Optimization paused.', 'warn');
        }
        if (window.lucide) lucide.createIcons();
    });

    // Wait a short moment to ensure tabs are visible and dimensions are accurate
    setTimeout(() => {
        initMap();
        initCharts();
        initSlider();
    }, 200);

});
