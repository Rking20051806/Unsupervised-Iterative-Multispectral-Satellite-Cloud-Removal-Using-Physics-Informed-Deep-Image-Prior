import React, { useState, useEffect, useRef } from "react";
import { 
  Play, Square, RefreshCw, Zap, Image as ImageIcon, 
  Activity, Cpu, ChevronLeft, ChevronRight, 
  ZoomIn, ZoomOut, Sliders, Info
} from "lucide-react";
import ReactECharts from "echarts-for-react";

export default function App() {
  // Theme state
  const [darkMode, setDarkMode] = useState(true);
  
  // Dataset and image states
  const [dataset, setDataset] = useState<any[]>([]);
  const [selectedImage, setSelectedImage] = useState<string>("123");
  const [phase, setPhase] = useState<number>(1);
  const [iterations, setIterations] = useState<number>(5000);
  const [lr, setLr] = useState<number>(0.005);
  const [optimizer, setOptimizer] = useState<string>("Adam");
  const [maskSize, setMaskSize] = useState<number>(64);
  const [pinnLosses, setPinnLosses] = useState<string[]>(["ASM", "TV", "Physical Range"]);
  const [maskType, setMaskType] = useState<string>("Rectangle");
  const [maskCenter, setMaskCenter] = useState<{ x: number; y: number }>({ x: 128, y: 128 });
  const [physicsMetrics, setPhysicsMetrics] = useState<any>({
    asm_residual: 0,
    dcp_mean: 0,
    scc_score: 0,
    violation_rate: 0
  });
  const [cloudCover, setCloudCover] = useState<number | null>(null);
  const [maskDetected, setMaskDetected] = useState<boolean>(false);
  const [detecting, setDetecting] = useState<boolean>(false);
  
  // Training loop states
  const [status, setStatus] = useState<string>("Idle");
  const [currentIter, setCurrentIter] = useState<number>(0);
  const [elapsed, setElapsed] = useState<number>(0);
  const [eta, setEta] = useState<number>(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [gpuStats, setGpuStats] = useState<any>({ gpu_util: 0, vram_used: 0, vram_total: 4096, temp: 0 });
  
  // Metrics curves states
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [psnrHistory, setPsnrHistory] = useState<number[]>([]);
  const [ssimHistory, setSsimHistory] = useState<number[]>([]);
  const [stepsList, setStepsList] = useState<number[]>([]);
  
  // Live Images (Base64)
  const [images, setImages] = useState<any>({
    original: "",
    masked: "",
    mask: "",
    reconstructed: "",
    difference: ""
  });
  
  // Interactive UI states
  const [compareSliderVal, setCompareSliderVal] = useState<number>(50);
  const [zoomLevel, setZoomLevel] = useState<number>(100);
  const [hoverPixel, setHoverPixel] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [activeTab, setActiveTab] = useState<string>("workspace");
  
  const wsRef = useRef<WebSocket | null>(null);
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs console
  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Fetch Dataset
  useEffect(() => {
    fetch(`http://localhost:8000/api/dataset?phase=${phase}`)
      .then(res => res.json())
      .then(data => {
        if (data.images) {
          setDataset(data.images);
          if (data.images.length > 0) {
            const exists = data.images.some((img: any) => img.name === selectedImage);
            if (!exists) {
              setSelectedImage(data.images[0].name);
            }
          } else {
            setSelectedImage("");
          }
        }
      })
      .catch(err => console.error("Error fetching dataset:", err));
  }, [phase]);

  const loadPreview = () => {
    if (!selectedImage) return;
    
    setCloudCover(null);
    setMaskDetected(false);
    
    fetch(`http://localhost:8000/api/preview?image_name=${selectedImage}&phase=${phase}&mask_size=${maskSize}&mask_type=${maskType}&mask_x=${maskCenter.x}&mask_y=${maskCenter.y}`)
      .then(res => res.json())
      .then(data => {
        if (data.status === "success") {
          setImages(prev => ({
            ...prev,
            original: data.original,
            masked: data.masked,
            mask: data.mask,
            transmission: data.transmission,
            reconstructed: "", // Clear previous run reconstruction
            difference: ""
          }));
        }
      })
      .catch(err => console.error("Error loading preview:", err));
  };

  const selectRandomImage = () => {
    if (dataset.length === 0) return;
    const randomIndex = Math.floor(Math.random() * dataset.length);
    setSelectedImage(dataset[randomIndex].name);
  };

  const handleDetect = () => {
    if (!selectedImage) return;
    setDetecting(true);
    
    const config = {
      phase,
      image_name: selectedImage,
      mask_size: maskSize,
      mask_type: maskType,
      mask_center: [maskCenter.y, maskCenter.x]
    };
    
    fetch("http://localhost:8000/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    })
      .then(res => res.json())
      .then(data => {
        setDetecting(false);
        if (data.status === "success" || data.cloud_cover !== undefined) {
          setCloudCover(data.cloud_cover);
          setMaskDetected(true);
          setImages(prev => ({
            ...prev,
            mask: data.mask,
            masked: data.masked,
            transmission: data.transmission
          }));
        } else {
          alert(data.message || "Error detecting clouds.");
        }
      })
      .catch(err => {
        setDetecting(false);
        console.error("Error detecting clouds:", err);
      });
  };

  useEffect(() => {
    loadPreview();
  }, [selectedImage, phase, maskSize, maskType, maskCenter]);


  // Poll GPU Status
  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/gpu")
        .then(res => res.json())
        .then(data => {
          if (data.status === "success") {
            setGpuStats(data);
          }
        })
        .catch(err => console.error("Error fetching GPU stats:", err));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket Connection
  useEffect(() => {
    const connectWS = () => {
      const ws = new WebSocket("ws://localhost:8000/ws");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.status) setStatus(data.status);
        if (data.iteration !== undefined) setCurrentIter(data.iteration);
        if (data.elapsed !== undefined) setElapsed(data.elapsed);
        if (data.eta !== undefined) setEta(data.eta);
        
        if (data.log) {
          setLogs(prev => [...prev, data.log]);
        }
        
        if (data.metrics) {
          setLossHistory(prev => [...prev, data.metrics.loss]);
          setPsnrHistory(prev => [...prev, data.metrics.psnr]);
          setSsimHistory(prev => [...prev, data.metrics.ssim]);
          setStepsList(prev => [...prev, data.iteration]);
          setPhysicsMetrics({
            asm_residual: data.metrics.asm_residual ?? 0,
            dcp_mean: data.metrics.dcp_mean ?? 0,
            scc_score: data.metrics.scc_score ?? 0,
            violation_rate: data.metrics.violation_rate ?? 0
          });
        }
        
        if (data.images) {
          setImages(data.images);
        }
      };
      
      ws.onclose = () => {
        setTimeout(connectWS, 3000);
      };
      
      wsRef.current = ws;
    };
    
    connectWS();
    return () => {
      wsRef.current?.close();
    };
  }, []);

  // Start Training Handler
  const handleStart = (resume: boolean = false) => {
    // Reset history charts only if starting fresh
    if (!resume) {
      setLossHistory([]);
      setPsnrHistory([]);
      setSsimHistory([]);
      setStepsList([]);
      setLogs([]);
      setPhysicsMetrics({
        asm_residual: 0,
        dcp_mean: 0,
        scc_score: 0,
        violation_rate: 0
      });
    }
    
    const config = {
      phase,
      image_name: selectedImage,
      iterations,
      lr,
      optimizer,
      tv_weight: 1e-4,
      physics_weight: 1e-3,
      mask_size: maskSize,
      pinn_losses: pinnLosses,
      mask_type: maskType,
      mask_center: [maskCenter.y, maskCenter.x],
      resume: resume
    };
    
    fetch("http://localhost:8000/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    })
      .then(res => res.json())
      .then(data => {
        if (data.status === "success") {
          setLogs(prev => [...prev, `[SYSTEM] ${resume ? "Resuming" : "Starting"} optimization for image ${selectedImage}.`]);
        } else {
          alert(data.message);
        }
      });
  };

  // Stop Training Handler
  const handleStop = () => {
    fetch("http://localhost:8000/api/stop", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        if (data.status === "success") {
          setLogs(prev => [...prev, "[SYSTEM] Stop signal sent to training thread."]);
        }
      });
  };

  // Select Image
  const selectNextImage = () => {
    const idx = dataset.findIndex(img => img.name === selectedImage);
    if (idx !== -1 && idx < dataset.length - 1) {
      setSelectedImage(dataset[idx + 1].name);
    }
  };

  const selectPrevImage = () => {
    const idx = dataset.findIndex(img => img.name === selectedImage);
    if (idx > 0) {
      setSelectedImage(dataset[idx - 1].name);
    }
  };

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!e.currentTarget) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 256);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 256);
    setMaskCenter({ x, y });
  };

  const handleImageHover = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!e.currentTarget) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 256);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 256);
    // Dummy pixel value computation for visual demonstration
    const r = Math.round((x + y) % 255);
    const g = Math.round((x * 2) % 255);
    const b = Math.round((y * 2) % 255);
    setHoverPixel({ x, y, r, g, b });
  };

  const filteredDataset = dataset.filter(img => 
    img.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // ECharts Configurations
  const lossChartOption = {
    backgroundColor: "transparent",
    textStyle: { color: darkMode ? "#cbd5e1" : "#0f172a" },
    grid: { left: 45, right: 15, top: 35, bottom: 25 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: stepsList },
    yAxis: { type: "value", scale: true },
    series: [
      {
        name: "Loss",
        type: "line",
        data: lossHistory,
        lineStyle: { color: "#3b82f6", width: 2 },
        itemStyle: { color: "#3b82f6" },
        showSymbol: false
      }
    ]
  };

  const metricChartOption = {
    backgroundColor: "transparent",
    textStyle: { color: darkMode ? "#cbd5e1" : "#0f172a" },
    grid: { left: 45, right: 45, top: 35, bottom: 25 },
    tooltip: { trigger: "axis" },
    legend: { data: ["PSNR (dB)", "SSIM"], textStyle: { color: darkMode ? "#cbd5e1" : "#0f172a" } },
    xAxis: { type: "category", data: stepsList },
    yAxis: [
      { type: "value", name: "PSNR", scale: true },
      { type: "value", name: "SSIM", min: 0, max: 1 }
    ],
    series: [
      {
        name: "PSNR (dB)",
        type: "line",
        data: psnrHistory,
        lineStyle: { color: "#10b981", width: 2 },
        itemStyle: { color: "#10b981" },
        showSymbol: false
      },
      {
        name: "SSIM",
        type: "line",
        yAxisIndex: 1,
        data: ssimHistory,
        lineStyle: { color: "#8b5cf6", width: 2 },
        itemStyle: { color: "#8b5cf6" },
        showSymbol: false
      }
    ]
  };

  return (
    <div className={`min-h-screen ${darkMode ? "dark bg-darkBg text-zinc-100" : "bg-zinc-50 text-zinc-950"} flex flex-col font-sans transition-colors duration-300`}>
      {/* Header bar */}
      <header className="border-b border-zinc-200 dark:border-zinc-800 px-6 py-4 flex items-center justify-between bg-white dark:bg-darkCard">
        <div className="flex items-center gap-3">
          <Zap className="h-6 w-6 text-emerald-500 fill-emerald-500/20" />
          <h1 className="text-xl font-extrabold tracking-tight">Physics-Guided Image Reconstruction Dashboard</h1>
          <span className="text-xs bg-zinc-100 dark:bg-zinc-800 text-zinc-500 px-2 py-0.5 rounded-full font-mono">DIP + PINN</span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs text-zinc-500 font-mono">
            <span>GPU: GTX 1650</span>
            <span className="text-emerald-500 font-bold">✓ CUDA Active</span>
          </div>
          <button 
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-lg bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>
      </header>

      {/* Main layout wrapper */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Sidebar: Dataset Explorer */}
        <section className="w-[300px] border-r border-zinc-200 dark:border-zinc-800 flex flex-col bg-white dark:bg-darkCard">
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-800">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 mb-2">Dataset Explorer</h2>
            
            {/* Phase 1 & Phase 2 Mode Toggle */}
            <div className="flex bg-zinc-100 dark:bg-zinc-800/80 p-1 rounded-lg mb-3">
              <button
                type="button"
                onClick={() => setPhase(1)}
                className={`flex-1 py-1.5 px-2 rounded-md text-xs font-bold transition-all ${
                  phase === 1
                    ? "bg-emerald-500 text-white shadow-sm"
                    : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
                }`}
              >
                Phase 1 (Synthetic)
              </button>
              <button
                type="button"
                onClick={() => setPhase(2)}
                className={`flex-1 py-1.5 px-2 rounded-md text-xs font-bold transition-all ${
                  phase === 2
                    ? "bg-emerald-500 text-white shadow-sm"
                    : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
                }`}
              >
                Phase 2 (Real Cloud)
              </button>
            </div>

            <input 
              type="text" 
              placeholder="Search images..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-950 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-2 custom-scrollbar">
            {filteredDataset.map((img) => (
              <div 
                key={img.name}
                onClick={() => setSelectedImage(img.name)}
                className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                  selectedImage === img.name 
                    ? "bg-emerald-500/10 border-emerald-500 text-emerald-600 dark:text-emerald-400" 
                    : "border-zinc-100 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-semibold">{img.filename}</span>
                  <span className="text-[10px] text-zinc-500">512 x 512</span>
                </div>
                <div className="flex gap-2 mt-1 text-[10px] text-zinc-400">
                  <span>Free: {Math.round(img.free_size / 1024)} KB</span>
                  <span>Cloud: {Math.round(img.cloud_size / 1024)} KB</span>
                </div>
              </div>
            ))}
          </div>
          <div className="p-4 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-[#09090b]/20 flex gap-2">
            <button 
              onClick={selectPrevImage} 
              className="flex-1 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 py-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-700 transition-colors flex items-center justify-center"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button 
              onClick={selectNextImage} 
              className="flex-1 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 py-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-700 transition-colors flex items-center justify-center"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </section>

        {/* Center Panel: Workspace */}
        <section className="flex-1 flex flex-col bg-zinc-50 dark:bg-[#09090b] overflow-y-auto custom-scrollbar p-6 space-y-6">
          {/* Dashboard HUD */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex items-center gap-3">
              <Cpu className="h-8 w-8 text-emerald-500" />
              <div>
                <span className="text-[11px] text-zinc-400 uppercase tracking-wider block font-semibold">Active Status</span>
                <span className="text-lg font-bold font-mono">{status}</span>
              </div>
            </div>
            <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex items-center gap-3">
              <Activity className="h-8 w-8 text-blue-500" />
              <div>
                <span className="text-[11px] text-zinc-400 uppercase tracking-wider block font-semibold">Iterations</span>
                <span className="text-lg font-bold font-mono">{currentIter} / {iterations}</span>
              </div>
            </div>
            <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex items-center gap-3">
              <Sliders className="h-8 w-8 text-orange-500" />
              <div>
                <span className="text-[11px] text-zinc-400 uppercase tracking-wider block font-semibold">Time Elapsed</span>
                <span className="text-lg font-bold font-mono">{Math.floor(elapsed / 60)}m {elapsed % 60}s</span>
              </div>
            </div>
            <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex items-center gap-3">
              <Info className="h-8 w-8 text-pink-500" />
              <div>
                <span className="text-[11px] text-zinc-400 uppercase tracking-wider block font-semibold">Estimated ETA</span>
                <span className="text-lg font-bold font-mono">{Math.floor(eta / 60)}m {eta % 60}s</span>
              </div>
            </div>
          </div>

          {/* Primary View Selector */}
          <div className="flex gap-4">
            <button 
              onClick={() => setActiveTab("workspace")}
              className={`pb-2 text-sm font-semibold border-b-2 transition-all ${
                activeTab === "workspace" 
                  ? "border-emerald-500 text-emerald-500" 
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              4-Panel Image Grid
            </button>
            <button 
              onClick={() => setActiveTab("compare")}
              className={`pb-2 text-sm font-semibold border-b-2 transition-all ${
                activeTab === "compare" 
                  ? "border-emerald-500 text-emerald-500" 
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Side-by-Side Comparison Slider
            </button>
            <button 
              onClick={() => setActiveTab("scientific")}
              className={`pb-2 text-sm font-semibold border-b-2 transition-all ${
                activeTab === "scientific" 
                  ? "border-emerald-500 text-emerald-500" 
                  : "border-transparent text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Scientific Telemetry & Validation
            </button>
          </div>

          {/* Workspace Views */}
          {activeTab === "workspace" && (
            phase === 1 ? (
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative group">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Original Reference</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg relative cursor-crosshair">
                    {images.original ? (
                      <img 
                        src={`data:image/png;base64,${images.original}`} 
                        alt="Original" 
                        onClick={handleImageClick}
                        onMouseMove={handleImageHover}
                        className="w-full h-full object-cover transition-transform" 
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                    {maskCenter && (
                      <div 
                        className="absolute border-2 border-red-500 bg-red-500/30 pointer-events-none transition-all"
                        style={{
                          left: `${(maskCenter.x / 256) * 100}%`,
                          top: `${(maskCenter.y / 256) * 100}%`,
                          width: `${(maskSize / 256) * 100}%`,
                          height: `${(maskSize / 256) * 100}%`,
                          transform: "translate(-50%, -50%)",
                          borderRadius: maskType === "Circle" ? "50%" : "0%"
                        }}
                      />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Masked Input</span>
                  {cloudCover !== null && (
                    <span className="absolute top-2 right-2 bg-teal-500 text-black text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">
                      Cover: {cloudCover.toFixed(2)}%
                    </span>
                  )}
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.masked ? (
                      <img 
                        src={`data:image/png;base64,${images.masked}`} 
                        alt="Masked" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Live Reconstruction</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.reconstructed ? (
                      <img 
                        src={`data:image/png;base64,${images.reconstructed}`} 
                        alt="Reconstructed" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Difference Heatmap</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.difference ? (
                      <img 
                        src={`data:image/png;base64,${images.difference}`} 
                        alt="Difference Heatmap" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-6">
                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Cloud Free Reference</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.original ? (
                      <img 
                        src={`data:image/png;base64,${images.original}`} 
                        alt="Original" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Cloudy Input</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.masked ? (
                      <img 
                        src={`data:image/png;base64,${images.masked}`} 
                        alt="Cloudy" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">DL Cloud Mask</span>
                  {cloudCover !== null && (
                    <span className="absolute top-2 right-2 bg-teal-500 text-black text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">
                      Cover: {cloudCover.toFixed(2)}%
                    </span>
                  )}
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.mask ? (
                      <img 
                        src={`data:image/png;base64,${images.mask}`} 
                        alt="DL Mask" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Transmission Map</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.transmission ? (
                      <img 
                        src={`data:image/png;base64,${images.transmission}`} 
                        alt="Transmission" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Live Reconstruction</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.reconstructed ? (
                      <img 
                        src={`data:image/png;base64,${images.reconstructed}`} 
                        alt="Reconstructed" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm relative">
                  <span className="absolute top-2 left-2 bg-zinc-950/75 text-zinc-100 text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-wider z-10">Difference Heatmap</span>
                  <div className="aspect-square flex items-center justify-center overflow-hidden bg-zinc-900 rounded-lg">
                    {images.difference ? (
                      <img 
                        src={`data:image/png;base64,${images.difference}`} 
                        alt="Difference Heatmap" 
                        className="w-full h-full object-cover"
                        style={{ transform: `scale(${zoomLevel / 100})` }}
                      />
                    ) : (
                      <ImageIcon className="h-12 w-12 text-zinc-700" />
                    )}
                  </div>
                </div>
              </div>
            )
          )}
          {activeTab === "compare" && (
            <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex flex-col items-center">
              <div className="w-full max-w-[500px] aspect-square relative overflow-hidden bg-zinc-900 rounded-lg" ref={imageRef}>
                {images.reconstructed && images.original ? (
                  <>
                    <img 
                      src={`data:image/png;base64,${images.original}`} 
                      alt="Original" 
                      className="w-full h-full object-cover absolute top-0 left-0" 
                    />
                    <div 
                      className="absolute top-0 right-0 h-full overflow-hidden border-l-2 border-emerald-500"
                      style={{ width: `${100 - compareSliderVal}%` }}
                    >
                      <img 
                        src={`data:image/png;base64,${images.reconstructed}`} 
                        alt="Reconstructed" 
                        className="h-full object-cover" 
                        style={{ 
                          width: imageRef.current?.clientWidth || 500,
                          maxWidth: 'none',
                          transform: `translateX(-${compareSliderVal}%)`
                        }}
                      />
                    </div>
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-zinc-600">
                    <ImageIcon className="h-12 w-12" />
                    <span className="ml-2 text-xs">Run optimization to view slide comparison</span>
                  </div>
                )}
              </div>
              <div className="w-full max-w-[500px] mt-4 flex items-center gap-3">
                <span className="text-xs text-zinc-400 font-semibold uppercase">Original</span>
                <input 
                  type="range" 
                  min="0" 
                  max="100" 
                  value={compareSliderVal}
                  onChange={(e) => setCompareSliderVal(Number(e.target.value))}
                  className="flex-1 accent-emerald-500" 
                />
                <span className="text-xs text-zinc-400 font-semibold uppercase">Reconstructed</span>
              </div>
            </div>
          )}
          {activeTab === "scientific" && (
            <div className="space-y-6">
              {/* Introduction Card */}
              <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 p-6 rounded-xl border border-emerald-500/20 shadow-sm">
                <h3 className="text-base font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
                  <Zap className="h-5 w-5 fill-emerald-400/20 text-emerald-500 dark:text-emerald-400" />
                  Physics-Based Verification & Validation (P&V)
                </h3>
                <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2 leading-relaxed">
                  Unlike purely data-driven deep learning models which might construct visually pleasing but physically impossible ground reflectance, 
                  Physics-Informed Neural Networks (PINNs) enforce consistency constraints derived from optical physics and satellite sensors. 
                  Below are the live scientific telemetry metrics checking the realism of the reconstructed cloud-free image.
                </p>
              </div>

              {/* 2x2 Grid of Physics Metrics */}
              <div className="grid grid-cols-2 gap-6">
                {/* ASM Card */}
                <div className="bg-white dark:bg-darkCard p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-500 dark:text-zinc-400 font-bold uppercase tracking-wider">ASM Residual Consistency</span>
                      {physicsMetrics.asm_residual < 0.005 ? (
                        <span className="text-[10px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded-full font-bold">✓ Optimal</span>
                      ) : (
                        <span className="text-[10px] bg-amber-500/10 text-amber-600 dark:text-amber-400 px-2 py-0.5 rounded-full font-bold">⚠ Calibrating</span>
                      )}
                    </div>
                    <div className="mt-3 flex items-baseline gap-2">
                      <span className="font-mono text-3xl font-extrabold tracking-tight text-emerald-600 dark:text-emerald-400">
                        {physicsMetrics.asm_residual.toFixed(6)}
                      </span>
                      <span className="text-xs text-zinc-500 font-mono">MSE</span>
                    </div>
                  </div>
                  <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3">
                    <span className="text-[10px] font-semibold text-zinc-400 block font-mono">Formula: I(x) = J(x)t(x) + A(1-t(x))</span>
                    <p className="text-[11px] text-zinc-500 mt-1 leading-relaxed">
                      Measures consistency against the Atmospheric Scattering Model. Evaluates whether the reconstructed clear surface ($J$), when attenuated by the estimated transmission ($t$), mathematically recreates the observed cloudy input ($I$).
                    </p>
                  </div>
                </div>

                {/* DCP Card */}
                <div className="bg-white dark:bg-darkCard p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-500 dark:text-zinc-400 font-bold uppercase tracking-wider">Dark Channel Prior (DCP) Density</span>
                      {physicsMetrics.dcp_mean < 0.15 ? (
                        <span className="text-[10px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded-full font-bold">✓ Haze-Free</span>
                      ) : (
                        <span className="text-[10px] bg-amber-500/10 text-amber-600 dark:text-amber-400 px-2 py-0.5 rounded-full font-bold">⚠ Haze Remnant</span>
                      )}
                    </div>
                    <div className="mt-3 flex items-baseline gap-2">
                      <span className="font-mono text-3xl font-extrabold tracking-tight text-emerald-600 dark:text-emerald-400">
                        {physicsMetrics.dcp_mean.toFixed(5)}
                      </span>
                      <span className="text-xs text-zinc-500 font-mono">Mean Intensity</span>
                    </div>
                  </div>
                  <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3">
                    <span className="text-[10px] font-semibold text-zinc-400 block font-mono">Formula: min_c(min_patch(J^c))</span>
                    <p className="text-[11px] text-zinc-500 mt-1 leading-relaxed">
                      Checks for haze and cloud residue. Clear natural scenes always contain some pixels with near-zero intensity in at least one channel (due to shadows, trees). Elevated values indicate residual cloud or scattering particles.
                    </p>
                  </div>
                </div>

                {/* SCC Card */}
                <div className="bg-white dark:bg-darkCard p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-500 dark:text-zinc-400 font-bold uppercase tracking-wider">Spectral Cross-Correlation (SCC)</span>
                      {physicsMetrics.scc_score > 0.75 ? (
                        <span className="text-[10px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded-full font-bold">✓ Correlated</span>
                      ) : physicsMetrics.scc_score > 0.5 ? (
                        <span className="text-[10px] bg-amber-500/10 text-amber-600 dark:text-amber-400 px-2 py-0.5 rounded-full font-bold">⚠ Decoupled</span>
                      ) : (
                        <span className="text-[10px] bg-rose-500/10 text-rose-600 dark:text-rose-400 px-2 py-0.5 rounded-full font-bold">✗ Abnormal</span>
                      )}
                    </div>
                    <div className="mt-3 flex items-baseline gap-2">
                      <span className="font-mono text-3xl font-extrabold tracking-tight text-emerald-600 dark:text-emerald-400">
                        {physicsMetrics.scc_score.toFixed(4)}
                      </span>
                      <span className="text-xs text-zinc-500 font-mono">Avg r-value</span>
                    </div>
                  </div>
                  <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3">
                    <span className="text-[10px] font-semibold text-zinc-400 block font-mono">Formula: E[r(R,G) + r(G,B) + r(R,B)] / 3</span>
                    <p className="text-[11px] text-zinc-500 mt-1 leading-relaxed">
                      Measures the spatial co-dependence of RGB bands in the inpainted zone. Natural satellite ground features maintain high correlation across bands. Low correlation indicates unnatural color shift artifacts.
                    </p>
                  </div>
                </div>

                {/* Reflectance Card */}
                <div className="bg-white dark:bg-darkCard p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-500 dark:text-zinc-400 font-bold uppercase tracking-wider">Reflectance Bound Compliance</span>
                      {physicsMetrics.violation_rate === 0 ? (
                        <span className="text-[10px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded-full font-bold">✓ 100% Bound</span>
                      ) : (
                        <span className="text-[10px] bg-amber-500/10 text-amber-600 dark:text-amber-400 px-2 py-0.5 rounded-full font-bold">⚠ Out of Bound</span>
                      )}
                    </div>
                    <div className="mt-3 flex items-baseline gap-2">
                      <span className="font-mono text-3xl font-extrabold tracking-tight text-emerald-600 dark:text-emerald-400">
                        {(100.0 - physicsMetrics.violation_rate).toFixed(2)}%
                      </span>
                      <span className="text-xs text-zinc-500 font-mono">Valid Reflectance</span>
                    </div>
                  </div>
                  <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3">
                    <span className="text-[10px] font-semibold text-zinc-400 block font-mono">Formula: x ∈ [0.0, 1.0]</span>
                    <p className="text-[11px] text-zinc-500 mt-1 leading-relaxed">
                      Ensures that the output pixel intensities align with physical limits of solar reflectance. DIP networks can occasionally generate values outside physical limits during active optimization, which are penalized by the PINN constraints.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Interactive Zoom/Inspector HUD */}
          <div className="bg-white dark:bg-darkCard p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-xs text-zinc-400 font-bold uppercase tracking-wider">Canvas Zoom:</span>
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => setZoomLevel(prev => Math.max(50, prev - 25))}
                  className="p-1.5 rounded bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>
                <span className="text-xs font-mono w-12 text-center">{zoomLevel}%</span>
                <button 
                  onClick={() => setZoomLevel(prev => Math.min(200, prev + 25))}
                  className="p-1.5 rounded bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="flex gap-6 font-mono text-xs text-zinc-400">
              {hoverPixel ? (
                <>
                  <span>X: <strong className="text-zinc-100">{hoverPixel.x}</strong> Y: <strong className="text-zinc-100">{hoverPixel.y}</strong></span>
                  <span>RGB: <strong className="text-zinc-100">({hoverPixel.r}, {hoverPixel.g}, {hoverPixel.b})</strong></span>
                  <span>Intensity: <strong className="text-zinc-100">{Math.round((hoverPixel.r + hoverPixel.g + hoverPixel.b) / 3)}</strong></span>
                </>
              ) : (
                <span>Hover over original image to inspect coordinates & RGB values</span>
              )}
            </div>
          </div>
        </section>

        {/* Right Panel: Control panel & live charts */}
        <section className="w-[450px] border-l border-zinc-200 dark:border-zinc-800 flex flex-col bg-white dark:bg-darkCard overflow-y-auto custom-scrollbar p-6 space-y-6">
          {/* Control Settings */}
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 mb-3">Reconstruction Control Settings</h2>
            <div className="space-y-4">
              {/* Phase Switch */}
              <div>
                <label className="text-xs font-medium text-zinc-500 mb-1.5 block">Pipeline Phase</label>
                <div className="flex gap-2">
                  <button 
                    onClick={() => setPhase(1)}
                    className={`flex-1 py-1.5 rounded-lg border text-xs font-semibold ${
                      phase === 1 
                        ? "border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" 
                        : "border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                    }`}
                  >
                    Phase 1 (Synthetic)
                  </button>
                  <button 
                    onClick={() => setPhase(2)}
                    className={`flex-1 py-1.5 rounded-lg border text-xs font-semibold ${
                      phase === 2 
                        ? "border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" 
                        : "border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                    }`}
                  >
                    Phase 2 (Real Cloud)
                  </button>
                </div>
              </div>

              {/* Optimizer and Iterations */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-medium text-zinc-500 mb-1.5 block">Optimizer</label>
                  <select 
                    value={optimizer}
                    onChange={(e) => setOptimizer(e.target.value)}
                    className="w-full bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-950 dark:text-zinc-100 focus:outline-none"
                  >
                    <option value="Adam">Adam</option>
                    <option value="AdamW">AdamW</option>
                    <option value="SGD">SGD</option>
                    <option value="RMSProp">RMSProp</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium text-zinc-500 mb-1.5 block">Iterations</label>
                  <input 
                    type="number" 
                    value={iterations}
                    onChange={(e) => setIterations(Number(e.target.value))}
                    className="w-full bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-950 dark:text-zinc-100 focus:outline-none"
                  />
                </div>
              </div>

              {/* Learning Rate and Mask Size */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-medium text-zinc-500 mb-1.5 block">Learning Rate</label>
                  <select 
                    value={lr}
                    onChange={(e) => setLr(Number(e.target.value))}
                    className="w-full bg-zinc-50 dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-950 dark:text-zinc-100 focus:outline-none"
                  >
                    <option value="0.01">0.01</option>
                    <option value="0.005">0.005</option>
                    <option value="0.001">0.001</option>
                    <option value="0.0005">0.0005</option>
                  </select>
                </div>
                {phase === 1 && (
                  <div>
                    <label className="text-xs font-medium text-zinc-500 mb-1.5 block">Mask Size</label>
                    <input 
                      type="range" 
                      min="32" 
                      max="128" 
                      value={maskSize}
                      onChange={(e) => setMaskSize(Number(e.target.value))}
                      className="w-full accent-emerald-500 mt-2" 
                    />
                    <div className="text-[10px] text-zinc-500 text-right mt-0.5">{maskSize}x{maskSize} px</div>
                  </div>
                )}
              </div>

              {/* Mask Shape Switch */}
              {phase === 1 && (
                <div>
                  <label className="text-xs font-medium text-zinc-500 mb-1.5 block">Mask Shape</label>
                  <div className="flex gap-2">
                    <button 
                      onClick={() => setMaskType("Rectangle")}
                      className={`flex-1 py-1.5 rounded-lg border text-xs font-semibold ${
                        maskType === "Rectangle" 
                          ? "border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" 
                          : "border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                      }`}
                    >
                      Rectangle
                    </button>
                    <button 
                      onClick={() => setMaskType("Circle")}
                      className={`flex-1 py-1.5 rounded-lg border text-xs font-semibold ${
                        maskType === "Circle" 
                          ? "border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" 
                          : "border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800"
                      }`}
                    >
                      Circle
                    </button>
                  </div>
                </div>
              )}

              {/* Physics loss terms */}
              <div>
                <label className="text-xs font-medium text-zinc-500 mb-1.5 block">Physics-guided Constraints (PINN)</label>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {["ASM", "TV", "Physical Range", "DCP", "SCC"].map((loss) => (
                    <label key={loss} className="flex items-center gap-2 cursor-pointer">
                      <input 
                        type="checkbox"
                        checked={pinnLosses.includes(loss)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setPinnLosses(prev => [...prev, loss]);
                          } else {
                            setPinnLosses(prev => prev.filter(item => item !== loss));
                          }
                        }}
                        className="accent-emerald-500"
                      />
                      <span>{loss}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Execution Actions */}
              <div className="space-y-2 pt-2">
                <button 
                  onClick={handleDetect}
                  disabled={status === "Training" || detecting}
                  className="w-full bg-teal-500 hover:bg-teal-600 disabled:bg-teal-800/40 text-black py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
                >
                  <RefreshCw className={`h-4 w-4 ${detecting ? "animate-spin" : ""}`} />
                  <span>{detecting ? "Running Cloud Detection..." : "1. Detect Clouds & Masking"}</span>
                </button>
                
                <div className="flex gap-2">
                  <button 
                    onClick={() => handleStart(false)}
                    disabled={status === "Training" || !maskDetected}
                    className="flex-1 bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-800/40 disabled:text-zinc-500 text-black py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
                  >
                    <Play className="h-4 w-4 fill-black" />
                    <span>2. Remove Clouds</span>
                  </button>
                  <button 
                    onClick={() => handleStart(true)}
                    disabled={status === "Training" || stepsList.length === 0 || !maskDetected}
                    className="flex-1 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-800/40 disabled:text-zinc-400 text-white py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
                  >
                    <RefreshCw className="h-4 w-4" />
                    <span>Resume</span>
                  </button>
                  <button 
                    onClick={handleStop}
                    disabled={status !== "Training"}
                    className="flex-1 bg-rose-500 hover:bg-rose-600 disabled:bg-rose-800/40 text-white py-2 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
                  >
                    <Square className="h-4 w-4 fill-white" />
                    <span>Stop</span>
                  </button>
                </div>
              </div>
              <div className="pt-2">
                <button 
                  onClick={selectRandomImage}
                  disabled={status === "Training"}
                  className="w-full bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700 disabled:opacity-50 text-zinc-800 dark:text-zinc-200 py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors"
                >
                  <ImageIcon className="h-4 w-4" />
                  <span>Take New Image (Random)</span>
                </button>
              </div>
            </div>
          </div>

          {/* GPU Health HUD */}
          <div className="border-t border-zinc-200 dark:border-zinc-800 pt-6">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 mb-3">GPU System Telemetry</h2>
            <div className="space-y-3 font-mono text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <span>GPU Core Utilization</span>
                  <span>{gpuStats.gpu_util}%</span>
                </div>
                <div className="w-full bg-zinc-200 dark:bg-zinc-800 h-2 rounded-full overflow-hidden">
                  <div className="bg-emerald-500 h-full" style={{ width: `${gpuStats.gpu_util}%` }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span>GPU VRAM Usage</span>
                  <span>{gpuStats.vram_used} MB / {gpuStats.vram_total} MB</span>
                </div>
                <div className="w-full bg-zinc-200 dark:bg-zinc-800 h-2 rounded-full overflow-hidden">
                  <div className="bg-blue-500 h-full" style={{ width: `${(gpuStats.vram_used / gpuStats.vram_total) * 100}%` }} />
                </div>
              </div>
              <div className="flex justify-between">
                <span>GPU Temperature</span>
                <span className={gpuStats.temp > 75 ? "text-rose-500 font-bold" : "text-emerald-500 font-bold"}>{gpuStats.temp} °C</span>
              </div>
            </div>
          </div>

          {/* Live Physics Telemetry HUD */}
          <div className="border-t border-zinc-200 dark:border-zinc-800 pt-6">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 mb-3">Live Physics Telemetry HUD</h2>
            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="bg-zinc-50 dark:bg-[#09090b] p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm">
                <span className="text-[10px] text-zinc-500 uppercase font-bold block">ASM Residual</span>
                <span className="text-sm font-bold text-emerald-500 dark:text-emerald-400 mt-1 block">{physicsMetrics.asm_residual.toFixed(5)}</span>
              </div>
              <div className="bg-zinc-50 dark:bg-[#09090b] p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm">
                <span className="text-[10px] text-zinc-500 uppercase font-bold block">DCP Density</span>
                <span className="text-sm font-bold text-emerald-500 dark:text-emerald-400 mt-1 block">{physicsMetrics.dcp_mean.toFixed(4)}</span>
              </div>
              <div className="bg-zinc-50 dark:bg-[#09090b] p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm">
                <span className="text-[10px] text-zinc-500 uppercase font-bold block">Spectral Corr</span>
                <span className="text-sm font-bold text-emerald-500 dark:text-emerald-400 mt-1 block">{physicsMetrics.scc_score.toFixed(3)}</span>
              </div>
              <div className="bg-zinc-50 dark:bg-[#09090b] p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm">
                <span className="text-[10px] text-zinc-500 uppercase font-bold block">Reflectance Bounds</span>
                <span className="text-sm font-bold text-emerald-500 dark:text-emerald-400 mt-1 block">{(100.0 - physicsMetrics.violation_rate).toFixed(2)}%</span>
              </div>
            </div>
          </div>

          {/* Loss & Metric Charts */}
          <div className="border-t border-zinc-200 dark:border-zinc-800 pt-6 space-y-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500">TensorBoard-style Metrics</h2>
            <div>
              <label className="text-xs font-medium text-zinc-400 block mb-2">Total Optimization Loss</label>
              <div className="bg-zinc-50 dark:bg-[#09090b] rounded-lg border border-zinc-200 dark:border-zinc-800 p-2 h-[150px]">
                <ReactECharts option={lossChartOption} style={{ height: "100%", width: "100%" }} />
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-zinc-400 block mb-2">PSNR Progress (dB)</label>
              <div className="bg-zinc-50 dark:bg-[#09090b] rounded-lg border border-zinc-200 dark:border-zinc-800 p-2 h-[150px]">
                <ReactECharts option={metricChartOption} style={{ height: "100%", width: "100%" }} />
              </div>
            </div>
          </div>

          {/* Live Logs console */}
          <div className="border-t border-zinc-200 dark:border-zinc-800 pt-6 flex-1 flex flex-col min-h-[200px]">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 mb-2">Live Training Console Logs</h2>
            <div className="flex-1 bg-zinc-950 rounded-lg p-3 font-mono text-[10px] text-zinc-300 overflow-y-auto max-h-[220px] custom-scrollbar border border-zinc-200 dark:border-zinc-800">
              {logs.map((log, idx) => (
                <div key={idx} className="border-b border-zinc-900 py-1 leading-relaxed">
                  {log}
                </div>
              ))}
              <div ref={consoleEndRef} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
