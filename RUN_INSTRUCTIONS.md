# 🚀 Complete Project Execution & Setup Guide (RUN_INSTRUCTIONS)

This project contains two major components:
1. **Core Backend & Flask Interactive Explorer** (PINN + DIP Reconstruction Pipeline, Sentinel-2 COG, RICE Benchmarking)
2. **React + Vite Modern Frontend Web Dashboard** (Located inside `new pr pinn dip/frontend`)

---

## 📋 System Prerequisites

* **Python:** `3.10` or higher (Recommended: Python 3.11 / 3.12)
* **Node.js:** `v18.0.0` or higher (with `npm`)
* **Git:** Installed and configured
* **GPU (Optional but Recommended):** NVIDIA GPU with CUDA support for accelerated PyTorch DIP inference.

---

## 🐍 1. Python Backend & Main Web Explorer Setup

### Step 1: Create and Activate Virtual Environment
```bash
# In the root directory (Pinn+dipp)
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Windows (CMD)
.\.venv\Scripts\activate.bat

# Activate on Linux / macOS
source .venv/bin/activate
```

### Step 2: Install Python Libraries & Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Key libraries: `torch`, `torchvision`, `numpy`, `scipy`, `rasterio`, `flask`, `opencv-python`, `matplotlib`, `scikit-image`, `scikit-learn`, `geopandas`, `rich`)*

---

## 🌐 2. Running the Main Applications

### Option A: Launch Interactive Full-Stack Flask Web Explorer
This starts the full multispectral Sentinel-2, RICE Explorer, and PINN+DIP comparison server:
```bash
python app.py
```
> 📍 Open your browser at: **`http://127.0.0.1:5000`**

### Option B: Run Core Sentinel-2 COG Cloud Removal via CLI
```bash
python reconstruct_s2.py --cog cogs/chandigarh_sample.tif --iterations 1500 --lr 0.001
```

### Option C: Run RICE-1 / RICE-2 Benchmark Evaluation
```bash
python rice1_thin_cloud_dip.py
python rice2dip_pinn_dip.py
```

### Option D: Calculate Metrics & Generate PDF Reports
```bash
python calculate_metrics_cli.py
python generate_rice2_pdf_report.py
```

---

## ⚛️ 3. React + Vite Frontend Setup (`new pr pinn dip/frontend`)

The repository includes a modern React 19 + Vite + TailwindCSS + Apache ECharts dashboard.

### Step 1: Navigate to Frontend Directory
```bash
cd "new pr pinn dip/frontend"
```

### Step 2: Install NPM Packages
```bash
npm install
```

*(Packages installed: `react`, `react-dom`, `lucide-react`, `echarts`, `echarts-for-react`, `tailwindcss`, `vite`, `typescript`)*

### Step 3: Run Frontend Development Server
```bash
npm run dev
```
> 📍 Open your browser at: **`http://localhost:5173`**

### Step 4: Build for Production
```bash
npm run build
```

---

## 📂 4. Pre-Trained Weights & Large Datasets (Google Drive)

Because GitHub has a strict 100 MB file limit, heavy pre-trained weights (`models/`), raw Sentinel-2 satellite tiles (`*.SAFE`), and COG files (`cogs/`) should be downloaded from Google Drive:

1. **`models/`** (~19.9 GB) ➔ Place inside root directory.
2. **`cogs/`** (~10.9 GB) ➔ Place inside root directory.
3. **`RICE_DATASET/`** (~516 MB) ➔ Place inside root directory.

---

## 🛠️ Quick Troubleshooting

* **CUDA Out of Memory (OOM):** Lower the patch size or tile dimensions in `configs/` or CLI flags (`--patch-size 256` or `--patch-size 128`).
* **PowerShell Execution Policy Error:** Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell terminal.
* **Port 5000 / 5173 already in use:** Specify a custom port via `python app.py --port 5001` or `npm run dev -- --port 3000`.
