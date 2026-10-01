import os
import glob
import asyncio
import threading
import subprocess
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.training import TrainingRunner

app = FastAPI(title="Physics Guided Image Reconstruction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections: Set[WebSocket] = set()
runner = TrainingRunner()
training_thread = None
main_loop = None

class TrainingConfig(BaseModel):
    phase: int
    image_name: str
    iterations: int
    lr: float
    optimizer: str
    tv_weight: float
    physics_weight: float
    mask_size: int
    pinn_losses: list[str]
    mask_type: str = "Rectangle"
    mask_center: list[int] = [128, 128]
    resume: bool = False

class DetectionConfig(BaseModel):
    phase: int
    image_name: str
    mask_size: int
    mask_type: str = "Rectangle"
    mask_center: list[int] = [128, 128]

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()

# Thread-safe WebSocket Broadcaster
def broadcast_training_state(payload: dict):
    if not active_connections or main_loop is None:
        return
    
    async def send_to_all():
        for connection in list(active_connections):
            try:
                await connection.send_json(payload)
            except Exception:
                if connection in active_connections:
                    active_connections.remove(connection)
                    
    # Schedule the coroutine to run on the main thread's event loop
    asyncio.run_coroutine_threadsafe(send_to_all(), main_loop)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        # Send initial status
        await websocket.send_json({
            "status": "Idle" if not runner.is_running else "Training",
            "log": "Connected to WebSocket Server."
        })
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/api/dataset")
def get_dataset(phase: int = 1):
    """
    Returns files list in the cloud Free images (Phase 1) or paired cloud/free images (Phase 2).
    """
    free_images = sorted(glob.glob("cloud Free images/*.png"))
    
    if phase == 1:
        images_list = []
        for f in free_images:
            base_name = os.path.basename(f).split(".")[0]
            images_list.append({
                "name": base_name,
                "filename": os.path.basename(f),
                "free_size": os.path.getsize(f),
                "cloud_size": 0
            })
        return {
            "images": images_list,
            "count": len(images_list)
        }
    else:
        cloud_images = sorted(glob.glob("cloud Images/*.png"))
        paired_names = []
        free_dict = {os.path.basename(f).split(".")[0]: f for f in free_images}
        cloud_dict = {os.path.basename(f).split(".")[0]: f for f in cloud_images}
        
        for base_name, free_path in sorted(free_dict.items()):
            cloudy_base = f"{base_name}c"
            if cloudy_base in cloud_dict:
                cloudy_path = cloud_dict[cloudy_base]
                paired_names.append({
                    "name": base_name,
                    "filename": os.path.basename(free_path),
                    "free_size": os.path.getsize(free_path),
                    "cloud_size": os.path.getsize(cloudy_path)
                })
            
        return {
            "images": paired_names,
            "count": len(paired_names)
        }

@app.get("/api/preview")
def get_preview(image_name: str, phase: int, mask_size: int, mask_type: str, mask_x: int, mask_y: int):
    """
    Generates preview base64 representations of the original reference, 
    binary mask, and masked inputs before training begins.
    """
    from PIL import Image
    import torchvision.transforms as T
    import numpy as np
    import torch
    from backend.training import tensor_to_base64
    from models.segmenter import detect_clouds_dl
    
    try:
        if phase == 1:
            clean_path = os.path.join("cloud Free images", f"{image_name}.png")
            if not os.path.exists(clean_path):
                return {"status": "error", "message": f"Image {image_name} not found."}
            orig_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            orig_tensor = T.ToTensor()(orig_img).unsqueeze(0)
            cloudy_tensor = orig_tensor
            
            # Generate custom synthetic mask
            h_img, w_img = orig_tensor.shape[2], orig_tensor.shape[3]
            mask_np = np.zeros((h_img, w_img), dtype=np.float32)
            
            if mask_type == "Circle":
                radius = mask_size // 2
                y_grid, x_grid = np.ogrid[:h_img, :w_img]
                mask_np[(x_grid - mask_x)**2 + (y_grid - mask_y)**2 <= radius**2] = 1.0
            else:  # Rectangle
                mh, mw = mask_size, mask_size
                y1, y2 = max(0, mask_y - mh // 2), min(h_img, mask_y + mh // 2)
                x1, x2 = max(0, mask_x - mw // 2), min(w_img, mask_x + mw // 2)
                mask_np[y1:y2, x1:x2] = 1.0
                
            mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
            masked_tensor = cloudy_tensor * (1.0 - mask_tensor)
            transmission_b64 = tensor_to_base64(torch.ones_like(mask_tensor))
        else:  # Phase 2
            from models.segmenter import compute_physical_attenuation
            cloudy_path = os.path.join("cloud Images", f"{image_name}c.png")
            clean_path = os.path.join("cloud Free images", f"{image_name}.png")
            if not os.path.exists(cloudy_path) or not os.path.exists(clean_path):
                return {"status": "error", "message": f"Paired images for {image_name} not found."}
            cloudy_img = Image.open(cloudy_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            clean_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            orig_tensor = T.ToTensor()(clean_img).unsqueeze(0)
            cloudy_tensor = T.ToTensor()(cloudy_img).unsqueeze(0)
            
            # Mask is empty/black until explicit detection button is clicked
            mask_tensor = torch.zeros((1, 1, 256, 256))
            transmission_b64 = tensor_to_base64(torch.ones_like(mask_tensor))
            masked_tensor = cloudy_tensor  # Cloudy image itself is the input
            
        return {
            "status": "success",
            "original": tensor_to_base64(orig_tensor),
            "masked": tensor_to_base64(masked_tensor),
            "mask": tensor_to_base64(mask_tensor),
            "transmission": transmission_b64
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/detect")
def detect_clouds(config: DetectionConfig):
    try:
        res = runner.detect_clouds_step(config.dict())
        return res
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/start")
def start_training(config: TrainingConfig):
    global training_thread
    if runner.is_running:
        return {"status": "error", "message": "Training is already in progress."}
        
    if runner.cached_mask is None or runner.current_image != config.image_name or runner.current_phase != config.phase:
        return {"status": "error", "message": "Please perform cloud detection / masking step first!"}
        
    runner_config = config.dict()
    
    # Run PyTorch loop in a separate thread so it does not block the FastAPI server
    def thread_target():
        runner.run_training(runner_config, broadcast_training_state)
        
    training_thread = threading.Thread(target=thread_target, daemon=True)
    training_thread.start()
    
    return {"status": "success", "message": "Training started."}

@app.post("/api/stop")
def stop_training():
    if not runner.is_running:
        return {"status": "error", "message": "No training in progress."}
    runner.should_stop = True
    return {"status": "success", "message": "Stop signal sent."}

@app.get("/api/gpu")
def get_gpu_status():
    """
    Retrieves live VRAM and temperature metrics directly from nvidia-smi.
    """
    try:
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        parts = [p.strip() for p in result.split(",")]
        return {
            "gpu_util": int(parts[0]),
            "vram_used": int(parts[1]),
            "vram_total": int(parts[2]),
            "temp": int(parts[3]),
            "status": "success"
        }
    except Exception as e:
        return {
            "gpu_util": 0,
            "vram_used": 0,
            "vram_total": 4096,
            "temp": 0,
            "status": "error",
            "message": str(e)
        }
