import os
import io
import base64
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PR_DIR = os.path.dirname(BACKEND_DIR)

import sys
sys.path.append(PR_DIR)

from models.dip import HighCapacityDIPUNet  # type: ignore
from models.segmenter import compute_physical_attenuation  # type: ignore
from models.losses import total_variation_loss, physical_range_penalty, dark_channel_prior_loss, spectral_correlation_metric  # type: ignore
from models.pinn import atmospheric_scattering_loss  # type: ignore

def tensor_to_base64(tensor):
    """
    Converts a PyTorch tensor (1, 3, H, W) or (1, 1, H, W) to a base64 PNG string.
    """
    if tensor.shape[1] == 1:
        # Grayscale / Mask: expand to 3 channels for ToPILImage
        tensor = tensor.repeat(1, 3, 1, 1)
    pil_img = T.ToPILImage()(tensor.squeeze(0).clamp(0.0, 1.0).cpu())
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def difference_heatmap_base64(img_tensor, ref_tensor):
    """
    Calculates absolute difference and maps to a jet colormap, returning base64.
    """
    diff = torch.abs(img_tensor - ref_tensor)
    diff_gray = torch.mean(diff, dim=1).squeeze(0).cpu().numpy() # (H, W)
    
    # Normalize to [0, 1]
    d_min, d_max = diff_gray.min(), diff_gray.max()
    if d_max > d_min:
        diff_gray = (diff_gray - d_min) / (d_max - d_min)
        
    # Map using matplotlib colormap
    cmap = plt.get_cmap("jet")
    rgba_img = (cmap(diff_gray) * 255)  # type: ignore
    if not isinstance(rgba_img, np.ndarray):
        rgba_img = np.array(rgba_img)
    rgba_img = rgba_img.astype(np.uint8) # (H, W, 4)
    pil_img = Image.fromarray(rgba_img).convert("RGB")
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class TrainingRunner:
    def __init__(self):
        self.is_running = False
        self.should_stop = False
        self.net = None
        self.net_input = None
        self.optimizer = None
        self.current_step = 0
        self.current_image = None
        self.current_phase = None
        self.cached_mask = None
        self.cached_transmission = None
        self.cloud_cover_percent = None
        
    def run_training(self, config, ws_broadcast_fn):
        """
        Runs the DIP + PINN optimization.
        """
        self.is_running = True
        self.should_stop = False
        
        phase = config.get("phase", 1)
        image_name = config.get("image_name", "123")
        iterations = config.get("iterations", 5000)
        lr = config.get("lr", 0.005)
        optimizer_name = config.get("optimizer", "Adam")
        tv_weight = config.get("tv_weight", 1e-4)
        physics_weight = config.get("physics_weight", 1e-3)
        pinn_losses = config.get("pinn_losses", ["ASM", "TV", "Physical Range"])
        resume = config.get("resume", False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 1. Load images based on phase
            if (self.cached_mask is not None and 
                self.current_image == image_name and 
                self.current_phase == phase):
                mask_tensor = self.cached_mask.to(device)
                if phase == 1:
                    clean_path = os.path.join(PR_DIR, "cloud Free images", f"{image_name}.png")
                    orig_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                    orig_tensor = T.ToTensor()(orig_img).unsqueeze(0).to(device)
                    cloudy_tensor = orig_tensor
                    transmission = torch.ones_like(mask_tensor)
                else:
                    cloudy_path = os.path.join(PR_DIR, "cloud Images", f"{image_name}c.png")
                    clean_path = os.path.join(PR_DIR, "cloud Free images", f"{image_name}.png")
                    cloudy_img = Image.open(cloudy_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                    clean_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                    orig_tensor = T.ToTensor()(clean_img).unsqueeze(0).to(device)
                    cloudy_tensor = T.ToTensor()(cloudy_img).unsqueeze(0).to(device)
                    if hasattr(self, 'cached_transmission') and self.cached_transmission is not None:
                        transmission = self.cached_transmission.to(device)
                    else:
                        attenuation = compute_physical_attenuation(cloudy_tensor, orig_tensor)
                        transmission = torch.clamp(1.0 - attenuation, min=0.1, max=1.0)
            elif phase == 1:
                clean_path = os.path.join(PR_DIR, "cloud Free images", f"{image_name}.png")
                if not os.path.exists(clean_path):
                    raise FileNotFoundError(f"Reference image {clean_path} not found.")
                orig_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                orig_tensor = T.ToTensor()(orig_img).unsqueeze(0).to(device)
                cloudy_tensor = orig_tensor
                
                # Generate synthetic custom mask based on position and shape
                h_img, w_img = orig_tensor.shape[2], orig_tensor.shape[3]
                mask_np = np.zeros((h_img, w_img), dtype=np.float32)
                
                mask_type = config.get("mask_type", "Rectangle")
                mask_size = config.get("mask_size", 64)
                mask_center = config.get("mask_center", [h_img // 2, w_img // 2])
                cy, cx = mask_center[0], mask_center[1]
                
                if mask_type == "Circle":
                    radius = mask_size // 2
                    y_grid, x_grid = np.ogrid[:h_img, :w_img]
                    mask_np[(x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2] = 1.0
                else:  # Rectangle
                    mh, mw = mask_size, mask_size
                    y1, y2 = max(0, cy - mh // 2), min(h_img, cy + mh // 2)
                    x1, x2 = max(0, cx - mw // 2), min(w_img, cx + mw // 2)
                    mask_np[y1:y2, x1:x2] = 1.0
                    
                mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).to(device)
                transmission = torch.ones_like(mask_tensor)
            else:  # Phase 2
                cloudy_path = os.path.join(PR_DIR, "cloud Images", f"{image_name}c.png")
                clean_path = os.path.join(PR_DIR, "cloud Free images", f"{image_name}.png")
                if not os.path.exists(cloudy_path) or not os.path.exists(clean_path):
                    raise FileNotFoundError(f"Paired images for {image_name} not found.")
                
                cloudy_img = Image.open(cloudy_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                clean_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                
                orig_tensor = T.ToTensor()(clean_img).unsqueeze(0).to(device)
                cloudy_tensor = T.ToTensor()(cloudy_img).unsqueeze(0).to(device)
                
                # Compute physical attenuation
                attenuation = compute_physical_attenuation(cloudy_tensor, orig_tensor)
                transmission = torch.clamp(1.0 - attenuation, min=0.1, max=1.0)
                
                # Use SegmentationUNet for DL cloud mask prediction!
                from models.segmenter import detect_clouds_dl  # type: ignore
                mask_tensor, _ = detect_clouds_dl(cloudy_tensor, orig_tensor, device)
                
            masked_tensor = cloudy_tensor * (1.0 - mask_tensor)
            
            # Prepare base64 references for the UI
            orig_b64 = tensor_to_base64(orig_tensor)
            masked_b64 = tensor_to_base64(cloudy_tensor if phase == 2 else masked_tensor)
            mask_b64 = tensor_to_base64(mask_tensor)
            transmission_b64 = tensor_to_base64(transmission if phase == 2 else torch.ones_like(mask_tensor))
            
            # 2. Setup or Resume DIP Network
            is_resumed = (
                resume and 
                self.net is not None and 
                self.current_image == image_name and 
                self.current_phase == phase
            )
            
            if is_resumed:
                net = self.net
                net_input = self.net_input
                optimizer = self.optimizer
                start_step = self.current_step
                print(f"Resuming training for {image_name} from step {start_step}...")
            else:
                net = HighCapacityDIPUNet(in_channels=32, out_channels=3, use_skip=True).to(device)
                
                # DIP Inpainting Conditioning:
                # We feed masked image as guidance + normalized coordinate meshgrid + high-entropy uniform spatial noise
                torch.manual_seed(42)
                if phase == 1:
                    base_noise = torch.randn(1, 27, 256, 256, device=device) * 0.5 + 0.5
                    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 256, device=device), torch.linspace(-1, 1, 256, device=device), indexing='ij')
                    grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
                    net_input = torch.cat([masked_tensor, grid, base_noise], dim=1) # [1, 32, 256, 256]
                else:
                    net_input = masked_tensor.repeat(1, 11, 1, 1)[:, :32, :, :]
                
                # Setup optimizer
                if optimizer_name == "AdamW":
                    optimizer = optim.AdamW(net.parameters(), lr=lr)
                elif optimizer_name == "SGD":
                    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
                elif optimizer_name == "RMSProp":
                    optimizer = optim.RMSprop(net.parameters(), lr=lr)
                else:
                    optimizer = optim.Adam(net.parameters(), lr=lr)
                
                start_step = 0
                self.net = net
                self.net_input = net_input
                self.optimizer = optimizer
                self.current_image = image_name
                self.current_phase = phase
                print(f"Starting fresh training for {image_name}...")
                
            start_time = time.time()
            total_iters = start_step + iterations
            
            # Extract customized loss weights
            loss_weights_dict = config.get("loss_weights", {})
            w_asm = float(loss_weights_dict.get("asm", 0.0)) * float(physics_weight)
            w_tv = float(loss_weights_dict.get("tv", 0.05)) if "tv" in loss_weights_dict else float(tv_weight)
            w_range = float(loss_weights_dict.get("range", 10.0)) * 1e-3
            w_dcp = float(loss_weights_dict.get("dcp", 0.0)) * 1e-2
            w_scc = float(loss_weights_dict.get("scc", 0.0)) * 1e-2

            # 3. Optimization Loop
            assert net is not None and optimizer is not None and net_input is not None
            for step in range(start_step + 1, total_iters + 1):
                if self.should_stop:
                    ws_broadcast_fn({"status": "Stopped", "log": "Optimization cancelled by user."})
                    break
                    
                optimizer.zero_grad()
                predicted_clear = net(net_input)
                
                # 1. Unmasked Context Reconstruction Loss (L1 + MSE hybrid for high-frequency sharp details)
                clear_mask = 1.0 - mask_tensor
                l1_context = torch.sum(torch.abs(predicted_clear - orig_tensor) * clear_mask) / (torch.sum(clear_mask) * 3 + 1e-8)
                mse_context = torch.sum((predicted_clear - orig_tensor) ** 2 * clear_mask) / (torch.sum(clear_mask) * 3 + 1e-8)
                data_loss = 0.5 * l1_context + 0.5 * mse_context
                
                # 2. Cross-Mask Boundary Gradient Loss:
                # Enforces edges and features crossing from known clear pixels INTO the masked hole to continue seamlessly
                mask_left = mask_tensor[:, :, :, :-1]
                mask_right = mask_tensor[:, :, :, 1:]
                cross_x = ((mask_left + mask_right) == 1).float()

                mask_top = mask_tensor[:, :, :-1, :]
                mask_bottom = mask_tensor[:, :, 1:, :]
                cross_y = ((mask_top + mask_bottom) == 1).float()

                dx_pred = predicted_clear[:, :, :, 1:] - predicted_clear[:, :, :, :-1]
                dx_gt = orig_tensor[:, :, :, 1:] - orig_tensor[:, :, :, :-1]
                dy_pred = predicted_clear[:, :, 1:, :] - predicted_clear[:, :, :-1, :]
                dy_gt = orig_tensor[:, :, 1:, :] - orig_tensor[:, :, :-1, :]

                boundary_x = (torch.abs(dx_pred - dx_gt) * cross_x).sum() / (cross_x.sum() * 3 + 1e-8)
                boundary_y = (torch.abs(dy_pred - dy_gt) * cross_y).sum() / (cross_y.sum() * 3 + 1e-8)
                boundary_loss = boundary_x + boundary_y

                # 3. Mask-Aware Total Variation Loss (Smooths the generated content inside the hole)
                tv_loss = torch.tensor(0.0, device=device)
                if "TV" in pinn_losses:
                    tv_dx = torch.abs(predicted_clear[:, :, :, 1:] - predicted_clear[:, :, :, :-1])
                    tv_dy = torch.abs(predicted_clear[:, :, 1:, :] - predicted_clear[:, :, :-1, :])
                    m_x = mask_tensor[:, :, :, 1:] * mask_tensor[:, :, :, :-1]
                    m_y = mask_tensor[:, :, 1:, :] * mask_tensor[:, :, :-1, :]
                    tv_loss = ((tv_dx * m_x).sum() + (tv_dy * m_y).sum()) / ((m_x.sum() + m_y.sum()) * 3 + 1e-8)
                    
                range_loss = torch.tensor(0.0, device=device)
                if "Physical Range" in pinn_losses:
                    range_loss = physical_range_penalty(predicted_clear)
                    
                pinn_loss = torch.tensor(0.0, device=device)
                if phase == 2 and "ASM" in pinn_losses:
                    pinn_loss = atmospheric_scattering_loss(cloudy_tensor, predicted_clear, transmission, atmospheric_light=1.0)
                    
                dcp_loss = torch.tensor(0.0, device=device)
                if "DCP" in pinn_losses:
                    dcp_loss = dark_channel_prior_loss(predicted_clear, mask_tensor)
                    
                scc_loss = torch.tensor(0.0, device=device)
                if "SCC" in pinn_losses:
                    scc_val = spectral_correlation_metric(predicted_clear, mask_tensor)
                    scc_loss = 1.0 - scc_val
                    
                # Combined Loss with boundary continuity and user customized weights
                loss = data_loss + (1.0 * boundary_loss if phase == 1 else 0.0)
                if "TV" in pinn_losses:
                    loss += w_tv * tv_loss
                if "Physical Range" in pinn_losses:
                    loss += w_range * range_loss
                if phase == 2 and "ASM" in pinn_losses:
                    loss += w_asm * pinn_loss
                if "DCP" in pinn_losses:
                    loss += w_dcp * dcp_loss
                if "SCC" in pinn_losses:
                    loss += w_scc * scc_loss
                    
                loss.backward()
                optimizer.step()
                self.current_step = step
                
                # Stream logs and images every 100 steps
                if step % 100 == 0 or step == start_step + 1 or step == total_iters:
                    elapsed = time.time() - start_time
                    active_steps = step - start_step
                    eta = (elapsed / active_steps) * (total_iters - step) if active_steps > 1 else 0
                    with torch.no_grad():
                        # In Phase 1:
                        # - If mode is 'learn_context' (Button 2): network progressively learns surrounding clear pixels with live convergence animation while strictly preserving the black masked hole!
                        # - If mode is 'reconstruct' (Button 3 / Run Engine): unmasked context stays 100% original crisp while the masked black spot is seamlessly inpainted with the synthesized texture!
                        mode = config.get("mode", "reconstruct")
                        if phase == 1:
                            if mode == "learn_context":
                                # Live progressive DIP convergence on surrounding clear pixels + black masked spot
                                final_output = predicted_clear * (1.0 - mask_tensor)
                            else:
                                final_output = orig_tensor * (1.0 - mask_tensor) + predicted_clear * mask_tensor
                        else:
                            final_output = predicted_clear

                        # Calculate evaluation metrics
                        mse_val = nn.functional.mse_loss(final_output, orig_tensor).item()
                        mae_val = torch.mean(torch.abs(final_output - orig_tensor)).item()
                        
                        # Simplified PSNR
                        psnr_val = 20 * np.log10(1.0 / np.sqrt(mse_val)) if mse_val > 0 else 99.0
                        
                        # Simplified SSIM
                        mu1 = final_output.mean()
                        mu2 = orig_tensor.mean()
                        sigma1_sq = ((final_output - mu1) ** 2).mean()
                        sigma2_sq = ((orig_tensor - mu2) ** 2).mean()
                        sigma12 = ((final_output - mu1) * (orig_tensor - mu2)).mean()
                        ssim_val = (((2 * mu1 * mu2 + 1e-4) * (2 * sigma12 + 9e-4)) / ((mu1**2 + mu2**2 + 1e-4) * (sigma1_sq + sigma2_sq + 9e-4))).item()
                        
                        # Compute detailed physical telemetry
                        val_transmission = transmission if phase == 2 else torch.ones_like(mask_tensor)
                        val_cloudy = cloudy_tensor
                        asm_residual_val = atmospheric_scattering_loss(val_cloudy, final_output, val_transmission, atmospheric_light=1.0).item()
                        
                        # DCP Mean inside the mask
                        min_c, _ = torch.min(final_output, dim=1, keepdim=True)
                        dark = -torch.nn.functional.max_pool2d(-min_c, kernel_size=15, stride=1, padding=7)
                        if mask_tensor is not None:
                            dcp_mean_val = (torch.sum(dark * mask_tensor) / (torch.sum(mask_tensor) + 1e-8)).item()
                        else:
                            dcp_mean_val = torch.mean(dark).item()
                            
                        # SCC Score inside the mask
                        scc_score_val = spectral_correlation_metric(final_output, mask_tensor).item()
                        
                        # Reflectance violations rate
                        total_pixels = final_output.numel()
                        out_of_bounds = torch.sum((final_output < 0.0) | (final_output > 1.0)).item()
                        violation_rate_val = (out_of_bounds / total_pixels) * 100.0
                        
                        # Generate base64 images
                        recon_b64 = tensor_to_base64(final_output)
                        diff_b64 = difference_heatmap_base64(final_output, orig_tensor)
                        
                        # Prepare payload
                        elapsed_sec = int(elapsed)
                        eta_sec = int(eta)
                        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_sec))
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

                        payload = {
                            "status": "Training" if step < total_iters else "Completed",
                            "iteration": step,
                            "total_iterations": total_iters,
                            "elapsed": round(elapsed),
                            "eta": round(eta),
                            "elapsed_hhmmss": elapsed_str,
                            "eta_hhmmss": eta_str,
                            "metrics": {
                                "loss": loss.item(),
                                "data_loss": data_loss.item(),
                                "tv_loss": tv_loss.item() if "TV" in pinn_losses else 0.0,
                                "pinn_loss": pinn_loss.item() if (phase == 2 and "ASM" in pinn_losses) else 0.0,
                                "psnr": psnr_val,
                                "ssim": ssim_val,
                                "mse": mse_val,
                                "mae": mae_val,
                                "asm_residual": asm_residual_val,
                                "dcp_mean": dcp_mean_val,
                                "scc_score": scc_score_val,
                                "violation_rate": violation_rate_val
                            },
                            "images": {
                                "original": orig_b64,
                                "masked": masked_b64,
                                "mask": mask_b64,
                                "reconstructed": recon_b64,
                                "difference": diff_b64,
                                "transmission": transmission_b64
                            },
                            "log": f"[{time.strftime('%H:%M:%S')}] [DIP+PINN] Iteration {step}/{iterations} | Loss: {loss.item():.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | Time: {elapsed_str} (ETA: {eta_str})"
                        }
                        print(payload["log"])
                        ws_broadcast_fn(payload)
                        
            if not self.should_stop:
                print("Optimization completed successfully.")
                ws_broadcast_fn({"status": "Completed", "log": "Optimization completed successfully."})
                
        except Exception as e:
            ws_broadcast_fn({"status": "Error", "log": f"Error during optimization: {str(e)}"})
            
        finally:
            self.is_running = False
            self.should_stop = False

    def detect_clouds_step(self, config):
        """
        Runs the cloud detection step (U-Net or Synthetic) and returns previews and cloud cover percent.
        Caches the generated mask inside self.cached_mask, self.cached_transmission, and self.cloud_cover_percent.
        """
        phase = config.get("phase", 1)
        image_name = config.get("image_name", "123")
        mask_size = config.get("mask_size", 64)
        mask_type = config.get("mask_type", "Rectangle")
        mask_center = config.get("mask_center", [128, 128])
        cy, cx = mask_center[0], mask_center[1]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load original images
        if phase == 1:
            clean_path = os.path.join(PR_DIR, "cloud Free images", f"{image_name}.png")
            if not os.path.exists(clean_path):
                raise FileNotFoundError(f"Reference image {clean_path} not found.")
            orig_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            orig_tensor = T.ToTensor()(orig_img).unsqueeze(0).to(device)
            cloudy_tensor = orig_tensor
            
            # Generate synthetic mask
            h_img, w_img = orig_tensor.shape[2], orig_tensor.shape[3]
            mask_np = np.zeros((h_img, w_img), dtype=np.float32)
            if mask_type == "Circle":
                radius = mask_size // 2
                y_grid, x_grid = np.ogrid[:h_img, :w_img]
                mask_np[(x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2] = 1.0
            else: # Rectangle
                mh, mw = mask_size, mask_size
                y1, y2 = max(0, cy - mh // 2), min(h_img, cy + mh // 2)
                x1, x2 = max(0, cx - mw // 2), min(w_img, cx + mw // 2)
                mask_np[y1:y2, x1:x2] = 1.0
                
            mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            transmission = torch.ones_like(mask_tensor)
        else: # Phase 2
            cloudy_path = os.path.join(PR_DIR, "cloud Images", f"{image_name}c.png")
            clean_path = os.path.join(PR_DIR, "cloud Free images", f"{image_name}.png")
            if not os.path.exists(cloudy_path) or not os.path.exists(clean_path):
                raise FileNotFoundError(f"Paired images for {image_name} not found.")
            
            cloudy_img = Image.open(cloudy_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            clean_img = Image.open(clean_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            
            orig_tensor = T.ToTensor()(clean_img).unsqueeze(0).to(device)
            cloudy_tensor = T.ToTensor()(cloudy_img).unsqueeze(0).to(device)
            
            # Compute physical attenuation
            attenuation = compute_physical_attenuation(cloudy_tensor, orig_tensor)
            transmission = torch.clamp(1.0 - attenuation, min=0.1, max=1.0)
            
            # Run cloud segmenter U-Net
            from models.segmenter import detect_clouds_dl  # type: ignore
            mask_tensor, _ = detect_clouds_dl(cloudy_tensor, orig_tensor, device)
            
        masked_tensor = cloudy_tensor * (1.0 - mask_tensor)
        
        # Calculate cloud cover percentage
        cloud_pixels = torch.sum(mask_tensor > 0.5).item()
        total_pixels = mask_tensor.numel()
        cloud_cover_percent = (cloud_pixels / total_pixels) * 100.0
        
        # Cache results in runner
        self.cached_mask = mask_tensor.cpu()
        self.cached_transmission = transmission.cpu()
        self.cloud_cover_percent = cloud_cover_percent
        self.current_image = image_name
        self.current_phase = phase
        
        # Return previews
        return {
            "status": "success",
            "cloud_cover": cloud_cover_percent,
            "original": tensor_to_base64(orig_tensor),
            "masked": tensor_to_base64(cloudy_tensor if phase == 2 else masked_tensor),
            "mask": tensor_to_base64(mask_tensor),
            "transmission": tensor_to_base64(transmission)
        }
