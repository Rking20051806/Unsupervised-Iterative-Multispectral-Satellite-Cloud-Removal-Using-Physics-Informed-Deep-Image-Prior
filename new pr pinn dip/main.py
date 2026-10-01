import os
import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# Import models & custom utilities
from models.dip import HighCapacityDIPUNet, get_parameter_count
from models.segmenter import compute_physical_attenuation
from models.losses import total_variation_loss, physical_range_penalty
from models.pinn import atmospheric_scattering_loss

def parse_args():
    parser = argparse.ArgumentParser(description="DIP & PINN Satellite Image Reconstruction & Cloud Removal")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Phase 1 (Synthetic) or Phase 2 (Real Cloud Removal)")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to process")
    parser.add_argument("--iterations", type=int, default=5000, help="Maximum number of optimization iterations")
    parser.add_argument("--lr", type=type(0.1), default=0.005, help="Learning rate")
    parser.add_argument("--tv_weight", type=type(0.1), default=1e-4, help="Total Variation loss weight")
    parser.add_argument("--physics_weight", type=type(0.1), default=1e-3, help="Physics consistency weight")
    return parser.parse_args()

def setup_directories():
    dirs = [
        "phase1/cloudfree", "phase1/masks", "phase1/masked", "phase1/reconstructed", "phase1/results",
        "phase2/cloud", "phase2/cloudfree", "phase2/output", "phase2/metrics"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

def calculate_ssim(img1, img2):
    """
    Simplified SSIM computation for structural similarity assessment.
    """
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.item()

def run_dip_pinn(cloudy_tensor, orig_tensor, mask_tensor, phase_name, img_name, 
                 max_iters, lr, tv_weight, physics_weight, device, output_dir, stats_dir):
    """
    Optimizes a DIP network for a single image, regularized by PINN TV and Atmospheric scattering model loss.
    Logs metrics at benchmarks: 500, 1000, 1500, 2000, 3000, 4000, 5000.
    """
    _, _, h, w = orig_tensor.shape
    
    # Prepare input masked image
    input_masked = cloudy_tensor * (1.0 - mask_tensor)
    
    # 1. Initialize High Capacity U-Net
    net = HighCapacityDIPUNet(in_channels=32, out_channels=3, use_skip=False).to(device)
    print(f"[{phase_name}] Model Initialized. Trainable Parameters: {get_parameter_count(net):,}")
    
    # 2. Input fixed guidance image
    net_input = input_masked.repeat(1, 11, 1, 1)[:, :32, :, :]
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Checkpoints to log
    benchmarks = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    # For Phase 2 Physics Model: transmission (t) = 1.0 - attenuation
    if phase_name == "Phase 2":
        attenuation = compute_physical_attenuation(cloudy_tensor, orig_tensor)
        transmission = torch.clamp(1.0 - attenuation, min=0.1, max=1.0)
    else:
        transmission = torch.ones_like(mask_tensor) # Dummy for Phase 1
        
    history = {b: {} for b in benchmarks}
    
    print(f"[{phase_name}] Optimizing reconstruction for {img_name}...")
    for step in range(1, max_iters + 1):
        optimizer.zero_grad()
        
        predicted_clear = net(net_input)
        
        # 1. Data Loss (Only on unmasked/known region)
        data_loss = nn.functional.mse_loss(predicted_clear * (1.0 - mask_tensor), input_masked)
        
        # 2. TV Loss inside the mask
        tv_loss = total_variation_loss(predicted_clear, mask_tensor)
        
        # 3. Physics / Range Constraint
        range_loss = physical_range_penalty(predicted_clear)
        
        # 4. Phase 2 PINN Loss: Atmospheric Scattering Model
        pinn_loss = torch.tensor(0.0, device=device)
        if phase_name == "Phase 2":
            pinn_loss = atmospheric_scattering_loss(cloudy_tensor, predicted_clear, transmission, atmospheric_light=1.0)
            
        # Total loss
        loss = data_loss + tv_weight * tv_loss + 1e-3 * range_loss
        if phase_name == "Phase 2":
            loss += physics_weight * pinn_loss
            
        loss.backward()
        optimizer.step()
        
        # Evaluate and save at benchmarks
        if step in benchmarks:
            with torch.no_grad():
                psnr_val = calculate_psnr(predicted_clear, orig_tensor)
                ssim_val = calculate_ssim(predicted_clear, orig_tensor)
                
                print(f"  Step {step:04d} | Total Loss: {loss.item():.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
                
                # Save reconstructed image
                out_path = os.path.join(output_dir, f"{img_name}_iter_{step:04d}.png")
                T.ToPILImage()(predicted_clear.squeeze(0).cpu()).save(out_path)
                
                # Save stats
                history[step] = {
                    "loss": loss.item(),
                    "psnr": psnr_val,
                    "ssim": ssim_val
                }
                
    # Save stats log
    stats_file = os.path.join(stats_dir, f"{img_name}_metrics.txt")
    with open(stats_file, "w") as f:
        f.write(f"Benchmark Metrics for {img_name} ({phase_name})\n")
        f.write("="*40 + "\n")
        for b in benchmarks:
            if b in history and "psnr" in history[b]:
                h_b = history[b]
                f.write(f"Iteration {b:04d} -> Loss: {h_b['loss']:.6f} | PSNR: {h_b['psnr']:.2f} dB | SSIM: {h_b['ssim']:.4f}\n")
                
    # Plot final curves
    iters = [b for b in benchmarks if b in history and "psnr" in history[b]]
    psnrs = [history[b]["psnr"] for b in iters]
    ssims = [history[b]["ssim"] for b in iters]
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(iters, psnrs, 'o-', color="tab:blue")
    plt.title("PSNR over Benchmarks")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(iters, ssims, 'o-', color="tab:orange")
    plt.title("SSIM over Benchmarks")
    plt.xlabel("Iterations")
    plt.ylabel("SSIM")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, f"{img_name}_metrics_curves.png"))
    plt.close()

def main():
    args = parse_args()
    setup_directories()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    if args.phase == 1:
        print("\n--- Starting Phase 1: Synthetic Mask Reconstruction ---")
        # Read from 'cloud Free images'
        clean_images = sorted(glob.glob("cloud Free images/*.png"))
        if not clean_images:
            print("Error: No images found in 'cloud Free images/' directory.")
            return
            
        num_to_process = min(args.num_images, len(clean_images))
        print(f"Found {len(clean_images)} images. Processing {num_to_process} images.")
        
        for i in range(num_to_process):
            img_path = clean_images[i]
            img_name = os.path.basename(img_path).split(".")[0]
            print(f"\nProcessing {img_path}...")
            
            # Load and save to phase1/cloudfree/
            orig_img = Image.open(img_path).convert("RGB")
            # Resize to 256x256 for fast execution on GTX 1650
            w_new, h_new = 256, 256
            orig_img = orig_img.resize((w_new, h_new), Image.Resampling.BILINEAR)
                
            orig_img.save(f"phase1/cloudfree/{img_name}.png")
            orig_tensor = T.ToTensor()(orig_img).unsqueeze(0).to(device)
            
            # Generate synthetic mask: 64x64 square in center
            h_img, w_img = orig_tensor.shape[2], orig_tensor.shape[3]
            mask_np = np.zeros((h_img, w_img), dtype=np.float32)
            mh, mw = 64, 64
            cy, cx = h_img // 2, w_img // 2
            mask_np[cy - mh//2 : cy + mh//2, cx - mw//2 : cx + mw//2] = 1.0
            
            # Save mask and masked images
            mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            T.ToPILImage()(mask_tensor.squeeze(0).cpu()).save(f"phase1/masks/{img_name}_mask.png")
            
            masked_tensor = orig_tensor * (1.0 - mask_tensor)
            T.ToPILImage()(masked_tensor.squeeze(0).cpu()).save(f"phase1/masked/{img_name}_masked.png")
            
            # Run DIP + PINN TV reconstruction
            run_dip_pinn(
                cloudy_tensor=orig_tensor, # Phase 1 input is clean image with mask applied
                orig_tensor=orig_tensor,
                mask_tensor=mask_tensor,
                phase_name="Phase 1",
                img_name=img_name,
                max_iters=args.iterations,
                lr=args.lr,
                tv_weight=args.tv_weight,
                physics_weight=args.physics_weight,
                device=device,
                output_dir="phase1/reconstructed",
                stats_dir="phase1/results"
            )
            
    elif args.phase == 2:
        print("\n--- Starting Phase 2: Real Cloud Removal ---")
        # Load from 'cloud Images' and pair with 'cloud Free images'
        cloudy_images = sorted(glob.glob("cloud Images/*.png"))
        if not cloudy_images:
            print("Error: No images found in 'cloud Images/' directory.")
            return
            
        num_to_process = min(args.num_images, len(cloudy_images))
        print(f"Found {len(cloudy_images)} paired cloud images. Processing {num_to_process} images.")
        
        for i in range(num_to_process):
            cloudy_path = cloudy_images[i]
            cloudy_basename = os.path.basename(cloudy_path).split(".")[0]
            if cloudy_basename.endswith('c'):
                img_name = cloudy_basename[:-1]
            else:
                img_name = cloudy_basename
            clean_path = os.path.join("cloud Free images", f"{img_name}.png")
            
            if not os.path.exists(clean_path):
                print(f"Warning: Corresponding reference image for {cloudy_path} not found. Skipping.")
                continue
                
            print(f"\nProcessing cloudy image {cloudy_path} with reference {clean_path}...")
            
            # Load images
            cloudy_img = Image.open(cloudy_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")
            
            # Resize to 256x256 for fast execution on GTX 1650
            w_new, h_new = 256, 256
            cloudy_img = cloudy_img.resize((w_new, h_new), Image.Resampling.BILINEAR)
            clean_img = clean_img.resize((w_new, h_new), Image.Resampling.BILINEAR)
                
            # Copy to phase2/ folders
            cloudy_img.save(f"phase2/cloud/{img_name}.png")
            clean_img.save(f"phase2/cloudfree/{img_name}.png")
            
            cloudy_tensor = T.ToTensor()(cloudy_img).unsqueeze(0).to(device)
            clean_tensor = T.ToTensor()(clean_img).unsqueeze(0).to(device)
            
            # Run DL cloud detection
            from models.segmenter import detect_clouds_dl
            mask_tensor, _ = detect_clouds_dl(cloudy_tensor, clean_tensor, device)
            
            # Run DIP + PINN (ASM constraint) reconstruction
            run_dip_pinn(
                cloudy_tensor=cloudy_tensor,
                orig_tensor=clean_tensor,
                mask_tensor=mask_tensor,
                phase_name="Phase 2",
                img_name=img_name,
                max_iters=args.iterations,
                lr=args.lr,
                tv_weight=args.tv_weight,
                physics_weight=args.physics_weight,
                device=device,
                output_dir="phase2/output",
                stats_dir="phase2/metrics"
            )

if __name__ == "__main__":
    main()
