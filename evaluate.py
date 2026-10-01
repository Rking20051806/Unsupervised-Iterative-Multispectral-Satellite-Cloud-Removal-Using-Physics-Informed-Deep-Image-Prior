import os
import sys
import time
import json
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from datasets import RICEDataset
from dip_loop import optimize_dip, estimate_atmospheric_light_dcp
from prs_metric import calculate_prs
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

def calculate_psnr(ref_arr, pred_arr):
    return float(psnr_metric(ref_arr, pred_arr, data_range=1.0))

def calculate_ssim(ref_arr, pred_arr):
    if len(ref_arr.shape) == 3:
        return float(ssim_metric(ref_arr, pred_arr, channel_axis=2, data_range=1.0))
    return float(ssim_metric(ref_arr, pred_arr, data_range=1.0))

def calculate_sam(ref_arr, pred_arr):
    """
    Calculates Spectral Angle Mapper (SAM) between predicted and reference array.
    """
    ref_flat = ref_arr.reshape(-1, ref_arr.shape[-1]).astype(np.float32)
    pred_flat = pred_arr.reshape(-1, pred_arr.shape[-1]).astype(np.float32)
    dot = np.sum(ref_flat * pred_flat, axis=1)
    norms = np.linalg.norm(ref_flat, axis=1) * np.linalg.norm(pred_flat, axis=1) + 1e-8
    return float(np.mean(np.arccos(np.clip(dot / norms, -1, 1))))

def run_dcp_baseline(I: torch.Tensor, patch_size: int = 15, omega: float = 0.95, t0: float = 0.1) -> torch.Tensor:
    """
    Traditional Dark Channel Prior (DCP) dehazing baseline.
    """
    c = I.shape[0]
    # 1. Estimate A
    A = estimate_atmospheric_light_dcp(I, patch_size) # [1, C, 1, 1]
    A = A.squeeze(0) # [C, 1, 1]
    
    # 2. Normalize and compute dark channel
    I_normalized = I / torch.clamp(A, min=1e-3)
    rgb_normalized = I_normalized[:3]
    
    min_channel = torch.min(rgb_normalized, dim=0, keepdim=True)[0]
    pad = patch_size // 2
    min_channel_padded = F.pad(-min_channel, (pad, pad, pad, pad), mode='replicate')
    dark_channel = -F.max_pool2d(min_channel_padded, kernel_size=patch_size, stride=1).squeeze(0)
    
    # 3. Estimate transmission
    t = 1.0 - omega * dark_channel
    t = torch.clamp(t, min=t0, max=1.0)
    t_full = t.unsqueeze(0).repeat(c, 1, 1)
    
    # 4. Recover clean image J
    J = (I - A) / t_full + A
    return torch.clamp(J, 0.0, 1.0)

def evaluate_benchmarks(dataset_root, split="RICE1", num_samples=3, out_dir="results", iters=200):
    """
    Runs multi-variant benchmarking comparing:
      1. Naive DCP Baseline
      2. DIP Only (no physics/priors)
      3. DIP + PINN (with physical RTE + TV + NDVI losses)
      4. DIP + PINN + DPS (sequential second stage diffusion)
    Saves outputs and generates a markdown results comparison table.
    """
    print(f"Loading {split} dataset...")
    dataset = RICEDataset(root_dir=dataset_root, split=split)
    
    run_dir = os.path.join(out_dir, f"{split}_benchmark_runs")
    os.makedirs(run_dir, exist_ok=True)
    
    # Benchmark stats structure
    results = {}
    
    variants = ["DCP Baseline", "DIP Only", "DIP + PINN", "DIP + PINN + DPS"]
    
    for v in variants:
        results[v] = {"psnr": [], "ssim": [], "sam": [], "prs": [], "time": []}
        
    num_samples = min(num_samples, len(dataset))
    
    for i in range(num_samples):
        sample = dataset[i]
        cloudy = sample['cloudy']
        clean = sample['clean']
        mask = sample['mask']
        name = sample['name']
        
        print(f"\n==================================================")
        print(f"Evaluating Image {name} ({i+1}/{num_samples})")
        print(f"==================================================")
        
        clean_np = clean.permute(1, 2, 0).numpy()
        cloudy_np = cloudy.permute(1, 2, 0).numpy()
        
        # --- 1. DCP Baseline ---
        print("\n--- Running DCP Baseline ---")
        t0 = time.time()
        pred_J_dcp = run_dcp_baseline(cloudy)
        elapsed_dcp = time.time() - t0
        
        pred_dcp_np = pred_J_dcp.permute(1, 2, 0).numpy()
        psnr_dcp = calculate_psnr(clean_np, pred_dcp_np)
        ssim_dcp = calculate_ssim(clean_np, pred_dcp_np)
        sam_dcp = calculate_sam(clean_np, pred_dcp_np)
        
        # Calculate PRS (Requires Tensors)
        t_dummy = torch.ones((1, cloudy.shape[1], cloudy.shape[2])) * 0.8
        A_dummy = torch.ones((cloudy.shape[0], 1, 1)) * 0.8
        prs_dcp = calculate_prs(cloudy, pred_J_dcp, t_dummy, A_dummy, has_nir=(cloudy.shape[0]>=4))["PRS"]
        
        results["DCP Baseline"]["psnr"].append(psnr_dcp)
        results["DCP Baseline"]["ssim"].append(ssim_dcp)
        results["DCP Baseline"]["sam"].append(sam_dcp)
        results["DCP Baseline"]["prs"].append(prs_dcp)
        results["DCP Baseline"]["time"].append(elapsed_dcp)
        
        # --- 2. DIP Only ---
        print("\n--- Running DIP Only ---")
        # We simulate "DIP Only" by setting iters small or run it with minimal/no physical constraints
        # To avoid editing the losses file configuration structure dynamically, we call optimize_dip with small iterations
        # or minimal constraints if we can. Since optimize_dip has fixed weights in dip_loop, we just run standard DIP
        # using a simple reconstruction loss. (But for a quick benchmark without rewrites, we run the base optimizer).
        # We can simulate this by running optimization with 0 iters as a dummy or low iterations.
        # Instead, let's run optimization with 100 iters and minimal loss weights by temporarily bypassing or mock optimization.
        # But to do it "properly" as in tasks: we run DIP without PINN loss.
        # Let's write a quick inline optimizer here for DIP Only so it is perfectly correct:
        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        I = cloudy.unsqueeze(0).to(device)
        from dip_loop import SimpleUNet
        dip_model = SimpleUNet(in_channels=cloudy.shape[0], out_channels=cloudy.shape[0]).to(device)
        optimizer = torch.optim.Adam(dip_model.parameters(), lr=0.01)
        z = torch.randn_like(I).to(device) * 0.1
        
        for _ in range(iters):
            optimizer.zero_grad()
            J_dip_only, _, _ = dip_model(z)
            # Reconstruct loss only
            loss = F.mse_loss(J_dip_only, I)
            loss.backward()
            optimizer.step()
            
        dip_model.eval()
        with torch.no_grad():
            J_dip_out, _, _ = dip_model(z)
        pred_J_dip = J_dip_out.squeeze(0).cpu()
        elapsed_dip = time.time() - t0
        
        pred_dip_np = pred_J_dip.permute(1, 2, 0).numpy()
        psnr_dip = calculate_psnr(clean_np, pred_dip_np)
        ssim_dip = calculate_ssim(clean_np, pred_dip_np)
        sam_dip = calculate_sam(clean_np, pred_dip_np)
        prs_dip = calculate_prs(cloudy, pred_J_dip, t_dummy, A_dummy, has_nir=(cloudy.shape[0]>=4))["PRS"]
        
        results["DIP Only"]["psnr"].append(psnr_dip)
        results["DIP Only"]["ssim"].append(ssim_dip)
        results["DIP Only"]["sam"].append(sam_dip)
        results["DIP Only"]["prs"].append(prs_dip)
        results["DIP Only"]["time"].append(elapsed_dip)
        
        # --- 3. DIP + PINN ---
        print("\n--- Running DIP + PINN ---")
        t0 = time.time()
        pred_J_pinn, pred_t_pinn = optimize_dip(cloudy, mask, num_iters=iters, use_gpu=torch.cuda.is_available(), use_dps=False)
        elapsed_pinn = time.time() - t0
        
        pred_pinn_np = pred_J_pinn.permute(1, 2, 0).numpy()
        psnr_pinn = calculate_psnr(clean_np, pred_pinn_np)
        ssim_pinn = calculate_ssim(clean_np, pred_pinn_np)
        sam_pinn = calculate_sam(clean_np, pred_pinn_np)
        prs_pinn = calculate_prs(cloudy, pred_J_pinn, pred_t_pinn.unsqueeze(0)[:1], A_dummy, has_nir=(cloudy.shape[0]>=4))["PRS"]
        
        results["DIP + PINN"]["psnr"].append(psnr_pinn)
        results["DIP + PINN"]["ssim"].append(ssim_pinn)
        results["DIP + PINN"]["sam"].append(sam_pinn)
        results["DIP + PINN"]["prs"].append(prs_pinn)
        results["DIP + PINN"]["time"].append(elapsed_pinn)
        
        # --- 4. DIP + PINN + DPS ---
        print("\n--- Running DIP + PINN + DPS ---")
        t0 = time.time()
        pred_J_dps, pred_t_dps = optimize_dip(cloudy, mask, num_iters=iters, use_gpu=torch.cuda.is_available(), use_dps=True)
        elapsed_dps = time.time() - t0
        
        pred_dps_np = pred_J_dps.permute(1, 2, 0).numpy()
        psnr_dps = calculate_psnr(clean_np, pred_dps_np)
        ssim_dps = calculate_ssim(clean_np, pred_dps_np)
        sam_dps = calculate_sam(clean_np, pred_dps_np)
        prs_dps = calculate_prs(cloudy, pred_J_dps, pred_t_dps.unsqueeze(0)[:1], A_dummy, has_nir=(cloudy.shape[0]>=4))["PRS"]
        
        results["DIP + PINN + DPS"]["psnr"].append(psnr_dps)
        results["DIP + PINN + DPS"]["ssim"].append(ssim_dps)
        results["DIP + PINN + DPS"]["sam"].append(sam_dps)
        results["DIP + PINN + DPS"]["prs"].append(prs_dps)
        results["DIP + PINN + DPS"]["time"].append(elapsed_dps)
        
        # Save output comparisons
        comparison_pil = Image.new('RGB', (256 * 5, 256))
        comparison_pil.paste(Image.fromarray((cloudy_np[::-1] * 255).astype(np.uint8)), (0, 0)) # Cloudy
        comparison_pil.paste(Image.fromarray((pred_dcp_np[::-1] * 255).astype(np.uint8)), (256, 0)) # DCP
        comparison_pil.paste(Image.fromarray((pred_dip_np[::-1] * 255).astype(np.uint8)), (256 * 2, 0)) # DIP
        comparison_pil.paste(Image.fromarray((pred_pinn_np[::-1] * 255).astype(np.uint8)), (256 * 3, 0)) # DIP + PINN
        comparison_pil.paste(Image.fromarray((pred_dps_np[::-1] * 255).astype(np.uint8)), (256 * 4, 0)) # DIP + PINN + DPS
        comparison_pil.save(os.path.join(run_dir, f"{name.split('.')[0]}_comparison.png"))
        
    # Generate aggregate stats and output markdown table
    md_content = f"# CloudClear Multi-Variant Benchmark Results ({split} Subset)\n\n"
    md_content += "| Model Variant | PSNR (dB) (H) | SSIM (H) | SAM (L) | PRS (H) | Runtime (s) (L) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    
    for v in variants:
        mean_psnr = np.mean(results[v]["psnr"])
        mean_ssim = np.mean(results[v]["ssim"])
        mean_sam = np.mean(results[v]["sam"])
        mean_prs = np.mean(results[v]["prs"])
        mean_time = np.mean(results[v]["time"])
        
        md_content += f"| **{v}** | {mean_psnr:.2f} | {mean_ssim:.4f} | {mean_sam:.4f} | {mean_prs:.4f} | {mean_time:.2f}s |\n"
        
    print("\nBenchmark Evaluation Summary:")
    print(md_content)
    
    with open(os.path.join(run_dir, "benchmark_report.md"), "w") as f:
        f.write(md_content)
        
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Benchmark results successfully saved to {run_dir}/")

if __name__ == "__main__":
    DATASET_ROOT = "RICE_DATASET"
    evaluate_benchmarks(DATASET_ROOT, split="RICE1", num_samples=1, iters=100)
