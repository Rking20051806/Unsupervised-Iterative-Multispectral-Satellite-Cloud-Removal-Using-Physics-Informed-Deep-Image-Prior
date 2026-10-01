import os
import time
import numpy as np
import torch
import rasterio
from s2_loader import Sentinel2Loader
from dip_loop import optimize_dip

def generate_blending_window(h: int, w: int) -> np.ndarray:
    """
    Creates a 2D linear blending window that goes to 0 at the edges.
    """
    # 1D windows
    w_y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    w_y = w_y / max(1.0, np.max(w_y))
    
    w_x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    w_x = w_x / max(1.0, np.max(w_x))
    
    # 2D window
    window_2d = w_y[:, None] * w_x[None, :]
    return np.clip(window_2d, 1e-4, 1.0)

def reconstruct_tile(product_dir: str, output_path: str, patch_size: int = 256, overlap: int = 32, 
                     dip_iters: int = 200, crop_box: dict | None = None, max_cloudy_patches: int = -1,
                     use_dps: bool = False, dps_weight: float = 0.5, **dip_kwargs):
    """
    Orchestrates the full or cropped Sentinel-2 tile reconstruction with physics-informed DIP.
    """
    start_time = time.time()
    loader = Sentinel2Loader(product_dir)
    
    # 1. Load full bands
    bands_4ch, cloud_mask, meta = loader.load_full_bands()
    
    C, H, W = bands_4ch.shape
    
    # If crop_box is provided, we restrict the reconstruction to that region
    if crop_box:
        cy, cx, ch, cw = crop_box['y'], crop_box['x'], crop_box['h'], crop_box['w']
        print(f"Running cropped reconstruction on region: y={cy}, x={cx}, h={ch}, w={cw}")
        bands_4ch = bands_4ch[:, cy:cy+ch, cx:cx+cw]
        cloud_mask = cloud_mask[cy:cy+ch, cx:cx+cw]
        H, W = ch, cw
        
        # Update metadata transform for the crop
        orig_transform = meta['transform']
        crop_transform = rasterio.Affine(
            orig_transform.a, orig_transform.b, orig_transform.c + cx * orig_transform.a,
            orig_transform.d, orig_transform.e, orig_transform.f + cy * orig_transform.e
        )
        meta.update(width=cw, height=ch, transform=crop_transform)
        
    print(f"Image size for processing: {H}x{W}")
    
    # 2. Generate patches
    all_patches = loader.generate_patches(H, W, patch_size=patch_size, overlap=overlap)
    cloudy_patches, clear_patches = loader.classify_patches(cloud_mask, all_patches, threshold=0.01)
    
    print(f"Total Patches: {len(all_patches)}")
    print(f"  Clear Patches (will be kept as is): {len(clear_patches)}")
    print(f"  Cloudy Patches (will be optimized): {len(cloudy_patches)}")
    
    if max_cloudy_patches > 0:
        print(f"Limiting cloudy patch optimization to first {max_cloudy_patches} patches for demo.")
        cloudy_patches = cloudy_patches[:max_cloudy_patches]
        
    # 3. Blending Setup
    accum_image = np.zeros((C, H, W), dtype=np.float32)
    accum_weight = np.zeros((H, W), dtype=np.float32)
    
    # Copy clear patches directly to accumulator to save time
    for patch in clear_patches:
        y, x, h, w = patch['y'], patch['x'], patch['h'], patch['w']
        window = generate_blending_window(h, w)
        accum_image[:, y:y+h, x:x+w] += bands_4ch[:, y:y+h, x:x+w] * window
        accum_weight[y:y+h, x:x+w] += window
        
    # Process cloudy patches using DIP+PINN
    for i, patch in enumerate(cloudy_patches):
        y, x, h, w = patch['y'], patch['x'], patch['h'], patch['w']
        print(f"\n[Patch {i+1}/{len(cloudy_patches)}] Optimizing cloudy patch at (y:{y}, x:{x}) with cloud fraction {patch['cloud_fraction']:.2f}")
        
        # Crop cloudy image and mask tensors
        patch_image = torch.from_numpy(bands_4ch[:, y:y+h, x:x+w])
        patch_mask = torch.from_numpy(cloud_mask[y:y+h, x:x+w])
        
        cx_full = (crop_box['x'] if crop_box else 0) + x
        cy_full = (crop_box['y'] if crop_box else 0) + y

        # Default physics parameters tuned for Sentinel-2 MSI bands if not overridden:
        # B02 (0.4924 um), B03 (0.5598 um), B04 (0.6646 um), B08 (0.8328 um)
        s2_lambdas = [0.4924, 0.5598, 0.6646, 0.8328]
        s2_beta_abs = [0.002, 0.015, 0.010, 0.005]
        
        opt_kwargs = {
            'lr': dip_kwargs.get('lr', 0.005),
            'lambda_asm': dip_kwargs.get('lambda_asm', 0.6),
            'lambda_rte': dip_kwargs.get('lambda_rte', 0.3),
            'lambda_ndvi': dip_kwargs.get('lambda_ndvi', 0.08),
            'lambda_tv': dip_kwargs.get('lambda_tv', 5e-5),
            'lambda_sam': dip_kwargs.get('lambda_sam', 0.08),
            'lambda_t_prior': dip_kwargs.get('lambda_t_prior', 0.3),
            'lambda_perceptual': dip_kwargs.get('lambda_perceptual', 0.10),
            'lambda_edge': dip_kwargs.get('lambda_edge', 0.08),
            'alpha': dip_kwargs.get('alpha', 1.3),
            'lambdas': dip_kwargs.get('lambdas', s2_lambdas),
            'beta_abs': dip_kwargs.get('beta_abs', s2_beta_abs),
        }
        # Include any remaining custom kwargs
        for k, v in dip_kwargs.items():
            if k not in opt_kwargs and k != 'lr':
                opt_kwargs[k] = v

        # Optimize DIP with physics and spectral constraints
        pred_J, pred_t = optimize_dip(
            cloudy_image_tensor=patch_image,
            mask_tensor=patch_mask,
            num_iters=dip_iters,
            use_gpu=torch.cuda.is_available(),
            use_dps=use_dps,
            dps_weight=dps_weight,
            s2_loader=loader,
            x_start=cx_full,
            y_start=cy_full,
            crop_size=h,
            **opt_kwargs
        )
        
        # ── Adaptive Reconstruction per cloud type (Thick Cloud, Thin Cloud / Haze) ──
        # If transmission map is available, apply physical dehaze for thin cloud/haze
        # and rely on DIP generative inpainting for thick clouds.
        with torch.no_grad():
            I_patch = patch_image.float()
            J_dip = pred_J.float()
            t_patch = pred_t.float() if pred_t.dim() == 3 else pred_t.squeeze(0).float()
            
            # Safe transmission threshold to avoid divide-by-zero on thick cloud cores
            t_safe = torch.clamp(t_patch, min=0.08, max=0.99)
            # Estimate atmospheric airlight from top 1% brightest pixels
            flat_I = I_patch.reshape(C, -1)
            k_bright = max(10, int(flat_I.shape[1] * 0.01))
            bright_idx = torch.topk(flat_I[:3].mean(dim=0), k=k_bright)[1]
            A_est = flat_I[:, bright_idx].mean(dim=1, keepdim=True).unsqueeze(-1)
            A_est = torch.clamp(A_est, 0.65, 0.90)
            
            # Atmospheric Scattering Model (ASM) physical inversion for haze/thin cloud
            J_dehazed = torch.clamp((I_patch - A_est * (1.0 - t_safe)) / t_safe, 0.0, 1.0)
            
            # Cloud density: Thick cloud (high reflectance + low transmission) vs Thin/Haze
            cloud_brightness = I_patch[:3].mean(dim=0, keepdim=True)
            thick_cloud_weight = torch.clamp((cloud_brightness - 0.35) / 0.40, 0.0, 1.0) * (1.0 - t_safe[:1])
            
            # Combined radiance: DIP synthesis in thick core + Physical ASM in thin haze
            J_adaptive = (1.0 - thick_cloud_weight) * J_dehazed + thick_cloud_weight * J_dip
            
            # ── Ground Chromaticity & Tone Preservation ──
            # Match reconstructed patch channel means to the surrounding clear terrain pixels
            # to prevent color shifts (yellowish/reddish or desaturated patches)
            clear_mask_patch = (M_patch < 0.2)
            if clear_mask_patch.sum() > 50:
                for c in range(C):
                    mean_clear_orig = I_patch[c][clear_mask_patch.squeeze(0)].mean()
                    mean_recon = J_adaptive[c].mean()
                    scale_factor = torch.clamp(mean_clear_orig / (mean_recon + 1e-6), 0.80, 1.25)
                    J_adaptive[c] = torch.clamp(J_adaptive[c] * scale_factor, 0.0, 1.0)
            
            # Soft mask feathering to eliminate border seams
            M_patch_t = patch_mask.float()
            if M_patch_t.dim() == 2:
                M_patch_t = M_patch_t.unsqueeze(0)
            
            import scipy.ndimage as ndimage
            M_np = M_patch_t.squeeze().numpy()
            M_feathered = ndimage.gaussian_filter(M_np, sigma=2.0)
            M_feathered_t = torch.from_numpy(M_feathered).unsqueeze(0).expand(C, -1, -1)
            
            # Preserve original clear pixels exactly, restore only cloudy/hazy pixels
            J_final_patch = (1.0 - M_feathered_t) * I_patch + M_feathered_t * J_adaptive
            pred_np = torch.clamp(J_final_patch, 0.0, 1.0).numpy()
        
        # Stitched update
        window = generate_blending_window(h, w)
        
        accum_image[:, y:y+h, x:x+w] += pred_np * window
        accum_weight[y:y+h, x:x+w] += window
        
    # 4. Final normalization and assembly
    # For any pixels that were not covered by any patch (should not happen with proper grid), use fallback
    mask_valid = accum_weight > 0
    final_image = np.copy(bands_4ch)
    
    for c in range(C):
        final_image[c, mask_valid] = accum_image[c, mask_valid] / accum_weight[mask_valid]
        
    # Clamp reflectance to [0, 1]
    final_image = np.clip(final_image, 0.0, 1.0)
    
    # ── No-Reference Image Quality Metrics Calculation ──
    # Computes Entropy, Tenengrad Gradient Sharpness, and Contrast on reconstructed image
    nr_metrics = compute_no_reference_metrics(final_image)
    print("\n--- Reconstructed Sentinel-2 Single-Image Quality Metrics ---")
    for k, v in nr_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("-----------------------------------------------------------\n")
    
    # Update metadata to save as standard GeoTIFF
    meta.update(driver='GTiff', dtype='uint16')
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        for c in range(C):
            # Scale back to uint16 (reflectance * 10000) for standard GeoTIFF format
            band_data = (final_image[c] * 10000.0).astype(np.uint16)
            dst.write(band_data, c + 1)
            
    print(f"Reconstruction finished in {time.time() - start_time:.2f} seconds!")
    return output_path


def compute_no_reference_metrics(img_4ch: np.ndarray) -> dict[str, float]:
    """
    Computes single-image (no reference ground truth needed) metrics for Sentinel-2 reconstruction:
      - Spatial Gradient / Tenengrad Sharpness (detail preservation)
      - Shannon Entropy (information content)
      - Dynamic Range / Contrast (contrast preservation)
      - Spectral Angle Mapper (SAM) with clear baseline
    """
    import scipy.ndimage as ndimage
    
    # Convert RGB for spatial metrics
    rgb = np.clip(img_4ch[:3].transpose(1, 2, 0), 0.0, 1.0)
    gray = np.mean(rgb, axis=2)
    
    # 1. Gradient / Tenengrad Sharpness
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    sharpness = float(np.mean(np.sqrt(gx**2 + gy**2 + 1e-8)))
    
    # 2. Shannon Entropy (Information Content)
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-8)) / 256.0)
    
    # 3. Michelson Contrast
    p95 = np.percentile(gray, 95)
    p5 = np.percentile(gray, 5)
    contrast = float((p95 - p5) / (p95 + p5 + 1e-8))
    
    return {
        "Tenengrad Sharpness": round(sharpness, 4),
        "Shannon Entropy": round(entropy, 4),
        "Contrast (P95-P5)": round(contrast, 4)
    }

if __name__ == '__main__':
    # Test script on a small crop
    product_dir = "."
    output_path = "results/reconstructed_crop.tif"
    
    # A small 1024x1024 region containing some cloud
    test_crop = {'y': 3000, 'x': 3000, 'h': 512, 'w': 512}
    
    try:
        reconstruct_tile(
            product_dir=product_dir,
            output_path=output_path,
            patch_size=256,
            overlap=32,
            dip_iters=100,
            crop_box=test_crop,
            max_cloudy_patches=2 # Limit to 2 patches for fast test
        )
    except Exception as e:
        print("Reconstruction test failed:", e)
