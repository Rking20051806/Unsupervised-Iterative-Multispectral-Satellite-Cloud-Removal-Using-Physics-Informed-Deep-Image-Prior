import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import rasterio
from tqdm import tqdm

# Add current directory to path to ensure local imports work
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from s2_loader import Sentinel2Loader
from dip_loop import optimize_dip

def get_feather_mask(h, w, overlap):
    """
    Generates a 2D weight mask that tapers to 0 at the boundaries to enable seamless blending.
    """
    mask = np.ones((h, w), dtype=np.float32)
    if overlap <= 0:
        return mask
        
    for i in range(min(overlap, h // 2)):
        val = i / overlap
        mask[i, :] *= val
        mask[-1-i, :] *= val
        
    for j in range(min(overlap, w // 2)):
        val = j / overlap
        mask[:, j] *= val
        mask[:, -1-j] *= val
        
    return mask

def process_full_tile(product_dir: str, output_path: str, patch_size: int = 256, overlap: int = 32, 
                      iters: int = 150, use_dps: bool = False, dps_weight: float = 0.5, cloud_threshold: float = 0.05):
    """
    Loads a full Sentinel-2 tile, processes cloudy patches with the DIP+PINN(+DPS) pipeline,
    and stitches them seamlessly back into the full-scale image.
    """
    print(f"Initializing Sentinel-2 loader for: {product_dir}")
    loader = Sentinel2Loader(product_dir)
    
    # 1. Load full bands and cloud mask
    bands_4ch, cloud_mask, meta = loader.load_full_bands()
    C, H, W = bands_4ch.shape
    print(f"Full tile loaded. Resolution: {W}x{H}, Channels: {C}")
    
    # 2. Generate and classify patches
    all_patches = loader.generate_patches(H, W, patch_size=patch_size, overlap=overlap)
    cloudy_patches, clear_patches = loader.classify_patches(cloud_mask, all_patches, threshold=cloud_threshold)
    
    print(f"Total patches: {len(all_patches)}")
    print(f"  Clear patches (copied directly): {len(clear_patches)}")
    print(f"  Cloudy patches (to process): {len(cloudy_patches)}")
    
    if len(cloudy_patches) == 0:
        print("No cloudy patches found above the threshold. Saving original image.")
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(bands_4ch * 10000.0) # Scale back to Sentinel-2 DN scale
        return
        
    # 3. Setup Accumulators for blending
    # Initialise output image with original bands where clear, and 0 where cloudy patches exist
    accum_image = np.zeros_like(bands_4ch)
    accum_weight = np.zeros((H, W), dtype=np.float32)
    
    # Pre-fill clear-sky pixels directly from original image
    clear_pixels = (cloud_mask < 0.5)
    for c in range(C):
        accum_image[c, clear_pixels] = bands_4ch[c, clear_pixels]
    accum_weight[clear_pixels] = 1.0
    
    # 4. Process Cloudy Patches
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Processing cloudy patches on: {device}")
    
    for idx, patch in enumerate(tqdm(cloudy_patches, desc="Reconstructing patches")):
        x, y, h, w = patch['x'], patch['y'], patch['h'], patch['w']
        
        # Extract patch bands & mask
        patch_bands, patch_mask = loader.load_crop(x, y, w, h)
        
        # Convert to PyTorch tensors
        patch_bands_t = torch.from_numpy(patch_bands)
        patch_mask_t = torch.from_numpy(patch_mask)
        
        # Run optimization with physics and loader spatial priors
        clean_patch_t, _ = optimize_dip(
            cloudy_image_tensor=patch_bands_t,
            mask_tensor=patch_mask_t,
            num_iters=iters,
            lr=0.005,
            use_gpu=torch.cuda.is_available(),
            use_dps=use_dps,
            dps_weight=dps_weight,
            s2_loader=loader,
            x_start=x,
            y_start=y,
            crop_size=h
        )
        
        clean_patch = clean_patch_t.numpy() # [C, h, w]
        
        # Get feathering mask
        feather = get_feather_mask(h, w, overlap=overlap)
        
        # Accumulate
        for c in range(C):
            accum_image[c, y:y+h, x:x+w] += clean_patch[c] * feather
        accum_weight[y:y+h, x:x+w] += feather
        
    # 5. Normalize Accumulator to get final image
    final_image = accum_image / np.clip(accum_weight, 1e-8, None)
    
    # Clamp to physical range [0, 1]
    final_image = np.clip(final_image, 0.0, 1.0)
    
    # Scale back to Sentinel-2 L2A Reflectance scale (0 - 10000)
    final_scaled = (final_image * 10000.0).astype(np.uint16)
    
    # 6. Save reconstructed GeoTIFF
    print(f"Writing final reconstructed GeoTIFF to: {output_path}")
    meta.update(dtype=rasterio.uint16)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(final_scaled)
        
    print("Inference completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-tile Sentinel-2 cloud removal and stitching.")
    parser.add_argument("--product_dir", type=str, default=".", help="Path to Sentinel-2 product folder")
    parser.add_argument("--output", type=str, default="reconstructed_full.tif", help="Output file path")
    parser.add_argument("--patch_size", type=int, default=256, help="Stitch patch size")
    parser.add_argument("--overlap", type=int, default=32, help="Patch overlap width")
    parser.add_argument("--iters", type=int, default=150, help="DIP iterations per patch")
    parser.add_argument("--use_dps", action="store_true", help="Enable sequential DPS prior")
    parser.add_argument("--dps_weight", type=float, default=0.5, help="DPS conditioning weight")
    parser.add_argument("--cloud_threshold", type=float, default=0.05, help="Minimum cloud fraction to trigger patch reconstruction")
    
    args = parser.parse_args()
    process_full_tile(
        product_dir=args.product_dir,
        output_path=args.output,
        patch_size=args.patch_size,
        overlap=args.overlap,
        iters=args.iters,
        use_dps=args.use_dps,
        dps_weight=args.dps_weight,
        cloud_threshold=args.cloud_threshold
    )
