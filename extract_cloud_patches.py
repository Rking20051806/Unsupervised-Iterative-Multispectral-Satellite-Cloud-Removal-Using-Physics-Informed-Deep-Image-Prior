import os
import rasterio
import numpy as np
from rasterio.windows import Window
import gc
from rasterio.enums import Resampling

# Configuration
SCL_PATH = 'cogs/china/R20m/T43SFB_20260607T054251_SCL_20m_cog.tif'
OUTPUT_DIR = 'cogs/cloud_patches'
PATCH_SIZE = 256
MIN_CLOUD_PCT = 1
MAX_CLOUD_PCT = 35

# Cloud classes in Sentinel-2 SCL
CLOUD_CLASSES = [4, 8, 9, 10, 11]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_patches():
    print(f"Reading SCL mask from: {SCL_PATH}")
    
    with rasterio.open(SCL_PATH) as src:
        width = src.width
        height = src.height
        
        # Profile for the output patch
        out_profile = src.profile.copy()
        out_profile.update({
            'driver': 'GTiff',
            'height': PATCH_SIZE,
            'width': PATCH_SIZE,
            'tiled': True,
            'blockxsize': 128,
            'blockysize': 128,
            'compress': 'deflate',
            'nodata': 0,
            'dtype': 'uint8'
        })
        
        patch_count = 0
        
        # Loop through the image in chunks
        for y in range(0, height - PATCH_SIZE, PATCH_SIZE):
            for x in range(0, width - PATCH_SIZE, PATCH_SIZE):
                
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                scl_data = src.read(1, window=window)
                
                # Create boolean mask for clouds
                cloud_mask = np.isin(scl_data, CLOUD_CLASSES)
                
                # Calculate percentage
                cloud_pixels = np.sum(cloud_mask)
                total_pixels = PATCH_SIZE * PATCH_SIZE
                cloud_pct = (cloud_pixels / total_pixels) * 100
                
                if MIN_CLOUD_PCT <= cloud_pct <= MAX_CLOUD_PCT:
                    # Convert to binary mask (255 for cloud, 0 for background)
                    # This removes all other background data and makes it visible
                    binary_mask = np.where(cloud_mask, 255, 0).astype(np.uint8)
                    
                    # Update transform for the specific window
                    window_transform = src.window_transform(window)
                    out_profile.update({
                        'transform': window_transform
                    })
                    
                    # Sort into percentage folders and rename
                    pct_str = f"{int(cloud_pct):02d}"
                    pct_folder = os.path.join(OUTPUT_DIR, f"{pct_str}_percent")
                    os.makedirs(pct_folder, exist_ok=True)
                    
                    filename = f"mask_pct{pct_str}_x{x}_y{y}.tif"
                    out_path = os.path.join(pct_folder, filename)
                    
                    with rasterio.open(out_path, 'w', **out_profile) as dest:
                        dest.write(binary_mask, 1)
                        
                    patch_count += 1
                    if patch_count % 50 == 0:
                        print(f"  Extracted {patch_count} patches so far...")

    print(f"\nDone! Extracted a total of {patch_count} cloud patches.")
    print(f"Saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    if os.path.exists(SCL_PATH):
        extract_patches()
    else:
        print(f"Error: Could not find SCL file at {SCL_PATH}")
