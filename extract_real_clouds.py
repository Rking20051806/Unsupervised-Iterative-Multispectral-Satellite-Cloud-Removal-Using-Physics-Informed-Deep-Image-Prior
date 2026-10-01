import os
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.enums import Resampling
import glob

SCL_PATH = 'cogs/china/R20m/T43SFB_20260607T054251_SCL_20m_cog.tif'
B04_PATH = 'cogs/china/R10m/T43SFB_20260607T054251_B04_10m_cog.tif'
B03_PATH = 'cogs/china/R10m/T43SFB_20260607T054251_B03_10m_cog.tif'
B02_PATH = 'cogs/china/R10m/T43SFB_20260607T054251_B02_10m_cog.tif'
OUTPUT_DIR = 'cogs/cloud_assets'
PATCH_SIZE = 256
MIN_CLOUD_PCT = 1
MAX_CLOUD_PCT = 35

CLOUD_CLASSES = [4, 8, 9, 10, 11]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_real_clouds():
    print(f"Extracting real cloud RGB patches...")
    
    with rasterio.open(SCL_PATH) as scl_src, \
         rasterio.open(B04_PATH) as b04_src, \
         rasterio.open(B03_PATH) as b03_src, \
         rasterio.open(B02_PATH) as b02_src:
        
        width = b04_src.width
        height = b04_src.height
        
        # We will save RGBA (4 bands)
        out_profile = b04_src.profile.copy()
        out_profile.update({
            'driver': 'GTiff',
            'count': 4,
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
        
        # Loop through the 10m image
        for y in range(0, height - PATCH_SIZE, PATCH_SIZE):
            for x in range(0, width - PATCH_SIZE, PATCH_SIZE):
                window_10m = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                
                # SCL is 20m, so window coords are halved
                window_20m = Window(x // 2, y // 2, PATCH_SIZE // 2, PATCH_SIZE // 2)
                
                # Read SCL and upscale to 10m
                scl_data = scl_src.read(1, window=window_20m, out_shape=(PATCH_SIZE, PATCH_SIZE), resampling=Resampling.nearest)
                
                # Check cloud percentage
                cloud_mask = np.isin(scl_data, CLOUD_CLASSES)
                cloud_pct = (np.sum(cloud_mask) / (PATCH_SIZE * PATCH_SIZE)) * 100
                
                if MIN_CLOUD_PCT <= cloud_pct <= MAX_CLOUD_PCT:
                    # Read RGB (B04, B03, B02) which are uint16
                    r = b04_src.read(1, window=window_10m).astype(np.float32)
                    g = b03_src.read(1, window=window_10m).astype(np.float32)
                    b = b02_src.read(1, window=window_10m).astype(np.float32)
                    
                    rgb = np.stack([r, g, b])
                    # Stretch to 8-bit (simple stretch from 0-3000 reflectance)
                    rgb = np.clip(rgb / 3000.0, 0, 1) * 255.0
                    rgb = rgb.astype(np.uint8)
                    
                    # Create alpha channel (255 where cloud, 0 where clear)
                    alpha = np.where(cloud_mask, 255, 0).astype(np.uint8)
                    
                    # Stack to form RGBA (4 bands)
                    rgba = np.vstack([rgb, np.expand_dims(alpha, axis=0)])
                    
                    # Also zero out the RGB where there is no cloud (to save space/be clean)
                    for b_idx in range(3):
                        rgba[b_idx] = np.where(cloud_mask, rgba[b_idx], 0)
                        
                    window_transform = b04_src.window_transform(window_10m)
                    out_profile.update({'transform': window_transform})
                    
                    pct_str = f"{int(cloud_pct):02d}"
                    pct_folder = os.path.join(OUTPUT_DIR, f"{pct_str}_percent")
                    os.makedirs(pct_folder, exist_ok=True)
                    
                    filename = f"real_cloud_pct{pct_str}_x{x}_y{y}.tif"
                    out_path = os.path.join(pct_folder, filename)
                    
                    with rasterio.open(out_path, 'w', **out_profile) as dest:
                        dest.write(rgba)
                        
                    patch_count += 1
                    if patch_count % 20 == 0:
                        print(f"  Extracted {patch_count} real RGB patches...")

    print(f"Done! Extracted {patch_count} real RGB cloud patches to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_real_clouds()
