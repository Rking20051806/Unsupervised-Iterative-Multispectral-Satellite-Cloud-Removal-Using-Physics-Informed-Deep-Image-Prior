import os
import glob
import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image

def convert_jp2_to_cog(src_path, dst_path):
    print(f"Converting {src_path} -> {dst_path}")
    with rasterio.open(src_path) as src:
        kwargs = src.meta.copy()
        
        # Update kwargs for Cloud Optimized GeoTIFF (COG)
        # 1. Driver must be GTiff
        # 2. Must be tiled (tiled=True)
        # 3. Block size (blockxsize, blockysize) usually 256 or 512
        # 4. Compress
        kwargs.update({
            'driver': 'GTiff',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'deflate',
            'interleave': 'pixel'
        })
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            # Read and write the data
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
                
            # Create overviews (pyramids)
            overviews = [2, 4, 8, 16]
            dst.build_overviews(overviews, Resampling.nearest)
            dst.update_tags(ns='rio_overview', resampling='nearest')

def main():
    # Pick bands to convert
    bands = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2']
    
    datasets = [
        ('China.SAFE', 'China_COGs'),
        ('Vidarbha_Nagpur_Maharashtra.SAFE', os.path.join('cogs', 'nagpur')),
        ('Vidarbha_Yavatmal_Maharashtra.SAFE', os.path.join('cogs', 'yavatmal'))
    ]
    
    cog_paths = []
    
    for safe_dir, out_dir in datasets:
        print(f"Processing {safe_dir}...")
        jp2_files = glob.glob(f'{safe_dir}/**/*.jp2', recursive=True)
        target_files = [f for f in jp2_files if any(f.endswith(b) for b in bands)]
        
        os.makedirs(out_dir, exist_ok=True)
        
        for f in target_files:
            basename = os.path.basename(f)
            dst_name = basename.replace('.jp2', '_cog.tif')
            dst_path = os.path.join(out_dir, dst_name)
            if not os.path.exists(dst_path):
                convert_jp2_to_cog(f, dst_path)
            cog_paths.append(dst_path)
            
    print("Conversion complete!")
    
    # Generate visualization (optional, we can just skip it or only use the last dataset's paths)
    # The visualization logic expects B04, B03, B02 to be in cog_paths.
    # Since we are processing multiple datasets, creating one combined TCI might mix them up.
    # Let's create TCI for each output directory separately.
    
    for _, out_dir in datasets:
        red = green = blue = None
        # get paths for this output dir
        dir_cogs = [p for p in cog_paths if p.startswith(out_dir)]
        if not dir_cogs:
            continue
            
        for p in dir_cogs:
            if 'B04' in p:
                with rasterio.open(p) as src:
                    out_shape = (src.height // 16, src.width // 16)
                    red = src.read(1, out_shape=out_shape)
            if 'B03' in p:
                with rasterio.open(p) as src:
                    out_shape = (src.height // 16, src.width // 16)
                    green = src.read(1, out_shape=out_shape)
            if 'B02' in p:
                with rasterio.open(p) as src:
                    out_shape = (src.height // 16, src.width // 16)
                    blue = src.read(1, out_shape=out_shape)
                    
        if red is not None and green is not None and blue is not None:
            rgb = np.stack((red, green, blue), axis=-1).astype(np.float32)
            # Normalize and brighten
            rgb = rgb / 10000.0 * 2.5
            rgb = np.clip(rgb, 0, 1) * 255.0
            rgb = rgb.astype(np.uint8)
            
            img = Image.fromarray(rgb)
            
            out_img = os.path.join(out_dir, 'tci_overview.png')
            img.save(out_img)
            print(f"Saved visualization to {out_img}")

if __name__ == '__main__':
    main()
