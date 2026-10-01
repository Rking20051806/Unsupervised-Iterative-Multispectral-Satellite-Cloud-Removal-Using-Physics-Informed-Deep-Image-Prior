"""
Convert ALL JP2 bands from Chandigarh_Punjab_Haryana.SAFE to high-quality
Cloud-Optimized GeoTIFFs (COGs).

- 10m bands: B02, B03, B04, B08, TCI, AOT, WVP  (7 files)
- 20m bands: B01-B12, B8A, TCI, SCL, AOT, WVP   (14 files)
- 60m bands: B01-B12, B8A, B09, TCI, SCL, AOT, WVP (15 files)

Each band is converted at its NATIVE resolution with:
  - DEFLATE compression (lossless, high quality)
  - 512x512 tiling
  - Internal overviews (pyramids) for fast zoom
"""

import os
import glob
import time
import rasterio
from rasterio.enums import Resampling


SAFE_DIR = "Chandigarh_Punjab_Haryana.SAFE"
OUT_DIR = os.path.join("cogs", "chandigarh")

# COG creation parameters - high quality
COG_PROFILE = {
    'driver': 'GTiff',
    'tiled': True,
    'blockxsize': 512,
    'blockysize': 512,
    'compress': 'deflate',
    'predictor': 2,           # Horizontal differencing for better compression
    'zlevel': 9,              # Maximum compression level
    'interleave': 'pixel',
}

# Overview levels for fast web viewing
OVERVIEW_LEVELS = [2, 4, 8, 16, 32]
OVERVIEW_RESAMPLING = Resampling.average  # Better quality than nearest


def convert_jp2_to_cog(src_path, dst_path):
    """Convert a single JP2 file to a high-quality Cloud-Optimized GeoTIFF."""
    basename = os.path.basename(src_path)
    print(f"  Converting: {basename}")
    start = time.time()

    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update(COG_PROFILE)

        # For multi-band files (like TCI with 3 bands), keep pixel interleave
        # For single-band files, band interleave is fine
        if src.count == 1:
            meta['interleave'] = 'band'

        with rasterio.open(dst_path, 'w', **meta) as dst:
            # Copy all bands
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
                dst.write(data, band_idx)

            # Copy color interpretation if available
            dst.colorinterp = src.colorinterp

            # Build internal overviews (pyramids) for fast zooming
            dst.build_overviews(OVERVIEW_LEVELS, OVERVIEW_RESAMPLING)
            dst.update_tags(ns='rio_overview', resampling='average')

    elapsed = time.time() - start
    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"    -> {os.path.basename(dst_path)} ({size_mb:.1f} MB, {elapsed:.1f}s)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find ALL JP2 files across all resolutions
    jp2_files = sorted(glob.glob(f'{SAFE_DIR}/**/*.jp2', recursive=True))

    if not jp2_files:
        print(f"ERROR: No JP2 files found in {SAFE_DIR}")
        return

    print(f"=" * 60)
    print(f" Chandigarh / Punjab / Haryana - JP2 -> COG Conversion")
    print(f" Source: {SAFE_DIR}")
    print(f" Output: {OUT_DIR}")
    print(f" Total files: {len(jp2_files)}")
    print(f"=" * 60)

    # Group by resolution for organized output
    groups = {'R10m': [], 'R20m': [], 'R60m': []}
    for f in jp2_files:
        for res in groups:
            if res in f:
                groups[res].append(f)
                break

    converted = 0
    skipped = 0
    total_start = time.time()

    for resolution, files in groups.items():
        if not files:
            continue

        print(f"\n--- {resolution} ({len(files)} bands) ---")

        # Create resolution subdirectory
        res_out = os.path.join(OUT_DIR, resolution)
        os.makedirs(res_out, exist_ok=True)

        for jp2_path in files:
            basename = os.path.basename(jp2_path)
            cog_name = basename.replace('.jp2', '_cog.tif')
            cog_path = os.path.join(res_out, cog_name)

            if os.path.exists(cog_path):
                print(f"  SKIP (exists): {cog_name}")
                skipped += 1
                continue

            convert_jp2_to_cog(jp2_path, cog_path)
            converted += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f" DONE!")
    print(f" Converted: {converted} files")
    print(f" Skipped:   {skipped} files (already existed)")
    print(f" Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f" Output:     {os.path.abspath(OUT_DIR)}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
