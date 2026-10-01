"""
Convert ALL JP2 bands from China, Nagpur, and Yavatmal SAFE folders
to high-quality Cloud-Optimized GeoTIFFs (COGs).

Matches the same quality as the Chandigarh conversion:
  - DEFLATE compression (lossless)
  - 512x512 tiling
  - Internal overviews (pyramids) for fast zoom
  - All bands at all resolutions (R10m, R20m, R60m)
"""

import os
import gc
import glob
import time
import rasterio
from rasterio.enums import Resampling


# COG creation parameters - high quality
COG_PROFILE = {
    'driver': 'GTiff',
    'tiled': True,
    'blockxsize': 512,
    'blockysize': 512,
    'compress': 'deflate',
    'zlevel': 6,
    'interleave': 'pixel',
}

OVERVIEW_LEVELS = [2, 4, 8, 16, 32]
OVERVIEW_RESAMPLING = Resampling.average


# Datasets to convert: (SAFE_DIR, COG_OUTPUT_DIR)
DATASETS = [
    ('China.SAFE', os.path.join('cogs', 'china')),
    ('Vidarbha_Nagpur_Maharashtra.SAFE', os.path.join('cogs', 'nagpur')),
    ('Vidarbha_Yavatmal_Maharashtra.SAFE', os.path.join('cogs', 'yavatmal')),
]


def convert_jp2_to_cog(src_path, dst_path):
    """Convert a single JP2 file to a high-quality COG.
    
    Writes data first, closes the file, then reopens in r+ mode
    to build overviews. This avoids the IReadBlock/TIFFReadEncodedTile
    error that occurs when building overviews in the same write session.
    """
    basename = os.path.basename(src_path)
    print(f"  Converting: {basename}")
    start = time.time()

    # Step 1: Read source and write data to COG
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update(COG_PROFILE)
        color_interp = src.colorinterp

        if src.count == 1:
            meta['interleave'] = 'band'

        with rasterio.open(dst_path, 'w', **meta) as dst:
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
                dst.write(data, band_idx)
            dst.colorinterp = color_interp
            # Do NOT build overviews here - causes IReadBlock errors

    # Step 2: Reopen in r+ mode to build overviews (file is fully written now)
    with rasterio.open(dst_path, 'r+') as dst:
        dst.build_overviews(OVERVIEW_LEVELS, OVERVIEW_RESAMPLING)
        dst.update_tags(ns='rio_overview', resampling='average')

    elapsed = time.time() - start
    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"    -> {os.path.basename(dst_path)} ({size_mb:.1f} MB, {elapsed:.1f}s)")


def safe_cleanup(filepath):
    """Try to remove a file, handling Windows file locking."""
    gc.collect()  # Release any dangling rasterio handles
    for attempt in range(3):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"  Cleaned up partial file: {os.path.basename(filepath)}")
            return
        except PermissionError:
            time.sleep(1)  # Wait for file lock to release
    print(f"  WARNING: Could not clean up {os.path.basename(filepath)} (file locked)")


def main():
    grand_total_start = time.time()
    grand_converted = 0
    grand_skipped = 0
    grand_errors = 0

    for safe_dir, out_base in DATASETS:
        if not os.path.isdir(safe_dir):
            print(f"SKIP: {safe_dir} not found")
            continue

        # Find ALL JP2 files
        jp2_files = sorted(glob.glob(f'{safe_dir}/**/*.jp2', recursive=True))

        if not jp2_files:
            print(f"SKIP: No JP2 files in {safe_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f" {safe_dir} -> {out_base}")
        print(f" Total JP2 files: {len(jp2_files)}")
        print(f"{'=' * 60}")

        # Group by resolution
        groups = {'R10m': [], 'R20m': [], 'R60m': []}
        for f in jp2_files:
            for res in groups:
                if res in f:
                    groups[res].append(f)
                    break

        for resolution, files in groups.items():
            if not files:
                continue

            print(f"\n--- {resolution} ({len(files)} bands) ---")

            res_out = os.path.join(out_base, resolution)
            os.makedirs(res_out, exist_ok=True)

            for jp2_path in files:
                basename = os.path.basename(jp2_path)
                cog_name = basename.replace('.jp2', '_cog.tif')
                cog_path = os.path.join(res_out, cog_name)

                if os.path.exists(cog_path):
                    print(f"  SKIP (exists): {cog_name}")
                    grand_skipped += 1
                    continue

                try:
                    convert_jp2_to_cog(jp2_path, cog_path)
                    grand_converted += 1
                except Exception as e:
                    print(f"  ERROR converting {cog_name}: {e}")
                    safe_cleanup(cog_path)
                    grand_errors += 1

    grand_elapsed = time.time() - grand_total_start

    print(f"\n{'=' * 60}")
    print(f" ALL DATASETS DONE!")
    print(f" Converted: {grand_converted} files")
    print(f" Skipped:   {grand_skipped} files (already existed)")
    print(f" Errors:    {grand_errors} files")
    print(f" Total time: {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
