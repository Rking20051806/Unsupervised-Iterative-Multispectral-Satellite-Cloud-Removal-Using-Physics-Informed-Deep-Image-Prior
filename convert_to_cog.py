import os
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

PROJECT_ROOT = Path(__file__).resolve().parent

S2_MAPS = {
    "Vidarbha_Nagpur_Maharashtra.SAFE": "Vidarbha_Nagpur_Maharashtra.SAFE",
    "Vidarbha_Yavatmal_Maharashtra.SAFE": "Vidarbha_Yavatmal_Maharashtra.SAFE",
    "China.SAFE": "China.SAFE"
}

BANDS_TO_CONVERT = ["B02", "B03", "B04", "B08", "TCI", "SCL"]

def find_band_jp2(safe_dir: Path, band: str) -> Path:
    search_dir = safe_dir / "GRANULE" if (safe_dir / "GRANULE").exists() else safe_dir
    band_lower = band.lower()
    for root, dirs, files in os.walk(search_dir):
        dirs[:] = [d for d in dirs if d.lower() not in {'.git', '.venv', 'isro', '__pycache__', 'scratch', 'outputs', 'results', 'cogs'}]
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith('.jp2'):
                if f"_{band_lower}_" in file_lower:
                    return Path(root) / file
    return None

def convert_band_to_cog(src_jp2: Path, dst_cog: Path) -> bool:
    tmp_tif = dst_cog.with_suffix('.tmp.tif')
    try:
        with rasterio.open(src_jp2) as src:
            profile = src.profile.copy()
            profile.update({
                'driver': 'GTiff',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': 'deflate',
                'interleave': 'pixel'
            })
            data = src.read()
            with rasterio.open(tmp_tif, 'w', **profile) as dst:
                dst.write(data)
                overviews = [2, 4, 8, 16, 32, 64]
                res = Resampling.nearest if 'scl' in src_jp2.name.lower() else Resampling.bilinear
                dst.build_overviews(overviews, res)
                dst.update_tags(ns='rio_overview', resampling=res.name)

        cog_prof = cog_profiles.get("deflate")
        cog_translate(str(tmp_tif), str(dst_cog), cog_prof, quiet=True)
        if tmp_tif.exists():
            tmp_tif.unlink()
            
        valid, errors, _ = cog_validate(str(dst_cog))
        if not valid:
            print(f"    [Warning] COG validation errors for {dst_cog.name}: {errors}")
        return True
    except Exception as e:
        print(f"    [Error] Failed to convert {src_jp2.name} to COG: {e}")
        if tmp_tif.exists():
            tmp_tif.unlink()
        return False

def convert_to_cog():
    cog_dir = PROJECT_ROOT / "cogs"
    cog_dir.mkdir(exist_ok=True)

    for map_id, safe_name in S2_MAPS.items():
        safe_path = PROJECT_ROOT / safe_name
        if not safe_path.exists():
            print(f"Skipping {safe_name}, directory does not exist.")
            continue

        map_cog_dir = cog_dir / map_id
        map_cog_dir.mkdir(exist_ok=True)
        print(f"\n--- Processing COGs for {map_id} ---")

        for band in BANDS_TO_CONVERT:
            dst_cog = map_cog_dir / f"{band}.tif"
            if dst_cog.exists():
                valid, _, _ = cog_validate(str(dst_cog))
                if valid:
                    print(f"  [Skip] {band}.tif already exists and is a valid COG.")
                    continue

            src_jp2 = find_band_jp2(safe_path, band)
            if not src_jp2:
                print(f"  [Missing] Could not locate JP2 for band {band}")
                continue

            print(f"  Converting {band} ({src_jp2.name}) -> {dst_cog.name} ...")
            convert_band_to_cog(src_jp2, dst_cog)

if __name__ == "__main__":
    convert_to_cog()

