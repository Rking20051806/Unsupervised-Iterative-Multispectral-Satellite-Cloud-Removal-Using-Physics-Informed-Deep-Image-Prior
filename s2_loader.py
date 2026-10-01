import os
import glob
from pathlib import Path
import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
import torch

class Sentinel2Loader:
    def __init__(self, product_dir: str):
        self.product_dir = Path(product_dir)
        self.band_paths = self._find_band_paths()
        self._dataset_cache = {}

    def get_dataset(self, band_key: str) -> rasterio.DatasetReader:
        if band_key not in self._dataset_cache:
            if band_key not in self.band_paths:
                raise FileNotFoundError(f"Band path for {band_key} not located.")
            self._dataset_cache[band_key] = rasterio.open(self.band_paths[band_key])
        return self._dataset_cache[band_key]

    def close(self):
        for src in self._dataset_cache.values():
            try:
                src.close()
            except Exception:
                pass
        self._dataset_cache.clear()

        
    def _find_band_paths(self) -> dict[str, Path]:
        """
        Locates B02, B03, B04, B08, TCI, and SCL paths. Checks cogs/{product_name} first for COGs per band.
        """
        paths = {}
        required = ['B02', 'B03', 'B04', 'B08', 'TCI', 'SCL']
        
        # Define all possible COG search directories
        possible_cog_dirs = [
            self.product_dir.parent / "cogs" / self.product_dir.name,
            self.product_dir.parent / "China_COGs",
            self.product_dir.parent / "cogs"
        ]
        
        # Check explicit COG directories first
        for cog_dir in possible_cog_dirs:
            if cog_dir.exists():
                for root, dirs, files in os.walk(cog_dir):
                    for file in files:
                        file_lower = file.lower()
                        if file_lower.endswith('.tif'):
                            path = Path(root) / file
                            if '_b02' in file_lower or file_lower == 'b02.tif':
                                paths.setdefault('B02', path)
                            elif '_b03' in file_lower or file_lower == 'b03.tif':
                                paths.setdefault('B03', path)
                            elif '_b04' in file_lower or file_lower == 'b04.tif':
                                paths.setdefault('B04', path)
                            elif '_b08' in file_lower or file_lower == 'b08.tif':
                                paths.setdefault('B08', path)
                            elif '_tci' in file_lower or file_lower == 'tci.tif':
                                paths.setdefault('TCI', path)
                            elif '_scl' in file_lower or file_lower == 'scl.tif':
                                paths.setdefault('SCL', path)

        missing = [r for r in required if r not in paths]
        if missing:
            search_dir = self.product_dir / "GRANULE" if (self.product_dir / "GRANULE").exists() else self.product_dir
            for root, dirs, files in os.walk(search_dir):
                dirs[:] = [d for d in dirs if d.lower() not in {'.git', '.venv', 'isro', '__pycache__', 'scratch', 'outputs', 'results'}]
                for file in files:
                    file_lower = file.lower()
                    if file_lower.endswith('.jp2') or file_lower.endswith('.tif'):
                        path = Path(root) / file
                        if '_b02_10m' in file_lower or file_lower == 'b02.tif':
                            paths.setdefault('B02', path)
                        elif '_b03_10m' in file_lower or file_lower == 'b03.tif':
                            paths.setdefault('B03', path)
                        elif '_b04_10m' in file_lower or file_lower == 'b04.tif':
                            paths.setdefault('B04', path)
                        elif '_b08_10m' in file_lower or file_lower == 'b08.tif':
                            paths.setdefault('B08', path)
                        elif '_tci_10m' in file_lower or file_lower == 'tci.tif' or 'tci_overview' in file_lower:
                            paths.setdefault('TCI', path)
                        elif '_scl_20m' in file_lower or file_lower == 'scl.tif':
                            paths.setdefault('SCL', path)

        still_missing = [r for r in ['B02', 'B03', 'B04', 'B08', 'SCL'] if r not in paths]
        if still_missing:
            raise FileNotFoundError(f"Missing required Sentinel-2 bands inside {self.product_dir}: {still_missing}")
            
        print(f"Located bands: {[f'{k}: {v.name}' for k, v in paths.items()]}")
        return paths

    def get_metadata(self) -> dict:
        """
        Reads metadata from the Blue band (B02) as the coordinate reference system.
        Returns a flat dict with width, height, crs, and transform guaranteed.
        """
        src = self.get_dataset('B02')
        meta = src.meta.copy()
        meta['width'] = src.width
        meta['height'] = src.height
        meta['crs'] = str(src.crs) if src.crs else 'EPSG:32643'
        return dict(meta)

    def load_full_bands(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Loads Blue, Green, Red, NIR bands, stacks them into a 4-channel array (shape: [4, H, W]),
        loads the SCL band, upsamples it to 10m spatial resolution, creates a binary cloud mask,
        and returns (bands_4ch, cloud_mask, metadata).
        """
        print("Loading 10m bands (B02, B03, B04, B08)...")
        # Load 10m bands
        bands = []
        meta = self.get_dataset('B02').meta.copy()
            
        for b in ['B02', 'B03', 'B04', 'B08']:
            src = self.get_dataset(b)
            arr = src.read(1) # shape [H, W]
            normalized = arr.astype(np.float32) / 10000.0
            bands.append(np.clip(normalized, 0.0, 1.0)) # Normalize and clip reflectance to [0, 1]
                    
        bands_4ch = np.stack(bands, axis=0) # shape [4, H, W]
        H, W = bands_4ch.shape[1], bands_4ch.shape[2]
        
        print("Loading 20m SCL band and upsampling to 10m...")
        scl_20m = self.get_dataset('SCL').read(1)

            
        # SCL is at 20m, bands are at 10m. We upsample by 2x using nearest-neighbor.
        scl_10m = np.repeat(np.repeat(scl_20m, 2, axis=0), 2, axis=1)
        
        # Crop or pad to match the 10m bands size exactly in case of minor rounding differences
        scl_10m = scl_10m[:H, :W]
        if scl_10m.shape[0] < H or scl_10m.shape[1] < W:
            scl_10m = np.pad(scl_10m, ((0, H - scl_10m.shape[0]), (0, W - scl_10m.shape[1])), mode='edge')

        # SCL cloud classes + brightness refinement for full tile:
        # 2 = dark area, 3 = cloud shadow, 8 = med cloud, 9 = high cloud, 10 = thin cirrus
        raw_rgb_mean = (bands_4ch[0] + bands_4ch[1] + bands_4ch[2]) / 3.0
        thick_core = (raw_rgb_mean > 0.28)
        scl_cloud = (scl_10m == 2) | (scl_10m == 3) | (scl_10m == 8) | (scl_10m == 9) | (scl_10m == 10)
        
        cloud_mask = (scl_cloud | thick_core).astype(np.float32)
        
        # Update metadata for 4 bands
        meta.update(count=4, dtype=rasterio.float32)
        return bands_4ch, cloud_mask, dict(meta)

    def load_crop(self, col_off: int, row_off: int, width: int, height: int, mask_source: str = "scl") -> tuple[np.ndarray, np.ndarray]:
        """
        Loads a crop from B02, B03, B04, B08 (shape: [4, height, width]).
        If mask_source == 'scl', loads SCL upsampled to 10m (shape: [height, width]).
        If mask_source == 'dn', threshold Blue/Red bands.
        Uses windowed reading for extremely fast crop extraction.
        """
        bands = []
        for b in ['B02', 'B03', 'B04', 'B08']:
            src = self.get_dataset(b)
            arr = src.read(1, window=rasterio.windows.Window(col_off, row_off, width, height))  # type: ignore
            normalized = arr.astype(np.float32) / 10000.0
            
            if normalized.shape[0] < height or normalized.shape[1] < width:
                normalized = np.pad(normalized, ((0, height - normalized.shape[0]), (0, width - normalized.shape[1])), mode='edge')
                
            bands.append(np.clip(normalized, 0.0, 1.0)) # Normalize and clip to [0, 1]
                
        bands_4ch = np.stack(bands, axis=0)
        
        import scipy.ndimage as ndimage
        
        # ── Helper: read SCL at 20m and upsample to 10m ──
        def _read_scl_10m():
            col_off_20m = col_off // 2
            row_off_20m = row_off // 2
            width_20m = max(1, width // 2)
            height_20m = max(1, height // 2)
            src_scl = self.get_dataset('SCL')
            scl_crop_20m = src_scl.read(1, window=rasterio.windows.Window(col_off_20m, row_off_20m, width_20m, height_20m))  # type: ignore
            scl_10m = np.repeat(np.repeat(scl_crop_20m, 2, axis=0), 2, axis=1)[:height, :width]
            if scl_10m.shape[0] < height or scl_10m.shape[1] < width:
                scl_10m = np.pad(scl_10m, ((0, height - scl_10m.shape[0]), (0, width - scl_10m.shape[1])), mode='edge')
            return scl_10m
        
        # ── Helper: convert reflectance band to display DN [0-255] ──
        def _band_to_dn(band):
            p2 = np.percentile(band, 2)
            p98 = np.percentile(band, 98)
            return (np.clip((band - p2) / (p98 - p2 + 1e-8), 0.0, 1.0) * 255.0).astype(np.uint8)
        
        if mask_source == "dn":
            # ══════════════════════════════════════════════════════════
            # HYBRID (SCL + DN 240-255) — Best for DIP+PINN downstream
            # ══════════════════════════════════════════════════════════
            scl_10m = _read_scl_10m()
            
            # 1. SCL base: dark area (2) + cloud shadow (3) + thick cloud (8,9) + thin cirrus (10)
            scl_mask = (scl_10m == 2) | (scl_10m == 3) | (scl_10m == 8) | (scl_10m == 9) | (scl_10m == 10)
            
            # 2. DN threshold on display-stretched bands
            b_dn = _band_to_dn(bands_4ch[0])  # B02 (Blue)
            g_dn = _band_to_dn(bands_4ch[1])  # B03 (Green)
            r_dn = _band_to_dn(bands_4ch[2])  # B04 (Red)
            
            # Tier 1: Bright cloud cores — ALL three bands >= 240
            dn_core = (r_dn >= 240) & (g_dn >= 240) & (b_dn >= 240)
            
            # Tier 2: Medium clouds — at least 2 of 3 bands >= 220
            #         AND average brightness >= 200
            band_passes_220 = (r_dn >= 220).astype(np.int8) + (g_dn >= 220).astype(np.int8) + (b_dn >= 220).astype(np.int8)
            avg_dn = (r_dn.astype(np.float32) + g_dn.astype(np.float32) + b_dn.astype(np.float32)) / 3.0
            dn_medium = (band_passes_220 >= 2) & (avg_dn >= 200)
            
            # Tier 3: Thin/hazy clouds — adaptive scene-relative detection
            #         Clouds are (a) brighter than terrain baseline, (b) spectrally neutral
            scene_median_dn = np.median(avg_dn)
            color_range_dn = (
                np.maximum.reduce([r_dn.astype(np.float32), g_dn.astype(np.float32), b_dn.astype(np.float32)]) -
                np.minimum.reduce([r_dn.astype(np.float32), g_dn.astype(np.float32), b_dn.astype(np.float32)])
            )
            # Cloud if: brightness well above scene median AND spectrally flat
            dn_haze = (avg_dn > scene_median_dn + 40) & (color_range_dn < 35)
            
            # Tier 4: Raw reflectance fallback for very bright pixels
            #         (catches clouds regardless of stretch)
            raw_brightness = (bands_4ch[0] + bands_4ch[1] + bands_4ch[2]) / 3.0
            raw_bright = raw_brightness > 0.35
            raw_range = (
                np.maximum.reduce([bands_4ch[0], bands_4ch[1], bands_4ch[2]]) -
                np.minimum.reduce([bands_4ch[0], bands_4ch[1], bands_4ch[2]])
            )
            raw_cloud = raw_bright & (raw_range < 0.08)
            
            # Tier 5: Shadow detection (Dark pixels)
            # Shadows have very low overall brightness and low NIR
            raw_nir = bands_4ch[3]
            dn_shadow = (avg_dn < np.maximum(20.0, scene_median_dn * 0.4)) & (raw_nir < 0.15)
            
            # 3. Union: SCL OR bright-core OR medium OR haze OR raw-bright OR shadow
            cloud_mask = scl_mask | dn_core | dn_medium | dn_haze | raw_cloud | dn_shadow
            
            # 4. Morphological cleanup — preserve individual cloud shapes
            #    Opening: remove isolated 1-2px salt noise
            cloud_mask = ndimage.binary_opening(
                cloud_mask,
                structure=np.ones((3, 3), dtype=bool),
                iterations=1,
            )
            #    Closing: fill small holes inside clouds
            cloud_mask = ndimage.binary_closing(
                cloud_mask,
                structure=np.ones((5, 5), dtype=bool),
                iterations=1,
            )
            #    Gentle dilation: expand mask to catch translucent cloud fringes
            cloud_mask = ndimage.binary_dilation(
                cloud_mask,
                iterations=4,
            ).astype(np.float32)
            
        else:
            # ══════════════════════════════════════════════════════════
            # ESA SCL + SPECTRAL BRIGHTNESS REFINEMENT
            # ══════════════════════════════════════════════════════════
            scl_10m = _read_scl_10m()
            
            # SCL classes: 2=dark area, 3=cloud shadow, 8=medium cloud, 9=high cloud, 10=thin cirrus
            scl_mask = (scl_10m == 2) | (scl_10m == 3) | (scl_10m == 8) | (scl_10m == 9) | (scl_10m == 10)
            
            # Additional safety: any pixel with high RGB reflectance (> 0.28) is a thick cloud core
            raw_rgb_mean = (bands_4ch[0] + bands_4ch[1] + bands_4ch[2]) / 3.0
            thick_core = (raw_rgb_mean > 0.28)
            
            combined_mask = scl_mask | thick_core
            
            # Morphological fill & dilation to guarantee thick cloud cores & translucent borders are covered
            combined_mask = ndimage.binary_closing(combined_mask, structure=np.ones((5, 5), dtype=bool), iterations=2)
            cloud_mask = ndimage.binary_dilation(combined_mask, iterations=4).astype(np.float32)
        
        return bands_4ch, cloud_mask

    def load_crop_downsampled(self, col_off: int, row_off: int, width: int, height: int, out_width: int, out_height: int, mask_source: str = "scl") -> tuple[np.ndarray, np.ndarray]:
        """
        Loads a crop from B02, B03, B04, B08 downsampled directly on read to (out_height, out_width)
        using bilinear resampling. Loads SCL downsampled/upsampled on read to (out_height, out_width)
        using nearest resampling, and constructs the cloud mask. If mask_source is 'dn', threshold Blue/Red.
        """
        bands = []
        for b in ['B02', 'B03', 'B04', 'B08']:
            src = self.get_dataset(b)
            arr = src.read(
                1,
                window=rasterio.windows.Window(col_off, row_off, width, height),  # type: ignore
                out_shape=(out_height, out_width),
                resampling=Resampling.bilinear
            )
            normalized = arr.astype(np.float32) / 10000.0
            bands.append(np.clip(normalized, 0.0, 1.0)) # Normalize and clip to [0, 1]
                
        bands_4ch = np.stack(bands, axis=0)
        
        import scipy.ndimage as ndimage
        
        # ── Helper: read SCL downsampled ──
        def _read_scl_ds():
            col_off_20m = col_off // 2
            row_off_20m = row_off // 2
            width_20m = max(1, width // 2)
            height_20m = max(1, height // 2)
            src_scl = self.get_dataset('SCL')
            return src_scl.read(
                1,
                window=rasterio.windows.Window(col_off_20m, row_off_20m, width_20m, height_20m),  # type: ignore
                out_shape=(out_height, out_width),
                resampling=Resampling.nearest
            )
        
        # ── Helper: convert reflectance band to display DN [0-255] ──
        def _band_to_dn(band):
            p2 = np.percentile(band, 2)
            p98 = np.percentile(band, 98)
            return (np.clip((band - p2) / (p98 - p2 + 1e-8), 0.0, 1.0) * 255.0).astype(np.uint8)
        
        if mask_source == "dn":
            # ══════════════════════════════════════════════════════════
            # HYBRID (SCL + DN 240-255) — Best for DIP+PINN downstream
            # ══════════════════════════════════════════════════════════
            scl_ds = _read_scl_ds()
            
            # 1. SCL base: dark area (2) + cloud shadow (3) + thick cloud (8,9) + thin cirrus (10)
            scl_mask = (scl_ds == 2) | (scl_ds == 3) | (scl_ds == 8) | (scl_ds == 9) | (scl_ds == 10)
            
            # 2. DN threshold on display-stretched bands
            b_dn = _band_to_dn(bands_4ch[0])  # B02 (Blue)
            g_dn = _band_to_dn(bands_4ch[1])  # B03 (Green)
            r_dn = _band_to_dn(bands_4ch[2])  # B04 (Red)
            
            # Tier 1: Bright cloud cores — ALL three bands >= 240
            dn_core = (r_dn >= 240) & (g_dn >= 240) & (b_dn >= 240)
            
            # Tier 2: Medium clouds — at least 2 of 3 bands >= 220
            band_passes_220 = (r_dn >= 220).astype(np.int8) + (g_dn >= 220).astype(np.int8) + (b_dn >= 220).astype(np.int8)
            avg_dn = (r_dn.astype(np.float32) + g_dn.astype(np.float32) + b_dn.astype(np.float32)) / 3.0
            dn_medium = (band_passes_220 >= 2) & (avg_dn >= 200)
            
            # Tier 3: Thin/hazy clouds — adaptive scene-relative detection
            scene_median_dn = np.median(avg_dn)
            color_range_dn = (
                np.maximum.reduce([r_dn.astype(np.float32), g_dn.astype(np.float32), b_dn.astype(np.float32)]) -
                np.minimum.reduce([r_dn.astype(np.float32), g_dn.astype(np.float32), b_dn.astype(np.float32)])
            )
            dn_haze = (avg_dn > scene_median_dn + 40) & (color_range_dn < 35)
            
            # Tier 4: Raw reflectance fallback
            raw_brightness = (bands_4ch[0] + bands_4ch[1] + bands_4ch[2]) / 3.0
            raw_bright = raw_brightness > 0.35
            raw_range = (
                np.maximum.reduce([bands_4ch[0], bands_4ch[1], bands_4ch[2]]) -
                np.minimum.reduce([bands_4ch[0], bands_4ch[1], bands_4ch[2]])
            )
            raw_cloud = raw_bright & (raw_range < 0.08)
            
            # Tier 5: Shadow detection (Dark pixels)
            raw_nir = bands_4ch[3]
            dn_shadow = (avg_dn < np.maximum(20.0, scene_median_dn * 0.4)) & (raw_nir < 0.15)
            
            # 3. Union
            cloud_mask = scl_mask | dn_core | dn_medium | dn_haze | raw_cloud | dn_shadow
            
            # 4. Morphological cleanup
            cloud_mask = ndimage.binary_opening(
                cloud_mask, structure=np.ones((3, 3), dtype=bool), iterations=1,
            )
            cloud_mask = ndimage.binary_closing(
                cloud_mask, structure=np.ones((5, 5), dtype=bool), iterations=1,
            )
            cloud_mask = ndimage.binary_dilation(
                cloud_mask, iterations=4,
            ).astype(np.float32)
            
        else:
            # ══════════════════════════════════════════════════════════
            # ESA SCL + SPECTRAL BRIGHTNESS REFINEMENT
            # ══════════════════════════════════════════════════════════
            scl_ds = _read_scl_ds()
            
            # SCL classes: 2=dark area, 3=cloud shadow, 8=medium cloud, 9=high cloud, 10=thin cirrus
            scl_mask = (scl_ds == 2) | (scl_ds == 3) | (scl_ds == 8) | (scl_ds == 9) | (scl_ds == 10)
            
            raw_rgb_mean = (bands_4ch[0] + bands_4ch[1] + bands_4ch[2]) / 3.0
            thick_core = (raw_rgb_mean > 0.28)
            
            combined_mask = scl_mask | thick_core
            combined_mask = ndimage.binary_closing(combined_mask, structure=np.ones((5, 5), dtype=bool), iterations=2)
            cloud_mask = ndimage.binary_dilation(combined_mask, iterations=4).astype(np.float32)
            
        return bands_4ch, cloud_mask


    def get_ui_overview(self, max_dim: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads highly downsampled overviews of TCI (RGB) and SCL cloud mask for fast frontend visualization.
        """
        # Ensure we have TCI path cached
        if 'TCI' not in self._dataset_cache:
            if 'TCI' in self.band_paths:
                self._dataset_cache['TCI'] = rasterio.open(self.band_paths['TCI'])
            else:
                # Fallback: create TCI overview on the fly from B04, B03, B02
                pass
                
        if 'TCI' in self._dataset_cache:
            src = self._dataset_cache['TCI']
            h, w = src.height, src.width
            factor = max(1, max(h, w) // max_dim)
            out_shape = (src.count, h // factor, w // factor)
            tci_rgb = src.read(
                out_shape=out_shape,
                resampling=Resampling.bilinear
            )
            tci_rgb = np.transpose(tci_rgb, (1, 2, 0))
        else:
            # If TCI JP2 is not present, construct from Red, Green, Blue
            src_b04 = self.get_dataset('B04')
            h, w = src_b04.height, src_b04.width
            factor = max(1, max(h, w) // max_dim)
            h_out, w_out = h // factor, w // factor
            
            b04 = src_b04.read(1, out_shape=(h_out, w_out), resampling=Resampling.bilinear)
            b03 = self.get_dataset('B03').read(1, out_shape=(h_out, w_out), resampling=Resampling.bilinear)
            b02 = self.get_dataset('B02').read(1, out_shape=(h_out, w_out), resampling=Resampling.bilinear)
            
            # Simple scaling to 0-255
            rgb = np.stack([b04, b03, b02], axis=-1).astype(np.float32) / 10000.0
            tci_rgb = (np.clip(rgb * 2.5, 0.0, 1.0) * 255.0).astype(np.uint8)
            out_shape = (3, h_out, w_out)
            
        src_scl = self.get_dataset('SCL')
        h_out, w_out = out_shape[1], out_shape[2]
        scl_arr = src_scl.read(
            out_shape=(1, h_out, w_out),
            resampling=Resampling.nearest
        )[0]
            
        mask_arr = np.zeros_like(scl_arr, dtype=np.uint8)
        # SCL: 3=shadow, 8=med cloud, 9=high cloud, 10=thin cirrus
        mask_arr[(scl_arr == 3) | (scl_arr == 8) | (scl_arr == 9) | (scl_arr == 10)] = 255
        
        return tci_rgb, mask_arr


    def generate_patches(self, height: int, width: int, patch_size: int = 256, overlap: int = 32) -> list[dict]:
        """
        Generates patch coordinates (y, x, h, w) across the image dimension.
        """
        patches = []
        stride = patch_size - overlap
        
        y = 0
        while y < height:
            h_curr = min(patch_size, height - y)
            y_start = y if (y + patch_size <= height) else (height - patch_size)
            if y_start < 0: y_start = 0
            
            x = 0
            while x < width:
                w_curr = min(patch_size, width - x)
                x_start = x if (x + patch_size <= width) else (width - patch_size)
                if x_start < 0: x_start = 0
                
                patches.append({
                    'y': y_start,
                    'x': x_start,
                    'h': patch_size if (y_start + patch_size <= height) else h_curr,
                    'w': patch_size if (x_start + patch_size <= width) else w_curr
                })
                
                if x + patch_size >= width:
                    break
                x += stride
                
            if y + patch_size >= height:
                break
            y += stride
            
        return patches

    def classify_patches(self, cloud_mask: np.ndarray, patch_coords: list[dict], threshold: float = 0.0) -> tuple[list[dict], list[dict]]:
        """
        Splits patch list into cloudy and clear patches.
        A patch is cloudy if fraction of cloud mask pixels > threshold.
        """
        cloudy_patches = []
        clear_patches = []
        
        for patch in patch_coords:
            y, x, h, w = patch['y'], patch['x'], patch['h'], patch['w']
            mask_crop = cloud_mask[y:y+h, x:x+w]
            cloud_fraction = np.mean(mask_crop > 0.5)
            
            p_info = patch.copy()
            p_info['cloud_fraction'] = float(cloud_fraction)
            
            if cloud_fraction > threshold:
                cloudy_patches.append(p_info)
            else:
                clear_patches.append(p_info)
                
        return cloudy_patches, clear_patches

if __name__ == '__main__':
    # Simple test run if path exists
    product_dir = "."
    try:
        loader = Sentinel2Loader(product_dir)
        meta = loader.get_metadata()
        print("Meta:", meta)
        tci, mask = loader.get_ui_overview(max_dim=512)
        print("UI Overview TCI shape:", tci.shape, "Mask shape:", mask.shape)
        # Generate patches
        patches = loader.generate_patches(10980, 10980, patch_size=256, overlap=32)
        print(f"Generated {len(patches)} patches in total.")
    except Exception as e:
        print("Loader test failed:", e)
