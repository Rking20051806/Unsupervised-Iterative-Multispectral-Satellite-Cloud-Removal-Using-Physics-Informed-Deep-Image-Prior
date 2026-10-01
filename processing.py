import os
import glob
from pathlib import Path
import json
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import numpy as np
from minio import Minio
import psycopg2
from pystac import Item, Asset, Catalog, Collection, SpatialExtent, TemporalExtent, Extent
from datetime import datetime, timezone
import tempfile
import sys

# Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "sentinel-2"

DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "username"
DB_PASS = "password"
DB_NAME = "postgis"

def init_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"Created bucket {BUCKET_NAME}")
    return client

def create_stacked_cog(safe_dir: str, out_path: str):
    print(f"Creating stacked COG from {safe_dir}...")
    paths = {}
    for root, _, files in os.walk(safe_dir):
        for f in files:
            f_lower = f.lower()
            if f_lower.endswith('.jp2'):
                if '_b02_10m' in f_lower: paths['B02'] = os.path.join(root, f)
                elif '_b03_10m' in f_lower: paths['B03'] = os.path.join(root, f)
                elif '_b04_10m' in f_lower: paths['B04'] = os.path.join(root, f)
                elif '_b08_10m' in f_lower: paths['B08'] = os.path.join(root, f)
                elif '_scl_20m' in f_lower: paths['SCL'] = os.path.join(root, f)
                
    required = ['B02', 'B03', 'B04', 'B08', 'SCL']
    missing = [r for r in required if r not in paths]
    if missing:
        raise FileNotFoundError(f"Missing bands {missing} in {safe_dir}")
        
    with rasterio.open(paths['B02']) as src:
        meta = src.meta.copy()
        
    meta.update({
        'driver': 'GTiff',
        'count': 5, # B02, B03, B04, B08, SCL
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'deflate',
        'interleave': 'pixel',
        'dtype': 'uint16'
    })
    
    with rasterio.open(out_path, 'w', **meta) as dst:
        # Write 10m bands directly
        print("  Writing B02 (1)")
        with rasterio.open(paths['B02']) as src:
            dst.write(src.read(1), 1)
        print("  Writing B03 (2)")
        with rasterio.open(paths['B03']) as src:
            dst.write(src.read(1), 2)
        print("  Writing B04 (3)")
        with rasterio.open(paths['B04']) as src:
            dst.write(src.read(1), 3)
        print("  Writing B08 (4)")
        with rasterio.open(paths['B08']) as src:
            dst.write(src.read(1), 4)
            
        # Write 20m SCL upsampled to 10m
        print("  Writing SCL upsampled (5)")
        with rasterio.open(paths['SCL']) as src:
            scl_data = src.read(
                1,
                out_shape=(meta['height'], meta['width']),
                resampling=Resampling.nearest
            )
            dst.write(scl_data, 5)
            
        # Overviews for fast rendering (6 levels of pyramids)
        print("  Building overviews...")
        overviews = [2, 4, 8, 16, 32, 64]
        dst.build_overviews(overviews, Resampling.nearest)
        dst.update_tags(ns='rio_overview', resampling='nearest')
        
    return meta

def ingest_to_pgstac(item_id, minio_url, bounds, dt, collection_id="sentinel-2-l2a"):
    print(f"Registering {item_id} in pgSTAC...")
    # Initialize pgSTAC collection if not exists
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    conn.autocommit = True
    cur = conn.cursor()
    
    # Check if collection exists
    cur.execute("SELECT id FROM collections WHERE id = %s;", (collection_id,))
    if not cur.fetchone():
        print(f"Creating collection {collection_id}...")
        collection = Collection(
            id=collection_id,
            description="Sentinel-2 L2A Stacked COGs",
            extent=Extent(SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]), TemporalExtent([[datetime(2015, 1, 1, tzinfo=timezone.utc), None]])),
            title="Sentinel-2 L2A"
        )
        cur.execute("INSERT INTO collections (content) VALUES (%s);", (json.dumps(collection.to_dict()),))

    # Create STAC Item
    item = Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[
                [bounds.left, bounds.bottom],
                [bounds.right, bounds.bottom],
                [bounds.right, bounds.top],
                [bounds.left, bounds.top],
                [bounds.left, bounds.bottom]
            ]]
        },
        bbox=[bounds.left, bounds.bottom, bounds.right, bounds.top],
        datetime=dt,
        properties={},
        collection=collection_id
    )
    
    # Add Asset pointing to MinIO
    item.add_asset("stacked_cog", Asset(
        href=minio_url,
        media_type="image/tiff; application=geotiff; profile=cloud-optimized",
        roles=["data"],
        title="Stacked COG (B02, B03, B04, B08, SCL)"
    ))
    
    # Insert Item to pgSTAC
    item_dict = item.to_dict()
    item_dict['collection'] = collection_id
    
    cur.execute("SELECT id FROM items WHERE id = %s;", (item_id,))
    if cur.fetchone():
        print(f"Item {item_id} already exists. Updating...")
        cur.execute("SELECT update_item(%s::jsonb);", (json.dumps(item_dict),))
    else:
        cur.execute("SELECT create_item(%s::jsonb);", (json.dumps(item_dict),))
            
    cur.close()
    conn.close()
    print("Registration successful.")

def main():
    datasets = [
        "China.SAFE",
        "Vidarbha_Nagpur_Maharashtra.SAFE",
        "Vidarbha_Yavatmal_Maharashtra.SAFE"
    ]
    
    minio_client = init_minio()
    
    for safe_dir in datasets:
        if not os.path.exists(safe_dir):
            print(f"Dataset {safe_dir} not found. Skipping.")
            continue
            
        item_id = safe_dir.split('.')[0].replace('_', '-').lower()
        object_name = f"{item_id}.tif"
        minio_url = f"s3://{BUCKET_NAME}/{object_name}"
        
        try:
            # Check if it's already uploaded
            minio_client.stat_object(BUCKET_NAME, object_name)
            print(f"Object {object_name} already exists in MinIO. Skipping COG generation.")
            # Read bounds from the remote COG
            env = rasterio.Env(
                AWS_ACCESS_KEY_ID=MINIO_ACCESS_KEY,
                AWS_SECRET_ACCESS_KEY=MINIO_SECRET_KEY,
                AWS_S3_ENDPOINT=MINIO_ENDPOINT,
                AWS_HTTPS='NO',
                AWS_VIRTUAL_HOSTING='FALSE'
            )
            with env:
                with rasterio.open(minio_url) as src:
                    bounds_latlon = rasterio.warp.transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                    bounds = rasterio.coords.BoundingBox(*bounds_latlon)
        except Exception:
            print(f"Object {object_name} not in MinIO or error accessing. Building COG...")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_cog = os.path.join(tmpdir, f"{item_id}.tif")
                meta = create_stacked_cog(safe_dir, tmp_cog)
                
                # Read bounds and CRS for STAC
                with rasterio.open(tmp_cog) as src:
                    bounds_latlon = rasterio.warp.transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                    bounds = rasterio.coords.BoundingBox(*bounds_latlon)
                    
                print(f"Uploading to MinIO {minio_url}...")
                minio_client.fput_object(BUCKET_NAME, object_name, tmp_cog)
                
        dt = datetime(2023, 1, 1, tzinfo=timezone.utc) # Generic dt for now
        ingest_to_pgstac(item_id, minio_url, bounds, dt)
            
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
