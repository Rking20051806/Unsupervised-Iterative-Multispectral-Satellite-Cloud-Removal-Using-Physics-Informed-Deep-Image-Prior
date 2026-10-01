# cloud_thickness_analysis.py
"""
Script to automatically evaluate cloud thickness in the images located in
`d:/new pr pinn dip/cloud Images`.
It classifies an image as **THICK** when the proportion of bright (cloud) pixels
exceeds a configurable threshold.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------
# Configuration – adjust these values for your dataset
# ---------------------------------------------------------------------
IMAGE_DIR = Path(r"d:/new pr pinn dip/cloud Images")  # folder with cloud images
BRIGHT_THRESHOLD = 200      # pixel intensity (0‑255) considered as cloud
THICK_CLOUD_PERCENT = 30.0   # % of bright pixels → classify as thick cloud


def compute_cloud_coverage(image_path: Path) -> float:
    """Return the percentage of bright pixels in *image_path*.

    The image is converted to grayscale; any pixel with intensity greater than
    ``BRIGHT_THRESHOLD`` is counted as cloud.
    """
    with Image.open(image_path) as img:
        gray = img.convert("L")
        arr = np.array(gray)
        bright = (arr > BRIGHT_THRESHOLD).sum()
        return bright / arr.size * 100.0


def main() -> None:
    if not IMAGE_DIR.is_dir():
        print(f"❌ Directory not found: {IMAGE_DIR}")
        return

    thick_images = []
    print("Analyzing cloud images for thickness...")
    for entry in sorted(IMAGE_DIR.iterdir()):
        if entry.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        pct = compute_cloud_coverage(entry)
        status = "THICK" if pct >= THICK_CLOUD_PERCENT else "thin"
        print(f"{entry.name}: {pct:.2f}% ({status})")
        if status == "THICK":
            thick_images.append(entry.name)

    print("\n✨ Images classified as THICK clouds:")
    for name in thick_images:
        print(name)
    total = len(list(IMAGE_DIR.glob("*")))
    print(f"\n📊 Total thick images: {len(thick_images)} / {total}")

if __name__ == "__main__":
    main()
