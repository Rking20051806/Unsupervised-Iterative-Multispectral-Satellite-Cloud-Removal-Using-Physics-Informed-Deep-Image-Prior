from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rice import RiceDataset


def main():
    dataset_root = PROJECT_ROOT / "RICE_DATASET"

    for split in ["RICE1", "RICE2"]:
        dataset = RiceDataset(dataset_root, split=split, image_size=256)
        print(f"{split}: {len(dataset)} samples")
        first_item = dataset[0]
        print(f"  first image: {first_item['image_path']}")


if __name__ == "__main__":
    main()
