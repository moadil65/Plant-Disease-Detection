"""
PlantVillage Dataset Download Script
This script downloads the PlantVillage dataset from Kaggle.

SETUP (one-time):
1. Go to https://www.kaggle.com/settings → API → Create New Token
2. This downloads a kaggle.json file
3. Place it at ~/.kaggle/kaggle.json
4. Run: chmod 600 ~/.kaggle/kaggle.json

Then run this script:
    python download_data.py
"""

import os
import subprocess
import sys
from pathlib import Path


def download_dataset():
    data_dir = Path("data")
    dataset_dir = data_dir / "PlantVillage"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"Dataset already exists at {dataset_dir}")
        count = sum(1 for _ in dataset_dir.rglob("*.jpg")) + sum(1 for _ in dataset_dir.rglob("*.JPG"))
        print(f"   Found {count} images")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Kaggle API credentials not found!")
        print("   1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print("   2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json")
        print("   3. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    print("Downloading PlantVillage dataset from Kaggle...")
    os.makedirs(data_dir, exist_ok=True)

    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "abdallahalidev/plantvillage-dataset",
        "-p", str(data_dir),
        "--unzip"
    ], check=True)

    possible_paths = [
        data_dir / "plantvillage dataset" / "color",
        data_dir / "plantvillage_dataset" / "color",
        data_dir / "color",
        data_dir / "PlantVillage",
    ]

    source = None
    for p in possible_paths:
        if p.exists():
            source = p
            break

    if source and source != dataset_dir:
        source.rename(dataset_dir)
        print(f"Dataset organized at {dataset_dir}")
    elif source is None:
        # List what we got so user can fix
        print("Download complete but folder structure unexpected.")
        print("   Contents of data/:")
        for item in sorted(data_dir.rglob("*"))[:20]:
            print(f"   {item}")
        print("\n   Please move the folder containing class subdirectories to data/PlantVillage/")
        return

    count = sum(1 for _ in dataset_dir.rglob("*.jpg")) + sum(1 for _ in dataset_dir.rglob("*.JPG"))
    print(f"Done! {count} images across {sum(1 for d in dataset_dir.iterdir() if d.is_dir())} classes")


if __name__ == "__main__":
    download_dataset()
