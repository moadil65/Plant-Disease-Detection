"""
Exploratory Data Analysis (EDA)
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, OUTPUT_DIR, CLASS_NAMES, get_plant_and_condition, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", palette="husl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Helper

def get_real_data():
    """Load class counts and sample images from real dataset."""
    from src.dataset import get_class_counts
    counts = get_class_counts()
    if not counts:
        return None, None
    return counts, DATA_DIR


def get_demo_data():
    """Generate synthetic class counts for demo visualization."""

    base_counts = {
        name: int(np.random.normal(1400, 400))
        for name in CLASS_NAMES
    }

    for key in list(base_counts.keys())[:5]:
        base_counts[key] = int(base_counts[key] * 0.4)
    base_counts = {k: max(v, 150) for k, v in base_counts.items()}
    return base_counts, None


# Plot 1: Class Distribution

def plot_class_distribution(counts: dict):
    """Bar chart of images per class."""
    fig, ax = plt.subplots(figsize=(14, 10))

    names = list(counts.keys())
    values = list(counts.values())

    plants = [get_plant_and_condition(n)[0] for n in names]
    unique_plants = sorted(set(plants))
    color_map = {p: plt.cm.tab20(i / len(unique_plants))
                 for i, p in enumerate(unique_plants)}
    colors = [color_map[p] for p in plants]

    sorted_pairs = sorted(zip(names, values, colors), key=lambda x: x[1])
    names, values, colors = zip(*sorted_pairs)

    display_names = [n.replace("___", " — ").replace("_", " ") for n in names]

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(display_names, fontsize=7)
    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title("PlantVillage Dataset — Images per Class", fontsize=14, fontweight='bold')

    # Add count labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=6)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 01_class_distribution.png")


# Plot 2: Plant-level Summary

def plot_plant_summary(counts: dict):
    """Grouped bar chart: healthy vs diseased per plant."""
    plant_data = {}
    for class_name, count in counts.items():
        plant, condition = get_plant_and_condition(class_name)
        if plant not in plant_data:
            plant_data[plant] = {"healthy": 0, "diseased": 0, "total": 0}
        if "healthy" in condition.lower():
            plant_data[plant]["healthy"] += count
        else:
            plant_data[plant]["diseased"] += count
        plant_data[plant]["total"] += count

    plants = sorted(plant_data.keys())
    healthy = [plant_data[p]["healthy"] for p in plants]
    diseased = [plant_data[p]["diseased"] for p in plants]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plants))
    width = 0.35

    ax.bar(x - width / 2, healthy, width, label="Healthy", color="#2ecc71", edgecolor='white')
    ax.bar(x + width / 2, diseased, width, label="Diseased", color="#e74c3c", edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in plants], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Number of Images")
    ax.set_title("Healthy vs Diseased Images by Plant", fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_plant_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 02_plant_summary.png")


# Plot 3: Class Imbalance Analysis

def plot_imbalance(counts: dict):
    """Show the imbalance ratio and stats."""
    values = list(counts.values())
    total = sum(values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(values, bins=20, color="#3498db", edgecolor='white', alpha=0.8)
    axes[0].axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.0f}')
    axes[0].axvline(np.median(values), color='orange', linestyle='--', label=f'Median: {np.median(values):.0f}')
    axes[0].set_xlabel("Images per Class")
    axes[0].set_ylabel("Number of Classes")
    axes[0].set_title("Distribution of Class Sizes")
    axes[0].legend()

    stats_text = (
        f"Total Images: {total:,}\n"
        f"Number of Classes: {len(values)}\n"
        f"Largest Class: {max(values):,}\n"
        f"Smallest Class: {min(values):,}\n"
        f"Imbalance Ratio: {max(values) / max(min(values), 1):.1f}x\n"
        f"Mean: {np.mean(values):,.0f}\n"
        f"Std Dev: {np.std(values):,.0f}"
    )
    axes[1].text(0.5, 0.5, stats_text, transform=axes[1].transAxes,
                 fontsize=13, verticalalignment='center', horizontalalignment='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[1].set_title("Dataset Statistics")
    axes[1].axis('off')

    plt.suptitle("Class Imbalance Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_imbalance_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 03_imbalance_analysis.png")


# Plot 4: Sample Images Grid

def plot_sample_images(data_dir: Path):
    """Show 2 random sample images from each class in a grid."""
    if data_dir is None:
        print("Skipping sample images (demo mode — no real images)")
        return

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    n_classes = len(class_dirs)
    cols = 6
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i, class_dir in enumerate(class_dirs):
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))
        if images:
            img_path = np.random.choice(images)
            img = Image.open(img_path).resize((224, 224))
            axes[i].imshow(img)

        plant, condition = get_plant_and_condition(class_dir.name)
        title = f"{plant}\n{condition}"
        axes[i].set_title(title, fontsize=6, pad=2)
        axes[i].axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Sample Images from Each Class", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_sample_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 04_sample_images.png")


# Plot 5: Disease Distribution Pie

def plot_disease_pie(counts: dict):
    """Pie chart showing proportion of healthy vs each disease category."""
    healthy_total = sum(v for k, v in counts.items() if "healthy" in k.lower())
    diseased_total = sum(v for k, v in counts.items() if "healthy" not in k.lower())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall split
    axes[0].pie(
        [healthy_total, diseased_total],
        labels=["Healthy", "Diseased"],
        colors=["#2ecc71", "#e74c3c"],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12}
    )
    axes[0].set_title("Overall: Healthy vs Diseased", fontsize=13, fontweight='bold')

    # Disease breakdown (top 10 diseases)
    disease_counts = {
        get_plant_and_condition(k)[1]: v
        for k, v in counts.items()
        if "healthy" not in k.lower()
    }
    # Merge same diseases across plants
    merged = {}
    for k, v in counts.items():
        if "healthy" not in k.lower():
            _, cond = get_plant_and_condition(k)
            merged[cond] = merged.get(cond, 0) + v

    sorted_diseases = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:10]
    labels, sizes = zip(*sorted_diseases)

    axes[1].pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 8} if len(labels) > 8 else {'fontsize': 9}
    )
    axes[1].set_title("Top 10 Disease Categories", fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_disease_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved 05_disease_distribution.png")


# Main

def main():
    parser = argparse.ArgumentParser(description="PlantVillage EDA")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data (no dataset needed)")
    args = parser.parse_args()

    print("=" * 50)
    print("PlantVillage — Exploratory Data Analysis")
    print("=" * 50)

    if args.demo:
        print("📋 Running in DEMO mode (synthetic counts)\n")
        counts, data_dir = get_demo_data()
    else:
        result = get_real_data()
        if result[0] is None:
            print("Dataset not found. Run with --demo or download the dataset first.")
            sys.exit(1)
        counts, data_dir = result

    plot_class_distribution(counts)
    plot_plant_summary(counts)
    plot_imbalance(counts)
    plot_sample_images(data_dir)
    plot_disease_pie(counts)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print("   Open them to explore the dataset!")


if __name__ == "__main__":
    main()
