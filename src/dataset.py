"""
Data Pipeline
Handles loading, splitting, augmenting, and batching the PlantVillage dataset.
Includes weighted sampling to handle class imbalance (up to 36x in this dataset).
"""

from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
)


# Transforms

def get_train_transforms():
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """Validation/test transforms — no augmentation."""
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),   # 256 for 224 crop
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Dataset Splitting

def create_splits(dataset):
    """
    Split dataset indices into train/val/test while stratifying by class.
    Returns three lists of indices.
    """
    targets = [dataset.targets[i] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    # First split: train vs (val + test)
    val_test_ratio = 1.0 - TRAIN_RATIO
    train_idx, valtest_idx = train_test_split(
        indices, test_size=val_test_ratio,
        stratify=targets, random_state=RANDOM_SEED
    )

    # Second split: val vs test
    valtest_targets = [targets[i] for i in valtest_idx]
    relative_test = VAL_RATIO / (VAL_RATIO + (1.0 - TRAIN_RATIO - VAL_RATIO))
    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=relative_test,
        stratify=valtest_targets, random_state=RANDOM_SEED
    )

    return train_idx, val_idx, test_idx


# Weighted Sampler (fixes class imbalance)

def make_weighted_sampler(dataset, train_idx):
    """
    Build a WeightedRandomSampler so each class is sampled equally
    during training, regardless of how many images it has.

    Without this, the model would see Soybean (5090 imgs) 33x more
    than Potato healthy (152 imgs) per epoch — badly skewing learning.
    """
    # Count samples per class in the training split
    train_targets = [dataset.targets[i] for i in train_idx]
    class_counts = Counter(train_targets)
    num_classes = len(class_counts)

    # Weight for each class = 1 / count  (rare classes get higher weight)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # Assign weight to each sample
    sample_weights = [class_weights[dataset.targets[i]] for i in train_idx]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # Sampler draws len(train_idx) samples per epoch with replacement
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True
    )

    print(f"   ⚖️  WeightedRandomSampler active — balancing {num_classes} classes")
    return sampler


# DataLoaders

def get_dataloaders(data_dir=None, batch_size=None, use_weighted_sampler=True):
    """
    Returns train, val, test DataLoaders and the class-to-index mapping.

    Args:
        data_dir: path to PlantVillage folder (auto-detected if None)
        batch_size: batch size (uses config default if None)
        use_weighted_sampler: balance classes during training (recommended)

    Usage:
        train_loader, val_loader, test_loader, class_to_idx = get_dataloaders()
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    batch_size = batch_size or BATCH_SIZE

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            f"Make sure your PlantVillage folder is at data/PlantVillage/"
        )


    full_dataset = datasets.ImageFolder(data_dir, transform=get_val_transforms())
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_to_idx)

    print(f"Dataset: {len(full_dataset)} images, {num_classes} classes")

    train_idx, val_idx, test_idx = create_splits(full_dataset)
    print(f"   Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Build separate datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(data_dir, transform=get_train_transforms())
    val_dataset   = datasets.ImageFolder(data_dir, transform=get_val_transforms())

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(val_dataset,   val_idx)
    test_subset  = Subset(val_dataset,   test_idx)

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    if use_weighted_sampler:
        sampler = make_weighted_sampler(full_dataset, train_idx)
        train_loader = DataLoader(
            train_subset, batch_size=batch_size,
            sampler=sampler,
            **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_subset, batch_size=batch_size,
            shuffle=True, **loader_kwargs
        )

    val_loader = DataLoader(
        val_subset,  batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader, class_to_idx


# Utility

def denormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def get_class_counts(data_dir=None):
    """Return dict of {class_name: image_count}."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    counts = {}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            n = len([f for f in class_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            counts[class_dir.name] = n
    return counts