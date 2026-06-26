"""
Training Engine
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MODEL_DIR, OUTPUT_DIR, NUM_EPOCHS,
    LEARNING_RATE, FINETUNE_LR, EARLY_STOP_PATIENCE, BATCH_SIZE
)
from src.dataset import get_dataloaders
from src.model import get_model, unfreeze_layers, model_summary


# Training Step

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return avg loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    return running_loss / total, 100 * correct / total


# Validation Step

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate, return avg loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100 * correct / total


# Training Loop

def train(model_name: str, num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE):
    """Full training pipeline with two phases: frozen head → fine-tune."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(" No GPU found — training on CPU will be slow.")
        print("      Consider using Google Colab (free GPU) for training.")

    # ── Load Data ───────────────────────────────────────
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        batch_size=batch_size, use_weighted_sampler=True
    )

    # Detect actual number of classes from dataset (27 or 38)
    num_classes = len(class_to_idx)
    print(f"   Classes detected: {num_classes}")

    # Save class mapping for inference
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(MODEL_DIR / "class_mapping.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f" Class mapping saved to models/class_mapping.json")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "phase": []}
    best_val_acc = 0.0

    # Phase 1: Train classification head only
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training classification head ({model_name})")
    print(f"{'='*60}")

    # num_classes=num_classes makes it work for any dataset size
    model = get_model(model_name, num_classes=num_classes, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    model_summary(model, model_name)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    patience_counter = 0
    phase1_epochs = min(num_epochs // 2, 10)

    for epoch in range(1, phase1_epochs + 1):
        print(f"\nEpoch {epoch}/{phase1_epochs}  [Phase 1 — head only]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["phase"].append(1)

        print(f"  Train  →  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val    →  Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pth")
            print(f" New best saved  (val_acc = {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f" Early stopping triggered at epoch {epoch}")
                break

    # Phase 2: Fine-tune backbone
    print(f"🔧 PHASE 2: Fine-tuning backbone layers ({model_name})")

    # Reload best Phase 1 weights before unfreezing
    model.load_state_dict(
        torch.load(MODEL_DIR / f"{model_name}_best.pth",
                   map_location=device, weights_only=True)
    )
    model = unfreeze_layers(model, model_name, num_layers=2)

    # Differential LR: backbone gets 10x lower LR than the head
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "fc" not in n and "classifier" not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and ("fc" in n or "classifier" in n)]

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": FINETUNE_LR},
        {"params": head_params,     "lr": FINETUNE_LR * 10},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    patience_counter = 0
    phase2_epochs = num_epochs - phase1_epochs

    for epoch in range(1, phase2_epochs + 1):
        print(f"\nEpoch {epoch}/{phase2_epochs}  [Phase 2 — fine-tuning]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["phase"].append(2)

        print(f"  Train  →  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val    →  Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pth")
            print(f" New best saved  (val_acc = {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f" Early stopping triggered at epoch {epoch}")
                break

    # Save & Plot 
    with open(MODEL_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, model_name)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"   Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"   Model saved to    : models/{model_name}_best.pth")
    print(f"{'='*60}")

    return model, history


# Plot Training Curves

def plot_training_curves(history: dict, model_name: str):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    phases = history.get("phase", [])
    phase2_start = next((i + 1 for i, p in enumerate(phases) if p == 2), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, ylabel, title in [
        (ax1, history["train_loss"], history["val_loss"], "Loss",         f"{model_name} — Loss"),
        (ax2, history["train_acc"],  history["val_acc"],  "Accuracy (%)", f"{model_name} — Accuracy"),
    ]:
        ax.plot(epochs, train_vals, 'b-o', markersize=3, label="Train")
        ax.plot(epochs, val_vals,   'r-o', markersize=3, label="Val")
        if phase2_start:
            ax.axvline(x=phase2_start, color='gray', linestyle='--',
                       alpha=0.7, label="Fine-tune start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"06_training_curves_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → outputs/06_training_curves_{model_name}.png")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--model", type=str, default="mobilenetv2",
                        choices=["resnet50", "mobilenetv2"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch_size)