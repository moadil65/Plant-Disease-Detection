"""
Evaluation & Grad-CAM
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MODEL_DIR, OUTPUT_DIR, DATA_DIR, IMG_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, get_plant_and_condition
)
from src.dataset import get_dataloaders, denormalize
from src.model import get_model


# Full Test Evaluation

@torch.no_grad()
def evaluate_test_set(model, test_loader, device, class_names):
    """Run model on full test set and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}%\n")

    # Classification report
    display_names = [n.replace("___", " — ").replace("_", " ") for n in class_names]
    report = classification_report(all_labels, all_preds, target_names=display_names)
    print(report)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    print(f"✅ Saved classification_report.txt")

    return all_preds, all_labels, accuracy


# Confusion Matrix

def plot_confusion_matrix(all_labels, all_preds, class_names, model_name):
    """Plot and save the 38×38 confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    display_names = [n.replace("___", "\n").replace("_", " ")[:25] for n in class_names]

    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(
        cm_normalized, annot=False, fmt='.2f', cmap='Blues',
        xticklabels=display_names, yticklabels=display_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title(f'{model_name} — Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"07_confusion_matrix_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix")

    # Also plot top misclassifications
    plot_top_errors(cm, class_names, model_name)


def plot_top_errors(cm, class_names, model_name):
    """Find and plot the most confused class pairs."""
    np.fill_diagonal(cm, 0)
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i][j] > 0 and i != j:
                errors.append((class_names[i], class_names[j], cm[i][j]))

    errors.sort(key=lambda x: x[2], reverse=True)
    top_n = min(15, len(errors))

    if top_n == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{e[0].split('___')[1][:15]} → {e[1].split('___')[1][:15]}" for e in errors[:top_n]]
    values = [e[2] for e in errors[:top_n]]

    ax.barh(range(top_n), values, color='#e74c3c', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of Misclassifications")
    ax.set_title(f"{model_name} — Top {top_n} Confused Pairs", fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"08_top_errors_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top error analysis")


# Grad-CAM

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap."""
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class, output


def get_target_layer(model, model_name):
    if model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "mobilenetv2":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def visualize_gradcam(model, model_name, data_dir, class_names, device, n_samples=8):

    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Collect sample images
    all_images = []
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
            if imgs:
                all_images.append(np.random.choice(imgs))

    np.random.shuffle(all_images)
    samples = all_images[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3))
    if n_samples == 1:
        axes = [axes]

    for i, img_path in enumerate(samples):
        # Original image
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))

        # Model input
        input_tensor = transform(img).unsqueeze(0).to(device)
        cam, pred_class, output = grad_cam.generate(input_tensor)

        probs = torch.softmax(output, dim=1)
        confidence = probs[0, pred_class].item()

        # Resize CAM to image size
        import cv2
        cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.5 * np.array(img_resized) / 255.0 + 0.5 * heatmap

        # True label from folder name
        true_name = img_path.parent.name
        pred_name = class_names[pred_class]
        _, true_cond = get_plant_and_condition(true_name)
        _, pred_cond = get_plant_and_condition(pred_name)

        # Plot
        axes[i][0].imshow(img_resized)
        axes[i][0].set_title(f"True: {true_cond}", fontsize=8)
        axes[i][0].axis('off')

        axes[i][1].imshow(cam_resized, cmap='jet')
        axes[i][1].set_title("Grad-CAM", fontsize=8)
        axes[i][1].axis('off')

        axes[i][2].imshow(overlay)
        color = 'green' if true_name == pred_name else 'red'
        axes[i][2].set_title(f"Pred: {pred_cond} ({confidence:.0%})", fontsize=8, color=color)
        axes[i][2].axis('off')

    plt.suptitle(f"{model_name} — Grad-CAM Visualizations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"09_gradcam_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM visualizations")


#CLI

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "mobilenetv2"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_DIR / f"{args.model}_best.pth"

    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("   Run training first: python -m src.train --model", args.model)
        sys.exit(1)

    # Load model
    model = get_model(args.model, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)

    # Load class mapping
    with open(MODEL_DIR / "class_mapping.json") as f:
        idx_to_class = json.load(f)
    class_names = [idx_to_class[str(i)] for i in range(NUM_CLASSES)]

    # Get test loader
    _, _, test_loader, _ = get_dataloaders()

    # Evaluate
    all_preds, all_labels, accuracy = evaluate_test_set(
        model, test_loader, device, class_names
    )

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, args.model)

    # Grad-CAM
    if DATA_DIR.exists():
        visualize_gradcam(model, args.model, DATA_DIR, class_names, device)


if __name__ == "__main__":
    main()
