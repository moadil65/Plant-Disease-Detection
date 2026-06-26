"""
Plant Disease Detection — Streamlit App
Run:  streamlit run streamlit_app/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_plant_and_condition


# Disease Info
DISEASE_INFO = {
    "healthy": {
        "severity": "None", "color": "green",
        "description": "The leaf appears healthy with no visible signs of disease.",
        "recommendation": "Continue regular care — proper watering, fertilization, and monitoring."
    },
    "Apple_scab": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
        "recommendation": "Apply fungicide (captan or myclobutanil). Remove fallen leaves. Ensure good air circulation."
    },
    "Black_rot": {
        "severity": "High", "color": "red",
        "description": "Fungal infection causing dark, circular lesions that expand concentrically.",
        "recommendation": "Prune infected areas. Apply copper-based fungicide. Remove mummified fruit."
    },
    "Cedar_apple_rust": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing bright orange spots on leaves.",
        "recommendation": "Remove nearby cedar trees if possible. Apply fungicide in spring."
    },
    "Powdery_mildew": {
        "severity": "Moderate", "color": "orange",
        "description": "White powdery coating on leaf surfaces caused by fungal infection.",
        "recommendation": "Improve air circulation. Apply sulfur-based or potassium bicarbonate fungicide."
    },
    "Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "High", "color": "red",
        "description": "Fungal disease causing rectangular gray-brown lesions between leaf veins.",
        "recommendation": "Plant resistant hybrids. Apply foliar fungicide. Practice crop rotation."
    },
    "Common_rust_": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing raised, rust-colored pustules on leaves.",
        "recommendation": "Plant resistant varieties. Apply fungicide early in season."
    },
    "Northern_Leaf_Blight": {
        "severity": "High", "color": "red",
        "description": "Fungal disease causing long, cigar-shaped gray-green lesions.",
        "recommendation": "Use resistant varieties. Apply foliar fungicide. Rotate crops."
    },
    "Esca_(Black_Measles)": {
        "severity": "Very High", "color": "red",
        "description": "Complex fungal disease causing tiger-stripe patterns on leaves.",
        "recommendation": "No cure available. Prune infected vines. Apply wound protectants after pruning."
    },
    "Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing angular, reddish-brown spots on leaves.",
        "recommendation": "Remove infected leaves. Apply mancozeb fungicide. Ensure good drainage."
    },
    "Haunglongbing_(Citrus_greening)": {
        "severity": "Very High", "color": "red",
        "description": "Bacterial disease spread by psyllids, causing mottled yellowing and misshapen fruit.",
        "recommendation": "No cure. Control psyllid vectors. Remove infected trees to prevent spread."
    },
    "Bacterial_spot": {
        "severity": "Moderate-High", "color": "orange",
        "description": "Bacterial infection causing small, dark, water-soaked spots on leaves.",
        "recommendation": "Apply copper-based bactericide. Avoid overhead watering. Use disease-free seeds."
    },
    "Early_blight": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing concentric ring patterns on lower leaves.",
        "recommendation": "Apply chlorothalonil fungicide. Mulch around plants. Remove infected foliage."
    },
    "Late_blight": {
        "severity": "Very High", "color": "red",
        "description": "Devastating oomycete disease causing water-soaked lesions that rapidly destroy foliage.",
        "recommendation": "Apply fungicide immediately (metalaxyl). Destroy infected plants. Monitor weather."
    },
    "Leaf_Mold": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing yellow patches on upper leaf surface and olive-green mold below.",
        "recommendation": "Improve ventilation. Reduce humidity. Apply fungicide if severe."
    },
    "Septoria_leaf_spot": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing small, circular spots with dark borders and gray centers.",
        "recommendation": "Remove infected leaves. Apply chlorothalonil. Avoid overhead irrigation."
    },
    "Spider_mites Two-spotted_spider_mite": {
        "severity": "Moderate", "color": "orange",
        "description": "Tiny arachnids causing stippled, yellowed leaves with fine webbing.",
        "recommendation": "Spray with insecticidal soap or neem oil. Increase humidity. Use predatory mites."
    },
    "Target_Spot": {
        "severity": "Moderate", "color": "orange",
        "description": "Fungal disease causing small, dark brown spots with concentric rings.",
        "recommendation": "Apply fungicide. Remove debris. Maintain proper plant spacing."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Very High", "color": "red",
        "description": "Viral disease causing severe leaf curling, yellowing, and stunted growth.",
        "recommendation": "No cure. Control whitefly vectors. Remove infected plants. Use resistant varieties."
    },
    "Tomato_mosaic_virus": {
        "severity": "High", "color": "red",
        "description": "Viral disease causing mottled light/dark green patterns and leaf distortion.",
        "recommendation": "No cure. Remove infected plants. Disinfect tools. Use resistant varieties."
    },
    "Leaf_scorch": {
        "severity": "Moderate", "color": "orange",
        "description": "Causes leaf edges to brown and curl, often spreading inward.",
        "recommendation": "Improve watering consistency. Provide afternoon shade. Mulch to retain moisture."
    },
}


# Model Builder
def build_mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model

def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_f  = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


# Model Loading

@st.cache_resource
def load_model(model_name):
    model_path   = PROJECT_ROOT / "models" / f"{model_name}_best.pth"
    mapping_path = PROJECT_ROOT / "models" / "class_mapping.json"

    if not model_path.exists():
        return None, None, f"Model file not found: {model_path}\nMake sure you placed mobilenetv2_best.pth in the models/ folder."
    if not mapping_path.exists():
        return None, None, f"class_mapping.json not found in models/ folder."

    with open(mapping_path) as f:
        idx_to_class = json.load(f)

    num_classes = len(idx_to_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_mobilenetv2(num_classes) if model_name == "mobilenetv2" else build_resnet50(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    return model, idx_to_class, None


# Prediction
def predict(model, image, idx_to_class, device):
    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs  = F.softmax(output, dim=1)[0]

    top5_probs, top5_idx = probs.topk(5)
    results = []
    for prob, idx in zip(top5_probs, top5_idx):
        class_name = idx_to_class[str(idx.item())]
        plant, condition = get_plant_and_condition(class_name)
        results.append({
            "class_name":  class_name,
            "plant":       plant,
            "condition":   condition,
            "probability": prob.item(),
        })

    return results, input_tensor


# Grad-CAM
def generate_gradcam(model, input_tensor, model_name, pred_class_idx):
    model.eval()
    target_layer = model.features[-1] if model_name == "mobilenetv2" else model.layer4[-1]

    activations, gradients = [], []
    fwd = target_layer.register_forward_hook(lambda m,i,o: activations.append(o.detach()))
    bwd = target_layer.register_full_backward_hook(lambda m,gi,go: gradients.append(go[0].detach()))

    out = model(input_tensor)
    model.zero_grad()
    out[0, pred_class_idx].backward()
    fwd.remove(); bwd.remove()

    weights = gradients[0].mean(dim=(2,3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))


# UI
def main():
    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌿",
        layout="wide"
    )

    # Header
    st.title("Plant Disease Detection")
    st.markdown(
        "Upload a photo of a plant leaf for instant AI-powered disease diagnosis. "
        "Trained on **35,000+ images** across **38 disease classes** and **14 crop species**."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        model_name   = st.selectbox("Model", ["mobilenetv2"], help="mobilenetv2 = fast & accurate")
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
        st.markdown("---")
        st.markdown("### Model Stats")
        st.markdown("- **Test Accuracy:** 97.29%")
        st.markdown("- **Architecture:** MobileNetV2")
        st.markdown("- **Classes:** 38")
        st.markdown("- **Training:** Transfer learning")
        st.markdown("---")
        st.markdown("### Supported Crops")
        st.markdown(
            "Apple · Blueberry · Cherry · Corn · Grape · Orange · "
            "Peach · Pepper · Potato · Raspberry · Soybean · Squash · "
            "Strawberry · Tomato"
        )

    # Load model
    model, idx_to_class, error = load_model(model_name)
    if error:
        st.error(f" {error}")
        return

    num_classes = len(idx_to_class)
    device = next(model.parameters()).device

    # Layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Upload Leaf Image")
        uploaded = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit photo of a single plant leaf"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

    if not uploaded:
        with col2:
            st.subheader("Diagnosis")
            st.info("<- Upload a leaf image to get started!")
            st.markdown("**Tips for best results:**")
            st.markdown("- Use a clear, well-lit photo")
            st.markdown("- Make sure the leaf fills most of the frame")
            st.markdown("- Avoid blurry or heavily shadowed images")
        return

    # Run prediction
    with st.spinner("Analyzing leaf..."):
        results, input_tensor = predict(model, image, idx_to_class, device)

    top = results[0]

    # Get disease info — try exact match then fallback
    condition_key = top["class_name"].split("___")[1] if "___" in top["class_name"] else top["condition"]
    info = DISEASE_INFO.get(condition_key, DISEASE_INFO.get("healthy"))

    with col2:
        st.subheader("Diagnosis")

        # Severity badge color
        sev_colors = {"None": "🟢", "Moderate": "🟡", "Moderate-High": "🟠", "High": "🔴", "Very High": "🔴"}
        sev_icon   = sev_colors.get(info["severity"], "⚪")

        # Result card
        st.markdown(f"### {top['plant']} — {top['condition']}")

        m1, m2 = st.columns(2)
        m1.metric("Confidence",  f"{top['probability']:.1%}")
        m2.metric("Severity",    f"{sev_icon} {info['severity']}")

        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Recommendation:** {info['recommendation']}")

        st.markdown("---")
        st.markdown("#### Top 5 Predictions")

        fig, ax = plt.subplots(figsize=(7, 2.8))
        labels = [f"{r['plant']} — {r['condition']}" for r in results]
        probs  = [r['probability'] for r in results]
        colors = ['#2ecc71' if i == 0 else '#bdc3c7' for i in range(len(labels))]
        ax.barh(range(len(labels)), probs, color=colors, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Grad-CAM
    if show_gradcam:
        st.markdown("---")
        st.subheader("Grad-CAM — What the Model Sees")

        pred_idx = int([k for k, v in idx_to_class.items() if v == top["class_name"]][0])
        cam      = generate_gradcam(model, input_tensor, model_name, pred_idx)

        img_rsz  = image.resize((IMG_SIZE, IMG_SIZE))
        heatmap  = plt.cm.jet(cam)[:, :, :3]
        overlay  = (0.5 * np.array(img_rsz) / 255.0 + 0.5 * heatmap).clip(0, 1)

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.image(img_rsz, caption="Original", use_container_width=True)
        with gc2:
            fig2, ax2 = plt.subplots()
            ax2.imshow(cam, cmap='jet')
            ax2.axis('off')
            ax2.set_title("Attention Map", fontsize=9)
            st.pyplot(fig2)
            plt.close()
        with gc3:
            st.image((overlay * 255).astype(np.uint8), caption="Overlay", use_container_width=True)

        st.caption(
            "🔴 Red/yellow = regions the model focused on most for this prediction. "
            "Green/blue = low attention. Healthy leaves show diffuse attention; "
            "diseased leaves show tight focus on lesion areas."
        )


if __name__ == "__main__":
    main()