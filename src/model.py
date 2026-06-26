"""
Model Definitions
"""

import torch
import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES


def get_resnet50(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
   
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model


def get_mobilenetv2(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
    
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model


def unfreeze_layers(model, model_name: str, num_layers: int = 2):
    
    if model_name == "resnet50":
        # ResNet has layer1, layer2, layer3, layer4
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        for layer in layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = True

    elif model_name == "mobilenetv2":
        # MobileNetV2 has 19 inverted residual blocks in model.features
        feature_blocks = list(model.features.children())
        for block in feature_blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfroze {num_layers} layer(s) — "
          f"trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.1f}%)")

    return model


def get_model(model_name: str, **kwargs):
    """Factory function to get a model by name."""
    models_dict = {
        "resnet50": get_resnet50,
        "mobilenetv2": get_mobilenetv2,
    }
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")
    return models_dict[model_name](**kwargs)


def model_summary(model, model_name: str):
    """Print a summary of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen

    print(f"\n{'=' * 50}")
    print(f"Model: {model_name}")
    print(f"{'=' * 50}")
    print(f"   Total params:     {total:>12,}")
    print(f"   Trainable params: {trainable:>12,} ({100 * trainable / total:.1f}%)")
    print(f"   Frozen params:    {frozen:>12,} ({100 * frozen / total:.1f}%)")
    print(f"{'=' * 50}\n")
