"""
Project Configuration
All hyperparameters and paths in one place.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Dataset
IMG_SIZE = 224 
NUM_CLASSES = 38 
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-5
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 5

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class name mapping (plant, condition)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def get_plant_and_condition(class_name: str) -> tuple[str, str]:
    """
    Split class folder name into (plant, condition).
    Handles both triple-underscore (Apple___scab) and
    double-underscore (Apple__scab) separators from different
    versions of the PlantVillage dataset.
    """

    if "___" in class_name:
        parts = class_name.split("___", 1)
    elif "__" in class_name:
        parts = class_name.split("__", 1)
    else:
        return class_name.replace("_", " "), "Unknown"

    plant = parts[0].replace("_", " ").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return plant, condition