# backend/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "generated_reports")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

SECRET_KEY = "dermai_secret_key_change_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# 3 Clásicos + 2 Híbridos
MODEL_TYPES = {
    "classic": ["ResNet50", "EfficientNetB0", "MobileNetV2"],
    "hybrid": ["DenseNet121_Attention", "EfficientNet_SpatialFusion"]
}
