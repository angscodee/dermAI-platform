"""
model_utils.py
Carga de modelos y funciones de predicción.
"""

import os
import time
import numpy as np
from app_logger import get_logger

logger = get_logger("model_utils")

# ---------------------------------------------------------------------------
# Rutas de modelos
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "app", "models")

_MODEL_FILES = {
    "EfficientNetB0": "efficientnetb0_isic2019.keras",
    "ResNet50":        "resnet50_isic2019.keras",
    "MobileNetV2":     "mobilenetv2_isic2019.keras",
    "DenseNet121":     "densenet121_isic2019.keras",
    "InceptionV3":     "inceptionv3_isic2019.keras",
}

# Umbral de decisión por defecto
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
def load_models() -> dict:
    """
    Intenta cargar los modelos .keras / .h5 desde MODELS_DIR.
    Devuelve un dict {nombre: modelo}.  Si no existe ningún archivo retorna {}.
    """
    try:
        from tensorflow import keras  # importación diferida para no romper tests
    except ImportError:
        return {}

    loaded = {}
    for name, filename in _MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        # También intentar con extensión .h5
        if not os.path.exists(path):
            path_h5 = path.replace(".keras", ".h5")
            if os.path.exists(path_h5):
                path = path_h5
            else:
                logger.debug("Modelo no encontrado en disco: %s", name)
                continue
        try:
            model = keras.models.load_model(path)
            logger.info("Modelo cargado: %s desde %s", name, path)
            # Forzar build del modelo para que Grad-CAM funcione (fix #3)
            _warm_up_model(model)
            loaded[name] = model
        except Exception as e:
            logger.error("Error cargando modelo %s: %s", name, e, exc_info=True)

    logger.info("Total modelos cargados: %d", len(loaded))
    return loaded


def _warm_up_model(model):
    """
    Realiza una pasada dummy para que el grafo del modelo quede construido.
    Esto evita el error "layer sequential has never been called" en Grad-CAM.
    """
    try:
        input_shape = model.input_shape  # (None, H, W, C)
        dummy = np.zeros((1, *input_shape[1:]), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception:
        pass  # Si falla el warm-up no bloqueamos la carga


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predict_image(model, processed_image: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
    """
    Predicción estándar.

    Returns:
        (diagnosis: str, confidence_percent: float, raw_confidence: float)
    """
    raw = float(model.predict(processed_image, verbose=0)[0][0])
    diagnosis = "Maligno" if raw >= threshold else "Benigno"
    confidence_percent = raw * 100 if raw >= threshold else (1 - raw) * 100
    logger.info("Predicción: raw=%.4f  diagnosis=%s  conf=%.1f%%  threshold=%.2f",
                raw, diagnosis, confidence_percent, threshold)
    return diagnosis, confidence_percent, raw


def predict_image_with_debug(model, processed_image: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
    """Igual que predict_image pero imprime información de debug."""
    result = predict_image(model, processed_image, threshold)
    logger.debug("DEBUG predict: raw=%.4f  diagnosis=%s  conf=%.1f%%", result[2], result[0], result[1])
    return result


def predict_image_with_custom_threshold(model, processed_image: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
    """Predicción usando un umbral de decisión personalizado."""
    return predict_image(model, processed_image, threshold)


def get_model_info(model) -> dict:
    """Retorna información básica de un modelo Keras."""
    try:
        total_params = model.count_params()
        input_shape = model.input_shape[1:]
    except Exception:
        total_params = 0
        input_shape = (224, 224, 3)
    return {
        "total_params": total_params,
        "input_shape": input_shape,
        "model_type": type(model).__name__,
    }
