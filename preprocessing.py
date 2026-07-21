"""
preprocessing.py
Preprocesamiento de imágenes para los modelos de deep learning.
"""

import numpy as np
from PIL import Image, ImageOps

# Tamaño de entrada esperado por los modelos
TARGET_SIZE = (224, 224)


def preprocess_image(image_input, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """
    Preprocesa una imagen PIL o un array NumPy para la inferencia.

    Pasos:
      1. Convierte a PIL si es array.
      2. Convierte a RGB (elimina canal alpha si existe).
      3. Redimensiona a target_size.
      4. Normaliza píxeles a [0, 1].
      5. Añade dimensión de batch → shape (1, H, W, 3).

    Returns:
        np.ndarray de shape (1, target_size[0], target_size[1], 3).
    """
    # Aceptar PIL.Image o np.ndarray
    if isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input.astype(np.uint8))
    elif isinstance(image_input, Image.Image):
        pil_image = image_input
    else:
        raise TypeError(f"Tipo de imagen no soportado: {type(image_input)}")

    # Convertir a RGB
    pil_image = pil_image.convert("RGB")

    # Redimensionar
    pil_image = pil_image.resize(target_size, Image.LANCZOS)

    # Convertir a array y normalizar
    img_array = np.array(pil_image, dtype=np.float32) / 255.0

    # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
