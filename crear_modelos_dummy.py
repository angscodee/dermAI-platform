"""
crear_modelos_dummy.py
Crea modelos Keras mínimos para probar la UI sin necesitar los modelos reales.
Ejecutar UNA SOLA VEZ:  python crear_modelos_dummy.py
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silenciar logs de TF

import tensorflow as tf
from tensorflow import keras

MODELS_DIR = os.path.join(os.path.dirname(__file__), "app", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAMES = [
    "efficientnetb0_isic2019",
    "resnet50_isic2019",
    "mobilenetv2_isic2019",
    "densenet121_isic2019",
    "inceptionv3_isic2019",
]

def build_dummy_model():
    """Modelo mínimo: Conv2D → GlobalAvgPool → Dense(1, sigmoid)."""
    inp = keras.Input(shape=(224, 224, 3), name="input_image")
    x = keras.layers.Conv2D(16, 3, activation="relu", padding="same", name="conv1")(inp)
    x = keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="conv2")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Warm-up: una pasada dummy para que el grafo quede construido (fix Grad-CAM)
    model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
    return model

print(f"Creando modelos dummy en: {MODELS_DIR}\n")

for name in MODEL_NAMES:
    path = os.path.join(MODELS_DIR, f"{name}.keras")
    if os.path.exists(path):
        print(f"  ✓ Ya existe: {name}.keras — saltando")
        continue
    print(f"  ⏳ Creando {name}.keras …", end=" ", flush=True)
    model = build_dummy_model()
    model.save(path)
    print("✅")

print("\n¡Listo! Ahora puedes correr:  streamlit run app.py")
