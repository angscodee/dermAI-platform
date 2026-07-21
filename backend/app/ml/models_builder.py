# backend/app/ml/models_builder.py
"""
Constructor y Cargador de Redes Neuronales Entrenadas (.keras)
Soporta:
1. DenseNet121_Attention (Híbrido)
2. EfficientNet_SpatialFusion (Híbrido)
3. ResNet50 (Clásico)
4. EfficientNetB0 (Clásico)
5. MobileNetV2 (Clásico)
6. InceptionV3 (Clásico)
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SHAPE = (224, 224, 3)

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_models"))

MODEL_FILE_MAP = {
    "DenseNet121_Attention": "densenet121_isic2019.keras",
    "EfficientNet_SpatialFusion": "efficientnetb0_isic2019.keras",
    "ResNet50": "resnet50_isic2019.keras",
    "EfficientNetB0": "efficientnetb0_isic2019.keras",
    "MobileNetV2": "mobilenetv2_isic2019.keras",
    "InceptionV3": "inceptionv3_isic2019.keras"
}

def _try_load_saved_model(name: str):
    filename = MODEL_FILE_MAP.get(name)
    if filename:
        candidate_paths = [
            os.path.join(SAVED_MODELS_DIR, filename),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models_output", filename)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models_output", filename)),
            os.path.abspath(filename)
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    print(f"Cargando modelo real entrenado desde: {p}")
                    model = keras.models.load_model(p, compile=False)
                    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    return model
                except Exception as e:
                    print(f"Advertencia al cargar {p}: {e}")
    return None

def build_classic_model(arch_name: str) -> keras.Model:
    saved = _try_load_saved_model(arch_name)
    if saved:
        return saved

    inputs = keras.Input(shape=IMG_SHAPE, name="image_input")
    try:
        if arch_name == "ResNet50":
            base = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
        elif arch_name == "EfficientNetB0":
            base = keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
        elif arch_name == "MobileNetV2":
            base = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)
        elif arch_name == "InceptionV3":
            base = keras.applications.InceptionV3(weights="imagenet", include_top=False, input_tensor=inputs)
        else:
            base = keras.applications.ResNet50(weights=None, include_top=False, input_tensor=inputs)
    except Exception:
        base = keras.applications.ResNet50(weights=None, include_top=False, input_tensor=inputs)

    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="diagnosis_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"Classic_{arch_name}")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def attention_gate(x, g, inter_channels):
    theta_x = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(g)
    
    if theta_x.shape[1:3] != phi_g.shape[1:3]:
        phi_g = layers.Resizing(theta_x.shape[1], theta_x.shape[2])(phi_g)
        
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(f)
    rate = layers.multiply([x, psi_f])
    return rate

def build_hybrid_densenet_attention() -> keras.Model:
    saved = _try_load_saved_model("DenseNet121_Attention")
    if saved:
        return saved

    inputs = keras.Input(shape=IMG_SHAPE, name="image_input")
    try:
        base = keras.applications.DenseNet121(weights="imagenet", include_top=False, input_tensor=inputs)
    except Exception:
        base = keras.applications.DenseNet121(weights=None, include_top=False, input_tensor=inputs)
    base.trainable = False
    
    feat = base.output
    attn_feat = attention_gate(feat, feat, 128)
    
    x = layers.GlobalAveragePooling2D()(attn_feat)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="diagnosis_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_DenseNet_Attention")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def build_hybrid_efficientnet_spatial_fusion() -> keras.Model:
    saved = _try_load_saved_model("EfficientNet_SpatialFusion")
    if saved:
        return saved

    inputs = keras.Input(shape=IMG_SHAPE, name="image_input")
    try:
        base = keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    except Exception:
        base = keras.applications.EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)
    base.trainable = False
    
    feat = base.output
    conv3x3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(feat)
    conv5x5 = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(feat)
    
    fused = layers.concatenate([conv3x3, conv5x5], axis=-1)
    x = layers.GlobalAveragePooling2D()(fused)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="diagnosis_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Hybrid_EfficientNet_SpatialFusion")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def build_model_by_type(name: str) -> keras.Model:
    if name == "DenseNet121_Attention":
        return build_hybrid_densenet_attention()
    elif name == "EfficientNet_SpatialFusion":
        return build_hybrid_efficientnet_spatial_fusion()
    else:
        return build_classic_model(name)
