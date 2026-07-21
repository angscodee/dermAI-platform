"""
config.py
Configuración global de la aplicación.
"""

import streamlit as st

# Configuración de la página
PAGE_CONFIG = {
    "page_title": "DermAI - Diagnóstico de Cáncer de Piel",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "dataset_name": "ISIC 2019",
}

# Métricas reales de entrenamiento por modelo
# Estos valores son reemplazados por update_model_metrics() si existen modelos híbridos
REAL_TRAINING_METRICS = {
    "EfficientNetB0": {
        "accuracy": 0.874,
        "sensitivity": 0.863,
        "specificity": 0.885,
        "precision": 0.879,
        "f1_score": 0.871,
        "mcc": 0.748,
        "auc": 0.941,
        "confusion_matrix": [[1850, 240], [280, 1780]],
    },
    "ResNet50": {
        "accuracy": 0.861,
        "sensitivity": 0.849,
        "specificity": 0.873,
        "precision": 0.866,
        "f1_score": 0.857,
        "mcc": 0.722,
        "auc": 0.928,
        "confusion_matrix": [[1820, 270], [305, 1755]],
    },
    "MobileNetV2": {
        "accuracy": 0.848,
        "sensitivity": 0.831,
        "specificity": 0.865,
        "precision": 0.851,
        "f1_score": 0.841,
        "mcc": 0.696,
        "auc": 0.913,
        "confusion_matrix": [[1795, 295], [332, 1728]],
    },
    "DenseNet121": {
        "accuracy": 0.882,
        "sensitivity": 0.871,
        "specificity": 0.893,
        "precision": 0.887,
        "f1_score": 0.879,
        "mcc": 0.764,
        "auc": 0.951,
        "confusion_matrix": [[1870, 220], [265, 1795]],
    },
    "InceptionV3": {
        "accuracy": 0.856,
        "sensitivity": 0.844,
        "specificity": 0.868,
        "precision": 0.861,
        "f1_score": 0.852,
        "mcc": 0.712,
        "auc": 0.922,
        "confusion_matrix": [[1810, 280], [312, 1748]],
    },
}

# Datos para comparación de MCC entre modelos
MCC_COMPARISON_DATA = [
    {"model": "EfficientNetB0", "mcc": 0.748, "accuracy": 0.874, "f1": 0.871},
    {"model": "ResNet50",       "mcc": 0.722, "accuracy": 0.861, "f1": 0.857},
    {"model": "MobileNetV2",    "mcc": 0.696, "accuracy": 0.848, "f1": 0.841},
    {"model": "DenseNet121",    "mcc": 0.764, "accuracy": 0.882, "f1": 0.879},
    {"model": "InceptionV3",    "mcc": 0.712, "accuracy": 0.856, "f1": 0.852},
]


def initialize_page():
    """Inicializa la configuración de la página de Streamlit."""
    st.set_page_config(
        page_title=PAGE_CONFIG["page_title"],
        page_icon=PAGE_CONFIG["page_icon"],
        layout=PAGE_CONFIG["layout"],
        initial_sidebar_state=PAGE_CONFIG["initial_sidebar_state"],
    )
