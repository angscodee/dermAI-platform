"""
hybrid_data_integrator.py
Integración de métricas de modelos híbridos con las métricas base.

Fix #8: update_model_metrics() devuelve los datos en lugar de mutar
variables globales del módulo importador.
"""

import os
import json
from config import REAL_TRAINING_METRICS, MCC_COMPARISON_DATA

HYBRID_METRICS_FILE = os.path.join(
    os.path.dirname(__file__), "app", "models", "hybrid", "metrics.json"
)


from typing import Optional


def update_model_metrics() -> tuple:
    """
    Lee las métricas de modelos híbridos desde disco (si existen) y las
    fusiona con las métricas base.

    Returns:
        (updated_metrics: dict, updated_mcc: list)
        Los llamadores deben guardar estos valores en st.session_state,
        NO mutar las variables del módulo config.
    """
    updated_metrics = dict(REAL_TRAINING_METRICS)
    updated_mcc = list(MCC_COMPARISON_DATA)

    if not os.path.exists(HYBRID_METRICS_FILE):
        return updated_metrics, updated_mcc

    try:
        with open(HYBRID_METRICS_FILE, "r", encoding="utf-8") as f:
            hybrid_data = json.load(f)

        hybrid_metrics = hybrid_data.get("metrics", {})
        hybrid_mcc = hybrid_data.get("mcc_comparison", [])

        # Fusionar métricas
        for name, values in hybrid_metrics.items():
            updated_metrics[name] = values

        # Fusionar MCC (evitar duplicados por nombre)
        existing_names = {entry["model"] for entry in updated_mcc}
        for entry in hybrid_mcc:
            if entry["model"] not in existing_names:
                updated_mcc.append(entry)
                existing_names.add(entry["model"])

    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"[hybrid_data_integrator] No se pudieron cargar métricas híbridas: {e}")

    return updated_metrics, updated_mcc
