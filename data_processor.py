"""
data_processor.py
Procesamiento de datos, comparación de modelos y preparación de métricas.

Mejoras v2:
  - Type hints completos en todas las funciones públicas (Prioridad Media #6)
  - bootstrap_accuracy_ci() para intervalos de confianza (Prioridad Media #8)
  - Logging estructurado
"""

import time
import math
import random
import numpy as np
import pandas as pd
from typing import Optional

from config import REAL_TRAINING_METRICS
from app_logger import get_logger

logger = get_logger("data_processor")


# ---------------------------------------------------------------------------
# Comparación de modelos
# ---------------------------------------------------------------------------
def compare_all_models(models: dict, processed_image: np.ndarray) -> list[dict]:
    """
    Ejecuta la inferencia con todos los modelos disponibles sobre la misma
    imagen y retorna una lista de dicts con los resultados.

    Args:
        models: dict {nombre_modelo: modelo_keras}
        processed_image: array de shape (1, H, W, 3) normalizado [0,1]

    Returns:
        Lista de dicts con keys: Modelo, Diagnóstico, Confianza (%), Tiempo (ms), Raw Score
    """
    results: list[dict] = []
    logger.info("Comparando %d modelos", len(models))
    for name, model in models.items():
        try:
            t0 = time.perf_counter()
            raw = float(model.predict(processed_image, verbose=0)[0][0])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            diagnosis = "Maligno" if raw >= 0.5 else "Benigno"
            confidence = raw * 100 if raw >= 0.5 else (1 - raw) * 100
            results.append({
                "Modelo": name,
                "Diagnóstico": diagnosis,
                "Confianza (%)": round(confidence, 1),
                "Tiempo (ms)": round(elapsed_ms, 1),
                "Raw Score": round(raw, 4),
            })
            logger.debug("Modelo %s → %s (%.1f%%) en %.1f ms", name, diagnosis, confidence, elapsed_ms)
        except Exception as e:
            logger.error("Error en modelo %s durante comparación: %s", name, e, exc_info=True)
            results.append({
                "Modelo": name,
                "Diagnóstico": "Error",
                "Confianza (%)": 0.0,
                "Tiempo (ms)": 0.0,
                "Raw Score": 0.0,
            })
    return results


# ---------------------------------------------------------------------------
# Análisis de consistencia
# ---------------------------------------------------------------------------
def analyze_consistency(comparison_results: list[dict]) -> dict:
    """
    Analiza la consistencia de los diagnósticos entre modelos.

    Args:
        comparison_results: lista de dicts retornada por compare_all_models()

    Returns:
        Dict con keys: consistent, agreement_rate, majority_diagnosis,
                       malignant_votes, benign_votes
    """
    if not comparison_results:
        return {"consistent": False, "agreement_rate": 0.0, "majority_diagnosis": "N/A"}

    diagnoses = [r["Diagnóstico"] for r in comparison_results if r["Diagnóstico"] != "Error"]
    if not diagnoses:
        return {"consistent": False, "agreement_rate": 0.0, "majority_diagnosis": "N/A"}

    malignant_count = diagnoses.count("Maligno")
    benign_count = diagnoses.count("Benigno")
    total = len(diagnoses)

    majority = "Maligno" if malignant_count >= benign_count else "Benigno"
    agreement = max(malignant_count, benign_count) / total

    return {
        "consistent": agreement >= 0.8,
        "agreement_rate": round(agreement * 100, 1),
        "majority_diagnosis": majority,
        "malignant_votes": malignant_count,
        "benign_votes": benign_count,
    }


# ---------------------------------------------------------------------------
# Métricas del modelo seleccionado
# ---------------------------------------------------------------------------
def get_model_metrics(model_name: str,
                      training_metrics: Optional[dict] = None) -> tuple[dict, bool]:
    """
    Retorna las métricas del modelo dado.

    Si se pasa ``training_metrics`` (proveniente de st.session_state), se usa ese
    dict en lugar del global de config.

    Args:
        model_name: nombre del modelo, ej. 'EfficientNetB0'
        training_metrics: dict opcional con métricas actualizadas

    Returns:
        Tupla (metrics_data: dict, is_real_data: bool)
    """
    source = training_metrics if training_metrics is not None else REAL_TRAINING_METRICS
    if model_name in source:
        logger.debug("Métricas reales encontradas para %s", model_name)
        return source[model_name], True

    logger.warning("Modelo '%s' no encontrado en métricas; usando datos simulados", model_name)
    simulated: dict = {
        "accuracy":         0.85,
        "sensitivity":      0.83,
        "specificity":      0.87,
        "precision":        0.84,
        "f1_score":         0.835,
        "mcc":              0.70,
        "auc":              0.92,
        "confusion_matrix": [[1750, 340], [350, 1710]],
    }
    return simulated, False


# ---------------------------------------------------------------------------
# Bootstrap para intervalo de confianza de accuracy  (Prioridad Media #8)
# ---------------------------------------------------------------------------
def bootstrap_accuracy_ci(metrics_data: dict,
                           n_bootstrap: int = 1000,
                           confidence: float = 0.95,
                           seed: int = 42) -> tuple[float, float]:
    """
    Calcula un intervalo de confianza bootstrap para la accuracy del modelo.

    Método: re-muestrea la matriz de confusión asumiendo que los recuentos
    son sumas de Bernoulli independientes, y recalcula accuracy en cada iteración.

    Args:
        metrics_data: dict con clave 'confusion_matrix' [[TN,FP],[FN,TP]]
                      y clave 'accuracy' (float).
        n_bootstrap: número de iteraciones bootstrap (default 1000).
        confidence: nivel de confianza, ej. 0.95 para IC del 95%.
        seed: semilla para reproducibilidad.

    Returns:
        (lower: float, upper: float) — límites del IC en [0, 1].
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    cm = metrics_data.get("confusion_matrix")
    if cm is None:
        acc = metrics_data.get("accuracy", 0.85)
        return _normal_approx_ci(acc, n=4150, confidence=confidence)

    cm_arr = np.array(cm)
    tn, fp = int(cm_arr[0, 0]), int(cm_arr[0, 1])
    fn, tp = int(cm_arr[1, 0]), int(cm_arr[1, 1])
    n_total = tn + fp + fn + tp

    if n_total == 0:
        return (0.0, 0.0)

    # Construir array de resultados individuales: 1=correcto, 0=incorrecto
    correct = np.array([1] * (tp + tn) + [0] * (fp + fn))

    boot_accs: list[float] = []
    for _ in range(n_bootstrap):
        sample = np_rng.choice(correct, size=n_total, replace=True)
        boot_accs.append(float(sample.mean()))

    boot_accs.sort()
    alpha = 1 - confidence
    lower_idx = int(math.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    return (boot_accs[lower_idx], boot_accs[upper_idx])


def _normal_approx_ci(acc: float, n: int, confidence: float) -> tuple[float, float]:
    """IC de Wilson aproximado cuando no hay matriz de confusión."""
    # z para nivel de confianza (solo 95% hardcodeado para evitar dependencia de scipy)
    z = 1.96 if confidence >= 0.95 else 1.645
    margin = z * math.sqrt(acc * (1 - acc) / n)
    return (max(0.0, acc - margin), min(1.0, acc + margin))


# ---------------------------------------------------------------------------
# Utilidades de formato
# ---------------------------------------------------------------------------
def create_mcc_comparison_dataframe(mcc_data: list[dict]) -> pd.DataFrame:
    """
    Crea un DataFrame formateado para mostrar la comparación de MCC.

    Args:
        mcc_data: lista de dicts con keys 'model', 'mcc', 'accuracy', 'f1'

    Returns:
        pd.DataFrame con columnas renombradas y valores formateados.
    """
    if not mcc_data:
        return pd.DataFrame()

    df = pd.DataFrame(mcc_data)
    rename_map = {
        "model":    "Modelo",
        "mcc":      "MCC",
        "accuracy": "Accuracy",
        "f1":       "F1-Score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["MCC", "Accuracy", "F1-Score"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.3f}")

    return df


def format_metrics_for_display(metrics: dict) -> dict:
    """
    Retorna un dict con los valores de métricas formateados como strings.

    Args:
        metrics: dict con claves accuracy, sensitivity, specificity, etc.

    Returns:
        Dict con los mismos valores como strings porcentuales/decimales.
    """
    return {
        "accuracy":    f"{metrics.get('accuracy', 0)*100:.1f}%",
        "sensitivity": f"{metrics.get('sensitivity', 0)*100:.1f}%",
        "specificity": f"{metrics.get('specificity', 0)*100:.1f}%",
        "precision":   f"{metrics.get('precision', 0)*100:.1f}%",
        "f1_score":    f"{metrics.get('f1_score', 0)*100:.1f}%",
        "mcc":         f"{metrics.get('mcc', 0):.3f}",
        "auc":         f"{metrics.get('auc', 0):.3f}",
    }
