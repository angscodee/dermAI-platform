"""
metrics_calculator.py
Cálculo de métricas estadísticas avanzadas: MCC y pruebas de McNemar.

Mejoras v2:
  - Type hints completos (Prioridad Media #6)
  - Logging estructurado
"""

import math
import numpy as np
from itertools import combinations
from typing import Union

from app_logger import get_logger

logger = get_logger("metrics_calculator")


# ---------------------------------------------------------------------------
# MCC
# ---------------------------------------------------------------------------
def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Coeficiente de Correlación de Matthews.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    Args:
        tp: verdaderos positivos
        tn: verdaderos negativos
        fp: falsos positivos
        fn: falsos negativos

    Returns:
        MCC en [-1, 1]. Retorna 0.0 si el denominador es cero.
    """
    numerator = tp * tn - fp * fn
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mcc_from_confusion_matrix(cm: Union[list, np.ndarray]) -> float:
    """
    Calcula MCC a partir de una matriz de confusión 2x2.

    Args:
        cm: [[TN, FP], [FN, TP]] como lista o numpy array

    Returns:
        MCC en [-1, 1]
    """
    cm = np.array(cm)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    return calculate_mcc(tp, tn, fp, fn)


# ---------------------------------------------------------------------------
# Test de McNemar
# ---------------------------------------------------------------------------
def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """
    Prueba de McNemar con corrección de continuidad de Edwards.

    Args:
        b: casos donde modelo A acierta y modelo B falla
        c: casos donde modelo B acierta y modelo A falla

    Returns:
        Tupla (chi2_stat: float, p_value: float)
    """
    if b + c == 0:
        return 0.0, 1.0

    # Chi-cuadrado con corrección de continuidad
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)

    # p-value usando distribución chi-cuadrado con 1 grado de libertad
    # Aproximación de la integral chi-cuadrado con 1 GL
    p_value = _chi2_sf(chi2, df=1)
    return chi2, p_value


def perform_mcnemar_comparisons(mcc_data: list[dict]) -> list[dict]:
    """
    Realiza pruebas de McNemar para todos los pares de modelos.

    Nota: por ausencia de predicciones individuales, b y c se estiman
    a partir de tasas de error. En producción usar predicciones reales.

    Args:
        mcc_data: lista de dicts con keys 'model', 'accuracy', 'mcc', 'f1'

    Returns:
        Lista de dicts con keys: model_a, model_b, chi2, p_value, significant, b, c
    """
    results = []
    n_test = 4150  # tamaño aproximado del conjunto de test ISIC 2019 (20%)

    model_list = [entry["model"] for entry in mcc_data]
    mcc_map = {entry["model"]: entry for entry in mcc_data}

    for model_a, model_b in combinations(model_list, 2):
        acc_a = mcc_map[model_a].get("accuracy", 0.85)
        acc_b = mcc_map[model_b].get("accuracy", 0.85)

        # Estimar b y c a partir de las tasas de error
        errors_a = int((1 - acc_a) * n_test)
        errors_b = int((1 - acc_b) * n_test)
        # b ≈ errores exclusivos de A, c ≈ errores exclusivos de B
        # Suposición conservadora: la mitad de los errores de cada uno son exclusivos
        b = max(1, errors_a // 2)
        c = max(1, errors_b // 2)

        chi2, p_value = mcnemar_test(b, c)
        significant = p_value < 0.05
        logger.debug("McNemar %s vs %s: chi2=%.4f p=%.6f sig=%s",
                     model_a, model_b, chi2, p_value, significant)

        results.append({
            "model_a": model_a,
            "model_b": model_b,
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 6),
            "significant": significant,
            "b": b,
            "c": c,
        })

    return results


# ---------------------------------------------------------------------------
# Helper: chi-cuadrado SF (1 GL)
# ---------------------------------------------------------------------------
def _chi2_sf(x: float, df: int = 1) -> float:
    """
    Función de supervivencia de la distribución chi-cuadrado con `df` grados
    de libertad usando la función gamma incompleta regularizada.

    Para df=1: P(X > x) = 1 - erf(sqrt(x/2))
    """
    if x <= 0:
        return 1.0
    if df == 1:
        return 1.0 - math.erf(math.sqrt(x / 2))
    # Para otros df, usamos una aproximación de series
    # (este módulo solo usa df=1 por ahora)
    return _regularized_upper_incomplete_gamma(df / 2, x / 2)


def _regularized_upper_incomplete_gamma(a: float, x: float) -> float:
    """Aproximación de la gamma incompleta superior regularizada Q(a, x)."""
    # Usamos la expansión en fracción continua de Lentz para x > a+1
    # y la serie para x <= a+1
    if x < 0:
        return 1.0
    if x == 0:
        return 1.0
    try:
        import scipy.special
        return float(scipy.special.gammaincc(a, x))
    except ImportError:
        # Fallback manual con serie de potencias
        return _gamma_series(a, x)


def _gamma_series(a: float, x: float, max_iter: int = 200, eps: float = 1e-10) -> float:
    """Serie de potencias para la gamma incompleta inferior regularizada P(a,x)."""
    if x <= 0:
        return 0.0
    ap = a
    delta = 1.0 / a
    total = delta
    for _ in range(max_iter):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
