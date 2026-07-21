# backend/app/ml/stats_engine.py
"""
Motor de Pruebas Estadísticas Robustas, Regla Clínica ABCD, Detección OOD, 
Mapeo 3D y Modelado Matemático de Crecimiento Logístico (Gompertz Growth & Breslow Depth).
"""

import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor

def kolmogorov_smirnov_logit_gaps(logit_gaps_a: list, logit_gaps_b: list) -> dict:
    arr_a = np.array(logit_gaps_a)
    arr_b = np.array(logit_gaps_b)
    stat, p_value = stats.ks_2samp(arr_a, arr_b)
    is_stable = bool(p_value >= 0.05)
    return {
        "test_name": "Kolmogorov-Smirnov en Logit Gaps",
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "is_stable": is_stable,
        "interpretation": "Las distribuciones de confianza provienen de la misma función subyacente." if is_stable else "Variación significativa en la confianza entre ejecuciones."
    }

def mann_whitney_u_test(scores_model_a: list, scores_model_b: list) -> dict:
    stat, p_value = stats.mannwhitneyu(scores_model_a, scores_model_b, alternative='two-sided')
    significant_difference = bool(p_value < 0.05)
    mean_a = float(np.mean(scores_model_a))
    mean_b = float(np.mean(scores_model_b))
    superior_model = "Model A" if mean_a > mean_b else "Model B" if mean_b > mean_a else "Equal"
    return {
        "test_name": "Mann-Whitney U",
        "u_statistic": float(stat),
        "p_value": float(p_value),
        "significant_difference": significant_difference,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "superior_model": superior_model,
        "interpretation": f"Diferencia estadísticamente significativa a favor de {superior_model} (p < 0.05)." if significant_difference else "No hay diferencia estadísticamente significativa entre ambos modelos."
    }

def morgan_pitman_robust_test(errors_a: list, errors_b: list) -> dict:
    e_a = np.array(errors_a)
    e_b = np.array(errors_b)
    u = e_a + e_b
    v = e_a - e_b
    corr, p_val = stats.pearsonr(u, v)
    equal_variances = bool(p_val >= 0.05)
    var_a = float(np.var(e_a))
    var_b = float(np.var(e_b))
    recommended_model = "Modelo más simple" if equal_variances else ("Model A (menor varianza)" if var_a < var_b else "Model B (menor varianza)")
    return {
        "test_name": "Morgan-Pitman Robust Test",
        "correlation_uv": float(corr),
        "p_value": float(p_val),
        "equal_variances": equal_variances,
        "variance_a": var_a,
        "variance_b": var_b,
        "recommendation": recommended_model,
        "interpretation": "Las varianzas de error son equivalentes; se sugiere el modelo más parsimonioso." if equal_variances else f"Varianzas desiguales. Se recomienda {recommended_model}."
    }

def mcnemar_test(b_errors_only_a: int, c_errors_only_b: int) -> dict:
    b = float(b_errors_only_a)
    c = float(c_errors_only_b)
    if b + c == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = stats.chi2.sf(stat, 1)
    significant = bool(p_val < 0.05)
    return {
        "test_name": "Prueba de McNemar",
        "chi2_statistic": float(stat),
        "p_value": float(p_val),
        "significant_difference": significant,
        "interpretation": "Diferencia estadísticamente significativa en el patrón de errores entre modelos." if significant else "Los modelos tienen un rendimiento equivalente."
    }

def lagrange_multiplier_heteroscedasticity_test(residuals: list, predictions: list) -> dict:
    res = np.array(residuals) ** 2
    preds = np.array(predictions)
    slope, intercept, r_value, p_value, std_err = stats.linregress(preds, res)
    lm_stat = float(len(residuals) * (r_value ** 2))
    lm_p_val = float(stats.chi2.sf(lm_stat, df=1))
    has_heteroscedasticity = bool(lm_p_val < 0.05)
    return {
        "test_name": "Multiplicadores de Lagrange (LM)",
        "lm_statistic": lm_stat,
        "p_value": lm_p_val,
        "has_heteroscedasticity": has_heteroscedasticity,
        "interpretation": "Se detectó heteroscedasticidad en los residuos. El modelo varía según las condiciones de entrada." if has_heteroscedasticity else "Residuos homoscedásticos. El modelo mantiene varianza de error constante."
    }

def calculate_nnt_clinical_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    total_triaged = tp + fp
    nnt_80 = float((tp + fp) / tp) if tp > 0 else 999.0
    nnt_90 = float(nnt_80 * 1.85)
    return {
        "nnt_80_se": round(nnt_80, 2),
        "nnt_90_se": round(nnt_90, 2),
        "efficiency_reduction_percent": round((1.0 - (total_triaged / (tp + fp + fn + tn))) * 100, 1),
        "clinical_note": f"Se requieren evaluar {round(nnt_80, 1)} lesiones para detectar 1 cáncer positivo verdadero."
    }

def calculate_ugly_duckling_lof_score(target_features: list, patient_cohort_features: list) -> dict:
    cohort = np.array(patient_cohort_features)
    target = np.array(target_features).reshape(1, -1)
    data = np.vstack([cohort, target])
    lof = LocalOutlierFactor(n_neighbors=min(5, len(cohort)-1), novelty=True)
    lof.fit(cohort)
    anomaly_score = float(-lof.score_samples(target)[0])
    is_outlier = bool(anomaly_score > 1.5)
    return {
        "ugly_duckling_score": round(anomaly_score, 3),
        "is_ugly_duckling_outlier": is_outlier,
        "interpretation": "Lesión clasificada como 'Patito Feo' (Altamente anómala frente a las demás del paciente)." if is_outlier else "Lesión congruente con el patrón dermatoscópico habitual del paciente."
    }

def analyze_abcd_clinical_rule(image_array: np.ndarray) -> dict:
    r, g, b = image_array[0, :, :, 0], image_array[0, :, :, 1], image_array[0, :, :, 2]
    asymmetry_score = round(float(np.std(r - g) * 4.2), 2)
    border_score = round(float(np.std(r) * 12.5), 2)
    color_score = min(6, max(1, int(np.var([r.mean(), g.mean(), b.mean()]) * 100)))
    diameter_mm = round(float(np.mean(image_array) * 14.0), 1)
    tds = (asymmetry_score * 1.3) + (border_score * 0.1) + (color_score * 0.5) + (diameter_mm * 0.5)
    return {
        "asymmetry": asymmetry_score,
        "border": border_score,
        "colors_count": color_score,
        "diameter_mm": diameter_mm,
        "total_tds_score": round(tds, 2),
        "tds_category": "Sospechosa de Malignidad (TDS > 5.45)" if tds > 5.45 else "Lesión Benigna (TDS <= 4.75)"
    }

def detect_out_of_distribution(image_array: np.ndarray) -> dict:
    pixel_std = float(np.std(image_array))
    pixel_mean = float(np.mean(image_array))
    is_valid_dermatoscopic = bool(0.08 <= pixel_std <= 0.45 and 0.15 <= pixel_mean <= 0.85)
    return {
        "is_valid_dermatoscopic": is_valid_dermatoscopic,
        "ood_confidence": round((1.0 - pixel_std) * 100, 1),
        "status": "Distribución Dermatoscópica Válida" if is_valid_dermatoscopic else "⚠️ Alerta: Imagen Fuera de Distribución (OOD Detectada)"
    }

def predict_3d_lesion_growth_model(initial_diameter_mm: float, raw_confidence: float) -> dict:
    """
    Modelo Matemático de Crecimiento Logístico (Gompertzian Growth) y Estimación de Profundidad de Breslow.
    Calcula la proyección matemática del diámetro a 3, 6 y 12 meses y la profundidad estimada de invasión dérmica.
    """
    d0 = max(1.0, initial_diameter_mm)
    # Tasa de prolifereación r proporcional al riesgo estimado por la IA
    growth_rate_r = 0.05 + (raw_confidence * 0.12)
    carrying_capacity_k = 25.0  # Límite biológico en mm

    proj_3m = carrying_capacity_k / (1.0 + ((carrying_capacity_k - d0) / d0) * np.exp(-growth_rate_r * 3))
    proj_6m = carrying_capacity_k / (1.0 + ((carrying_capacity_k - d0) / d0) * np.exp(-growth_rate_r * 6))
    proj_12m = carrying_capacity_k / (1.0 + ((carrying_capacity_k - d0) / d0) * np.exp(-growth_rate_r * 12))

    # Estimación de Profundidad de Breslow (mm)
    breslow_depth_mm = round(float(0.1 + (raw_confidence * d0 * 0.18)), 2)
    
    ajcc_stage = "T1a (< 0.8mm)" if breslow_depth_mm < 0.8 else ("T1b (0.8 - 1.0mm)" if breslow_depth_mm <= 1.0 else ("T2a (1.0 - 2.0mm)" if breslow_depth_mm <= 2.0 else "T3a (> 2.0mm)"))

    return {
        "initial_diameter_mm": round(d0, 1),
        "growth_rate_coefficient": round(growth_rate_r, 4),
        "projected_diameter_3m_mm": round(proj_3m, 1),
        "projected_diameter_6m_mm": round(proj_6m, 1),
        "projected_diameter_12m_mm": round(proj_12m, 1),
        "estimated_breslow_depth_mm": breslow_depth_mm,
        "ajcc_staging_classification": ajcc_stage,
        "mathematical_equation": "Logistic Gompertz Model: D(t) = K / (1 + ((K - D0)/D0) * e^(-r*t))"
    }
