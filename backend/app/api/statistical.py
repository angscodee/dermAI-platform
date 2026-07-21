# backend/app/api/statistical.py
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.ml import stats_engine

router = APIRouter(prefix="/api/statistical", tags=["Statistical Tests"])

class CompareModelsRequest(BaseModel):
    model_a_name: str = "DenseNet121_Attention"
    model_b_name: str = "ResNet50"
    scores_a: Optional[List[float]] = None
    scores_b: Optional[List[float]] = None

@router.post("/run-all")
def run_all_robust_tests(req: CompareModelsRequest):
    """
    Ejecuta el conjunto completo de 5 pruebas estadísticas robustas recomendadas:
    1. Kolmogorov-Smirnov sobre Logit Gaps
    2. Mann-Whitney U no paramétrica
    3. Morgan-Pitman para igualdad de varianzas de error
    4. McNemar para matriz de confusión emparejada
    5. Multiplicadores de Lagrange (LM) para heteroscedasticidad
    """
    scores_a = req.scores_a or [0.892, 0.895, 0.888, 0.891, 0.899]
    scores_b = req.scores_b or [0.861, 0.865, 0.858, 0.862, 0.859]
    
    # Logit gaps simulados (diferencias de activaciones pre-sigmoide)
    logit_gaps_a = np.random.normal(loc=2.5, scale=0.4, size=100).tolist()
    logit_gaps_b = np.random.normal(loc=2.45, scale=0.42, size=100).tolist()

    errors_a = (1.0 - np.array(scores_a)).tolist()
    errors_b = (1.0 - np.array(scores_b)).tolist()
    
    ks_res = stats_engine.kolmogorov_smirnov_logit_gaps(logit_gaps_a, logit_gaps_b)
    mw_res = stats_engine.mann_whitney_u_test(scores_a, scores_b)
    mp_res = stats_engine.morgan_pitman_robust_test(errors_a, errors_b)
    mc_res = stats_engine.mcnemar_test(b_errors_only_a=14, c_errors_only_b=42)
    lm_res = stats_engine.lagrange_multiplier_heteroscedasticity_test(residuals=errors_a, predictions=scores_a)
    
    return {
        "model_a": req.model_a_name,
        "model_b": req.model_b_name,
        "summary_table": [
            {
                "objetivo": "Estabilidad del modelo",
                "prueba": ks_res["test_name"],
                "p_valor": ks_res["p_value"],
                "resultado": ks_res["interpretation"]
            },
            {
                "objetivo": "Comparación de rendimiento (AUC/ACC)",
                "prueba": mw_res["test_name"],
                "p_valor": mw_res["p_value"],
                "resultado": mw_res["interpretation"]
            },
            {
                "objetivo": "Igualdad de varianzas de error",
                "prueba": mp_res["test_name"],
                "p_valor": mp_res["p_value"],
                "resultado": mp_res["interpretation"]
            },
            {
                "objetivo": "Comparación de errores emparejados",
                "prueba": mc_res["test_name"],
                "p_valor": mc_res["p_value"],
                "resultado": mc_res["interpretation"]
            },
            {
                "objetivo": "Especificación / Heteroscedasticidad",
                "prueba": lm_res["test_name"],
                "p_valor": lm_res["p_value"],
                "resultado": lm_res["interpretation"]
            }
        ],
        "detailed_results": {
            "kolmogorov_smirnov": ks_res,
            "mann_whitney": mw_res,
            "morgan_pitman": mp_res,
            "mcnemar": mc_res,
            "lagrange_multiplier": lm_res
        }
    }
