# backend/app/api/ensemble.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/ensemble", tags=["Multimodal Ensemble"])

class MetadataInput(BaseModel):
    age: int = 58
    sex: str = "Male"
    anatomical_site: str = "Torso / Espalda Superior"
    lesion_diameter_mm: float = 6.8

class EnsembleSimulationRequest(BaseModel):
    cnn_model: str = "DenseNet121_Attention"
    use_lightgbm: bool = True
    use_xgboost: bool = True
    use_catboost: bool = True
    cnn_weight: float = 0.60  # Weight for Vision CNN (0.0 to 1.0)
    metadata: MetadataInput = MetadataInput()

@router.post("/simulate")
def simulate_multimodal_ensemble(req: EnsembleSimulationRequest):
    """
    Simula el Ensamble Multimodal (ISIC 2024 1st Place Solution)
    Fusión de predicciones de Visión por Computadora + Modelos Tabulares GBDT.
    """
    # Predicción base simulada de Visión por Computadora (CNN)
    p_cnn = 0.885 if "DenseNet" in req.cnn_model else 0.842
    
    # Predicciones simuladas de modelos GBDT basados en metadata (Edad, Sitio, Diámetro)
    site_risk_bonus = 0.05 if "Torso" in req.metadata.anatomical_site else 0.02
    age_risk_bonus = (req.metadata.age / 100.0) * 0.10
    diameter_risk = (req.metadata.lesion_diameter_mm / 15.0) * 0.15
    
    p_lgb = min(0.99, 0.78 + site_risk_bonus + age_risk_bonus + diameter_risk)
    p_xgb = min(0.99, 0.76 + site_risk_bonus + age_risk_bonus + diameter_risk * 1.1)
    p_cat = min(0.99, 0.80 + site_risk_bonus + age_risk_bonus * 0.9 + diameter_risk)
    
    # Cálculo del peso efectivo de GBDT
    active_gbdts = []
    gbdt_preds = []
    
    if req.use_lightgbm:
        active_gbdts.append("LightGBM")
        gbdt_preds.append(p_lgb)
    if req.use_xgboost:
        active_gbdts.append("XGBoost")
        gbdt_preds.append(p_xgb)
    if req.use_catboost:
        active_gbdts.append("CatBoost")
        gbdt_preds.append(p_cat)
        
    p_gbdt_avg = float(sum(gbdt_preds) / len(gbdt_preds)) if gbdt_preds else p_cnn
    
    # Predicción final ensamblada ponderada
    w_cnn = req.cnn_weight
    w_gbdt = 1.0 - w_cnn
    
    p_ensemble = (w_cnn * p_cnn) + (w_gbdt * p_gbdt_avg)
    
    # Métricas pAUC > 80% TPR (ISIC 2024 Metric)
    pauc_cnn = 0.165
    pauc_gbdt = 0.158
    pauc_ensemble = min(0.198, 0.165 + (len(active_gbdts) * 0.009) + (w_cnn * 0.012))
    
    chart_comparison = [
        {"model": "Solo Visión (CNN)", "pauc": pauc_cnn, "type": "vision"},
        {"model": "Solo Tabular (GBDT)", "pauc": pauc_gbdt, "type": "tabular"},
        {"model": "Ensamble Multimodal", "pauc": round(pauc_ensemble, 4), "type": "ensemble"}
    ]
    
    return {
        "cnn_model": req.cnn_model,
        "active_gbdts": active_gbdts,
        "weights": {
            "cnn_weight": w_cnn,
            "gbdt_weight": round(w_gbdt, 2)
        },
        "individual_predictions": {
            "vision_cnn_prob": round(p_cnn, 4),
            "lightgbm_prob": round(p_lgb, 4),
            "xgboost_prob": round(p_xgb, 4),
            "catboost_prob": round(p_cat, 4),
            "gbdt_combined_prob": round(p_gbdt_avg, 4)
        },
        "ensemble_final_prob": round(p_ensemble, 4),
        "ensemble_diagnosis": "Maligno / Enfermo" if p_ensemble >= 0.5 else "Benigno / Sano",
        "pauc_metric_comparison": chart_comparison,
        "gain_percentage": round(((pauc_ensemble - pauc_cnn) / pauc_cnn) * 100, 2)
    }
