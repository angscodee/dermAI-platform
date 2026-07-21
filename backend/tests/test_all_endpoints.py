# backend/tests/test_all_endpoints.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_all_endpoints_response():
    # 1. Root
    res = client.get("/")
    assert res.status_code == 200

    # 2. EDA Summary
    res = client.get("/api/eda/summary")
    assert res.status_code == 200

    # 3. EDA Waterfall
    res = client.get("/api/eda/waterfall-correlations")
    assert res.status_code == 200

    # 4. EDA Ablation
    res = client.get("/api/eda/ablation-study")
    assert res.status_code == 200

    # 5. EDA Patient Risk
    res = client.get("/api/eda/patient-risk-stratification")
    assert res.status_code == 200

    # 6. EDA Validation Concordance
    res = client.get("/api/eda/validation-concordance")
    assert res.status_code == 200

    # 7. Ensemble Simulation
    res = client.post("/api/ensemble/simulate", json={
        "cnn_model": "DenseNet121_Attention",
        "use_lightgbm": True,
        "use_xgboost": True,
        "use_catboost": True,
        "cnn_weight": 0.6,
        "metadata": {
            "age": 58,
            "sex": "Male",
            "anatomical_site": "Torso / Espalda Superior",
            "lesion_diameter_mm": 6.8
        }
    })
    assert res.status_code == 200
