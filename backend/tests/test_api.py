# backend/tests/test_api.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_eda_summary():
    res = client.get("/api/eda/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_samples" in data
    assert data["total_samples"] == 25331

def test_training_run():
    res = client.post("/api/training/run", json={"epochs": 2, "batch_size": 32})
    assert res.status_code == 200
    data = res.json()
    assert len(data["models_summary"]) == 5

def test_statistical_tests():
    res = client.post("/api/statistical/run-all", json={"model_a_name": "DenseNet121_Attention", "model_b_name": "ResNet50"})
    assert res.status_code == 200
    data = res.json()
    assert len(data["summary_table"]) == 5
