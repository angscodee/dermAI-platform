"""
tests/test_data_processor.py
Tests unitarios para data_processor.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd

from data_processor import (
    analyze_consistency,
    create_mcc_comparison_dataframe,
    format_metrics_for_display,
    get_model_metrics,
)


# ---------------------------------------------------------------------------
# analyze_consistency
# ---------------------------------------------------------------------------
class TestAnalyzeConsistency:
    def test_empty_input(self):
        result = analyze_consistency([])
        assert result["consistent"] is False
        assert result["agreement_rate"] == 0.0

    def test_all_malignant(self):
        results = [{"Diagnóstico": "Maligno"} for _ in range(5)]
        result = analyze_consistency(results)
        assert result["consistent"] is True
        assert result["agreement_rate"] == 100.0
        assert result["majority_diagnosis"] == "Maligno"

    def test_all_benign(self):
        results = [{"Diagnóstico": "Benigno"} for _ in range(4)]
        result = analyze_consistency(results)
        assert result["consistent"] is True
        assert result["majority_diagnosis"] == "Benigno"

    def test_mixed_inconsistent(self):
        results = [
            {"Diagnóstico": "Maligno"},
            {"Diagnóstico": "Benigno"},
            {"Diagnóstico": "Maligno"},
            {"Diagnóstico": "Benigno"},
        ]
        result = analyze_consistency(results)
        assert result["consistent"] is False
        assert result["agreement_rate"] == 50.0

    def test_error_entries_ignored(self):
        results = [
            {"Diagnóstico": "Maligno"},
            {"Diagnóstico": "Error"},
            {"Diagnóstico": "Maligno"},
        ]
        result = analyze_consistency(results)
        # Solo se cuentan "Maligno" y "Benigno", no "Error"
        assert result["majority_diagnosis"] == "Maligno"
        assert result["malignant_votes"] == 2


# ---------------------------------------------------------------------------
# create_mcc_comparison_dataframe
# ---------------------------------------------------------------------------
class TestMCCDataframe:
    def _sample_data(self):
        return [
            {"model": "ModelA", "mcc": 0.75, "accuracy": 0.87, "f1": 0.86},
            {"model": "ModelB", "mcc": 0.60, "accuracy": 0.82, "f1": 0.81},
        ]

    def test_returns_dataframe(self):
        df = create_mcc_comparison_dataframe(self._sample_data())
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = create_mcc_comparison_dataframe(self._sample_data())
        assert "Modelo" in df.columns
        assert "MCC" in df.columns

    def test_correct_row_count(self):
        df = create_mcc_comparison_dataframe(self._sample_data())
        assert len(df) == 2

    def test_empty_input(self):
        df = create_mcc_comparison_dataframe([])
        assert df.empty


# ---------------------------------------------------------------------------
# format_metrics_for_display
# ---------------------------------------------------------------------------
class TestFormatMetrics:
    def _sample_metrics(self):
        return {
            "accuracy": 0.874,
            "sensitivity": 0.863,
            "specificity": 0.885,
            "precision": 0.879,
            "f1_score": 0.871,
            "mcc": 0.748,
            "auc": 0.941,
        }

    def test_returns_dict(self):
        result = format_metrics_for_display(self._sample_metrics())
        assert isinstance(result, dict)

    def test_accuracy_format(self):
        result = format_metrics_for_display(self._sample_metrics())
        assert result["accuracy"] == "87.4%"

    def test_mcc_format(self):
        result = format_metrics_for_display(self._sample_metrics())
        assert result["mcc"] == "0.748"

    def test_missing_keys_default_zero(self):
        result = format_metrics_for_display({})
        assert result["accuracy"] == "0.0%"
        assert result["mcc"] == "0.000"


# ---------------------------------------------------------------------------
# get_model_metrics
# ---------------------------------------------------------------------------
class TestGetModelMetrics:
    def test_known_model_returns_real_data(self):
        metrics, is_real = get_model_metrics("EfficientNetB0")
        assert is_real is True
        assert "accuracy" in metrics
        assert 0 < metrics["accuracy"] < 1

    def test_unknown_model_returns_simulated(self):
        metrics, is_real = get_model_metrics("NonExistentModel123")
        assert is_real is False
        assert "accuracy" in metrics

    def test_custom_training_metrics_override(self):
        custom = {
            "CustomModel": {
                "accuracy": 0.99, "sensitivity": 0.98, "specificity": 0.99,
                "precision": 0.98, "f1_score": 0.98, "mcc": 0.95, "auc": 0.99,
                "confusion_matrix": [[100, 1], [1, 100]],
            }
        }
        metrics, is_real = get_model_metrics("CustomModel", training_metrics=custom)
        assert is_real is True
        assert metrics["accuracy"] == 0.99
