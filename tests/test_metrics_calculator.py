"""
tests/test_metrics_calculator.py
Tests unitarios para metrics_calculator.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest

from metrics_calculator import (
    calculate_mcc,
    mcc_from_confusion_matrix,
    mcnemar_test,
    perform_mcnemar_comparisons,
)


# ---------------------------------------------------------------------------
# calculate_mcc
# ---------------------------------------------------------------------------
class TestCalculateMCC:
    def test_perfect_classifier(self):
        """Un clasificador perfecto devuelve MCC = 1.0"""
        mcc = calculate_mcc(tp=100, tn=100, fp=0, fn=0)
        assert math.isclose(mcc, 1.0, abs_tol=1e-9)

    def test_inverse_classifier(self):
        """Un clasificador completamente invertido devuelve MCC = -1.0"""
        mcc = calculate_mcc(tp=0, tn=0, fp=100, fn=100)
        assert math.isclose(mcc, -1.0, abs_tol=1e-9)

    def test_random_classifier(self):
        """Un clasificador aleatorio (50/50) devuelve MCC ≈ 0"""
        mcc = calculate_mcc(tp=50, tn=50, fp=50, fn=50)
        assert math.isclose(mcc, 0.0, abs_tol=1e-9)

    def test_zero_denominator_returns_zero(self):
        """Si el denominador es cero, retorna 0.0 sin lanzar excepción"""
        mcc = calculate_mcc(tp=10, tn=0, fp=0, fn=0)
        assert mcc == 0.0

    def test_known_value(self):
        """
        TP=70, TN=60, FP=10, FN=20
        MCC = (70*60 - 10*20) / sqrt((80)(90)(70)(80))
            = (4200 - 200) / sqrt(40320000)
            = 4000 / 6349.8... ≈ 0.6299
        """
        mcc = calculate_mcc(tp=70, tn=60, fp=10, fn=20)
        expected = (70 * 60 - 10 * 20) / math.sqrt(80 * 90 * 70 * 80)
        assert math.isclose(mcc, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# mcc_from_confusion_matrix
# ---------------------------------------------------------------------------
class TestMCCFromMatrix:
    def test_list_input(self):
        """Acepta lista de listas [[TN, FP], [FN, TP]]"""
        cm = [[60, 10], [20, 70]]
        mcc = mcc_from_confusion_matrix(cm)
        expected = calculate_mcc(tp=70, tn=60, fp=10, fn=20)
        assert math.isclose(mcc, expected, rel_tol=1e-6)

    def test_perfect_classifier_matrix(self):
        cm = [[100, 0], [0, 100]]
        mcc = mcc_from_confusion_matrix(cm)
        assert math.isclose(mcc, 1.0, abs_tol=1e-9)

    def test_numpy_array_input(self):
        import numpy as np
        cm = np.array([[60, 10], [20, 70]])
        mcc = mcc_from_confusion_matrix(cm)
        assert -1.0 <= mcc <= 1.0


# ---------------------------------------------------------------------------
# mcnemar_test
# ---------------------------------------------------------------------------
class TestMcNemarTest:
    def test_identical_models(self):
        """b == c → chi2 ~= 0, p ~= 1.0"""
        chi2, p = mcnemar_test(b=10, c=10)
        # Con corrección de continuidad: (|10-10| - 1)^2 / 20 = 1/20 = 0.05
        assert chi2 >= 0
        assert 0.0 <= p <= 1.0

    def test_very_different_models(self):
        """Gran diferencia entre b y c → p < 0.05 (significativo)"""
        chi2, p = mcnemar_test(b=100, c=1)
        assert p < 0.05

    def test_both_zero(self):
        """b=0, c=0 → chi2=0, p=1.0"""
        chi2, p = mcnemar_test(b=0, c=0)
        assert chi2 == 0.0
        assert p == 1.0

    def test_symmetry(self):
        """mcnemar_test(b, c) == mcnemar_test(c, b) porque usamos |b-c|"""
        chi2_1, p1 = mcnemar_test(b=30, c=10)
        chi2_2, p2 = mcnemar_test(b=10, c=30)
        assert math.isclose(chi2_1, chi2_2, rel_tol=1e-9)
        assert math.isclose(p1, p2, rel_tol=1e-9)

    def test_known_value(self):
        """
        b=20, c=5  → chi2 = (|20-5| - 1)^2 / 25 = 196/25 = 7.84
        p debe ser < 0.05
        """
        chi2, p = mcnemar_test(b=20, c=5)
        assert math.isclose(chi2, (abs(20 - 5) - 1) ** 2 / 25, rel_tol=1e-6)
        assert p < 0.05


# ---------------------------------------------------------------------------
# perform_mcnemar_comparisons
# ---------------------------------------------------------------------------
class TestPerformMcNemarComparisons:
    def _sample_mcc_data(self):
        return [
            {"model": "ModelA", "mcc": 0.75, "accuracy": 0.87, "f1": 0.86},
            {"model": "ModelB", "mcc": 0.60, "accuracy": 0.80, "f1": 0.79},
            {"model": "ModelC", "mcc": 0.70, "accuracy": 0.85, "f1": 0.84},
        ]

    def test_returns_correct_number_of_pairs(self):
        """3 modelos → 3 pares (combinaciones de 2)"""
        results = perform_mcnemar_comparisons(self._sample_mcc_data())
        assert len(results) == 3

    def test_result_structure(self):
        results = perform_mcnemar_comparisons(self._sample_mcc_data())
        for r in results:
            assert "model_a" in r
            assert "model_b" in r
            assert "chi2" in r
            assert "p_value" in r
            assert "significant" in r
            assert isinstance(r["significant"], bool)

    def test_p_value_range(self):
        results = perform_mcnemar_comparisons(self._sample_mcc_data())
        for r in results:
            assert 0.0 <= r["p_value"] <= 1.0

    def test_single_model_no_pairs(self):
        single = [{"model": "OnlyOne", "mcc": 0.75, "accuracy": 0.87, "f1": 0.86}]
        results = perform_mcnemar_comparisons(single)
        assert results == []

    def test_empty_input(self):
        results = perform_mcnemar_comparisons([])
        assert results == []
