"""
tests/test_model_utils.py
Unit tests for src/model_utils.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from src.model_utils import evaluate_model, compare_models, log_cv_results
from sklearn.model_selection import cross_validate, StratifiedKFold


@pytest.fixture
def trained_model():
    X, y = make_classification(n_samples=500, n_features=10,
                                weights=[0.9, 0.1], random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X[:400], y[:400])
    return model, X[400:], y[400:]


def test_evaluate_model_returns_required_keys(trained_model):
    model, X_test, y_test = trained_model
    result = evaluate_model("LR", model, X_test, y_test, "Test")
    for key in ["model", "dataset", "AUC-PR", "F1", "ROC-AUC", "TP", "FP", "TN", "FN"]:
        assert key in result, f"Missing key: {key}"


def test_evaluate_model_metrics_in_range(trained_model):
    model, X_test, y_test = trained_model
    result = evaluate_model("LR", model, X_test, y_test, "Test")
    assert 0 <= result["AUC-PR"]  <= 1
    assert 0 <= result["F1"]      <= 1
    assert 0 <= result["ROC-AUC"] <= 1


def test_evaluate_model_confusion_matrix_sums(trained_model):
    model, X_test, y_test = trained_model
    result = evaluate_model("LR", model, X_test, y_test, "Test")
    total = result["TP"] + result["FP"] + result["TN"] + result["FN"]
    assert total == len(y_test)


def test_evaluate_model_raises_without_predict_proba():
    from sklearn.svm import SVC
    X, y = make_classification(n_samples=200, random_state=42)
    model = SVC()  # no predict_proba by default
    model.fit(X[:150], y[:150])
    with pytest.raises(TypeError, match="predict_proba"):
        evaluate_model("SVM", model, X[150:], y[150:], "Test")


def test_compare_models_returns_dataframe(trained_model):
    import pandas as pd
    model, X_test, y_test = trained_model
    r1 = evaluate_model("LR", model, X_test, y_test, "DataA")
    r2 = evaluate_model("LR", model, X_test, y_test, "DataB")
    df = compare_models([r1, r2])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "AUC-PR" in df.columns


def test_log_cv_results_returns_correct_keys(trained_model):
    model, X_test, y_test = trained_model
    X, y = make_classification(n_samples=500, n_features=10,
                                weights=[0.9, 0.1], random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv,
                            scoring=["average_precision", "f1"])
    result = log_cv_results(scores, "LR", "Test")
    assert "cv_aucpr" in result
    assert "cv_f1" in result
    assert "+/-" in result["cv_aucpr"]
