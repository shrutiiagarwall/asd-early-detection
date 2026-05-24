"""
tests/test_models.py
--------------------
Unit tests for model training utilities.
Uses minimal synthetic data — no real dataset required.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="module")
def synthetic_data():
    """Balanced binary classification dataset (100 samples, 17 features)."""
    X, y = make_classification(
        n_samples=200, n_features=17, n_informative=10,
        n_redundant=3, random_state=42, class_sep=1.5,
    )
    return X.astype(np.float32), y.astype(int)


class TestGetModels:

    def test_returns_dict(self):
        from src.models.train import get_models
        models = get_models()
        assert isinstance(models, dict)
        assert len(models) >= 3

    def test_all_have_predict_proba(self):
        from src.models.train import get_models
        import numpy as np
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=17, random_state=42)
        for name, model in get_models().items():
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (100, 2), f"{name}: predict_proba shape wrong"
            assert np.allclose(proba.sum(axis=1), 1.0), f"{name}: probabilities don't sum to 1"


class TestOptimizeThreshold:

    def test_returns_correct_types(self, synthetic_data):
        from src.models.train import optimize_threshold, get_models
        X, y = synthetic_data
        model = get_models()["Random Forest"]
        model.fit(X, y)
        y_pred, thresh, y_prob = optimize_threshold(model, X, y)
        assert isinstance(thresh, float)
        assert 0.0 <= thresh <= 1.0
        assert y_pred.shape == y.shape
        assert y_prob.shape == y.shape

    def test_threshold_respects_min_recall(self, synthetic_data):
        from src.models.train import optimize_threshold, get_models
        from sklearn.metrics import recall_score
        X, y = synthetic_data
        model = get_models()["Random Forest"]
        model.fit(X, y)
        y_pred, thresh, _ = optimize_threshold(model, X, y, min_recall=0.80)
        recall = recall_score(y, y_pred)
        # Either recall >= 0.80 OR fallback to 0.5 (no valid threshold found)
        assert recall >= 0.79 or thresh == 0.5


class TestCrossValidation:

    def test_cv_returns_best_model(self, synthetic_data):
        from src.models.train import train_with_cv, get_models
        X, y = synthetic_data
        models = {"LR": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=300)}
        best, name, results = train_with_cv(models, X, y, label="test", cv_folds=3)
        assert best is not None
        assert "LR" in results
        assert "mean_roc_auc" in results["LR"]
        assert 0.0 <= results["LR"]["mean_roc_auc"] <= 1.0


class TestEvaluationMetrics:

    def test_evaluate_model_keys(self, synthetic_data):
        from src.evaluation.metrics import evaluate_model
        from src.models.train import get_models
        X, y = synthetic_data
        model = get_models()["Random Forest"]
        model.fit(X, y)
        result = evaluate_model(model, X, y, "Test RF")
        for key in ["accuracy", "roc_auc", "sensitivity", "specificity", "ppv", "npv", "f1"]:
            assert key in result, f"Missing key: {key}"
            assert 0.0 <= result[key] <= 1.0, f"{key} out of [0,1]"

    def test_uncertainty_dataframe_shape(self, synthetic_data):
        from src.evaluation.metrics import monte_carlo_uncertainty
        from src.models.train import get_models
        X, y = synthetic_data
        model = get_models()["Random Forest"]
        model.fit(X, y)
        udf = monte_carlo_uncertainty(model, X)
        assert len(udf) == len(X)
        assert "mean_probability" in udf.columns
        assert "uncertainty_std" in udf.columns
        assert "uncertain_flag" in udf.columns
        assert udf["mean_probability"].between(0, 1).all()
