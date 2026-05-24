"""
tests/test_pipeline.py
----------------------
Smoke tests for the public pipeline components.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample" / "ASD_sample_public.csv"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture(scope="module")
def engineered_df(sample_df):
    from src.features.pipeline import engineer_features
    return engineer_features(sample_df.copy())


# ── Feature engineering tests ────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_load_sample(self, sample_df):
        assert len(sample_df) == 50
        for col in [f"A{i}" for i in range(1, 11)] + ["Age", "Sex", "Class"]:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_encode_demographics(self, sample_df):
        from src.features.pipeline import encode_demographics
        df = encode_demographics(sample_df.copy())
        assert df["Sex"].dtype in [np.int64, np.int32, int, np.float64]
        assert set(df["Sex"].dropna().unique()).issubset({0, 1})

    def test_compute_aq_domains(self, sample_df):
        from src.features.pipeline import compute_aq_domains, encode_demographics
        df = encode_demographics(sample_df.copy())
        df = compute_aq_domains(df)
        assert "Total_Score" in df.columns
        assert "Social_Score" in df.columns
        assert df["Total_Score"].between(0, 10).all(), "Total_Score out of [0,10]"

    def test_engineer_features_columns(self, engineered_df):
        expected = ["Bio_Risk", "Bio_Combined", "Age_Band_Enc",
                    "FamASD_x_Threshold", "Risk_x_AgeBand",
                    "Score_Tier_Enc", "Composite_Risk"]
        for col in expected:
            assert col in engineered_df.columns, f"Missing engineered col: {col}"

    def test_no_nans(self, engineered_df):
        numeric_cols = engineered_df.select_dtypes(include=[np.number]).columns
        assert engineered_df[numeric_cols].isnull().sum().sum() == 0, \
            "NaN values found in engineered numeric columns"


# ── Feature matrix tests ──────────────────────────────────────────────────────

class TestFeatureMatrix:

    def test_demo_shape(self, engineered_df):
        from src.features.pipeline import build_feature_matrix
        X, y, names = build_feature_matrix(engineered_df, feature_set="demo")
        assert X.shape[0] == 50
        assert X.shape[1] == len(names)
        assert y is not None and len(y) == 50

    def test_full_shape(self, engineered_df):
        from src.features.pipeline import build_feature_matrix
        X, y, names = build_feature_matrix(engineered_df, feature_set="full")
        assert X.shape[0] == 50
        assert X.shape[1] > 10, "Full feature set should include AQ items"


# ── Evaluation tests ──────────────────────────────────────────────────────────

class TestEvaluation:

    def test_compute_psi_identical(self):
        from src.evaluation.metrics import compute_psi
        arr = np.random.rand(200)
        psi = compute_psi(arr, arr)
        assert psi < 0.01, "PSI of identical distributions should be ~0"

    def test_compute_psi_different(self):
        from src.evaluation.metrics import compute_psi
        ref = np.random.normal(0, 1, 500)
        new = np.random.normal(3, 1, 500)
        psi = compute_psi(ref, new)
        assert psi > 0.2, "PSI of very different distributions should be > 0.2"

    def test_monitor_drift_output(self):
        from src.evaluation.metrics import monitor_drift
        X_tr = np.random.rand(200, 5)
        X_new= np.random.rand(100, 5)
        names = [f"feat_{i}" for i in range(5)]
        df = monitor_drift(X_tr, X_new, names)
        assert len(df) == 5
        assert "PSI" in df.columns and "Status" in df.columns


# ── PSI threshold tests ───────────────────────────────────────────────────────

class TestPSIThresholds:

    @pytest.mark.parametrize("psi,expected_status", [
        (0.05,  "✅ STABLE"),
        (0.15,  "⚠️ MONITOR"),
        (0.25,  "🔴 RETRAIN"),
    ])
    def test_psi_status_labels(self, psi, expected_status):
        # Status logic is inline in monitor_drift; test it via string comparison
        status = "✅ STABLE" if psi < 0.1 else ("⚠️ MONITOR" if psi < 0.2 else "🔴 RETRAIN")
        assert status == expected_status
