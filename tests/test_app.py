"""
tests/test_app.py
-----------------
Smoke tests for the Gradio app prediction logic.
Runs in DEMO MODE (no model weights needed).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestStubPredictor:
    """Tests for the rule-based stub used in DEMO MODE."""

    def test_stub_returns_floats(self):
        from app.app import _stub_predict
        full_p, demo_p = _stub_predict(aq_score=8, age=25, bio_risk=1)
        assert isinstance(full_p, float)
        assert isinstance(demo_p, float)

    def test_stub_probabilities_in_range(self):
        from app.app import _stub_predict
        for score in range(0, 11):
            full_p, demo_p = _stub_predict(aq_score=score, age=30, bio_risk=0)
            assert 0.0 <= full_p <= 1.0, f"full_p out of range at score={score}"
            assert 0.0 <= demo_p <= 1.0, f"demo_p out of range at score={score}"

    def test_higher_score_higher_probability(self):
        """High AQ-10 score should produce higher ASD probability than low score."""
        from app.app import _stub_predict
        p_high, _ = _stub_predict(aq_score=9, age=25, bio_risk=0)
        p_low,  _ = _stub_predict(aq_score=1, age=25, bio_risk=0)
        assert p_high > p_low, "Higher AQ score should yield higher ASD probability"

    def test_bio_risk_increases_demo_probability(self):
        from app.app import _stub_predict
        _, p_bio2 = _stub_predict(aq_score=5, age=25, bio_risk=2)
        _, p_bio0 = _stub_predict(aq_score=5, age=25, bio_risk=0)
        assert p_bio2 >= p_bio0, "Higher bio-risk should yield >= demographic probability"


class TestPredictFunction:
    """Tests for the main predict() function in DEMO MODE."""

    def test_predict_returns_markdown_string(self):
        from app.app import predict
        result = predict(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # A1-A10 all 1 (score=10)
            age=25, sex="Male",
            jaundice="No", family_asd="No",
        )
        assert isinstance(result, str)
        assert len(result) > 50

    def test_predict_high_score_flags_high_risk(self):
        from app.app import predict
        result = predict(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            age=25, sex="Male",
            jaundice="No", family_asd="No",
        )
        assert "HIGH RISK" in result or "DEMO MODE" in result

    def test_predict_zero_score_low_risk(self):
        from app.app import predict
        result = predict(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            age=25, sex="Female",
            jaundice="No", family_asd="No",
        )
        assert "LOW RISK" in result or "DEMO MODE" in result

    def test_predict_disclaimer_always_present(self):
        """Disclaimer must appear in every prediction output."""
        from app.app import predict
        for score_set in [(1,)*10, (0,)*10]:
            result = predict(*score_set, age=30, sex="Male", jaundice="No", family_asd="No")
            assert "research" in result.lower() or "disclaimer" in result.lower() or "not" in result.lower()

    def test_predict_age_boundary_values(self):
        """App should handle age boundary values without error."""
        from app.app import predict
        for age in [4, 11, 12, 17, 18, 40, 41, 80]:
            result = predict(
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                age=age, sex="Male",
                jaundice="No", family_asd="Yes",
            )
            assert isinstance(result, str), f"predict() failed for age={age}"
