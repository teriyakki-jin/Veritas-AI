"""Tests for src/models/fusion.py"""

import numpy as np
import pytest
from models.fusion import (
    apply_temperature,
    logits_to_credibility,
    score_to_verdict,
    FusionEngine,
    LIAR_CREDIBILITY,
    FEVER_CREDIBILITY,
    FNN_CREDIBILITY,
)


class TestApplyTemperature:
    def test_output_sums_to_one(self):
        logits = np.array([1.0, 2.0, 0.5])
        probs = apply_temperature(logits, temperature=1.0)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_zero_temperature_defaults_to_one(self):
        logits = np.array([1.0, 2.0, 0.5])
        probs_zero = apply_temperature(logits, temperature=0.0)
        probs_one = apply_temperature(logits, temperature=1.0)
        np.testing.assert_allclose(probs_zero, probs_one)

    def test_negative_temperature_defaults_to_one(self):
        logits = np.array([1.0, 2.0])
        probs = apply_temperature(logits, temperature=-1.0)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_high_temperature_flattens_distribution(self):
        logits = np.array([10.0, 0.0])
        probs_low = apply_temperature(logits, temperature=0.1)
        probs_high = apply_temperature(logits, temperature=10.0)
        # Low temperature should be more peaked
        assert probs_low[0] > probs_high[0]

    def test_all_zeros_returns_uniform(self):
        logits = np.array([0.0, 0.0, 0.0])
        probs = apply_temperature(logits, temperature=1.0)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3], atol=1e-6)


class TestLogitsToCredibility:
    def test_output_in_unit_interval(self):
        logits = np.array([1.0, 2.0, 0.5, -1.0, 0.0, 3.0])
        score = logits_to_credibility(logits, LIAR_CREDIBILITY)
        assert 0.0 <= score <= 1.0

    def test_fever_supports_gives_high_score(self):
        # Class 0 = SUPPORTS = credibility 1.0
        logits = np.array([10.0, -10.0, -10.0])
        score = logits_to_credibility(logits, FEVER_CREDIBILITY)
        assert score > 0.9

    def test_fever_refutes_gives_low_score(self):
        # Class 1 = REFUTES = credibility 0.0
        logits = np.array([-10.0, 10.0, -10.0])
        score = logits_to_credibility(logits, FEVER_CREDIBILITY)
        assert score < 0.1

    def test_fnn_real_gives_high_score(self):
        # Class 1 = real = credibility 1.0
        logits = np.array([-10.0, 10.0])
        score = logits_to_credibility(logits, FNN_CREDIBILITY)
        assert score > 0.9

    def test_fnn_fake_gives_low_score(self):
        # Class 0 = fake = credibility 0.0
        logits = np.array([10.0, -10.0])
        score = logits_to_credibility(logits, FNN_CREDIBILITY)
        assert score < 0.1


class TestScoreToVerdict:
    def test_true_at_high_score(self):
        assert score_to_verdict(0.80) == "TRUE"
        assert score_to_verdict(0.75) == "TRUE"

    def test_mostly_true(self):
        assert score_to_verdict(0.65) == "MOSTLY TRUE"
        assert score_to_verdict(0.55) == "MOSTLY TRUE"

    def test_half_true(self):
        assert score_to_verdict(0.50) == "HALF TRUE"
        assert score_to_verdict(0.45) == "HALF TRUE"

    def test_mostly_false(self):
        assert score_to_verdict(0.30) == "MOSTLY FALSE"
        assert score_to_verdict(0.25) == "MOSTLY FALSE"

    def test_false_at_low_score(self):
        assert score_to_verdict(0.10) == "FALSE"
        assert score_to_verdict(0.0) == "FALSE"

    def test_boundary_values(self):
        assert score_to_verdict(0.749) == "MOSTLY TRUE"
        assert score_to_verdict(0.549) == "HALF TRUE"
        assert score_to_verdict(0.449) == "MOSTLY FALSE"
        assert score_to_verdict(0.249) == "FALSE"


class TestFusionEngine:
    def setup_method(self):
        self.engine = FusionEngine()

    def test_fuse_all_three_models(self):
        outputs = {
            "liar":  np.array([0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),
            "fever": np.array([1.5, -0.5, 0.2]),
            "fnn":   np.array([-0.3, 1.2]),
        }
        result = self.engine.fuse(outputs)
        assert "credibility_score" in result
        assert "verdict" in result
        assert "model_details" in result
        assert 0.0 <= result["credibility_score"] <= 1.0

    def test_fuse_single_model(self):
        outputs = {"fnn": np.array([-1.0, 2.0])}
        result = self.engine.fuse(outputs)
        assert result["credibility_score"] > 0.5  # "real" class dominates

    def test_fuse_empty_input_returns_half(self):
        result = self.engine.fuse({})
        assert result["credibility_score"] == 0.5

    def test_fuse_model_details_present(self):
        outputs = {
            "liar":  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "fever": np.array([1.0, 1.0, 1.0]),
        }
        result = self.engine.fuse(outputs)
        for name in ["liar", "fever"]:
            assert name in result["model_details"]
            detail = result["model_details"][name]
            assert "credibility_score" in detail
            assert "predicted_class" in detail
            assert "probabilities" in detail

    def test_fuse_unknown_model_ignored(self):
        outputs = {"unknown_model": np.array([1.0, 2.0])}
        result = self.engine.fuse(outputs)
        assert result["credibility_score"] == 0.5
        assert "unknown_model" not in result["model_details"]

    def test_custom_weights_applied(self):
        engine_biased = FusionEngine(weights={"fnn": 1.0, "liar": 0.0, "fever": 0.0})
        outputs = {
            "fnn":   np.array([10.0, -10.0]),  # strongly fake
            "liar":  np.array([-10.0, -10.0, -10.0, -10.0, -10.0, 10.0]),  # strongly true
            "fever": np.array([10.0, -10.0, -10.0]),  # strongly supports
        }
        result = engine_biased.fuse(outputs)
        # With fnn weight=1.0 and fnn predicting fake, score should be low
        assert result["credibility_score"] < 0.2
