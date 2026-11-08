"""Unit and integration test verifying clinical risk calibration and monotonic sanity progression."""

import pytest
from app.schemas import PatientInput, PredictionResponse
from app.services.predictor import predictor_service
from src.evaluation.audit_predictions import BENCHMARK_PROFILES, audit_profile_predictions


def test_clinical_benchmark_profiles_monotonic_progression():
    """Verify that predictions across four escalating risk profiles yield monotonically increasing risk."""
    audited = audit_profile_predictions()
    assert len(audited) == 4

    scores = [item["response"].risk_score for item in audited]
    probs = [item["response"].probability for item in audited]

    # Monotonic score progression: Low -> Moderate -> High -> High
    assert scores[0] <= scores[1] <= scores[2] <= scores[3]
    # Strict probability ordering between low, moderate, and high tiers
    assert probs[0] < probs[1] < probs[2] <= probs[3]


def test_clinical_benchmark_profiles_risk_tier_boundaries():
    """Verify each benchmark profile satisfies clinical tier boundaries."""
    healthy_res: PredictionResponse = predictor_service.predict(BENCHMARK_PROFILES[0]["patient"])
    assert healthy_res.risk_category == "Low"
    assert healthy_res.risk_score < 20

    borderline_res: PredictionResponse = predictor_service.predict(BENCHMARK_PROFILES[1]["patient"])
    assert borderline_res.risk_category == "Moderate"
    assert 30 <= borderline_res.risk_score <= 65

    diabetic_res: PredictionResponse = predictor_service.predict(BENCHMARK_PROFILES[2]["patient"])
    assert diabetic_res.risk_category == "High"
    assert diabetic_res.risk_score >= 75

    severe_res: PredictionResponse = predictor_service.predict(BENCHMARK_PROFILES[3]["patient"])
    assert severe_res.risk_category == "High"
    assert severe_res.risk_score >= 90

