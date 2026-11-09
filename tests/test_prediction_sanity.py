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


def test_blood_glucose_incremental_rate_of_change_bounded():
    """Verify increments of 1 unit in blood glucose around the threshold (100 to 110 mg/dL) produce bounded rate-of-change deltas (< 25%)."""
    base_patient = PatientInput(
        gender="Female",
        age=45.0,
        hypertension=0,
        heart_disease=0,
        smoking_history="never",
        bmi=26.5,
        HbA1c_level=5.8,
        blood_glucose_level=100.0,
    )

    probs = []
    for g in range(100, 111):
        p = base_patient.model_copy(update={"blood_glucose_level": float(g)})
        res = predictor_service.predict(p)
        probs.append(res.probability)

    for i in range(len(probs) - 1):
        delta = abs(probs[i + 1] - probs[i])
        assert delta < 0.25, (
            f"Glucose shift from {100 + i} to {101 + i} mg/dL produced excessive delta {delta:.4f} (>= 0.25)"
        )
