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


def test_elevated_hba1c_profile_non_zero_elevated():
    """Assert Profile (Age 32, BMI 25.3, HbA1c 6.6%, Glucose 102) evaluates to elevated risk >= 40."""
    patient = PatientInput(
        gender="Male",
        age=32.0,
        hypertension=0,
        heart_disease=0,
        smoking_history="never",
        bmi=25.3,
        HbA1c_level=6.6,
        blood_glucose_level=102.0,
    )
    res: PredictionResponse = predictor_service.predict(patient)
    assert res.risk_score >= 40, f"Expected risk score >= 40, got {res.risk_score}%"
    assert res.risk_category in ("Moderate", "High"), f"Expected Moderate or High tier, got {res.risk_category}"


def test_monotonic_risk_progression_hba1c_glucose():
    """Verify monotonicity: Risk(HbA1c=7.4, Glucose=200) > Risk(HbA1c=6.6, Glucose=200)."""
    p_66 = PatientInput(
        gender="Male",
        age=32.0,
        hypertension=0,
        heart_disease=0,
        smoking_history="never",
        bmi=25.3,
        HbA1c_level=6.6,
        blood_glucose_level=200.0,
    )
    p_74 = PatientInput(
        gender="Male",
        age=32.0,
        hypertension=0,
        heart_disease=0,
        smoking_history="never",
        bmi=25.3,
        HbA1c_level=7.4,
        blood_glucose_level=200.0,
    )
    res_66 = predictor_service.predict(p_66)
    res_74 = predictor_service.predict(p_74)

    assert res_74.probability > res_66.probability, (
        f"Expected Risk(7.4, 200) > Risk(6.6, 200), but got {res_74.probability} <= {res_66.probability}"
    )
    assert res_74.risk_score >= res_66.risk_score


def test_blood_glucose_incremental_rate_of_change_bounded():
    """Verify single-unit glucose increments around 108 mg/dL produce continuous, smooth transitions (< 5% shift per single mg/dL)."""
    base_patient = PatientInput(
        gender="Male",
        age=32.0,
        hypertension=0,
        heart_disease=0,
        smoking_history="never",
        bmi=25.3,
        HbA1c_level=6.6,
        blood_glucose_level=100.0,
    )

    probs = []
    for g in range(104, 113):  # 104 to 112 covering 108 mg/dL
        p = base_patient.model_copy(update={"blood_glucose_level": float(g)})
        res = predictor_service.predict(p)
        probs.append((g, res.probability))

    for i in range(len(probs) - 1):
        g_curr, p_curr = probs[i]
        g_next, p_next = probs[i + 1]
        delta = abs(p_next - p_curr)
        assert delta < 0.05, (
            f"Single-unit glucose shift from {g_curr} to {g_next} mg/dL produced excessive delta {delta:.4f} (>= 0.05)"
        )
        assert p_next >= p_curr - 1e-6, (
            f"Risk unexpectedly decreased from glucose {g_curr} ({p_curr}) to {g_next} ({p_next})"
        )
