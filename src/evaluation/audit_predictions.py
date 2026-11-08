"""Model prediction sanity and probability calibration audit across clinical benchmark tiers."""

import logging
import sys
from typing import Any, Dict, List

from app.schemas import PatientInput, PredictionResponse
from app.services.predictor import predictor_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PROFILES: List[Dict[str, Any]] = [
    {
        "name": "Healthy Young Adult",
        "description": "Age 22, BMI 21.0, Glucose 85, HbA1c 4.8%, Non-smoker",
        "patient": PatientInput(
            gender="Female",
            age=22,
            hypertension=0,
            heart_disease=0,
            smoking_history="never",
            bmi=21.0,
            HbA1c_level=4.8,
            blood_glucose_level=85,
        ),
        "expected_tier": "Low",
        "max_risk_score": 20,
    },
    {
        "name": "Borderline / Pre-diabetic",
        "description": "Age 48, BMI 27.5, Glucose 128, HbA1c 6.0%, No complications",
        "patient": PatientInput(
            gender="Male",
            age=48,
            hypertension=0,
            heart_disease=0,
            smoking_history="never",
            bmi=27.5,
            HbA1c_level=6.0,
            blood_glucose_level=128,
        ),
        "expected_tier": "Moderate",
        "min_risk_score": 30,
        "max_risk_score": 65,
    },
    {
        "name": "Clinical Diabetic",
        "description": "Age 60, BMI 34.0, Glucose 190, HbA1c 7.6%, Hypertension",
        "patient": PatientInput(
            gender="Female",
            age=60,
            hypertension=1,
            heart_disease=0,
            smoking_history="former",
            bmi=34.0,
            HbA1c_level=7.6,
            blood_glucose_level=190,
        ),
        "expected_tier": "High",
        "min_risk_score": 75,
    },
    {
        "name": "Severe Metabolic Burden",
        "description": "Age 68, BMI 38.0, Glucose 240, HbA1c 8.8%, Heart Disease",
        "patient": PatientInput(
            gender="Male",
            age=68,
            hypertension=1,
            heart_disease=1,
            smoking_history="current",
            bmi=38.0,
            HbA1c_level=8.8,
            blood_glucose_level=240,
        ),
        "expected_tier": "High",
        "min_risk_score": 90,
    },
]


def audit_profile_predictions() -> List[Dict[str, Any]]:
    """Execute prediction inference across benchmark profiles and return audited results.

    Returns:
        List of audited profile dictionaries with prediction responses and compliance flags.
    """
    logger.info("Starting Clinical Prediction Sanity & Calibration Audit...")
    audited_results = []

    for item in BENCHMARK_PROFILES:
        res: PredictionResponse = predictor_service.predict(item["patient"])
        logger.info(
            "Profile: %-26s | Prob: %.4f | Score: %3d%% | Tier: %-8s (Expected: %s)",
            item["name"],
            res.probability,
            res.risk_score,
            res.risk_category,
            item["expected_tier"],
        )

        meets_tier = res.risk_category == item["expected_tier"]
        meets_score = True
        if "max_risk_score" in item and res.risk_score > item["max_risk_score"]:
            meets_score = False
        if "min_risk_score" in item and res.risk_score < item["min_risk_score"]:
            meets_score = False

        audited_results.append({
            "name": item["name"],
            "description": item["description"],
            "response": res,
            "meets_tier": meets_tier,
            "meets_score": meets_score,
        })

    return audited_results


def run_prediction_audit() -> bool:
    """Run CLI audit verification banner and return True if all benchmarks pass."""
    results = audit_profile_predictions()

    print("\n" + "=" * 80)
    print("CLINICAL MODEL CALIBRATION & RISK STRATIFICATION SANITY AUDIT")
    print("=" * 80)
    print(f"{'Profile Name':<28} | {'Probability':<12} | {'Risk Score':<10} | {'Tier':<10} | {'Status':<6}")
    print("-" * 80)

    all_passed = True
    for item in results:
        res: PredictionResponse = item["response"]
        status = "PASS" if item["meets_tier"] and item["meets_score"] else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(
            f"{item['name']:<28} | {res.probability:<12.4f} | {f'{res.risk_score}%':<10} | {res.risk_category:<10} | {status:<6}"
        )

    print("-" * 80)

    # Verify monotonic progression across risk scores
    scores = [item["response"].risk_score for item in results]
    monotonic = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    strict_progression = scores[0] < scores[1] < scores[2] <= scores[3]

    print(f"Monotonic Progression Audit: {'PASSED' if (monotonic and strict_progression) else 'FAILED'}")
    print(f"Score Progression: {' -> '.join(str(s) + '%' for s in scores)}")
    print("=" * 80 + "\n")

    return all_passed and monotonic and strict_progression


if __name__ == "__main__":
    success = run_prediction_audit()
    sys.exit(0 if success else 1)

