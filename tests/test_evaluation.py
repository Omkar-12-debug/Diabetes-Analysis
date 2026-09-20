"""Unit and integration tests for model calibration, demographic fairness, and evaluation pipeline.

Tests:
- Calibration metrics (Brier score, ECE, MCE) on mocked and deterministic arrays.
- Calibration reliability curve plotting.
- Demographic feature reconstruction from one-hot encodings.
- Subgroup fairness auditing across gender and age slices.
- Sample-size protection and warning for tiny cohorts (e.g. 'Other' gender N=2).
- Dynamic champion loading and end-to-end evaluation execution.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import (
    compute_brier_score,
    compute_expected_calibration_error,
    evaluate_calibration,
    plot_calibration_curve,
)
from src.evaluation.fairness import (
    compute_subgroup_metrics,
    evaluate_fairness,
    evaluate_fairness_for_attribute,
    reconstruct_demographic_features,
)
from src.evaluation.evaluate_all import (
    CHAMPION_URI,
    get_predictions,
    load_champion_model,
    run_evaluation,
)


# ==========================================
# 1. Calibration Unit Tests
# ==========================================

def test_brier_score_bounds_and_correctness():
    """Verify Brier score bounds and exact values for perfect and imperfect predictions."""
    y_true = np.array([0, 0, 1, 1])

    # Perfect predictions -> Brier = 0.0
    perfect_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert compute_brier_score(y_true, perfect_prob) == pytest.approx(0.0)

    # Inverted predictions -> Brier = 1.0
    inverted_prob = np.array([1.0, 1.0, 0.0, 0.0])
    assert compute_brier_score(y_true, inverted_prob) == pytest.approx(1.0)

    # 50-50 guess -> Brier = 0.25
    half_prob = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_brier_score(y_true, half_prob) == pytest.approx(0.25)


def test_expected_calibration_error_calculation():
    """Test ECE and MCE computation on deterministic mocked probability bins."""
    # Perfectly calibrated cases
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])

    ece, mce, details = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert "bins" in details
    assert details["n_bins"] == 10


def test_evaluate_calibration_dict_structure():
    """Verify evaluate_calibration returns complete dictionary with required keys."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.1, size=200)
    y_prob = np.clip(y_true * 0.7 + np.random.normal(0, 0.2, size=200), 0.0, 1.0)

    result = evaluate_calibration(y_true, y_prob, n_bins=5)

    required_keys = [
        "brier_score",
        "expected_calibration_error",
        "maximum_calibration_error",
        "calibration_curve",
        "bin_details",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert 0.0 <= result["brier_score"] <= 1.0
    assert 0.0 <= result["expected_calibration_error"] <= 1.0
    assert len(result["calibration_curve"]["prob_true"]) > 0


def test_plot_calibration_curve(tmp_path):
    """Verify plot_calibration_curve generates a valid non-empty image file."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.1, 0.2, 0.6, 0.2])

    save_file = tmp_path / "test_calibration.png"
    plot_calibration_curve(y_true, y_prob, save_path=save_file, n_bins=5)

    assert save_file.exists()
    assert save_file.stat().st_size > 1000  # Non-trivial image file


# ==========================================
# 2. Fairness Unit Tests
# ==========================================

def test_reconstruct_demographic_features():
    """Verify one-hot encoded columns are correctly reconstructed to string labels."""
    df = pd.DataFrame({
        "num__age": [25, 60],
        "cat__gender_Female": [1, 0],
        "cat__gender_Male": [0, 1],
        "cat__gender_Other": [0, 0],
        "cat__age_group_18-35": [1, 0],
        "cat__age_group_51-65": [0, 1],
    })

    reconstructed = reconstruct_demographic_features(df)

    assert "gender" in reconstructed.columns
    assert "age_group" in reconstructed.columns
    assert reconstructed["gender"].tolist() == ["Female", "Male"]
    assert reconstructed["age_group"].tolist() == ["18-35", "51-65"]


def test_subgroup_metrics_normal():
    """Verify metrics calculation for a well-supported demographic subgroup."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y_pred = np.array([0, 0, 0, 1, 1])

    metrics = compute_subgroup_metrics(
        y_true, y_prob, y_pred, subgroup_name="gender", slice_label="Female", min_samples=3,
    )

    assert metrics["support"] == 5
    assert metrics["positives"] == 2
    assert metrics["negatives"] == 3
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["f1"] == 1.0
    assert metrics["sample_size_warning"] is False
    assert metrics["warning_message"] is None


def test_subgroup_metrics_small_cohort_handling():
    """Verify small cohort (e.g. Other N=2) handles metric edge cases and flags sample warning."""
    y_true = np.array([0, 0])
    y_prob = np.array([0.1, 0.2])
    y_pred = np.array([0, 0])

    metrics = compute_subgroup_metrics(
        y_true, y_prob, y_pred, subgroup_name="gender", slice_label="Other", min_samples=30,
    )

    assert metrics["support"] == 2
    assert metrics["positives"] == 0
    assert metrics["negatives"] == 2
    assert metrics["sample_size_warning"] is True
    assert "small sample size" in metrics["warning_message"]
    # Recall is None when there are no positive examples
    assert metrics["recall"] is None
    # FPR is 0.0 since FP=0 and TN=2
    assert metrics["false_positive_rate"] == 0.0


def test_fairness_audit_all_categories_retained():
    """Verify that fairness analysis retains all demographic categories without dropping any."""
    df = pd.DataFrame({
        "gender": ["Female"] * 40 + ["Male"] * 40 + ["Other"] * 2,
        "age_group": ["<18"] * 20 + ["18-35"] * 20 + ["36-50"] * 20 + ["51-65"] * 12 + ["65+"] * 10,
    })
    y_true = np.random.binomial(1, 0.1, size=len(df))
    y_prob = np.random.uniform(0, 1, size=len(df))
    y_pred = (y_prob >= 0.5).astype(int)

    results = evaluate_fairness(df, y_true, y_prob, y_pred, demographic_cols=["gender", "age_group"])

    gender_slices = results["fairness_audit"]["gender"]["slices"]
    expected_genders = {"Female", "Male", "Other"}
    assert set(gender_slices.keys()) == expected_genders
    # Verify 'Other' is present and has sample size warning
    assert gender_slices["Other"]["sample_size_warning"] is True
    assert gender_slices["Female"]["sample_size_warning"] is False

    age_slices = results["fairness_audit"]["age_group"]["slices"]
    expected_ages = {"<18", "18-35", "36-50", "51-65", "65+"}
    assert set(age_slices.keys()) == expected_ages


# ==========================================
# 3. Integration Tests with Champion Model
# ==========================================

def test_dynamic_champion_model_loading():
    """Verify that load_champion_model dynamically loads the active champion from MLflow."""
    model = load_champion_model(CHAMPION_URI)
    assert model is not None

    # Verify predictions on mock input
    mock_x = np.zeros((2, 27))
    probs, preds = get_predictions(model, mock_x)

    assert len(probs) == 2
    assert len(preds) == 2
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})


def test_run_evaluation_pipeline(tmp_path):
    """Integration test verifying run_evaluation executes on processed test data."""
    test_data_path = Path("data/processed/test.parquet")
    if not test_data_path.exists():
        pytest.skip("data/processed/test.parquet does not exist.")

    report = run_evaluation(
        processed_test_path=test_data_path,
        model_uri=CHAMPION_URI,
        export_report=False,
    )

    assert report["registered_model_name"] == "DiabetesRiskModel"
    assert report["overall_performance"]["pr_auc"] > 0.50
    assert report["calibration"]["brier_score"] < 0.20
    assert "gender" in report["fairness"]["fairness_audit"]
    assert "age_group" in report["fairness"]["fairness_audit"]
