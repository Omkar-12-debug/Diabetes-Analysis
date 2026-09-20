"""Unit and integration tests for Model Quality Gate and Promotion Engine.

Tests:
- All thresholds passing triggers PASSED decision and executes promotion.
- Sub-threshold metrics (PR-AUC, Recall, Brier) trigger FAILED decision and halt promotion.
- Fairness disparity breach (|Recall_Male - Recall_Female| > 0.08) triggers gate failure.
- Missing or corrupted artifacts trigger artifact integrity failure.
- End-to-end integration test on active champion model verifying reports/quality_gate_report.json.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.quality_gate import (
    MAX_BRIER_SCORE,
    MAX_GENDER_RECALL_DISPARITY,
    MIN_TEST_PR_AUC,
    MIN_TEST_RECALL,
    QualityGateController,
    evaluate_gates,
    verify_artifacts,
)


@pytest.fixture
def passing_inputs():
    """Ideal metrics dictionary meeting all quality gate standards."""
    candidate_metrics = {
        "pr_auc": 0.8850,
        "recall": 0.9050,
        "brier_score": 0.0550,
    }
    baseline_pr_auc = 0.8124
    fairness_audit = {
        "gender": {
            "slices": {
                "Male": {"recall": 0.9100},
                "Female": {"recall": 0.8900},
            }
        }
    }
    artifact_status = {"valid": True}
    return candidate_metrics, baseline_pr_auc, fairness_audit, artifact_status


def test_quality_gate_all_pass(passing_inputs):
    """Verify that candidate satisfying all criteria achieves PASSED decision."""
    metrics, baseline, fairness, artifacts = passing_inputs
    result = evaluate_gates(metrics, baseline, fairness, artifacts)

    assert result["overall_decision"] == "PASSED"
    assert len(result["failure_reasons"]) == 0
    for gate_name, gate_info in result["gates"].items():
        assert gate_info["status"] == "PASSED", f"Gate {gate_name} unexpectedly failed"


def test_quality_gate_pr_auc_degradation_fails(passing_inputs):
    """Verify that PR-AUC below 0.85 or below baseline triggers FAILED decision."""
    metrics, baseline, fairness, artifacts = passing_inputs

    # Case A: Below 0.85 absolute threshold
    low_pr_metrics = {**metrics, "pr_auc": 0.7200}
    result_a = evaluate_gates(low_pr_metrics, baseline, fairness, artifacts)
    assert result_a["overall_decision"] == "FAILED"
    assert result_a["gates"]["primary_pr_auc"]["status"] == "FAILED"
    assert any("PR-AUC" in r for r in result_a["failure_reasons"])

    # Case B: Below baseline even if relatively high
    high_baseline = 0.8900
    result_b = evaluate_gates(metrics, high_baseline, fairness, artifacts)
    assert result_b["overall_decision"] == "FAILED"
    assert result_b["gates"]["primary_pr_auc"]["status"] == "FAILED"


def test_quality_gate_clinical_recall_failure(passing_inputs):
    """Verify that sensitivity below 0.88 triggers FAILED decision."""
    metrics, baseline, fairness, artifacts = passing_inputs
    low_recall_metrics = {**metrics, "recall": 0.8200}

    result = evaluate_gates(low_recall_metrics, baseline, fairness, artifacts)
    assert result["overall_decision"] == "FAILED"
    assert result["gates"]["clinical_recall"]["status"] == "FAILED"
    assert any("Recall" in r for r in result["failure_reasons"])


def test_quality_gate_brier_calibration_failure(passing_inputs):
    """Verify that poorly calibrated model (Brier > 0.10) triggers FAILED decision."""
    metrics, baseline, fairness, artifacts = passing_inputs
    poor_calibration_metrics = {**metrics, "brier_score": 0.1450}

    result = evaluate_gates(poor_calibration_metrics, baseline, fairness, artifacts)
    assert result["overall_decision"] == "FAILED"
    assert result["gates"]["calibration_brier"]["status"] == "FAILED"
    assert any("Brier" in r for r in result["failure_reasons"])


def test_quality_gate_fairness_disparity_failure(passing_inputs):
    """Verify that gender recall disparity exceeding 0.08 triggers FAILED decision."""
    metrics, baseline, _, artifacts = passing_inputs
    disparate_fairness = {
        "gender": {
            "slices": {
                "Male": {"recall": 0.9400},
                "Female": {"recall": 0.8400},  # Disparity = 0.10 > 0.08
            }
        }
    }

    result = evaluate_gates(metrics, baseline, disparate_fairness, artifacts)
    assert result["overall_decision"] == "FAILED"
    assert result["gates"]["fairness_disparity"]["status"] == "FAILED"
    assert any("Fairness" in r or "disparity" in r.lower() for r in result["failure_reasons"])


def test_artifact_integrity_verification_missing_file(tmp_path):
    """Verify that missing preprocessor or unexecutable model triggers artifact failure."""
    non_existent_path = tmp_path / "non_existent_preprocessor.joblib"
    fake_model = MagicMock()

    status = verify_artifacts(fake_model, preprocessor_path=non_existent_path)
    assert status["valid"] is False
    assert status["preprocessor_exists"] is False

    # Corrupted / None model
    real_preprocessor = Path("models/preprocessor.joblib")
    status_no_model = verify_artifacts(None, preprocessor_path=real_preprocessor)
    assert status_no_model["valid"] is False
    assert status_no_model["model_executable"] is False


def test_controller_promotion_action():
    """Verify that controller executes client.set_registered_model_alias upon promotion."""
    controller = QualityGateController()
    mock_client = MagicMock()
    controller.client = mock_client

    controller.promote_candidate("3")

    mock_client.set_registered_model_alias.assert_called_once_with(
        name=controller.registered_model_name,
        alias=controller.champion_alias,
        version="3",
    )
    assert mock_client.set_model_version_tag.call_count == 2


def test_end_to_end_quality_gate_execution(tmp_path):
    """Integration test: Execute run_gate on current champion and test dataset."""
    controller = QualityGateController()
    report_file = tmp_path / "test_quality_gate_report.json"

    result = controller.run_gate(
        auto_promote=False,  # Don't mutate registry during unit test
        export_report=True,
        report_path=report_file,
    )

    assert result["overall_decision"] == "PASSED"
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["overall_decision"] == "PASSED"
    assert "gates" in saved_data["gate_details"] or "primary_pr_auc" in saved_data["gate_details"]
    assert saved_data["candidate_metrics"]["pr_auc"] >= MIN_TEST_PR_AUC
    assert saved_data["candidate_metrics"]["recall"] >= MIN_TEST_RECALL
    assert saved_data["candidate_metrics"]["brier_score"] <= MAX_BRIER_SCORE
