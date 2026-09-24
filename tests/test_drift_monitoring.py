"""Integration and contract test suite for data/prediction drift and delayed-label monitoring."""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.models import AssessmentHistory, GroundTruthLabel
from app.db.session import SessionLocal, init_db
from app.main import app
from src.monitoring.drift import (
    DRIFT_REPORT_HTML_PATH,
    DRIFT_REPORT_JSON_PATH,
    compute_data_drift,
    generate_drifted_batch,
    generate_in_distribution_batch,
    load_reference_data,
    run_drift_pipeline,
)
from src.monitoring.performance import compute_production_performance


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure database schema is created prior to test executions."""
    init_db()
    yield


client = TestClient(app)


@pytest.fixture
def reference_dataset():
    """Provide reference patient cohort for drift evaluations."""
    return load_reference_data(sample_size=1000, random_state=42)


# ==========================================
# 1. Statistical Drift Engine Tests
# ==========================================

def test_in_distribution_batch_stability(reference_dataset):
    """Verify that in-distribution cohorts do not trigger dataset drift alerts (p > 0.05)."""
    in_dist_batch = generate_in_distribution_batch(
        reference_df=reference_dataset,
        sample_size=500,
        random_state=123,
    )

    summary, _ = compute_data_drift(
        reference_df=reference_dataset,
        current_df=in_dist_batch,
        drift_share_threshold=0.30,
    )

    assert summary["dataset_drift_detected"] is False
    assert summary["drift_share"] < 0.30
    assert summary["drifted_features_count"] < 3

    # Primary biomarkers must remain statistically stable
    features = summary["features"]
    assert features["blood_glucose_level"]["p_value"] > 0.05
    assert features["blood_glucose_level"]["drift_detected"] is False
    assert features["HbA1c_level"]["p_value"] > 0.05
    assert features["HbA1c_level"]["drift_detected"] is False


def test_clinical_biomarker_drift_detection(reference_dataset):
    """Verify that clinical perturbations (+35 glucose, +1.2 HbA1c, +10 age) trigger drift detection."""
    drifted_batch = generate_drifted_batch(
        reference_df=reference_dataset,
        sample_size=500,
        glucose_shift=35.0,
        hba1c_shift=1.2,
        age_shift=10.0,
        random_state=999,
    )

    summary, _ = compute_data_drift(
        reference_df=reference_dataset,
        current_df=drifted_batch,
        drift_share_threshold=0.30,
    )

    assert summary["dataset_drift_detected"] is True
    assert summary["drift_share"] >= 0.30

    features = summary["features"]
    # Check that each perturbed feature is explicitly flagged
    for drifted_col in ["blood_glucose_level", "HbA1c_level", "age"]:
        assert drifted_col in features, f"Feature '{drifted_col}' missing from features drift map"
        assert features[drifted_col]["drift_detected"] is True, f"Expected drift on {drifted_col}"
        assert features[drifted_col]["p_value"] < 0.05, f"Expected p < 0.05 for {drifted_col}"


def test_drift_pipeline_report_exports():
    """Verify that run_drift_pipeline exports both JSON and HTML report artifacts."""
    summary = run_drift_pipeline(
        batch_type="drifted",
        sample_size=300,
        export_json=True,
        export_html=True,
    )

    assert "dataset_drift_detected" in summary
    assert "features" in summary
    assert DRIFT_REPORT_JSON_PATH.exists()
    assert DRIFT_REPORT_HTML_PATH.exists()

    # Validate HTML artifact contains HTML elements
    html_content = Path(DRIFT_REPORT_HTML_PATH).read_text(encoding="utf-8")
    assert "<html" in html_content.lower()
    assert len(html_content) > 1000


# ==========================================
# 2. Delayed-Label Performance Tests
# ==========================================

def test_delayed_label_performance_empty():
    """Verify performance evaluation handles empty labeled data gracefully."""
    db = SessionLocal()
    try:
        # Evaluate with an impossible model version to guarantee no matches
        result = compute_production_performance(
            db=db,
            model_version=f"nonexistent-version-{uuid.uuid4().hex[:8]}",
        )
        assert result["status"] == "insufficient_data"
        assert result["total_evaluated"] == 0
        assert result["metrics"]["pr_auc"] is None
        assert result["metrics"]["brier_score"] is None
    finally:
        db.close()


def test_delayed_label_performance_with_verified_records():
    """Verify performance metrics calculation with verified ground-truth labels."""
    db = SessionLocal()
    try:
        test_version = f"test-eval-{uuid.uuid4().hex[:6]}"

        # Seed 4 assessment records (2 positive, 2 negative)
        cases = [
            (0.85, 85, "High", 1),
            (0.75, 75, "High", 1),
            (0.15, 15, "Low", 0),
            (0.25, 25, "Low", 0),
        ]

        for prob, score, cat, true_label in cases:
            assessment = AssessmentHistory(
                patient_id=f"test-p-{uuid.uuid4().hex[:6]}",
                input_features={"age": 50.0, "bmi": 28.0},
                risk_probability=prob,
                risk_score=score,
                risk_category=cat,
                model_version=test_version,
            )
            db.add(assessment)
            db.flush()

            label = GroundTruthLabel(
                assessment_id=assessment.id,
                patient_id=assessment.patient_id,
                verified_diabetes_label=true_label,
            )
            db.add(label)

        db.commit()

        # Compute production performance
        result = compute_production_performance(db=db, model_version=test_version)

        assert result["status"] == "evaluated"
        assert result["total_evaluated"] == 4
        assert result["positive_cases"] == 2
        assert result["negative_cases"] == 2
        assert result["metrics"]["pr_auc"] is not None
        assert result["metrics"]["roc_auc"] is not None
        assert result["metrics"]["brier_score"] is not None
        assert result["metrics"]["precision"] == 1.0
        assert result["metrics"]["recall"] == 1.0
        assert result["metrics"]["f1_score"] == 1.0
        assert 0.0 <= result["metrics"]["brier_score"] <= 1.0
        assert result["calibration_status"] in ["well_calibrated", "poorly_calibrated"]
    finally:
        db.close()


# ==========================================
# 3. Monitoring FastAPI Route Tests
# ==========================================

def test_get_drift_report_endpoint():
    """Verify GET /monitoring/drift/report returns 200 with structured summary and HTML path."""
    response = client.get("/monitoring/drift/report")
    assert response.status_code == 200

    data = response.json()
    assert "dataset_drift_detected" in data
    assert "drift_share" in data
    assert "features" in data
    assert "html_report_path" in data


def test_simulate_drift_endpoint_drifted():
    """Verify POST /monitoring/drift/simulate with drifted batch flags dataset drift."""
    payload = {
        "batch_type": "drifted",
        "sample_size": 250,
        "glucose_shift": 40.0,
        "hba1c_shift": 1.5,
        "age_shift": 12.0,
    }
    response = client.post("/monitoring/drift/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["batch_type"] == "drifted"
    assert data["dataset_drift_detected"] is True
    assert data["features"]["blood_glucose_level"]["drift_detected"] is True


def test_simulate_drift_endpoint_in_distribution():
    """Verify POST /monitoring/drift/simulate with in_distribution batch reports no drift."""
    payload = {
        "batch_type": "in_distribution",
        "sample_size": 250,
    }
    response = client.post("/monitoring/drift/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["batch_type"] == "in_distribution"
    assert data["dataset_drift_detected"] is False


def test_get_performance_endpoint():
    """Verify GET /monitoring/performance returns 200 with performance schema."""
    response = client.get("/monitoring/performance")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "total_evaluated" in data
    assert "metrics" in data
    assert "brier_score" in data["metrics"]
