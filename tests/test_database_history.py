"""Integration tests for database persistence, patient assessment history, and delayed-label tracking.

Tests:
- Automatic assessment persistence on /predict calls (custom patient_id and auto-generated UUID).
- Longitudinal assessment trajectory retrieval via GET /history/{patient_id}.
- Empty history handling for non-existent patients.
- Delayed ground-truth clinical outcome binding via POST /feedback/ground-truth.
- Delayed ground-truth updates on previously recorded assessments.
- 404 response on ground-truth binding for non-existent assessment IDs.
- 422 validation rejection on invalid ground-truth labels (non 0/1).
- Global history retrieval with pagination via GET /history.
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from app.db.session import init_db
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database tables are initialized before tests."""
    init_db()


@pytest.fixture
def sample_patient_payload():
    """Valid clinical profile for test patients."""
    return {
        "gender": "Female",
        "age": 52.0,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "former",
        "bmi": 31.0,
        "HbA1c_level": 6.8,
        "blood_glucose_level": 160.0,
    }


def test_predict_persists_assessment_with_custom_patient_id(sample_patient_payload):
    """Verify calling /predict with explicit patient_id logs the assessment into the database."""
    patient_id = f"test-patient-{uuid.uuid4().hex[:8]}"
    payload = {**sample_patient_payload, "patient_id": patient_id}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] == patient_id
    assert "assessment_id" in data
    assert data["assessment_id"] is not None
    assert isinstance(data["assessment_id"], int)

    # Verify queryable via history endpoint
    hist_resp = client.get(f"/history/{patient_id}")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["patient_id"] == patient_id
    assert hist_data["total_records"] == 1
    assert hist_data["assessments"][0]["id"] == data["assessment_id"]
    assert hist_data["assessments"][0]["risk_score"] == data["risk_score"]
    assert hist_data["assessments"][0]["risk_category"] == data["risk_category"]


def test_predict_auto_generates_uuid_when_omitted(sample_patient_payload):
    """Verify calling /predict without patient_id auto-assigns a valid UUID and persists."""
    response = client.post("/predict", json=sample_patient_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] is not None
    assert len(data["patient_id"]) > 10  # Valid UUID format
    assert data["assessment_id"] is not None

    # Retrieve via auto-generated patient ID
    hist_resp = client.get(f"/history/{data['patient_id']}")
    assert hist_resp.status_code == 200
    assert hist_resp.json()["total_records"] == 1


def test_patient_history_chronological_trajectory(sample_patient_payload):
    """Verify multiple assessments for the same patient are ordered chronologically."""
    patient_id = f"trajectory-{uuid.uuid4().hex[:8]}"

    # Initial assessment: Elevated risk
    p1 = {**sample_patient_payload, "patient_id": patient_id, "bmi": 34.0, "blood_glucose_level": 180.0}
    r1 = client.post("/predict", json=p1)
    assert r1.status_code == 200
    id1 = r1.json()["assessment_id"]

    # Second assessment: Moderate reduction
    p2 = {**sample_patient_payload, "patient_id": patient_id, "bmi": 29.0, "blood_glucose_level": 140.0}
    r2 = client.post("/predict", json=p2)
    assert r2.status_code == 200
    id2 = r2.json()["assessment_id"]

    # Third assessment: Significant lifestyle improvement
    p3 = {**sample_patient_payload, "patient_id": patient_id, "bmi": 24.5, "blood_glucose_level": 95.0}
    r3 = client.post("/predict", json=p3)
    assert r3.status_code == 200
    id3 = r3.json()["assessment_id"]

    # Fetch longitudinal trajectory
    hist_resp = client.get(f"/history/{patient_id}")
    assert hist_resp.status_code == 200
    history = hist_resp.json()

    assert history["total_records"] == 3
    assessments = history["assessments"]
    returned_ids = [a["id"] for a in assessments]
    assert returned_ids == [id1, id2, id3], "Assessments must be strictly ordered chronologically"


def test_nonexistent_patient_history_returns_empty_list():
    """Verify requesting history for non-existent patient returns empty list with HTTP 200."""
    nonexistent_id = f"unknown-{uuid.uuid4().hex}"
    response = client.get(f"/history/{nonexistent_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == nonexistent_id
    assert data["total_records"] == 0
    assert data["assessments"] == []


def test_record_ground_truth_label(sample_patient_payload):
    """Verify recording delayed clinical ground-truth feedback for an assessment."""
    # Create assessment
    patient_id = f"feedback-{uuid.uuid4().hex[:8]}"
    pred_resp = client.post("/predict", json={**sample_patient_payload, "patient_id": patient_id})
    assert pred_resp.status_code == 200
    assessment_id = pred_resp.json()["assessment_id"]

    # Submit ground-truth confirmation
    feedback_payload = {
        "assessment_id": assessment_id,
        "verified_diabetes_label": 1,
        "notes": "Oral glucose tolerance test (OGTT) 2-hour post-load glucose >= 200 mg/dL.",
    }
    fb_resp = client.post("/feedback/ground-truth", json=feedback_payload)
    assert fb_resp.status_code == 200
    fb_data = fb_resp.json()

    assert fb_data["assessment_id"] == assessment_id
    assert fb_data["patient_id"] == patient_id
    assert fb_data["verified_diabetes_label"] == 1
    assert "OGTT" in fb_data["notes"]
    assert "verified_at" in fb_data

    # Verify reflected in patient history
    hist_resp = client.get(f"/history/{patient_id}")
    assert hist_resp.status_code == 200
    record = hist_resp.json()["assessments"][0]
    assert record["verified_diabetes_label"] == 1
    assert record["verified_at"] is not None
    assert "OGTT" in record["notes"]


def test_update_existing_ground_truth_label(sample_patient_payload):
    """Verify updating a previously submitted ground-truth label updates cleanly."""
    pred_resp = client.post("/predict", json=sample_patient_payload)
    assessment_id = pred_resp.json()["assessment_id"]

    # First label
    client.post("/feedback/ground-truth", json={"assessment_id": assessment_id, "verified_diabetes_label": 1})
    # Updated label
    update_resp = client.post(
        "/feedback/ground-truth",
        json={
            "assessment_id": assessment_id,
            "verified_diabetes_label": 0,
            "notes": "Secondary laboratory re-test determined previous result was pre-diabetic.",
        },
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["verified_diabetes_label"] == 0
    assert "Secondary laboratory re-test" in update_resp.json()["notes"]


def test_ground_truth_nonexistent_assessment_returns_404():
    """Verify submitting ground truth for a nonexistent assessment returns 404."""
    response = client.post(
        "/feedback/ground-truth",
        json={"assessment_id": 99999999, "verified_diabetes_label": 1},
    )
    assert response.status_code == 404
    assert "not exist" in response.json()["detail"].lower()


def test_delayed_label_validation_error():
    """Verify submitting invalid verified_diabetes_label returns HTTP 422."""
    response = client.post(
        "/feedback/ground-truth",
        json={"assessment_id": 1, "verified_diabetes_label": 5},
    )
    assert response.status_code == 422


def test_global_history_pagination(sample_patient_payload):
    """Verify GET /history returns paginated assessments."""
    # Ensure at least 3 records exist
    for _ in range(3):
        client.post("/predict", json=sample_patient_payload)

    response = client.get("/history?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] >= 3
    assert len(data["assessments"]) == 2
