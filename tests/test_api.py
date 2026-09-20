"""Integration tests for FastAPI REST API service using TestClient.

Tests:
- GET /health: 200 OK and valid health payload.
- POST /predict: 200 OK, valid probability, integer risk score, and risk category.
- POST /predict: 422 Unprocessable Entity on corrupted inputs (age, BMI, HbA1c, glucose, hypertension).
- POST /explain: 200 OK with non-empty SHAP feature attributions.
- POST /what-if: 200 OK with probability delta and mandatory clinical disclaimer.
- GET /model-info: 200 OK with active champion model governance metadata.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.features.counterfactual_dice import MANDATORY_DISCLAIMER

client = TestClient(app)


@pytest.fixture
def valid_patient_payload():
    """Valid clinical profile for an elevated-risk patient."""
    return {
        "gender": "Female",
        "age": 55.0,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "former",
        "bmi": 32.5,
        "HbA1c_level": 7.2,
        "blood_glucose_level": 180.0,
    }


@pytest.fixture
def healthy_patient_payload():
    """Valid clinical profile for a low-risk patient."""
    return {
        "gender": "Male",
        "age": 28.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 22.0,
        "HbA1c_level": 5.1,
        "blood_glucose_level": 85.0,
    }


# ==========================================
# 1. Health & Info Endpoints
# ==========================================

def test_health_endpoint():
    """Verify GET /health returns HTTP 200 with healthy state and model info."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "diabetes-mlops-api"
    assert "model_name" in data
    assert "model_version" in data
    assert "timestamp" in data


def test_model_info_endpoint():
    """Verify GET /model-info returns HTTP 200 with governance and architecture metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200

    data = response.json()
    assert data["model_name"] == "DiabetesRiskModel"
    assert "@champion" in data["alias"]
    assert "XGBoost" in data["algorithm"]
    assert "status" in data


# ==========================================
# 2. Prediction Endpoint Tests
# ==========================================

def test_predict_endpoint_high_risk(valid_patient_payload):
    """Verify POST /predict returns valid 200 response with high risk score."""
    response = client.post("/predict", json=valid_patient_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_category"] in ["Low", "Moderate", "High"]
    assert data["risk_score"] == round(data["probability"] * 100)
    assert data["model_name"] == "DiabetesRiskModel"
    assert "model_version" in data


def test_predict_endpoint_low_risk(healthy_patient_payload):
    """Verify POST /predict handles healthy patient and classifies as Low risk."""
    response = client.post("/predict", json=healthy_patient_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["risk_score"] < 30
    assert data["risk_category"] == "Low"
    assert data["prediction"] == 0


@pytest.mark.parametrize(
    "corrupted_field,invalid_value",
    [
        ("age", -5.0),
        ("age", 135.0),
        ("gender", "InvalidSex"),
        ("hypertension", 2),
        ("heart_disease", -1),
        ("smoking_history", "vaping"),
        ("bmi", 5.0),
        ("bmi", 120.0),
        ("HbA1c_level", 2.0),
        ("HbA1c_level", 25.0),
        ("blood_glucose_level", 10.0),
        ("blood_glucose_level", 650.0),
    ],
)
def test_predict_validation_errors(valid_patient_payload, corrupted_field, invalid_value):
    """Verify that physiological boundary breaches return HTTP 422 with structured details."""
    payload = valid_patient_payload.copy()
    payload[corrupted_field] = invalid_value

    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    data = response.json()
    assert "details" in data or "detail" in data


# ==========================================
# 3. Explainability & What-If Endpoints
# ==========================================

def test_explain_endpoint(valid_patient_payload):
    """Verify POST /explain returns HTTP 200 with complete SHAP attributions."""
    response = client.post("/explain", json=valid_patient_payload)
    assert response.status_code == 200

    data = response.json()
    assert "base_value" in data
    assert "predicted_margin" in data
    assert "predicted_probability" in data
    assert len(data["attributions"]) == 27
    assert len(data["top_risk_increasing_factors"]) > 0
    assert len(data["top_risk_decreasing_factors"]) > 0

    first_attr = data["attributions"][0]
    assert "feature" in first_attr
    assert "attribution_value" in first_attr
    assert first_attr["direction"] in ["increases_risk", "decreases_risk"]


def test_what_if_endpoint_automated_counterfactual(valid_patient_payload):
    """Verify POST /what-if returns HTTP 200 with probability delta and disclaimer."""
    request_payload = {
        "patient": valid_patient_payload,
        "target_risk_threshold": 0.50,
    }

    response = client.post("/what-if", json=request_payload)
    assert response.status_code == 200

    data = response.json()
    assert "original_probability" in data
    assert "counterfactual_probability" in data
    assert data["counterfactual_probability"] < data["original_probability"]
    assert data["probability_delta"] < 0.0
    assert data["non_modifiable_preserved"] is True
    assert "feature_changes" in data
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_what_if_endpoint_manual_overrides(valid_patient_payload):
    """Verify POST /what-if processes manual lifestyle overrides."""
    request_payload = {
        "patient": valid_patient_payload,
        "overrides": {
            "bmi": 24.0,
            "blood_glucose_level": 95.0,
            "HbA1c_level": 5.4,
        },
    }

    response = client.post("/what-if", json=request_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["counterfactual_probability"] < data["original_probability"]
    assert data["probability_delta"] < 0.0
    assert "bmi" in data["feature_changes"]
    assert data["disclaimer"] == MANDATORY_DISCLAIMER
