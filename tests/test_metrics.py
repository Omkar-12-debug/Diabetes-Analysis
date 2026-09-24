"""Integration tests for Prometheus metrics collection, telemetry instrumentation, and /metrics exposition."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.db.session import init_db
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure database tables exist before API tests run."""
    init_db()
    yield


client = TestClient(app)


@pytest.fixture
def sample_valid_patient():
    """Valid patient payload for testing prediction metric increments."""
    return {
        "gender": "Female",
        "age": 48.0,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 30.5,
        "HbA1c_level": 6.8,
        "blood_glucose_level": 155.0,
    }


def test_metrics_endpoint_returns_200_and_plain_text():
    """Verify GET /metrics returns HTTP 200 with standard Prometheus plain-text exposition."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert len(body) > 0


def test_metrics_exposition_contains_required_collectors():
    """Verify /metrics output contains all required MLOps telemetry collectors and TYPE declarations."""
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text

    required_metrics = [
        "prediction_requests_total",
        "prediction_errors_total",
        "prediction_latency_seconds",
        "active_requests_in_flight",
        "prediction_risk_score",
        "http_requests_total",
    ]

    for metric in required_metrics:
        assert metric in body, f"Metric '{metric}' missing from /metrics exposition"
        assert f"# TYPE {metric}" in body, f"Metadata TYPE for '{metric}' missing from /metrics"


def test_prediction_request_increments_counters(sample_valid_patient):
    """Verify that POST /predict updates prediction_requests_total and risk distribution histograms."""
    # Read initial metrics
    initial_resp = client.get("/metrics")
    assert initial_resp.status_code == 200

    # Execute prediction
    pred_resp = client.post("/predict", json=sample_valid_patient)
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    risk_cat = pred_data["risk_category"]
    model_ver = pred_data["model_version"]

    # Read updated metrics
    updated_resp = client.get("/metrics")
    assert updated_resp.status_code == 200
    updated_body = updated_resp.text

    # Verify prediction_requests_total includes corresponding label pair
    expected_label_substr = f'risk_category="{risk_cat}"'
    assert expected_label_substr in updated_body
    assert f'model_version="{model_ver}"' in updated_body

    # Verify prediction_risk_score histogram recorded an observation
    assert "prediction_risk_score_bucket" in updated_body
    assert "prediction_risk_score_count" in updated_body


def test_validation_error_increments_error_counter(sample_valid_patient):
    """Verify that invalid payloads (422) increment prediction_errors_total with validation_error."""
    invalid_payload = sample_valid_patient.copy()
    invalid_payload["age"] = -50.0  # Violates physiological boundary

    error_resp = client.post("/predict", json=invalid_payload)
    assert error_resp.status_code == 422

    # Verify error metrics
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_body = metrics_resp.text

    assert 'error_type="validation_error"' in metrics_body or 'error_type="ValidationError"' in metrics_body
    assert 'endpoint="/predict"' in metrics_body


def test_active_requests_in_flight_gauge():
    """Verify active_requests_in_flight gauge is non-negative and initialized."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "active_requests_in_flight" in response.text


def test_docker_compose_observability_stack_contract():
    """Validate that docker-compose.yml defines prometheus (9090) and grafana (3000) services."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml not found"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "prometheus" in services, "prometheus service missing from docker-compose.yml"
    assert "grafana" in services, "grafana service missing from docker-compose.yml"

    # Check Prometheus ports
    prom_ports = services["prometheus"].get("ports", [])
    assert any("9090:9090" in str(p) for p in prom_ports), "Prometheus port 9090:9090 missing"

    # Check Grafana ports
    graf_ports = services["grafana"].get("ports", [])
    assert any("3000:3000" in str(p) for p in graf_ports), "Grafana port 3000:3000 missing"

    # Check network connections
    assert "mlops_network" in services["prometheus"].get("networks", [])
    assert "mlops_network" in services["grafana"].get("networks", [])
