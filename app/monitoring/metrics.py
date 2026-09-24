"""Prometheus metrics collectors for clinical inference and API observability."""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# 1. Total prediction requests stratified by model version and risk tier
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of diabetes risk prediction requests processed",
    ["model_version", "risk_category"],
)

# 2. Total errors categorized by endpoint and error type
prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total prediction and API error counts categorized by endpoint and error type",
    ["endpoint", "error_type"],
)

# 3. Request and inference latency distribution in seconds
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Inference and request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# 4. Concurrently processing requests currently in flight
active_requests_in_flight = Gauge(
    "active_requests_in_flight",
    "Number of requests currently in flight across API endpoints",
)

# 5. Continuous diabetes risk score (0-100) distribution
prediction_risk_score = Histogram(
    "prediction_risk_score",
    "Distribution of calibrated diabetes risk scores (0-100)",
    buckets=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
)

# 6. General HTTP requests counter for throughput and error rate tracking
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests received by method, endpoint, and HTTP status code",
    ["method", "endpoint", "status_code"],
)


def get_latest_metrics() -> bytes:
    """Collect and format all registered Prometheus metrics into plain-text exposition."""
    return generate_latest()


__all__ = [
    "CONTENT_TYPE_LATEST",
    "active_requests_in_flight",
    "get_latest_metrics",
    "http_requests_total",
    "prediction_errors_total",
    "prediction_latency_seconds",
    "prediction_requests_total",
    "prediction_risk_score",
]
