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

# 7. Persistent gauge tracking total stored assessments in database
total_stored_assessments = Gauge(
    "total_stored_assessments",
    "Total persistent clinical assessments stored in the database",
)


def sync_metrics_from_db(db_session_factory=None) -> None:
    """Sync Prometheus metrics with persistent database assessment counts on startup."""
    import logging
    from sqlalchemy import func
    from app.db.models import AssessmentHistory
    from app.db.session import SessionLocal

    metrics_logger = logging.getLogger(__name__)

    session_creator = db_session_factory or SessionLocal
    db = session_creator()
    try:
        total = db.query(AssessmentHistory).count()
        total_stored_assessments.set(total)
        metrics_logger.info("Initialized total_stored_assessments gauge to %d", total)

        # Retrieve active model version if available, defaulting to '2'
        active_version = "2"
        try:
            from app.services.predictor import predictor_service
            active_version = getattr(predictor_service, "model_version", "2") or "2"
        except Exception:
            pass

        # Query counts grouped by model_version and risk_category
        grouped_counts = (
            db.query(
                AssessmentHistory.model_version,
                AssessmentHistory.risk_category,
                func.count(AssessmentHistory.id),
            )
            .group_by(AssessmentHistory.model_version, AssessmentHistory.risk_category)
            .all()
        )

        seen_labels = set()
        for m_ver, r_cat, count in grouped_counts:
            ver = str(m_ver) if m_ver else active_version
            cat = str(r_cat)
            seen_labels.add((ver, cat))
            if count > 0:
                prediction_requests_total.labels(
                    model_version=ver,
                    risk_category=cat,
                ).inc(count)
                metrics_logger.info(
                    "Seeded prediction_requests_total(model_version=%s, risk_category=%s) with %d",
                    ver,
                    cat,
                    count,
                )

        # Initialize default risk tiers with 0 for the active version if not already present
        for cat in ["Low", "Moderate", "High"]:
            if (active_version, cat) not in seen_labels:
                prediction_requests_total.labels(
                    model_version=active_version,
                    risk_category=cat,
                ).inc(0)

    except Exception as exc:
        metrics_logger.warning("Could not sync metrics with database counts: %s", exc)
    finally:
        db.close()


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
    "sync_metrics_from_db",
    "total_stored_assessments",
]
