"""Clinical inference routes for diabetes risk prediction."""

import time
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.monitoring.cloudwatch import log_prediction_cloudwatch
from app.monitoring.metrics import (
    prediction_errors_total,
    prediction_requests_total,
    prediction_risk_score,
)
from app.schemas import PatientInput, PredictionResponse
from app.services.history_service import history_service
from app.services.predictor import predictor_service

router = APIRouter(tags=["Clinical Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Diabetes Risk",
    description=(
        "Ingests clinical and biometric patient features, executes preprocessor transformations, "
        "persists the assessment event into history, and outputs a 0–100 risk score and risk tier."
    ),
)
def predict_risk(
    patient: PatientInput,
    db: Session = Depends(get_db),  # noqa: B008
) -> PredictionResponse:
    """Predict diabetes onset risk for an individual patient profile and log assessment."""
    # Ensure patient has an identifier
    if not patient.patient_id:
        patient.patient_id = str(uuid.uuid4())

    # Execute ML inference
    start_time = time.perf_counter()
    try:
        response = predictor_service.predict(patient)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        prediction_requests_total.labels(
            model_version=response.model_version,
            risk_category=response.risk_category,
        ).inc()
        prediction_risk_score.observe(float(response.risk_score))

        # AWS CloudWatch Telemetry (graceful console fallback when credentials absent)
        log_prediction_cloudwatch(
            payload=patient.model_dump(),
            risk_score=float(response.risk_score),
            latency=latency_ms,
        )
    except Exception as exc:
        prediction_errors_total.labels(
            endpoint="/predict",
            error_type=type(exc).__name__,
        ).inc()
        raise

    # Persist assessment record in database
    input_features = {k: v for k, v in patient.model_dump().items() if k != "patient_id"}
    db_record = history_service.log_assessment(
        db=db,
        patient_id=patient.patient_id,
        input_features=input_features,
        prediction_response=response,
    )

    # Attach database identification to response
    response.patient_id = patient.patient_id
    response.assessment_id = db_record.id

    return response
