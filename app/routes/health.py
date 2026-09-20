"""Health audit and readiness routes."""

import datetime
from fastapi import APIRouter

from app.schemas import HealthResponse
from app.services.predictor import predictor_service
from src.evaluation.evaluate_all import REGISTERED_MODEL_NAME

router = APIRouter(tags=["System Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness",
    description="Returns API service availability, registered model identifier, and active champion version.",
)
def health_check() -> HealthResponse:
    """Return service status and active model metadata."""
    return HealthResponse(
        status="healthy",
        service="diabetes-mlops-api",
        model_name=REGISTERED_MODEL_NAME,
        model_version=predictor_service.model_version,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
