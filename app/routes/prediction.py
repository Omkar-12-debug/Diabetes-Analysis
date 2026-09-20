"""Clinical inference routes for diabetes risk prediction."""

from fastapi import APIRouter

from app.schemas import PatientInput, PredictionResponse
from app.services.predictor import predictor_service

router = APIRouter(tags=["Clinical Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Diabetes Risk",
    description="Ingests clinical and biometric patient features, executes preprocessor transformations, and outputs a 0–100 risk score and risk tier.",
)
def predict_risk(patient: PatientInput) -> PredictionResponse:
    """Predict diabetes onset risk for an individual patient profile."""
    return predictor_service.predict(patient)
