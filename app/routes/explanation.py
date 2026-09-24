"""Explainable AI and counterfactual sensitivity routes."""

from fastapi import APIRouter

from app.schemas import (
    ExplanationResponse,
    PatientInput,
    WhatIfRequest,
    WhatIfResponse,
)
from app.services.counterfactual import counterfactual_service
from app.services.explainer import explainer_service

router = APIRouter(tags=["Explainability & What-If"])


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Local SHAP Risk Attribution",
    description="Computes patient-specific SHAP values across all 27 clinical feature dimensions, ranking top risk-increasing and protective factors.",
)
def explain_prediction(patient: PatientInput) -> ExplanationResponse:
    """Generate local SHAP feature attributions for a single patient profile."""
    return explainer_service.explain(patient)


@router.post(
    "/what-if",
    response_model=WhatIfResponse,
    summary="Counterfactual What-If Simulation",
    description="Explores actionable counterfactual sensitivity scenarios, modifying only modifiable features (BMI, glucose, HbA1c, smoking) while strictly freezing non-modifiable factors (age, gender, history).",
)
def simulate_what_if(request: WhatIfRequest) -> WhatIfResponse:
    """Generate or evaluate counterfactual sensitivity scenarios for a patient profile."""
    return counterfactual_service.simulate(request)
