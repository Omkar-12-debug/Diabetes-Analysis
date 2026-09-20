"""Pydantic v2 schemas for request validation and response formatting.

Strictly validates physiological boundaries matching Phase 2 data auditing:
- age: [0.0, 120.0]
- gender: ['Female', 'Male', 'Other']
- hypertension: 0 or 1
- heart_disease: 0 or 1
- smoking_history: ['never', 'No Info', 'current', 'former', 'ever', 'not current']
- bmi: [10.0, 100.0]
- HbA1c_level: [3.0, 20.0]
- blood_glucose_level: [30.0, 500.0]
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
    """Clinical patient profile input with physiological boundary validation."""

    age: float = Field(
        ...,
        ge=0.0,
        le=120.0,
        description="Patient age in years (0.0 to 120.0)",
    )
    gender: Literal["Female", "Male", "Other"] = Field(
        ...,
        description="Biological sex or self-reported gender identity",
    )
    hypertension: Literal[0, 1] = Field(
        ...,
        description="Clinically diagnosed hypertension (0: No, 1: Yes)",
    )
    heart_disease: Literal[0, 1] = Field(
        ...,
        description="Diagnosed cardiovascular or coronary heart disease (0: No, 1: Yes)",
    )
    smoking_history: Literal[
        "never", "No Info", "current", "former", "ever", "not current"
    ] = Field(
        ...,
        description="Smoking history category",
    )
    bmi: float = Field(
        ...,
        ge=10.0,
        le=100.0,
        description="Body Mass Index in kg/m^2 (10.0 to 100.0)",
    )
    HbA1c_level: float = Field(
        ...,
        ge=3.0,
        le=20.0,
        description="Glycated hemoglobin percentage (3.0% to 20.0%)",
    )
    blood_glucose_level: float = Field(
        ...,
        ge=30.0,
        le=500.0,
        description="Blood glucose concentration in mg/dL (30.0 to 500.0)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "age": 55.0,
                "hypertension": 1,
                "heart_disease": 0,
                "smoking_history": "former",
                "bmi": 32.5,
                "HbA1c_level": 7.2,
                "blood_glucose_level": 180.0,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Inference outcome with calibrated probability and discrete risk tiers."""

    prediction: int = Field(..., description="Binary classification (0: Non-Diabetic, 1: Diabetic)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated positive risk probability [0.0, 1.0]")
    risk_score: int = Field(..., ge=0, le=100, description="Integer clinical risk score from 0 to 100")
    risk_category: Literal["Low", "Moderate", "High"] = Field(
        ...,
        description="Risk tier: Low (<30), Moderate (30-70), High (>70)",
    )
    model_name: str = Field(..., description="Registered model name")
    model_version: str = Field(..., description="Active champion model version number")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp")


class FeatureAttribution(BaseModel):
    """Single feature SHAP attribution."""

    feature: str = Field(..., description="Feature name")
    attribution_value: float = Field(..., description="SHAP attribution value (+ increases risk, - decreases risk)")
    direction: Literal["increases_risk", "decreases_risk"] = Field(..., description="Direction of risk impact")
    raw_value: Optional[Any] = Field(None, description="Raw or transformed input value")


class ExplanationResponse(BaseModel):
    """Local SHAP feature attribution explanation."""

    base_value: float = Field(..., description="Baseline expected log-odds margin")
    predicted_margin: float = Field(..., description="Sum of base value and all attributions")
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted positive class probability")
    risk_classification: str = Field(..., description="Risk tier label")
    attributions: List[FeatureAttribution] = Field(..., description="Full 27-feature attribution list")
    top_risk_increasing_factors: List[FeatureAttribution] = Field(
        ...,
        description="Top factors driving risk higher",
    )
    top_risk_decreasing_factors: List[FeatureAttribution] = Field(
        ...,
        description="Top protective factors lowering risk",
    )
    model_version: str = Field(..., description="Active champion model version")


class WhatIfRequest(BaseModel):
    """What-if sensitivity simulation request."""

    patient: PatientInput = Field(..., description="Baseline patient clinical profile")
    target_risk_threshold: Optional[float] = Field(
        0.50,
        ge=0.01,
        le=0.99,
        description="Target maximum allowable probability",
    )
    overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional manual modifiable overrides (e.g. {'bmi': 25.0})",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient": {
                    "gender": "Female",
                    "age": 55.0,
                    "hypertension": 1,
                    "heart_disease": 0,
                    "smoking_history": "former",
                    "bmi": 32.5,
                    "HbA1c_level": 7.2,
                    "blood_glucose_level": 180.0,
                },
                "target_risk_threshold": 0.50,
            }
        }
    )


class WhatIfResponse(BaseModel):
    """What-if counterfactual scenario outcome."""

    original_probability: float = Field(..., description="Baseline predicted probability")
    counterfactual_probability: float = Field(..., description="Adjusted predicted probability")
    probability_delta: float = Field(..., description="Probability change (counterfactual - original)")
    target_threshold: float = Field(..., description="Requested risk threshold")
    target_achieved: bool = Field(..., description="Whether counterfactual meets target")
    feature_changes: Dict[str, Any] = Field(..., description="Dictionary of modified clinical variables")
    counterfactual_patient: Dict[str, Any] = Field(..., description="Full counterfactual patient profile")
    non_modifiable_preserved: bool = Field(..., description="Confirms age, gender, hypertension, heart_disease untouched")
    disclaimer: str = Field(..., description="Mandatory non-causal medical disclaimer")


class ModelInfoResponse(BaseModel):
    """Registered model governance and performance metadata."""

    model_name: str = Field(..., description="Registered model name")
    alias: str = Field(..., description="Active MLflow alias")
    model_version: str = Field(..., description="Active model version")
    algorithm: str = Field(..., description="Underlying ML algorithm")
    status: str = Field(..., description="Quality gate promotion status")
    metrics_summary: Dict[str, Any] = Field(..., description="Validation and test metrics")
    thresholds: Dict[str, Any] = Field(..., description="Quality gate thresholds")


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field("healthy", description="Service health state")
    service: str = Field(..., description="Service identifier")
    model_name: str = Field(..., description="Active registered model name")
    model_version: str = Field(..., description="Active model version")
    timestamp: str = Field(..., description="Current system timestamp")
