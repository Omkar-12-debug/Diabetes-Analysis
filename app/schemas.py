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
    patient_id: Optional[str] = Field(
        None,
        description="Optional unique patient identifier. An auto-generated UUID is assigned if omitted.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "patient-12345",
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
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    assessment_id: Optional[int] = Field(None, description="Unique database assessment record ID")


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


class AssessmentHistoryResponse(BaseModel):
    """Historical assessment record capturing patient features and inference outcome."""

    id: int = Field(..., description="Assessment database primary key ID")
    patient_id: str = Field(..., description="Unique patient identifier")
    created_at: str = Field(..., description="ISO 8601 timestamp of assessment")
    input_features: Dict[str, Any] = Field(..., description="Evaluated clinical and biometric features")
    risk_probability: float = Field(..., description="Calibrated positive risk probability [0.0, 1.0]")
    risk_score: int = Field(..., description="Integer clinical risk score (0–100)")
    risk_category: str = Field(..., description="Categorical risk tier: Low, Moderate, High")
    model_version: str = Field(..., description="Model version used for inference")
    verified_diabetes_label: Optional[int] = Field(
        None,
        description="Confirmed clinical diagnostic outcome (0: Non-Diabetic, 1: Diabetic, null if unverified)",
    )
    verified_at: Optional[str] = Field(None, description="ISO 8601 timestamp of diagnostic verification")
    notes: Optional[str] = Field(None, description="Clinical feedback or diagnostic notes")

    model_config = ConfigDict(from_attributes=True)


class HistoryListResponse(BaseModel):
    """Collection of assessment records with pagination metadata."""

    patient_id: Optional[str] = Field(None, description="Patient identifier filter, or null for global history")
    total_records: int = Field(..., description="Total count of assessment records returned")
    assessments: List[AssessmentHistoryResponse] = Field(..., description="Ordered assessment history records")


class DelayedLabelInput(BaseModel):
    """Payload for delayed clinical ground truth verification."""

    assessment_id: int = Field(..., description="Assessment record ID to bind diagnosis to")
    verified_diabetes_label: Literal[0, 1] = Field(
        ...,
        description="Confirmed clinical diagnostic outcome (0: Non-Diabetic, 1: Diabetic)",
    )
    notes: Optional[str] = Field(None, description="Clinical diagnostic notes or lab confirmation details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "assessment_id": 1,
                "verified_diabetes_label": 1,
                "notes": "Diagnosis confirmed via fasting plasma glucose test (>126 mg/dL) at 3-month follow-up.",
            }
        }
    )


class DelayedLabelResponse(BaseModel):
    """Response confirming ground truth diagnostic binding."""

    id: int = Field(..., description="Ground truth record ID")
    assessment_id: int = Field(..., description="Bound assessment ID")
    patient_id: str = Field(..., description="Patient ID")
    verified_diabetes_label: int = Field(..., description="Confirmed diabetic status (0 or 1)")
    verified_at: str = Field(..., description="ISO 8601 verification timestamp")
    notes: Optional[str] = Field(None, description="Clinical notes recorded")
    message: str = Field(
        "Ground truth clinical verification successfully recorded.",
        description="Operation status message",
    )

