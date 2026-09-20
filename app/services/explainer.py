"""Explainability service for patient-level local SHAP attributions."""

import logging
from typing import Optional

from app.schemas import ExplanationResponse, FeatureAttribution, PatientInput
from app.services.predictor import predictor_service
from src.features.xai_shap import ShapExplainerEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ExplainerService:
    """Service wrapping local SHAP feature attribution generation."""

    def __init__(self, shap_engine: Optional[ShapExplainerEngine] = None) -> None:
        """Initialize service with optional pre-configured ShapExplainerEngine."""
        self.shap_engine = shap_engine or ShapExplainerEngine(
            model=predictor_service.model,
            preprocessor=predictor_service.preprocessor,
            n_background=100,
        )

    def explain(self, patient: PatientInput) -> ExplanationResponse:
        """Compute local SHAP feature attributions for a single patient profile.

        Args:
            patient: Validated patient inputs.

        Returns:
            ExplanationResponse with base margin, feature attributions, and top drivers.
        """
        patient_dict = patient.model_dump()
        raw_result = self.shap_engine.explain_local(patient_dict)

        attributions = [
            FeatureAttribution(
                feature=a["feature"],
                attribution_value=round(a["shap_value"], 4),
                direction=a["effect"],
                raw_value=patient_dict.get(a["feature"], None),
            )
            for a in raw_result["attributions"]
        ]

        increasing_factors = [
            FeatureAttribution(
                feature=a["feature"],
                attribution_value=round(a["shap_value"], 4),
                direction=a["effect"],
                raw_value=patient_dict.get(a["feature"], None),
            )
            for a in raw_result["top_risk_increasing_factors"]
        ]

        decreasing_factors = [
            FeatureAttribution(
                feature=a["feature"],
                attribution_value=round(a["shap_value"], 4),
                direction=a["effect"],
                raw_value=patient_dict.get(a["feature"], None),
            )
            for a in raw_result["top_risk_decreasing_factors"]
        ]

        return ExplanationResponse(
            base_value=round(raw_result["base_value"], 4),
            predicted_margin=round(raw_result["predicted_margin"], 4),
            predicted_probability=round(raw_result["predicted_probability"], 4),
            risk_classification=raw_result["risk_classification"],
            attributions=attributions,
            top_risk_increasing_factors=increasing_factors,
            top_risk_decreasing_factors=decreasing_factors,
            model_version=predictor_service.model_version,
        )


# Singleton instance for route handlers
explainer_service = ExplainerService()
