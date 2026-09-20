"""Counterfactual service providing what-if sensitivity simulations."""

import copy
import logging
from typing import Optional

from app.schemas import WhatIfRequest, WhatIfResponse
from app.services.predictor import predictor_service
from src.features.counterfactual_dice import (
    MANDATORY_DISCLAIMER,
    MODIFIABLE_FEATURES,
    NON_MODIFIABLE_FEATURES,
    CounterfactualExplainerEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CounterfactualService:
    """Service generating actionable what-if simulations with strict clinical guardrails."""

    def __init__(self, cf_engine: Optional[CounterfactualExplainerEngine] = None) -> None:
        """Initialize service with optional pre-configured CounterfactualExplainerEngine."""
        self.cf_engine = cf_engine or CounterfactualExplainerEngine(
            model=predictor_service.model,
            preprocessor=predictor_service.preprocessor,
        )

    def simulate(self, request: WhatIfRequest) -> WhatIfResponse:
        """Execute counterfactual sensitivity exploration on a patient profile.

        Args:
            request: Validated WhatIfRequest containing patient and optional overrides.

        Returns:
            WhatIfResponse detailing probability changes and required lifestyle modifications.
        """
        patient_dict = request.patient.model_dump()
        target_threshold = request.target_risk_threshold or 0.50

        # Case A: User supplied manual overrides
        if request.overrides:
            orig_prob = self.cf_engine.predict_probability(patient_dict)
            cf_patient = copy.deepcopy(patient_dict)

            feature_changes = {}
            for mod_key, new_val in request.overrides.items():
                if mod_key in MODIFIABLE_FEATURES and mod_key in cf_patient:
                    old_val = cf_patient[mod_key]
                    if old_val != new_val:
                        cf_patient[mod_key] = new_val
                        feature_changes[mod_key] = {
                            "original": old_val,
                            "counterfactual": new_val,
                        }

            # Enforce non-modifiable feature preservation
            preserved = True
            for non_mod in NON_MODIFIABLE_FEATURES:
                if cf_patient.get(non_mod) != patient_dict.get(non_mod):
                    cf_patient[non_mod] = patient_dict[non_mod]
                    preserved = False

            new_prob = self.cf_engine.predict_probability(cf_patient)
            delta = float(new_prob - orig_prob)

            return WhatIfResponse(
                original_probability=round(orig_prob, 4),
                counterfactual_probability=round(new_prob, 4),
                probability_delta=round(delta, 4),
                target_threshold=target_threshold,
                target_achieved=bool(new_prob < target_threshold),
                feature_changes=feature_changes,
                counterfactual_patient=cf_patient,
                non_modifiable_preserved=preserved,
                disclaimer=MANDATORY_DISCLAIMER,
            )

        # Case B: Automated search for minimal modifications lowering risk
        result = self.cf_engine.generate_counterfactual(
            patient_dict=patient_dict,
            target_risk_threshold=target_threshold,
        )

        return WhatIfResponse(
            original_probability=round(result["original_probability"], 4),
            counterfactual_probability=round(result["counterfactual_probability"], 4),
            probability_delta=round(result["probability_delta"], 4),
            target_threshold=target_threshold,
            target_achieved=result["target_achieved"],
            feature_changes=result["feature_changes"],
            counterfactual_patient=result["counterfactual_patient"],
            non_modifiable_preserved=result["non_modifiable_preserved"],
            disclaimer=result["disclaimer"],
        )


# Singleton instance for route handlers
counterfactual_service = CounterfactualService()
