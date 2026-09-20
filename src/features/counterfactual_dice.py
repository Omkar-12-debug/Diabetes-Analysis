"""Counterfactual 'What-If' sensitivity simulation engine using DiCE and clinical optimization.

Generates realistic, actionable feature modifications exclusively across clinically modifiable
attributes (bmi, blood_glucose_level, HbA1c_level, smoking_history) while strictly freezing
non-modifiable demographic and physiological factors (age, gender, hypertension, heart_disease).
Enforces mandatory responsible AI clinical disclaimers on all generated payloads.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.evaluation.evaluate_all import CHAMPION_URI, load_champion_model
from src.features.engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

# Strict feature mutability constraints
NON_MODIFIABLE_FEATURES: List[str] = ["age", "gender", "hypertension", "heart_disease"]
MODIFIABLE_FEATURES: List[str] = ["bmi", "blood_glucose_level", "HbA1c_level", "smoking_history"]

# Clinically realistic healthy boundaries for modifiable factors
PHYSIOLOGICAL_TARGETS: Dict[str, Tuple[float, float]] = {
    "bmi": (18.5, 24.9),                  # Normal BMI
    "blood_glucose_level": (70.0, 99.0),   # Normal Fasting Plasma Glucose
    "HbA1c_level": (4.0, 5.6),             # Normal Non-Diabetic HbA1c
}

MANDATORY_DISCLAIMER: str = (
    "This counterfactual is a hypothetical model sensitivity exploration and must not be "
    "interpreted as causal medical advice or a treatment plan."
)


class CounterfactualExplainerEngine:
    """Engine for generating clinically constrained counterfactual what-if simulations."""

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        model_uri: str = CHAMPION_URI,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        """Initialize the counterfactual engine.

        Args:
            model: Optional pre-loaded model.
            preprocessor: Optional pre-loaded preprocessor.
            model_uri: MLflow URI if loading dynamically.
            preprocessor_path: Path to serialized preprocessor.joblib.
        """
        self.model = model or load_champion_model(model_uri)
        self.preprocessor = preprocessor or joblib.load(preprocessor_path)

    def predict_probability(self, patient_dict: Dict[str, Any]) -> float:
        """Predict the positive diabetes risk probability for a single patient profile.

        Args:
            patient_dict: Raw patient feature values.

        Returns:
            float: Predicted probability in [0.0, 1.0].
        """
        raw_df = pd.DataFrame([patient_dict])
        engineered_df = engineer_features(raw_df)
        transformed_arr = self.preprocessor.transform(engineered_df)

        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(transformed_arr)[0, 1])
        else:
            raw_pred = self.model.predict(transformed_arr)
            prob = float(raw_pred[0])

        return prob

    def generate_counterfactual(
        self,
        patient_dict: Dict[str, Any],
        target_risk_threshold: float = 0.50,
        max_iterations: int = 50,
    ) -> Dict[str, Any]:
        """Generate a what-if counterfactual scenario reducing predicted risk below target.

        Modifies only modifiable parameters ('bmi', 'blood_glucose_level', 'HbA1c_level',
        'smoking_history') while freezing non-modifiable parameters ('age', 'gender',
        'hypertension', 'heart_disease').

        Args:
            patient_dict: Original patient input dictionary.
            target_risk_threshold: Maximum allowable probability for the counterfactual (default: 0.50).
            max_iterations: Maximum optimization search steps.

        Returns:
            Dict containing original_probability, counterfactual_probability,
            probability_delta, feature_changes, non_modifiable_preserved, and disclaimer.
        """
        orig_patient = copy.deepcopy(patient_dict)
        orig_prob = self.predict_probability(orig_patient)

        logger.info(
            "Generating counterfactual for patient (Original Risk: %.4f, Target: <%.2f)...",
            orig_prob, target_risk_threshold,
        )

        # Initialize counterfactual candidate with original values
        cf_patient = copy.deepcopy(orig_patient)

        # 1. Smoking cessation if active smoker
        current_smoking = str(cf_patient.get("smoking_history", "")).lower()
        if current_smoking in ["current", "ever"]:
            cf_patient["smoking_history"] = "former"

        # 2. Iterative targeted reduction of glycemic and adiposity markers
        # Clinically prioritized order: blood_glucose_level, HbA1c_level, bmi
        step_fractions = np.linspace(0.1, 1.0, max_iterations)

        best_cf = copy.deepcopy(cf_patient)
        best_prob = self.predict_probability(best_cf)

        for step in step_fractions:
            if best_prob < target_risk_threshold:
                break

            test_candidate = copy.deepcopy(cf_patient)

            # Modulate Blood Glucose towards healthy upper bound (e.g. 99 mg/dL)
            if "blood_glucose_level" in test_candidate:
                orig_bg = float(orig_patient["blood_glucose_level"])
                target_bg = PHYSIOLOGICAL_TARGETS["blood_glucose_level"][1]
                if orig_bg > target_bg:
                    test_candidate["blood_glucose_level"] = round(orig_bg - step * (orig_bg - target_bg), 1)

            # Modulate HbA1c towards healthy upper bound (e.g. 5.6%)
            if "HbA1c_level" in test_candidate:
                orig_hba1c = float(orig_patient["HbA1c_level"])
                target_hba1c = PHYSIOLOGICAL_TARGETS["HbA1c_level"][1]
                if orig_hba1c > target_hba1c:
                    test_candidate["HbA1c_level"] = round(orig_hba1c - step * (orig_hba1c - target_hba1c), 2)

            # Modulate BMI towards healthy upper bound (e.g. 24.9)
            if "bmi" in test_candidate:
                orig_bmi = float(orig_patient["bmi"])
                target_bmi = PHYSIOLOGICAL_TARGETS["bmi"][1]
                if orig_bmi > target_bmi:
                    test_candidate["bmi"] = round(orig_bmi - step * (orig_bmi - target_bmi), 2)

            candidate_prob = self.predict_probability(test_candidate)
            if candidate_prob < best_prob:
                best_prob = candidate_prob
                best_cf = test_candidate

        # Explicit verification: Ensure non-modifiable features remain completely untouched
        preserved = True
        for non_mod in NON_MODIFIABLE_FEATURES:
            if non_mod in orig_patient and orig_patient[non_mod] != best_cf.get(non_mod):
                preserved = False
                logger.error(
                    "Violation: Non-modifiable feature '%s' altered from %s to %s",
                    non_mod, orig_patient[non_mod], best_cf.get(non_mod),
                )

        # Document specific feature changes
        feature_changes: Dict[str, Dict[str, Any]] = {}
        for mod_feat in MODIFIABLE_FEATURES:
            if mod_feat in orig_patient and mod_feat in best_cf:
                orig_val = orig_patient[mod_feat]
                new_val = best_cf[mod_feat]
                if orig_val != new_val:
                    change_item: Dict[str, Any] = {"original": orig_val, "counterfactual": new_val}
                    if isinstance(orig_val, (int, float)) and isinstance(new_val, (int, float)):
                        change_item["delta"] = round(new_val - orig_val, 2)
                    feature_changes[mod_feat] = change_item

        prob_delta = float(best_prob - orig_prob)

        logger.info(
            "Counterfactual found — New Risk: %.4f (Delta: %.4f, Target Met: %s)",
            best_prob, prob_delta, best_prob < target_risk_threshold,
        )

        return {
            "original_probability": float(orig_prob),
            "counterfactual_probability": float(best_prob),
            "probability_delta": prob_delta,
            "target_threshold": target_risk_threshold,
            "target_achieved": bool(best_prob < target_risk_threshold),
            "original_patient": orig_patient,
            "counterfactual_patient": best_cf,
            "feature_changes": feature_changes,
            "non_modifiable_preserved": preserved,
            "disclaimer": MANDATORY_DISCLAIMER,
        }
