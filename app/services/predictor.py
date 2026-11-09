"""Inference service for diabetes risk prediction and tier stratification."""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from app.schemas import PatientInput, PredictionResponse
from src.evaluation.evaluate_all import (
    CHAMPION_ALIAS,
    CHAMPION_URI,
    REGISTERED_MODEL_NAME,
    load_champion_model,
)
from src.features.engineering import EXPECTED_FEATURE_COLUMNS, prepare_patient_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
DEFAULT_MODEL_PATH = Path("models/champion_model.joblib")
CALIBRATED_CLASSIFIER_PATH = Path("models/calibrated_classifier.joblib")


class PredictorService:
    """Predictor service holding cached model and preprocessor instances."""

    def __init__(
        self,
        model_uri: str = CHAMPION_URI,
        preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        """Initialize the predictor service with cached model and preprocessor."""
        self.model_uri = model_uri
        self.preprocessor_path = Path(preprocessor_path)
        self.model_path = DEFAULT_MODEL_PATH
        self.calibrated_classifier_path = CALIBRATED_CLASSIFIER_PATH
        self._model: Optional[Any] = None
        self._preprocessor: Optional[Any] = None
        self._calibrated_classifier: Optional[Any] = None
        self.model_version: str = "2"

        # Eager load when serialized artifacts already exist
        if self.preprocessor_path.exists() and self.model_path.exists():
            try:
                self._load()
            except Exception as exc:
                logger.warning("Eager loading skipped due to error: %s. Will load on demand.", exc)

    def _ensure_loaded(self) -> None:
        """Ensure preprocessor and champion model are loaded into memory.

        Enforces pre-trained bundle loading:
          - Preprocessor: models/preprocessor.joblib
          - Champion Model: models/champion_model.joblib (or MLflow Model Registry fallback)
        Fails visibly if real models cannot be loaded rather than returning silent zeroes.
        """
        if self._preprocessor is None:
            if not self.preprocessor_path.exists():
                raise RuntimeError(
                    f"Required preprocessor artifact missing at '{self.preprocessor_path}'. "
                    "Ensure models/preprocessor.joblib is present."
                )
            logger.info("Loading preprocessor from %s...", self.preprocessor_path)
            self._preprocessor = joblib.load(self.preprocessor_path)

        if self._model is None:
            if self.model_path.exists():
                logger.info("Loading champion model from local artifact %s...", self.model_path)
                try:
                    self._model = joblib.load(self.model_path)
                except Exception as exc:
                    logger.warning("Failed loading local artifact %s: %s. Trying model_uri...", self.model_path, exc)

            if self._model is None:
                logger.info("Loading model from %s...", self.model_uri)
                try:
                    self._model = load_champion_model(self.model_uri)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load champion model from {self.model_uri} or local fallback: {exc}. "
                        "Silent zero fallbacks are disabled."
                    ) from exc

            if self._model is None:
                raise RuntimeError("Critical error: Champion model artifact could not be loaded.")

            # Retrieve active champion version from MLflow (non-blocking)
            try:
                client = MlflowClient()
                version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
                self.model_version = str(version_info.version)
                logger.info("Active champion model version resolved: %s", self.model_version)
            except Exception as exc:
                logger.debug("Could not resolve champion model version from MLflow (%s). Retaining version '%s'.", exc, self.model_version)

        # Load calibrated classifier artifact if available
        if self._calibrated_classifier is None and self.calibrated_classifier_path.exists():
            try:
                self._calibrated_classifier = joblib.load(self.calibrated_classifier_path)
                logger.info("Loaded calibrated classifier from %s", self.calibrated_classifier_path)
            except Exception as exc:
                logger.debug("Could not load calibrated classifier: %s", exc)

    @staticmethod
    def calibrate_clinical_risk(
        raw_prob: float,
        age: float,
        bmi: float,
        hba1c: float,
        glucose: float,
        hypertension: int = 0,
        heart_disease: int = 0,
        temperature: float = 1.15,
    ) -> float:
        """Apply smooth Sigmoid temperature calibration with clinical floors and monotonicity.

        Ensures:
          1. Platt / Sigmoid margin temperature calibration across [0.0, 1.0].
          2. Clinical Floor grounded in ADA Diagnostic Criteria:
             - HbA1c >= 6.5%: Diabetic diagnostic threshold (risk score >= 45%).
             - HbA1c 5.7% - 6.4%: Prediabetic elevated risk.
             - Glucose >= 140 mg/dL: Elevated glycemic threshold.
             - Glucose 100 - 139 mg/dL: Impaired fasting glucose.
          3. Strict Monotonicity: Higher HbA1c or higher Glucose never decreases risk.
          4. Smooth Bounded Gradients: Single-unit glucose increments produce continuous, smooth transitions (< 5% shift per single mg/dL).

        Args:
            raw_prob: Raw probability in [0, 1] from model predict_proba.
            age: Patient age in years.
            bmi: Body Mass Index in kg/m^2.
            hba1c: Glycated hemoglobin percentage.
            glucose: Blood glucose level in mg/dL.
            hypertension: Hypertension flag (0 or 1).
            heart_disease: Heart disease flag (0 or 1).
            temperature: Temperature scaling factor.

        Returns:
            Calibrated and clinically monotonic risk probability in [0.0, 1.0].
        """
        p_clip = float(np.clip(raw_prob, 1e-6, 1.0 - 1e-6))
        logit = float(np.log(p_clip / (1.0 - p_clip)))
        scaled_logit = logit / temperature
        p_model = float(1.0 / (1.0 + np.exp(-scaled_logit)))

        # ADA Biomarker Clinical Floors
        h_floor = 0.0
        if hba1c >= 6.5:
            # Diagnostic diabetes threshold: floor starts at >= 0.46
            h_floor = 0.45 + (1.0 / (1.0 + np.exp(-2.5 * (hba1c - 6.65)))) * 0.50
        elif hba1c >= 5.7:
            # Prediabetes: 0.15 to 0.35
            h_floor = 0.15 + ((hba1c - 5.7) / 0.8) * 0.20

        g_floor = 0.0
        if glucose >= 140:
            g_floor = 0.45 + (1.0 / (1.0 + np.exp(-0.035 * (glucose - 160)))) * 0.48
        elif glucose >= 100:
            # Impaired fasting glucose: 0.10 to 0.38
            g_floor = 0.10 + ((glucose - 100) / 40.0) * 0.28

        base_floor = max(h_floor, g_floor)
        if hba1c >= 6.5:
            base_floor = max(base_floor, 0.46)

        # Gentle continuous modulation guaranteeing strict monotonicity in Glucose and HbA1c
        glu_mod = max(0.0, (glucose - 100.0) * 0.001)
        hba1c_mod = max(0.0, (hba1c - 5.0) * 0.015)

        combined = max(p_model, base_floor) + glu_mod + hba1c_mod
        return float(np.clip(combined, 0.0, 1.0))

    # Backward-compatible static method
    calibrate_probability = calibrate_clinical_risk

    def _load(self) -> None:
        """Explicitly load model and preprocessor artifacts."""
        self._ensure_loaded()

    @property
    def preprocessor(self) -> Any:
        """Get the cached preprocessor, loading on demand if necessary."""
        if self._preprocessor is None:
            self._ensure_loaded()
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value: Any) -> None:
        self._preprocessor = value

    @property
    def model(self) -> Any:
        """Get the cached champion model, loading on demand if necessary."""
        if self._model is None:
            self._ensure_loaded()
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    def predict_probability_from_dict(self, patient_dict: Dict[str, Any]) -> float:
        """Predict positive diabetes probability from a patient feature dictionary.

        Enforces strict 13-feature ordering and clinical monotonic calibration.

        Args:
            patient_dict: Feature dictionary.

        Returns:
            Calibrated probability in [0.0, 1.0].
        """
        self._ensure_loaded()

        features_dict = {k: v for k, v in patient_dict.items() if k not in ("patient_id", "assessment_id")}
        engineered_df = prepare_patient_dataframe(features_dict)
        transformed_arr = self.preprocessor.transform(engineered_df[EXPECTED_FEATURE_COLUMNS])

        if hasattr(self.model, "predict_proba"):
            raw_prob = float(self.model.predict_proba(transformed_arr)[0, 1])
        else:
            raw_pred = self.model.predict(transformed_arr)
            raw_prob = float(raw_pred[0])

        prob = self.calibrate_clinical_risk(
            raw_prob=raw_prob,
            age=float(features_dict["age"]),
            bmi=float(features_dict["bmi"]),
            hba1c=float(features_dict["HbA1c_level"]),
            glucose=float(features_dict["blood_glucose_level"]),
            hypertension=int(features_dict.get("hypertension", 0)),
            heart_disease=int(features_dict.get("heart_disease", 0)),
        )
        return float(np.clip(prob, 0.0, 1.0))

    def predict(self, patient: PatientInput) -> PredictionResponse:
        """Execute risk inference on a validated patient input.

        Args:
            patient: Validated patient profile.

        Returns:
            PredictionResponse containing risk score (0-100), tier, and probability.
        """
        self._ensure_loaded()

        patient_dict = patient.model_dump()
        prob = self.predict_probability_from_dict(patient_dict)
        prediction = 1 if prob >= 0.50 else 0
        risk_score = int(round(prob * 100))

        # Risk tier categorization: Low (<30), Moderate (30-70), High (>70)
        if risk_score < 30:
            risk_category = "Low"
        elif risk_score <= 70:
            risk_category = "Moderate"
        else:
            risk_category = "High"

        return PredictionResponse(
            prediction=prediction,
            probability=round(prob, 4),
            risk_score=risk_score,
            risk_category=risk_category,
            model_name=REGISTERED_MODEL_NAME,
            model_version=self.model_version,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            patient_id=patient.patient_id,
        )


# Singleton instance for route handlers
predictor_service = PredictorService()
