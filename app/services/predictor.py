"""Inference service for diabetes risk prediction and tier stratification."""

import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

from app.schemas import PatientInput, PredictionResponse
from src.evaluation.evaluate_all import (
    CHAMPION_ALIAS,
    CHAMPION_URI,
    REGISTERED_MODEL_NAME,
    load_champion_model,
)
from src.features.engineering import add_engineered_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


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
        self._model: Optional[Any] = None
        self._preprocessor: Optional[Any] = None
        self.model_version: str = "2"

        # Attempt eager load only if the preprocessor artifact already exists.
        # This prevents pytest collection crashes on clean machines / CI runners.
        if self.preprocessor_path.exists():
            try:
                self._load()
            except Exception as exc:
                logger.warning("Eager loading skipped due to error: %s. Will load on demand.", exc)

    def _ensure_loaded(self) -> None:
        """Ensure model and preprocessor are loaded, generating them if absent."""
        if self._preprocessor is None:
            if not self.preprocessor_path.exists():
                logger.info("Preprocessor missing at %s. Running preprocessing pipeline...", self.preprocessor_path)
                try:
                    from src.preprocessing.pipeline import run_pipeline
                    run_pipeline()
                except Exception as exc:
                    logger.error("Failed to run preprocessing pipeline: %s", exc)

            if self.preprocessor_path.exists():
                logger.info("Loading preprocessor from %s...", self.preprocessor_path)
                self._preprocessor = joblib.load(self.preprocessor_path)
            else:
                raise FileNotFoundError(
                    f"Preprocessor artifact not found at {self.preprocessor_path} and could not be auto-generated."
                )

        if self._model is None:
            logger.info("Loading model from %s...", self.model_uri)
            self._model = load_champion_model(self.model_uri)

            # Retrieve active champion version from MLflow
            try:
                client = MlflowClient()
                version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
                self.model_version = str(version_info.version)
                logger.info("Active champion model version resolved: %s", self.model_version)
            except Exception as exc:
                logger.warning("Could not resolve champion model version from MLflow: %s. Using default.", exc)

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

    def predict(self, patient: PatientInput) -> PredictionResponse:
        """Execute risk inference on a validated patient input.

        Args:
            patient: Validated patient profile.

        Returns:
            PredictionResponse containing risk score (0-100), tier, and probability.
        """
        self._ensure_loaded()

        patient_dict = patient.model_dump()
        features_dict = {k: v for k, v in patient_dict.items() if k != "patient_id"}
        raw_df = pd.DataFrame([features_dict])

        # Feature engineering & transformation
        engineered_df = add_engineered_features(raw_df)
        transformed_arr = self.preprocessor.transform(engineered_df)

        # Probability inference
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(transformed_arr)[0, 1])
        else:
            raw_pred = self.model.predict(transformed_arr)
            prob = float(raw_pred[0])

        prob = max(0.0, min(1.0, prob))
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