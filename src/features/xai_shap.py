"""Explainable AI (XAI) engine using SHAP TreeExplainer.

Provides background reference sampling, global feature importance summary generation,
and granular local single-patient attribution analysis with mapped clinical risk drivers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.evaluation.evaluate_all import CHAMPION_URI, load_champion_model
from src.features.engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
DEFAULT_TRAIN_PATH = Path("data/processed/train.parquet")
DEFAULT_VAL_PATH = Path("data/processed/val.parquet")
DEFAULT_SHAP_SUMMARY_PATH = Path("reports/figures/shap_summary.png")


class ShapExplainerEngine:
    """SHAP explanation engine for the XGBoost champion model."""

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        background_data: Optional[np.ndarray] = None,
        n_background: int = 150,
        model_uri: str = CHAMPION_URI,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
        train_path: Union[str, Path] = DEFAULT_TRAIN_PATH,
    ) -> None:
        """Initialize SHAP TreeExplainer with background reference samples.

        Args:
            model: Optional pre-loaded XGBoost champion model.
            preprocessor: Optional pre-loaded scikit-learn ColumnTransformer.
            background_data: Optional numpy array of background reference data.
            n_background: Number of reference rows to sample from training set.
            model_uri: MLflow model URI if loading dynamically.
            preprocessor_path: Path to serialized preprocessor.joblib.
            train_path: Path to processed train.parquet.
        """
        self.model = model or load_champion_model(model_uri)
        self.preprocessor = preprocessor or joblib.load(preprocessor_path)

        # Obtain feature names from preprocessor
        if hasattr(self.preprocessor, "get_feature_names_out"):
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        else:
            self.feature_names = [f"feature_{i}" for i in range(27)]

        # Background sampling for TreeExplainer
        if background_data is not None:
            self.background_data = background_data
        else:
            train_path = Path(train_path)
            if train_path.exists():
                train_df = pd.read_parquet(train_path)
                features_df = train_df.drop(columns=["diabetes"], errors="ignore")
                sample_n = min(n_background, len(features_df))
                self.background_data = features_df.sample(n=sample_n, random_state=42).values
                logger.info("Sampled %d background reference rows for SHAP explainer.", sample_n)
            else:
                self.background_data = None
                logger.warning("Train dataset not found at %s. Explainer using path-dependent tree mode.", train_path)

        # Initialize TreeExplainer
        if self.background_data is not None:
            self.explainer = shap.TreeExplainer(self.model, data=self.background_data, model_output="raw")
        else:
            self.explainer = shap.TreeExplainer(self.model, model_output="raw")

        self.expected_value = float(self.explainer.expected_value)
        logger.info("TreeExplainer initialized with expected base value: %.4f", self.expected_value)

    def explain_global(
        self,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        save_path: Union[str, Path] = DEFAULT_SHAP_SUMMARY_PATH,
        max_samples: int = 500,
    ) -> Path:
        """Generate and save global SHAP summary plot across validation cohort.

        Args:
            X_val: Feature matrix or DataFrame. If None, loaded from val.parquet.
            save_path: Output path for shap_summary.png.
            max_samples: Maximum number of rows to evaluate for plotting.

        Returns:
            Path: Destination path of the saved figure.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if X_val is None:
            val_df = pd.read_parquet(DEFAULT_VAL_PATH)
            X_val = val_df.drop(columns=["diabetes"], errors="ignore")

        if hasattr(X_val, "sample") and len(X_val) > max_samples:
            X_sample = X_val.sample(n=max_samples, random_state=42)
        elif isinstance(X_val, np.ndarray) and len(X_val) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(X_val), size=max_samples, replace=False)
            X_sample = X_val[indices]
        else:
            X_sample = X_val

        X_array = X_sample.values if hasattr(X_sample, "values") else X_sample

        logger.info("Computing global SHAP values on %d samples...", len(X_array))
        shap_values = self.explainer(X_array)

        # Generate summary plot
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values.values,
            X_array,
            feature_names=self.feature_names,
            show=False,
            max_display=15,
        )
        plt.title("SHAP Global Feature Attributions — DiabetesRiskModel (@champion)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved global SHAP summary plot to %s", save_path)
        return save_path

    def explain_local(
        self,
        patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    ) -> Dict[str, Any]:
        """Generate patient-specific local attribution breakdown for an individual record.

        Accepts raw patient fields, executes feature engineering, runs preprocessor,
        and computes feature-level SHAP attributions.

        Args:
            patient_data: Dictionary or DataFrame with raw patient features (e.g.,
                'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
                'bmi', 'HbA1c_level', 'blood_glucose_level').

        Returns:
            Dict containing base_value, predicted_margin, predicted_probability,
            and ordered list of attributions with direction of risk impact.
        """
        # Format input to DataFrame
        if isinstance(patient_data, dict):
            raw_df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, pd.Series):
            raw_df = pd.DataFrame([patient_data.to_dict()])
        elif isinstance(patient_data, pd.DataFrame):
            raw_df = patient_data.copy()
        else:
            raise ValueError("patient_data must be a dict, pd.Series, or pd.DataFrame.")

        # Check if features are already preprocessed (27 dimensions) or raw (8 raw columns)
        if raw_df.shape[1] == 27 and all(c in self.feature_names for c in raw_df.columns):
            transformed_arr = raw_df.values
            raw_values_dict = {}
        else:
            # 1. Feature Engineering
            engineered_df = engineer_features(raw_df)
            raw_values_dict = raw_df.iloc[0].to_dict()

            # 2. Preprocessing Transformation
            transformed_arr = self.preprocessor.transform(engineered_df)

        # Compute SHAP values
        shap_res = self.explainer(transformed_arr)
        shap_values_row = shap_res.values[0]

        # Predicted margin and probability
        predicted_margin = float(self.expected_value + np.sum(shap_values_row))
        # Logistic sigmoid probability conversion from log-odds margin
        predicted_prob = float(1.0 / (1.0 + np.exp(-predicted_margin)))

        # Build feature attribution records
        attributions: List[Dict[str, Any]] = []
        for feat_name, shap_val in zip(self.feature_names, shap_values_row):
            effect = "increases_risk" if shap_val > 0 else "decreases_risk"
            attributions.append({
                "feature": feat_name,
                "shap_value": float(shap_val),
                "effect": effect,
                "abs_importance": float(abs(shap_val)),
            })

        # Sort descending by absolute attribution magnitude
        attributions.sort(key=lambda x: x["abs_importance"], reverse=True)

        increasing_factors = [a for a in attributions if a["shap_value"] > 0][:5]
        decreasing_factors = [a for a in attributions if a["shap_value"] < 0][:5]

        return {
            "base_value": float(self.expected_value),
            "predicted_margin": predicted_margin,
            "predicted_probability": predicted_prob,
            "risk_classification": "Diabetic / Elevated Risk" if predicted_prob >= 0.5 else "Non-Diabetic / Low Risk",
            "raw_inputs": raw_values_dict,
            "top_risk_increasing_factors": increasing_factors,
            "top_risk_decreasing_factors": decreasing_factors,
            "attributions": attributions,
        }
