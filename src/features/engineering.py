"""Feature engineering module for diabetes prediction pipeline.

Constructs clinically grounded derived features using row-local operations.
All transformations are deterministic and leakage-safe (no population-level
statistics are computed), so they can be applied independently to any split.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Age group bin edges and labels
AGE_BINS = [0, 18, 35, 50, 65, float("inf")]
AGE_LABELS = ["<18", "18-35", "36-50", "51-65", "65+"]

# BMI clinical category thresholds
BMI_BINS = [0, 18.5, 24.9, 29.9, float("inf")]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]

# Names of all engineered feature columns
ENGINEERED_FEATURE_NAMES: List[str] = [
    "age_group",
    "bmi_category",
    "cardiometabolic_risk",
    "glucose_hba1c_interaction",
    "age_bmi_interaction",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically grounded derived features to the DataFrame.

    Creates 5 new columns:
        - age_group: Categorical age bins (<18, 18-35, 36-50, 51-65, 65+).
        - bmi_category: Clinical BMI tiers (Underweight, Normal, Overweight, Obese).
        - cardiometabolic_risk: Integer sum of hypertension + heart_disease (0, 1, or 2).
        - glucose_hba1c_interaction: blood_glucose_level * HbA1c_level.
        - age_bmi_interaction: age * bmi.

    All operations are row-local (no population-level fitting), making them
    safe to apply independently to train, validation, and test splits.

    Args:
        df: Input DataFrame with raw features.

    Returns:
        pd.DataFrame: Copy of input with 5 additional engineered columns.
    """
    result = df.copy()

    # Age group binning
    result["age_group"] = pd.cut(
        result["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)

    # BMI category binning
    result["bmi_category"] = pd.cut(
        result["bmi"],
        bins=BMI_BINS,
        labels=BMI_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)

    # Cardiometabolic composite risk
    result["cardiometabolic_risk"] = (
        result["hypertension"].astype(int) + result["heart_disease"].astype(int)
    )

    # Continuous interaction terms
    result["glucose_hba1c_interaction"] = (
        result["blood_glucose_level"] * result["HbA1c_level"]
    )
    result["age_bmi_interaction"] = result["age"] * result["bmi"]

    logger.info(
        "Engineered %d features: %s. Shape: %s.",
        len(ENGINEERED_FEATURE_NAMES),
        ENGINEERED_FEATURE_NAMES,
        result.shape,
    )

    return result


# Convenience alias for downstream service integration
engineer_features = add_engineered_features
