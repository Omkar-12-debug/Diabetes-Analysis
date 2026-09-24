"""Leakage-safe preprocessing pipeline using scikit-learn ColumnTransformer.

Builds a unified pipeline that scales numeric features, one-hot encodes
categoricals, and passes through binary flags. The preprocessor is fitted
strictly on the training partition to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature column groups (after feature engineering)
NUMERIC_FEATURES: List[str] = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "glucose_hba1c_interaction",
    "age_bmi_interaction",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "smoking_history",
    "age_group",
    "bmi_category",
]

BINARY_FEATURES: List[str] = [
    "hypertension",
    "heart_disease",
    "cardiometabolic_risk",
]

TARGET_COL: str = "diabetes"

PREPROCESSOR_SAVE_PATH = Path("models/preprocessor.joblib")


def build_preprocessor() -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for the diabetes feature set.

    Pipeline composition:
        - Numeric features: StandardScaler
        - Categorical features: OneHotEncoder (handle_unknown='ignore')
        - Binary features: passthrough (no transformation)

    Returns:
        ColumnTransformer: Unfitted preprocessing transformer.
    """
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: A fitted ColumnTransformer instance.

    Returns:
        List[str]: Output feature column names.
    """
    return list(preprocessor.get_feature_names_out())


def fit_and_transform_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    save_path: Path = PREPROCESSOR_SAVE_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           ColumnTransformer]:
    """Fit preprocessor on training data and transform all splits.

    The preprocessor is fitted ONLY on train_df to prevent data leakage.
    Transformed arrays are returned along with the fitted preprocessor.

    Args:
        train_df: Training DataFrame (with target column).
        val_df: Validation DataFrame (with target column).
        test_df: Test DataFrame (with target column).
        target_col: Name of the target column to separate.
        save_path: Path to serialize the fitted preprocessor.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor).
    """
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    # Build and fit preprocessor ONLY on training data
    preprocessor = build_preprocessor()
    logger.info("Fitting preprocessor on training data (%d samples)...", len(X_train))
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transform validation and test sets using train-fitted preprocessor
    logger.info("Transforming validation data (%d samples)...", len(X_val))
    X_val_transformed = preprocessor.transform(X_val)

    logger.info("Transforming test data (%d samples)...", len(X_test))
    X_test_transformed = preprocessor.transform(X_test)

    # Serialize preprocessor
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)
    logger.info("Preprocessor serialized to %s.", save_path)

    feature_names = get_feature_names(preprocessor)
    logger.info(
        "Transformation complete. Output dimensionality: %d features. "
        "Shapes: train=%s, val=%s, test=%s.",
        len(feature_names),
        X_train_transformed.shape,
        X_val_transformed.shape,
        X_test_transformed.shape,
    )

    return (
        X_train_transformed,
        X_val_transformed,
        X_test_transformed,
        y_train,
        y_val,
        y_test,
        preprocessor,
    )


def save_processed_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path = Path("data/processed"),
) -> None:
    """Save preprocessed splits as parquet files.

    Each parquet file contains the transformed features and the target column.

    Args:
        X_train, X_val, X_test: Transformed feature arrays.
        y_train, y_val, y_test: Target label arrays.
        feature_names: Column names for the transformed features.
        output_dir: Directory to save parquet files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        df = pd.DataFrame(X, columns=feature_names)
        df[TARGET_COL] = y
        filepath = output_dir / f"{name}.parquet"
        df.to_parquet(filepath, index=False)
        logger.info("Saved %s split to %s (%d rows, %d cols).",
                     name, filepath, len(df), len(df.columns))


def run_pipeline() -> None:
    """Execute the full preprocessing pipeline end-to-end.

    Steps:
        1. Load and validate raw data.
        2. Stratified split (with deduplication).
        3. Feature engineering on each split independently.
        4. Fit preprocessor on train, transform all splits.
        5. Save processed parquet files and serialized preprocessor.
    """
    from src.data.ingest import ingest_data
    from src.data.validate import validate_dataset
    from src.features.engineering import add_engineered_features
    from src.preprocessing.split import stratified_split

    # Step 1: Ingest and validate
    logger.info("=" * 60)
    logger.info("PHASE 3 PIPELINE: Starting preprocessing pipeline...")
    logger.info("=" * 60)

    raw_df = ingest_data()
    validate_dataset(raw_df)

    # Step 2: Stratified split with deduplication
    train_df, val_df, test_df = stratified_split(raw_df, drop_duplicates=True)

    # Step 3: Feature engineering on each split independently (leakage-safe)
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)
    test_df = add_engineered_features(test_df)

    # Step 4: Fit preprocessor on train, transform all splits
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = (
        fit_and_transform_splits(train_df, val_df, test_df)
    )

    # Step 5: Save processed splits
    feature_names = get_feature_names(preprocessor)
    save_processed_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

    logger.info("=" * 60)
    logger.info("PHASE 3 PIPELINE: Complete. All artifacts saved.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
