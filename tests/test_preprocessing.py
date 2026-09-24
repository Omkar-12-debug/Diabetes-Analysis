"""Test suite for preprocessing pipeline, splitting, and feature engineering.

Tests cover split shapes and stratification, feature derivation correctness,
pipeline leakage safety, unseen category handling, determinism, and serialization.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    AGE_LABELS,
    BMI_LABELS,
    ENGINEERED_FEATURE_NAMES,
    add_engineered_features,
)
from src.preprocessing.pipeline import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_preprocessor,
    fit_and_transform_splits,
    get_feature_names,
)
from src.preprocessing.split import stratified_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_dataset() -> pd.DataFrame:
    """Load the full raw dataset for integration tests."""
    from src.data.ingest import ingest_data
    return ingest_data()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic dataset with known properties for unit tests."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "gender": np.random.choice(["Female", "Male", "Other"], size=n, p=[0.58, 0.41, 0.01]),
        "age": np.random.uniform(1, 90, size=n).round(1),
        "hypertension": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "heart_disease": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "smoking_history": np.random.choice(
            ["never", "No Info", "current", "former", "ever", "not current"],
            size=n,
        ),
        "bmi": np.random.uniform(12, 60, size=n).round(1),
        "HbA1c_level": np.random.uniform(3.5, 9.0, size=n).round(1),
        "blood_glucose_level": np.random.choice(
            [80, 100, 126, 140, 160, 200, 260, 300], size=n
        ),
        "diabetes": np.random.choice([0, 1], size=n, p=[0.915, 0.085]),
    })


# ---------------------------------------------------------------------------
# Split Tests
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    """Tests for stratified train/val/test splitting."""

    def test_split_shapes_with_dedup(self, raw_dataset: pd.DataFrame):
        """Assert split shapes match 70/15/15 ratio after deduplication."""
        train, val, test = stratified_split(raw_dataset, drop_duplicates=True)
        total = len(train) + len(val) + len(test)

        # After deduplication: 100,000 - 3,854 = 96,146 rows
        assert total == 96146, f"Expected 96146 total rows after dedup, got {total}"
        assert abs(len(train) / total - 0.70) < 0.01
        assert abs(len(val) / total - 0.15) < 0.01
        assert abs(len(test) / total - 0.15) < 0.01

    def test_split_shapes_without_dedup(self, sample_df: pd.DataFrame):
        """Assert split shapes match 70/15/15 ratio without deduplication."""
        train, val, test = stratified_split(sample_df, drop_duplicates=False)
        total = len(train) + len(val) + len(test)

        assert total == 1000
        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150

    def test_class_distribution_preserved(self, raw_dataset: pd.DataFrame):
        """Assert each partition preserves ~8.5% ± 1% positive class rate."""
        train, val, test = stratified_split(raw_dataset, drop_duplicates=True)

        for name, split_df in [("train", train), ("val", val), ("test", test)]:
            pos_rate = split_df["diabetes"].mean()
            assert 0.075 <= pos_rate <= 0.095, (
                f"{name} split positive rate {pos_rate:.4f} outside acceptable range [0.075, 0.095]."
            )

    def test_no_index_overlap(self, sample_df: pd.DataFrame):
        """Assert no row indices overlap between partitions."""
        # Use original indices for overlap test
        working = sample_df.copy()
        working["_orig_idx"] = range(len(working))
        train, val, test = stratified_split(working, drop_duplicates=False)

        train_idx = set(train["_orig_idx"])
        val_idx = set(val["_orig_idx"])
        test_idx = set(test["_orig_idx"])

        assert len(train_idx & val_idx) == 0, "Train and val share indices"
        assert len(train_idx & test_idx) == 0, "Train and test share indices"
        assert len(val_idx & test_idx) == 0, "Val and test share indices"

    def test_invalid_ratios_raise_error(self, sample_df: pd.DataFrame):
        """Assert ValueError when ratios don't sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            stratified_split(sample_df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Tests for derived clinical feature creation."""

    def test_all_engineered_columns_created(self, sample_df: pd.DataFrame):
        """Assert all 5 derived columns are present."""
        result = add_engineered_features(sample_df)
        for col in ENGINEERED_FEATURE_NAMES:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_age_group_binning(self):
        """Assert age_group produces correct bins for known inputs."""
        df = pd.DataFrame({
            "gender": ["Male"] * 5,
            "age": [5.0, 25.0, 40.0, 55.0, 70.0],
            "hypertension": [0] * 5,
            "heart_disease": [0] * 5,
            "smoking_history": ["never"] * 5,
            "bmi": [22.0] * 5,
            "HbA1c_level": [5.0] * 5,
            "blood_glucose_level": [100] * 5,
            "diabetes": [0] * 5,
        })
        result = add_engineered_features(df)
        expected = ["<18", "18-35", "36-50", "51-65", "65+"]
        assert list(result["age_group"]) == expected

    def test_bmi_category_binning(self):
        """Assert bmi_category produces correct clinical tiers."""
        df = pd.DataFrame({
            "gender": ["Female"] * 4,
            "age": [40.0] * 4,
            "hypertension": [0] * 4,
            "heart_disease": [0] * 4,
            "smoking_history": ["never"] * 4,
            "bmi": [16.0, 22.0, 27.5, 35.0],
            "HbA1c_level": [5.0] * 4,
            "blood_glucose_level": [100] * 4,
            "diabetes": [0] * 4,
        })
        result = add_engineered_features(df)
        expected = ["Underweight", "Normal", "Overweight", "Obese"]
        assert list(result["bmi_category"]) == expected

    def test_cardiometabolic_risk(self):
        """Assert cardiometabolic_risk equals hypertension + heart_disease."""
        df = pd.DataFrame({
            "gender": ["Male"] * 4,
            "age": [50.0] * 4,
            "hypertension": [0, 1, 0, 1],
            "heart_disease": [0, 0, 1, 1],
            "smoking_history": ["never"] * 4,
            "bmi": [25.0] * 4,
            "HbA1c_level": [5.5] * 4,
            "blood_glucose_level": [120] * 4,
            "diabetes": [0] * 4,
        })
        result = add_engineered_features(df)
        assert list(result["cardiometabolic_risk"]) == [0, 1, 1, 2]

    def test_interaction_terms(self):
        """Assert interaction terms are correct products."""
        df = pd.DataFrame({
            "gender": ["Male"],
            "age": [45.0],
            "hypertension": [0],
            "heart_disease": [0],
            "smoking_history": ["never"],
            "bmi": [30.0],
            "HbA1c_level": [6.5],
            "blood_glucose_level": [180],
            "diabetes": [0],
        })
        result = add_engineered_features(df)
        assert result["glucose_hba1c_interaction"].iloc[0] == pytest.approx(180 * 6.5)
        assert result["age_bmi_interaction"].iloc[0] == pytest.approx(45.0 * 30.0)

    def test_original_columns_preserved(self, sample_df: pd.DataFrame):
        """Assert engineering does not modify or drop original columns."""
        result = add_engineered_features(sample_df)
        for col in sample_df.columns:
            assert col in result.columns
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:
    """Tests for the ColumnTransformer preprocessing pipeline."""

    def _prepare_engineered_splits(self, sample_df: pd.DataFrame):
        """Helper: split and engineer features on sample data."""
        train, val, test = stratified_split(sample_df, drop_duplicates=False)
        train = add_engineered_features(train)
        val = add_engineered_features(val)
        test = add_engineered_features(test)
        return train, val, test

    def test_no_nan_in_transformed_output(self, sample_df: pd.DataFrame):
        """Assert pipeline produces no NaN values in any split."""
        train, val, test = self._prepare_engineered_splits(sample_df)
        X_train, X_val, X_test, y_train, y_val, y_test, _ = fit_and_transform_splits(
            train, val, test,
            save_path=Path("models/test_preprocessor_nonan.joblib"),
        )

        assert not np.isnan(X_train).any(), "NaN found in transformed train"
        assert not np.isnan(X_val).any(), "NaN found in transformed val"
        assert not np.isnan(X_test).any(), "NaN found in transformed test"

    def test_train_fitted_scaler_values(self, sample_df: pd.DataFrame):
        """Assert numeric features in training set are approximately standard-scaled."""
        train, val, test = self._prepare_engineered_splits(sample_df)
        X_train, _, _, _, _, _, preprocessor = fit_and_transform_splits(
            train, val, test,
            save_path=Path("models/test_preprocessor_scaler.joblib"),
        )

        # Numeric features are the first len(NUMERIC_FEATURES) columns
        n_numeric = len(NUMERIC_FEATURES)
        train_numeric = X_train[:, :n_numeric]

        # Mean should be approximately 0, std approximately 1
        means = train_numeric.mean(axis=0)
        stds = train_numeric.std(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=0.05)
        np.testing.assert_allclose(stds, 1.0, atol=0.15)

    def test_handle_unknown_categories(self, sample_df: pd.DataFrame):
        """Assert unseen categories in test data produce zeros (not errors)."""
        train, val, test = self._prepare_engineered_splits(sample_df)

        # Inject unseen category into test set
        test = test.copy()
        test.loc[0, "gender"] = "Nonbinary"  # Not in training categories

        # Should not raise any error
        X_train, X_val, X_test, _, _, _, _ = fit_and_transform_splits(
            train, val, test,
            save_path=Path("models/test_preprocessor_unknown.joblib"),
        )

        assert not np.isnan(X_test).any(), "NaN found after unseen category"

    def test_determinism_across_runs(self, sample_df: pd.DataFrame):
        """Assert two consecutive runs with same random_state produce identical results."""
        # Run 1
        train1, val1, test1 = stratified_split(sample_df, random_state=42, drop_duplicates=False)
        train1 = add_engineered_features(train1)
        val1 = add_engineered_features(val1)
        test1 = add_engineered_features(test1)
        X_tr1, X_v1, X_te1, y_tr1, y_v1, y_te1, _ = fit_and_transform_splits(
            train1, val1, test1,
            save_path=Path("models/test_preprocessor_det1.joblib"),
        )

        # Run 2
        train2, val2, test2 = stratified_split(sample_df, random_state=42, drop_duplicates=False)
        train2 = add_engineered_features(train2)
        val2 = add_engineered_features(val2)
        test2 = add_engineered_features(test2)
        X_tr2, X_v2, X_te2, y_tr2, y_v2, y_te2, _ = fit_and_transform_splits(
            train2, val2, test2,
            save_path=Path("models/test_preprocessor_det2.joblib"),
        )

        np.testing.assert_array_equal(X_tr1, X_tr2, err_msg="Train arrays differ between runs")
        np.testing.assert_array_equal(X_v1, X_v2, err_msg="Val arrays differ between runs")
        np.testing.assert_array_equal(X_te1, X_te2, err_msg="Test arrays differ between runs")
        np.testing.assert_array_equal(y_tr1, y_tr2, err_msg="Train labels differ between runs")
        np.testing.assert_array_equal(y_v1, y_v2, err_msg="Val labels differ between runs")
        np.testing.assert_array_equal(y_te1, y_te2, err_msg="Test labels differ between runs")


class TestSerialization:
    """Tests for preprocessor serialization and reuse."""

    def test_serialized_preprocessor_loads_and_transforms(self, sample_df: pd.DataFrame):
        """Assert saved preprocessor can be loaded and used on new data."""
        train, val, test = stratified_split(sample_df, drop_duplicates=False)
        train = add_engineered_features(train)
        val = add_engineered_features(val)
        test = add_engineered_features(test)

        save_path = Path("models/test_preprocessor_serial.joblib")
        _, _, X_test_orig, _, _, _, _ = fit_and_transform_splits(
            train, val, test, save_path=save_path,
        )

        # Load and re-transform
        loaded_preprocessor = joblib.load(save_path)
        X_test_reloaded = loaded_preprocessor.transform(
            test.drop(columns=[TARGET_COL])
        )

        np.testing.assert_array_equal(X_test_orig, X_test_reloaded)


class TestIntegrationPipeline:
    """Integration test against the full raw dataset."""

    def test_full_pipeline_on_raw_data(self, raw_dataset: pd.DataFrame):
        """End-to-end pipeline: split → engineer → fit → transform on real data."""
        train, val, test = stratified_split(raw_dataset, drop_duplicates=True)

        # Verify deduplication
        total = len(train) + len(val) + len(test)
        assert total == 96146

        # Engineer features
        train = add_engineered_features(train)
        val = add_engineered_features(val)
        test = add_engineered_features(test)

        # Fit and transform
        save_path = Path("models/test_preprocessor_integration.joblib")
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = (
            fit_and_transform_splits(train, val, test, save_path=save_path)
        )

        # Shape checks
        assert X_train.shape[0] == len(train)
        assert X_val.shape[0] == len(val)
        assert X_test.shape[0] == len(test)
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

        # No NaN
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_val).any()
        assert not np.isnan(X_test).any()

        # Feature names
        feature_names = get_feature_names(preprocessor)
        assert len(feature_names) == X_train.shape[1]
