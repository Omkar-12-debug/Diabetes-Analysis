"""Test suite for data validation and auditing engine."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import IngestionError, ingest_data, verify_dataset_integrity
from src.data.validate import (
    DataValidationError,
    audit_data_quality,
    run_validation_and_audit,
    validate_binary_flags,
    validate_categorical_values,
    validate_dataset,
    validate_missing_values,
    validate_physiological_ranges,
    validate_schema,
)


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing a clean, valid sample DataFrame."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Other", "Female"],
            "age": [45.0, 60.0, 25.0, 0.8],
            "hypertension": [0, 1, 0, 0],
            "heart_disease": [0, 0, 0, 0],
            "smoking_history": ["never", "current", "No Info", "former"],
            "bmi": [25.4, 30.2, 22.0, 18.5],
            "HbA1c_level": [5.7, 7.2, 4.5, 5.0],
            "blood_glucose_level": [100, 180, 85, 90],
            "diabetes": [0, 1, 0, 0],
        }
    )


def test_valid_data_passes(sample_valid_df: pd.DataFrame):
    """Ensure valid DataFrame passes all validation checks without raising errors."""
    validate_dataset(sample_valid_df)


# Schema & Type Tests
def test_schema_missing_column(sample_valid_df: pd.DataFrame):
    """Ensure schema validation fails when a required column is missing."""
    corrupted_df = sample_valid_df.drop(columns=["blood_glucose_level"])
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate_schema(corrupted_df)


def test_schema_unexpected_column(sample_valid_df: pd.DataFrame):
    """Ensure schema validation fails when an extraneous column is present."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df["unauthorized_column"] = 123
    with pytest.raises(DataValidationError, match="Unexpected columns"):
        validate_schema(corrupted_df)


def test_schema_invalid_dtype(sample_valid_df: pd.DataFrame):
    """Ensure schema validation fails if numeric columns have string types."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df["age"] = ["forty", "sixty", "twenty", "one"]
    with pytest.raises(DataValidationError, match="expected numeric type"):
        validate_schema(corrupted_df)


# Missing Value Tests
def test_missing_values_rejection(sample_valid_df: pd.DataFrame):
    """Ensure null/NaN values are strictly rejected."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df.loc[0, "bmi"] = np.nan
    with pytest.raises(DataValidationError, match="Missing value assertion failed"):
        validate_missing_values(corrupted_df)


# Physiological Range Tests
@pytest.mark.parametrize(
    "col,invalid_val",
    [
        ("age", -1.0),
        ("age", 135.0),
        ("bmi", 5.0),
        ("bmi", 115.0),
        ("HbA1c_level", 2.1),
        ("HbA1c_level", 25.0),
        ("blood_glucose_level", 15.0),
        ("blood_glucose_level", 600.0),
    ],
)
def test_physiological_range_violations(
    sample_valid_df: pd.DataFrame, col: str, invalid_val: float
):
    """Ensure out-of-range physiological values are caught and raise DataValidationError."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df.loc[0, col] = invalid_val
    with pytest.raises(DataValidationError, match=f"Physiological range check failed for '{col}'"):
        validate_physiological_ranges(corrupted_df)


# Categorical Tests
def test_invalid_gender_rejection(sample_valid_df: pd.DataFrame):
    """Ensure invalid gender strings are rejected."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df.loc[0, "gender"] = "UnknownGender"
    with pytest.raises(DataValidationError, match="Categorical check failed for 'gender'"):
        validate_categorical_values(corrupted_df)


def test_invalid_smoking_history_rejection(sample_valid_df: pd.DataFrame):
    """Ensure invalid smoking categories are rejected."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df.loc[0, "smoking_history"] = "cigar_smoker"
    with pytest.raises(DataValidationError, match="Categorical check failed for 'smoking_history'"):
        validate_categorical_values(corrupted_df)


# Binary Flag Tests
@pytest.mark.parametrize("col", ["hypertension", "heart_disease", "diabetes"])
def test_non_binary_flag_rejection(sample_valid_df: pd.DataFrame, col: str):
    """Ensure binary flags reject numbers other than 0 and 1."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df.loc[0, col] = 2
    with pytest.raises(DataValidationError, match=f"Binary check failed for '{col}'"):
        validate_binary_flags(corrupted_df)


# Audit Quality Functions Test
def test_audit_data_quality_metrics():
    """Verify auditing engine detects duplicates, infants, and conflicting labels correctly."""
    synthetic_df = pd.DataFrame(
        {
            "gender": ["Female", "Female", "Male", "Male", "Other"],
            "age": [0.5, 0.5, 45.0, 45.0, 30.0],
            "hypertension": [0, 0, 0, 0, 0],
            "heart_disease": [0, 0, 0, 0, 0],
            "smoking_history": ["never", "never", "No Info", "No Info", "current"],
            "bmi": [20.0, 20.0, 25.0, 25.0, 22.0],
            "HbA1c_level": [5.0, 5.0, 6.0, 6.0, 5.5],
            "blood_glucose_level": [100, 100, 120, 120, 110],
            # Row 0 and 1: exact duplicate
            # Row 2 and 3: identical features, but conflicting target labels (0 vs 1)
            "diabetes": [0, 0, 0, 1, 0],
        }
    )

    audit = audit_data_quality(synthetic_df)

    assert audit["total_records"] == 5
    assert audit["exact_duplicates"]["count"] == 1  # 1 duplicate row
    assert audit["infant_records_age_under_1"]["count"] == 2  # rows 0 and 1 have age 0.5
    assert audit["conflicting_target_feature_groups"]["group_count"] == 1  # rows 2 & 3
    assert audit["conflicting_target_feature_groups"]["affected_rows"] == 2
    assert audit["demographic_subgroups"]["other_gender"]["count"] == 1
    assert audit["demographic_subgroups"]["no_info_smoking"]["count"] == 2
    assert audit["class_distribution"]["non_diabetic_0"]["count"] == 4
    assert audit["class_distribution"]["diabetic_1"]["count"] == 1


# Integration Tests on Raw Dataset
def test_raw_dataset_ingestion_and_auditing():
    """Integration test validating the full raw dataset in data/raw/."""
    raw_df = ingest_data()
    assert len(raw_df) == 100000
    assert len(raw_df.columns) == 9

    # Strict validation must pass
    validate_dataset(raw_df)

    # Audit findings must match exact known baseline metrics
    audit = audit_data_quality(raw_df)
    assert audit["total_records"] == 100000
    assert audit["exact_duplicates"]["count"] == 3854
    assert audit["infant_records_age_under_1"]["count"] == 911
    assert audit["conflicting_target_feature_groups"]["group_count"] == 91
    assert audit["demographic_subgroups"]["other_gender"]["count"] == 18
    assert audit["demographic_subgroups"]["no_info_smoking"]["count"] == 35816
    assert audit["class_distribution"]["non_diabetic_0"]["count"] == 91500
    assert audit["class_distribution"]["diabetic_1"]["count"] == 8500


def test_ingestion_error_on_corrupted_shape():
    """Verify IngestionError is raised when dataset dimensions are wrong."""
    small_df = pd.DataFrame({"gender": ["Male"]})
    with pytest.raises(IngestionError):
        verify_dataset_integrity(small_df)
