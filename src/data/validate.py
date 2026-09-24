"""Automated validation suite and data auditing engine for diabetes dataset.

Executes schema, type, range, categorical, duplicate, and label-conflict audits.
Generates structured validation reports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expected Schema Constants
EXPECTED_COLUMNS: List[str] = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]

NUMERIC_COLUMNS: List[str] = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]

PERMITTED_CATEGORIES: Dict[str, List[str]] = {
    "gender": ["Female", "Male", "Other"],
    "smoking_history": ["never", "No Info", "current", "former", "ever", "not current"],
}

BINARY_COLUMNS: List[str] = ["hypertension", "heart_disease", "diabetes"]

PHYSIOLOGICAL_RANGES: Dict[str, Dict[str, float]] = {
    "age": {"min": 0.0, "max": 120.0},
    "bmi": {"min": 10.0, "max": 100.0},
    "HbA1c_level": {"min": 3.0, "max": 20.0},
    "blood_glucose_level": {"min": 30.0, "max": 500.0},
}


class DataValidationError(Exception):
    """Raised when data fails validation checks."""

    pass


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has the exact expected columns and datatypes.

    Args:
        df: Input DataFrame to check.

    Raises:
        DataValidationError: If columns do not match expected schema.
    """
    current_cols = list(df.columns)
    if current_cols != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(current_cols)
        unexpected = set(current_cols) - set(EXPECTED_COLUMNS)
        raise DataValidationError(
            f"Schema validation failed. Missing columns: {sorted(missing)}, "
            f"Unexpected columns: {sorted(unexpected)}"
        )

    # Check that numeric columns are indeed numeric
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"Column '{col}' expected numeric type, got {df[col].dtype}."
            )


def validate_missing_values(df: pd.DataFrame) -> None:
    """Validate that there are 0 missing/null values in the dataset.

    Args:
        df: Input DataFrame to check.

    Raises:
        DataValidationError: If any missing values are found.
    """
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        missing_detail = {k: int(v) for k, v in null_counts.items() if v > 0}
        raise DataValidationError(
            f"Missing value assertion failed. Found {total_nulls} null values: {missing_detail}"
        )


def validate_physiological_ranges(df: pd.DataFrame) -> None:
    """Validate that physiological features fall within valid human physiological boundaries.

    Args:
        df: Input DataFrame to check.

    Raises:
        DataValidationError: If any values fall outside accepted ranges.
    """
    for col, bounds in PHYSIOLOGICAL_RANGES.items():
        min_val = bounds["min"]
        max_val = bounds["max"]
        out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
        if not out_of_bounds.empty:
            count = len(out_of_bounds)
            sample_violators = out_of_bounds[col].head(3).tolist()
            raise DataValidationError(
                f"Physiological range check failed for '{col}'. "
                f"Expected between {min_val} and {max_val}, but found {count} violating records. "
                f"Sample violations: {sample_violators}"
            )


def validate_categorical_values(df: pd.DataFrame) -> None:
    """Validate that categorical features contain only approved values.

    Args:
        df: Input DataFrame to check.

    Raises:
        DataValidationError: If unexpected categories are detected.
    """
    for col, allowed in PERMITTED_CATEGORIES.items():
        unique_vals = set(df[col].dropna().unique())
        invalid_vals = unique_vals - set(allowed)
        if invalid_vals:
            raise DataValidationError(
                f"Categorical check failed for '{col}'. "
                f"Found unpermitted categories: {sorted(invalid_vals)}. "
                f"Allowed categories: {allowed}"
            )


def validate_binary_flags(df: pd.DataFrame) -> None:
    """Validate that binary flag columns contain strictly 0 or 1.

    Args:
        df: Input DataFrame to check.

    Raises:
        DataValidationError: If non-binary values are found.
    """
    for col in BINARY_COLUMNS:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            invalid = unique_vals - {0, 1}
            raise DataValidationError(
                f"Binary check failed for '{col}'. Expected {0, 1}, but found: {sorted(invalid)}"
            )


def validate_dataset(df: pd.DataFrame) -> None:
    """Run full validation pipeline on the dataset.

    Args:
        df: Input DataFrame.

    Raises:
        DataValidationError: If any validation test fails.
    """
    validate_schema(df)
    validate_missing_values(df)
    validate_physiological_ranges(df)
    validate_categorical_values(df)
    validate_binary_flags(df)
    logger.info("All strict validation rules passed successfully.")


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute in-depth quality auditing to discover and document known data anomalies.

    Identifies:
        - Exact duplicate rows
        - Infant records (age < 1 year)
        - Discordant target labels for identical feature vectors
        - Demographic subgroup sample sizes ('Other' gender, 'No Info' smoking)
        - Class balance / imbalance metrics

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, Any]: Audit findings dictionary.
    """
    total_records = len(df)

    # 1. Exact Duplicate Rows
    num_duplicates = int(df.duplicated().sum())
    pct_duplicates = round((num_duplicates / total_records) * 100, 4)

    # 2. Age < 1 Year (Infant records)
    infant_records = df[df["age"] < 1.0]
    num_infants = int(len(infant_records))
    pct_infants = round((num_infants / total_records) * 100, 4)

    # 3. Conflicting Feature Combinations with Discordant Labels
    feature_cols = [c for c in EXPECTED_COLUMNS if c != "diabetes"]
    grouped = df.groupby(feature_cols)["diabetes"].nunique()
    conflicting_groups = grouped[grouped > 1]
    num_conflicting_groups = int(len(conflicting_groups))

    # Total rows affected by conflicting groups
    if num_conflicting_groups > 0:
        conflicting_indices = df.set_index(feature_cols).index.isin(conflicting_groups.index)
        num_conflicting_rows = int(conflicting_indices.sum())
    else:
        num_conflicting_rows = 0

    # 4. Demographic Subgroups
    gender_counts = df["gender"].value_counts().to_dict()
    other_gender_count = int(gender_counts.get("Other", 0))
    other_gender_pct = round((other_gender_count / total_records) * 100, 4)

    smoking_counts = df["smoking_history"].value_counts().to_dict()
    no_info_smoking_count = int(smoking_counts.get("No Info", 0))
    no_info_smoking_pct = round((no_info_smoking_count / total_records) * 100, 4)

    # 5. Class Imbalance
    diabetes_counts = df["diabetes"].value_counts().to_dict()
    non_diabetic_count = int(diabetes_counts.get(0, 0))
    diabetic_count = int(diabetes_counts.get(1, 0))
    non_diabetic_pct = round((non_diabetic_count / total_records) * 100, 2)
    diabetic_pct = round((diabetic_count / total_records) * 100, 2)

    audit_results = {
        "total_records": total_records,
        "exact_duplicates": {
            "count": num_duplicates,
            "percentage": pct_duplicates,
        },
        "infant_records_age_under_1": {
            "count": num_infants,
            "percentage": pct_infants,
        },
        "conflicting_target_feature_groups": {
            "group_count": num_conflicting_groups,
            "affected_rows": num_conflicting_rows,
        },
        "demographic_subgroups": {
            "gender_counts": gender_counts,
            "other_gender": {
                "count": other_gender_count,
                "percentage": other_gender_pct,
            },
            "smoking_history_counts": smoking_counts,
            "no_info_smoking": {
                "count": no_info_smoking_count,
                "percentage": no_info_smoking_pct,
            },
        },
        "class_distribution": {
            "non_diabetic_0": {
                "count": non_diabetic_count,
                "percentage": non_diabetic_pct,
            },
            "diabetic_1": {
                "count": diabetic_count,
                "percentage": diabetic_pct,
            },
            "imbalance_ratio": f"{non_diabetic_count}:{diabetic_count} (~{round(non_diabetic_count/max(diabetic_count, 1), 1)}:1)",
        },
    }

    return audit_results


def run_validation_and_audit(
    df: Optional[pd.DataFrame] = None,
    output_report_path: str | Path = "reports/data_validation_report.json",
) -> Dict[str, Any]:
    """Run full validation checks and auditing suite, saving structured report to JSON.

    Args:
        df: Input DataFrame. If None, loads from data/raw/diabetes_prediction_dataset.csv.
        output_report_path: Destination path for JSON report.

    Returns:
        Dict[str, Any]: The complete audit report dictionary.
    """
    if df is None:
        raw_path = Path("data/raw/diabetes_prediction_dataset.csv")
        if not raw_path.exists():
            from src.data.ingest import ingest_data
            df = ingest_data()
        else:
            df = pd.read_csv(raw_path)

    logger.info("Executing strict validation suite...")
    validate_dataset(df)

    logger.info("Executing exploratory quality audit...")
    audit_findings = audit_data_quality(df)

    report = {
        "validation_status": "PASSED",
        "dataset_shape": {
            "rows": len(df),
            "columns": len(df.columns),
        },
        "schema_checks": {
            "expected_columns": EXPECTED_COLUMNS,
            "missing_values_count": 0,
            "physiological_ranges": PHYSIOLOGICAL_RANGES,
            "permitted_categories": PERMITTED_CATEGORIES,
            "binary_flags": BINARY_COLUMNS,
        },
        "audit_findings": audit_findings,
    }

    report_path = Path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report saved successfully to %s", report_path)
    return report


if __name__ == "__main__":
    from src.data.ingest import ingest_data

    raw_df = ingest_data()
    run_validation_and_audit(raw_df)
