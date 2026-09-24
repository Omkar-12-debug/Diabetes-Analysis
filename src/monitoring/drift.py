"""Clinical data and prediction drift monitoring engine using Evidently AI.

Loads baseline reference distributions, generates controlled synthetic in-distribution
and drifted clinical cohorts, executes statistical drift evaluations, and exports
structured JSON summaries and interactive HTML diagnostics.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report

from src.data.ingest import generate_deterministic_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/diabetes_prediction_dataset.csv")
DRIFT_REPORT_JSON_PATH = Path("reports/drift_report.json")
DRIFT_REPORT_HTML_PATH = Path("reports/figures/drift_report.html")

FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
TARGET_COLUMN = "diabetes"


def load_reference_data(sample_size: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Load baseline reference patient cohort from raw dataset or fallback generator.

    Args:
        sample_size: Number of records to sample for baseline reference.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame containing reference patient profiles.
    """
    if not RAW_DATA_PATH.exists():
        logger.warning("Raw dataset missing at %s. Generating deterministic baseline...", RAW_DATA_PATH)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_deterministic_dataset(RAW_DATA_PATH)

    df = pd.read_csv(RAW_DATA_PATH)
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if TARGET_COLUMN in df.columns:
        available_cols.append(TARGET_COLUMN)

    sampled = df[available_cols].sample(
        n=min(sample_size, len(df)),
        random_state=random_state,
        replace=False,
    ).reset_index(drop=True)

    logger.info("Loaded reference dataset with %d rows and %d features.", len(sampled), len(available_cols))
    return sampled


def generate_in_distribution_batch(
    reference_df: pd.DataFrame,
    sample_size: int = 1000,
    random_state: int = 123,
) -> pd.DataFrame:
    """Generate an in-distribution patient cohort where no drift is expected (p > 0.05).

    Args:
        reference_df: Baseline reference dataframe.
        sample_size: Number of records for the batch.
        random_state: Random seed for sampling.

    Returns:
        pd.DataFrame representing an in-distribution production batch.
    """
    batch = reference_df.sample(
        n=min(sample_size, len(reference_df)),
        random_state=random_state,
        replace=True,
    ).copy().reset_index(drop=True)
    return batch


def generate_drifted_batch(
    reference_df: pd.DataFrame,
    sample_size: int = 1000,
    glucose_shift: float = 35.0,
    hba1c_shift: float = 1.2,
    age_shift: float = 10.0,
    random_state: int = 999,
) -> pd.DataFrame:
    """Generate a drifted patient cohort with controlled clinical biomarker shifts.

    Applies physiological perturbations:
    - Mean blood glucose elevated by +35.0 mg/dL
    - Mean HbA1c level elevated by +1.2%
    - Mean age elevated by +10.0 years

    Args:
        reference_df: Baseline reference dataframe.
        sample_size: Number of records for the batch.
        glucose_shift: Additive shift to blood glucose level in mg/dL.
        hba1c_shift: Additive shift to HbA1c percentage.
        age_shift: Additive shift to patient age in years.
        random_state: Random seed for sampling.

    Returns:
        pd.DataFrame representing a clinically drifted production cohort.
    """
    batch = reference_df.sample(
        n=min(sample_size, len(reference_df)),
        random_state=random_state,
        replace=True,
    ).copy().reset_index(drop=True)

    if "blood_glucose_level" in batch.columns:
        batch["blood_glucose_level"] = (batch["blood_glucose_level"] + glucose_shift).clip(
            lower=50.0, upper=500.0
        )

    if "HbA1c_level" in batch.columns:
        batch["HbA1c_level"] = (batch["HbA1c_level"] + hba1c_shift).clip(
            lower=3.5, upper=18.0
        )

    if "age" in batch.columns:
        batch["age"] = (batch["age"] + age_shift).clip(
            lower=1.0, upper=110.0
        )

    # Shift target distribution proportionally to biomarker increases if present
    if TARGET_COLUMN in batch.columns:
        elevated_risk_mask = (batch["blood_glucose_level"] >= 180.0) | (batch["HbA1c_level"] >= 7.0)
        batch.loc[elevated_risk_mask, TARGET_COLUMN] = 1

    logger.info(
        "Generated drifted cohort (glucose=+%.1f, hba1c=+%.1f, age=+%.1f).",
        glucose_shift, hba1c_shift, age_shift,
    )
    return batch


def compute_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_share_threshold: float = 0.30,
) -> tuple[dict[str, Any], Report]:
    """Compute Evidently data drift report comparing reference vs current batches.

    Args:
        reference_df: Baseline patient cohort.
        current_df: Active production or simulated patient batch.
        drift_share_threshold: Proportion of drifted columns required to trigger dataset drift.

    Returns:
        Tuple of (structured_summary_dict, evidently_report_object).
    """
    # Exclude target if evaluating feature-only drift, or include TargetDriftPreset if present
    eval_cols = [c for c in FEATURE_COLUMNS if c in reference_df.columns and c in current_df.columns]
    ref_eval = reference_df[eval_cols].copy()
    curr_eval = current_df[eval_cols].copy()

    metrics: list[Any] = [DataDriftPreset(drift_share=drift_share_threshold)]

    has_target = (
        TARGET_COLUMN in reference_df.columns
        and TARGET_COLUMN in current_df.columns
    )
    if has_target:
        ref_eval[TARGET_COLUMN] = reference_df[TARGET_COLUMN]
        curr_eval[TARGET_COLUMN] = current_df[TARGET_COLUMN]
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=ref_eval, current_data=curr_eval)

    raw_dict = report.as_dict()
    metrics_list = raw_dict.get("metrics", [])

    # Metric 0 is DatasetDriftMetric
    dataset_drift_metric = metrics_list[0].get("result", {}) if len(metrics_list) > 0 else {}
    dataset_drift_detected = bool(dataset_drift_metric.get("dataset_drift", False))
    drift_share = float(dataset_drift_metric.get("share_of_drifted_columns", 0.0))
    drifted_cols_count = int(dataset_drift_metric.get("number_of_drifted_columns", 0))
    total_cols_count = int(dataset_drift_metric.get("number_of_columns", len(eval_cols)))

    # Metric 1 is DataDriftTable containing per-column drift p-values
    drift_table_metric = metrics_list[1].get("result", {}) if len(metrics_list) > 1 else {}
    drift_by_columns_raw = drift_table_metric.get("drift_by_columns", {})

    features_drift: dict[str, dict[str, Any]] = {}
    for col_name, col_data in drift_by_columns_raw.items():
        features_drift[col_name] = {
            "p_value": round(float(col_data.get("drift_score", 1.0)), 6),
            "drift_detected": bool(col_data.get("drift_detected", False)),
            "stat_test": str(col_data.get("stattest_name", col_data.get("stat_test", "unknown"))),
            "threshold": float(col_data.get("stattest_threshold", col_data.get("threshold", 0.05))),
        }

    summary: dict[str, Any] = {
        "dataset_drift_detected": dataset_drift_detected,
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": drift_share_threshold,
        "drifted_features_count": drifted_cols_count,
        "total_features_count": total_cols_count,
        "features": features_drift,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    return summary, report


def run_drift_pipeline(
    batch_type: str = "drifted",
    sample_size: int = 1000,
    export_json: bool = True,
    export_html: bool = True,
) -> dict[str, Any]:
    """Execute the end-to-end drift pipeline and export diagnostic reports.

    Args:
        batch_type: 'drifted' or 'in_distribution'.
        sample_size: Number of records in active cohort.
        export_json: Whether to write reports/drift_report.json.
        export_html: Whether to write reports/figures/drift_report.html.

    Returns:
        Structured drift report dictionary.
    """
    logger.info("=== Starting Data Drift Evaluation Pipeline (batch_type='%s') ===", batch_type)
    ref_df = load_reference_data(sample_size=3000)

    if batch_type == "in_distribution":
        curr_df = generate_in_distribution_batch(ref_df, sample_size=sample_size)
    else:
        curr_df = generate_drifted_batch(ref_df, sample_size=sample_size)

    summary, report = compute_data_drift(ref_df, curr_df)

    # Attach batch metadata
    summary["batch_type"] = batch_type

    if export_json:
        DRIFT_REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DRIFT_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Drift summary report exported to %s", DRIFT_REPORT_JSON_PATH)

    if export_html:
        DRIFT_REPORT_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(DRIFT_REPORT_HTML_PATH))
        logger.info("Interactive HTML drift dashboard exported to %s", DRIFT_REPORT_HTML_PATH)

    logger.info(
        "=== Drift Pipeline Complete: dataset_drift=%s (drift_share=%.2f%%, drifted=%d/%d) ===",
        summary["dataset_drift_detected"],
        summary["drift_share"] * 100,
        summary["drifted_features_count"],
        summary["total_features_count"],
    )
    return summary


def main() -> None:
    """CLI entrypoint for standalone drift pipeline execution."""
    run_drift_pipeline(batch_type="drifted")


if __name__ == "__main__":
    main()
