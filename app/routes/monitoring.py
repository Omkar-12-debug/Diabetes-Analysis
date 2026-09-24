"""Routes for data/target drift evaluation and delayed-label model performance telemetry."""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from src.monitoring.drift import (
    DRIFT_REPORT_HTML_PATH,
    DRIFT_REPORT_JSON_PATH,
    compute_data_drift,
    generate_drifted_batch,
    generate_in_distribution_batch,
    load_reference_data,
    run_drift_pipeline,
)
from src.monitoring.performance import compute_production_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Drift"])


class DriftSimulateRequest(BaseModel):
    """Configuration for synthetic cohort generation and drift simulation."""

    batch_type: str = Field(
        default="drifted",
        description="Batch distribution type: 'drifted' or 'in_distribution'",
    )
    sample_size: int = Field(
        default=1000,
        ge=50,
        le=10000,
        description="Number of records to simulate in active cohort",
    )
    glucose_shift: float = Field(
        default=35.0,
        ge=0.0,
        le=200.0,
        description="Blood glucose level additive shift in mg/dL",
    )
    hba1c_shift: float = Field(
        default=1.2,
        ge=0.0,
        le=5.0,
        description="HbA1c level additive shift in percentage points",
    )
    age_shift: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        description="Patient age additive shift in years",
    )


@router.get(
    "/drift/report",
    summary="Get Data & Target Drift Report",
    description="Returns the latest computed Evidently AI data and prediction drift report, computing one if absent.",
)
def get_drift_report() -> dict[str, Any]:
    """Retrieve the latest data drift summary report."""
    if not DRIFT_REPORT_JSON_PATH.exists():
        logger.info("Drift report not found at %s. Executing default drift pipeline...", DRIFT_REPORT_JSON_PATH)
        return run_drift_pipeline(batch_type="drifted")

    with open(DRIFT_REPORT_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    data["html_report_path"] = str(DRIFT_REPORT_HTML_PATH)
    return data


@router.post(
    "/drift/simulate",
    summary="Simulate Clinical Cohort Drift",
    description="Generates an in-distribution or clinically drifted patient cohort and runs Evidently drift analysis.",
)
def simulate_drift(request: DriftSimulateRequest) -> dict[str, Any]:
    """Simulate a production batch and execute statistical data drift detection."""
    ref_df = load_reference_data(sample_size=3000)

    if request.batch_type == "in_distribution":
        curr_df = generate_in_distribution_batch(ref_df, sample_size=request.sample_size)
    else:
        curr_df = generate_drifted_batch(
            ref_df,
            sample_size=request.sample_size,
            glucose_shift=request.glucose_shift,
            hba1c_shift=request.hba1c_shift,
            age_shift=request.age_shift,
        )

    summary, report = compute_data_drift(ref_df, curr_df)
    summary["batch_type"] = request.batch_type

    # Save reports
    DRIFT_REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    DRIFT_REPORT_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(DRIFT_REPORT_HTML_PATH))
    summary["html_report_path"] = str(DRIFT_REPORT_HTML_PATH)

    return summary


@router.get(
    "/performance",
    summary="Delayed-Label Production Performance",
    description="Computes production classification metrics (PR-AUC, ROC-AUC, Recall, Precision, Brier Score) from verified diagnostic outcomes.",
)
def get_production_performance(
    model_version: str | None = Query(default=None, description="Filter evaluation to a specific model version"),
    threshold: float = Query(default=0.50, ge=0.0, le=1.0, description="Decision threshold for positive risk classification"),
    db: Session = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    """Calculate production performance metrics pairing historical inferences with verified outcomes."""
    return compute_production_performance(
        db=db,
        model_version=model_version,
        threshold=threshold,
    )
