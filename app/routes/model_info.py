"""Model governance and metadata route."""

import json
from pathlib import Path
from fastapi import APIRouter

from app.schemas import ModelInfoResponse
from app.services.predictor import predictor_service
from src.evaluation.evaluate_all import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
)

router = APIRouter(tags=["Model Governance"])

QUALITY_GATE_REPORT_PATH = Path("reports/quality_gate_report.json")
MODEL_COMPARISON_PATH = Path("reports/model_comparison.json")


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Information & Governance",
    description="Returns registered model architecture, active champion alias, evaluation performance metrics, and quality gate approval status.",
)
def get_model_info() -> ModelInfoResponse:
    """Retrieve metadata and evaluation metrics for the active champion model."""
    metrics_summary = {}
    thresholds = {}
    status = "Active Champion"

    if QUALITY_GATE_REPORT_PATH.exists():
        try:
            with open(QUALITY_GATE_REPORT_PATH, "r", encoding="utf-8") as f:
                qg_data = json.load(f)
            metrics_summary = qg_data.get("candidate_metrics", {})
            thresholds = qg_data.get("thresholds", {})
            if qg_data.get("overall_decision") == "PASSED":
                status = "Promoted by Quality Gate (Active)"
        except Exception:
            pass

    if not metrics_summary and MODEL_COMPARISON_PATH.exists():
        try:
            with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
            metrics_summary = comp_data.get("champion", {}).get("test_metrics", {})
        except Exception:
            pass

    return ModelInfoResponse(
        model_name=REGISTERED_MODEL_NAME,
        alias=f"@{CHAMPION_ALIAS}",
        model_version=predictor_service.model_version,
        algorithm="XGBoost (XGBClassifier)",
        status=status,
        metrics_summary=metrics_summary,
        thresholds=thresholds,
    )
