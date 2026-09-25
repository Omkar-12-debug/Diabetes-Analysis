"""Production delayed-label performance evaluation engine.

Computes clinical classification metrics (PR-AUC, ROC-AUC, Recall, Precision,
F1-score) and probabilistic calibration (Brier score) by pairing historical
production predictions with delayed ground-truth diagnostic lab confirmations.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy.orm import Session

from app.db.models import AssessmentHistory, GroundTruthLabel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def query_labeled_assessments(
    db: Session,
    model_version: str | None = None,
) -> list[tuple[AssessmentHistory, GroundTruthLabel]]:
    """Query assessment history records that possess bound ground-truth diagnostic outcomes.

    Args:
        db: SQLAlchemy database session.
        model_version: Optional model version string filter.

    Returns:
        List of paired (AssessmentHistory, GroundTruthLabel) ORM records.
    """
    query = (
        db.query(AssessmentHistory, GroundTruthLabel)
        .join(GroundTruthLabel, AssessmentHistory.id == GroundTruthLabel.assessment_id)
    )

    if model_version:
        query = query.filter(AssessmentHistory.model_version == model_version)

    records = query.order_by(GroundTruthLabel.verified_at.asc()).all()
    logger.info("Found %d labeled assessment pairs in production database.", len(records))
    return records


def compute_production_performance(
    db: Session,
    model_version: str | None = None,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Calculate production performance metrics pairing historical inferences with verified outcomes.

    Args:
        db: Active SQLAlchemy database session.
        model_version: Optional model version filter.
        threshold: Decision threshold for binary risk categorization (default 0.50).

    Returns:
        Structured performance dictionary containing PR-AUC, ROC-AUC, Precision,
        Recall, F1-Score, Brier score, and sample cohort sizes.
    """
    pairs = query_labeled_assessments(db=db, model_version=model_version)

    if not pairs:
        logger.warning("No verified ground-truth labels available for performance evaluation.")
        return {
            "status": "insufficient_data",
            "message": "No verified ground truth labels available for evaluation.",
            "total_evaluated": 0,
            "positive_cases": 0,
            "negative_cases": 0,
            "metrics": {
                "pr_auc": None,
                "roc_auc": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "brier_score": None,
            },
            "threshold": threshold,
            "model_version": model_version,
            "evaluated_at": datetime.now(UTC).isoformat(),
        }

    y_true = np.array([gt.verified_diabetes_label for _, gt in pairs], dtype=int)
    y_prob = np.array([a.risk_probability for a, _ in pairs], dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    total_evaluated = len(y_true)
    positive_cases = int(np.sum(y_true))
    negative_cases = int(total_evaluated - positive_cases)

    # Calculate discrimination metrics if both binary classes exist in evaluation set
    has_both_classes = len(np.unique(y_true)) > 1
    roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4) if has_both_classes else None
    pr_auc = round(float(average_precision_score(y_true, y_prob)), 4) if has_both_classes else None

    # Calculate point metrics
    precision = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    brier = round(float(brier_score_loss(y_true, y_prob)), 4)

    logger.info(
        "Production Performance Evaluated (N=%d): PR-AUC=%s, ROC-AUC=%s, Recall=%.4f, Precision=%.4f, Brier=%.4f",
        total_evaluated, pr_auc, roc_auc, recall, precision, brier,
    )

    return {
        "status": "evaluated",
        "total_evaluated": total_evaluated,
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "threshold": threshold,
        "model_version": model_version or "all",
        "metrics": {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "brier_score": brier,
        },
        "calibration_status": "well_calibrated" if brier <= 0.15 else "poorly_calibrated",
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
