"""Demographic fairness analysis and subgroup performance auditing.

Evaluates recall, false positive rate (FPR), precision, F1-score, and PR-AUC across
demographic slices (gender, age_group). Explicitly handles statistical limitations
and warnings for small cohorts (e.g., 'Other' gender).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIN_SAMPLE_THRESHOLD = 30  # Threshold below which sample size warning is flagged


def reconstruct_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct original categorical labels from one-hot encoded columns.

    Args:
        df: DataFrame containing one-hot encoded 'cat__gender_*' and 'cat__age_group_*'.

    Returns:
        DataFrame with 'gender' and 'age_group' string columns added.
    """
    df = df.copy()

    # Reconstruct Gender
    gender_cols = [c for c in df.columns if c.startswith("cat__gender_")]
    if gender_cols:
        df["gender"] = df[gender_cols].idxmax(axis=1).str.replace("cat__gender_", "", regex=False)
    elif "gender" not in df.columns:
        df["gender"] = "Unknown"

    # Reconstruct Age Group
    age_cols = [c for c in df.columns if c.startswith("cat__age_group_")]
    if age_cols:
        df["age_group"] = df[age_cols].idxmax(axis=1).str.replace("cat__age_group_", "", regex=False)
    elif "age_group" not in df.columns:
        df["age_group"] = "Unknown"

    return df


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    subgroup_name: str,
    slice_label: str,
    min_samples: int = MIN_SAMPLE_THRESHOLD,
) -> Dict[str, Any]:
    """Compute performance and fairness metrics for a single demographic subgroup slice.

    Args:
        y_true: Ground truth binary labels for the slice.
        y_prob: Predicted probabilities for the slice.
        y_pred: Binary predictions for the slice.
        subgroup_name: Name of demographic attribute (e.g., "gender").
        slice_label: Specific value (e.g., "Female", "Male", "Other").
        min_samples: Sample size threshold for flagging statistical unreliability.

    Returns:
        Dict containing support, class counts, recall, FPR, F1, PR-AUC, and warnings.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    support = int(len(y_true))
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))

    sample_size_warning = support < min_samples
    warning_message = None
    if sample_size_warning:
        warning_message = (
            f"Subgroup '{slice_label}' has very small sample size (N={support} < {min_samples}). "
            f"Metrics have high variance and cannot be used for definitive statistical inferences."
        )
        logger.warning(warning_message)

    # Confusion matrix components
    if support > 0:
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    else:
        tn = fp = fn = tp = 0

    # Recall (Sensitivity / TPR)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None

    # False Positive Rate (FPR = FP / (FP + TN))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None

    # Precision
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (0.0 if positives > 0 else None)

    # F1-Score
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if support > 0 else None

    # PR-AUC
    pr_auc = None
    if positives > 0 and negatives > 0:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = None

    # ROC-AUC
    roc_auc = None
    if positives > 0 and negatives > 0:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None

    return {
        "demographic_attribute": subgroup_name,
        "subgroup_value": str(slice_label),
        "support": support,
        "positives": positives,
        "negatives": negatives,
        "positive_prevalence": float(positives / support) if support > 0 else 0.0,
        "recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "sample_size_warning": sample_size_warning,
        "warning_message": warning_message,
    }


def evaluate_fairness_for_attribute(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    attribute_col: str,
    min_samples: int = MIN_SAMPLE_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate fairness across all unique categories of a demographic attribute.

    Args:
        df: DataFrame containing the demographic attribute column.
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        y_pred: Binary predictions.
        attribute_col: Column name to group by (e.g., 'gender', 'age_group').
        min_samples: Minimum count threshold for statistical reliability.

    Returns:
        Dict with slices breakdown and disparity analysis across well-supported slices.
    """
    slices: Dict[str, Dict[str, Any]] = {}
    unique_values = sorted(df[attribute_col].unique().tolist())

    for val in unique_values:
        mask = (df[attribute_col] == val).values
        sub_y_true = y_true[mask]
        sub_y_prob = y_prob[mask]
        sub_y_pred = y_pred[mask]

        metrics = compute_subgroup_metrics(
            sub_y_true,
            sub_y_prob,
            sub_y_pred,
            subgroup_name=attribute_col,
            slice_label=str(val),
            min_samples=min_samples,
        )
        slices[str(val)] = metrics

    # Disparity calculations on statistically valid subgroups (support >= min_samples)
    valid_slices = [s for s in slices.values() if not s["sample_size_warning"]]

    disparities = {}
    if len(valid_slices) >= 2:
        valid_recalls = [s["recall"] for s in valid_slices if s["recall"] is not None]
        valid_fprs = [s["false_positive_rate"] for s in valid_slices if s["false_positive_rate"] is not None]

        if valid_recalls:
            min_recall = min(valid_recalls)
            max_recall = max(valid_recalls)
            disparities["recall_disparity_diff"] = float(max_recall - min_recall)
            disparities["recall_disparity_ratio"] = float(min_recall / max_recall) if max_recall > 0 else 0.0

        if valid_fprs:
            min_fpr = min(valid_fprs)
            max_fpr = max(valid_fprs)
            disparities["fpr_disparity_diff"] = float(max_fpr - min_fpr)
            disparities["fpr_disparity_ratio"] = float(max_fpr / min_fpr) if min_fpr > 0 else None

    return {
        "attribute": attribute_col,
        "slices": slices,
        "disparities": disparities,
    }


def evaluate_fairness(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    demographic_cols: Optional[List[str]] = None,
    min_samples: int = MIN_SAMPLE_THRESHOLD,
) -> Dict[str, Any]:
    """Complete demographic fairness audit across gender and age groups.

    Args:
        df: Feature DataFrame (with one-hot encoded or raw columns).
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        y_pred: Binary predictions.
        demographic_cols: Demographic columns to audit (default: ['gender', 'age_group']).
        min_samples: Threshold for small cohort warnings.

    Returns:
        Dict containing fairness audit results per attribute and overall summary.
    """
    df_reconstructed = reconstruct_demographic_features(df)

    if demographic_cols is None:
        demographic_cols = ["gender", "age_group"]

    results = {}
    for col in demographic_cols:
        if col in df_reconstructed.columns:
            logger.info("Evaluating demographic fairness for attribute: %s", col)
            results[col] = evaluate_fairness_for_attribute(
                df_reconstructed, y_true, y_prob, y_pred, col, min_samples=min_samples,
            )
        else:
            logger.warning("Demographic attribute column '%s' not found in dataset.", col)

    return {
        "fairness_audit": results,
        "sample_size_threshold": min_samples,
    }
