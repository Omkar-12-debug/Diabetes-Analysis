"""Shared evaluation utilities, metrics computation, plotting, and MLflow logging.

Provides a consistent interface for computing all required metrics, generating
diagnostic plots, and instrumenting MLflow experiment runs across all model
training modules.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "diabetes"
EXPERIMENT_NAME = "diabetes-risk-assessment"
REGISTERED_MODEL_NAME = "DiabetesRiskModel"


def load_processed_splits(
    processed_dir: str | Path = "data/processed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed parquet splits and separate features from target.

    Args:
        processed_dir: Directory containing train.parquet, val.parquet, test.parquet.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    processed_dir = Path(processed_dir)

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    X_train = train_df.drop(columns=[TARGET_COL]).values
    y_train = train_df[TARGET_COL].values
    X_val = val_df.drop(columns=[TARGET_COL]).values
    y_val = val_df[TARGET_COL].values
    X_test = test_df.drop(columns=[TARGET_COL]).values
    y_test = test_df[TARGET_COL].values

    logger.info(
        "Loaded splits: train=%d, val=%d, test=%d (%d features).",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute all required evaluation metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        y_pred: Binary predictions.

    Returns:
        Dict with keys: pr_auc, roc_auc, recall, precision, f1, balanced_accuracy,
        brier_score, confusion_matrix (dict with tn, fp, fn, tp).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: File path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Non-Diabetic", "Diabetic"],
        cmap="Blues", ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """Generate and save an ROC curve with AUC annotation.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        save_path: File path to save the plot.
        title: Plot title.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """Generate and save a Precision-Recall curve with AP annotation.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        save_path: File path to save the plot.
        title: Plot title.
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ap_val = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color="blue", lw=2, label=f"PR (AP = {ap_val:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def log_mlflow_run(
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    y_val_pred: np.ndarray,
    flavor: str = "sklearn",
) -> str:
    """Log a complete model training run to MLflow.

    Args:
        model: Trained model object.
        model_name: Human-readable model name (e.g., "LogisticRegression").
        params: Hyperparameters dict.
        train_metrics: Metrics dict for training set.
        val_metrics: Metrics dict for validation set.
        test_metrics: Metrics dict for test set.
        y_val_true: Validation ground truth (for artifact plots).
        y_val_prob: Validation predicted probabilities.
        y_val_pred: Validation binary predictions.
        flavor: MLflow model flavor ("sklearn" or "xgboost").

    Returns:
        Tuple[str, str]: The MLflow run ID and the model URI.
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_param("model_type", model_name)
        if "random_state" not in params:
            mlflow.log_param("random_state", 42)
        for key, value in params.items():
            if key == "model_type":
                continue
            mlflow.log_param(key, value)

        # Log metrics with split prefixes
        for prefix, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ]:
            for metric_name, metric_value in metrics.items():
                if metric_name == "confusion_matrix":
                    # Log confusion matrix components individually
                    for cm_key, cm_val in metric_value.items():
                        mlflow.log_metric(f"{prefix}_{cm_key}", cm_val)
                else:
                    mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)

        # Generate and log artifact plots
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            cm_path = tmpdir / "confusion_matrix.png"
            plot_confusion_matrix(y_val_true, y_val_pred, cm_path, f"{model_name} - Confusion Matrix (Val)")
            mlflow.log_artifact(str(cm_path))

            roc_path = tmpdir / "roc_curve.png"
            plot_roc_curve(y_val_true, y_val_prob, roc_path, f"{model_name} - ROC Curve (Val)")
            mlflow.log_artifact(str(roc_path))

            pr_path = tmpdir / "pr_curve.png"
            plot_pr_curve(y_val_true, y_val_prob, pr_path, f"{model_name} - PR Curve (Val)")
            mlflow.log_artifact(str(pr_path))

        # Log model
        if flavor == "xgboost":
            model_info = mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

        run_id = run.info.run_id
        model_uri = model_info.model_uri
        logger.info(
            "MLflow run logged: %s (run_id=%s, model_uri=%s, val_pr_auc=%.4f).",
            model_name, run_id, model_uri, val_metrics["pr_auc"],
        )
        return run_id, model_uri
