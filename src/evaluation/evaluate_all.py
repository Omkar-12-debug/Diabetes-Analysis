"""Orchestrator for advanced model evaluation, calibration, and demographic fairness.

Dynamically loads the active champion model from the MLflow Model Registry via the
'@champion' alias, audits probability calibration against ground truth on the test set,
conducts demographic fairness analysis across gender and age slices, and exports
structured audit reports and diagnostic figures.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

from src.evaluation.calibration import evaluate_calibration, plot_calibration_curve
from src.evaluation.fairness import evaluate_fairness
from src.training.evaluate import EXPERIMENT_NAME, REGISTERED_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"
CHAMPION_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
EVAL_REPORT_PATH = Path("reports/evaluation_report.json")
CALIBRATION_PLOT_PATH = Path("reports/figures/calibration_curve.png")


def load_champion_model(model_uri: str = CHAMPION_URI) -> Any:
    """Dynamically load the active champion model from the MLflow Model Registry.

    Attempts flavor-specific loading first (e.g. XGBoost / Sklearn) for full
    predict_proba support, falling back to pyfunc if needed.

    Args:
        model_uri: MLflow model URI using registered model alias syntax.

    Returns:
        Loaded model object.
    """
    logger.info("Dynamically loading champion model from '%s'...", model_uri)

    try:
        import mlflow.xgboost
        model = mlflow.xgboost.load_model(model_uri)
        logger.info("Successfully loaded champion model via mlflow.xgboost.")
        return model
    except Exception as exc:
        logger.warning("Could not load via mlflow.xgboost (%s), falling back to mlflow.pyfunc.", exc)

    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Successfully loaded champion model via mlflow.sklearn.")
        return model
    except Exception as exc:
        logger.warning("Could not load via mlflow.sklearn (%s), falling back to mlflow.pyfunc.", exc)

    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Successfully loaded champion model via mlflow.pyfunc.")
    return model


def get_predictions(model: Any, X: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate probability estimates and binary predictions from model.

    Args:
        model: Loaded model object.
        X: Feature matrix or DataFrame.

    Returns:
        Tuple of (probabilities array [0..1], binary predictions array {0, 1}).
    """
    if hasattr(model, "predict_proba"):
        X_input = X.values if hasattr(X, "values") else X
        probs = model.predict_proba(X_input)[:, 1]
    elif hasattr(model, "predict"):
        raw_pred = model.predict(X)
        if hasattr(raw_pred, "ndim") and raw_pred.ndim == 2 and raw_pred.shape[1] > 1:
            probs = raw_pred[:, 1]
        elif np.all((raw_pred >= 0.0) & (raw_pred <= 1.0)) and not np.all(np.isin(raw_pred, [0, 1])):
            probs = raw_pred
        else:
            # Binary predictions only
            probs = raw_pred.astype(float)
    else:
        raise ValueError("Loaded model lacks predict or predict_proba methods.")

    preds = (probs >= 0.5).astype(int)
    return np.asarray(probs, dtype=float), np.asarray(preds, dtype=int)


def run_evaluation(
    processed_test_path: str | Path = "data/processed/test.parquet",
    model_uri: str = CHAMPION_URI,
    export_report: bool = True,
) -> Dict[str, Any]:
    """Run full calibration and fairness evaluation pipeline on test data.

    Args:
        processed_test_path: Path to preprocessed test split.
        model_uri: Model URI to load.
        export_report: Whether to save report and plots to disk.

    Returns:
        Comprehensive evaluation dictionary.
    """
    processed_test_path = Path(processed_test_path)
    if not processed_test_path.exists():
        raise FileNotFoundError(f"Processed test set not found at {processed_test_path}")

    logger.info("Loading test dataset from %s...", processed_test_path)
    df_test = pd.read_parquet(processed_test_path)

    if "diabetes" not in df_test.columns:
        raise ValueError("Test dataset missing target column 'diabetes'.")

    X_test = df_test.drop(columns=["diabetes"])
    y_test = df_test["diabetes"].values

    # Load champion
    model = load_champion_model(model_uri)

    # Predictions
    logger.info("Generating predictions on %d test samples...", len(y_test))
    y_prob, y_pred = get_predictions(model, X_test)

    # Base test metrics
    overall_metrics = {
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }
    logger.info(
        "Overall Test Metrics — PR-AUC: %.4f, ROC-AUC: %.4f, Recall: %.4f, F1: %.4f",
        overall_metrics["pr_auc"],
        overall_metrics["roc_auc"],
        overall_metrics["recall"],
        overall_metrics["f1"],
    )

    # Calibration Analysis
    logger.info("Running probability calibration audit...")
    calibration_results = evaluate_calibration(y_test, y_prob, n_bins=10)

    # Demographic Fairness Analysis
    logger.info("Running demographic fairness audit across gender and age groups...")
    fairness_results = evaluate_fairness(df_test, y_test, y_prob, y_pred)

    # Build report
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_uri": model_uri,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alias": CHAMPION_ALIAS,
        "test_dataset_size": len(y_test),
        "overall_performance": overall_metrics,
        "calibration": calibration_results,
        "fairness": fairness_results,
    }

    if export_report:
        # Generate and save calibration curve plot
        CALIBRATION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plot_calibration_curve(
            y_true=y_test,
            y_prob=y_prob,
            save_path=CALIBRATION_PLOT_PATH,
            model_name=f"{REGISTERED_MODEL_NAME} (@{CHAMPION_ALIAS})",
        )

        # Export JSON report
        EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved evaluation report to %s", EVAL_REPORT_PATH)

        # Log evaluation run to MLflow
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="Champion-Audit-Phase5"):
                mlflow.log_param("model_uri", model_uri)
                mlflow.log_param("test_samples", len(y_test))
                mlflow.log_metric("eval_test_pr_auc", overall_metrics["pr_auc"])
                mlflow.log_metric("eval_test_roc_auc", overall_metrics["roc_auc"])
                mlflow.log_metric("eval_test_recall", overall_metrics["recall"])
                mlflow.log_metric("eval_test_f1", overall_metrics["f1"])
                mlflow.log_metric("eval_brier_score", calibration_results["brier_score"])
                mlflow.log_metric("eval_ece", calibration_results["expected_calibration_error"])
                mlflow.log_metric("eval_mce", calibration_results["maximum_calibration_error"])

                mlflow.log_artifact(str(CALIBRATION_PLOT_PATH))
                mlflow.log_artifact(str(EVAL_REPORT_PATH))
                logger.info("Logged Phase 5 evaluation artifacts to MLflow experiment '%s'.", EXPERIMENT_NAME)
        except Exception as exc:
            logger.warning("Could not log evaluation run to MLflow: %s", exc)

    return report


def main() -> None:
    """Execute end-to-end evaluation."""
    logger.info("=== Starting Advanced Evaluation & Audit Pipeline ===")
    run_evaluation()
    logger.info("=== Advanced Evaluation & Audit Pipeline Complete ===")


if __name__ == "__main__":
    main()
