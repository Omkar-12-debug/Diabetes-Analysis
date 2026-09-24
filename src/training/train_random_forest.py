"""Random Forest candidate model with cost-sensitive trees.

Uses balanced_subsample class weighting to handle the ~10.34:1 class imbalance
by resampling class weights per bootstrap sample.
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.training.evaluate import (
    EXPERIMENT_NAME,
    compute_metrics,
    load_processed_splits,
    log_mlflow_run,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "RandomForest"
PARAMS = {
    "model_type": MODEL_NAME,
    "class_weight": "balanced_subsample",
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "random_state": 42,
}


def train(X_train: np.ndarray, y_train: np.ndarray):
    """Train a Random Forest model.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Tuple of (trained model, parameters dict).
    """
    model = RandomForestClassifier(
        class_weight=PARAMS["class_weight"],
        n_estimators=PARAMS["n_estimators"],
        max_depth=PARAMS["max_depth"],
        min_samples_leaf=PARAMS["min_samples_leaf"],
        random_state=PARAMS["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("%s training complete.", MODEL_NAME)
    return model, PARAMS.copy()


def train_and_evaluate():
    """Train, evaluate, and log the Random Forest model to MLflow.

    Returns:
        Dict with model_name, run_id, val_metrics, test_metrics.
    """
    import mlflow

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_splits()
    model, params = train(X_train, y_train)

    # Predictions
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = model.predict(X_train)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)

    # Metrics
    train_metrics = compute_metrics(y_train, y_train_prob, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_prob, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_prob, y_test_pred)

    # Log to MLflow
    run_id, model_uri = log_mlflow_run(
        model=model,
        model_name=MODEL_NAME,
        params=params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        y_val_true=y_val,
        y_val_prob=y_val_prob,
        y_val_pred=y_val_pred,
        flavor="sklearn",
    )

    logger.info(
        "%s — Val PR-AUC: %.4f | Val Recall: %.4f | Val F1: %.4f",
        MODEL_NAME, val_metrics["pr_auc"], val_metrics["recall"], val_metrics["f1"],
    )

    return {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "model_uri": model_uri,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    train_and_evaluate()
