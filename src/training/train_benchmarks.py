"""Benchmark models: Decision Tree and Gaussian Naive Bayes.

Provides historical continuity and baseline comparisons against simpler classical
methods evaluated in earlier phases or baseline experiments.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.training.evaluate import (
    EXPERIMENT_NAME,
    compute_metrics,
    load_processed_splits,
    log_mlflow_run,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_CONFIGS = [
    {
        "model_name": "DecisionTree",
        "class": DecisionTreeClassifier,
        "params": {
            "model_type": "DecisionTree",
            "class_weight": "balanced",
            "max_depth": 8,
            "min_samples_leaf": 10,
            "random_state": 42,
        },
    },
    {
        "model_name": "GaussianNB",
        "class": GaussianNB,
        "params": {
            "model_type": "GaussianNB",
            "var_smoothing": 1e-9,
        },
    },
]


def train_benchmark(
    name: str,
    cls: Any,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a single benchmark model.

    Args:
        name: Name of the benchmark model.
        cls: Model class.
        params: Hyperparameter dictionary.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Tuple of (trained model, params copy).
    """
    init_params = {k: v for k, v in params.items() if k != "model_type"}
    model = cls(**init_params)
    model.fit(X_train, y_train)
    logger.info("%s benchmark training complete.", name)
    return model, params.copy()


def train_and_evaluate_all_benchmarks() -> List[Dict[str, Any]]:
    """Train, evaluate, and log all benchmark models to MLflow.

    Returns:
        List of dicts with model_name, run_id, val_metrics, test_metrics.
    """
    import mlflow

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_splits()
    results = []

    for cfg in BENCHMARK_CONFIGS:
        name = cfg["model_name"]
        cls = cfg["class"]
        params = cfg["params"]

        model, logged_params = train_benchmark(name, cls, params, X_train, y_train)

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
            model_name=name,
            params=logged_params,
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
            name, val_metrics["pr_auc"], val_metrics["recall"], val_metrics["f1"],
        )

        results.append({
            "model_name": name,
            "run_id": run_id,
            "model_uri": model_uri,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        })

    return results


if __name__ == "__main__":
    train_and_evaluate_all_benchmarks()
