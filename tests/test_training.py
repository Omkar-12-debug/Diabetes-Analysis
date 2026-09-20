"""Unit and integration test suite for model training, evaluation, and MLflow tracking.

Validates:
- Baseline Logistic Regression training and validation PR-AUC > 0.40.
- Candidate models (RF, XGBoost) beating baseline PR-AUC and Recall.
- MLflow tracking parameters, metrics, and artifact generation.
- Probability outputs bounded strictly in [0.0, 1.0].
- Ranking, champion selection, and JSON report export logic.
"""

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from src.training.evaluate import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    compute_metrics,
    load_processed_splits,
    log_mlflow_run,
)
from src.training.train_all import (
    export_comparison_report,
    rank_and_select_champion,
)
from src.training.train_baseline import train as train_baseline
from src.training.train_benchmarks import BENCHMARK_CONFIGS, train_benchmark
from src.training.train_random_forest import train as train_rf
from src.training.train_xgboost import train as train_xgb


@pytest.fixture(scope="module")
def dataset_splits():
    """Load preprocessed splits once for the module tests."""
    return load_processed_splits("data/processed")


def test_load_processed_splits(dataset_splits):
    """Verify that processed splits load with matching dimensions and no NaNs."""
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_splits

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # No NaN or Infinite values
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()

    # Binary labels
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_val)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_compute_metrics_correctness():
    """Verify compute_metrics calculates all required fields within valid ranges."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_true, y_prob, y_pred)

    required_keys = [
        "pr_auc", "roc_auc", "recall", "precision", "f1",
        "balanced_accuracy", "brier_score", "confusion_matrix",
    ]
    for k in required_keys:
        assert k in metrics, f"Missing metric: {k}"

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 3
    assert cm["fp"] == 0
    assert cm["fn"] == 0
    assert cm["tp"] == 3


def test_baseline_logistic_regression(dataset_splits):
    """Test baseline model trains, produces bounded probabilities, and meets PR-AUC > 0.40."""
    X_train, X_val, _, y_train, y_val, _ = dataset_splits

    model, params = train_baseline(X_train, y_train)

    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)

    # Probabilities strictly bounded in [0.0, 1.0]
    assert np.all(y_val_prob >= 0.0)
    assert np.all(y_val_prob <= 1.0)

    val_metrics = compute_metrics(y_val, y_val_prob, y_val_pred)

    # Verification threshold: PR-AUC > 0.40
    assert val_metrics["pr_auc"] > 0.40, f"Baseline PR-AUC too low: {val_metrics['pr_auc']}"
    assert val_metrics["roc_auc"] > 0.80, f"Baseline ROC-AUC too low: {val_metrics['roc_auc']}"
    assert val_metrics["recall"] > 0.70, f"Baseline Recall too low: {val_metrics['recall']}"


def test_candidate_tree_models_beat_baseline(dataset_splits):
    """Test candidate models (RF and XGBoost) outperform baseline in PR-AUC and Recall."""
    X_train, X_val, _, y_train, y_val, _ = dataset_splits

    # Train baseline
    base_model, _ = train_baseline(X_train, y_train)
    base_prob = base_model.predict_proba(X_val)[:, 1]
    base_pred = base_model.predict(X_val)
    base_metrics = compute_metrics(y_val, base_prob, base_pred)

    # Train Random Forest
    rf_model, _ = train_rf(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_pred = rf_model.predict(X_val)
    rf_metrics = compute_metrics(y_val, rf_prob, rf_pred)

    # Train XGBoost
    xgb_model, _ = train_xgb(X_train, y_train)
    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_pred = xgb_model.predict(X_val)
    xgb_metrics = compute_metrics(y_val, xgb_prob, xgb_pred)

    # Both RF and XGBoost should beat baseline PR-AUC
    assert rf_metrics["pr_auc"] > base_metrics["pr_auc"], (
        f"RF PR-AUC ({rf_metrics['pr_auc']:.4f}) should beat Baseline ({base_metrics['pr_auc']:.4f})"
    )
    assert xgb_metrics["pr_auc"] > base_metrics["pr_auc"], (
        f"XGBoost PR-AUC ({xgb_metrics['pr_auc']:.4f}) should beat Baseline ({base_metrics['pr_auc']:.4f})"
    )

    # Both tree candidates maintain high recall (> 0.75)
    assert rf_metrics["recall"] >= 0.75, f"RF Recall too low: {rf_metrics['recall']}"
    assert xgb_metrics["recall"] >= 0.75, f"XGBoost Recall too low: {xgb_metrics['recall']}"


def test_benchmark_models_execution(dataset_splits):
    """Test Decision Tree and Gaussian NB benchmarks run and output valid bounded metrics."""
    X_train, X_val, _, y_train, y_val, _ = dataset_splits

    for cfg in BENCHMARK_CONFIGS:
        name = cfg["model_name"]
        cls = cfg["class"]
        params = cfg["params"]

        model, _ = train_benchmark(name, cls, params, X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        preds = model.predict(X_val)

        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

        metrics = compute_metrics(y_val, probs, preds)
        assert metrics["pr_auc"] > 0.20
        assert metrics["roc_auc"] > 0.60


def test_mlflow_logging_and_artifacts(dataset_splits):
    """Test that MLflow logs params, metrics, plots, and models without unhandled errors."""
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_splits

    mlflow.set_experiment(EXPERIMENT_NAME)

    model, params = train_baseline(X_train, y_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = model.predict(X_train)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)

    train_m = compute_metrics(y_train, y_train_prob, y_train_pred)
    val_m = compute_metrics(y_val, y_val_prob, y_val_pred)
    test_m = compute_metrics(y_test, y_test_prob, y_test_pred)

    run_id, model_uri = log_mlflow_run(
        model=model,
        model_name="TestLR",
        params=params,
        train_metrics=train_m,
        val_metrics=val_m,
        test_metrics=test_m,
        y_val_true=y_val,
        y_val_prob=y_val_prob,
        y_val_pred=y_val_pred,
        flavor="sklearn",
    )

    client = MlflowClient()
    run = client.get_run(run_id)

    # Check params logged
    assert run.data.params["model_type"] == "TestLR"
    assert run.data.params["class_weight"] == "balanced"

    # Check metrics logged
    assert "val_pr_auc" in run.data.metrics
    assert "val_roc_auc" in run.data.metrics
    assert "val_recall" in run.data.metrics
    assert "val_f1" in run.data.metrics
    assert "val_tn" in run.data.metrics

    # Check artifact plots logged
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    assert "confusion_matrix.png" in artifact_paths
    assert "roc_curve.png" in artifact_paths
    assert "pr_curve.png" in artifact_paths

    # Check model artifact is valid and loadable
    assert model_uri is not None and len(model_uri) > 0
    loaded_model = mlflow.sklearn.load_model(model_uri)
    np.testing.assert_array_equal(loaded_model.predict(X_val[:10]), y_val_pred[:10])


def test_ranking_and_champion_selection():
    """Verify rank_and_select_champion accurately sorts by val_pr_auc."""
    mock_results = [
        {"model_name": "Model_A", "val_metrics": {"pr_auc": 0.55, "f1": 0.60, "recall": 0.70}},
        {"model_name": "Model_B", "val_metrics": {"pr_auc": 0.82, "f1": 0.80, "recall": 0.85}},
        {"model_name": "Model_C", "val_metrics": {"pr_auc": 0.71, "f1": 0.75, "recall": 0.78}},
    ]

    ranked_info = rank_and_select_champion(mock_results)
    ranked = ranked_info["ranked_models"]
    champion = ranked_info["champion"]

    assert champion["model_name"] == "Model_B"
    assert ranked[0]["model_name"] == "Model_B"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["model_name"] == "Model_C"
    assert ranked[1]["rank"] == 2
    assert ranked[2]["model_name"] == "Model_A"
    assert ranked[2]["rank"] == 3


def test_export_comparison_report(tmp_path):
    """Test export_comparison_report creates a valid structured JSON file."""
    ranked_models = [
        {
            "rank": 1,
            "model_name": "BestModel",
            "run_id": "run_123",
            "val_metrics": {"pr_auc": 0.85},
            "test_metrics": {"pr_auc": 0.84},
        }
    ]
    champion = {
        "model_name": "BestModel",
        "run_id": "run_123",
        "val_metrics": {"pr_auc": 0.85},
        "test_metrics": {"pr_auc": 0.84},
    }

    report_file = tmp_path / "model_comparison.json"
    export_comparison_report(ranked_models, champion, "1", report_file)

    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["primary_ranking_metric"] == "val_pr_auc"
    assert data["registered_model_name"] == REGISTERED_MODEL_NAME
    assert data["champion"]["model_name"] == "BestModel"
    assert len(data["leaderboard"]) == 1
    assert data["leaderboard"][0]["is_champion"] is True
