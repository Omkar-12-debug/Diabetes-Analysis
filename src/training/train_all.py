"""Orchestrator script for training all baseline, candidate, and benchmark models.

Ranks models by validation PR-AUC, selects the champion, registers it in the MLflow
Model Registry under 'DiabetesRiskModel', assigns the '@champion' alias, and exports
a comprehensive summary to reports/model_comparison.json.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import mlflow
from mlflow.tracking import MlflowClient

from src.training.evaluate import EXPERIMENT_NAME, REGISTERED_MODEL_NAME
from src.training.train_baseline import train_and_evaluate as train_baseline
from src.training.train_benchmarks import train_and_evaluate_all_benchmarks
from src.training.train_random_forest import train_and_evaluate as train_random_forest
from src.training.train_xgboost import train_and_evaluate as train_xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COMPARISON_REPORT_PATH = Path("reports/model_comparison.json")


def run_all_training() -> List[Dict[str, Any]]:
    """Execute training and evaluation across all defined models.

    Returns:
        List of model result dictionaries with metrics and run IDs.
    """
    results: List[Dict[str, Any]] = []

    logger.info("=== [1/4] Training Baseline (Logistic Regression) ===")
    results.append(train_baseline())

    logger.info("=== [2/4] Training Candidate 1 (Random Forest) ===")
    results.append(train_random_forest())

    logger.info("=== [3/4] Training Candidate 2 (XGBoost) ===")
    results.append(train_xgboost())

    logger.info("=== [4/4] Training Benchmarks (Decision Tree & Gaussian NB) ===")
    results.extend(train_and_evaluate_all_benchmarks())

    return results


def rank_and_select_champion(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank models by validation PR-AUC in descending order and select champion.

    Args:
        results: List of model evaluation results.

    Returns:
        Dict with ranked_models list and champion dict.
    """
    # Sort descending by validation PR-AUC
    ranked = sorted(
        results,
        key=lambda r: r["val_metrics"]["pr_auc"],
        reverse=True,
    )

    for rank_idx, res in enumerate(ranked, start=1):
        res["rank"] = rank_idx

    champion = ranked[0]
    logger.info(
        "Champion Selected: %s (Val PR-AUC: %.4f, Val F1: %.4f, Val Recall: %.4f)",
        champion["model_name"],
        champion["val_metrics"]["pr_auc"],
        champion["val_metrics"]["f1"],
        champion["val_metrics"]["recall"],
    )
    return {"ranked_models": ranked, "champion": champion}


def register_champion_model(champion: Dict[str, Any]) -> str:
    """Register the champion model in MLflow Model Registry and set @champion alias.

    Args:
        champion: Dictionary containing champion model details.

    Returns:
        str: Registered model version number.
    """
    run_id = champion["run_id"]
    model_name = champion["model_name"]
    model_uri = champion.get("model_uri") or f"runs:/{run_id}/model"

    logger.info(
        "Registering champion %s (run_id=%s) to registry '%s'...",
        model_name, run_id, REGISTERED_MODEL_NAME,
    )

    # Register model
    registered_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()
    version_num = str(registered_version.version)

    # Assign @champion alias
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=version_num,
    )
    logger.info(
        "Assigned '@champion' alias to version %s of '%s'.",
        version_num, REGISTERED_MODEL_NAME,
    )

    # Add description / tags
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version_num,
        description=(
            f"Champion model selected via validation PR-AUC ({champion['val_metrics']['pr_auc']:.4f}). "
            f"Model type: {model_name}, run_id: {run_id}."
        ),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version_num,
        key="model_type",
        value=model_name,
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version_num,
        key="val_pr_auc",
        value=f"{champion['val_metrics']['pr_auc']:.4f}",
    )

    return version_num


def export_comparison_report(
    ranked_models: List[Dict[str, Any]],
    champion: Dict[str, Any],
    champion_version: str,
    output_path: Path = COMPARISON_REPORT_PATH,
) -> None:
    """Save model comparison details and ranking summary to JSON.

    Args:
        ranked_models: Ranked list of model evaluation results.
        champion: Champion model dict.
        champion_version: Registered version string.
        output_path: Target path for the report file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "primary_ranking_metric": "val_pr_auc",
        "registered_model_name": REGISTERED_MODEL_NAME,
        "champion": {
            "model_name": champion["model_name"],
            "run_id": champion["run_id"],
            "registered_version": champion_version,
            "alias": "champion",
            "val_metrics": champion["val_metrics"],
            "test_metrics": champion["test_metrics"],
        },
        "leaderboard": [
            {
                "rank": m["rank"],
                "model_name": m["model_name"],
                "run_id": m["run_id"],
                "is_champion": m["model_name"] == champion["model_name"],
                "val_metrics": m["val_metrics"],
                "test_metrics": m["test_metrics"],
            }
            for m in ranked_models
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.info("Model comparison report saved to %s.", output_path)


def main() -> None:
    """Run the complete training pipeline, model ranking, registration, and reporting."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = run_all_training()
    ranked_info = rank_and_select_champion(results)
    ranked_models = ranked_info["ranked_models"]
    champion = ranked_info["champion"]

    champion_version = register_champion_model(champion)
    export_comparison_report(ranked_models, champion, champion_version)

    logger.info("=== Training and Registration Pipeline Complete ===")


if __name__ == "__main__":
    main()
