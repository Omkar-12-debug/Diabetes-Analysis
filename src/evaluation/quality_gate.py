"""Autonomous Quality Gate & Model Promotion Engine.

Evaluates candidate models against strict clinical sensitivity, discriminative ranking,
probability calibration, demographic fairness, and artifact integrity thresholds before
promoting them to '@champion' in the MLflow Model Registry.
"""

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.evaluation.calibration import compute_brier_score
from src.evaluation.evaluate_all import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    get_predictions,
    load_champion_model,
)
from src.evaluation.fairness import evaluate_fairness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Gate Threshold Specifications
MIN_TEST_PR_AUC: float = 0.85
MIN_TEST_RECALL: float = 0.88
MAX_BRIER_SCORE: float = 0.10
MAX_GENDER_RECALL_DISPARITY: float = 0.08

DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
DEFAULT_TEST_DATA_PATH = Path("data/processed/test.parquet")
DEFAULT_REPORT_PATH = Path("reports/quality_gate_report.json")
DEFAULT_MODEL_COMPARISON_PATH = Path("reports/model_comparison.json")


def get_baseline_test_pr_auc(comparison_path: Union[str, Path] = DEFAULT_MODEL_COMPARISON_PATH) -> float:
    """Retrieve test PR-AUC of the Logistic Regression baseline model.

    Args:
        comparison_path: Path to reports/model_comparison.json.

    Returns:
        float: Baseline test PR-AUC (fallback to 0.8124 if report unavailable).
    """
    comparison_path = Path(comparison_path)
    if comparison_path.exists():
        try:
            with open(comparison_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for model_item in data.get("leaderboard", []):
                if model_item.get("model_name") == "LogisticRegression":
                    return float(model_item["test_metrics"]["pr_auc"])
        except Exception as exc:
            logger.warning("Could not parse baseline PR-AUC from %s: %s", comparison_path, exc)

    return 0.8124  # Baseline Logistic Regression test PR-AUC benchmark


def verify_artifacts(
    model: Any,
    preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
) -> Dict[str, Any]:
    """Verify that the preprocessor and model artifacts are intact, loadable, and executable.

    Args:
        model: Loaded candidate model instance.
        preprocessor_path: Path to preprocessor joblib file.

    Returns:
        Dict detailing verification status of all required artifacts.
    """
    preprocessor_path = Path(preprocessor_path)
    prep_exists = preprocessor_path.exists()
    prep_loadable = False
    prep_obj = None

    if prep_exists:
        try:
            prep_obj = joblib.load(preprocessor_path)
            prep_loadable = hasattr(prep_obj, "transform")
        except Exception as exc:
            logger.error("Failed to load preprocessor artifact: %s", exc)

    model_valid = model is not None and (hasattr(model, "predict") or hasattr(model, "predict_proba"))

    all_valid = prep_exists and prep_loadable and model_valid

    return {
        "valid": all_valid,
        "preprocessor_exists": prep_exists,
        "preprocessor_loadable": prep_loadable,
        "model_executable": model_valid,
        "preprocessor_instance": prep_obj,
    }


def evaluate_gates(
    candidate_metrics: Dict[str, Any],
    baseline_pr_auc: float,
    fairness_audit: Dict[str, Any],
    artifact_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit all 5 quality gate criteria against explicit clinical and performance thresholds.

    Args:
        candidate_metrics: Dict containing 'pr_auc', 'recall', 'brier_score', etc.
        baseline_pr_auc: Benchmark PR-AUC from the baseline model.
        fairness_audit: Demographic fairness audit results dictionary.
        artifact_status: Artifact verification dictionary.

    Returns:
        Dict with evaluation outcomes for each gate and overall PASS/FAIL decision.
    """
    gates: Dict[str, Dict[str, Any]] = {}
    failure_reasons: List[str] = []

    # 1. Primary Metric Gate: PR-AUC >= 0.85 and >= baseline
    cand_pr_auc = float(candidate_metrics.get("pr_auc", 0.0))
    pr_auc_pass = (cand_pr_auc >= MIN_TEST_PR_AUC) and (cand_pr_auc >= baseline_pr_auc)
    gates["primary_pr_auc"] = {
        "metric": "PR-AUC",
        "value": cand_pr_auc,
        "threshold": f">= {MIN_TEST_PR_AUC} and >= baseline ({baseline_pr_auc:.4f})",
        "status": "PASSED" if pr_auc_pass else "FAILED",
        "details": (
            f"Candidate PR-AUC ({cand_pr_auc:.4f}) meets minimum standard ({MIN_TEST_PR_AUC}) "
            f"and matches/exceeds baseline ({baseline_pr_auc:.4f})."
            if pr_auc_pass
            else f"Candidate PR-AUC ({cand_pr_auc:.4f}) degraded below required threshold or baseline."
        ),
    }
    if not pr_auc_pass:
        failure_reasons.append(f"PR-AUC gate failed: {cand_pr_auc:.4f} < max({MIN_TEST_PR_AUC}, {baseline_pr_auc:.4f})")

    # 2. Clinical Sensitivity Gate: Recall >= 0.88
    cand_recall = float(candidate_metrics.get("recall", 0.0))
    recall_pass = cand_recall >= MIN_TEST_RECALL
    gates["clinical_recall"] = {
        "metric": "Recall (Sensitivity)",
        "value": cand_recall,
        "threshold": f">= {MIN_TEST_RECALL}",
        "status": "PASSED" if recall_pass else "FAILED",
        "details": (
            f"Candidate Recall ({cand_recall:.4f}) satisfies clinical sensitivity standard ({MIN_TEST_RECALL})."
            if recall_pass
            else f"Candidate Recall ({cand_recall:.4f}) failed clinical sensitivity requirement ({MIN_TEST_RECALL})."
        ),
    }
    if not recall_pass:
        failure_reasons.append(f"Clinical sensitivity gate failed: Recall {cand_recall:.4f} < {MIN_TEST_RECALL}")

    # 3. Probability Calibration Gate: Brier Score <= 0.10
    cand_brier = float(candidate_metrics.get("brier_score", 1.0))
    brier_pass = cand_brier <= MAX_BRIER_SCORE
    gates["calibration_brier"] = {
        "metric": "Brier Score",
        "value": cand_brier,
        "threshold": f"<= {MAX_BRIER_SCORE}",
        "status": "PASSED" if brier_pass else "FAILED",
        "details": (
            f"Candidate Brier score ({cand_brier:.4f}) satisfies probability calibration upper bound ({MAX_BRIER_SCORE})."
            if brier_pass
            else f"Candidate Brier score ({cand_brier:.4f}) indicates uncalibrated risk probabilities (> {MAX_BRIER_SCORE})."
        ),
    }
    if not brier_pass:
        failure_reasons.append(f"Calibration gate failed: Brier score {cand_brier:.4f} > {MAX_BRIER_SCORE}")

    # 4. Demographic Fairness Gate: Gender Recall Disparity <= 0.08
    gender_slices = fairness_audit.get("gender", {}).get("slices", {})
    male_recall = gender_slices.get("Male", {}).get("recall")
    female_recall = gender_slices.get("Female", {}).get("recall")

    if male_recall is not None and female_recall is not None:
        recall_disparity = float(abs(male_recall - female_recall))
        fairness_pass = recall_disparity <= MAX_GENDER_RECALL_DISPARITY
        fairness_details = (
            f"Gender recall disparity |{male_recall:.4f} - {female_recall:.4f}| = {recall_disparity:.4f} "
            f"is within tolerance ({MAX_GENDER_RECALL_DISPARITY})."
            if fairness_pass
            else f"Gender recall disparity ({recall_disparity:.4f}) exceeds tolerance ({MAX_GENDER_RECALL_DISPARITY})."
        )
    else:
        recall_disparity = None
        fairness_pass = False
        fairness_details = "Could not compute gender recall disparity: male or female subgroup slice missing."

    gates["fairness_disparity"] = {
        "metric": "Gender Recall Disparity (|Recall_Male - Recall_Female|)",
        "value": recall_disparity,
        "threshold": f"<= {MAX_GENDER_RECALL_DISPARITY}",
        "status": "PASSED" if fairness_pass else "FAILED",
        "details": fairness_details,
    }
    if not fairness_pass:
        failure_reasons.append(f"Fairness gate failed: disparity {recall_disparity} > {MAX_GENDER_RECALL_DISPARITY}")

    # 5. Artifact Integrity Gate
    art_pass = bool(artifact_status.get("valid", False))
    gates["artifact_integrity"] = {
        "metric": "Artifact Integrity",
        "value": "Intact" if art_pass else "Corrupted/Missing",
        "threshold": "All artifacts present and loadable",
        "status": "PASSED" if art_pass else "FAILED",
        "details": (
            "Preprocessor (models/preprocessor.joblib) and model artifact are intact and executable."
            if art_pass
            else "Artifact verification failed: preprocessor or model artifact is missing/corrupted."
        ),
    }
    if not art_pass:
        failure_reasons.append("Artifact integrity gate failed.")

    overall_decision = "PASSED" if all(g["status"] == "PASSED" for g in gates.values()) else "FAILED"

    return {
        "overall_decision": overall_decision,
        "gates": gates,
        "failure_reasons": failure_reasons,
    }


class QualityGateController:
    """Controller evaluating candidate models and orchestrating MLflow model promotion."""

    def __init__(
        self,
        registered_model_name: str = REGISTERED_MODEL_NAME,
        champion_alias: str = CHAMPION_ALIAS,
    ) -> None:
        """Initialize controller.

        Args:
            registered_model_name: Name of registered model in MLflow.
            champion_alias: Logical alias for production champion.
        """
        self.registered_model_name = registered_model_name
        self.champion_alias = champion_alias
        self.client = MlflowClient()

    def promote_candidate(self, candidate_version: str) -> None:
        """Assign the @champion alias to the candidate version in MLflow Model Registry.

        Args:
            candidate_version: Model version number to promote.
        """
        logger.info(
            "Promoting version %s to '@%s' in registry '%s'...",
            candidate_version, self.champion_alias, self.registered_model_name,
        )
        self.client.set_registered_model_alias(
            name=self.registered_model_name,
            alias=self.champion_alias,
            version=str(candidate_version),
        )
        self.client.set_model_version_tag(
            name=self.registered_model_name,
            version=str(candidate_version),
            key="promotion_status",
            value="promoted_by_quality_gate",
        )
        self.client.set_model_version_tag(
            name=self.registered_model_name,
            version=str(candidate_version),
            key="promoted_at",
            value=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        logger.info("Promotion successful: version %s is now '@%s'.", candidate_version, self.champion_alias)

    def run_gate(
        self,
        candidate_version: Optional[str] = None,
        model_uri: Optional[str] = None,
        test_path: Union[str, Path] = DEFAULT_TEST_DATA_PATH,
        preprocessor_path: Union[str, Path] = DEFAULT_PREPROCESSOR_PATH,
        auto_promote: bool = True,
        export_report: bool = True,
        report_path: Union[str, Path] = DEFAULT_REPORT_PATH,
    ) -> Dict[str, Any]:
        """Execute autonomous quality gate audit and conditional promotion.

        Args:
            candidate_version: Model version to evaluate. If None, resolves from active champion.
            model_uri: Model URI. If None, resolved from version or champion alias.
            test_path: Path to preprocessed test split.
            preprocessor_path: Path to preprocessor artifact.
            auto_promote: Whether to assign @champion alias upon passing.
            export_report: Whether to write quality_gate_report.json.
            report_path: File path for gate report.

        Returns:
            Dict containing full quality gate audit results.
        """
        test_path = Path(test_path)
        report_path = Path(report_path)

        # 1. Resolve candidate version and model URI
        if candidate_version is None:
            try:
                alias_model = self.client.get_model_version_by_alias(self.registered_model_name, self.champion_alias)
                candidate_version = str(alias_model.version)
                logger.info("Target candidate version resolved from active champion: version %s", candidate_version)
            except Exception:
                candidate_version = "latest"
                logger.info("Champion alias not found; using version 'latest'.")

        if model_uri is None:
            if candidate_version != "latest":
                model_uri = f"models:/{self.registered_model_name}/{candidate_version}"
            else:
                model_uri = f"models:/{self.registered_model_name}@{self.champion_alias}"

        # 2. Load model and verify artifacts
        logger.info("Loading candidate model from %s...", model_uri)
        try:
            model = load_champion_model(model_uri)
        except Exception as exc:
            logger.error("Could not load candidate model: %s", exc)
            model = None

        artifact_status = verify_artifacts(model, preprocessor_path)

        # 3. Generate predictions and evaluation metrics on test dataset
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {test_path}")

        df_test = pd.read_parquet(test_path)
        X_test = df_test.drop(columns=["diabetes"], errors="ignore")
        y_test = df_test["diabetes"].values

        if model is not None and artifact_status["valid"]:
            from sklearn.metrics import average_precision_score, recall_score
            y_prob, y_pred = get_predictions(model, X_test)
            candidate_metrics = {
                "pr_auc": float(average_precision_score(y_test, y_prob)),
                "recall": float(recall_score(y_test, y_pred)),
                "brier_score": float(compute_brier_score(y_test, y_prob)),
            }
            fairness_result = evaluate_fairness(df_test, y_test, y_prob, y_pred)
            fairness_audit = fairness_result.get("fairness_audit", {})
        else:
            candidate_metrics = {"pr_auc": 0.0, "recall": 0.0, "brier_score": 1.0}
            fairness_audit = {}

        # 4. Retrieve baseline PR-AUC benchmark
        baseline_pr_auc = get_baseline_test_pr_auc()

        # 5. Evaluate all gate criteria
        logger.info("Auditing quality gate criteria...")
        gate_evaluation = evaluate_gates(
            candidate_metrics=candidate_metrics,
            baseline_pr_auc=baseline_pr_auc,
            fairness_audit=fairness_audit,
            artifact_status=artifact_status,
        )

        decision = gate_evaluation["overall_decision"]
        logger.info("Quality Gate Evaluation Result: %s", decision)

        # 6. Execute conditional promotion or rollback
        promoted = False
        if decision == "PASSED":
            logger.info("✅ ALL QUALITY GATES PASSED.")
            if auto_promote and candidate_version != "latest":
                self.promote_candidate(candidate_version)
                promoted = True
        else:
            logger.warning("❌ QUALITY GATE FAILED. Reasons: %s", gate_evaluation["failure_reasons"])
            logger.warning("Refusing promotion. Existing registry status remains untouched.")

        # 7. Compile and export report
        report_payload = {
            "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "candidate": {
                "registered_model_name": self.registered_model_name,
                "version": candidate_version,
                "model_uri": model_uri,
            },
            "overall_decision": decision,
            "promoted_to_champion": promoted,
            "failure_reasons": gate_evaluation["failure_reasons"],
            "thresholds": {
                "min_test_pr_auc": MIN_TEST_PR_AUC,
                "min_test_recall": MIN_TEST_RECALL,
                "max_brier_score": MAX_BRIER_SCORE,
                "max_gender_recall_disparity": MAX_GENDER_RECALL_DISPARITY,
                "baseline_pr_auc": baseline_pr_auc,
            },
            "candidate_metrics": candidate_metrics,
            "gate_details": gate_evaluation["gates"],
        }

        if export_report:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=2)
            logger.info("Saved quality gate report to %s", report_path)

        return report_payload


def main() -> None:
    """CLI runner for autonomous quality gate. Exits with code 0 on PASS, 1 on FAIL."""
    logger.info("=== Starting Autonomous Model Quality Gate ===")
    controller = QualityGateController()
    try:
        result = controller.run_gate()
    except Exception as exc:
        logger.exception("Unexpected error executing quality gate: %s", exc)
        sys.exit(1)

    if result["overall_decision"] == "PASSED":
        logger.info("=== Quality Gate PASSED: Model is Approved for Champion Deployment ===")
        sys.exit(0)
    else:
        logger.error("=== Quality Gate FAILED: Promotion Aborted ===")
        for reason in result["failure_reasons"]:
            logger.error("  - %s", reason)
        sys.exit(1)


if __name__ == "__main__":
    main()
