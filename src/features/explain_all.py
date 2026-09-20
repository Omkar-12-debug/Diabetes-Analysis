"""CLI orchestrator for Explainable AI (SHAP) and Counterfactual What-If simulation.

Executes:
1. Global SHAP summary generation and export to reports/figures/shap_summary.png.
2. Logging of explainability artifacts to MLflow.
3. Single-instance local SHAP attribution demonstration on a high-risk patient profile.
4. Counterfactual 'What-If' sensitivity exploration with risk reduction delta computation.
"""

import json
import logging
from pathlib import Path

import mlflow

from src.features.counterfactual_dice import CounterfactualExplainerEngine
from src.features.xai_shap import DEFAULT_SHAP_SUMMARY_PATH, ShapExplainerEngine
from src.training.evaluate import EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_HIGH_RISK_PATIENT = {
    "gender": "Female",
    "age": 55.0,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "former",
    "bmi": 32.5,
    "HbA1c_level": 7.2,
    "blood_glucose_level": 180.0,
}


def run_explainability_pipeline() -> None:
    """Run full XAI pipeline: global SHAP, MLflow logging, local attribution, and counterfactuals."""
    logger.info("=== [1/3] Initializing SHAP Explainer & Generating Global Summary ===")
    shap_engine = ShapExplainerEngine()
    summary_plot_path = shap_engine.explain_global(save_path=DEFAULT_SHAP_SUMMARY_PATH, max_samples=400)
    logger.info("Global summary generated at: %s", summary_plot_path)

    # Log figure to MLflow
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="Champion-Explainability-SHAP"):
            mlflow.log_artifact(str(summary_plot_path))
            logger.info("Logged %s to MLflow experiment '%s'.", summary_plot_path, EXPERIMENT_NAME)
    except Exception as exc:
        logger.warning("Could not log SHAP summary to MLflow: %s", exc)

    logger.info("=== [2/3] Demonstrating Local SHAP Risk Attribution ===")
    logger.info("Patient Profile: %s", json.dumps(SAMPLE_HIGH_RISK_PATIENT, indent=2))

    local_explanation = shap_engine.explain_local(SAMPLE_HIGH_RISK_PATIENT)
    logger.info("Patient Predicted Risk Probability: %.4f (%s)",
                local_explanation["predicted_probability"], local_explanation["risk_classification"])
    logger.info("Top Risk-Increasing Drivers:")
    for factor in local_explanation["top_risk_increasing_factors"][:3]:
        logger.info("  [+] %s: SHAP attribution = +%.4f", factor["feature"], factor["shap_value"])

    logger.info("Top Protective / Risk-Decreasing Drivers:")
    for factor in local_explanation["top_risk_decreasing_factors"][:3]:
        logger.info("  [-] %s: SHAP attribution = %.4f", factor["feature"], factor["shap_value"])

    logger.info("=== [3/3] Generating Counterfactual 'What-If' Simulation ===")
    cf_engine = CounterfactualExplainerEngine(model=shap_engine.model, preprocessor=shap_engine.preprocessor)
    cf_result = cf_engine.generate_counterfactual(SAMPLE_HIGH_RISK_PATIENT, target_risk_threshold=0.50)

    logger.info("Original Risk: %.4f -> Counterfactual Risk: %.4f (Delta: %.4f)",
                cf_result["original_probability"],
                cf_result["counterfactual_probability"],
                cf_result["probability_delta"])
    logger.info("Required Lifestyle / Clinical Modifications:")
    for feat, change in cf_result["feature_changes"].items():
        logger.info("  * %s: %s -> %s (delta: %s)",
                    feat, change["original"], change["counterfactual"], change.get("delta", "N/A"))

    logger.info("Non-Modifiable Demographics Preserved (age, gender, history): %s",
                cf_result["non_modifiable_preserved"])
    logger.info("Mandatory Clinical Disclaimer:\n  \"%s\"", cf_result["disclaimer"])

    logger.info("=== Explainability & Counterfactual Pipeline Complete ===")


if __name__ == "__main__":
    run_explainability_pipeline()
