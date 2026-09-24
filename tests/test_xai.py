"""Unit and integration tests for Explainable AI (SHAP) and Counterfactual What-If simulation.

Tests:
- SHAP TreeExplainer initialization and background reference set sampling.
- Additivity property: sum of base_value and SHAP attributions matches predicted margin.
- Attribution dimension mapping matches the 27 transformed features.
- Counterfactual engine preserves non-modifiable features (age, gender, hypertension, heart_disease).
- Counterfactual engine drives risk reduction below threshold.
- Inclusion of mandatory non-causal medical disclaimer string in all outputs.
"""

from pathlib import Path
import numpy as np
import pytest

from src.features.counterfactual_dice import (
    MANDATORY_DISCLAIMER,
    NON_MODIFIABLE_FEATURES,
    CounterfactualExplainerEngine,
)
from src.features.xai_shap import ShapExplainerEngine


@pytest.fixture(scope="module")
def sample_patient():
    """Sample high-risk patient profile."""
    return {
        "gender": "Female",
        "age": 58.0,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "current",
        "bmi": 34.0,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 190.0,
    }


@pytest.fixture(scope="module")
def shap_engine():
    """Instantiate ShapExplainerEngine for tests."""
    return ShapExplainerEngine(n_background=50)


@pytest.fixture(scope="module")
def cf_engine(shap_engine):
    """Instantiate CounterfactualExplainerEngine using preloaded model and preprocessor."""
    return CounterfactualExplainerEngine(
        model=shap_engine.model,
        preprocessor=shap_engine.preprocessor,
    )


def test_shap_explainer_initialization(shap_engine):
    """Verify explainer initializes with valid base value and feature dimensionality."""
    assert shap_engine.explainer is not None
    assert isinstance(shap_engine.expected_value, float)
    assert len(shap_engine.feature_names) == 27


def test_shap_local_attribution_additivity(shap_engine, sample_patient):
    """Test SHAP additivity property: base_value + sum(shap_values) == predicted_margin."""
    result = shap_engine.explain_local(sample_patient)

    assert "base_value" in result
    assert "predicted_margin" in result
    assert "predicted_probability" in result
    assert "attributions" in result

    # Compute sum of attributions
    total_shap = sum(a["shap_value"] for a in result["attributions"])
    reconstructed_margin = result["base_value"] + total_shap

    # Check margin match within floating point precision
    assert result["predicted_margin"] == pytest.approx(reconstructed_margin, rel=1e-3, abs=1e-3)

    # Check predicted probability is valid in [0, 1]
    assert 0.0 <= result["predicted_probability"] <= 1.0


def test_shap_attributions_match_feature_dimensions(shap_engine, sample_patient):
    """Verify local attribution list matches all 27 transformed feature dimensions."""
    result = shap_engine.explain_local(sample_patient)
    attributions = result["attributions"]

    assert len(attributions) == 27

    attribution_feature_names = {a["feature"] for a in attributions}
    assert attribution_feature_names == set(shap_engine.feature_names)

    for attr in attributions:
        assert isinstance(attr["shap_value"], float)
        assert attr["effect"] in ["increases_risk", "decreases_risk"]
        assert attr["abs_importance"] >= 0.0


def test_counterfactual_preserves_non_modifiable_features(cf_engine, sample_patient):
    """Verify non-modifiable factors (age, gender, hypertension, heart_disease) remain identical."""
    result = cf_engine.generate_counterfactual(sample_patient, target_risk_threshold=0.50)

    assert result["non_modifiable_preserved"] is True

    original = result["original_patient"]
    counterfactual = result["counterfactual_patient"]

    for non_mod in NON_MODIFIABLE_FEATURES:
        assert original[non_mod] == counterfactual[non_mod], (
            f"Non-modifiable feature '{non_mod}' was changed from {original[non_mod]} to {counterfactual[non_mod]}"
        )


def test_counterfactual_risk_reduction_and_delta(cf_engine, sample_patient):
    """Verify counterfactual achieves risk reduction and correctly reports delta."""
    result = cf_engine.generate_counterfactual(sample_patient, target_risk_threshold=0.50)

    orig_p = result["original_probability"]
    cf_p = result["counterfactual_probability"]
    delta = result["probability_delta"]

    assert orig_p >= 0.50  # High risk patient
    assert cf_p < orig_p    # Counterfactual strictly lowers risk
    assert delta < 0.0      # Delta is negative (risk reduction)
    assert delta == pytest.approx(cf_p - orig_p, abs=1e-5)
    assert result["target_achieved"] is True


def test_counterfactual_mandatory_disclaimer(cf_engine, sample_patient):
    """Verify counterfactual response contains the mandatory responsible AI disclaimer."""
    result = cf_engine.generate_counterfactual(sample_patient)

    assert "disclaimer" in result
    assert result["disclaimer"] == MANDATORY_DISCLAIMER
    assert "causal medical advice" in result["disclaimer"].lower()
    assert "hypothetical model sensitivity" in result["disclaimer"].lower()


def test_global_shap_summary_generation(shap_engine, tmp_path):
    """Verify global SHAP summary plot is generated and saved cleanly."""
    save_file = tmp_path / "test_shap_summary.png"
    out_path = shap_engine.explain_global(save_path=save_file, max_samples=50)

    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # Non-empty image file
