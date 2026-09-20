"""Probability calibration analysis and reliability diagram generation.

Computes Brier Score, Expected Calibration Error (ECE), Maximum Calibration
Error (MCE), and generates publication-grade calibration plots to evaluate
the trustworthiness of predicted probabilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score loss.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive-class probabilities [0.0, 1.0].

    Returns:
        float: Brier score loss (lower is better, 0.0 is perfect).
    """
    return float(brier_score_loss(y_true, y_prob))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Dict[str, Any]]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Divides predictions into equal-width bins, computing the weighted mean absolute
    difference between confidence (mean predicted prob) and accuracy (fraction of positives).

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive-class probabilities.
        n_bins: Number of probability bins.

    Returns:
        Tuple of (ece, mce, bin_details_dict).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_samples = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges[1:-1])  # 0 to n_bins - 1

    ece = 0.0
    mce = 0.0
    bin_details = []

    for b in range(n_bins):
        mask = bin_assignments == b
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            diff = abs(bin_acc - bin_conf)
            weight = bin_count / n_samples
            ece += weight * diff
            mce = max(mce, diff)

            bin_details.append({
                "bin_index": b,
                "bin_range": [float(bin_edges[b]), float(bin_edges[b + 1])],
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "absolute_error": float(diff),
            })

    return float(ece), float(mce), {"bins": bin_details, "n_bins": n_bins}


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Comprehensive probability calibration audit.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        Dict containing brier_score, ece, mce, and curve points.
    """
    brier = compute_brier_score(y_true, y_prob)
    ece, mce, bin_info = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    logger.info(
        "Calibration evaluated — Brier Score: %.4f, ECE: %.4f, MCE: %.4f",
        brier, ece, mce,
    )

    return {
        "brier_score": brier,
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "calibration_curve": {
            "prob_true": [float(x) for x in prob_true],
            "prob_pred": [float(x) for x in prob_pred],
        },
        "bin_details": bin_info["bins"],
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    n_bins: int = 10,
    model_name: str = "DiabetesRiskModel (@champion)",
) -> None:
    """Generate and export a two-panel calibration reliability diagram.

    Top panel: Reliability curve comparing predicted probability vs true fraction.
    Bottom panel: Histogram showing distribution and sharpness of predicted probabilities.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive-class probabilities.
        save_path: Path to save the plot image.
        n_bins: Number of bins.
        model_name: Model title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = compute_brier_score(y_true, y_prob)
    ece, _, _ = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Panel 1: Reliability Diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=1.5)
    ax1.plot(
        prob_pred,
        prob_true,
        "s-",
        color="#1f77b4",
        lw=2,
        label=f"{model_name}\n(Brier={brier:.4f}, ECE={ece:.4f})",
    )
    ax1.set_ylabel("Fraction of Positives (Empirical Risk)", fontsize=10)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_title(f"Calibration Reliability Diagram — {model_name}", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Predicted Probability Histogram (Sharpness)
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), color="#aec7e8", edgecolor="#1f77b4", density=True)
    ax2.set_xlabel("Mean Predicted Probability (Risk Score)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration diagram to %s", save_path)
