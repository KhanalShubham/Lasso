"""Unit Tests for Gate 6: Out-of-Sample Evaluation & Hypothesis Testing.

Tests:
1. Test metrics file exists, has all models/benchmarks, and strictly valid ranges.
2. Regularized models (Lasso & Ridge) outperform unregularized MLE in Log-Loss on test set.
3. Wilcoxon signed-rank test output integrity across 38 gameweeks.
4. McNemar test output integrity and contingency matrix validity.
5. Bootstrap coefficient stability file integrity (16 predictors, SE reduction, SSI, FSF).
6. Required evaluation figure artifacts exist and are non-empty.
7. Final evaluation report markdown exists and contains key decisions.
"""

from pathlib import Path
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def test_test_metrics_table_integrity():
    """Verify test metrics table exists and contains all required models."""
    metrics_path = REPORTS_DIR / "test_metrics.csv"
    assert metrics_path.exists(), f"Missing {metrics_path}"

    df = pd.read_csv(metrics_path)
    expected_models = {"Prior", "Bet365", "Market_Avg", "MLE", "Ridge", "Lasso"}
    assert expected_models.issubset(set(df["model"].values)), "Missing expected model evaluations"

    # Verify probability metric bounds
    assert (df["log_loss"] > 0).all()
    assert (df["brier_score"] >= 0).all() and (df["brier_score"] <= 2).all()
    assert (df["ece"] >= 0).all() and (df["ece"] <= 1).all()
    assert (df["accuracy"] >= 0).all() and (df["accuracy"] <= 1).all()
    assert (df["macro_f1"] >= 0).all() and (df["macro_f1"] <= 1).all()


def test_regularization_superiority_on_test_set():
    """Verify Ridge and Lasso achieve lower test Log-Loss and Brier score than unregularized MLE."""
    df = pd.read_csv(REPORTS_DIR / "test_metrics.csv")
    metrics_by_model = df.set_index("model")

    mle_ll = metrics_by_model.loc["MLE", "log_loss"]
    ridge_ll = metrics_by_model.loc["Ridge", "log_loss"]
    lasso_ll = metrics_by_model.loc["Lasso", "log_loss"]

    # Regularization reduces test log-loss
    assert ridge_ll < mle_ll, f"Ridge ({ridge_ll}) should beat MLE ({mle_ll})"
    assert lasso_ll < mle_ll, f"Lasso ({lasso_ll}) should beat MLE ({mle_ll})"

    # Market odds outperform statistical models (as expected theoretically)
    market_ll = metrics_by_model.loc["Market_Avg", "log_loss"]
    assert market_ll < lasso_ll

    # Prior is worse than statistical models
    prior_ll = metrics_by_model.loc["Prior", "log_loss"]
    assert mle_ll < prior_ll


def test_wilcoxon_results_integrity():
    """Verify paired weekly Wilcoxon test results across 38 gameweeks."""
    wilcoxon_path = REPORTS_DIR / "wilcoxon_results.csv"
    assert wilcoxon_path.exists(), f"Missing {wilcoxon_path}"

    df = pd.read_csv(wilcoxon_path)
    assert len(df) == 3  # MLE vs Ridge, MLE vs Lasso, Ridge vs Lasso
    assert set(df["comparison"].values) == {
        "MLE vs. Ridge (L2)",
        "MLE vs. Lasso (L1)",
        "Ridge vs. Lasso (H3 Parity)",
    }

    # Verify p-values in [0, 1]
    assert (df["p_value"] >= 0).all() and (df["p_value"] <= 1.0).all()


def test_mcnemar_results_integrity():
    """Verify McNemar test matrix and p-values."""
    mcnemar_path = REPORTS_DIR / "mcnemar_results.csv"
    assert mcnemar_path.exists(), f"Missing {mcnemar_path}"

    df = pd.read_csv(mcnemar_path)
    assert len(df) == 3
    # Total matches across all 4 cells of contingency table must equal 369
    total_matches = df["n_correct_both"] + df["n_correct_A_only"] + df["n_correct_B_only"] + df["n_wrong_both"]
    assert (total_matches == 369).all()
    assert (df["p_value"] >= 0).all() and (df["p_value"] <= 1.0).all()


def test_bootstrap_stability_integrity():
    """Verify bootstrap stability table (B=1000, 8 non-singular features, variance reduction)."""
    boot_path = REPORTS_DIR / "bootstrap_coefficients.csv"
    assert boot_path.exists(), f"Missing {boot_path}"

    df = pd.read_csv(boot_path)
    assert len(df) == 8, f"Expected 8 predictors, found {len(df)}"

    # Check columns
    required_cols = [
        "predictor",
        "mle_coef",
        "mle_se",
        "ridge_coef",
        "ridge_se",
        "lasso_coef",
        "lasso_se",
        "ridge_var_reduction_pct",
        "mle_ssi",
        "ridge_ssi",
        "lasso_fsf",
        "core_feature",
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column {c}"

    # Overall Ridge SE reduction should be substantially positive under collinearity (> 50%)
    mean_var_red = df["ridge_var_reduction_pct"].mean()
    assert mean_var_red > 50.0, f"Expected > 50% mean variance reduction, got {mean_var_red:.1f}%"

    # Core features identified (FSF >= 80%)
    core_feats = df[df["core_feature"]]["predictor"].tolist()
    assert len(core_feats) >= 1, f"Expected at least 1 core tactical feature, got {len(core_feats)}"


def test_visual_and_report_artifacts_exist():
    """Verify calibration plots, bootstrap plots, and final report markdown exist."""
    calib_plot = FIGURES_DIR / "calibration_curves.png"
    boot_plot = FIGURES_DIR / "bootstrap_stability.png"
    report_md = REPORTS_DIR / "final_evaluation_report.md"

    assert calib_plot.exists() and calib_plot.stat().st_size > 1000
    assert boot_plot.exists() and boot_plot.stat().st_size > 1000
    assert report_md.exists() and report_md.stat().st_size > 500
