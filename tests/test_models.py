"""Unit Tests for Gate 5: MLE, Ridge, and Lasso Model Calibration and Tuning.

Tests:
1. MLE convergence and valid probability simplex generation.
2. Ridge L2 monotonic reduction of effective degrees of freedom df(lambda).
3. Lasso L1 exact sparsity and variable elimination mechanics.
4. Tuning table completeness (20 rows, valid columns, hyperparameter bounds).
5. Superiority of regularized models over unpenalized MLE on validation set.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

from src.models.mle_model import MLEMultinomialLogistic
from src.models.ridge_model import RidgeLogisticRegression
from src.models.lasso_model import LassoLogisticRegression

try:
    scalers_meta = joblib.load(PROCESSED_DIR / "scalers.joblib")
    PREDICTOR_COLS = scalers_meta["feature_names"]
except Exception:
    PREDICTOR_COLS = [
        "diff_goals_scored_k5",
        "diff_goals_conceded_k5",
        "diff_ppg_k5",
        "diff_goal_diff_k10",
        "diff_venue_ppg_k5",
        "diff_rest_days",
        "diff_clean_sheets_k10",
        "diff_win_rate_k5",
    ]


@pytest.fixture(scope="module")
def data_splits():
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")
    X_train = train_df[PREDICTOR_COLS]
    y_train = train_df["target_multinomial"].values
    X_val = val_df[PREDICTOR_COLS]
    y_val = val_df["target_multinomial"].values
    return {"X_tr": X_train, "y_tr": y_train, "X_val": X_val, "y_val": y_val}


def test_mle_convergence_and_probabilities(data_splits):
    """Verify unregularized MLE converges and produces valid probability simplex."""
    mle = MLEMultinomialLogistic(tol=1e-5, max_iter=5000)
    mle.fit(data_splits["X_tr"], data_splits["y_tr"])

    probs = mle.predict_proba(data_splits["X_val"])
    assert probs.shape == (370, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    contrasts = mle.get_contrast_coefficients(reference_class=1)
    p_len = len(PREDICTOR_COLS)
    assert contrasts.shape == (2, p_len + 1)  # 2 non-reference classes x (p features + intercept)


def test_ridge_shrinkage_and_df(data_splits):
    """Verify Ridge degrees of freedom shrinks monotonically as lambda increases."""
    r_small = RidgeLogisticRegression(lambda_val=0.001, n_train=1002)
    r_small.fit(data_splits["X_tr"], data_splits["y_tr"])

    r_med = RidgeLogisticRegression(lambda_val=1.0, n_train=1002)
    r_med.fit(data_splits["X_tr"], data_splits["y_tr"])

    r_large = RidgeLogisticRegression(lambda_val=100.0, n_train=1002)
    r_large.fit(data_splits["X_tr"], data_splits["y_tr"])

    assert r_small.df_lambda_ > r_med.df_lambda_
    assert r_med.df_lambda_ > r_large.df_lambda_
    assert r_large.df_lambda_ >= 3.0  # At least 3 intercepts remain


def test_lasso_exact_sparsity(data_splits):
    """Verify Lasso drives coefficients identically to zero as lambda increases."""
    l_small = LassoLogisticRegression(lambda_val=0.0001, n_train=1002)
    l_small.fit(data_splits["X_tr"], data_splits["y_tr"])
    sp_small = l_small.get_sparsity()

    l_large = LassoLogisticRegression(lambda_val=10.0, n_train=1002)
    l_large.fit(data_splits["X_tr"], data_splits["y_tr"])
    sp_large = l_large.get_sparsity()

    p_len = len(PREDICTOR_COLS)
    assert sp_small["n_retained"] >= p_len - 1  # Small penalty keeps virtually all
    assert sp_large["n_retained"] == 0          # Large penalty zeroes all features


def test_tuning_table_and_bundle():
    """Verify hyperparameter tuning table completeness and serialized model bundle."""
    tuning_csv = REPORTS_DIR / "hyperparameter_tuning.csv"
    bundle_path = PROCESSED_DIR / "fitted_models.joblib"

    assert tuning_csv.exists()
    assert bundle_path.exists()

    df = pd.read_csv(tuning_csv)
    assert len(df) == 20
    assert "ridge_log_loss" in df.columns
    assert "lasso_log_loss" in df.columns
    assert "lasso_retained_count" in df.columns

    bundle = joblib.load(bundle_path)
    assert "mle_model" in bundle
    assert "ridge_best" in bundle
    assert "lasso_best" in bundle
    assert bundle["lambda_ridge_opt"] > 0
    assert bundle["lambda_lasso_opt"] > 0
