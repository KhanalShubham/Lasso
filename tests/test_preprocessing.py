"""Unit Tests for Gate 4: Chronological Validation and Frozen Preprocessing.

Tests:
1. Exact row counts per chronological partition (Train: 1,002, Val: 370, Test: 369).
2. Strict season boundaries per partition (no temporal mixing).
3. Zero missing values across all 16 differential candidate predictors.
4. Scaler fitted strictly on training partition and frozen for val/test (Invariant 5).
5. Training class frequencies correctly computed strictly on training split.
6. VIF summary completeness (16 features, all VIF >= 1.0, condition numbers computed).
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

PREDICTOR_COLS = [
    "diff_sot_k5",
    "diff_sotA_k5",
    "diff_shots_k5",
    "diff_shotsA_k5",
    "diff_sot_ratio_k5",
    "diff_shot_dominance_k5",
    "diff_corners_k5",
    "diff_cornersA_k5",
    "diff_fouls_k5",
    "diff_yellows_k5",
    "diff_goal_diff_k5",
    "diff_goal_diff_k10",
    "diff_clean_sheets_k10",
    "diff_ppg_k5",
    "diff_venue_ppg_k5",
    "diff_rest_days",
]


@pytest.fixture(scope="module")
def processed_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_data.csv")
    scalers_meta = joblib.load(PROCESSED_DIR / "scalers.joblib")
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "meta": scalers_meta,
    }


def test_partition_row_counts(processed_data):
    """Verify exact modelling-ready row counts across the three chronological partitions."""
    train_df = processed_data["train"]
    val_df = processed_data["val"]
    test_df = processed_data["test"]

    assert len(train_df) == 1002, f"Expected 1,002 training matches, got {len(train_df)}"
    assert len(val_df) == 370, f"Expected 370 validation matches, got {len(val_df)}"
    assert len(test_df) == 369, f"Expected 369 test matches, got {len(test_df)}"
    assert len(train_df) + len(val_df) + len(test_df) == 1741


def test_partition_seasons(processed_data):
    """Verify strict chronological non-overlapping seasons across partitions."""
    train_seasons = set(processed_data["train"]["season"].unique())
    val_seasons = set(processed_data["val"]["season"].unique())
    test_seasons = set(processed_data["test"]["season"].unique())

    assert train_seasons == {"2019-2020", "2020-2021", "2021-2022"}
    assert val_seasons == {"2022-2023"}
    assert test_seasons == {"2023-2024"}

    # Zero overlap
    assert len(train_seasons.intersection(val_seasons)) == 0
    assert len(train_seasons.intersection(test_seasons)) == 0
    assert len(val_seasons.intersection(test_seasons)) == 0


def test_zero_nulls_in_all_partitions(processed_data):
    """Verify zero missing values across all 16 candidate predictors."""
    for split_name in ["train", "val", "test"]:
        df = processed_data[split_name]
        null_counts = df[PREDICTOR_COLS].isnull().sum()
        assert null_counts.sum() == 0, f"Found nulls in {split_name}: {null_counts[null_counts > 0]}"


def test_scaler_frozen_strictly_on_training(processed_data):
    """Verify scaler was fitted on training set (mean 0, std 1) and frozen for val/test."""
    train_df = processed_data["train"]
    val_df = processed_data["val"]
    test_df = processed_data["test"]
    meta = processed_data["meta"]

    # Training set must have approximately 0 mean and 1 std
    for col in PREDICTOR_COLS:
        col_mean = train_df[col].mean()
        col_std = train_df[col].std(ddof=0)
        assert abs(col_mean) < 1e-4, f"Training mean for {col} not 0: {col_mean}"
        assert abs(col_std - 1.0) < 1e-4, f"Training std for {col} not 1: {col_std}"

    # Validation and test sets must NOT be identically mean 0 or std 1 (they are out-of-sample)
    val_means = [val_df[c].mean() for c in PREDICTOR_COLS]
    test_means = [test_df[c].mean() for c in PREDICTOR_COLS]
    assert any(abs(m) > 0.01 for m in val_means), "Validation set appears refitted instead of frozen!"
    assert any(abs(m) > 0.01 for m in test_means), "Test set appears refitted instead of frozen!"

    # Scaler metadata checks
    assert len(meta["train_means"]) == 16
    assert len(meta["train_stds"]) == 16
    assert meta["train_sample_size"] == 1002


def test_training_class_frequencies_priors(processed_data):
    """Verify class-frequency priors are computed strictly on training partition."""
    meta = processed_data["meta"]
    freqs = meta["train_class_frequencies"]
    counts = meta["train_class_counts"]

    # Check exact training counts: 424 Home (2), 225 Draw (1), 353 Away (0)
    assert counts[2] == 424
    assert counts[1] == 225
    assert counts[0] == 353

    # Check frequencies
    assert abs(freqs[2] - 424 / 1002) < 1e-6
    assert abs(freqs[1] - 225 / 1002) < 1e-6
    assert abs(freqs[0] - 353 / 1002) < 1e-6
    assert abs(sum(freqs.values()) - 1.0) < 1e-6


def test_vif_summary_completeness():
    """Verify VIF summary table and condition number diagnostics."""
    vif_csv = REPORTS_DIR / "vif_summary.csv"
    diag_json = REPORTS_DIR / "vif_diagnostics.json"

    assert vif_csv.exists(), "vif_summary.csv does not exist"
    assert diag_json.exists(), "vif_diagnostics.json does not exist"

    vif_df = pd.read_csv(vif_csv)
    assert len(vif_df) == 16, f"Expected 16 features in VIF table, got {len(vif_df)}"

    required_cols = {"predictor", "vif", "r_squared", "collinearity_cluster", "top_correlate", "max_pearson_r"}
    assert required_cols.issubset(set(vif_df.columns))

    # All VIFs must be >= 1.0
    assert (vif_df["vif"] >= 1.0).all()
    # All R^2 in [0, 1)
    assert (vif_df["r_squared"] >= 0.0).all()
    assert (vif_df["r_squared"] < 1.0).all()

    with open(diag_json, "r", encoding="utf-8") as f:
        diag = json.load(f)

    assert diag["train_sample_size"] == 1002
    assert diag["condition_number_design_matrix_kappa_X"] > 1.0
    assert diag["condition_number_hessian_kappa_H0"] > 1.0
    assert diag["vif_threshold_ge_5_count"] >= 1
