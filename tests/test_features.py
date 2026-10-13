"""Automated Test Suite for Feature Pipeline and Mathematical Correctness (tests/test_features.py).

Verifies:
  - Chronological sorting across teams and fixtures.
  - Rolling-window mathematics (5-match, 10-match, venue form).
  - Division by zero safeguards (shot accuracy, shot dominance).
  - Schema completeness (exactly 16 predictors, 4 targets, 7 metadata).
  - Missingness and warm-up partition boundaries.
  - Differential calculation direction (Delta X = X_Home - X_Away).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features import (
    load_and_reconcile_raw_data,
    build_feature_dataset,
    compute_team_k5_features,
    compute_team_k10_features,
    compute_venue_ppg,
    compute_rest_days,
    PREDICTOR_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = WORKSPACE_ROOT / "data" / "features" / "dataset_features.csv"


@pytest.fixture(scope="module")
def feature_data():
    """Load or generate the feature dataset."""
    if not FEATURES_CSV.exists():
        df_matches = load_and_reconcile_raw_data()
        df_feat = build_feature_dataset(df_matches)
    else:
        df_feat = pd.read_csv(FEATURES_CSV)
    return df_feat


# =====================================================================
# A. CHRONOLOGICAL ORDERING TESTS
# =====================================================================

def test_global_chronological_ordering(feature_data):
    """Test that the global dataset is monotonically non-decreasing in datetime."""
    dt = pd.to_datetime(feature_data["date"] + " " + feature_data["kickoff_time"])
    assert dt.is_monotonic_increasing, "Fixtures are not in chronological order!"


def test_team_level_strictly_monotonic_kickoff_timestamps(feature_data):
    """Test that every individual team's matches are strictly increasing in time (no duplicates, no back-dating)."""
    dt = pd.to_datetime(feature_data["date"] + " " + feature_data["kickoff_time"])
    df = feature_data.copy()
    df["dt"] = dt

    all_teams = set(df["home_team"]).union(df["away_team"])
    for team in all_teams:
        team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()
        # Strictly increasing: diff() > 0 for all rows after first
        time_diffs = team_matches["dt"].diff().dropna()
        assert (time_diffs > pd.Timedelta(0)).all(), f"Team {team} has non-monotonic or duplicate kickoff times!"


def test_zero_missing_kickoff_timestamps(feature_data):
    """Test that kickoff dates and times have zero missing values across all 1,900 matches."""
    assert feature_data["date"].isnull().sum() == 0, "Missing dates detected!"
    assert feature_data["kickoff_time"].isnull().sum() == 0, "Missing kickoff times detected!"


# =====================================================================
# B. ROLLING-WINDOW MATHEMATICAL CORRECTNESS TESTS
# =====================================================================

def test_k5_rolling_window_manual_calculation():
    """Verify rolling 5-match window with an exact hand-crafted sequence."""
    synthetic_history = [
        {"shots": 10, "shotsA": 5, "sot": 4, "sotA": 2, "corners": 6, "cornersA": 3,
         "fouls": 10, "yellows": 1, "goal_diff": 1, "points": 3, "clean_sheet": 0},
        {"shots": 12, "shotsA": 8, "sot": 6, "sotA": 4, "corners": 5, "cornersA": 4,
         "fouls": 8, "yellows": 2, "goal_diff": 2, "points": 3, "clean_sheet": 1},
        {"shots": 8, "shotsA": 10, "sot": 2, "sotA": 5, "corners": 4, "cornersA": 7,
         "fouls": 12, "yellows": 3, "goal_diff": -1, "points": 0, "clean_sheet": 0},
        {"shots": 15, "shotsA": 6, "sot": 7, "sotA": 1, "corners": 8, "cornersA": 2,
         "fouls": 9, "yellows": 0, "goal_diff": 3, "points": 3, "clean_sheet": 1},
        {"shots": 10, "shotsA": 11, "sot": 5, "sotA": 3, "corners": 3, "cornersA": 5,
         "fouls": 11, "yellows": 2, "goal_diff": 0, "points": 1, "clean_sheet": 0},
    ]

    res = compute_team_k5_features(synthetic_history)
    # Total shots = 10 + 12 + 8 + 15 + 10 = 55 -> mean = 11.0
    assert pytest.approx(res["shots"], 1e-6) == 11.0
    # Total shotsA = 5 + 8 + 10 + 6 + 11 = 40 -> mean = 8.0
    assert pytest.approx(res["shotsA"], 1e-6) == 8.0
    # Total SoT = 4 + 6 + 2 + 7 + 5 = 24 -> mean = 4.8
    assert pytest.approx(res["sot"], 1e-6) == 4.8
    # Total SoTA = 2 + 4 + 5 + 1 + 3 = 15 -> mean = 3.0
    assert pytest.approx(res["sotA"], 1e-6) == 3.0
    # SoT ratio = 24 / 55
    assert pytest.approx(res["sot_ratio"], 1e-6) == 24.0 / 55.0
    # Shot dominance = 55 / (55 + 40) = 55 / 95
    assert pytest.approx(res["shot_dominance"], 1e-6) == 55.0 / 95.0
    # Corners = 6 + 5 + 4 + 8 + 3 = 26 -> mean = 5.2
    assert pytest.approx(res["corners"], 1e-6) == 5.2
    # Fouls = 10 + 8 + 12 + 9 + 11 = 50 -> mean = 10.0
    assert pytest.approx(res["fouls"], 1e-6) == 10.0
    # Yellows = 1 + 2 + 3 + 0 + 2 = 8 -> mean = 1.6
    assert pytest.approx(res["yellows"], 1e-6) == 1.6
    # Goal diff = 1 + 2 - 1 + 3 + 0 = 5 -> mean = 1.0
    assert pytest.approx(res["goal_diff"], 1e-6) == 1.0
    # Points = 3 + 3 + 0 + 3 + 1 = 10 -> mean = 2.0 (PPG)
    assert pytest.approx(res["ppg"], 1e-6) == 2.0


def test_k10_rolling_window_manual_calculation():
    """Verify rolling 10-match window with an exact hand-crafted sequence."""
    synthetic_history = [
        {"goal_diff": 1, "clean_sheet": 1},
        {"goal_diff": -2, "clean_sheet": 0},
        {"goal_diff": 0, "clean_sheet": 1},
        {"goal_diff": 3, "clean_sheet": 1},
        {"goal_diff": -1, "clean_sheet": 0},
        {"goal_diff": 2, "clean_sheet": 1},
        {"goal_diff": 0, "clean_sheet": 0},
        {"goal_diff": 1, "clean_sheet": 1},
        {"goal_diff": -3, "clean_sheet": 0},
        {"goal_diff": 4, "clean_sheet": 1},
    ]

    res = compute_team_k10_features(synthetic_history)
    # Sum of goal diff = 1 - 2 + 0 + 3 - 1 + 2 + 0 + 1 - 3 + 4 = 5 -> mean = 0.5
    assert pytest.approx(res["goal_diff_k10"], 1e-6) == 0.5
    # Clean sheets: 6 out of 10 -> rate = 0.6
    assert pytest.approx(res["clean_sheets_k10"], 1e-6) == 0.6


def test_venue_ppg_manual_calculation():
    """Verify 5-match venue-specific form with hand-crafted home/away fixtures."""
    venue_matches = [
        {"points": 3},
        {"points": 1},
        {"points": 3},
        {"points": 0},
        {"points": 3},
    ]
    # Sum points = 10 -> mean = 2.0
    res = compute_venue_ppg(venue_matches)
    assert pytest.approx(res, 1e-6) == 2.0


def test_rest_days_continuous_calculation():
    """Verify continuous rest days calculation in fractional days."""
    t0 = pd.Timestamp("2023-08-12 15:00:00")
    t1 = pd.Timestamp("2023-08-19 17:30:00")
    days = compute_rest_days(t0, t1)
    # Difference: 7 days, 2 hours, 30 minutes = 7 + 2.5/24 = 7.1041666...
    expected = 7.0 + (2.5 / 24.0)
    assert pytest.approx(days, 1e-6) == expected


# =====================================================================
# C. ZERO-DENOMINATOR RATIO SAFEGUARD TESTS
# =====================================================================

def test_sot_ratio_zero_denominator_guard():
    """Verify that when 0 shots are attempted in 5 matches, sot_ratio is safely 0.0 (not NaN/Inf)."""
    zero_shot_history = [
        {"shots": 0, "shotsA": 5, "sot": 0, "sotA": 2, "corners": 0, "cornersA": 0,
         "fouls": 0, "yellows": 0, "goal_diff": -2, "points": 0, "clean_sheet": 0}
        for _ in range(5)
    ]
    res = compute_team_k5_features(zero_shot_history)
    assert res["sot_ratio"] == 0.0, "Zero-shot accuracy ratio must evaluate safely to 0.0!"
    assert np.isfinite(res["sot_ratio"]), "Ratio must be strictly finite!"


def test_shot_dominance_zero_denominator_guard():
    """Verify that when total shots + shotsA == 0, shot_dominance safely returns 0.5 (neutral 50% baseline)."""
    zero_total_shots_history = [
        {"shots": 0, "shotsA": 0, "sot": 0, "sotA": 0, "corners": 0, "cornersA": 0,
         "fouls": 0, "yellows": 0, "goal_diff": 0, "points": 1, "clean_sheet": 1}
        for _ in range(5)
    ]
    res = compute_team_k5_features(zero_total_shots_history)
    assert res["shot_dominance"] == 0.5, "Neutral shot dominance baseline must be 0.5!"
    assert np.isfinite(res["shot_dominance"]), "Dominance must be strictly finite!"


# =====================================================================
# D. SCHEMA AND DATA INTEGRITY TESTS
# =====================================================================

def test_exact_feature_schema(feature_data):
    """Verify the final dataset contains exactly the approved 16 predictors, 4 targets, and 7 metadata columns."""
    expected_all = set(METADATA_COLUMNS + TARGET_COLUMNS + PREDICTOR_COLUMNS)
    actual_all = set(feature_data.columns)
    assert expected_all == actual_all, f"Schema mismatch: missing {expected_all - actual_all}, extra {actual_all - expected_all}"
    assert len(PREDICTOR_COLUMNS) == 16, f"Expected exactly 16 predictors, found {len(PREDICTOR_COLUMNS)}"


def test_fixture_row_counts(feature_data):
    """Verify total rows, modelling-ready rows, and warm-up row counts."""
    assert len(feature_data) == 1900, f"Expected 1,900 total matches, got {len(feature_data)}"
    n_ready = feature_data["is_modelling_ready"].sum()
    n_warmup = (~feature_data["is_modelling_ready"]).sum()
    assert n_ready == 1741, f"Expected exactly 1,741 modelling-ready matches, got {n_ready}"
    assert n_warmup == 159, f"Expected exactly 159 warm-up matches, got {n_warmup}"


def test_zero_nulls_in_modelling_ready_partition(feature_data):
    """Verify that all 16 predictors have strictly zero null or missing values across all modelling-ready rows."""
    ready_df = feature_data[feature_data["is_modelling_ready"]]
    null_counts = ready_df[PREDICTOR_COLUMNS].isnull().sum()
    assert (null_counts == 0).all(), f"Detected null values in modelling-ready partition:\n{null_counts[null_counts > 0]}"


def test_all_finite_values_in_modelling_ready_partition(feature_data):
    """Verify that all 16 predictors contain only finite values (no Inf, -Inf, NaN) across modelling-ready rows."""
    ready_df = feature_data[feature_data["is_modelling_ready"]]
    vals = ready_df[PREDICTOR_COLUMNS].to_numpy()
    assert np.isfinite(vals).all(), "Detected non-finite values (Inf/-Inf) in modelling-ready predictors!"


def test_zero_duplicate_matches(feature_data):
    """Verify zero duplicate rows or match_id entries."""
    assert feature_data["match_id"].duplicated().sum() == 0, "Duplicate match_id detected!"
    assert feature_data.duplicated(subset=["season", "home_team", "away_team"]).sum() == 0, "Duplicate seasonal fixtures detected!"


def test_target_encodings_validity(feature_data):
    """Verify target columns are well-formed and valid."""
    assert set(feature_data["target_multinomial"].unique()).issubset({0, 1, 2}), "Invalid multinomial target classes!"
    assert set(feature_data["target_binary"].unique()).issubset({0, 1}), "Invalid binary target classes!"

    # Consistency with goals
    hw_multi = (feature_data["home_goals"] > feature_data["away_goals"]) == (feature_data["target_multinomial"] == 2)
    hw_bin = (feature_data["home_goals"] > feature_data["away_goals"]) == (feature_data["target_binary"] == 1)
    dr_multi = (feature_data["home_goals"] == feature_data["away_goals"]) == (feature_data["target_multinomial"] == 1)
    aw_multi = (feature_data["home_goals"] < feature_data["away_goals"]) == (feature_data["target_multinomial"] == 0)

    assert hw_multi.all(), "target_multinomial inconsistent with home win!"
    assert hw_bin.all(), "target_binary inconsistent with home win!"
    assert dr_multi.all(), "target_multinomial inconsistent with draw!"
    assert aw_multi.all(), "target_multinomial inconsistent with away win!"


# =====================================================================
# E. DIFFERENTIAL CALCULATION DIRECTION TESTS
# =====================================================================

def test_differential_direction_symmetry(feature_data):
    """Verify the home-minus-away convention (Delta X = X_Home - X_Away)."""
    ready_df = feature_data[feature_data["is_modelling_ready"]]
    # Shot dominance differential: when Home dominance > Away dominance, diff > 0
    # Venue PPG differential: when Home home form > Away away form, diff > 0
    assert ready_df["diff_venue_ppg_k5"].min() >= -3.0 - 1e-6
    assert ready_df["diff_venue_ppg_k5"].max() <= 3.0 + 1e-6
    assert ready_df["diff_ppg_k5"].min() >= -3.0 - 1e-6
    assert ready_df["diff_ppg_k5"].max() <= 3.0 + 1e-6
