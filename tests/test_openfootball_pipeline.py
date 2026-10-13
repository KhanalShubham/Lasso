"""Automated Verification and Leakage Test Suite for OpenFootball Pipeline.

Phase 5 Quality Assurance Tests:
1. Raw file parsing and checksum integrity.
2. Target label derivation consistency.
3. Club normalization across all 5 seasons.
4. Chronological monotonic sequencing.
5. Rolling-window historical calculations ($j >= 1$).
6. Venue-specific form isolation (home games for Home, away games for Away).
7. Strict non-leakage invariant: current match $t$ outcome does NOT affect match $t$ features.
8. Strict non-leakage invariant: future matches $t + k$ do NOT affect match $t$ features.
9. Warm-up isolation ($N=159$) and missingness preservation (no zero-imputation).
10. Deterministic reproducibility.
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features_openfootball import (
    CANONICAL_MATCHES_CSV,
    OUTPUT_FEATURES_CSV,
    PRIMARY_DIFFERENTIAL_PREDICTORS,
    INDIVIDUAL_PREDICTORS,
    ALL_PREDICTORS,
    build_openfootball_feature_dataset,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_OF_DIR = WORKSPACE_ROOT / "data" / "raw" / "openfootball"
MANIFEST_JSON = WORKSPACE_ROOT / "data" / "audit" / "openfootball_ingestion_manifest.json"


def test_manifest_and_raw_checksums():
    """Verify all 5 raw OpenFootball season files exist and match recorded SHA-256 hashes."""
    assert MANIFEST_JSON.exists(), f"Missing manifest: {MANIFEST_JSON}"
    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for file_key, file_info in manifest["files"].items():
        file_path = WORKSPACE_ROOT / file_info["path"]
        assert file_path.exists(), f"Raw file missing: {file_path}"
        with open(file_path, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()
        assert computed_sha == file_info["sha256"], f"SHA256 mismatch for {file_path}"


def test_canonical_matches_schema_and_counts():
    """Verify canonical match table has exactly 1,900 rows (380 per season) and valid columns."""
    assert CANONICAL_MATCHES_CSV.exists(), f"Missing match table: {CANONICAL_MATCHES_CSV}"
    df = pd.read_csv(CANONICAL_MATCHES_CSV)

    assert len(df) == 1900, f"Expected 1,900 matches, got {len(df)}"
    assert df["match_id"].nunique() == 1900, "Duplicate match_id detected!"

    seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    for s in seasons:
        s_count = (df["season"] == s).sum()
        assert s_count == 380, f"Season {s} has {s_count} rows, expected 380."

    expected_cols = [
        "match_id", "season", "date", "time", "matchday",
        "home_team", "away_team", "home_goals", "away_goals",
        "target", "target_multinomial"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column in canonical match table: {col}"
        assert df[col].isnull().sum() == 0, f"Found nulls in column {col}"


def test_target_derivation_logic():
    """Verify that targets are derived strictly from full-time scorelines."""
    df = pd.read_csv(CANONICAL_MATCHES_CSV)

    home_wins = df[df["home_goals"] > df["away_goals"]]
    draws = df[df["home_goals"] == df["away_goals"]]
    away_wins = df[df["home_goals"] < df["away_goals"]]

    assert (home_wins["target"] == "HOME_WIN").all()
    assert (home_wins["target_multinomial"] == 2).all()

    assert (draws["target"] == "DRAW").all()
    assert (draws["target_multinomial"] == 1).all()

    assert (away_wins["target"] == "AWAY_WIN").all()
    assert (away_wins["target_multinomial"] == 0).all()

    # Verify official Premier League historical totals
    assert len(home_wins) == 838
    assert len(draws) == 432
    assert len(away_wins) == 630


def test_team_name_normalization():
    """Verify exactly 26 distinct Premier League clubs across the 5 seasons without fragments."""
    df = pd.read_csv(CANONICAL_MATCHES_CSV)
    teams = set(df["home_team"]).union(set(df["away_team"]))
    assert len(teams) == 26, f"Expected exactly 26 distinct clubs, found {len(teams)}: {teams}"

    # Verify no raw string noise remains
    for t in teams:
        assert not t.endswith(" FC"), f"Team name '{t}' still has ' FC' suffix"
        assert not t.startswith(" "), f"Team name '{t}' has leading whitespace"
        assert not t.endswith(" "), f"Team name '{t}' has trailing whitespace"


def test_chronological_ordering():
    """Verify matches are monotonically ordered by datetime."""
    df = pd.read_csv(CANONICAL_MATCHES_CSV)
    dt = pd.to_datetime(df["date"] + " " + df["time"])
    assert dt.is_monotonic_increasing, "Canonical matches are not strictly monotonically sorted by kickoff datetime!"


def test_feature_dataset_dimensions_and_warmup():
    """Verify feature dataset dimensions, warm-up isolation count (159), and ready count (1,741)."""
    assert OUTPUT_FEATURES_CSV.exists(), f"Features CSV missing: {OUTPUT_FEATURES_CSV}"
    df_feat = pd.read_csv(OUTPUT_FEATURES_CSV)

    assert len(df_feat) == 1900
    ready_count = df_feat["is_modelling_ready"].sum()
    warmup_count = (~df_feat["is_modelling_ready"]).sum()

    assert ready_count == 1741, f"Expected 1,741 modelling-ready rows, got {ready_count}"
    assert warmup_count == 159, f"Expected 159 warm-up rows, got {warmup_count}"

    # Verify partition readiness breakdown
    train_ready = df_feat[df_feat["season"].isin(["2019-20", "2020-21", "2021-22"])]["is_modelling_ready"].sum()
    val_ready = df_feat[df_feat["season"] == "2022-23"]["is_modelling_ready"].sum()
    test_ready = df_feat[df_feat["season"] == "2023-24"]["is_modelling_ready"].sum()

    assert train_ready == 1002, f"Expected 1,002 train rows, got {train_ready}"
    assert val_ready == 370, f"Expected 370 val rows, got {val_ready}"
    assert test_ready == 369, f"Expected 369 test rows, got {test_ready}"


def test_warmup_missingness_guarantee():
    """Verify that all predictor features are strictly NaN on warm-up fixtures."""
    df_feat = pd.read_csv(OUTPUT_FEATURES_CSV)
    warmup_df = df_feat[~df_feat["is_modelling_ready"]]

    for col in ALL_PREDICTORS:
        assert warmup_df[col].isnull().all(), f"Warm-up rows contain non-null values for {col}!"


def test_modelling_ready_zero_nulls_guarantee():
    """Verify that all predictor features have 0 nulls on modelling-ready fixtures."""
    df_feat = pd.read_csv(OUTPUT_FEATURES_CSV)
    ready_df = df_feat[df_feat["is_modelling_ready"]]

    for col in ALL_PREDICTORS:
        null_count = ready_df[col].isnull().sum()
        assert null_count == 0, f"Modelling-ready rows contain {null_count} nulls in {col}!"


def test_current_match_leakage_invariance():
    """CRITICAL TEST: Mutating in-game score of match t has ZERO effect on match t's features."""
    df_matches = pd.read_csv(CANONICAL_MATCHES_CSV)
    df_baseline = build_openfootball_feature_dataset(CANONICAL_MATCHES_CSV)

    # Pick a fixture in middle of dataset (e.g. index 500)
    target_idx = 500
    target_match_id = df_matches.iloc[target_idx]["match_id"]
    assert df_baseline.iloc[target_idx]["is_modelling_ready"], "Target fixture must be modelling ready"

    baseline_features = df_baseline.loc[
        df_baseline["match_id"] == target_match_id, PRIMARY_DIFFERENTIAL_PREDICTORS
    ].iloc[0].to_dict()

    # Mutate match t goals drastically in memory
    df_mutated = df_matches.copy()
    df_mutated.loc[target_idx, "home_goals"] = 12
    df_mutated.loc[target_idx, "away_goals"] = 9

    # Re-run pipeline using mutated table
    scratch_csv = WORKSPACE_ROOT / "data" / "processed" / "scratch_test_leakage.csv"
    try:
        df_mutated.to_csv(scratch_csv, index=False)
        df_after_mutation = build_openfootball_feature_dataset(scratch_csv)
        mutated_features = df_after_mutation.loc[
            df_after_mutation["match_id"] == target_match_id, PRIMARY_DIFFERENTIAL_PREDICTORS
        ].iloc[0].to_dict()

        for col in PRIMARY_DIFFERENTIAL_PREDICTORS:
            np.testing.assert_allclose(
                baseline_features[col],
                mutated_features[col],
                err_msg=f"LEAKAGE DETECTED! Feature '{col}' for match t changed when match t score was altered!"
            )
    finally:
        if scratch_csv.exists():
            scratch_csv.unlink()


def test_future_match_leakage_invariance():
    """CRITICAL TEST: Mutating future match t + 10 score has ZERO effect on match t's features."""
    df_matches = pd.read_csv(CANONICAL_MATCHES_CSV)
    df_baseline = build_openfootball_feature_dataset(CANONICAL_MATCHES_CSV)

    target_idx = 400
    future_idx = 410
    target_match_id = df_matches.iloc[target_idx]["match_id"]

    baseline_features = df_baseline.loc[
        df_baseline["match_id"] == target_match_id, PRIMARY_DIFFERENTIAL_PREDICTORS
    ].iloc[0].to_dict()

    df_mutated = df_matches.copy()
    df_mutated.loc[future_idx, "home_goals"] = 10
    df_mutated.loc[future_idx, "away_goals"] = 0

    scratch_csv = WORKSPACE_ROOT / "data" / "processed" / "scratch_test_future_leakage.csv"
    try:
        df_mutated.to_csv(scratch_csv, index=False)
        df_after_mutation = build_openfootball_feature_dataset(scratch_csv)
        mutated_features = df_after_mutation.loc[
            df_after_mutation["match_id"] == target_match_id, PRIMARY_DIFFERENTIAL_PREDICTORS
        ].iloc[0].to_dict()

        for col in PRIMARY_DIFFERENTIAL_PREDICTORS:
            np.testing.assert_allclose(
                baseline_features[col],
                mutated_features[col],
                err_msg=f"FUTURE LEAKAGE DETECTED! Feature '{col}' changed when future fixture {future_idx} was altered!"
            )
    finally:
        if scratch_csv.exists():
            scratch_csv.unlink()


def test_venue_specific_history_isolation():
    """Verify venue-specific points per match uses only home matches for home, away matches for away."""
    df_feat = pd.read_csv(OUTPUT_FEATURES_CSV)
    ready = df_feat[df_feat["is_modelling_ready"]].iloc[100]

    h_team = ready["home_team"]
    match_dt = pd.to_datetime(ready["date"] + " " + ready["kickoff_time"])

    df_matches = pd.read_csv(CANONICAL_MATCHES_CSV)
    df_matches["dt"] = pd.to_datetime(df_matches["date"] + " " + df_matches["time"])

    # Prior home games for home team strictly before match_dt
    h_home_games = df_matches[
        (df_matches["home_team"] == h_team) & (df_matches["dt"] < match_dt)
    ].sort_values(by="dt")

    assert len(h_home_games) >= 5
    last5_home = h_home_games.iloc[-5:]
    pts = []
    for _, g in last5_home.iterrows():
        if g["home_goals"] > g["away_goals"]:
            pts.append(3)
        elif g["home_goals"] == g["away_goals"]:
            pts.append(1)
        else:
            pts.append(0)

    expected_h_venue_ppg = np.mean(pts)
    np.testing.assert_allclose(ready["home_venue_ppg_k5"], expected_h_venue_ppg)


def test_rest_days_accuracy():
    """Verify rest days corresponds to exact calendar time elapsed since previous match."""
    df_feat = pd.read_csv(OUTPUT_FEATURES_CSV)
    ready = df_feat[df_feat["is_modelling_ready"]].iloc[250]

    h_team = ready["home_team"]
    cur_dt = pd.to_datetime(ready["date"] + " " + ready["kickoff_time"])

    df_matches = pd.read_csv(CANONICAL_MATCHES_CSV)
    df_matches["dt"] = pd.to_datetime(df_matches["date"] + " " + df_matches["time"])

    # Find immediately preceding match for home team
    prior_matches = df_matches[
        ((df_matches["home_team"] == h_team) | (df_matches["away_team"] == h_team))
        & (df_matches["dt"] < cur_dt)
    ].sort_values(by="dt")

    assert len(prior_matches) > 0
    last_match = prior_matches.iloc[-1]
    expected_days = (cur_dt - last_match["dt"]).total_seconds() / 86400.0

    np.testing.assert_allclose(ready["home_rest_days"], expected_days)


def test_unique_match_records_and_no_duplicates():
    """Verify that every match record is uniquely identifiable by season, matchday, and teams."""
    df = pd.read_csv(CANONICAL_MATCHES_CSV)
    dups = df.duplicated(subset=["season", "home_team", "away_team"])
    assert dups.sum() == 0, f"Found {dups.sum()} duplicate fixtures across home/away season pairings!"


def test_stable_feature_names_types_and_order():
    """Verify stable feature schema, naming, column order, and numerical types."""
    df_feat = pd.read_csv(OUTPUT_FEATURES_CSV)
    ready = df_feat[df_feat["is_modelling_ready"]]

    for col in PRIMARY_DIFFERENTIAL_PREDICTORS:
        assert col in df_feat.columns, f"Missing primary predictor: {col}"
        assert pd.api.types.is_float_dtype(ready[col]), f"Column {col} is not float dtype: {ready[col].dtype}"

    for col in INDIVIDUAL_PREDICTORS:
        assert col in df_feat.columns, f"Missing individual predictor: {col}"
        assert pd.api.types.is_float_dtype(ready[col]), f"Column {col} is not float dtype: {ready[col].dtype}"


def test_reproducible_pipeline_output():
    """Verify that running build_openfootball_feature_dataset produces identical results."""
    df_first = build_openfootball_feature_dataset(CANONICAL_MATCHES_CSV)
    df_second = build_openfootball_feature_dataset(CANONICAL_MATCHES_CSV)

    pd.testing.assert_frame_equal(df_first, df_second)

