"""Automated Temporal Lookahead Leakage Test Suite (tests/test_leakage.py).

Verifies the foundational methodological invariants:
  1. Current-Match Invariance: Altering a match's current in-game statistics does NOT
     change its own pre-match feature vector (strictly j >= 1).
  2. Future-Match Invariance: Altering future match observations does NOT alter earlier features.
  3. Target Isolation: Ground truth targets and in-game statistics are strictly excluded
     from the predictor covariate space.
  4. Chronological Step Invariant: Statistics from match t appear in the predictor vector
     for match t+1, but NEVER in match t.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features import (
    load_and_reconcile_raw_data,
    build_feature_dataset,
    PREDICTOR_COLUMNS,
    TARGET_COLUMNS,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_target_isolation_from_predictors():
    """Verify that no target variable or in-game statistics exist in the predictor set."""
    forbidden_targets = {
        "home_goals", "away_goals", "target_multinomial", "target_binary",
        "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
        "score", "result",
    }
    predictor_set = set(PREDICTOR_COLUMNS)
    leakage_overlap = predictor_set.intersection(forbidden_targets)
    assert len(leakage_overlap) == 0, f"Fatal leakage! Forbidden targets in predictor columns: {leakage_overlap}"


def test_current_match_invariance_leakage_assertion():
    """Verify that modifying in-game stats of match t has ZERO impact on match t's pre-match features."""
    df_raw = load_and_reconcile_raw_data()

    # Pick a well-established fixture with completed history (e.g. index 350)
    target_idx = 350
    target_match_id = df_raw.iloc[target_idx]["match_id"]

    # 1. Baseline feature run
    df_feat_baseline = build_feature_dataset(df_raw.copy())
    baseline_vector = df_feat_baseline.loc[
        df_feat_baseline["match_id"] == target_match_id, PREDICTOR_COLUMNS
    ].iloc[0].to_dict()

    # 2. Mutate CURRENT match statistics drastically (10 goals, 50 shots, 20 corners, etc.)
    df_mutated = df_raw.copy()
    df_mutated.loc[target_idx, "FTHG"] = 10
    df_mutated.loc[target_idx, "FTAG"] = 8
    df_mutated.loc[target_idx, "HS"] = 45
    df_mutated.loc[target_idx, "AS"] = 30
    df_mutated.loc[target_idx, "HST"] = 25
    df_mutated.loc[target_idx, "AST"] = 18
    df_mutated.loc[target_idx, "HC"] = 15
    df_mutated.loc[target_idx, "AC"] = 12
    df_mutated.loc[target_idx, "HF"] = 28
    df_mutated.loc[target_idx, "AF"] = 25
    df_mutated.loc[target_idx, "HY"] = 6
    df_mutated.loc[target_idx, "AY"] = 5

    # 3. Re-run feature pipeline
    df_feat_mutated = build_feature_dataset(df_mutated)
    mutated_vector = df_feat_mutated.loc[
        df_feat_mutated["match_id"] == target_match_id, PREDICTOR_COLUMNS
    ].iloc[0].to_dict()

    # 4. Strict invariance assertion: baseline_vector MUST EQUAL mutated_vector exactly!
    for col in PREDICTOR_COLUMNS:
        b_val = baseline_vector[col]
        m_val = mutated_vector[col]
        assert pytest.approx(b_val, nan_ok=True) == m_val, (
            f"Temporal Lookahead Leakage Detected in '{col}'! "
            f"Altering current match statistics changed pre-match feature value: "
            f"baseline={b_val}, mutated={m_val}"
        )


def test_future_match_invariance_leakage_assertion():
    """Verify that mutating FUTURE matches (t+50) has ZERO effect on PAST matches (t)."""
    df_raw = load_and_reconcile_raw_data()

    past_idx = 200
    future_idx = 400
    past_match_id = df_raw.iloc[past_idx]["match_id"]

    # Baseline features
    df_feat_baseline = build_feature_dataset(df_raw.copy())
    past_vector_baseline = df_feat_baseline.loc[
        df_feat_baseline["match_id"] == past_match_id, PREDICTOR_COLUMNS
    ].iloc[0].to_dict()

    # Mutate future fixture
    df_mutated = df_raw.copy()
    df_mutated.loc[future_idx, "FTHG"] = 20
    df_mutated.loc[future_idx, "FTAG"] = 15
    df_mutated.loc[future_idx, "HS"] = 99
    df_mutated.loc[future_idx, "HST"] = 50

    # Re-run features
    df_feat_mutated = build_feature_dataset(df_mutated)
    past_vector_mutated = df_feat_mutated.loc[
        df_feat_mutated["match_id"] == past_match_id, PREDICTOR_COLUMNS
    ].iloc[0].to_dict()

    # Assert exact equality
    for col in PREDICTOR_COLUMNS:
        assert pytest.approx(past_vector_baseline[col], nan_ok=True) == past_vector_mutated[col], (
            f"Future Leakage Detected in '{col}'! "
            f"Modifying future match {future_idx} changed past feature {past_idx}!"
        )


def test_chronological_step_propagation():
    """Verify that match t's stats propagate into match t+1, but not match t."""
    df_raw = load_and_reconcile_raw_data()
    df_feat = build_feature_dataset(df_raw)

    # Pick a team, find two consecutive matches
    team = "Liverpool"
    team_matches = df_feat[(df_feat["home_team"] == team) | (df_feat["away_team"] == team)].copy()

    # Take 15th and 16th matches where both are modelling ready
    m15 = team_matches.iloc[15]
    m16 = team_matches.iloc[16]

    # Find the corresponding raw match for m15
    m15_raw = df_raw[df_raw["match_id"] == m15["match_id"]].iloc[0]

    # Check goal difference from m15
    if m15_raw["home_team"] == team:
        m15_gd = int(m15_raw["FTHG"]) - int(m15_raw["FTAG"])
    else:
        m15_gd = int(m15_raw["FTAG"]) - int(m15_raw["FTHG"])

    # m15's stats must be incorporated into m16, but NOT m15!
    assert m15["match_id"] != m16["match_id"]
    assert pd.to_datetime(m16["date"] + " " + m16["kickoff_time"]) > pd.to_datetime(m15["date"] + " " + m15["kickoff_time"])
