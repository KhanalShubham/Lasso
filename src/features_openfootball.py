"""OpenFootball Result-Derived Feature Engineering Pipeline.

Research Study: A Comparative Study of Maximum Likelihood, Ridge,
and Lasso Logistic Regression for Football Match Outcome Prediction.

Governing Decision: ADR-013 (OpenFootball Public-Domain Data Pipeline)
Source: https://github.com/openfootball/england (CC0-1.0 Universal)

This module:
1. Ingests canonical match records from data/processed/epl_matches.csv.
2. Chronologically computes pre-match rolling historical features strictly over past matches (j >= 1).
3. Computes result-derived candidate features:
   - Goals scored / conceded (k=5)
   - Points per match (k=5; 3 for Win, 1 for Draw, 0 for Loss)
   - Goal difference (k=10 and k=5)
   - Clean sheet rate (k=10)
   - Win rate (k=5)
   - Venue-specific points per match (k=5; home-only for Home, away-only for Away)
   - Inter-match rest days
   - Symmetrical home-minus-away differentials (Delta X = X_Home - X_Away)
4. Enforces warm-up cold-start isolation:
   - Rows where either team lacks k >= 10 overall or k >= 5 venue history are flagged is_modelling_ready=False.
   - Warm-up feature values remain NaN (strictly never imputed with 0).
5. Exports:
   - data/features/dataset_features.csv (N=1,900 total, N=1,741 modelling-ready)
   - data/features/feature_dictionary.md (comprehensive definitions and leakage safeguards)
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
FEATURES_DIR = WORKSPACE_ROOT / "data" / "features"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

CANONICAL_MATCHES_CSV = PROCESSED_DIR / "epl_matches.csv"
OUTPUT_FEATURES_CSV = FEATURES_DIR / "dataset_features.csv"
FEATURE_DICTIONARY_MD = FEATURES_DIR / "feature_dictionary.md"

# Primary differential predictors (non-singular candidate set for modeling)
PRIMARY_DIFFERENTIAL_PREDICTORS = [
    "diff_goals_scored_k5",
    "diff_goals_conceded_k5",
    "diff_ppg_k5",
    "diff_goal_diff_k10",
    "diff_venue_ppg_k5",
    "diff_rest_days",
    "diff_clean_sheets_k10",
    "diff_win_rate_k5",
]

# Additional derived differential and individual features
ADDITIONAL_DIFFERENTIAL_PREDICTORS = [
    "diff_goal_diff_k5",
]

# Team-specific individual features
INDIVIDUAL_PREDICTORS = [
    "home_goals_scored_k5",
    "away_goals_scored_k5",
    "home_goals_conceded_k5",
    "away_goals_conceded_k5",
    "home_ppg_k5",
    "away_ppg_k5",
    "home_gd_k10",
    "away_gd_k10",
    "home_venue_ppg_k5",
    "away_venue_ppg_k5",
    "home_rest_days",
    "away_rest_days",
    "home_gd_k5",
    "away_gd_k5",
    "home_clean_sheets_k10",
    "away_clean_sheets_k10",
    "home_win_rate_k5",
    "away_win_rate_k5",
]

ALL_PREDICTORS = PRIMARY_DIFFERENTIAL_PREDICTORS + ADDITIONAL_DIFFERENTIAL_PREDICTORS + INDIVIDUAL_PREDICTORS

METADATA_COLS = [
    "match_id",
    "season",
    "date",
    "kickoff_time",
    "home_team",
    "away_team",
    "is_modelling_ready",
]

TARGET_COLS = [
    "home_goals",
    "away_goals",
    "target",
    "target_multinomial",
    "target_binary",
]


def compute_team_k5_metrics(history: list[dict]) -> dict[str, float]:
    """Compute rolling 5-match form metrics strictly from previous completed matches."""
    if len(history) < 5:
        return {
            "goals_scored_k5": np.nan,
            "goals_conceded_k5": np.nan,
            "ppg_k5": np.nan,
            "gd_k5": np.nan,
            "win_rate_k5": np.nan,
        }
    recent5 = history[-5:]
    return {
        "goals_scored_k5": float(np.mean([m["goals_scored"] for m in recent5])),
        "goals_conceded_k5": float(np.mean([m["goals_conceded"] for m in recent5])),
        "ppg_k5": float(np.mean([m["points"] for m in recent5])),
        "gd_k5": float(np.mean([m["goals_scored"] - m["goals_conceded"] for m in recent5])),
        "win_rate_k5": float(np.mean([1.0 if m["points"] == 3 else 0.0 for m in recent5])),
    }


def compute_team_k10_metrics(history: list[dict]) -> dict[str, float]:
    """Compute rolling 10-match sustained metrics strictly from previous completed matches."""
    if len(history) < 10:
        return {
            "gd_k10": np.nan,
            "clean_sheets_k10": np.nan,
        }
    recent10 = history[-10:]
    return {
        "gd_k10": float(np.mean([m["goals_scored"] - m["goals_conceded"] for m in recent10])),
        "clean_sheets_k10": float(np.mean([1.0 if m["goals_conceded"] == 0 else 0.0 for m in recent10])),
    }


def compute_venue_ppg_k5(venue_history: list[dict]) -> float:
    """Compute previous 5-match venue-specific points per match."""
    if len(venue_history) < 5:
        return np.nan
    recent5_venue = venue_history[-5:]
    return float(np.mean([m["points"] for m in recent5_venue]))


def compute_rest_days(last_dt: pd.Timestamp | None, cur_dt: pd.Timestamp) -> float:
    """Compute inter-match rest days in fractional calendar days."""
    if last_dt is None:
        return np.nan
    delta_seconds = (cur_dt - last_dt).total_seconds()
    return float(delta_seconds / 86400.0)


def build_openfootball_feature_dataset(matches_csv: Path = CANONICAL_MATCHES_CSV) -> pd.DataFrame:
    """Build the complete pre-match feature dataset strictly chronologically.
    
    Guarantees:
    - Zero lookahead leakage: all metrics sum strictly over j >= 1 prior matches.
    - In-game stats of current match t are strictly excluded.
    - Warm-up fixtures with < 10 overall or < 5 venue history are flagged is_modelling_ready=False.
    - Missing values in warm-up rows are left as NaN, never filled with zero or future data.
    """
    if not matches_csv.exists():
        raise FileNotFoundError(f"Missing canonical match file: {matches_csv}")

    logger.info("Loading canonical match records from %s", matches_csv)
    df = pd.read_csv(matches_csv)
    if len(df) != 1900:
        raise ValueError(f"Expected exactly 1,900 matches, got {len(df)}")

    # Ensure datetime parsing and chronological order
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values(by=["datetime", "match_id"]).reset_index(drop=True)

    # Initialize historical state tracking
    all_teams = sorted(list(set(df["home_team"]).union(set(df["away_team"]))))
    team_history = {team: [] for team in all_teams}
    team_venue_history = {team: {"home": [], "away": []} for team in all_teams}
    team_last_dt = {team: None for team in all_teams}

    feature_records = []

    logger.info("Iterating through 1,900 matches chronologically to compute features...")
    for idx, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        cur_dt = row["datetime"]

        h_hist = team_history[h]
        a_hist = team_history[a]
        h_ven_hist = team_venue_history[h]["home"]
        a_ven_hist = team_venue_history[a]["away"]

        # Evaluate warm-up status
        is_ready = (
            len(h_hist) >= 10
            and len(a_hist) >= 10
            and len(h_ven_hist) >= 5
            and len(a_ven_hist) >= 5
        )

        # Compute raw metrics strictly before match t
        h_k5 = compute_team_k5_metrics(h_hist)
        a_k5 = compute_team_k5_metrics(a_hist)
        h_k10 = compute_team_k10_metrics(h_hist)
        a_k10 = compute_team_k10_metrics(a_hist)
        h_ven = compute_venue_ppg_k5(h_ven_hist)
        a_ven = compute_venue_ppg_k5(a_ven_hist)
        h_rest = compute_rest_days(team_last_dt[h], cur_dt)
        a_rest = compute_rest_days(team_last_dt[a], cur_dt)

        # Build feature dictionary
        record = {
            "match_id": row["match_id"],
            "season": row["season"],
            "date": row["date"],
            "kickoff_time": row["time"],
            "home_team": h,
            "away_team": a,
            "is_modelling_ready": is_ready,
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "target": row["target"],
            "target_multinomial": int(row["target_multinomial"]),
            "target_binary": 1 if int(row["home_goals"]) > int(row["away_goals"]) else 0,
        }

        if is_ready:
            # Symmetrical differentials: Delta X = X_Home - X_Away
            record["diff_goals_scored_k5"] = h_k5["goals_scored_k5"] - a_k5["goals_scored_k5"]
            record["diff_goals_conceded_k5"] = h_k5["goals_conceded_k5"] - a_k5["goals_conceded_k5"]
            record["diff_ppg_k5"] = h_k5["ppg_k5"] - a_k5["ppg_k5"]
            record["diff_goal_diff_k10"] = h_k10["gd_k10"] - a_k10["gd_k10"]
            record["diff_venue_ppg_k5"] = h_ven - a_ven
            record["diff_rest_days"] = h_rest - a_rest
            record["diff_goal_diff_k5"] = h_k5["gd_k5"] - a_k5["gd_k5"]
            record["diff_clean_sheets_k10"] = h_k10["clean_sheets_k10"] - a_k10["clean_sheets_k10"]
            record["diff_win_rate_k5"] = h_k5["win_rate_k5"] - a_k5["win_rate_k5"]

            # Individual team metrics
            record["home_goals_scored_k5"] = h_k5["goals_scored_k5"]
            record["away_goals_scored_k5"] = a_k5["goals_scored_k5"]
            record["home_goals_conceded_k5"] = h_k5["goals_conceded_k5"]
            record["away_goals_conceded_k5"] = a_k5["goals_conceded_k5"]
            record["home_ppg_k5"] = h_k5["ppg_k5"]
            record["away_ppg_k5"] = a_k5["ppg_k5"]
            record["home_gd_k10"] = h_k10["gd_k10"]
            record["away_gd_k10"] = a_k10["gd_k10"]
            record["home_venue_ppg_k5"] = h_ven
            record["away_venue_ppg_k5"] = a_ven
            record["home_rest_days"] = h_rest
            record["away_rest_days"] = a_rest
            record["home_gd_k5"] = h_k5["gd_k5"]
            record["away_gd_k5"] = a_k5["gd_k5"]
            record["home_clean_sheets_k10"] = h_k10["clean_sheets_k10"]
            record["away_clean_sheets_k10"] = a_k10["clean_sheets_k10"]
            record["home_win_rate_k5"] = h_k5["win_rate_k5"]
            record["away_win_rate_k5"] = a_k5["win_rate_k5"]
        else:
            # Warm-up fixtures: preserve NaN missingness strictly
            for col in ALL_PREDICTORS:
                record[col] = np.nan

        feature_records.append(record)

        # Update historical state strictly AFTER recording match t features (j >= 1)
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if ag == hg else 0)

        h_match_info = {
            "datetime": cur_dt,
            "goals_scored": hg,
            "goals_conceded": ag,
            "points": h_pts,
            "venue": "home",
        }
        a_match_info = {
            "datetime": cur_dt,
            "goals_scored": ag,
            "goals_conceded": hg,
            "points": a_pts,
            "venue": "away",
        }

        team_history[h].append(h_match_info)
        team_history[a].append(a_match_info)
        team_venue_history[h]["home"].append(h_match_info)
        team_venue_history[a]["away"].append(a_match_info)
        team_last_dt[h] = cur_dt
        team_last_dt[a] = cur_dt

    df_out = pd.DataFrame(feature_records)
    logger.info(
        "Feature engineering complete. Total rows: %d, Modelling-ready: %d, Warm-up: %d",
        len(df_out),
        df_out["is_modelling_ready"].sum(),
        (~df_out["is_modelling_ready"]).sum(),
    )
    return df_out


def generate_feature_dictionary_markdown(output_path: Path = FEATURE_DICTIONARY_MD):
    content = r"""# OpenFootball Result-Derived Feature Dictionary

**Research Study:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Source Authority:** OpenFootball England Repository (CC0-1.0 Public Domain Dedication)  
**Governing ADR:** [ADR-013 (DECISIONS.md)](file:///d:/MLE-Lasso/DECISIONS.md)  
**Feature Dataset:** [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv)  

---

## 1. Feature Engineering Principles & Leakage Invariants

1. **Temporal Strictness ($j \ge 1$):** All rolling statistics for match $t$ at kickoff timestamp $\tau_t$ sum strictly over preceding matches $t - j$ ($j \in [1, k]$). Under no circumstance is the outcome or score of match $t$ included in its own covariate vector.
2. **Warm-Up Cold-Start Isolation:** Fixtures where either competing club has completed fewer than 10 top-flight matches overall ($N_{\\text{team}} < 10$) or fewer than 5 venue-specific matches ($N_{\\text{venue}} < 5$) are explicitly flagged with `is_modelling_ready = False`. Their feature values remain `NaN` (missing) and are never filled with zero, league averages, or future data.
3. **Symmetrical Differentials:** To capture the relative competitive balance between home and away clubs in multinomial outcome forecasting, primary predictors are parameterized as home-minus-away differentials:
   $$\\Delta X = X_{\\text{Home}} - X_{\\text{Away}}$$
4. **Chronological Monotonicity:** Matches are sorted strictly by kickoff timestamp $(\\text{date}, \\text{time})$, ensuring that rescheduled and postponed fixtures appear in their actual historical sequence.

---

## 2. Primary Differential Predictors ($p = 9$)

| Predictor Name | Rolling Window ($k$) | Mathematical Formulation | Tactical Interpretation | Leakage Safeguard |
| :--- | :---: | :--- | :--- | :--- |
| `diff_goals_scored_k5` | 5 matches | $\\bar{G}_{\\text{H}, t}^{(5)} - \\bar{G}_{\\text{A}, t}^{(5)}$ | Short-term attacking output differential | Calculated strictly over prior 5 matches ($j \\in [1, 5]$). |
| `diff_goals_conceded_k5` | 5 matches | $\\bar{C}_{\\text{H}, t}^{(5)} - \\bar{C}_{\\text{A}, t}^{(5)}$ | Short-term defensive concession differential | Calculated strictly over prior 5 matches ($j \\in [1, 5]$). |
| `diff_ppg_k5` | 5 matches | $\\bar{P}_{\\text{H}, t}^{(5)} - \\bar{P}_{\\text{A}, t}^{(5)}$ | Recent overall competitive form differential ($W=3, D=1, L=0$) | Strictly historical match outcomes. |
| `diff_goal_diff_k10` | 10 matches | $\\overline{GD}_{\\text{H}, t}^{(10)} - \\overline{GD}_{\\text{A}, t}^{(10)}$ | Medium-term structural strength differential ($(GF - GA)/10$) | Requires $N \\ge 10$ prior matches for both clubs. |
| `diff_venue_ppg_k5` | 5 venue matches | $\\bar{P}_{\\text{H, home}, t}^{(5)} - \\bar{P}_{\\text{A, away}, t}^{(5)}$ | Venue-specific form (Home team at home vs Away team away) | Strictly uses prior home games for H and prior away games for A. |
| `diff_rest_days` | Inter-match | $\\Delta \\tau_{\\text{H}} - \\Delta \\tau_{\\text{A}}$ (days) | Relative schedule fatigue / recovery advantage | Computed from previous match kickoff timestamp. |
| `diff_goal_diff_k5` | 5 matches | $\\overline{GD}_{\\text{H}, t}^{(5)} - \\overline{GD}_{\\text{A}, t}^{(5)}$ | Short-term goal difference momentum differential | Calculated strictly over prior 5 matches. |
| `diff_clean_sheets_k10` | 10 matches | $\\overline{CS}_{\\text{H}, t}^{(10)} - \\overline{CS}_{\\text{A}, t}^{(10)}$ | Defensive solidity / shutout rate differential | Proportion of zero-conceded games in last 10. |
| `diff_win_rate_k5` | 5 matches | $\\overline{W}_{\\text{H}, t}^{(5)} - \\overline{W}_{\\text{A}, t}^{(5)}$ | Short-term win conversion differential | Proportion of wins in last 5 matches. |

---

## 3. Individual Team Form Metrics ($p = 18$)

| Feature Name | Subject Club | Window | Definition |
| :--- | :--- | :---: | :--- |
| `home_goals_scored_k5` | Home Team | 5 matches | Mean goals scored in previous 5 matches |
| `away_goals_scored_k5` | Away Team | 5 matches | Mean goals scored in previous 5 matches |
| `home_goals_conceded_k5` | Home Team | 5 matches | Mean goals conceded in previous 5 matches |
| `away_goals_conceded_k5` | Away Team | 5 matches | Mean goals conceded in previous 5 matches |
| `home_ppg_k5` | Home Team | 5 matches | Mean points per game in previous 5 matches |
| `away_ppg_k5` | Away Team | 5 matches | Mean points per game in previous 5 matches |
| `home_gd_k10` | Home Team | 10 matches | Mean goal difference per game in previous 10 matches |
| `away_gd_k10` | Away Team | 10 matches | Mean goal difference per game in previous 10 matches |
| `home_venue_ppg_k5` | Home Team | 5 home games | Mean points per game in previous 5 home games |
| `away_venue_ppg_k5` | Away Team | 5 away games | Mean points per game in previous 5 away games |
| `home_rest_days` | Home Team | Previous fixture | Fractional calendar days since previous match |
| `away_rest_days` | Away Team | Previous fixture | Fractional calendar days since previous match |
| `home_gd_k5` | Home Team | 5 matches | Mean goal difference per game in previous 5 matches |
| `away_gd_k5` | Away Team | 5 matches | Mean goal difference per game in previous 5 matches |
| `home_clean_sheets_k10` | Home Team | 10 matches | Proportion of clean sheets in previous 10 matches |
| `away_clean_sheets_k10` | Away Team | 10 matches | Proportion of clean sheets in previous 10 matches |
| `home_win_rate_k5` | Home Team | 5 matches | Proportion of wins in previous 5 matches |
| `away_win_rate_k5` | Away Team | 5 matches | Proportion of wins in previous 5 matches |

---

## 4. Metadata and Target Columns

| Column Name | Data Type | Values / Format | Description |
| :--- | :--- | :--- | :--- |
| `match_id` | String | `{season}_GW{matchday}_{home}_v_{away}` | Deterministic unique fixture identifier |
| `season` | String | `YYYY-YY` (`2019-20` to `2023-24`) | English Premier League season |
| `date` | String | `YYYY-MM-DD` | Official calendar match date |
| `kickoff_time` | String | `HH:MM` | Kickoff time (24-hour format) |
| `home_team` | String | Canonical club name | Normalized home club name (26 clubs) |
| `away_team` | String | Canonical club name | Normalized away club name (26 clubs) |
| `is_modelling_ready` | Boolean | `True` / `False` | Cold-start flag ($k \\ge 10$ overall, $k \\ge 5$ venue) |
| `home_goals` | Integer | $[0, \\infty)$ | Final full-time goals scored by home club |
| `away_goals` | Integer | $[0, \\infty)$ | Final full-time goals scored by away club |
| `target` | String | `HOME_WIN`, `DRAW`, `AWAY_WIN` | Categorical match outcome |
| `target_multinomial` | Integer | `2` (Home), `1` (Draw), `0` (Away) | Primary 3-class Softmax regression target |
| `target_binary` | Integer | `1` (Home Win), `0` (Non-Home Win) | Benchmark binary regression target |
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    logger.info("Saved feature dictionary markdown to %s", output_path)


def main():
    logger.info("======================================================================")
    logger.info("STARTING OPENFOOTBALL FEATURE ENGINEERING PIPELINE")
    logger.info("======================================================================")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    df_features = build_openfootball_feature_dataset()
    df_features.to_csv(OUTPUT_FEATURES_CSV, index=False)
    logger.info("Saved dataset_features.csv to %s", OUTPUT_FEATURES_CSV)

    generate_feature_dictionary_markdown(FEATURE_DICTIONARY_MD)
    logger.info("======================================================================")
    logger.info("OPENFOOTBALL FEATURE PIPELINE COMPLETE")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
