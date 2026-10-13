"""Feature Engineering and Pre-Match Predictor Pipeline (Gate 3).

Research Study: A Comparative Study of Maximum Likelihood, Ridge,
and Lasso Logistic Regression for Football Match Outcome Prediction.

Generates:
  1. data/features/dataset_features.csv (N=1,900 total, N=1,741 modelling-ready)
  2. data/audit/feature_availability_audit.csv (16 approved Option A predictors)
  3. data/audit/feature_missingness_audit.csv (missingness and warm-up breakdown)

Enforces:
  - Zero temporal lookahead leakage (j >= 1 for all rolling windows).
  - Explicit home-minus-away differential conventions (Delta X = X_Home - X_Away).
  - Robust division guards for shot accuracy (0.0) and dominance (0.5).
  - Strict chronological sequencing by kickoff timestamp.
  - Raw data immutability.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FOOTBALL_DATA_DIR = RAW_DIR / "football_data"
FBREF_DIR = RAW_DIR / "fbref"
FEATURES_DIR = DATA_DIR / "features"
AUDIT_DIR = DATA_DIR / "audit"

SEASONS = [
    {"label": "2019-2020", "fbref_code": 1920, "fd_code": "1920"},
    {"label": "2020-2021", "fbref_code": 2021, "fd_code": "2021"},
    {"label": "2021-2022", "fbref_code": 2122, "fd_code": "2122"},
    {"label": "2022-2023", "fbref_code": 2223, "fd_code": "2223"},
    {"label": "2023-2024", "fbref_code": 2324, "fd_code": "2324"},
]

# Canonical team mapping between Football-Data.co.uk and FBref
FD_TO_FBREF_TEAM_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham",
    "Watford": "Watford",
    "West Brom": "West Brom",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}

PREDICTOR_COLUMNS = [
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

METADATA_COLUMNS = [
    "match_id",
    "season",
    "date",
    "kickoff_time",
    "home_team",
    "away_team",
    "is_modelling_ready",
]

TARGET_COLUMNS = [
    "home_goals",
    "away_goals",
    "target_multinomial",
    "target_binary",
]


def load_and_reconcile_raw_data() -> pd.DataFrame:
    """Load raw seasonal Football-Data CSVs and FBref schedules, reconciling into a single sorted DataFrame.
    
    Returns:
        pd.DataFrame: 1,900 reconciled matches sorted strictly chronologically.
    """
    logger.info("Loading raw Football-Data and FBref schedules...")
    all_fd = []
    for s in SEASONS:
        label = s["label"]
        fd_path = FOOTBALL_DATA_DIR / f"premier_league_{label}.csv"
        if not fd_path.exists():
            raise FileNotFoundError(f"Missing required raw file: {fd_path}")
        df_season = pd.read_csv(fd_path)
        if len(df_season) != 380:
            raise ValueError(f"Expected 380 rows in {fd_path}, got {len(df_season)}")
        df_season["season_label"] = label
        all_fd.append(df_season)

    df_fd = pd.concat(all_fd, ignore_index=True)
    logger.info("Loaded %d matches from Football-Data.", len(df_fd))

    # Standardize team names
    df_fd["home_team"] = df_fd["HomeTeam"].map(FD_TO_FBREF_TEAM_MAP)
    df_fd["away_team"] = df_fd["AwayTeam"].map(FD_TO_FBREF_TEAM_MAP)
    if df_fd["home_team"].isnull().any() or df_fd["away_team"].isnull().any():
        unmapped_h = df_fd[df_fd["home_team"].isnull()]["HomeTeam"].unique()
        unmapped_a = df_fd[df_fd["away_team"].isnull()]["AwayTeam"].unique()
        raise ValueError(f"Unmapped teams detected: Home={unmapped_h}, Away={unmapped_a}")

    # Parse datetime strictly
    df_fd["datetime"] = pd.to_datetime(
        df_fd["Date"] + " " + df_fd["Time"], format="%d/%m/%Y %H:%M"
    )
    if df_fd["datetime"].isnull().any():
        raise ValueError("Detected unparseable or null datetime values in raw Football-Data.")

    # Load FBref schedule for auxiliary validation
    fb_path = FBREF_DIR / "fbref_premier_league_2019_2024_schedule.csv"
    if not fb_path.exists():
        raise FileNotFoundError(f"Missing FBref schedule: {fb_path}")
    df_fb = pd.read_csv(fb_path)

    season_map = {s["fbref_code"]: s["label"] for s in SEASONS}
    df_fb["season_label"] = df_fb["season"].map(season_map).fillna(df_fb["season"].astype(str))

    # Merge to bring in game_id, referee, and venue
    merged = pd.merge(
        df_fd,
        df_fb[["season_label", "home_team", "away_team", "game_id", "referee", "venue"]],
        on=["season_label", "home_team", "away_team"],
        how="inner",
    )
    if len(merged) != 1900:
        raise ValueError(f"Cross-source reconciliation failure: merged {len(merged)} rows, expected 1900.")

    # Sort strictly chronologically by datetime, using home_team and away_team for deterministic secondary tie-breaking
    merged = merged.sort_values(by=["datetime", "home_team", "away_team"]).reset_index(drop=True)

    # Deterministic match_id: EPL_{YYYY}_{YYYY}_{YYYYMMDD}_{HOM}_{AWY}
    merged["match_id"] = merged.apply(
        lambda r: f"EPL_{r['season_label'][:4]}_{r['season_label'][5:]}_{r['datetime'].strftime('%Y%m%d')}_{r['home_team'][:3].upper()}_{r['away_team'][:3].upper()}",
        axis=1,
    )

    logger.info("Reconciliation passed: exactly 1,900 fixtures chronologically sorted.")
    return merged


def compute_team_k5_features(hist: list) -> dict:
    """Compute 5-match rolling features for a team from completed match history.
    
    Args:
        hist: Chronological list of completed past match dicts (excluding current match).
        
    Returns:
        dict of rolling 5-match metrics, or NaNs if len(hist) < 5.
    """
    if len(hist) < 5:
        return {
            "sot": np.nan,
            "sotA": np.nan,
            "shots": np.nan,
            "shotsA": np.nan,
            "sot_ratio": np.nan,
            "shot_dominance": np.nan,
            "corners": np.nan,
            "cornersA": np.nan,
            "fouls": np.nan,
            "yellows": np.nan,
            "goal_diff": np.nan,
            "ppg": np.nan,
        }

    last5 = hist[-5:]
    sot = sum(m["sot"] for m in last5)
    sotA = sum(m["sotA"] for m in last5)
    shots = sum(m["shots"] for m in last5)
    shotsA = sum(m["shotsA"] for m in last5)
    corners = sum(m["corners"] for m in last5)
    cornersA = sum(m["cornersA"] for m in last5)
    fouls = sum(m["fouls"] for m in last5)
    yellows = sum(m["yellows"] for m in last5)
    gd = sum(m["goal_diff"] for m in last5)
    pts = sum(m["points"] for m in last5)

    # Robust division guards
    sot_ratio = sot / shots if shots > 0 else 0.0
    shot_dom = shots / (shots + shotsA) if (shots + shotsA) > 0 else 0.5

    return {
        "sot": sot / 5.0,
        "sotA": sotA / 5.0,
        "shots": shots / 5.0,
        "shotsA": shotsA / 5.0,
        "sot_ratio": sot_ratio,
        "shot_dominance": shot_dom,
        "corners": corners / 5.0,
        "cornersA": cornersA / 5.0,
        "fouls": fouls / 5.0,
        "yellows": yellows / 5.0,
        "goal_diff": gd / 5.0,
        "ppg": pts / 5.0,
    }


def compute_team_k10_features(hist: list) -> dict:
    """Compute 10-match rolling features for a team from completed match history.
    
    Args:
        hist: Chronological list of completed past match dicts (excluding current match).
        
    Returns:
        dict of rolling 10-match metrics, or NaNs if len(hist) < 10.
    """
    if len(hist) < 10:
        return {"goal_diff_k10": np.nan, "clean_sheets_k10": np.nan}

    last10 = hist[-10:]
    gd = sum(m["goal_diff"] for m in last10) / 10.0
    cs = sum(m["clean_sheet"] for m in last10) / 10.0
    return {"goal_diff_k10": gd, "clean_sheets_k10": cs}


def compute_venue_ppg(venue_hist: list) -> float:
    """Compute 5-match rolling venue points per game.
    
    Args:
        venue_hist: Chronological list of completed venue-specific match dicts.
        
    Returns:
        float: venue PPG, or NaN if len(venue_hist) < 5.
    """
    if len(venue_hist) < 5:
        return np.nan
    last5 = venue_hist[-5:]
    return sum(m["points"] for m in last5) / 5.0


def compute_rest_days(last_dt: pd.Timestamp | None, cur_dt: pd.Timestamp) -> float:
    """Compute inter-match rest days in continuous days.
    
    Args:
        last_dt: Kickoff timestamp of the team's previous completed match.
        cur_dt: Kickoff timestamp of the current match.
        
    Returns:
        float: rest days, or NaN if last_dt is None (first match).
    """
    if last_dt is None:
        return np.nan
    delta_seconds = (cur_dt - last_dt).total_seconds()
    return delta_seconds / 86400.0


def build_feature_dataset(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Build the pre-match feature matrix across all 1,900 matches strictly sequentially.
    
    Guarantees:
      - Current match t is NEVER included in historical statistics (strictly j >= 1).
      - State update happens ONLY after pre-match features are recorded for match t.
      - Home-minus-away convention enforced: Delta X = X_Home - X_Away.
      - Warm-up rows flagged via is_modelling_ready (k >= 10 overall, k >= 5 venue).
    """
    logger.info("Building pre-match feature matrix strictly sequentially...")

    # Initialize sequential state tracking
    all_teams = set(df_matches["home_team"]).union(df_matches["away_team"])
    team_history = {t: [] for t in all_teams}
    team_venue_history = {t: {"home": [], "away": []} for t in all_teams}
    team_last_dt = {t: None for t in all_teams}

    feature_rows = []

    for idx, row in df_matches.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        cur_dt = row["datetime"]

        h_hist = team_history[h]
        a_hist = team_history[a]
        h_venue = team_venue_history[h]["home"]
        a_venue = team_venue_history[a]["away"]

        # 1. Warm-up evaluation
        is_ready = (
            len(h_hist) >= 10
            and len(a_hist) >= 10
            and len(h_venue) >= 5
            and len(a_venue) >= 5
        )

        # 2. Compute pre-match features for each team
        h_k5 = compute_team_k5_features(h_hist)
        a_k5 = compute_team_k5_features(a_hist)
        h_k10 = compute_team_k10_features(h_hist)
        a_k10 = compute_team_k10_features(a_hist)
        h_ven = compute_venue_ppg(h_venue)
        a_ven = compute_venue_ppg(a_venue)
        h_rest = compute_rest_days(team_last_dt[h], cur_dt)
        a_rest = compute_rest_days(team_last_dt[a], cur_dt)

        # 3. Construct 16 home-minus-away differentials
        # Delta X = X_Home - X_Away
        diff_sot_k5 = h_k5["sot"] - a_k5["sot"]
        diff_sotA_k5 = h_k5["sotA"] - a_k5["sotA"]
        diff_shots_k5 = h_k5["shots"] - a_k5["shots"]
        diff_shotsA_k5 = h_k5["shotsA"] - a_k5["shotsA"]
        diff_sot_ratio_k5 = h_k5["sot_ratio"] - a_k5["sot_ratio"]
        diff_shot_dominance_k5 = h_k5["shot_dominance"] - a_k5["shot_dominance"]
        diff_corners_k5 = h_k5["corners"] - a_k5["corners"]
        diff_cornersA_k5 = h_k5["cornersA"] - a_k5["cornersA"]
        diff_fouls_k5 = h_k5["fouls"] - a_k5["fouls"]
        diff_yellows_k5 = h_k5["yellows"] - a_k5["yellows"]
        diff_goal_diff_k5 = h_k5["goal_diff"] - a_k5["goal_diff"]
        diff_goal_diff_k10 = h_k10["goal_diff_k10"] - a_k10["goal_diff_k10"]
        diff_clean_sheets_k10 = h_k10["clean_sheets_k10"] - a_k10["clean_sheets_k10"]
        diff_ppg_k5 = h_k5["ppg"] - a_k5["ppg"]
        diff_venue_ppg_k5 = h_ven - a_ven
        diff_rest_days = h_rest - a_rest

        # 4. Target variables (ground truth)
        hg = int(row["FTHG"])
        ag = int(row["FTAG"])
        target_multi = 2 if hg > ag else (1 if hg == ag else 0)
        target_bin = 1 if hg > ag else 0

        feature_record = {
            # Metadata
            "match_id": row["match_id"],
            "season": row["season_label"],
            "date": cur_dt.strftime("%Y-%m-%d"),
            "kickoff_time": cur_dt.strftime("%H:%M"),
            "home_team": h,
            "away_team": a,
            "is_modelling_ready": is_ready,
            # Targets
            "home_goals": hg,
            "away_goals": ag,
            "target_multinomial": target_multi,
            "target_binary": target_bin,
            # 16 Differential Predictors
            "diff_sot_k5": diff_sot_k5,
            "diff_sotA_k5": diff_sotA_k5,
            "diff_shots_k5": diff_shots_k5,
            "diff_shotsA_k5": diff_shotsA_k5,
            "diff_sot_ratio_k5": diff_sot_ratio_k5,
            "diff_shot_dominance_k5": diff_shot_dominance_k5,
            "diff_corners_k5": diff_corners_k5,
            "diff_cornersA_k5": diff_cornersA_k5,
            "diff_fouls_k5": diff_fouls_k5,
            "diff_yellows_k5": diff_yellows_k5,
            "diff_goal_diff_k5": diff_goal_diff_k5,
            "diff_goal_diff_k10": diff_goal_diff_k10,
            "diff_clean_sheets_k10": diff_clean_sheets_k10,
            "diff_ppg_k5": diff_ppg_k5,
            "diff_venue_ppg_k5": diff_venue_ppg_k5,
            "diff_rest_days": diff_rest_days,
        }
        feature_rows.append(feature_record)

        # 5. STATE UPDATE (Strictly AFTER pre-match feature calculation)
        hs = int(row["HS"])
        as_ = int(row["AS"])
        hst = int(row["HST"])
        ast = int(row["AST"])
        hc = int(row["HC"])
        ac = int(row["AC"])
        hf = int(row["HF"])
        af = int(row["AF"])
        hy = int(row["HY"])
        ay = int(row["AY"])

        h_match_stat = {
            "shots": hs,
            "shotsA": as_,
            "sot": hst,
            "sotA": ast,
            "corners": hc,
            "cornersA": ac,
            "fouls": hf,
            "yellows": hy,
            "goal_diff": hg - ag,
            "clean_sheet": 1 if ag == 0 else 0,
            "points": 3 if hg > ag else (1 if hg == ag else 0),
        }
        a_match_stat = {
            "shots": as_,
            "shotsA": hs,
            "sot": ast,
            "sotA": hst,
            "corners": ac,
            "cornersA": hc,
            "fouls": af,
            "yellows": ay,
            "goal_diff": ag - hg,
            "clean_sheet": 1 if hg == 0 else 0,
            "points": 3 if ag > hg else (1 if ag == hg else 0),
        }

        team_history[h].append(h_match_stat)
        team_history[a].append(a_match_stat)
        team_venue_history[h]["home"].append(h_match_stat)
        team_venue_history[a]["away"].append(a_match_stat)
        team_last_dt[h] = cur_dt
        team_last_dt[a] = cur_dt

    df_features = pd.DataFrame(feature_rows)
    return df_features


def generate_audits(df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate availability and missingness audit tables for the 16 approved predictors.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (availability_df, missingness_df)
    """
    logger.info("Generating feature audit tables...")

    # 1. Feature Availability Audit (Option A Specification)
    availability_records = [
        {
            "feature_id": "X1",
            "feature_name": "diff_sot_k5",
            "tactical_concept": "Attacking Target Accuracy Volume",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "HST (Home Shots on Target), AST (Away Shots on Target)",
            "calculation": "Mean(HST_last5)_Home - Mean(AST_last5)_Away",
            "unit": "Shots on target / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X2",
            "feature_name": "diff_sotA_k5",
            "tactical_concept": "Defensive Concession Target Threat",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "AST (Conceded by Home), HST (Conceded by Away)",
            "calculation": "Mean(AST_last5)_Home - Mean(HST_last5)_Away",
            "unit": "Shots on target conceded / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X3",
            "feature_name": "diff_shots_k5",
            "tactical_concept": "Total Attacking Shot Attempt Volume",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "HS (Home Shots), AS (Away Shots)",
            "calculation": "Mean(HS_last5)_Home - Mean(AS_last5)_Away",
            "unit": "Shots attempted / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X4",
            "feature_name": "diff_shotsA_k5",
            "tactical_concept": "Total Conceded Shot Attempt Volume",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "AS (Conceded by Home), HS (Conceded by Away)",
            "calculation": "Mean(AS_last5)_Home - Mean(HS_last5)_Away",
            "unit": "Shots conceded / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X5",
            "feature_name": "diff_sot_ratio_k5",
            "tactical_concept": "Offensive Shot Precision Efficiency",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "HST, HS (Home) / AST, AS (Away)",
            "calculation": "[Sum(SoT)/Sum(Shots)]_Home - [Sum(SoT)/Sum(Shots)]_Away",
            "unit": "Ratio [0.0, 1.0]; 0.0 if shots == 0",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Guarded at 0.0; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X6",
            "feature_name": "diff_shot_dominance_k5",
            "tactical_concept": "Territorial Shot Share Dominance",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "HS, AS",
            "calculation": "[Shots/(Shots+ShotsA)]_Home - [Shots/(Shots+ShotsA)]_Away",
            "unit": "Share [0.0, 1.0]; 0.5 baseline if total shots == 0",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Guarded at 0.5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X7",
            "feature_name": "diff_corners_k5",
            "tactical_concept": "Deep Final Third Territorial Pressure",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "HC (Home Corners), AC (Away Corners)",
            "calculation": "Mean(HC_last5)_Home - Mean(AC_last5)_Away",
            "unit": "Corner kicks won / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X8",
            "feature_name": "diff_cornersA_k5",
            "tactical_concept": "Conceded Final Third Pressure",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "AC (Conceded by Home), HC (Conceded by Away)",
            "calculation": "Mean(AC_last5)_Home - Mean(HC_last5)_Away",
            "unit": "Corner kicks conceded / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X9",
            "feature_name": "diff_fouls_k5",
            "tactical_concept": "Pressing Aggression & Disruption",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "HF (Home Fouls), AF (Away Fouls)",
            "calculation": "Mean(HF_last5)_Home - Mean(AF_last5)_Away",
            "unit": "Fouls committed / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X10",
            "feature_name": "diff_yellows_k5",
            "tactical_concept": "Disciplinary Caution Risk",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "HY (Home Yellows), AY (Away Yellows)",
            "calculation": "Mean(HY_last5)_Home - Mean(AY_last5)_Away",
            "unit": "Yellow cards received / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X11",
            "feature_name": "diff_goal_diff_k5",
            "tactical_concept": "Short-Term Goal Difference Form",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "FTHG, FTAG",
            "calculation": "Mean(GF - GA)_Home_k5 - Mean(GF - GA)_Away_k5",
            "unit": "Goal differential / match",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X12",
            "feature_name": "diff_goal_diff_k10",
            "tactical_concept": "Sustained Structural Goal Dominance",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "FTHG, FTAG",
            "calculation": "Mean(GF - GA)_Home_k10 - Mean(GF - GA)_Away_k10",
            "unit": "Goal differential / match",
            "window": "k = 10",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 10; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X13",
            "feature_name": "diff_clean_sheets_k10",
            "tactical_concept": "Sustained Defensive Solidity Rate",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "FTHG, FTAG (GA == 0 indicator)",
            "calculation": "Rate(GA == 0)_Home_k10 - Rate(GA == 0)_Away_k10",
            "unit": "Clean sheet proportion [0.0, 1.0]",
            "window": "k = 10",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 10; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X14",
            "feature_name": "diff_ppg_k5",
            "tactical_concept": "Recent Psychological Points Momentum",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "FTR (Full Time Result: 3 for W, 1 for D, 0 for L)",
            "calculation": "Mean(Points)_Home_k5 - Mean(Points)_Away_k5",
            "unit": "Points / match [0.0, 3.0]",
            "window": "k = 5",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X15",
            "feature_name": "diff_venue_ppg_k5",
            "tactical_concept": "Symmetrical Venue-Specific Momentum",
            "raw_source_file": "Football-Data.co.uk (E0.csv)",
            "source_columns": "FTR by venue (Home matches for H, Away matches for A)",
            "calculation": "Mean(HomePoints)_Home_k5 - Mean(AwayPoints)_Away_k5",
            "unit": "Venue points / match [-3.0, +3.0]",
            "window": "k = 5 venue games",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if k < 5 venue games; non-null for all 1,741 ready fixtures",
        },
        {
            "feature_id": "X16",
            "feature_name": "diff_rest_days",
            "tactical_concept": "Inter-Match Physical Congestion Differential",
            "raw_source_file": "Football-Data / FBref Schedule",
            "source_columns": "Date, Time (Kickoff timestamps)",
            "calculation": "RestDays_Home - RestDays_Away (Delta tau / 86400)",
            "unit": "Continuous days",
            "window": "Continuous inter-match interval",
            "availability_status": "100% Available (0 raw nulls)",
            "missing_data_policy": "Excluded if first fixture; non-null for all 1,741 ready fixtures",
        },
    ]
    df_avail = pd.DataFrame(availability_records)

    # 2. Feature Missingness Audit by Season and Warm-up Status
    missingness_records = []
    for s in df_features["season"].unique():
        s_df = df_features[df_features["season"] == s]
        ready_df = s_df[s_df["is_modelling_ready"]]
        warmup_df = s_df[~s_df["is_modelling_ready"]]

        for feat in PREDICTOR_COLUMNS:
            missingness_records.append({
                "season": s,
                "feature_name": feat,
                "total_fixtures": len(s_df),
                "modelling_ready_count": len(ready_df),
                "warmup_count": len(warmup_df),
                "ready_missing_count": ready_df[feat].isnull().sum(),
                "ready_missing_pct": round(ready_df[feat].isnull().mean() * 100, 2),
                "warmup_missing_count": warmup_df[feat].isnull().sum(),
                "warmup_missing_pct": round(warmup_df[feat].isnull().mean() * 100, 2),
            })
    df_miss = pd.DataFrame(missingness_records)

    return df_avail, df_miss


def main():
    """Main execution entry point for feature engineering pipeline."""
    logger.info("=======================================================")
    logger.info("   Executing Gate 3: Feature Engineering Pipeline      ")
    logger.info("=======================================================")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and reconcile raw fixtures
    df_matches = load_and_reconcile_raw_data()

    # 2. Build sequential pre-match feature matrix
    df_features = build_feature_dataset(df_matches)

    # 3. Validate schema contracts
    expected_cols = METADATA_COLUMNS + TARGET_COLUMNS + PREDICTOR_COLUMNS
    actual_cols = list(df_features.columns)
    assert set(expected_cols) == set(actual_cols), f"Column mismatch: expected {expected_cols}, got {actual_cols}"

    # 4. Verify modelling-ready partition properties
    ready_df = df_features[df_features["is_modelling_ready"]]
    warmup_df = df_features[~df_features["is_modelling_ready"]]

    logger.info("Fixture summary:")
    logger.info("  Total matches: %d", len(df_features))
    logger.info("  Modelling-ready matches: %d (%.2f%%)", len(ready_df), len(ready_df) / len(df_features) * 100)
    logger.info("  Warm-up matches: %d (%.2f%%)", len(warmup_df), len(warmup_df) / len(df_features) * 100)

    # Assert 0 nulls across all 16 predictors in modelling-ready partition
    ready_nulls = ready_df[PREDICTOR_COLUMNS].isnull().sum()
    if ready_nulls.any():
        logger.error("Non-zero null count in modelling-ready partition:\n%s", ready_nulls[ready_nulls > 0])
        raise AssertionError("Modelling-ready partition contains null values!")
    logger.info("Modelling-ready invariant verified: exactly 0 nulls across all 16 predictors.")

    # Assert all values are finite
    is_finite = np.isfinite(ready_df[PREDICTOR_COLUMNS].values).all()
    assert is_finite, "Modelling-ready partition contains non-finite values (Inf / -Inf)!"
    logger.info("Modelling-ready invariant verified: all feature values are strictly finite.")

    # 5. Save dataset_features.csv
    out_features_path = FEATURES_DIR / "dataset_features.csv"
    df_features.to_csv(out_features_path, index=False)
    logger.info("Saved dataset_features.csv to %s (%d rows, %d cols).", out_features_path, len(df_features), len(df_features.columns))

    # 6. Generate and save audits
    df_avail, df_miss = generate_audits(df_features)
    out_avail_path = AUDIT_DIR / "feature_availability_audit.csv"
    out_miss_path = AUDIT_DIR / "feature_missingness_audit.csv"
    df_avail.to_csv(out_avail_path, index=False)
    df_miss.to_csv(out_miss_path, index=False)
    logger.info("Saved feature_availability_audit.csv to %s.", out_avail_path)
    logger.info("Saved feature_missingness_audit.csv to %s.", out_miss_path)

    logger.info("=======================================================")
    logger.info("   Gate 3 Feature Engineering Pipeline Complete       ")
    logger.info("=======================================================")


if __name__ == "__main__":
    main()
