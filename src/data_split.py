"""Chronological Partitioning, Frozen Standardization, and Collinearity Diagnostics (Gate 4).

This module implements Quality Gate 4 (TSK-012, TSK-013, TSK-014):
1. Partitions the feature repository into strict chronological splits:
   - Training Partition (Seasons 2019/20 - 2021/22): exactly 1,002 modelling-ready matches.
   - Validation Partition (Season 2022/23): exactly 370 modelling-ready matches (held strictly for tuning lambda).
   - Test Partition (Season 2023/24): exactly 369 modelling-ready matches (locked until final evaluation).
2. Fits and serializes frozen standardizers and training class-frequency priors (data/processed/scalers.joblib).
3. Computes training-split empirical Variance Inflation Factors (VIF), correlation clusters,
   and design/Hessian condition numbers (reports/vif_summary.csv).
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Workspace paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = WORKSPACE_ROOT / "data" / "features" / "dataset_features.csv"
PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
REPORTS_DIR = WORKSPACE_ROOT / "reports"

# Legacy 16 candidate predictors (ADR-007 Option A)
LEGACY_PREDICTOR_COLS = [
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

# OpenFootball public-domain result-derived candidate predictors (ADR-013)
OPENFOOTBALL_PREDICTOR_COLS = [
    "diff_goals_scored_k5",
    "diff_goals_conceded_k5",
    "diff_ppg_k5",
    "diff_goal_diff_k10",
    "diff_venue_ppg_k5",
    "diff_rest_days",
    "diff_clean_sheets_k10",
    "diff_win_rate_k5",
]

PREDICTOR_COLS = OPENFOOTBALL_PREDICTOR_COLS

TRAIN_SEASONS = ["2019-20", "2019-2020", "2020-21", "2020-2021", "2021-22", "2021-2022"]
VAL_SEASONS = ["2022-23", "2022-2023"]
TEST_SEASONS = ["2023-24", "2023-2024"]

LEGACY_CLUSTER_MAP = {
    "diff_shots_k5": "Shot Volume & Dominance",
    "diff_shotsA_k5": "Shot Volume & Dominance",
    "diff_shot_dominance_k5": "Shot Volume & Dominance",
    "diff_sot_k5": "Shot Quality & Precision",
    "diff_sotA_k5": "Shot Quality & Precision",
    "diff_sot_ratio_k5": "Shot Quality & Precision",
    "diff_goal_diff_k5": "Form & Goal Difference",
    "diff_goal_diff_k10": "Form & Goal Difference",
    "diff_ppg_k5": "Form & Goal Difference",
    "diff_venue_ppg_k5": "Form & Goal Difference",
    "diff_corners_k5": "Territorial Set Pieces",
    "diff_cornersA_k5": "Territorial Set Pieces",
    "diff_clean_sheets_k10": "Defensive Solidity",
    "diff_fouls_k5": "Disciplinary & Aggression",
    "diff_yellows_k5": "Disciplinary & Aggression",
    "diff_rest_days": "Schedule Congestion",
}

OPENFOOTBALL_CLUSTER_MAP = {
    "diff_goals_scored_k5": "Attacking Threat & Output",
    "diff_goals_conceded_k5": "Defensive Concession",
    "diff_ppg_k5": "Recent Form & Momentum",
    "diff_goal_diff_k10": "Sustained Structural Quality",
    "diff_venue_ppg_k5": "Venue-Specific Home/Away Form",
    "diff_rest_days": "Schedule Congestion & Rest",
    "diff_clean_sheets_k10": "Defensive Solidity & Shutouts",
    "diff_win_rate_k5": "Win Conversion Rate",
}

FEATURE_CLUSTER_MAP = OPENFOOTBALL_CLUSTER_MAP


def get_active_predictors(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Determine active predictor list and cluster mapping based on columns present."""
    if "diff_goals_scored_k5" in df.columns:
        return OPENFOOTBALL_PREDICTOR_COLS, OPENFOOTBALL_CLUSTER_MAP
    return LEGACY_PREDICTOR_COLS, LEGACY_CLUSTER_MAP


def load_and_validate_partitions(features_path: Path = FEATURES_CSV) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features dataset and partition chronologically with strict validation."""
    global PREDICTOR_COLS, FEATURE_CLUSTER_MAP
    if not features_path.exists():
        raise FileNotFoundError(f"Features dataset not found: {features_path}")

    df = pd.read_csv(features_path)
    active_cols, cluster_map = get_active_predictors(df)
    PREDICTOR_COLS = active_cols
    FEATURE_CLUSTER_MAP = cluster_map

    # Filter strictly to modelling-ready fixtures (warm-up cold-start isolation)
    ready = df[df["is_modelling_ready"]].copy().reset_index(drop=True)

    if len(ready) != 1741:
        raise ValueError(f"Expected exactly 1,741 modelling-ready matches, got {len(ready)}")

    train_df = ready[ready["season"].isin(TRAIN_SEASONS)].copy().reset_index(drop=True)
    val_df = ready[ready["season"].isin(VAL_SEASONS)].copy().reset_index(drop=True)
    test_df = ready[ready["season"].isin(TEST_SEASONS)].copy().reset_index(drop=True)

    # Invariant assertions
    if len(train_df) != 1002:
        raise ValueError(f"Expected exactly 1,002 training matches, got {len(train_df)}")
    if len(val_df) != 370:
        raise ValueError(f"Expected exactly 370 validation matches, got {len(val_df)}")
    if len(test_df) != 369:
        raise ValueError(f"Expected exactly 369 test matches, got {len(test_df)}")

    # Check zero nulls across active predictors
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        null_counts = part[PREDICTOR_COLS].isnull().sum()
        if null_counts.sum() > 0:
            raise ValueError(f"Found missing values in {name} partition: {null_counts[null_counts > 0].to_dict()}")

    return train_df, val_df, test_df


def fit_and_freeze_standardizers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR
) -> dict:
    """Fit standardizer strictly on training partition and apply frozen to validation/test."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit standardizer strictly on training split
    scaler = StandardScaler()
    scaler.fit(train_df[PREDICTOR_COLS])

    # Compute training class frequencies (zero lookahead leakage baseline)
    train_class_counts = train_df["target_multinomial"].value_counts().to_dict()
    train_n = len(train_df)
    train_class_freqs = {
        0: float(train_class_counts.get(0, 0) / train_n),  # Away Win
        1: float(train_class_counts.get(1, 0) / train_n),  # Draw
        2: float(train_class_counts.get(2, 0) / train_n),  # Home Win
    }

    scaler_metadata = {
        "feature_names": PREDICTOR_COLS,
        "scaler": scaler,
        "train_means": dict(zip(PREDICTOR_COLS, scaler.mean_)),
        "train_stds": dict(zip(PREDICTOR_COLS, scaler.scale_)),
        "train_class_frequencies": train_class_freqs,
        "train_class_counts": {int(k): int(v) for k, v in train_class_counts.items()},
        "train_sample_size": int(train_n),
        "val_sample_size": int(len(val_df)),
        "test_sample_size": int(len(test_df)),
    }

    # Serialize frozen scaler metadata
    scalers_path = output_dir / "scalers.joblib"
    joblib.dump(scaler_metadata, scalers_path)
    print(f"Serialized frozen scaler metadata to: {scalers_path}")

    # Generate unscaled and scaled datasets
    def make_scaled_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        scaled_vals = scaler.transform(df_in[PREDICTOR_COLS])
        for idx, col in enumerate(PREDICTOR_COLS):
            df_out[col] = scaled_vals[:, idx]
        return df_out

    train_scaled = make_scaled_df(train_df)
    val_scaled = make_scaled_df(val_df)
    test_scaled = make_scaled_df(test_df)

    # Save processed splits
    train_df.to_csv(output_dir / "train_unscaled.csv", index=False)
    val_df.to_csv(output_dir / "val_unscaled.csv", index=False)
    test_df.to_csv(output_dir / "test_unscaled.csv", index=False)

    train_scaled.to_csv(output_dir / "train_data.csv", index=False)
    val_scaled.to_csv(output_dir / "val_data.csv", index=False)
    test_scaled.to_csv(output_dir / "test_data.csv", index=False)

    print(f"Saved processed partition datasets to: {output_dir}")
    return scaler_metadata


def compute_vif_and_collinearity_diagnostics(
    train_df: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR
) -> pd.DataFrame:
    """Compute empirical VIFs, correlation partners, and condition numbers on the training split."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    X = train_df[PREDICTOR_COLS].copy()

    # 1. Compute VIF for each predictor
    vif_records = []
    for col in PREDICTOR_COLS:
        other_cols = [c for c in PREDICTOR_COLS if c != col]
        y_col = X[col]
        X_other = X[other_cols]

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X_other, y_col)
        r_squared = float(reg.score(X_other, y_col))

        if r_squared >= 0.999999:
            vif = 999999.0
        else:
            vif = float(1.0 / (1.0 - r_squared))

        # Find top correlation partner in training set
        corrs = X.corr()[col].drop(col)
        abs_corrs = corrs.abs()
        top_partner = abs_corrs.idxmax()
        max_r = float(corrs[top_partner])

        vif_records.append({
            "predictor": col,
            "vif": round(vif, 2),
            "r_squared": round(r_squared, 4),
            "collinearity_cluster": FEATURE_CLUSTER_MAP.get(col, "General"),
            "top_correlate": top_partner,
            "max_pearson_r": round(max_r, 4),
        })

    vif_df = pd.DataFrame(vif_records)
    vif_df = vif_df.sort_values(by="vif", ascending=False).reset_index(drop=True)

    # 2. Compute Design Matrix and Hessian Condition Numbers
    X_scaled = StandardScaler().fit_transform(X)
    X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    # Design matrix condition number (ratio of max to min singular value)
    s_vals = np.linalg.svd(X_with_intercept, compute_uv=False)
    kappa_X = float(s_vals[0] / s_vals[-1])

    # Empirical Hessian under training base frequencies:
    # H_0 = (1/n) * X^T W X where W_c = p_c * (1 - p_c)
    # Average class variance across 3 classes:
    p_home = float(train_df["target_multinomial"].value_counts().get(2, 0) / len(train_df))
    p_draw = float(train_df["target_multinomial"].value_counts().get(1, 0) / len(train_df))
    p_away = float(train_df["target_multinomial"].value_counts().get(0, 0) / len(train_df))
    w_avg = (p_home * (1 - p_home) + p_draw * (1 - p_draw) + p_away * (1 - p_away)) / 3.0

    H_0 = (w_avg / len(X_with_intercept)) * (X_with_intercept.T @ X_with_intercept)
    eig_vals = np.linalg.eigvalsh(H_0)
    kappa_H0 = float(eig_vals[-1] / eig_vals[0])

    # Save VIF summary CSV
    vif_csv_path = reports_dir / "vif_summary.csv"
    vif_df.to_csv(vif_csv_path, index=False)
    print(f"Saved VIF summary to: {vif_csv_path}")

    # Save detailed JSON diagnostic
    diagnostics = {
        "train_sample_size": len(train_df),
        "condition_number_design_matrix_kappa_X": round(kappa_X, 2),
        "condition_number_hessian_kappa_H0": round(kappa_H0, 2),
        "vif_threshold_ge_5_count": int((vif_df["vif"] >= 5.0).sum()),
        "vif_threshold_ge_10_count": int((vif_df["vif"] >= 10.0).sum()),
        "max_vif_feature": str(vif_df.iloc[0]["predictor"]),
        "max_vif_value": float(vif_df.iloc[0]["vif"]),
        "vif_rankings": vif_df.to_dict(orient="records"),
    }
    diag_path = reports_dir / "vif_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved VIF diagnostics JSON to: {diag_path}")

    # Generate Markdown summary report
    p_count = len(vif_df)
    md_content = f"""# Training Partition Collinearity & VIF Diagnostics (Gate 4)

**Sample:** Training Partition (Seasons 2019/20 - 2021/22, $N = 1,002$ matches)  
**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 4  

---

## 1. Matrix Condition Numbers

- **Design Matrix Condition Number $\\kappa(\\mathbf{{X}})$:** `{round(kappa_X, 2)}`
- **Empirical Unpenalized Hessian Condition Number $\\kappa(\\mathbf{{H}}_0)$:** `{round(kappa_H0, 2)}`
- **Features with $\\text{{VIF}} \\ge 5.0$:** `{(vif_df['vif'] >= 5.0).sum()}` of {p_count} predictors
- **Features with $\\text{{VIF}} \\ge 10.0$:** `{(vif_df['vif'] >= 10.0).sum()}` of {p_count} predictors

---

## 2. Empirical Variance Inflation Factors

| Rank | Predictor | VIF ($1 / (1-R^2)$) | $R_j^2$ | Tactical Cluster | Top Correlate | Max Pearson $r$ |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
"""
    for i, r in vif_df.iterrows():
        md_content += f"| {i+1} | `{r['predictor']}` | **{r['vif']:.2f}** | {r['r_squared']:.4f} | {r['collinearity_cluster']} | `{r['top_correlate']}` | {r['max_pearson_r']:+.4f} |\n"

    md_path = reports_dir / "vif_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved VIF summary Markdown to: {md_path}")

    return vif_df


def run_pipeline():
    """Execute complete Gate 4 data splitting and diagnostic pipeline."""
    print("=" * 70)
    print("Starting Gate 4: Chronological Splitting & Preprocessing Pipeline")
    print("=" * 70)

    train_df, val_df, test_df = load_and_validate_partitions()
    print(f"Partitions verified: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    scaler_metadata = fit_and_freeze_standardizers(train_df, val_df, test_df)
    print(f"Training Class Frequencies: {scaler_metadata['train_class_frequencies']}")

    vif_df = compute_vif_and_collinearity_diagnostics(train_df)
    print(f"Top 3 VIF Features:\n{vif_df[['predictor', 'vif', 'top_correlate', 'max_pearson_r']].head(3)}")

    print("=" * 70)
    print("Gate 4 Pipeline Successfully Executed!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
