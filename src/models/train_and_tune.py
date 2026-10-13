"""Hyperparameter Tuning & Regularization Paths on Validation Season 2022/23 (Gate 5).

This script implements TSK-018:
1. Loads standardized training partition (N = 1,002) and validation partition (N = 370).
2. Fits Unregularized MLE on training set, evaluating baseline validation loss.
3. Evaluates Naive Training-Split Class-Frequency Prior and Market-Implied Odds benchmarks.
4. Executes a 20-point logarithmic grid search over lambda in [10^-4, 10^4] for Ridge and Lasso:
   - Records Multi-class Log-Loss, Multi-class Brier Score, Accuracy, and Macro-F1.
   - Traces Ridge effective degrees of freedom df(lambda).
   - Traces Lasso variable sparsity (n_retained) and the coordinate shrinkage path.
5. Identifies optimal regularization strengths (lambda_Ridge*, lambda_Lasso*).
6. Serializes tuned models to data/processed/fitted_models.joblib.
7. Generates publication figures:
   - reports/figures/fig6_hyperparameter_tuning.png
   - reports/figures/lasso_shrinkage_path.png
8. Exports reports/hyperparameter_tuning.csv and reports/tuning_summary.md.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, f1_score

# Relative imports from src.models
import sys
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.models.mle_model import MLEMultinomialLogistic
from src.models.ridge_model import RidgeLogisticRegression
from src.models.lasso_model import LassoLogisticRegression

# Paths
PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"
RAW_FD_DIR = WORKSPACE_ROOT / "data" / "raw" / "football_data"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Multi-class Brier Score: (1/n) * sum_i sum_c (p_ic - y_ic)^2."""
    n_samples = len(y_true)
    y_onehot = np.zeros((n_samples, 3))
    for i, label in enumerate(y_true):
        y_onehot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def get_market_odds_validation_benchmarks(val_df: pd.DataFrame) -> dict:
    """Extract bookmaker market odds for Season 2022/23 validation fixtures."""
    fd_csv = RAW_FD_DIR / "premier_league_2022-2023.csv"
    if not fd_csv.exists():
        return {}

    fd = pd.read_csv(fd_csv)
    # Map teams to FBref canonical names
    fd["home_fbref"] = fd["HomeTeam"].map(FD_TO_FBREF_TEAM_MAP)
    fd["away_fbref"] = fd["AwayTeam"].map(FD_TO_FBREF_TEAM_MAP)

    merged = pd.merge(
        val_df[["date", "home_team", "away_team", "target_multinomial"]],
        fd[["home_fbref", "away_fbref", "AvgH", "AvgD", "AvgA", "B365H", "B365D", "B365A"]],
        left_on=["home_team", "away_team"],
        right_on=["home_fbref", "away_fbref"],
        how="inner"
    )

    if len(merged) != len(val_df):
        print(f"Warning: Market odds matched {len(merged)} / {len(val_df)} validation fixtures.")
        return {}

    y_val = merged["target_multinomial"].values

    # 1. Market Average Odds
    inv_avg_h = 1.0 / merged["AvgH"].values
    inv_avg_d = 1.0 / merged["AvgD"].values
    inv_avg_a = 1.0 / merged["AvgA"].values
    sum_inv_avg = inv_avg_h + inv_avg_d + inv_avg_a

    # Order: 0: Away, 1: Draw, 2: Home
    prob_avg = np.column_stack([inv_avg_a / sum_inv_avg, inv_avg_d / sum_inv_avg, inv_avg_h / sum_inv_avg])
    ll_avg = float(log_loss(y_val, prob_avg))
    bs_avg = multiclass_brier_score(y_val, prob_avg)
    acc_avg = float(accuracy_score(y_val, np.argmax(prob_avg, axis=1)))
    f1_avg = float(f1_score(y_val, np.argmax(prob_avg, axis=1), average="macro"))

    # 2. Bet365 Odds
    inv_b_h = 1.0 / merged["B365H"].values
    inv_b_d = 1.0 / merged["B365D"].values
    inv_b_a = 1.0 / merged["B365A"].values
    sum_inv_b = inv_b_h + inv_b_d + inv_b_a

    prob_b365 = np.column_stack([inv_b_a / sum_inv_b, inv_b_d / sum_inv_b, inv_b_h / sum_inv_b])
    ll_b365 = float(log_loss(y_val, prob_b365))
    bs_b365 = multiclass_brier_score(y_val, prob_b365)
    acc_b365 = float(accuracy_score(y_val, np.argmax(prob_b365, axis=1)))
    f1_b365 = float(f1_score(y_val, np.argmax(prob_b365, axis=1), average="macro"))

    return {
        "market_avg": {"log_loss": ll_avg, "brier_score": bs_avg, "accuracy": acc_avg, "macro_f1": f1_avg},
        "bet365": {"log_loss": ll_b365, "brier_score": bs_b365, "accuracy": acc_b365, "macro_f1": f1_b365},
    }


def run_tuning():
    print("=" * 70)
    print("Executing Gate 5: Model Calibration and Hyperparameter Tuning")
    print("=" * 70)

    # 1. Load data partitions
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")

    X_train = train_df[PREDICTOR_COLS]
    y_train = train_df["target_multinomial"].values
    X_val = val_df[PREDICTOR_COLS]
    y_val = val_df["target_multinomial"].values

    n_train = len(X_train)
    n_val = len(X_val)
    print(f"Loaded partitions: Training N={n_train}, Validation N={n_val}")

    # 2. Baseline Benchmark 1: Training Class-Frequency Prior
    scalers_meta = joblib.load(PROCESSED_DIR / "scalers.joblib")
    train_freqs = scalers_meta["train_class_frequencies"]
    prior_prob = np.array([train_freqs[0], train_freqs[1], train_freqs[2]])
    val_prior_probs = np.tile(prior_prob, (n_val, 1))

    prior_ll = float(log_loss(y_val, val_prior_probs))
    prior_bs = multiclass_brier_score(y_val, val_prior_probs)
    prior_preds = np.argmax(val_prior_probs, axis=1)
    prior_acc = float(accuracy_score(y_val, prior_preds))
    prior_f1 = float(f1_score(y_val, prior_preds, average="macro"))

    print(f"\n[Baseline 1: Training Frequency Prior]")
    print(f"  Validation Log-Loss:    {prior_ll:.4f}")
    print(f"  Validation Brier Score: {prior_bs:.4f}")
    print(f"  Validation Accuracy:    {prior_acc:.4f}")
    print(f"  Validation Macro-F1:    {prior_f1:.4f}")

    # 3. Baseline Benchmark 2: Market Odds Consensus
    odds_benchmarks = get_market_odds_validation_benchmarks(val_df)
    if odds_benchmarks:
        mkt_ll = odds_benchmarks["market_avg"]["log_loss"]
        mkt_bs = odds_benchmarks["market_avg"]["brier_score"]
        mkt_acc = odds_benchmarks["market_avg"]["accuracy"]
        print(f"\n[Baseline 2: Market Average Odds Benchmark]")
        print(f"  Validation Log-Loss:    {mkt_ll:.4f}")
        print(f"  Validation Brier Score: {mkt_bs:.4f}")
        print(f"  Validation Accuracy:    {mkt_acc:.4f}")
    else:
        mkt_ll, mkt_bs, mkt_acc = np.nan, np.nan, np.nan

    # 4. Model 1: Unregularized MLE
    print(f"\nFitting Model 1: Unregularized MLE (lambda = 0, penalty=None)...")
    mle = MLEMultinomialLogistic(tol=1e-5, max_iter=5000, fit_intercept=True)
    mle.fit(X_train, y_train)

    mle_probs = mle.predict_proba(X_val)
    mle_preds = mle.predict(X_val)

    mle_ll = float(log_loss(y_val, mle_probs))
    mle_bs = multiclass_brier_score(y_val, mle_probs)
    mle_acc = float(accuracy_score(y_val, mle_preds))
    mle_f1 = float(f1_score(y_val, mle_preds, average="macro"))

    print(f"  Validation Log-Loss:    {mle_ll:.4f}")
    print(f"  Validation Brier Score: {mle_bs:.4f}")
    print(f"  Validation Accuracy:    {mle_acc:.4f}")
    print(f"  Validation Macro-F1:    {mle_f1:.4f}")

    # 5. Grid Search over 20-Point Logarithmic Grid: lambda in [10^-4, 10^4]
    lambdas = np.logspace(-4, 4, 20)
    tuning_records = []
    lasso_coef_paths = {col: [] for col in PREDICTOR_COLS}

    print(f"\nEvaluating Regularization Grid (20 lambda values from 10^-4 to 10^4)...")

    for idx, lmbda in enumerate(lambdas):
        # A. Ridge Logistic Regression
        ridge = RidgeLogisticRegression(lambda_val=lmbda, n_train=n_train, tol=1e-5, max_iter=5000)
        ridge.fit(X_train, y_train)
        ridge_probs = ridge.predict_proba(X_val)
        ridge_preds = ridge.predict(X_val)

        r_ll = float(log_loss(y_val, ridge_probs))
        r_bs = multiclass_brier_score(y_val, ridge_probs)
        r_acc = float(accuracy_score(y_val, ridge_preds))
        r_f1 = float(f1_score(y_val, ridge_preds, average="macro"))
        r_df = float(ridge.df_lambda_)

        # B. Lasso Logistic Regression
        lasso = LassoLogisticRegression(lambda_val=lmbda, n_train=n_train, tol=1e-4, max_iter=5000)
        lasso.fit(X_train, y_train)
        lasso_probs = lasso.predict_proba(X_val)
        lasso_preds = lasso.predict(X_val)

        l_ll = float(log_loss(y_val, lasso_probs))
        l_bs = multiclass_brier_score(y_val, lasso_probs)
        l_acc = float(accuracy_score(y_val, lasso_preds))
        l_f1 = float(f1_score(y_val, lasso_preds, average="macro"))
        l_sparsity = lasso.get_sparsity()

        # Track L1 norm of each predictor across the 3 classes for shrinkage path
        coef_matrix = lasso.model.coef_  # Shape (3, p)
        for c_idx, col in enumerate(PREDICTOR_COLS):
            # Euclidean norm across the 3 classes for this feature
            feature_norm = float(np.linalg.norm(coef_matrix[:, c_idx]))
            lasso_coef_paths[col].append(feature_norm)

        tuning_records.append({
            "step": idx + 1,
            "lambda": float(lmbda),
            "log10_lambda": float(np.log10(lmbda)),
            "C_ridge": float(ridge.C),
            "C_lasso": float(lasso.C),
            "ridge_log_loss": r_ll,
            "ridge_brier_score": r_bs,
            "ridge_accuracy": r_acc,
            "ridge_macro_f1": r_f1,
            "ridge_df": round(r_df, 2),
            "lasso_log_loss": l_ll,
            "lasso_brier_score": l_bs,
            "lasso_accuracy": l_acc,
            "lasso_macro_f1": l_f1,
            "lasso_retained_count": l_sparsity["n_retained"],
            "lasso_retained_pct": l_sparsity["pct_retained"],
            "lasso_retained_features": ";".join(l_sparsity["retained_features"]),
        })

        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  Step {idx+1:2d}/20 | lambda={lmbda:9.4f} | Ridge LL={r_ll:.4f} (df={r_df:4.1f}) | Lasso LL={l_ll:.4f} (ret={l_sparsity['n_retained']:2d}/16)")

    tuning_df = pd.DataFrame(tuning_records)
    tuning_csv = REPORTS_DIR / "hyperparameter_tuning.csv"
    tuning_df.to_csv(tuning_csv, index=False)
    print(f"\nSaved tuning grid records to: {tuning_csv}")

    # 6. Identify Optimal Hyperparameters (Minimizing Validation Log-Loss)
    best_ridge_row = tuning_df.loc[tuning_df["ridge_log_loss"].idxmin()]
    best_lasso_row = tuning_df.loc[tuning_df["lasso_log_loss"].idxmin()]

    lambda_ridge_opt = float(best_ridge_row["lambda"])
    lambda_lasso_opt = float(best_lasso_row["lambda"])

    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY (Season 2022/23 Validation Set)")
    print("=" * 70)
    print(f"Baseline MLE (lambda=0):")
    print(f"  Log-Loss = {mle_ll:.4f} | Brier = {mle_bs:.4f} | Accuracy = {mle_acc:.4f}")
    print("-" * 70)
    print(f"Optimal Ridge (L2):")
    print(f"  Optimal lambda* = {lambda_ridge_opt:.4f} (log10 = {best_ridge_row['log10_lambda']:.2f})")
    print(f"  Corresponding C = {best_ridge_row['C_ridge']:.6f}")
    print(f"  Validation Log-Loss = {best_ridge_row['ridge_log_loss']:.4f} (delta vs MLE: {mle_ll - best_ridge_row['ridge_log_loss']:+.4f})")
    print(f"  Validation Brier    = {best_ridge_row['ridge_brier_score']:.4f}")
    print(f"  Effective df(lambda)= {best_ridge_row['ridge_df']:.2f} / 19 parameters")
    print("-" * 70)
    print(f"Optimal Lasso (L1):")
    print(f"  Optimal lambda* = {lambda_lasso_opt:.4f} (log10 = {best_lasso_row['log10_lambda']:.2f})")
    print(f"  Corresponding C = {best_lasso_row['C_lasso']:.6f}")
    print(f"  Validation Log-Loss = {best_lasso_row['lasso_log_loss']:.4f} (delta vs MLE: {mle_ll - best_lasso_row['lasso_log_loss']:+.4f})")
    print(f"  Validation Brier    = {best_lasso_row['lasso_brier_score']:.4f}")
    print(f"  Features Retained   = {int(best_lasso_row['lasso_retained_count'])} of {len(PREDICTOR_COLS)} ({best_lasso_row['lasso_retained_pct']:.1f}%)")
    print(f"  Retained Feature Set: {best_lasso_row['lasso_retained_features']}")
    print("=" * 70)

    # 7. Refit Best Models on Training Split and Serialize
    print("\nRefitting optimal models on Training partition...")
    best_ridge_model = RidgeLogisticRegression(lambda_val=lambda_ridge_opt, n_train=n_train, tol=1e-5, max_iter=5000)
    best_ridge_model.fit(X_train, y_train)

    best_lasso_model = LassoLogisticRegression(lambda_val=lambda_lasso_opt, n_train=n_train, tol=1e-4, max_iter=5000)
    best_lasso_model.fit(X_train, y_train)

    fitted_models_bundle = {
        "mle_model": mle,
        "ridge_best": best_ridge_model,
        "lasso_best": best_lasso_model,
        "lambda_ridge_opt": lambda_ridge_opt,
        "lambda_lasso_opt": lambda_lasso_opt,
        "best_ridge_row": best_ridge_row.to_dict(),
        "best_lasso_row": best_lasso_row.to_dict(),
        "mle_validation": {"log_loss": mle_ll, "brier_score": mle_bs, "accuracy": mle_acc, "macro_f1": mle_f1},
        "prior_validation": {"log_loss": prior_ll, "brier_score": prior_bs, "accuracy": prior_acc, "macro_f1": prior_f1},
        "market_validation": odds_benchmarks,
        "predictor_cols": PREDICTOR_COLS,
    }

    models_joblib_path = PROCESSED_DIR / "fitted_models.joblib"
    joblib.dump(fitted_models_bundle, models_joblib_path)
    print(f"Serialized fitted models bundle to: {models_joblib_path}")

    # 8. Publication Figures
    print("\nGenerating Figure 6: Hyperparameter Tuning Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # A. Log-Loss Curve
    ax1.plot(tuning_df["log10_lambda"], tuning_df["ridge_log_loss"], "o-", color="#1e3a8a", linewidth=2, label="Ridge (L2)")
    ax1.plot(tuning_df["log10_lambda"], tuning_df["lasso_log_loss"], "s--", color="#b91c1c", linewidth=2, label="Lasso (L1)")
    ax1.axhline(mle_ll, color="#64748b", linestyle=":", linewidth=1.5, label=f"MLE Baseline ({mle_ll:.4f})")
    ax1.axhline(prior_ll, color="#0f172a", linestyle="-.", linewidth=1.2, label=f"Frequency Prior ({prior_ll:.4f})")
    if not np.isnan(mkt_ll):
        ax1.axhline(mkt_ll, color="#059669", linestyle="-", linewidth=1.2, label=f"Market Odds Avg ({mkt_ll:.4f})")

    ax1.axvline(best_ridge_row["log10_lambda"], color="#1e3a8a", linestyle=":", alpha=0.7)
    ax1.axvline(best_lasso_row["log10_lambda"], color="#b91c1c", linestyle=":", alpha=0.7)

    ax1.set_xlabel(r"Regularization Strength $\log_{10}(\lambda)$", fontweight="semibold")
    ax1.set_ylabel("Validation Multi-Class Log-Loss (nats)", fontweight="semibold")
    ax1.set_title("A. Validation Log-Loss vs. Penalty Strength", pad=12)
    ax1.legend(frameon=True, edgecolor="#cbd5e1", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # B. Sparsity & Degrees of Freedom
    p_tot = len(PREDICTOR_COLS)
    ax2.plot(tuning_df["log10_lambda"], tuning_df["ridge_df"], "o-", color="#1e3a8a", linewidth=2, label=f"Ridge df(λ) (Total={3*p_tot+1})")
    ax2.plot(tuning_df["log10_lambda"], tuning_df["lasso_retained_count"], "s--", color="#b91c1c", linewidth=2, label=f"Lasso Retained Features (Max={p_tot})")
    ax2.axhline(p_tot // 2, color="#dc2626", linestyle=":", linewidth=1.2, label=f"H3 Parsimony Threshold (<={p_tot//2} features)")
    ax2.axvline(best_lasso_row["log10_lambda"], color="#b91c1c", linestyle=":", alpha=0.7)

    ax2.set_xlabel(r"Regularization Strength $\log_{10}(\lambda)$", fontweight="semibold")
    ax2.set_ylabel("Effective Parameters / Features Retained", fontweight="semibold")
    ax2.set_title("B. Parameter Complexity & Feature Selection", pad=12)
    ax2.legend(frameon=True, edgecolor="#cbd5e1", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig6_path = FIGURES_DIR / "fig6_hyperparameter_tuning.png"
    fig.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig6_path}")

    # C. Lasso Shrinkage Path Figure
    print("Generating Figure: Lasso Shrinkage Paths...")
    fig_path, ax_path = plt.subplots(figsize=(11, 7))

    for col in PREDICTOR_COLS:
        ax_path.plot(tuning_df["log10_lambda"], lasso_coef_paths[col], marker="o", markersize=3, label=col)

    ax_path.axvline(best_lasso_row["log10_lambda"], color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Optimal $\lambda^*$ ({lambda_lasso_opt:.4f})")
    ax_path.set_xlabel(r"Regularization Penalty $\log_{10}(\lambda)$", fontweight="semibold")
    ax_path.set_ylabel(r"Coefficient Norm Across Classes $\|\boldsymbol{\beta}_j\|_2$", fontweight="semibold")
    ax_path.set_title("Multinomial Lasso Regularization Path (Order of Variable Elimination)", pad=15)
    ax_path.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, edgecolor="#cbd5e1", fontsize=8.5)
    ax_path.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    lasso_fig_path = FIGURES_DIR / "lasso_shrinkage_path.png"
    fig_path.savefig(lasso_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig_path)
    print(f"Saved: {lasso_fig_path}")

    # 9. Write Comprehensive Markdown Summary
    md_content = f"""# Hyperparameter Tuning & Model Validation Report (Gate 5)

**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 5  
**Validation Partition:** Season 2022/23 ($N = 370$ matches)  
**Training Partition:** Seasons 2019/20 - 2021/22 ($N = 1,002$ matches)  

---

## 1. Validation Performance Benchmark Summary

| Model / Benchmark | Penalty $\\lambda$ | Sklearn $C$ | Validation Log-Loss | Validation Brier | Validation Accuracy | Notes / Sparsity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Frequency Prior** | — | — | `{prior_ll:.4f}` | `{prior_bs:.4f}` | `{prior_acc:.4f}` | Static training marginal rates |
| **Market Average Odds** | — | — | `{mkt_ll:.4f}` | `{mkt_bs:.4f}` | `{mkt_acc:.4f}` | Consensus betting benchmark |
| **Unregularized MLE** | $0$ | $\\infty$ | `{mle_ll:.4f}` | `{mle_bs:.4f}` | `{mle_acc:.4f}` | Full unconstrained model |
| **Ridge ($L_2$) Optimal** | **`{lambda_ridge_opt:.4f}`** | **`{best_ridge_row['C_ridge']:.6f}`** | **`{best_ridge_row['ridge_log_loss']:.4f}`** | **`{best_ridge_row['ridge_brier_score']:.4f}`** | **`{best_ridge_row['ridge_accuracy']:.4f}`** | $\\text{{df}}(\\lambda) = {best_ridge_row['ridge_df']:.2f}$ |
| **Lasso ($L_1$) Optimal** | **`{lambda_lasso_opt:.4f}`** | **`{best_lasso_row['C_lasso']:.6f}`** | **`{best_lasso_row['lasso_log_loss']:.4f}`** | **`{best_lasso_row['lasso_brier_score']:.4f}`** | **`{best_lasso_row['lasso_accuracy']:.4f}`** | **{int(best_lasso_row['lasso_retained_count'])} / {len(PREDICTOR_COLS)} features** ({best_lasso_row['lasso_retained_pct']:.1f}%) |

---

## 2. Optimal Lasso Feature Selection

The optimal Lasso model ($\lambda^* = {lambda_lasso_opt:.4f}$) retained **{int(best_lasso_row['lasso_retained_count'])} of {len(PREDICTOR_COLS)} candidate predictors**:

- **Retained Features:**
"""
    for feat in best_lasso_row["lasso_retained_features"].split(";"):
        if feat:
            md_content += f"  - `{feat}`\n"

    md_content += f"""
- **Eliminated (Shrunk to Exactly Zero):**
"""
    eliminated = [c for c in PREDICTOR_COLS if c not in best_lasso_row["lasso_retained_features"].split(";")]
    for feat in eliminated:
        md_content += f"  - `{feat}`\n"

    md_content += f"""
---

## 3. Deliverable Artifacts Generated

1. `data/processed/fitted_models.joblib` (Serialized trained estimators)
2. `reports/hyperparameter_tuning.csv` (Full 20-point grid search diagnostics)
3. `reports/figures/fig6_hyperparameter_tuning.png` (Validation loss & parameter complexity curves)
4. `reports/figures/lasso_shrinkage_path.png` (Lasso shrinkage paths and variable elimination)
"""
    summary_md_path = REPORTS_DIR / "tuning_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved tuning summary report to: {summary_md_path}")

    print("=" * 70)
    print("Gate 5 Model Calibration & Tuning Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_tuning()
