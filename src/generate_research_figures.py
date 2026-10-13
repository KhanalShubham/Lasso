"""Publication-Quality Research Figures Generator.

Generates high-resolution (300 DPI) publication-ready figures for
the EPL Match Outcome Prediction Study:
  1. fig1_target_distribution.png (Pie & Bar Chart of match outcomes)
  2. fig2_seasonal_breakdown.png (Modelling-ready vs Warm-up sample across 5 seasons)
  3. fig3_collinearity_heatmap.png (16x16 Pearson correlation matrix)
  4. fig4_feature_distributions.png (4x4 multi-panel grid of all 16 differential features)
  5. fig5_collinear_pairs_scatter.png (Key collinear pairs with regression lines)

Outputs saved to:
  - reports/figures/
  - dashboard/figures/ (for web viewing and export)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = WORKSPACE_ROOT / "data" / "features" / "dataset_features.csv"
REPORTS_FIG_DIR = WORKSPACE_ROOT / "reports" / "figures"
DASHBOARD_FIG_DIR = WORKSPACE_ROOT / "dashboard" / "figures"

REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set clean academic style (black & white / minimalist)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.edgecolor": "#1e293b",
    "axes.linewidth": 1.2,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

FEATURE_TITLES = {
    "diff_sot_k5": "Shots on Target Diff (k=5)",
    "diff_sotA_k5": "Conceded Shots on Target Diff (k=5)",
    "diff_shots_k5": "Total Shots Diff (k=5)",
    "diff_shotsA_k5": "Conceded Total Shots Diff (k=5)",
    "diff_sot_ratio_k5": "Shot Accuracy Ratio Diff (k=5)",
    "diff_shot_dominance_k5": "Shot Dominance Share Diff (k=5)",
    "diff_corners_k5": "Corners Won Diff (k=5)",
    "diff_cornersA_k5": "Conceded Corners Diff (k=5)",
    "diff_fouls_k5": "Fouls Committed Diff (k=5)",
    "diff_yellows_k5": "Yellow Cards Diff (k=5)",
    "diff_goal_diff_k5": "Goal Difference Diff (k=5)",
    "diff_goal_diff_k10": "Goal Difference Diff (k=10)",
    "diff_clean_sheets_k10": "Clean Sheet Rate Diff (k=10)",
    "diff_ppg_k5": "Points Per Game Diff (k=5)",
    "diff_venue_ppg_k5": "Venue Form Diff (k=5)",
    "diff_rest_days": "Rest Days Diff",
}


def save_fig(fig, filename):
    """Save high-resolution figures to both reports and dashboard directories."""
    p1 = REPORTS_FIG_DIR / filename
    p2 = DASHBOARD_FIG_DIR / filename
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    print(f"Saved: {p1} and {p2}")
    plt.close(fig)


def generate_figures():
    df = pd.read_csv(FEATURES_CSV)
    ready = df[df["is_modelling_ready"]].copy()
    pred_cols = [c for c in df.columns if c.startswith("diff_")]

    # =========================================================================
    # Figure 1: Target Outcome Distribution (Pie & Bar)
    # =========================================================================
    print("Generating Figure 1: Target Outcome Distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Pie Chart
    outcome_counts = ready["target_multinomial"].value_counts().sort_index()
    # 0: Away Win, 1: Draw, 2: Home Win
    labels = [f"Away Win\n(N={outcome_counts[0]})", f"Draw\n(N={outcome_counts[1]})", f"Home Win\n(N={outcome_counts[2]})"]
    counts = [outcome_counts[0], outcome_counts[1], outcome_counts[2]]
    # Clean black and white with clean shading
    colors = ["#cbd5e1", "#94a3b8", "#1e293b"]
    explode = (0.02, 0.02, 0.05)

    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
        wedgeprops={"edgecolor": "#0f172a", "linewidth": 1.5},
        textprops={"fontsize": 11, "fontweight": "semibold"}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    # Make home win text white for contrast against dark wedge
    autotexts[2].set_color("white")
    ax1.set_title("A. Target Class Proportions (N = 1,741)", pad=15)

    # Bar Chart comparing Full vs Modelling-Ready
    full_counts = df["target_multinomial"].value_counts().sort_index()
    x = np.arange(3)
    width = 0.35

    rects1 = ax2.bar(x - width/2, [full_counts[2], full_counts[1], full_counts[0]], width, 
                    label="Full Sample (N=1,900)", color="#94a3b8", edgecolor="#0f172a", linewidth=1.2)
    rects2 = ax2.bar(x + width/2, [outcome_counts[2], outcome_counts[1], outcome_counts[0]], width, 
                    label="Modelling-Ready (N=1,741)", color="#1e293b", edgecolor="#0f172a", linewidth=1.2)

    ax2.set_ylabel("Number of Matches")
    ax2.set_title("B. Absolute Class Counts Comparison", pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Home Win (Class 2)", "Draw (Class 1)", "Away Win (Class 0)"])
    ax2.legend(frameon=True, edgecolor="#cbd5e1")
    ax2.set_ylim(0, 950)

    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="semibold")
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "fig1_target_distribution.png")

    # =========================================================================
    # Figure 2: Seasonal Sample Breakdown (Stacked Bar)
    # =========================================================================
    print("Generating Figure 2: Seasonal Sample Breakdown...")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    seasons = sorted(df["season"].unique())
    ready_by_s = [len(df[(df["season"] == s) & (df["is_modelling_ready"])]) for s in seasons]
    warmup_by_s = [len(df[(df["season"] == s) & (~df["is_modelling_ready"])]) for s in seasons]

    bars1 = ax.bar(seasons, ready_by_s, label="Modelling-Ready (k ≥ 10 matches)", 
                   color="#1e293b", edgecolor="#0f172a", linewidth=1.2, width=0.55)
    bars2 = ax.bar(seasons, warmup_by_s, bottom=ready_by_s, label="Warm-Up Excluded (Cold-Start)", 
                   color="#e2e8f0", edgecolor="#0f172a", linewidth=1.2, width=0.55, hatch="//")

    ax.set_ylabel("Number of Fixtures (Matches)")
    ax.set_title("Premier League Sample Composition by Season (2019/20 – 2023/24)", pad=15)
    ax.set_ylim(0, 480)
    ax.axhline(380, color="#64748b", linestyle=":", linewidth=1, label="Season Total (380 matches)")
    ax.legend(frameon=True, edgecolor="#cbd5e1", loc="upper left", fontsize=9.5)

    # Label values
    for i, (r_cnt, w_cnt) in enumerate(zip(ready_by_s, warmup_by_s)):
        # label ready
        ax.text(i, r_cnt / 2, f"{r_cnt}\n({round(r_cnt/380*100, 1)}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        # label warmup if present
        if w_cnt > 0:
            ax.text(i, r_cnt + w_cnt / 2, f"{w_cnt}", ha="center", va="center", color="#0f172a", fontweight="bold", fontsize=10)

    plt.tight_layout()
    save_fig(fig, "fig2_seasonal_sample_breakdown.png")

    # =========================================================================
    # Figure 3: Collinearity Heatmap (16x16 Matrix)
    # =========================================================================
    print("Generating Figure 3: Collinearity Heatmap...")
    fig, ax = plt.subplots(figsize=(13, 11))

    corr = ready[pred_cols].corr()
    short_labels = [FEATURE_TITLES.get(c, c) for c in pred_cols]

    # Clean monochromatic diverging palette with high contrast
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.5,
        linecolor="#ffffff",
        cbar_kws={"label": "Pearson Correlation Coefficient (r)", "shrink": 0.8},
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax,
        annot_kws={"size": 8, "fontweight": "semibold"}
    )

    ax.set_title("Multicollinearity Matrix of 16 Pre-Match Differential Features (N = 1,741)", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    save_fig(fig, "fig3_collinearity_heatmap.png")

    # =========================================================================
    # Figure 4: Feature Distributions Grid (4x4)
    # =========================================================================
    print("Generating Figure 4: 16 Feature Distributions Grid...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(pred_cols):
        ax = axes[i]
        vals = ready[col].dropna()
        # Clean rest days outlier purely for visual plotting
        if col == "diff_rest_days":
            vals = vals[(vals >= -20) & (vals <= 20)]

        # Histogram + Empirical KDE (Non-parametric density estimation)
        ax.hist(vals, bins=25, density=True, color="#cbd5e1", edgecolor="#1e293b", linewidth=0.8, alpha=0.85)
        sns.kdeplot(vals, ax=ax, color="#0f172a", linewidth=1.5, label="Empirical KDE")

        # Zero reference line
        ax.axvline(0, color="#f43f5e", linestyle="--", linewidth=1, alpha=0.8)

        short_t = FEATURE_TITLES.get(col, col)
        ax.set_title(short_t, fontsize=10, pad=6)
        ax.tick_params(axis="both", labelsize=8)
        # Stats annotation
        mu, sigma = vals.mean(), vals.std()
        ax.text(0.04, 0.85, f"μ={mu:.2f}\nσ={sigma:.2f}", transform=ax.transAxes, 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.9))

    plt.suptitle("Empirical Distributions of 16 Pre-Match Differentials (Modelling Sample, N = 1,741)", fontsize=14, y=0.995)
    plt.tight_layout()
    save_fig(fig, "fig4_feature_distributions.png")

    # =========================================================================
    # Figure 5: Top Collinear Pairs Scatterplots
    # =========================================================================
    print("Generating Figure 5: Collinear Clusters Scatterplots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    pairs = [
        ("diff_shots_k5", "diff_shot_dominance_k5", "Total Shots vs. Shot Dominance (r = 0.88)"),
        ("diff_shotsA_k5", "diff_shot_dominance_k5", "Conceded Shots vs. Shot Dominance (r = -0.89)"),
        ("diff_goal_diff_k5", "diff_ppg_k5", "Goal Difference vs. Points Per Game (r = 0.89)"),
        ("diff_goal_diff_k5", "diff_goal_diff_k10", "Goal Diff (k=5) vs. Goal Diff (k=10) (r = 0.84)"),
    ]

    for ax, (x_col, y_col, title) in zip(axes.flatten(), pairs):
        x_vals = ready[x_col]
        y_vals = ready[y_col]

        # Scatter
        ax.scatter(x_vals, y_vals, alpha=0.35, color="#1e293b", edgecolors="none", s=25)
        # Linear regression fit
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, m * x_line + b, color="#f43f5e", linewidth=2, label=f"Fit (slope={m:.2f})")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(FEATURE_TITLES.get(x_col, x_col), fontsize=10)
        ax.set_ylabel(FEATURE_TITLES.get(y_col, y_col), fontsize=10)
        ax.legend(frameon=True, edgecolor="#cbd5e1", fontsize=9)

    plt.suptitle("Collinear Tactical Clusters Illustrating Redundant Dimensionality", fontsize=14, y=0.995)
    plt.tight_layout()
    save_fig(fig, "fig5_collinear_pairs_scatter.png")

    print("\nAll 5 research figures successfully generated and saved to reports/figures/ and dashboard/figures/!")


if __name__ == "__main__":
    generate_figures()
