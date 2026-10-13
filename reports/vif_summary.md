# Training Partition Collinearity & VIF Diagnostics (Gate 4)

**Sample:** Training Partition (Seasons 2019/20 - 2021/22, $N = 1,002$ matches)  
**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 4  

---

## 1. Matrix Condition Numbers

- **Design Matrix Condition Number $\kappa(\mathbf{X})$:** `30.63`
- **Empirical Unpenalized Hessian Condition Number $\kappa(\mathbf{H}_0)$:** `938.36`
- **Features with $\text{VIF} \ge 5.0$:** `8` of 16 predictors
- **Features with $\text{VIF} \ge 10.0$:** `5` of 16 predictors

---

## 2. Empirical Variance Inflation Factors

| Rank | Predictor | VIF ($1 / (1-R^2)$) | $R_j^2$ | Tactical Cluster | Top Correlate | Max Pearson $r$ |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| 1 | `diff_shot_dominance_k5` | **74.14** | 0.9865 | Shot Volume & Dominance | `diff_shotsA_k5` | -0.8928 |
| 2 | `diff_shots_k5` | **48.23** | 0.9793 | Shot Volume & Dominance | `diff_shot_dominance_k5` | +0.8888 |
| 3 | `diff_sot_k5` | **45.06** | 0.9778 | Shot Quality & Precision | `diff_shots_k5` | +0.7924 |
| 4 | `diff_shotsA_k5` | **26.24** | 0.9619 | Shot Volume & Dominance | `diff_shot_dominance_k5` | -0.8928 |
| 5 | `diff_sot_ratio_k5` | **16.75** | 0.9403 | Shot Quality & Precision | `diff_sot_k5` | +0.6429 |
| 6 | `diff_goal_diff_k5` | **8.40** | 0.8810 | Form & Goal Difference | `diff_ppg_k5` | +0.8861 |
| 7 | `diff_goal_diff_k10` | **6.33** | 0.8420 | Form & Goal Difference | `diff_goal_diff_k5` | +0.8429 |
| 8 | `diff_ppg_k5` | **5.46** | 0.8168 | Form & Goal Difference | `diff_goal_diff_k5` | +0.8861 |
| 9 | `diff_sotA_k5` | **3.24** | 0.6918 | Shot Quality & Precision | `diff_shotsA_k5` | +0.7838 |
| 10 | `diff_venue_ppg_k5` | **2.36** | 0.5755 | Form & Goal Difference | `diff_goal_diff_k10` | +0.7196 |
| 11 | `diff_cornersA_k5` | **2.21** | 0.5481 | Territorial Set Pieces | `diff_shot_dominance_k5` | -0.7045 |
| 12 | `diff_clean_sheets_k10` | **2.16** | 0.5363 | Defensive Solidity | `diff_goal_diff_k10` | +0.6879 |
| 13 | `diff_corners_k5` | **1.98** | 0.4938 | Territorial Set Pieces | `diff_shots_k5` | +0.6757 |
| 14 | `diff_yellows_k5` | **1.33** | 0.2476 | Disciplinary & Aggression | `diff_fouls_k5` | +0.4268 |
| 15 | `diff_fouls_k5` | **1.25** | 0.2024 | Disciplinary & Aggression | `diff_yellows_k5` | +0.4268 |
| 16 | `diff_rest_days` | **1.03** | 0.0275 | Schedule Congestion | `diff_corners_k5` | -0.0957 |
