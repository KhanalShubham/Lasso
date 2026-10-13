# Training Partition Collinearity & VIF Diagnostics (Gate 4)

**Sample:** Training Partition (Seasons 2019/20 - 2021/22, $N = 1,002$ matches)  
**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 4  

---

## 1. Matrix Condition Numbers

- **Design Matrix Condition Number $\kappa(\mathbf{X})$:** `14.0`
- **Empirical Unpenalized Hessian Condition Number $\kappa(\mathbf{H}_0)$:** `195.96`
- **Features with $\text{VIF} \ge 5.0$:** `3` of 8 predictors
- **Features with $\text{VIF} \ge 10.0$:** `2` of 8 predictors

---

## 2. Empirical Variance Inflation Factors

| Rank | Predictor | VIF ($1 / (1-R^2)$) | $R_j^2$ | Tactical Cluster | Top Correlate | Max Pearson $r$ |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: |
| 1 | `diff_ppg_k5` | **24.63** | 0.9594 | Recent Form & Momentum | `diff_win_rate_k5` | +0.9662 |
| 2 | `diff_win_rate_k5` | **17.23** | 0.9419 | Win Conversion Rate | `diff_ppg_k5` | +0.9662 |
| 3 | `diff_goal_diff_k10` | **5.76** | 0.8263 | Sustained Structural Quality | `diff_ppg_k5` | +0.7560 |
| 4 | `diff_goals_scored_k5` | **4.24** | 0.7640 | Attacking Threat & Output | `diff_ppg_k5` | +0.7455 |
| 5 | `diff_goals_conceded_k5` | **3.53** | 0.7167 | Defensive Concession | `diff_ppg_k5` | -0.7013 |
| 6 | `diff_venue_ppg_k5` | **2.34** | 0.5735 | Venue-Specific Home/Away Form | `diff_goal_diff_k10` | +0.7196 |
| 7 | `diff_clean_sheets_k10` | **2.28** | 0.5619 | Defensive Solidity & Shutouts | `diff_goal_diff_k10` | +0.6879 |
| 8 | `diff_rest_days` | **1.01** | 0.0084 | Schedule Congestion & Rest | `diff_goal_diff_k10` | -0.0727 |
