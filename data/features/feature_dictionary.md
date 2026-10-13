# OpenFootball Result-Derived Feature Dictionary

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
