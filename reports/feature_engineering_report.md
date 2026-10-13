# Feature Engineering & Leakage Safeguards Report (reports/feature_engineering_report.md)

**Project Title:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Pipeline Version:** OpenFootball Result-Derived Features (ADR-013)  
**Date:** 13 October 2026  
**Governing Documents:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md) | [RESEARCH_METHODOLOGY_GUIDE.md](file:///d:/MLE-Lasso/RESEARCH_METHODOLOGY_GUIDE.md) | [DECISIONS.md](file:///d:/MLE-Lasso/DECISIONS.md)  
**Deliverable Dataset:** [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv) ($N=1,900$ total, $N=1,741$ modelling-ready)  

---

## 1. Overview & Methodological Objectives

This report details the mathematical formulation, implementation, and leakage validation of the result-derived feature engineering pipeline operating on the public-domain OpenFootball dataset.

Following the migration mandated by ADR-013, candidate predictors are constructed exclusively from historical match results, final scorelines, and match timing. The pipeline enforces four foundational guarantees:
1. **Strict Temporal Priority ($j \ge 1$):** Pre-match features for match $t$ are derived exclusively from matches $t-j$ where $j \ge 1$. The current match $t$ is strictly excluded from its own covariate vector.
2. **Cold-Start Warm-Up Isolation:** Matches where either competing club has completed fewer than 10 top-flight matches overall or fewer than 5 venue-specific matches are explicitly flagged as `is_modelling_ready = False`. Their feature values remain `NaN` and are never imputed with zero or future data.
3. **Symmetrical Home-Minus-Away Differentials:** Relative team strength is parameterized as $\Delta X = X_{\text{Home}} - X_{\text{Away}}$.
4. **Monotonic Chronological Sequencing:** Fixtures are processed strictly in order of kickoff timestamp $(\text{date}, \text{time})$, ensuring proper sequencing of rescheduled and postponed fixtures.

---

## 2. Mathematical Formulations of Engineered Features

Let match $t$ take place at kickoff timestamp $\tau_t$ between home club $H$ and away club $A$. Let $\mathcal{H}_t(C) = \{m \in \text{Completed Matches} \mid \tau_m < \tau_t, C \in m\}$ denote the chronological sequence of matches completed by club $C$ prior to match $t$.

### 2.1 Rolling Moving-Average Operator
For any match metric $x_{m}(C)$ (e.g. goals scored, goals conceded, points), the rolling moving-average operator over window $k$ is:
$$\bar{X}_{t}^{(k)}(C) = \frac{1}{k} \sum_{j=1}^{k} x_{t-j}(C)$$
where $t-j$ references the $j$-th most recent fixture in $\mathcal{H}_t(C)$. The summation strictly starts at $j=1$ (Invariant 1).

### 2.2 Points Calculation
Match points are defined according to standard association football league scoring:
$$P_m(C) = \begin{cases} 3 & \text{if } G_m(C) > G_m(\text{Opponent}) \quad (\text{Win}) \\ 1 & \text{if } G_m(C) = G_m(\text{Opponent}) \quad (\text{Draw}) \\ 0 & \text{if } G_m(C) < G_m(\text{Opponent}) \quad (\text{Loss}) \end{cases}$$

### 2.3 Venue-Specific Moving Average
Let $\mathcal{H}_{t, \text{venue}}(C) \subset \mathcal{H}_t(C)$ denote the sub-sequence where club $C$ played at the specified venue ($\text{home}$ for club $H$; $\text{away}$ for club $A$). The venue-specific form over window $k=5$ is:
$$\bar{X}_{t, \text{venue}}^{(5)}(C) = \frac{1}{5} \sum_{j=1}^{5} x_{t-j, \text{venue}}(C)$$

### 2.4 Inter-Match Rest Days
Let $\tau_{\text{last}}(C)$ denote the kickoff timestamp of club $C$'s immediately preceding match ($j=1$). Rest days in fractional calendar days is:
$$\Delta \tau(C) = \frac{\tau_t - \tau_{\text{last}}(C)}{86,400 \text{ seconds}}$$

---

## 3. Candidate Predictor Portfolio

### 3.1 Primary Symmetrical Differentials ($p = 9$)

| Predictor Identifier | Mathematical Definition | Window ($k$) | Tactical Dimension |
| :--- | :--- | :---: | :--- |
| `diff_goals_scored_k5` | $\bar{G}_{H, t}^{(5)} - \bar{G}_{A, t}^{(5)}$ | 5 matches | Short-term attacking firepower differential |
| `diff_goals_conceded_k5` | $\bar{C}_{H, t}^{(5)} - \bar{C}_{A, t}^{(5)}$ | 5 matches | Short-term defensive concession differential |
| `diff_ppg_k5` | $\bar{P}_{H, t}^{(5)} - \bar{P}_{A, t}^{(5)}$ | 5 matches | Recent overall form / momentum differential |
| `diff_goal_diff_k10` | $\overline{GD}_{H, t}^{(10)} - \overline{GD}_{A, t}^{(10)}$ | 10 matches | Sustained structural goal difference per match |
| `diff_venue_ppg_k5` | $\bar{P}_{H, \text{home}, t}^{(5)} - \bar{P}_{A, \text{away}, t}^{(5)}$ | 5 venue games | Home form of Home club vs. Road form of Away club |
| `diff_rest_days` | $\Delta \tau(H) - \Delta \tau(A)$ | Inter-match | Relative schedule fatigue / recovery advantage |
| `diff_goal_diff_k5` | $\overline{GD}_{H, t}^{(5)} - \overline{GD}_{A, t}^{(5)}$ | 5 matches | Short-term goal difference momentum differential |
| `diff_clean_sheets_k10` | $\overline{CS}_{H, t}^{(10)} - \overline{CS}_{A, t}^{(10)}$ | 10 matches | Sustained defensive solidity / shutout rate |
| `diff_win_rate_k5` | $\bar{W}_{H, t}^{(5)} - \bar{W}_{A, t}^{(5)}$ | 5 matches | Short-term win conversion efficiency |

### 3.2 Individual Team Form Metrics ($p = 18$)
The dataset also preserves raw individual team metrics:
- `home_goals_scored_k5`, `away_goals_scored_k5`
- `home_goals_conceded_k5`, `away_goals_conceded_k5`
- `home_ppg_k5`, `away_ppg_k5`
- `home_gd_k10`, `away_gd_k10`
- `home_venue_ppg_k5`, `away_venue_ppg_k5`
- `home_rest_days`, `away_rest_days`
- `home_gd_k5`, `away_gd_k5`
- `home_clean_sheets_k10`, `away_clean_sheets_k10`
- `home_win_rate_k5`, `away_win_rate_k5`

---

## 4. Cold-Start Warm-Up Isolation Breakdown

To avoid contaminating model parameters with arbitrary prior imputations or artificial zeros, fixtures lacking sufficient empirical top-flight history ($N < 10$ overall matches or $N < 5$ venue matches for either competing team) are flagged as `is_modelling_ready = False`.

### 4.1 Seasonal Partitioning and Warm-Up Counts

| Season | Gross Fixtures | Warm-Up Excluded Rows ($N_{\text{warmup}}$) | Modelling-Ready Rows ($N_{\text{ready}}$) | Warm-Up Fraction |
| :--- | :---: | :---: | :---: | :---: |
| **2019/20** | 380 | 100 | 280 | 26.3% |
| **2020/21** | 380 | 28 | 352 | 7.4% |
| **2021/22** | 380 | 10 | 370 | 2.6% |
| **2022/23** | 380 | 10 | 370 | 2.6% |
| **2023/24** | 380 | 11 | 369 | 2.9% |
| **Total** | **1,900** | **159** | **1,741** | **8.37%** |

### 4.2 Chronological Modelling Partitions
- **Training Partition (Seasons 2019/20, 2020/21, 2021/22):** Exactly $N_{\text{train}} = 1,002$ modelling-ready matches (280 + 352 + 370).
- **Validation Partition (Season 2022/23):** Exactly $N_{\text{val}} = 370$ modelling-ready matches.
- **Test Partition (Season 2023/24):** Exactly $N_{\text{test}} = 369$ modelling-ready matches.
- **Sum:** $1,002 + 370 + 369 = 1,741$ modelling-ready fixtures, exactly replicating the approved experimental partitioning protocol (ADR-002).

---

## 5. Missingness and Zero-Imputation Prohibition Audit

An exhaustive audit of [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv) confirms:
1. **Modelling-Ready Partition ($N=1,741$):**
   - Missing values in metadata: **0 (0.0%)**
   - Missing values in targets: **0 (0.0%)**
   - Missing values across all 27 candidate predictors: **0 (0.0%)**
2. **Warm-Up Partition ($N=159$):**
   - Missing values across all predictor columns: **159 (100.0% NaN)**
   - No zero-imputation was performed; true historical absence is preserved.

---

## 6. Automated Leakage Test Suite Results

The pipeline was subjected to automated property-based leakage assertions in [`tests/test_openfootball_pipeline.py`](file:///d:/MLE-Lasso/tests/test_openfootball_pipeline.py):

| Test Name | Invariant Tested | Execution Result |
| :--- | :--- | :---: |
| `test_current_match_leakage_invariance` | Modifying match $t$ score (12-9) in raw data does NOT alter match $t$ features | **PASSED** |
| `test_future_match_leakage_invariance` | Modifying match $t+10$ score (10-0) does NOT alter match $t$ features | **PASSED** |
| `test_venue_specific_history_isolation` | Home venue PPG strictly uses preceding home fixtures | **PASSED** |
| `test_rest_days_accuracy` | Rest days precisely matches timestamp difference | **PASSED** |
| `test_warmup_missingness_guarantee` | All 159 warm-up rows strictly maintain NaN missingness | **PASSED** |
| `test_modelling_ready_zero_nulls_guarantee` | All 1,741 modelling rows strictly have 0 nulls | **PASSED** |
| `test_reproducible_pipeline_output` | Successive pipeline runs yield byte-identical DataFrames | **PASSED** |

**Conclusion:** The OpenFootball result-derived feature engineering pipeline satisfies all methodological non-leakage invariants.
