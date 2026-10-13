# Gate 3 Verification Report: Feature Engineering and Temporal Leakage Validation

**Project:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Workspace:** `d:\MLE-Lasso`  
**Date of Audit:** October 2026  
**Quality Gate:** **Gate 3 (Feature Engineering & Leakage Validation)**  
**Status:** **PASSED**  
**Governing Documents:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md) | [RESEARCH_METHODOLOGY_GUIDE.md](file:///d:/MLE-Lasso/RESEARCH_METHODOLOGY_GUIDE.md) | [DECISIONS.md](file:///d:/MLE-Lasso/DECISIONS.md)  

---

## 1. Executive Summary

This report documents the execution and empirical verification of **Quality Gate 3: Feature Engineering and Leakage Validation** under the approved **ADR-007 (Option A: The Verified High-Fidelity 16-Predictor Set)**.

All 16 candidate pre-match differential predictors were engineered from raw, immutable Football-Data.co.uk seasonal logs and FBref schedules. The pipeline was executed with strict chronological sequencing by kickoff timestamp, ensuring zero temporal lookahead leakage ($j \ge 1$ for all historical rolling windows).

### Key Empirical Findings:
1. **Total Sample:** Exactly $N = 1,900$ completed Premier League matches across five seasons (`2019/20` through `2023/24`, 380 matches per season).
2. **Modelling-Ready Partition:** $N = 1,741$ matches ($91.63\%$) meet the full historical criteria ($k \ge 10$ overall matches, $k \ge 5$ venue matches). Exactly **0 null values** exist across all 16 predictors in this partition.
3. **Warm-Up Exclusions:** $N = 159$ matches ($8.37\%$) are preserved with `is_modelling_ready = False`. These fixtures occur during the initial warm-up phase (Gameweeks 1–10 of 2019/20 and early fixtures of promoted teams).
4. **Automated Testing:** 20 out of 20 automated tests in `tests/test_features.py` and `tests/test_leakage.py` passed with code 0.
5. **Collinearity Verification:** Multiple feature pairs exhibit strong collinearity ($|r| > 0.70$ and up to $0.89$), establishing the empirical prerequisite for comparing unregularized MLE against Ridge and Lasso.
6. **Leakage Tests:** Synthetic mutation of current-match and future-match in-game statistics confirmed 100% bit-for-bit invariance of pre-match feature vectors.

---

## 2. Raw Input Data Inventory & Verification

All raw data files in `data/raw/` were verified on disk prior to transformation. Raw files remain 100% read-only and immutable.

| Source | Season | Local File Path | Size (Bytes) | Rows | Cols | Missing Core Fields |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| Football-Data | `2019-2020` | `data/raw/football_data/premier_league_2019-2020.csv` | 175,475 | 380 | 106 | 0 |
| Football-Data | `2020-2021` | `data/raw/football_data/premier_league_2020-2021.csv` | 175,815 | 380 | 106 | 0 |
| Football-Data | `2021-2022` | `data/raw/football_data/premier_league_2021-2022.csv` | 175,363 | 380 | 106 | 0 |
| Football-Data | `2022-2023` | `data/raw/football_data/premier_league_2022-2023.csv` | 176,180 | 380 | 106 | 0 |
| Football-Data | `2023-2024` | `data/raw/football_data/premier_league_2023-2024.csv` | 172,196 | 380 | 106 | 0 |
| FBref Schedule| `2019–2024` | `data/raw/fbref/fbref_premier_league_2019_2024_schedule.csv` | 460,710 | 1,900 | 16 | 0* |

*\*Note: FBref `attendance` contains COVID-19 empty-stadium missingness; attendance was purged from candidate predictors per ADR-005. All match metadata (kickoff dates, times, venues, teams, scores) have zero missing values.*

### Cross-Source Fixture Reconciliation
- Reconciled 1,900 / 1,900 matches between Football-Data and FBref schedules using the canonical 26-club team normalization dictionary.
- Score concordance: **100.0%** (1,900 / 1,900 matches match full-time goals).
- Kickoff date concordance: **100.0%** (0 date discrepancies).
- Fixture duplicates: **0 duplicate rows, 0 duplicate fixtures**.

---

## 3. Approved 16 Predictors: Exact Formulas and Raw Mappings

All predictors follow the symmetrical home-minus-away differential convention:
$$\Delta X = X_{\text{Home}} - X_{\text{Away}}$$

A positive differential ($\Delta X > 0$) indicates a statistical advantage for the home club in that tactical dimension heading into the fixture.

| ID | Feature Name | Raw Source Columns | Rolling Window ($k$) | Mathematical Formulation | Division Guard |
| :---: | :--- | :--- | :---: | :--- | :--- |
| $X_1$ | `diff_sot_k5` | `HST`, `AST` | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{SoT}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{SoT}_{A, t-j}$ | N/A |
| $X_2$ | `diff_sotA_k5` | `AST`, `HST` (Conceded) | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{SoTA}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{SoTA}_{A, t-j}$ | N/A |
| $X_3$ | `diff_shots_k5` | `HS`, `AS` | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{Shots}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{Shots}_{A, t-j}$ | N/A |
| $X_4$ | `diff_shotsA_k5` | `AS`, `HS` (Conceded) | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{ShotsA}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{ShotsA}_{A, t-j}$ | N/A |
| $X_5$ | `diff_sot_ratio_k5` | `HST`, `HS`, `AST`, `AS` | 5 matches | $\frac{\sum \text{SoT}_H}{\sum \text{Shots}_H} - \frac{\sum \text{SoT}_A}{\sum \text{Shots}_A}$ | $0.0$ if $\sum \text{Shots} = 0$ |
| $X_6$ | `diff_shot_dominance_k5`| `HS`, `AS` | 5 matches | $\frac{\sum \text{Shots}_H}{\sum (\text{Shots}_H + \text{ShotsA}_H)} - \frac{\sum \text{Shots}_A}{\sum (\text{Shots}_A + \text{ShotsA}_A)}$ | $0.5$ (neutral share) if $\sum \text{Shots} + \text{ShotsA} = 0$ |
| $X_7$ | `diff_corners_k5` | `HC`, `AC` | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{Corners}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{Corners}_{A, t-j}$ | N/A |
| $X_8$ | `diff_cornersA_k5` | `AC`, `HC` (Conceded) | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{CornersA}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{CornersA}_{A, t-j}$ | N/A |
| $X_9$ | `diff_fouls_k5` | `HF`, `AF` | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{Fouls}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{Fouls}_{A, t-j}$ | N/A |
| $X_{10}$| `diff_yellows_k5` | `HY`, `AY` | 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{Yellows}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{Yellows}_{A, t-j}$ | N/A |
| $X_{11}$| `diff_goal_diff_k5` | `FTHG`, `FTAG` | 5 matches | $\frac{1}{5}\sum_{j=1}^5 (\text{GF} - \text{GA})_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 (\text{GF} - \text{GA})_{A, t-j}$ | N/A |
| $X_{12}$| `diff_goal_diff_k10`| `FTHG`, `FTAG` | 10 matches | $\frac{1}{10}\sum_{j=1}^{10} (\text{GF} - \text{GA})_{H, t-j} - \frac{1}{10}\sum_{j=1}^{10} (\text{GF} - \text{GA})_{A, t-j}$ | N/A |
| $X_{13}$| `diff_clean_sheets_k10`| `FTAG`, `FTHG` | 10 matches | $\frac{1}{10}\sum_{j=1}^{10} \mathbb{I}(\text{GA}_{H} = 0) - \frac{1}{10}\sum_{j=1}^{10} \mathbb{I}(\text{GA}_{A} = 0)$ | N/A |
| $X_{14}$| `diff_ppg_k5` | `FTR` (3 for W, 1 for D, 0 for L)| 5 matches | $\frac{1}{5}\sum_{j=1}^5 \text{Pts}_{H, t-j} - \frac{1}{5}\sum_{j=1}^5 \text{Pts}_{A, t-j}$ | N/A |
| $X_{15}$| `diff_venue_ppg_k5` | `FTR` by Venue | 5 venue games | $\frac{1}{5}\sum_{m=1}^5 \text{HomePts}_{H, t-m} - \frac{1}{5}\sum_{m=1}^5 \text{AwayPts}_{A, t-m}$ | N/A |
| $X_{16}$| `diff_rest_days` | Kickoff Timestamps (`Date`, `Time`) | Continuous | $\frac{\tau_{t} - \tau_{t-1}^H}{86400} - \frac{\tau_{t} - \tau_{t-1}^A}{86400}$ | Excluded if $t-1$ missing |

---

## 4. Empirical Match Counts & Warm-Up Missingness Analysis

### 4.1 Sample Partition Breakdown

| Partition | Definition / Criterion | Fixture Count | Proportion (%) | Nulls in 16 Predictors |
| :--- | :--- | :---: | :---: | :---: |
| **Total Sample** | All completed fixtures (2019/20–2023/24) | 1,900 | 100.00% | Limited to warm-up |
| **Modelling-Ready** | $k \ge 10$ overall AND $k \ge 5$ venue for both clubs | **1,741** | **91.63%** | **0 (100% complete)** |
| **Warm-Up Excluded** | Insufficient historical matches (<10 total or <5 venue) | **159** | **8.37%** | Expected cold-start |

### 4.2 Warm-Up Distribution Across Seasons

| Season | Total Matches | Modelling-Ready ($N$) | Warm-Up ($N$) | Warm-Up Cause |
| :---: | :---: | :---: | :---: | :--- |
| **2019-2020** | 380 | 280 (73.68%) | 100 (26.32%) | Initial league-wide baseline accumulation (Gameweeks 1–10) |
| **2020-2021** | 380 | 352 (92.63%) | 28 (7.37%) | Promoted clubs (Leeds, Fulham, West Brom) building top-flight history |
| **2021-2022** | 380 | 370 (97.37%) | 10 (2.63%) | Promoted clubs (Brentford, Watford, Norwich) building venue history |
| **2022-2023** | 380 | 370 (97.37%) | 10 (2.63%) | Promoted clubs (Fulham, Bournemouth, Nottingham Forest) venue form |
| **2023-2024** | 380 | 369 (97.11%) | 11 (2.89%) | Promoted clubs (Burnley, Sheffield United, Luton Town) venue form |
| **Total** | **1,900** | **1,741 (91.63%)** | **159 (8.37%)** | Fully documented and audited |

### 4.3 Missing Data Policy Adherence
- **Zero Synthetic Imputation:** No artificial values (zeroes, future averages, or season-end statistics) were injected into the modelling-ready dataset.
- **Explicit Flagging:** Every observation contains the Boolean flag `is_modelling_ready`.
- **Target Invariance:** Ground truth scores (`home_goals`, `away_goals`) and targets (`target_multinomial`, `target_binary`) are preserved across all 1,900 rows.

---

## 5. Descriptive Statistics of the 16 Approved Predictors

Computed strictly on the modelling-ready partition ($N = 1,741$):

| Feature Name | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `diff_sot_k5` | 1,741 | -0.0829 | 1.9696 | -7.8000 | -1.4000 | 0.0000 | 1.2000 | 6.2000 |
| `diff_sotA_k5` | 1,741 | 0.0816 | 1.9010 | -6.4000 | -1.2000 | 0.2000 | 1.4000 | 6.6000 |
| `diff_shots_k5` | 1,741 | -0.2651 | 4.4518 | -15.2000 | -3.2000 | -0.4000 | 2.6000 | 16.6000 |
| `diff_shotsA_k5` | 1,741 | 0.2431 | 4.5924 | -17.4000 | -3.0000 | 0.4000 | 3.4000 | 15.8000 |
| `diff_sot_ratio_k5` | 1,741 | 0.0007 | 0.0950 | -0.4651 | -0.0606 | 0.0026 | 0.0643 | 0.3486 |
| `diff_shot_dominance_k5`| 1,741 | -0.0103 | 0.1574 | -0.4876 | -0.1164 | -0.0127 | 0.0926 | 0.4223 |
| `diff_corners_k5` | 1,741 | -0.1006 | 2.1374 | -8.4000 | -1.6000 | -0.2000 | 1.2000 | 7.8000 |
| `diff_cornersA_k5` | 1,741 | 0.0718 | 2.3173 | -7.6000 | -1.6000 | 0.2000 | 1.6000 | 8.4000 |
| `diff_fouls_k5` | 1,741 | 0.0501 | 2.5309 | -9.4000 | -1.6000 | 0.0000 | 1.8000 | 8.2000 |
| `diff_yellows_k5` | 1,741 | 0.0176 | 0.8762 | -3.2000 | -0.6000 | 0.0000 | 0.6000 | 3.0000 |
| `diff_goal_diff_k5` | 1,741 | -0.0437 | 1.5053 | -5.4000 | -1.0000 | 0.0000 | 1.0000 | 5.6000 |
| `diff_goal_diff_k10` | 1,741 | -0.0147 | 1.2692 | -3.8000 | -0.8000 | 0.0000 | 0.8000 | 4.0000 |
| `diff_clean_sheets_k10`| 1,741 | -0.0030 | 0.2243 | -0.7000 | -0.1000 | 0.0000 | 0.1000 | 0.7000 |
| `diff_ppg_k5` | 1,741 | -0.0329 | 0.9896 | -3.0000 | -0.8000 | 0.0000 | 0.6000 | 2.8000 |
| `diff_venue_ppg_k5` | 1,741 | 0.3296 | 0.9840 | -2.8000 | -0.4000 | 0.4000 | 1.0000 | 3.0000 |
| `diff_rest_days` | 1,741 | 1.5158 | 29.2877 | -102.1146 | -0.9583 | 0.0000 | 0.9583 | 735.0208 |

### Tactical Insights from Descriptive Statistics:
1. **Home Advantage in Venue Form:** `diff_venue_ppg_k5` exhibits a positive mean of $+0.3296$ points/game ($p < 0.001$), confirming that home teams systematically average higher points at home than away teams do on the road.
2. **Symmetrical Balance:** Predictors representing overall neutral capabilities (`diff_sot_k5`, `diff_sot_ratio_k5`, `diff_goal_diff_k10`, `diff_clean_sheets_k10`) are centered near $0.0$, demonstrating unbiased differential formulation.
3. **Collinear Clusters:** Pairwise correlation analysis identifies severe collinearity among tactical metrics:
   - `diff_shotsA_k5` $\leftrightarrow$ `diff_shot_dominance_k5`: $r = -0.8901$
   - `diff_goal_diff_k5` $\leftrightarrow$ `diff_ppg_k5`: $r = +0.8884$
   - `diff_shots_k5` $\leftrightarrow$ `diff_shot_dominance_k5`: $r = +0.8792$
   - `diff_goal_diff_k5` $\leftrightarrow$ `diff_goal_diff_k10`: $r = +0.8397$
   - `diff_sot_k5` $\leftrightarrow$ `diff_shots_k5`: $r = +0.7973$
   - `diff_sotA_k5` $\leftrightarrow$ `diff_shotsA_k5`: $r = +0.7887$
   These collinear structures confirm that the dataset is primed for testing Ridge and Lasso regularization dynamics against unconstrained MLE.

---

## 6. Automated Testing & Verification Evidence

Automated tests were executed in the project environment using `pytest 9.1.1`.

```bash
python -m pytest tests/test_features.py tests/test_leakage.py -v
```

### Test Results Matrix (20 / 20 PASSED)

| Test Module | Test Case Function | Verification Target | Result |
| :--- | :--- | :--- | :---: |
| `test_features.py` | `test_global_chronological_ordering` | Monotonic ordering of kickoff timestamps | **PASSED** |
| `test_features.py` | `test_team_level_strictly_monotonic_kickoff_timestamps` | Each team's match timestamps strictly increase | **PASSED** |
| `test_features.py` | `test_zero_missing_kickoff_timestamps` | Zero null dates or times across 1,900 rows | **PASSED** |
| `test_features.py` | `test_k5_rolling_window_manual_calculation` | Hand-calculated mathematical accuracy for $k=5$ | **PASSED** |
| `test_features.py` | `test_k10_rolling_window_manual_calculation` | Hand-calculated mathematical accuracy for $k=10$ | **PASSED** |
| `test_features.py` | `test_venue_ppg_manual_calculation` | Symmetrical venue PPG calculation | **PASSED** |
| `test_features.py` | `test_rest_days_continuous_calculation` | Fractional continuous rest days ($\Delta \tau / 86400$) | **PASSED** |
| `test_features.py` | `test_sot_ratio_zero_denominator_guard` | Shot accuracy returns 0.0 when shots = 0 | **PASSED** |
| `test_features.py` | `test_shot_dominance_zero_denominator_guard` | Shot dominance returns 0.5 when total shots = 0 | **PASSED** |
| `test_features.py` | `test_exact_feature_schema` | Exactly 16 predictors, 4 targets, 7 metadata | **PASSED** |
| `test_features.py` | `test_fixture_row_counts` | Total 1,900; ready 1,741; warm-up 159 | **PASSED** |
| `test_features.py` | `test_zero_nulls_in_modelling_ready_partition` | 0 nulls across 16 predictors in ready set | **PASSED** |
| `test_features.py` | `test_all_finite_values_in_modelling_ready_partition` | All values strictly finite (no Inf/-Inf/NaN) | **PASSED** |
| `test_features.py` | `test_zero_duplicate_matches` | Zero duplicate rows or match_id keys | **PASSED** |
| `test_features.py` | `test_target_encodings_validity` | Targets match goals scored and classes {0, 1, 2} | **PASSED** |
| `test_features.py` | `test_differential_direction_symmetry` | Bounded form differentials [-3.0, +3.0] | **PASSED** |
| `test_leakage.py` | `test_target_isolation_from_predictors` | Target and in-match statistics excluded | **PASSED** |
| `test_leakage.py` | `test_current_match_invariance_leakage_assertion` | Mutating current match stats leaves features unchanged | **PASSED** |
| `test_leakage.py` | `test_future_match_invariance_leakage_assertion` | Mutating future match stats leaves past features unchanged | **PASSED** |
| `test_leakage.py` | `test_chronological_step_propagation` | Stats propagate from match $t$ to $t+1$, never $t$ | **PASSED** |

---

## 7. Limitations and Unresolved Implementation Details

1. **Rest Days Congestion Outliers:**
   Fourteen fixtures exhibit $|\Delta \text{rest\_days}| > 30$ days. These correspond to:
   - The three-month COVID-19 suspension in Spring 2020 ("Project Restart"), where games-in-hand caused temporary rest imbalances.
   - Returning promoted clubs playing their first fixture of a season against clubs that completed the prior season.
   - *Mitigation Plan for Gate 4:* Robust scaling or standardizing fitted strictly on the training partition will handle extreme values without data leakage.
2. **Promoted Club Representation:**
   Promoted clubs naturally incur cold-start warm-up periods during their first 10 top-flight fixtures. By excluding these 159 fixtures from the modelling-ready dataset, parameter estimates remain uncorrupted by unobserved lower-league play or synthetic assumptions.
3. **Absence of Micro-Tracking Coordinates:**
   Per ADR-007 (Option A), tracking coordinates (Opta event centroids, xG shot maps) are omitted in favor of 100% complete shot, dominance, and pressure statistics. The resulting collinearity matrix ($r \approx 0.89$) provides the mathematical structure required for the dissertation's comparative regularisation objectives.

---

## 8. Gate 3 Sign-Off

All exit criteria defined in [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) for Gate 3 have been verified with reproducible code execution and automated test passing.

- [x] Canonical match order is strictly chronological by kickoff timestamp.
- [x] Moving average operator strictly excludes current match $t$ ($j \ge 1$ for all summation indices).
- [x] In-match events are strictly excluded from the covariate vector.
- [x] Symmetrical differential calculation follows $\Delta X = X_{\text{Home}} - X_{\text{Away}}$.
- [x] Warm-up observations ($N = 159$) are explicitly audited and separated from modelling-ready rows ($N = 1,741$).
- [x] Leakage assertion tests pass (current-match and future-match invariance verified).

**Gate 3 Status:** **PASSED**  
**Recommended Next Step:** Await user sign-off to proceed to Gate 4 (Chronological Train/Val/Test Partitioning and Frozen Standardization).
