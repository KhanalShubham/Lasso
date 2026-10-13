# Architectural & Methodological Decision Log (DECISIONS.md)

**Project:** MLE vs Ridge vs Lasso Logistic Regression for Football Match Outcome Prediction  
**Status:** Living Document  
**Governing Document:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md)  

---

## Decision Status Summary

| ID | Title | Date | Status | Decider |
| :---: | :--- | :---: | :---: | :--- |
| **ADR-001** | Competition & Five-Season Scope Selection | 2026-10-12 | **ACCEPTED** | Research Proposal / Guide |
| **ADR-002** | Chronological Partitioning (Train/Val/Test) | 2026-10-12 | **ACCEPTED** | Methodology Guide / Constitution |
| **ADR-003** | Symmetric Multinomial Softmax Formulation | 2026-10-12 | **ACCEPTED** | Methodology Guide Sec. D.1 |
| **ADR-004** | Dual-Source Architecture (FBref + Football-Data) | 2026-10-12 | **ACCEPTED** | Data Audit Pipeline |
| **ADR-005** | Purging Attendance to Prevent COVID Missingness | 2026-10-12 | **ACCEPTED** | Methodology Guide Sec. C.3 |
| **ADR-006** | Promoted-Team Synthetic Baseline Prior Protocol | 2026-10-12 | **ACCEPTED** | Methodology Guide Sec. C.6 |
| **ADR-007** | Candidate Feature Set Specification ($p = 16$) | 2026-10-12 | **ACCEPTED (Option A)** | User Approval |
| **ADR-008** | Binary Benchmark vs 3-Class Primary Target | 2026-10-12 | **ACCEPTED** | Methodology Guide Sec. A.2 |
| **ADR-009** | Optimization Solvers & Framework Libraries | 2026-10-12 | **ACCEPTED** | Methodology Guide Sec. G.2 |
| **ADR-010** | Benchmark Baseline Specifications (Odds & Frequencies) | 2026-10-13 | **ACCEPTED** | Research Consensus |
| **ADR-011** | Gameweek Block-Bootstrap Resampling Protocol | 2026-10-13 | **ACCEPTED** | Methodology Design |
| **ADR-012** | Refined Hypothesis 3 (Parsimony & Predictive Parity) | 2026-10-13 | **ACCEPTED** | Research Consensus |
| **ADR-013** | Migration to OpenFootball Public-Domain Data Pipeline | 2026-10-13 | **ACCEPTED** | Project Direction / Migration |

---

## Detailed Architectural Decision Records (ADRs)

### ADR-001: Competition & Five-Season Scope Selection
- **Status:** **ACCEPTED**
- **Context:** The research requires dense, high-fidelity tracking data with uniform seasonal structures and sufficient sample size to evaluate coefficient shrinkage under collinearity.
- **Decision:** English Premier League across 5 contiguous seasons: `2019/20`, `2020/21`, `2021/22`, `2022/23`, and `2023/24`.
- **Rationale:** Exactly 380 matches per season $\times$ 5 seasons = 1,900 completed fixtures. Opta has been the single official data provider to the Premier League throughout this period, guaranteeing uniform measurement definitions.

---

### ADR-002: Chronological Partitioning Strategy
- **Status:** **ACCEPTED**
- **Context:** Football match prediction exhibits temporal autocorrelation and tactical drift. Standard random K-fold cross-validation suffers from lookahead leakage.
- **Decision:** Partitions are strictly chronological, with warm-up cold-start isolation enforced:
  - **Training Partition (Seasons 1–3):** `2019/20`, `2020/21`, `2021/22` (1,140 gross fixtures; exactly $N_{\text{train}} = 1,002$ modelling-ready observations).
  - **Validation Partition (Season 4):** `2022/23` (380 gross fixtures; exactly $N_{\text{val}} = 370$ modelling-ready observations, used exclusively for $\lambda$ hyperparameter grid searches).
  - **Test Partition (Season 5):** `2023/24` (380 gross fixtures; exactly $N_{\text{test}} = 369$ modelling-ready observations, held out untouched until final evaluation).
- **Rationale:** Reflects genuine sports forecasting where future outcomes are predicted from past history. A full contiguous validation season (2022/23) reflects a complete, balanced home/away round-robin cycle without mid-season fixture imbalance. Total modelling-ready observations sum to $N_{\text{modelling}} = 1,741$ (with 159 warm-up fixtures preserved for history).s.

---

### ADR-003: Symmetric Multinomial Softmax Parameterization
- **Status:** **ACCEPTED**
- **Context:** In classical unregularized multinomial logistic regression, one class $K$ is set as reference ($\boldsymbol{\beta}_K = \mathbf{0}$) to achieve parameter identification. Under regularized estimation ($L_1$ and $L_2$), fixing an arbitrary reference category causes asymmetric shrinkage dependent on the choice of baseline class.
- **Decision:** Model all $K = 3$ classes ($\text{Away Win}, \text{Draw}, \text{Home Win}$) symmetrically using the Softmax link:
  $$p_{ic} = \frac{\exp(\mathbf{x}_i^T \boldsymbol{\beta}_c)}{\sum_{r=1}^3 \exp(\mathbf{x}_i^T \boldsymbol{\beta}_r)}$$
  with identical penalty $\lambda$ applied across all parameter vectors $\boldsymbol{\beta}_1, \boldsymbol{\beta}_2, \boldsymbol{\beta}_3$.
- **Rationale:** Guarantees class-invariant shrinkage and preserves mathematical symmetry across outcomes.

---

### ADR-004: Dual-Source Verification Architecture
- **Status:** **ACCEPTED**
- **Context:** Web scraping APIs can experience schema shifts, rate limits, or page outages. A secondary source is required to validate data integrity.
- **Decision:** Use FBref as the primary source for fixture schedules, timestamps, kickoff times, match reports, and Opta IDs. Use Football-Data.co.uk as the secondary validation source for scores, outcome labels, and shot volume statistics.
- **Verification:** Reconciled 1,900 / 1,900 matches with 100.0% score concordance.

---

### ADR-005: Purging Stadium Attendance Metric
- **Status:** **ACCEPTED**
- **Context:** COVID-19 safety measures forced matches to be played behind closed doors for portions of 2019/20 (post-lockdown Project Restart) and almost the entirety of 2020/21.
- **Decision:** Attendance is completely purged from candidate feature vectors.
- **Rationale:** Prevents structural missingness (440 missing entries) and prevents the model from conflating empty stadium anomalies with tactical team strength.

---

### ADR-006: Promoted-Team Cold-Start Handling Protocol
- **Status:** **ACCEPTED**
- **Context:** Promoted clubs (3 per season, e.g. Luton Town, Sheffield United, Burnley) lack top-flight match data prior to Gameweek 1. Imputing with season-end stats causes lookahead leakage; setting values to zero biases differential predictors.
- **Decision:** Implement the **Synthetic Baseline Prior Protocol**:
  $$\bar{X}_{\text{promoted}, 0}^{(k)} = \frac{1}{3} \sum_{s=1}^3 \bar{X}_{\text{Relegated}, S-s}$$
  with linear transition decay $\bar{X}_t^{(k)} = \frac{m}{k} \bar{X}_{\text{observed}} + (1 - \frac{m}{k}) \bar{X}_0$ until $m \ge k$ top-flight matches have been completed.

---

### ADR-007: Candidate Feature Set Specification ($p = 16$)
- **Status:** **ACCEPTED (Option A Approved by User on 12 October 2026)**
- **Context:** The automated data audit revealed that `soccerdata.FBref.read_team_match_stats` does not support match-level passing (`PrgP`, `1/3`), possession (`PrgC`, `Att Pen`), defense (`Att 3rd`), or advanced goalkeeping (`PSxG+/-`) tables, and FBref strips `xG` from `all_comps` match logs.
- **Decision:** Implement **Option A: The Verified High-Fidelity 16-Predictor Set**. Replace the originally proposed unverified Opta tracking metrics with 100% complete, verified match statistics derived from our audited primary (FBref schedule) and secondary (Football-Data.co.uk) sources.
- **Approved 16 Candidate Predictors:**
  1. `diff_sot_k5`: Differential in rolling 5-match shots on target (Attack volume & threat).
  2. `diff_sotA_k5`: Differential in rolling 5-match conceded shots on target (Defensive concession).
  3. `diff_shots_k5`: Differential in rolling 5-match total shots attempted.
  4. `diff_shotsA_k5`: Differential in rolling 5-match total shots conceded.
  5. `diff_sot_ratio_k5`: Differential in rolling shot accuracy ratio (`SoT / Shots`, guarded against zero denominator).
  6. `diff_shot_dominance_k5`: Differential in territorial shot dominance share (`Shots / (Shots + ShotsA)`, guarded against zero denominator with neutral baseline 0.5).
  7. `diff_corners_k5`: Differential in rolling 5-match corners won (Deep attacking third pressure).
  8. `diff_cornersA_k5`: Differential in rolling 5-match corners conceded.
  9. `diff_fouls_k5`: Differential in rolling 5-match fouls committed (Defensive aggression & disruption).
  10. `diff_yellows_k5`: Differential in rolling 5-match yellow cards (Disciplinary risk).
  11. `diff_goal_diff_k5`: Differential in short-term rolling goal difference (`GF - GA`).
  12. `diff_goal_diff_k10`: Differential in sustained 10-match structural goal difference.
  13. `diff_clean_sheets_k10`: Differential in rolling 10-match clean sheet rate (`GA == 0`).
  14. `diff_ppg_k5`: Differential in rolling 5-match points per game (3 for W, 1 for D, 0 for L).
  15. `diff_venue_ppg_k5`: Symmetrical venue-specific form (Home team PPG over previous 5 home games minus Away team PPG over previous 5 away games).
  16. `diff_rest_days`: Differential in inter-match rest days derived from chronological kickoff timestamps ($\Delta \tau / 86400$).
- **Verified Data Sources:** FBref schedule (kickoff timestamps, scores, venues, referee, game_id) and Football-Data.co.uk (match shots, shots on target, corners, fouls, cards, scores).
- **Methodological Limitations Introduced:**
  - Omits micro-event spatial tracking coordinates (exact pass coordinates, high-press zonal turnovers).
  - Uses shot dominance share and corner kick volume as proxies for territorial control instead of direct tracking possession % and passes into final third.
  - Retains all core statistical properties needed to evaluate MLE vs. Ridge vs. Lasso: multi-collinear clusters ($\text{VIF} > 5$), variance inflation, and coefficient shrinkage dynamics.
- **Warm-Up Policy:** Prefer excluding observations from the modelling-ready dataset where required historical information ($k \ge 10$ overall matches and $k \ge 5$ venue matches) does not exist (resulting in 1,741 modelling-ready matches out of 1,900 total), while preserving an audited record of all 159 warm-up fixtures. Zero imputation or future data substitution is permitted.
- **Empirical Gate 3 Verification:**
  - Feature pipeline implemented in [`src/features.py`](file:///d:/MLE-Lasso/src/features.py).
  - Pre-match dataset generated in [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv) (1,900 rows, 27 cols).
  - Zero temporal lookahead leakage confirmed via 20 automated tests in [`tests/test_features.py`](file:///d:/MLE-Lasso/tests/test_features.py) and [`tests/test_leakage.py`](file:///d:/MLE-Lasso/tests/test_leakage.py).
  - Gate 3 officially passed on 12 October 2026.

---

### ADR-008: Binary Benchmark vs. Primary 3-Class Target
- **Status:** **ACCEPTED**
- **Context:** Match outcome forecasting in association football naturally comprises 3 classes (Home Win, Draw, Away Win). However, binary logistic regression provides an intuitive benchmark for visualizing odds-ratio curves and shrinkage paths.
- **Decision:** The primary target is 3-class multinomial logit ($\mathcal{Y} \in \{H, D, A\}$). Binary logistic regression ($\text{Home Win vs. Non-Home Win}$) will be implemented as an introductory benchmark check to validate solver convergence.

---

### ADR-009: Solvers & Optimization Framework
- **Status:** **ACCEPTED**
- **Context:** Unregularized MLE requires exact Hessian inversion and covariance standard error reporting; Ridge requires smooth quadratic gradient descent; Lasso requires non-smooth coordinate descent or proximal gradient methods.
- **Decision:**
  - **Unregularized MLE:** `statsmodels.discrete.discrete_model.MNLogit` (for Hessian inversion, standard errors, p-values, and VIFs) and `scikit-learn.linear_model.LogisticRegression(penalty=None, solver='lbfgs')`.
  - **Ridge ($L_2$):** `scikit-learn.linear_model.LogisticRegression(penalty='l2', solver='lbfgs')`.
  - **Lasso ($L_1$):** `scikit-learn.linear_model.LogisticRegression(penalty='l1', solver='saga')`.
- **Rationale:** Standard, battle-tested scientific Python implementations with verified mathematical guarantees.

---

### ADR-010: Benchmark Baseline Specifications (Odds & Training Frequencies)
- **Status:** **ACCEPTED**
- **Context:** Standalone Log-Loss and Brier scores require comparative benchmarks to establish whether models achieve meaningful predictive signal over naive heuristics and market consensus.
- **Decision:** Evaluate all models against two formal baseline benchmarks on the Season 2023/24 test set ($N=369$):
  1. **Training Class-Frequency Prior Baseline:** Naive forecast predicting the static marginal distribution computed strictly on the $N=1,002$ training partition ($\hat{p}_{\text{Home}} = 424/1002 \approx 0.4232$, $\hat{p}_{\text{Draw}} = 225/1002 \approx 0.2246$, $\hat{p}_{\text{Away}} = 353/1002 \approx 0.3523$). Testing frequencies or full-sample frequencies are strictly prohibited to prevent lookahead bias (Invariant 5).
  2. **Market-Implied Probability Benchmark:** Consensus betting odds extracted from Football-Data (`AvgH`, `AvgD`, `AvgA` and `B365H`, `B365D`, `B365A`), normalized by removing the bookmaker overround ($\hat{p}_{ic} = \frac{1/O_{ic}}{\sum_k 1/O_{ik}}$), serving as the efficient market benchmark.
- **Rationale:** Guarantees proper framing of model performance and satisfies peer-review standards in sports analytics forecasting.

---

### ADR-011: Temporal Gameweek Block-Bootstrap Resampling Protocol
- **Status:** **ACCEPTED**
- **Context:** Football fixtures within the same matchweek share macroeconomic, tactical, and environmental dependencies. Standard i.i.d. observation-level resampling violates the cluster structure of sports fixtures.
- **Decision:** Execute bootstrap resampling ($B = 1,000$) using **Matchweek Temporal Blocks** drawn from the training partition rather than random individual fixtures.
- **Rationale:** Preserves the cross-sectional correlation of simultaneous matches while computing robust Empirical Bootstrap Standard Errors $\widehat{\text{SE}}(\hat{\beta})$, the Sign Stability Index (SSI), and the Feature Selection Frequency (FSF).

---

### ADR-012: Refined Hypothesis 3 (Parsimony & Predictive Parity)
- **Status:** **ACCEPTED**
- **Context:** The original draft of $H_{3,\text{alt}}$ stated a dogmatic $< 50\%$ feature retention rule without statistical non-inferiority guarantees against Ridge.
- **Decision:** Refine $H_{3,\text{alt}}$ to explicitly pair parsimony with predictive parity:
  $$\text{Lasso isolates a substantially sparser tactical subset } (\le 50\%\text{ of features retained}) \text{ without statistically significant degradation in out-of-sample Log-Loss compared to Ridge}$$
- **Rationale:** Aligns hypothesis testing with established statistical learning principles (Occam's razor / 1-SE rule), evaluating both dimensional sparsity and predictive non-inferiority via the Wilcoxon signed-rank test.

---

### ADR-013: Migration to OpenFootball Public-Domain Data Pipeline
- **Status:** **ACCEPTED**
- **Context:** The original research pipeline utilized match logs from FBref (Opta-derived, subject to web scrapers and restrictive commercial terms of use) and Football-Data.co.uk (proprietary commercial compilation). To guarantee unconditional reproducibility, open science transparency, legal compliance, and suitability for open-access academic publication without licensing ambiguity, the project requires a fully public-domain data foundation.
- **Decision:** Migrate all data acquisition and preparation exclusively to the **OpenFootball England** repository:
  - **Source URL:** `https://github.com/openfootball/england`
  - **License Declaration:** CC0-1.0 Universal Public Domain Dedication (`LICENSE.md` checked 2026-10-13; SHA-256: `36ffd9dc085d529a7e60e1276d73ae5a030b020313e6c5408593a6ae2af39673`).
  - **Target Seasons:** All 5 contiguous EPL seasons: `2019-20`, `2020-21`, `2021-22`, `2022-23`, `2023-24` (exactly 380 fixtures per season $\times$ 5 = 1,900 total matches).
  - **Data Acquired:** Official match date, kickoff time, round/matchday, home club, away club, final scoreline, and 3-class outcome target (`HOME_WIN`, `DRAW`, `AWAY_WIN`).
- **Data Limitations & Predictor Revision:**
  - OpenFootball provides match schedules, dates, and final scorelines, but does **not** provide in-game event statistics (shots, shots on target, corners, fouls, cards, possession) or bookmaker odds.
  - The comparative research question (MLE vs. Ridge vs. Lasso multinomial logistic regression) and 3-class target remain identical.
  - Candidate predictors are revised to result-derived historical performance metrics:
    - 5-match rolling goals scored (`home_goals_scored_k5`, `away_goals_scored_k5`, differential: `diff_goals_scored_k5`)
    - 5-match rolling goals conceded (`home_goals_conceded_k5`, `away_goals_conceded_k5`, differential: `diff_goals_conceded_k5`)
    - 5-match rolling points per game (`home_ppg_k5`, `away_ppg_k5`, differential: `diff_ppg_k5`)
    - 10-match sustained goal difference per game (`home_gd_k10`, `away_gd_k10`, differential: `diff_goal_diff_k10`)
    - 5-match venue-specific form (`home_venue_ppg_k5` for home games, `away_venue_ppg_k5` for away games, differential: `diff_venue_ppg_k5`)
    - Inter-match rest days differential (`diff_rest_days`)
    - Short-term 5-match goal difference differential (`diff_goal_diff_k5`), 10-match clean sheet rate differential (`diff_clean_sheets_k10`), and 5-match win rate differential (`diff_win_rate_k5`).
- **Legacy Source Policy:**
  - Files in `data/raw/fbref/` and `data/raw/football_data/` are marked as **LEGACY** and completely excluded from the active modeling pipeline.
  - No fallback scraping or silent mixing of FBref/Football-Data is permitted. Raw legacy files are preserved solely for auditability.
- **Verification & Invariant Guard:**
  - Raw OpenFootball source files are stored immutably in `data/raw/openfootball/`.
  - Canonical match table stored at `data/processed/epl_matches.csv`.
  - Feature matrix stored at `data/features/dataset_features.csv`.
  - Quality Gate 3 is held in BLOCKED state until feature engineering tests and reports pass. No models are trained in this migration phase.

