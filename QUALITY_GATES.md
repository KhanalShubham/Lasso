# Research Quality Gates & Verification Framework (QUALITY_GATES.md)

**Project:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Status:** Operational  
**Authority:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md)  

---

## Quality Gate Lifecycle Overview

To maintain academic rigor and prevent cascading methodological errors, the project is structured into **six mandatory quality gates**. No phase of work may begin until the preceding gate has passed and received documented verification evidence.

```mermaid
graph LR
    G1["Gate 1: Context & Scope"] --> G2["Gate 2: Raw Data Collection & Audit"]
    G2 --> G3["Gate 3: Feature Engineering & Leakage Validation"]
    G3 --> G4["Gate 4: Chronological Partitions & Preprocessing"]
    G4 --> G5["Gate 5: MLE, Ridge & Lasso Implementation"]
    G5 --> G6["Gate 6: Evaluation, Statistical Tests & Reporting"]
```

---

## Gate 1: Context and Scope Verified

- **Objective:** Establish the source of truth, document authority, ADR log, and non-negotiable research invariants.
- **Prerequisites:** Workspace inspected, context audit complete.
- **Exit Criteria / Checklists:**
  - [x] `PROJECT_CONSTITUTION.md` created with non-negotiable scope (EPL 2019/20–2023/24, 1,900 matches).
  - [x] Document authority hierarchy explicitly defined.
  - [x] Methodological invariants (no lookahead leakage, raw data immutability, partition purity) codified.
  - [x] Unresolved decisions recorded in `DECISIONS.md`.
  - [x] Agent rules in `AGENTS.md` established.
- **Verification Evidence:** Context engineering audit completed on 2026-10-12.
- **Status:** **PASSED**

---

## Gate 2: Raw Data Collected and Audited

- **Objective:** Ingest primary and secondary match data for all five seasons, verify row counts, check for duplicates, and reconcile fixtures across sources.
- **Prerequisites:** Gate 1 passed.
- **Exit Criteria / Checklists:**
  - [x] Exactly 1,900 matches collected across 5 seasons (380 matches/season verified).
  - [x] Raw data saved in `data/raw/fbref/` and `data/raw/football_data/`.
  - [x] Raw data files preserved completely immutable and unmodified.
  - [x] Zero duplicate fixtures in primary and secondary datasets.
  - [x] 1,900 / 1,900 fixtures reconciled between FBref and Football-Data with 100.0% score concordance.
  - [x] Detailed audit report generated in `data/audit/data_audit_report.md`.
  - [x] Candidate predictor availability audited and documented in `data/audit/feature_availability_audit.csv`.
- **Verification Evidence:**
  - Audit report: [`data/audit/data_audit_report.md`](file:///d:/MLE-Lasso/data/audit/data_audit_report.md)
  - Fixture concordance: [`data/audit/fixture_matching_audit.csv`](file:///d:/MLE-Lasso/data/audit/fixture_matching_audit.csv) (1,900/1,900 PASS)
- **Status:** **PASSED**

---

## Gate 3: Feature Engineering Validated for Temporal Leakage

- **Objective:** Construct the 16 pre-match differential predictors ($\Delta X$) strictly from historical match data without temporal lookahead bias.
- **Prerequisites:** Gate 2 passed; ADR-007 (Option A) explicitly approved by user on 12 October 2026.
- **Exit Criteria / Checklists:**
  - [x] Canonical match order is strictly chronological by kickoff timestamp, not Gameweek round number.
  - [x] Moving average operator strictly excludes current match $t$ ($j \ge 1$ for all summation indices).
  - [x] In-match events (goals, cards, shots during the 90 minutes of the match being predicted) are strictly excluded from the covariate vector.
  - [x] Differential calculation follows $\Delta X = X_{\text{Home}} - X_{\text{Away}}$.
  - [x] Warm-up observations ($N = 159$) are explicitly audited and separated from modelling-ready rows ($N = 1,741$).
  - [x] Unit tests verify that mutating current/future match values does not alter pre-match feature vectors (leakage assertion test).
- **Verification Command:** `python -m pytest tests/test_features.py tests/test_leakage.py -v` (20/20 passed)
- **Deliverables Produced:**
  - Feature script: [`src/features.py`](file:///d:/MLE-Lasso/src/features.py)
  - Feature dataset: [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv) (1,900 rows, 27 cols)
  - Feature audits: [`data/audit/feature_availability_audit.csv`](file:///d:/MLE-Lasso/data/audit/feature_availability_audit.csv), [`data/audit/feature_missingness_audit.csv`](file:///d:/MLE-Lasso/data/audit/feature_missingness_audit.csv)
  - Comprehensive report: [`data/audit/feature_engineering_report.md`](file:///d:/MLE-Lasso/data/audit/feature_engineering_report.md)
  - Test suites: [`tests/test_features.py`](file:///d:/MLE-Lasso/tests/test_features.py), [`tests/test_leakage.py`](file:///d:/MLE-Lasso/tests/test_leakage.py)
- **Status:** **PASSED**

---

## Gate 4: Chronological Validation and Preprocessing Verified

- **Objective:** Partition the dataset into strict chronological splits and verify out-of-sample preprocessing isolation.
- **Prerequisites:** Gate 3 passed.
- **Exit Criteria / Checklists:**
  - [x] Training partition: Seasons 2019/20, 2020/21, 2021/22 (1,140 gross matches; $N_{\text{train}} = 1,002$ modelling-ready observations).
  - [x] Validation partition: Season 2022/23 (380 gross matches; $N_{\text{val}} = 370$ modelling-ready observations, strictly held out for $\lambda$ tuning).
  - [x] Test partition: Season 2023/24 (380 gross matches; $N_{\text{test}} = 369$ modelling-ready observations, strictly locked and unread until final evaluation).
  - [x] Standardization means ($\hat{\mu}$) and standard deviations ($\hat{\sigma}$) fitted **strictly on training split**.
  - [x] Imputation medians and class-frequency priors computed strictly on training split ($N=1,002$).
  - [x] Collinearity diagnostics (Variance Inflation Factors, VIF) computed on training split and documented.
- **Verification Command:** `pytest tests/test_preprocessing.py` (6/6 passed)
- **Deliverables Produced:**
  - Splitting pipeline: [`src/data_split.py`](file:///d:/MLE-Lasso/src/data_split.py)
  - Processed splits: `data/processed/train_data.csv`, `val_data.csv`, `test_data.csv`
  - Frozen standardizer: `data/processed/scalers.joblib`
  - Collinearity reports: [`reports/vif_summary.csv`](file:///d:/MLE-Lasso/reports/vif_summary.csv), [`reports/vif_summary.md`](file:///d:/MLE-Lasso/reports/vif_summary.md)
  - Test suite: [`tests/test_preprocessing.py`](file:///d:/MLE-Lasso/tests/test_preprocessing.py)
- **Status:** **PASSED**

---

## Gate 5: MLE, Ridge, and Lasso Implementations Validated

- **Objective:** Calibrate unregularized MLE, Ridge ($L_2$), and Lasso ($L_1$) multinomial logistic regression models on the training split and optimize regularization parameters on the validation split.
- **Prerequisites:** Gate 4 passed.
- **Exit Criteria / Checklists:**
  - [x] Unregularized MLE converges without singular matrix exceptions; condition number of Hessian documented.
  - [x] Ridge ($L_2$) cross-entropy loss convex optimization verified with `lbfgs` solver.
  - [x] Lasso ($L_1$) sparsity verified with `saga` solver; coordinate shrinkage path traced.
  - [x] Optimal regularization parameters ($\lambda_{\text{Ridge}}^* = 0.0886$, $\lambda_{\text{Lasso}}^* = 0.0127$) selected via minimum Log-Loss on Season 2022/23 validation partition.
  - [x] Convergence tolerances, solver iterations, and parameter vectors documented.
- **Verification Command:** `pytest tests/test_models.py` (4/4 passed)
- **Deliverables Produced:**
  - Estimators: `src/models/mle_model.py`, `ridge_model.py`, `lasso_model.py`
  - Tuning engine: [`src/models/train_and_tune.py`](file:///d:/MLE-Lasso/src/models/train_and_tune.py)
  - Trained models: `data/processed/fitted_models.joblib`
  - Tuning report: [`reports/hyperparameter_tuning.csv`](file:///d:/MLE-Lasso/reports/hyperparameter_tuning.csv), [`reports/tuning_summary.md`](file:///d:/MLE-Lasso/reports/tuning_summary.md)
  - Tuning figures: [`reports/figures/fig6_hyperparameter_tuning.png`](file:///d:/MLE-Lasso/reports/figures/fig6_hyperparameter_tuning.png), [`reports/figures/lasso_shrinkage_path.png`](file:///d:/MLE-Lasso/reports/figures/lasso_shrinkage_path.png)
  - Test suite: [`tests/test_models.py`](file:///d:/MLE-Lasso/tests/test_models.py)
- **Status:** **PASSED**

---

## Gate 6: Final Evaluation, Statistical Comparisons, and Reporting Completed

- **Objective:** Evaluate final models on the untouched Season 2023/24 test set, execute formal hypothesis tests, and generate publication-ready tables and figures.
- **Prerequisites:** Gate 5 passed.
- **Exit Criteria / Checklists:**
  - [x] Primary probabilistic metrics evaluated: Multi-class Log-Loss, Brier Score, Expected Calibration Error (ECE).
  - [x] Classification metrics evaluated: Accuracy, Macro-F1, Multi-class ROC-AUC.
  - [x] Hypothesis $H_1$ tested: Gameweek-level paired Wilcoxon signed-rank test on Log-Loss ($\alpha = 0.05$ with Bonferroni correction to $\alpha = 0.025$).
  - [x] Hypothesis $H_2$ tested: Block-bootstrap resampling ($B = 1,000$) computing Empirical Bootstrap Standard Errors and Sign Stability Index (SSI).
  - [x] Hypothesis $H_3$ tested: Lasso Feature Selection Frequency (FSF) quantifying sparsity and identifying the Core Tactical Feature Set ($\text{FSF} \ge 80\%$).
  - [x] Classification divergence tested via McNemar's test with continuity correction.
  - [x] Final research report and visualization figures exported to `reports/`.
- **Verification Command:** `python src/evaluation.py` and `pytest tests/test_evaluation.py` (6/6 passed; 36/36 repository tests passed)
- **Deliverables Produced:**
  - Metrics table: [`reports/test_metrics.csv`](file:///d:/MLE-Lasso/reports/test_metrics.csv)
  - Wilcoxon tests: [`reports/wilcoxon_results.csv`](file:///d:/MLE-Lasso/reports/wilcoxon_results.csv)
  - McNemar tests: [`reports/mcnemar_results.csv`](file:///d:/MLE-Lasso/reports/mcnemar_results.csv)
  - Bootstrap stability: [`reports/bootstrap_coefficients.csv`](file:///d:/MLE-Lasso/reports/bootstrap_coefficients.csv)
  - Visual figures: [`reports/figures/calibration_curves.png`](file:///d:/MLE-Lasso/reports/figures/calibration_curves.png), [`reports/figures/bootstrap_stability.png`](file:///d:/MLE-Lasso/reports/figures/bootstrap_stability.png)
  - Final report: [`reports/final_evaluation_report.md`](file:///d:/MLE-Lasso/reports/final_evaluation_report.md)
  - Test suite: [`tests/test_evaluation.py`](file:///d:/MLE-Lasso/tests/test_evaluation.py)
- **Status:** **PASSED**
