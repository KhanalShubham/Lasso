# Research Task Master Tracker (TASKS.md)

**Project:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Status:** Living Task Board  
**Governing Gates:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md)  

---

## Task Board Summary

| Task ID | Task Description | Gate | Status | Deliverable Artifact |
| :---: | :--- | :---: | :---: | :--- |
| **TSK-001** | Inspect project context & audit existing files | Gate 1 | **DONE** | Context audit report |
| **TSK-002** | Establish source-of-truth governance framework | Gate 1 | **DONE** | `PROJECT_CONSTITUTION.md`, `DECISIONS.md`, `QUALITY_GATES.md`, `AGENTS.md` |
| **TSK-003** | Ingest raw Football-Data Premier League CSVs (5 seasons) | Gate 2 | **DONE** | `data/raw/football_data/` (1,900 rows) |
| **TSK-004** | Retrieve FBref fixture schedules & match reports (5 seasons) | Gate 2 | **DONE** | `data/raw/fbref/` (1,900 rows) |
| **TSK-005** | Reconcile cross-source fixtures & team name normalization | Gate 2 | **DONE** | `data/audit/fixture_matching_audit.csv` |
| **TSK-006** | Audit 16 candidate predictor availability against raw APIs | Gate 2 | **DONE** | `data/audit/feature_availability_audit.csv`, `data_audit_report.md` |
| **TSK-007** | Resolve ADR-007: Confirm 16-Feature Set Specification | Gate 3 | **DONE** | Approved ADR-007 (Option A) in `DECISIONS.md` |
| **TSK-008** | Implement chronological rolling moving-average operator | Gate 3 | **DONE** | `src/features.py` |
| **TSK-009** | Establish warm-up isolation policy ($N=159$) | Gate 3 | **DONE** | `src/features.py`, `feature_missingness_audit.csv` |
| **TSK-010** | Construct pre-match differential feature matrix ($p = 16$) | Gate 3 | **DONE** | `data/features/dataset_features.csv` ($N=1,900$) |
| **TSK-011** | Unit test feature matrix for zero temporal lookahead leakage | Gate 3 | **DONE** | `tests/test_features.py`, `tests/test_leakage.py` (20/20 PASS) |
| **TSK-012** | Implement chronological train/validation/test splitter | Gate 4 | **DONE** | `src/data_split.py` |
| **TSK-013** | Fit and freeze training standardizers & imputation medians | Gate 4 | **DONE** | `data/processed/scalers.joblib` |
| **TSK-014** | Compute Variance Inflation Factors (VIF) & collinearity clusters | Gate 4 | **DONE** | `reports/vif_summary.csv` |
| **TSK-015** | Implement unregularized MLE Multinomial Logistic Regression | Gate 5 | **DONE** | `src/models/mle_model.py` |
| **TSK-016** | Implement Ridge ($L_2$) Multinomial Logistic Regression | Gate 5 | **DONE** | `src/models/ridge_model.py` |
| **TSK-017** | Implement Lasso ($L_1$) Multinomial Logistic Regression | Gate 5 | **DONE** | `src/models/lasso_model.py` |
| **TSK-018** | Tune $\lambda_{\text{Ridge}}$ and $\lambda_{\text{Lasso}}$ on Season 2022/23 validation set | Gate 5 | **DONE** | `reports/hyperparameter_tuning.csv` |
| **TSK-019** | Out-of-sample evaluation on Season 2023/24 test set | Gate 6 | **NOT STARTED** | `reports/test_metrics.csv` |
| **TSK-020** | Run Wilcoxon signed-rank test on gameweek log-loss (RQ1) | Gate 6 | **NOT STARTED** | `reports/wilcoxon_results.csv` |
| **TSK-021** | Run McNemar's test for classification divergence | Gate 6 | **NOT STARTED** | `reports/mcnemar_results.csv` |
| **TSK-022** | Execute block-bootstrap resampling for coefficient stability (RQ2/RQ3) | Gate 6 | **NOT STARTED** | `reports/bootstrap_coefficients.csv` |
| **TSK-023** | Trace Lasso regularization path and order of variable elimination | Gate 6 | **NOT STARTED** | `reports/figures/lasso_shrinkage_path.png` |
| **TSK-024** | Generate probability calibration curves & reliability diagrams (RQ4) | Gate 6 | **NOT STARTED** | `reports/figures/calibration_curves.png` |
| **TSK-025** | Compile final comparative dissertation analysis & thesis tables | Gate 6 | **NOT STARTED** | `reports/final_thesis_report.md` |

---

## Detailed Task Verification History

### TSK-001 through TSK-006 (Completed during Gate 1 & Gate 2)
- **Completed:** 2026-10-12
- **Evidence:**
  - Raw files saved in `data/raw/fbref/` and `data/raw/football_data/`.
  - Exactly 1,900 matches verified across both datasets.
  - Zero duplicates detected.
  - Cross-source fixture matching passed 1,900 / 1,900 with 100.0% score concordance.
  - Generated audit report: [`data/audit/data_audit_report.md`](file:///d:/MLE-Lasso/data/audit/data_audit_report.md).

### TSK-007 through TSK-011 (Completed during Gate 3)
- **Completed:** 2026-10-12
- **Evidence:**
  - User approved ADR-007 (Option A: 16 candidate predictors).
  - Implemented feature pipeline in [`src/features.py`](file:///d:/MLE-Lasso/src/features.py).
  - Generated [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv) (1,900 rows, 27 columns).
  - Validated warm-up exclusions: exactly 1,741 modelling-ready fixtures (0 nulls across all 16 predictors) and 159 warm-up fixtures.
  - Executed automated tests: [`tests/test_features.py`](file:///d:/MLE-Lasso/tests/test_features.py) and [`tests/test_leakage.py`](file:///d:/MLE-Lasso/tests/test_leakage.py) (20/20 tests passed).
  - Published comprehensive Gate 3 report: [`data/audit/feature_engineering_report.md`](file:///d:/MLE-Lasso/data/audit/feature_engineering_report.md).

### TSK-012 through TSK-014 (Completed during Gate 4)
- **Completed:** 2026-10-13
- **Evidence:**
  - Implemented chronological splitter in [`src/data_split.py`](file:///d:/MLE-Lasso/src/data_split.py).
  - Partition row counts verified: Train = 1,002, Val = 370, Test = 369 (Total = 1,741).
  - Fitted and serialized frozen StandardScaler metadata to `data/processed/scalers.joblib`.
  - Saved split datasets: `train_data.csv`, `val_data.csv`, `test_data.csv` (and unscaled variants).
  - Computed training class frequencies: Home = 42.32%, Draw = 22.46%, Away = 35.23%.
  - Computed training VIF summary in [`reports/vif_summary.csv`](file:///d:/MLE-Lasso/reports/vif_summary.csv) and [`reports/vif_summary.md`](file:///d:/MLE-Lasso/reports/vif_summary.md).
  - Executed automated test suite: [`tests/test_preprocessing.py`](file:///d:/MLE-Lasso/tests/test_preprocessing.py) (6/6 passed).

### TSK-015 through TSK-018 (Completed during Gate 5)
- **Completed:** 2026-10-13
- **Evidence:**
  - Implemented models: [`src/models/mle_model.py`](file:///d:/MLE-Lasso/src/models/mle_model.py), [`src/models/ridge_model.py`](file:///d:/MLE-Lasso/src/models/ridge_model.py), [`src/models/lasso_model.py`](file:///d:/MLE-Lasso/src/models/lasso_model.py).
  - Executed 20-point logarithmic grid search in [`src/models/train_and_tune.py`](file:///d:/MLE-Lasso/src/models/train_and_tune.py) across $\lambda \in [10^{-4}, 10^4]$.
  - Saved tuning grid diagnostics: [`reports/hyperparameter_tuning.csv`](file:///d:/MLE-Lasso/reports/hyperparameter_tuning.csv).
  - Optimal Ridge selected: $\lambda^* = 0.0886$ ($C = 0.005633$), Val Log-Loss = 1.0016, Brier = 0.5984, $\text{df}(\lambda) = 8.73$.
  - Optimal Lasso selected: $\lambda^* = 0.0127$ ($C = 0.078319$), Val Log-Loss = 1.0003, Brier = 0.5978, Retained = 9 of 16 features (56.2%).
  - Serialized trained model bundle: `data/processed/fitted_models.joblib`.
  - Generated figures: [`reports/figures/fig6_hyperparameter_tuning.png`](file:///d:/MLE-Lasso/reports/figures/fig6_hyperparameter_tuning.png), [`reports/figures/lasso_shrinkage_path.png`](file:///d:/MLE-Lasso/reports/figures/lasso_shrinkage_path.png).
  - Published tuning summary: [`reports/tuning_summary.md`](file:///d:/MLE-Lasso/reports/tuning_summary.md).
  - Executed automated test suite: [`tests/test_models.py`](file:///d:/MLE-Lasso/tests/test_models.py) (4/4 passed; 30/30 workspace tests pass).

### TSK-019 through TSK-025 (Completed during Gate 6)
- **Completed:** 2026-10-13
- **Evidence:**
  - Evaluated all candidate models and benchmarks on held-out Season 2023/24 test partition ($N = 369$, 38 Gameweeks).
  - Out-of-sample Log-Loss: Market Avg (0.9094) < Bet365 (0.9091) < **Lasso (0.9573)** < **Ridge (0.9636)** < **MLE (0.9846)** < Frequency Prior (1.0548).
  - Brier score: Market Avg (0.5332) < **Lasso (0.5658)** < **Ridge (0.5703)** < **MLE (0.5810)** < Prior (0.6380).
  - Expected Calibration Error (ECE): **Ridge (0.0370)** < Prior (0.0403) < **Lasso (0.0467)** < Market Avg (0.0528) < **MLE (0.0568)**.
  - Accuracy: Market Avg (58.54%) > **Lasso (56.37%)** > **MLE (56.10%)** > **Ridge (55.83%)** > Prior (46.34%).
  - Wilcoxon signed-rank test across 38 Gameweeks:
    - MLE vs. Ridge: $W = 489.0$, $p = 0.0436$ (one-sided; mean weekly reduction +0.0210 nats).
    - MLE vs. Lasso: $W = 495.0$, $p = 0.0359$ (one-sided; mean weekly reduction +0.0275 nats).
    - Ridge vs. Lasso ($H_3$ Parity): $W = 293.0$, $p = 0.2675$ (two-sided, non-significant; delta = 0.0063 nats).
  - McNemar classification divergence tests: zero statistically significant divergence between MLE, Ridge, and Lasso ($p = 1.000$ and $p = 0.8231$).
  - Block-bootstrap resampling ($B = 1,000$ matchweek blocks):
    - Ridge reduced standard errors across predictors by an average of 74.8% relative to unregularized MLE (e.g. `diff_shot_dominance_k5` SE dropped from 0.7073 to 0.0256: 96.4% variance reduction).
    - Identified Core Tactical Features ($\text{FSF} \ge 80\%$): `diff_goal_diff_k10` (99.9%), `diff_shotsA_k5` (97.0%), `diff_fouls_k5` (86.6%).
  - Deliverable artifacts generated:
    - [`reports/test_metrics.csv`](file:///d:/MLE-Lasso/reports/test_metrics.csv)
    - [`reports/wilcoxon_results.csv`](file:///d:/MLE-Lasso/reports/wilcoxon_results.csv)
    - [`reports/mcnemar_results.csv`](file:///d:/MLE-Lasso/reports/mcnemar_results.csv)
    - [`reports/bootstrap_coefficients.csv`](file:///d:/MLE-Lasso/reports/bootstrap_coefficients.csv)
    - [`reports/figures/calibration_curves.png`](file:///d:/MLE-Lasso/reports/figures/calibration_curves.png)
    - [`reports/figures/bootstrap_stability.png`](file:///d:/MLE-Lasso/reports/figures/bootstrap_stability.png)
    - [`reports/final_evaluation_report.md`](file:///d:/MLE-Lasso/reports/final_evaluation_report.md)
  - Executed automated test suite: [`tests/test_evaluation.py`](file:///d:/MLE-Lasso/tests/test_evaluation.py) (6/6 passed; 36/36 repository tests passed).

### Milestone Completion & Manuscript Sync
- **Status:** **ALL 6 RESEARCH QUALITY GATES PASSED (100% COMPLETE)**
- **Next Task:** Populate empirical LaTeX tables, numbers, and figure references in `manuscript.tex` for dissertation submission.
