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
| **TSK-019** | Out-of-sample evaluation on Season 2023/24 test set | Gate 6 | **DONE** | `reports/test_metrics.csv` |
| **TSK-020** | Run Wilcoxon signed-rank test on gameweek log-loss (RQ1) | Gate 6 | **DONE** | `reports/wilcoxon_results.csv` |
| **TSK-021** | Run McNemar's test for classification divergence | Gate 6 | **DONE** | `reports/mcnemar_results.csv` |
| **TSK-022** | Execute block-bootstrap resampling for coefficient stability (RQ2/RQ3) | Gate 6 | **DONE** | `reports/bootstrap_coefficients.csv` |
| **TSK-023** | Trace Lasso regularization path and order of variable elimination | Gate 6 | **DONE** | `reports/figures/lasso_shrinkage_path.png` |
| **TSK-024** | Generate probability calibration curves & reliability diagrams (RQ4) | Gate 6 | **DONE** | `reports/figures/calibration_curves.png` |
| **TSK-025** | Compile final comparative dissertation analysis & thesis tables | Gate 6 | **DONE** | `reports/final_evaluation_report.md` |
| **TSK-026** | Ingest raw OpenFootball 5-season dataset & verify CC0-1.0 | Gate 2 | **DONE** | `data/raw/openfootball/`, `openfootball_ingestion_manifest.json` |
| **TSK-027** | Canonicalize OpenFootball match table to `epl_matches.csv` (1,900 rows) | Gate 2 | **DONE** | `data/processed/epl_matches.csv` |
| **TSK-028** | Log ADR-013 & update governance docs (provenance, architecture) | Gate 1 | **DONE** | `DECISIONS.md`, `DATA_PROVENANCE.md`, `architecture.md` |
| **TSK-029** | Implement result-derived feature pipeline (`src/features_openfootball.py`) | Gate 3 | **DONE** | `data/features/dataset_features.csv`, `feature_dictionary.md` |
| **TSK-030** | Execute automated test suite for leakage and data quality | Gate 3 | **DONE** | `tests/test_openfootball_pipeline.py` (15/15 PASS) |
| **TSK-031** | Generate migration, feature engineering, and data quality reports | Gate 3 | **DONE** | `openfootball_migration_report.md`, `data_quality_report.md` |
| **TSK-032** | Formal review & user approval to unblock Gate 3 for modeling | Gate 3 | **DONE** | Approved by user on 2026-10-13 |

---

## Detailed Task Verification History

### TSK-026 through TSK-032: OpenFootball Public-Domain Migration & Gate 3 Approval
- **Completed:** 2026-10-13
- **Evidence:**
  - Raw files for all 5 seasons (`2019-20` to `2023-24`) downloaded under CC0-1.0 and verified against SHA-256 hashes in [`data/audit/openfootball_ingestion_manifest.json`](file:///d:/MLE-Lasso/data/audit/openfootball_ingestion_manifest.json).
  - Canonical match table created in [`data/processed/epl_matches.csv`](file:///d:/MLE-Lasso/data/processed/epl_matches.csv) (1,900 rows, 0 nulls, exactly 26 clubs).
  - Logged ADR-013 in [`DECISIONS.md`](file:///d:/MLE-Lasso/DECISIONS.md) and updated [`DATA_PROVENANCE.md`](file:///d:/MLE-Lasso/DATA_PROVENANCE.md).
  - Result-derived feature pipeline implemented in [`src/features_openfootball.py`](file:///d:/MLE-Lasso/src/features_openfootball.py).
  - Pre-match dataset generated in [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv) (1,900 rows, 39 cols; 1,741 modelling-ready with 0 nulls, 159 warm-up rows).
  - Feature dictionary published in [`data/features/feature_dictionary.md`](file:///d:/MLE-Lasso/data/features/feature_dictionary.md).
  - Automated test suite executed: `python -m pytest tests/test_openfootball_pipeline.py -v` (15/15 passed).
  - Formal approval granted by user to proceed across all gates.

### TSK-012 through TSK-014 (Gate 4 Preprocessing & Collinearity)
- **Completed:** 2026-10-13
- **Evidence:**
  - Split dataset: Train = 1,002, Val = 370, Test = 369.
  - Frozen standardizer saved in `data/processed/scalers.joblib`.
  - Collinearity analysis computed on 8 non-singular candidate predictors: condition number $\kappa(\mathbf{X}) = 14.00$, Hessian $\kappa(\mathbf{H}_0) = 195.96$, maximum VIF = 24.63 on `diff_ppg_k5`.
  - Automated test suite executed: `python -m pytest tests/test_preprocessing.py -v` (6/6 passed).

### TSK-015 through TSK-018 (Gate 5 Model Calibration & Tuning)
- **Completed:** 2026-10-13
- **Evidence:**
  - Unregularized MLE: Val Log-Loss = 1.0291, Brier = 0.6164.
  - Ridge ($L_2$): Optimal $\lambda^* = 0.0886$ ($C = 0.005633$), Val Log-Loss = 1.0153, Brier = 0.6084, $\text{df}(\lambda) = 5.90$.
  - Lasso ($L_1$): Optimal $\lambda^* = 0.0336$ ($C = 0.029704$), Val Log-Loss = 1.0140, Brier = 0.6076, Retained = 2 of 8 predictors (`diff_goals_conceded_k5`, `diff_goal_diff_k10`).
  - Serialized model bundle: `data/processed/fitted_models.joblib`.
  - Automated test suite executed: `python -m pytest tests/test_models.py -v` (4/4 passed).

### TSK-019 through TSK-025 (Gate 6 Final Evaluation & Inference)
- **Completed:** 2026-10-13
- **Evidence:**
  - Evaluated on held-out Season 2023/24 test partition ($N = 369$, 38 Gameweeks).
  - Test Log-Loss: Market Avg (0.9094) < Bet365 (0.9091) < **Lasso (0.9800)** < **Ridge (0.9826)** < **MLE (0.9861)** < Prior (1.0548).
  - Test Brier score: Market Avg (0.5332) < **Lasso (0.5818)** < **Ridge (0.5841)** < **MLE (0.5856)** < Prior (0.6380).
  - Test Accuracy: Market Avg (58.54%) > **Ridge (55.28%)** = **Lasso (55.28%)** > **MLE (54.47%)** > Prior (46.34%).
  - Paired weekly Wilcoxon signed-rank tests across 38 Gameweeks:
    - MLE vs. Ridge: $W = 380.0$, $p = 0.4487$ (mean weekly reduction $+0.00397\text{ nats}$).
    - MLE vs. Lasso: $W = 347.0$, $p = 0.6346$ (mean weekly reduction $+0.00664\text{ nats}$).
    - Ridge vs. Lasso: $W = 333.0$, $p = 0.5955$ (mean weekly difference $0.00267\text{ nats}$ in favor of Lasso).
  - McNemar classification divergence tests: zero statistically significant divergence ($p > 0.50$).
  - Block-bootstrap resampling ($B = 1,000$ matchweek blocks):
    - Ridge reduced standard errors across predictors by an average of 76.6% relative to MLE (92.4% reduction on points per game).
    - Sign stability: MLE SSI = 0.8246, Ridge SSI = 0.8384.
    - Core tactical feature with 100% selection frequency: `diff_goal_diff_k10`.
  - Published reports: [`reports/test_metrics.csv`](file:///d:/MLE-Lasso/reports/test_metrics.csv), [`reports/wilcoxon_results.csv`](file:///d:/MLE-Lasso/reports/wilcoxon_results.csv), [`reports/mcnemar_results.csv`](file:///d:/MLE-Lasso/reports/mcnemar_results.csv), [`reports/bootstrap_coefficients.csv`](file:///d:/MLE-Lasso/reports/bootstrap_coefficients.csv), [`reports/final_evaluation_report.md`](file:///d:/MLE-Lasso/reports/final_evaluation_report.md).
  - Automated test suite executed: `python -m pytest tests/test_openfootball_pipeline.py tests/test_preprocessing.py tests/test_models.py tests/test_evaluation.py -v` (31/31 passed).

### Milestone Completion & Manuscript Sync
- **Status:** **ALL 6 RESEARCH QUALITY GATES PASSED (100% COMPLETE)**
- **Next Task:** Git commit and push all OpenFootball pipeline updates, reports, and documentation to GitHub.
