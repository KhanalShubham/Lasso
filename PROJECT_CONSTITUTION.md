# Project Constitution: MLE-Lasso Research Study

**Project Name:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Short Name:** `MLE-Lasso`  
**Domain:** Sports Analytics & Statistical Machine Learning  
**Status:** Active Research Project  
**Authoritative Reference:** [RESEARCH_METHODOLOGY_GUIDE.md](file:///d:/MLE-Lasso/RESEARCH_METHODOLOGY_GUIDE.md)  

---

## 1. Constitutional Purpose & Scope

The purpose of this project is to conduct a controlled, mathematically rigorous comparative evaluation of three estimation frameworks for multinomial logistic regression when applied to association football match outcome forecasting:
1. **Classical Maximum Likelihood Estimation (MLE):** Unregularized baseline ($\lambda = 0$).
2. **Ridge Logistic Regression ($L_2$ Regularization):** Shrinkage via Euclidean penalty ($\lambda \|\boldsymbol{\beta}\|_2^2$).
3. **Lasso Logistic Regression ($L_1$ Regularization):** Shrinkage and exact feature selection via sparsity penalty ($\lambda \|\boldsymbol{\beta}\|_1$).

### Non-Negotiable Research Scope
- **Competition:** English Premier League (EPL).
- **Temporal Horizon:** Five contiguous seasons: `2019/20`, `2020/21`, `2021/22`, `2022/23`, and `2023/24`.
- **Target Observation Horizon:** Exactly $N = 1,900$ completed fixtures across the 5 seasons (380 matches per season), yielding $N_{\text{modelling}} = 1,741$ leak-free modelling observations after the audited warm-up isolation ($N_{\text{warmup}} = 159$).
- **Target Outcome:**
  - **Primary Model:** 3-class multinomial match outcome: $\mathcal{Y} \in \{\text{Home Win (H)}, \text{Draw (D)}, \text{Away Win (A)}\}$.
  - **Benchmark Auxiliary:** Binary outcome: $\mathcal{Y}_{\text{binary}} \in \{\text{Home Win (1)}, \text{Non-Home Win (0)}\}$.

---

## 2. Non-Negotiable Methodological Invariants (The "Golden Rules")

Any code, script, analysis, or prompt that violates any of the following six invariants is **strictly invalid**:

### Invariant 1: Absolute Prohibition of Temporal Lookahead Leakage
- **Rule:** Predictor features for a match taking place at kickoff timestamp $t$ may **only** be computed using information available strictly prior to $t$ ($t - j$ where $j \ge 1$).
- **Enforcement:** In-game statistics (goals, shots, cards, possession during the match being predicted) are strictly excluded from the covariate vector $\mathbf{x}_t$.
- **Sorting:** Fixtures must be processed in **strict chronological order of kickoff timestamp**, never by nominal matchweek or season-end tables.

### Invariant 2: Raw Data Immutability
- **Rule:** Raw data files located in `data/raw/` are **read-only and immutable**.
- **Enforcement:** No cleaning, rolling computation, imputation, column removal, or normalization may be performed directly inside `data/raw/`. All transformations must be performed via reproducible pipeline scripts into `data/processed/` or `data/features/`.

### Invariant 3: Purity of Controlled Comparison
- **Rule:** All three models (MLE, Ridge, Lasso) must be evaluated on the **identical feature matrix**, the **identical chronological partitions**, and optimize the **identical categorical cross-entropy loss function**.
- **Enforcement:** Differences in predictive loss, calibration, and coefficient behavior must be attributable solely to the regularization penalty ($\lambda = 0$ vs. $L_2$ vs. $L_1$), not to disparate preprocessing, variable subsets, or data splits.

### Invariant 4: No Silent Methodological Changes or Data Invention
- **Rule:** Never fabricate data, never invent placeholder metrics, and never silently substitute unverified variables.
- **Enforcement:** If a planned metric cannot be obtained from verified data sources, it must be flagged in [DECISIONS.md](file:///d:/MLE-Lasso/DECISIONS.md) and approved before modifying the feature contract.

### Invariant 5: Strict Partition Separation for Preprocessing
- **Rule:** All preprocessing parameters—including feature standardization means ($\hat{\mu}$) and standard deviations ($\hat{\sigma}$), missing-value imputation medians, promoted-team priors, and baseline class-frequency distributions—must be estimated **strictly on the training split**.
- **Enforcement:** Preprocessing parameters must be frozen and applied out-of-sample to validation and test partitions without recalculation. Fitting scalers or calculating priors on the full dataset is a critical violation.

### Invariant 6: Chronological Train-Validation-Test Partitions
- **Rule:** No random K-fold cross-validation across time.
- **Enforcement:** Partitions must follow strict chronological boundaries:
  - **Training Partition (Seasons 2019/20, 2020/21, 2021/22):** 1,140 gross matches; exactly $N_{\text{train}} = 1,002$ modelling-ready observations after excluding 138 cold-start warm-up matches.
  - **Validation / Tuning Partition (Season 2022/23):** 380 gross matches; exactly $N_{\text{val}} = 370$ modelling-ready observations after excluding 10 warm-up matches for promoted clubs, held out strictly for hyperparameter tuning of $\lambda$.
  - **Final Out-of-Sample Test Partition (Season 2023/24):** 380 gross matches; exactly $N_{\text{test}} = 369$ modelling-ready observations after excluding 11 warm-up matches for promoted clubs, held strictly untouched until final evaluation.

---

## 3. Document Authority Hierarchy

When resolving inconsistencies between project documents, the following strict hierarchy of authority governs:

1. **`PROJECT_CONSTITUTION.md` (Highest Authority):** Defines non-negotiable scope, invariants, and constraints.
2. **`RESEARCH_METHODOLOGY_GUIDE.md` (Scientific Authority):** Defines theoretical formulations, research questions, mathematical proofs, experimental designs, and statistical tests.
3. **`QUALITY_GATES.md` (Verification Authority):** Defines gate criteria and acceptance benchmarks required to advance between project stages.
4. **`DECISIONS.md` (Architectural Log):** Documents all accepted Architectural Decision Records (ADRs) and unresolved research questions.
5. **Technical Context Files (`architecture.md`, `dataflow.md`, `dataoverview.md`, `datatypes.md`, `mathlogic.md`, `modelused.md`):** Module-specific technical blueprints implementing the methodology.
6. **Code & Data Scripts (`src/`, `data/`):** Implementation artifacts that must adhere to all higher layers.

---

## 4. Prohibited Actions (Anti-Patterns)

The AI assistant and any contributing researcher must **NEVER**:
1. Run model training before passing Quality Gates 1 through 4.
2. Calculate rolling statistics using window $k=0$ (the current match).
3. Overwrite raw data files with transformed datasets.
4. Use random shuffle resamples for time-series football data.
5. Impute promoted clubs' missing stats using their end-of-season performance.
6. Report synthetic or simulated experimental numbers as verified results.
7. Change model hyperparameters or feature counts without documenting the rationale in [DECISIONS.md](file:///d:/MLE-Lasso/DECISIONS.md).
