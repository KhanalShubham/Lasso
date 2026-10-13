# Hyperparameter Tuning & Model Validation Report (Gate 5)

**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 5  
**Validation Partition:** Season 2022/23 ($N = 370$ matches)  
**Training Partition:** Seasons 2019/20 - 2021/22 ($N = 1,002$ matches)  

---

## 1. Validation Performance Benchmark Summary

| Model / Benchmark | Penalty $\lambda$ | Sklearn $C$ | Validation Log-Loss | Validation Brier | Validation Accuracy | Notes / Sparsity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Frequency Prior** | — | — | `1.0581` | `0.6391` | `0.4838` | Static training marginal rates |
| **Market Average Odds** | — | — | `nan` | `nan` | `nan` | Consensus betting benchmark |
| **Unregularized MLE** | $0$ | $\infty$ | `1.0291` | `0.6164` | `0.5000` | Full unconstrained model |
| **Ridge ($L_2$) Optimal** | **`0.0886`** | **`0.005633`** | **`1.0153`** | **`0.6084`** | **`0.5189`** | $\text{df}(\lambda) = 5.90$ |
| **Lasso ($L_1$) Optimal** | **`0.0336`** | **`0.029704`** | **`1.0140`** | **`0.6076`** | **`0.5108`** | **2 / 8 features** (25.0%) |

---

## 2. Optimal Lasso Feature Selection

The optimal Lasso model ($\lambda^* = 0.0336$) retained **2 of 8 candidate predictors**:

- **Retained Features:**
  - `diff_goals_conceded_k5`
  - `diff_goal_diff_k10`

- **Eliminated (Shrunk to Exactly Zero):**
  - `diff_goals_scored_k5`
  - `diff_ppg_k5`
  - `diff_venue_ppg_k5`
  - `diff_rest_days`
  - `diff_clean_sheets_k10`
  - `diff_win_rate_k5`

---

## 3. Deliverable Artifacts Generated

1. `data/processed/fitted_models.joblib` (Serialized trained estimators)
2. `reports/hyperparameter_tuning.csv` (Full 20-point grid search diagnostics)
3. `reports/figures/fig6_hyperparameter_tuning.png` (Validation loss & parameter complexity curves)
4. `reports/figures/lasso_shrinkage_path.png` (Lasso shrinkage paths and variable elimination)
