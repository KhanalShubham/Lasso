# Hyperparameter Tuning & Model Validation Report (Gate 5)

**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 5  
**Validation Partition:** Season 2022/23 ($N = 370$ matches)  
**Training Partition:** Seasons 2019/20 - 2021/22 ($N = 1,002$ matches)  

---

## 1. Validation Performance Benchmark Summary

| Model / Benchmark | Penalty $\lambda$ | Sklearn $C$ | Validation Log-Loss | Validation Brier | Validation Accuracy | Notes / Sparsity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Frequency Prior** | — | — | `1.0581` | `0.6391` | `0.4838` | Static training marginal rates |
| **Market Average Odds** | — | — | `0.9684` | `0.5755` | `0.5595` | Consensus betting benchmark |
| **Unregularized MLE** | $0$ | $\infty$ | `1.0146` | `0.6048` | `0.5459` | Full unconstrained model |
| **Ridge ($L_2$) Optimal** | **`0.0886`** | **`0.005633`** | **`1.0016`** | **`0.5984`** | **`0.5378`** | $\text{df}(\lambda) = 8.73$ |
| **Lasso ($L_1$) Optimal** | **`0.0127`** | **`0.078319`** | **`1.0003`** | **`0.5978`** | **`0.5324`** | **9 / 16 features** (56.2%) |

---

## 2. Optimal Lasso Feature Selection

The optimal Lasso model ($\lambda^* = 0.0127$) retained **9 of 16 candidate predictors**:

- **Retained Features:**
  - `diff_shots_k5`
  - `diff_shotsA_k5`
  - `diff_shot_dominance_k5`
  - `diff_corners_k5`
  - `diff_fouls_k5`
  - `diff_yellows_k5`
  - `diff_goal_diff_k5`
  - `diff_goal_diff_k10`
  - `diff_rest_days`

- **Eliminated (Shrunk to Exactly Zero):**
  - `diff_sot_k5`
  - `diff_sotA_k5`
  - `diff_sot_ratio_k5`
  - `diff_cornersA_k5`
  - `diff_clean_sheets_k10`
  - `diff_ppg_k5`
  - `diff_venue_ppg_k5`

---

## 3. Deliverable Artifacts Generated

1. `data/processed/fitted_models.joblib` (Serialized trained estimators)
2. `reports/hyperparameter_tuning.csv` (Full 20-point grid search diagnostics)
3. `reports/figures/fig6_hyperparameter_tuning.png` (Validation loss & parameter complexity curves)
4. `reports/figures/lasso_shrinkage_path.png` (Lasso shrinkage paths and variable elimination)
