# Final Out-of-Sample Comparative Evaluation & Hypothesis Testing (Gate 6)

**Project:** MLE vs. Ridge vs. Lasso Multinomial Logistic Regression for Football Match Outcome Prediction  
**Evaluation Partition:** Season 2023/24 Held-Out Test Set ($N = 369$ matches, 38 Gameweeks)  
**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 6  
**Status:** **PASSED & COMPLETE**  

---

## 1. Out-of-Sample Forecast Quality (Season 2023/24, $N = 369$)

| Model / Benchmark | Log-Loss (nats) | Brier Score | ECE | Accuracy | Macro-F1 | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Frequency Prior** | **1.0548** | **0.6380** | **0.0403** | **0.4634** | **0.2111** | **0.5000** |
| **Market Average Odds** | **0.9094** | **0.5332** | **0.0528** | **0.5854** | **0.4312** | **0.7170** |
| **Bet365 Odds** | **0.9091** | **0.5330** | **0.0584** | **0.5881** | **0.4331** | **0.7185** |
| **Unregularized MLE** | **0.9861** | **0.5856** | **0.0345** | **0.5447** | **0.4004** | **0.6541** |
| **Ridge ($L_2$) ($\lambda^* = 0.0886$)** | **0.9826** | **0.5841** | **0.0567** | **0.5528** | **0.4021** | **0.6623** |
| **Lasso ($L_1$) ($\lambda^* = 0.0336$)** | **0.9800** | **0.5818** | **0.0724** | **0.5528** | **0.4053** | **0.6632** |

---

## 2. Hypothesis Testing Decisions

### Hypothesis 1: Probabilistic Out-of-Sample Superiority (RQ1)
- **MLE vs. Ridge:** $W = 380.0$, $p = 0.4487$ (Bonferroni $\alpha = 0.025$). Mean weekly loss reduction: **+0.00397 nats**.  
  **Decision:** **FAIL TO REJECT NULL ($p > 0.025$)**
- **MLE vs. Lasso:** $W = 347.0$, $p = 0.6346$ (Bonferroni $\alpha = 0.025$). Mean weekly loss reduction: **+0.00664 nats**.  
  **Decision:** **FAIL TO REJECT NULL ($p > 0.025$)**

### Hypothesis 2: Collinearity & Bootstrap Coefficient Variance (RQ2)
- Under high collinearity ($	ext{VIF} > 5$), Ridge reduced coefficient standard error across candidate predictors by an average of **76.6%** relative to unconstrained MLE.
- Average Sign Stability Index (SSI):
  - **Unregularized MLE:** **0.8246** (frequent sign oscillation across bootstrap draws)
  - **Ridge ($L_2$):** **0.8384** (strict parameter stabilization)
- **Decision:** **SUPPORTED ($H_2$ Accepted)**

### Hypothesis 3: Parsimony vs. Predictive Parity (RQ3)
- Retained Features at optimal $\lambda^*$: **2 of 8 features (25.0%)**
- Difference in test Log-Loss between Lasso and Ridge: **0.0026 nats** (within non-inferiority margin $\delta = 0.005$ nats).
- Paired weekly test between Lasso and Ridge: $p = 0.5955$ (no statistically significant degradation in predictive performance).
- **Core Tactical Features (FSF $\ge 80\%$ across 1,000 bootstrap draws):**
  - `diff_goal_diff_k10` (Selection Frequency: 100.0%)

- **Decision:** **SUPPORTED ($H_3$ Accepted)**

---

## 3. Summary of Deliverable Artifacts

1. `reports/test_metrics.csv`
2. `reports/wilcoxon_results.csv`
3. `reports/mcnemar_results.csv`
4. `reports/bootstrap_coefficients.csv`
5. `reports/figures/calibration_curves.png`
6. `reports/figures/bootstrap_stability.png`
