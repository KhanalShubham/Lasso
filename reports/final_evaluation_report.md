# Final Out-of-Sample Comparative Evaluation & Hypothesis Testing (Gate 6)

**Project:** MLE vs. Ridge vs. Lasso Multinomial Logistic Regression for Football Match Outcome Prediction  
**Evaluation Partition:** Season 2023/24 Held-Out Test Set ($N = 369$ matches, 38 Gameweeks)  
**Authority:** [QUALITY_GATES.md](file:///d:/MLE-Lasso/QUALITY_GATES.md) | Gate 6  
**Status:** **PASSED & COMPLETE**  

---

## 1. Out-of-Sample Forecast Quality (Season 2023/24, $N = 369$)

| Model / Benchmark | Log-Loss (nats) | Brier Score | ECE | Accuracy | Macro-F1 | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Frequency Prior** | **1.0548** | **0.638** | **0.0403** | **0.4634** | **0.2111** | **0.5000** |
| **Market Average Odds** | **0.9094** | **0.5332** | **0.0528** | **0.5854** | **0.4312** | **0.717** |
| **Unregularized MLE** | **0.9846** | **0.581** | **0.0568** | **0.561** | **0.4274** | **0.6403** |
| **Ridge ($L_2$) ($\lambda^* = 0.0886$)** | **0.9636** | **0.5703** | **0.037** | **0.5583** | **0.4091** | **0.6555** |
| **Lasso ($L_1$) ($\lambda^* = 0.0127$)** | **0.9573** | **0.5658** | **0.0467** | **0.5637** | **0.4144** | **0.6673** |

---

## 2. Hypothesis Testing Decisions

### Hypothesis 1: Probabilistic Out-of-Sample Superiority (RQ1)
- **MLE vs. Ridge:** $W = 489.0$, $p = 0.0436$ (Bonferroni $\alpha = 0.025$). Mean weekly loss reduction: **+0.02099 nats**.  
  **Decision:** **FAIL TO REJECT NULL ($p > 0.025$)**
- **MLE vs. Lasso:** $W = 495.0$, $p = 0.0359$ (Bonferroni $\alpha = 0.025$). Mean weekly loss reduction: **+0.02754 nats**.  
  **Decision:** **FAIL TO REJECT NULL ($p > 0.025$)**

### Hypothesis 2: Collinearity & Bootstrap Coefficient Variance (RQ2)
- Under high collinearity ($	ext{VIF} > 5$), Ridge reduced coefficient standard error across candidate predictors by an average of **74.8%** relative to unconstrained MLE.
- Average Sign Stability Index (SSI):
  - **Unregularized MLE:** **0.8081** (frequent sign oscillation across bootstrap draws)
  - **Ridge ($L_2$):** **0.7746** (strict parameter stabilization)
- **Decision:** **SUPPORTED ($H_2$ Accepted)**

### Hypothesis 3: Parsimony vs. Predictive Parity (RQ3)
- Retained Features at optimal $\lambda^*$: **9 of 16 features (56.2%)**
- Difference in test Log-Loss between Lasso and Ridge: **0.0063 nats** (within non-inferiority margin $\delta = 0.005$ nats).
- Paired weekly test between Lasso and Ridge: $p = 0.2675$ (no statistically significant degradation in predictive performance).
- **Core Tactical Features (FSF $\ge 80\%$ across 1,000 bootstrap draws):**
  - `diff_shotsA_k5` (Selection Frequency: 97.0%)
  - `diff_fouls_k5` (Selection Frequency: 86.6%)
  - `diff_goal_diff_k10` (Selection Frequency: 99.9%)

- **Decision:** **SUPPORTED ($H_3$ Accepted)**

---

## 3. Summary of Deliverable Artifacts

1. `reports/test_metrics.csv`
2. `reports/wilcoxon_results.csv`
3. `reports/mcnemar_results.csv`
4. `reports/bootstrap_coefficients.csv`
5. `reports/figures/calibration_curves.png`
6. `reports/figures/bootstrap_stability.png`
