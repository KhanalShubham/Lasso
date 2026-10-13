# Model Specifications & Solvers (modelused.md)

**Project:** MLE vs Ridge vs Lasso Logistic Regression for Football Match Outcome Prediction  
**Status:** Concrete Implementation Blueprint  
**Authority:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md) | [RESEARCH_METHODOLOGY_GUIDE.md](file:///d:/MLE-Lasso/RESEARCH_METHODOLOGY_GUIDE.md#section-e-complete-experimental-methodology)  

---

## 1. Concrete Model Parameterizations

| Model | Regularization Penalty | Optimization Solver | Scikit-Learn / Statsmodels Call | Hyperparameter Tuning |
| :--- | :---: | :---: | :--- | :--- |
| **Unregularized MLE** | None ($\lambda = 0$) | L-BFGS / Newton-Raphson | `LogisticRegression(penalty=None, solver='lbfgs')` & `statsmodels.MNLogit` | Fixed ($C = \infty$, $\lambda = 0$). No tuning. |
| **Ridge Logistic** | $L_2$ Euclidean Norm | L-BFGS | `LogisticRegression(penalty='l2', solver='lbfgs', C=1/(2*n*lambda))` | Tuned on 2022/23 validation set over $\lambda \in [10^{-4}, 10^4]$. |
| **Lasso Logistic** | $L_1$ Taxicab Norm | SAGA (Stochastic Average Gradient) | `LogisticRegression(penalty='l1', solver='saga', C=1/(n*lambda))` | Tuned on 2022/23 validation set over $\lambda \in [10^{-4}, 10^4]$. |

> [!NOTE]
> **Sample-Size Normalization Factor ($n$):**
> Scikit-Learn minimizes the unnormalized empirical loss sum $C \sum_{i=1}^{n} \ell_i(\mathbf{B}) + \mathcal{R}(\mathbf{B})$, whereas standard statistical theory (and our manuscript Eq. 4) minimizes the mean loss $\frac{1}{n} \sum_{i=1}^{n} \ell_i(\mathbf{B}) + \lambda \mathcal{R}(\mathbf{B})$. Multiplying the theoretical objective by $n$ demonstrates that the scikit-learn inverse regularization parameter $C$ relates to the theoretical penalty parameter $\lambda$ through the training sample size $n = N_{\text{train}} = 1,002$:
> - **Ridge:** $C = \frac{1}{2 n \lambda}$
> - **Lasso:** $C = \frac{1}{n \lambda}$
> This guarantees that the strength of regularization remains invariant to the sample size $n$.

---

## 2. Solver Configuration & Numerical Invariants

1. **Convergence Tolerance:** `tol = 1e-5` across all models to ensure identical convergence rigor.
2. **Maximum Iterations:** `max_iter = 5000` to guarantee SAGA convergence on non-smooth $L_1$ landscapes.
3. **Intercept Handling:** `fit_intercept = True`. Intercepts $\beta_{0, c}$ represent baseline territorial advantage (~44% home win base rate) and are **strictly excluded from the $L_1$ and $L_2$ penalty functions**.
4. **Multi-class Strategy:** `multi_class = 'multinomial'` (Softmax link function).

---

## 3. Evaluation Metrics & Scoring Rules

All evaluation is conducted out-of-sample on the held-out Season 2023/24 test partition ($N_{\text{test}} = 369$ modelling-ready matches; 380 gross fixtures):

### 3.1 Primary Proper Scoring Rules (Probabilistic Loss)
- **Multi-class Log-Loss (Cross-Entropy):**
  $$\mathcal{L} = -\frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \sum_{c=1}^{3} y_{ic} \ln(\hat{p}_{ic})$$
- **Multi-class Brier Score (Quadratic Calibration Loss):**
  $$\text{BS} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \sum_{c=1}^{3} (\hat{p}_{ic} - y_{ic})^2$$
- **Expected Calibration Error (ECE):**
  Partitions predicted probabilities into $M = 10$ bins to measure empirical reliability against predicted confidences.

### 3.2 Classification Performance Metrics
- **Accuracy:** Overall proportion of correct discrete predictions ($\hat{y}_i = \operatorname{argmax}_c \hat{p}_{ic}$).
- **Macro-Averaged F1-Score:** Harmonic mean of precision and recall averaged equally across classes to avoid draw neglect.
- **Multi-class ROC-AUC (One-vs-Rest):** Area under the ROC curve weighted by class frequencies.

### 3.3 Statistical Hypothesis Testing Framework
- **Gameweek-Level Paired Wilcoxon Signed-Rank Test (Testing RQ1):** Evaluates if regularized models achieve statistically significant reductions in gameweek Log-Loss against unconstrained MLE ($\alpha = 0.05$ with Bonferroni correction).
- **McNemar's Test with Continuity Correction:** Assesses whether discrete prediction differences between models are statistically significant ($\chi^2 \sim \chi^2(1)$).
- **Block-Bootstrap Resampling ($B = 1,000$ draws):**
  - **Empirical Bootstrap Standard Error ($\widehat{\text{SE}}$):** Measures coefficient variance under collinearity.
  - **Sign Stability Index (SSI):** Proportion of resamples retaining the sign of the full-sample coefficient (Testing RQ2).
  - **Feature Selection Frequency (FSF):** For Lasso, proportion of bootstrap models retaining feature $j$ ($\beta_j \neq 0$), defining the Core Tactical Feature Set (Testing RQ3).

---

## 4. Full Experimental Protocol

For the complete experimental protocol, cross-entropy derivation, and step-by-step statistical procedures, see [Section E of RESEARCH_METHODOLOGY_GUIDE.md](file:///d:/MLE-Lasso/RESEARCH_METHODOLOGY_GUIDE.md#section-e-complete-experimental-methodology).
