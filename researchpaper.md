# Regularisation Under Collinearity: A Controlled Comparison of Maximum Likelihood, Ridge and Lasso Multinomial Logistic Regression for Football Match Outcome Forecasting

**Shubham Khanal**  
*Department of Computer Science / Machine Learning*  
*Preprint draft — October 13, 2026*  
**Repository:** [https://github.com/KhanalShubham/Lasso](https://github.com/KhanalShubham/Lasso)

---

### Abstract

Pre-match performance indicators in association football are strongly collinear, which is a setting where classical maximum likelihood estimation (MLE) is expected to produce unstable coefficients and poorly calibrated probabilities. This study isolates the effect of the regularisation penalty by fitting unregularised, $L_2$ (Ridge) and $L_1$ (Lasso) multinomial logistic regression to an identical feature matrix, identical chronological partitions and an identical cross-entropy loss. The data comprise 1,900 English Premier League fixtures from 2019/20 to 2023/24, of which 1,741 have sufficient history for leak-free pre-match features (16 rolling differential predictors). Models are estimated on 2019/20–2021/22 ($N = 1,002$), tuned on 2022/23 ($N = 370$) and evaluated once on 2023/24 ($N = 369$) using strictly proper scoring rules, with a training-frequency prior and a bookmaker-implied benchmark as reference points. Inference uses matchweek-block bootstrap resampling ($B = 1,000$) for coefficient standard errors, sign stability and Lasso selection frequency. 

On the training split, eight of the sixteen predictors have a variance inflation factor of at least 5 and the unpenalised Hessian has condition number 938.36. On the held-out 2023/24 season ($N = 369$), both penalties lower log-loss relative to MLE (0.9846 for MLE, 0.9636 for Ridge, 0.9573 for Lasso), but the gains are modest: the one-sided paired Wilcoxon tests on weekly log-loss give $p = 0.044$ (Ridge) and $p = 0.036$ (Lasso), neither of which meets the Bonferroni-adjusted level of 0.025. Matchweek-block bootstrap standard errors are on average 74.8% smaller under Ridge than under MLE, but sign stability is not uniformly improved: the mean sign stability index is 0.81 for MLE and 0.78 for Ridge, with large gains for some collinear predictors and losses for others. The Lasso retains nine of sixteen predictors, one more than the pre-specified parsimony threshold of eight, so the parsimony part of the third hypothesis is not met; its predictive accuracy is, however, formally non-inferior to Ridge (upper 95% confidence bound for the log-loss difference $+0.0015\text{ nats}$ against a margin of $0.005\text{ nats}$; $t = -2.408, p = 0.0106$). Draw probabilities are close to the base rate for every model, and the lower draw calibration error of the penalised models should be read in that light. All three models remain well behind bookmaker-implied probabilities (log-loss 0.9094). The contribution is a transparent, reproducible account of how shrinkage and sparsity behave on correlated sports covariates, together with an explicit treatment of post-selection inference limits.

**Keywords:** multinomial logistic regression; ridge; lasso; multicollinearity; proper scoring rules; football forecasting; bootstrap

---

## 1. Introduction

Probabilistic forecasting of football match outcomes is a long-standing applied statistics problem (Maher, 1982; Dixon and Coles, 1997). Most of the literature emphasises the choice of model family. This paper takes a different, deliberately narrow stance: it holds the model family, the covariates, the loss function and the data splits fixed, and varies only the regularisation penalty. Any difference in forecast quality, coefficient behaviour or sparsity can then be attributed to that single design choice.

The motivation is empirical. Rolling shot, form and goal-difference summaries of the kind used in match forecasting are mechanically related to one another. In the present data, for example, the rolling shot-dominance differential and the total-shots differential have a sample correlation of 0.88, shot dominance and conceded shots have $-0.89$, and rolling goal difference and points per game have 0.89 (Section 3). When columns of the design matrix are nearly linearly dependent, the information matrix of the unpenalised multinomial logit is ill-conditioned, coefficient variance inflates, and signs can become unstable (Hoerl and Kennard, 1970). Ridge regression addresses this by shrinking all coefficients smoothly; the Lasso (Tibshirani, 1996) additionally sets some coefficients exactly to zero.

### Research Questions
- **RQ1:** Do $L_1$ or $L_2$ penalties yield a statistically significant reduction in out-of-sample log-loss and Brier score relative to unregularised MLE?
- **RQ2:** How do Ridge and Lasso mitigate coefficient instability caused by collinear predictors?
- **RQ3:** Which predictors does the Lasso retain, and how stable is that selection?
- **RQ4:** Do regularised models produce better-calibrated probabilities, in particular for draws?

### Hypotheses
- **$H_1$:** Ridge and Lasso achieve lower out-of-sample multi-class log-loss and Brier score than unregularised MLE.
- **$H_2$:** Under collinearity ($\text{VIF} > 5$), MLE exhibits larger bootstrap coefficient variance and lower sign stability than Ridge; Lasso yields a sparser coefficient structure.
- **$H_3$:** The Lasso retains at most 8 of the 16 candidate predictors ($\le 50\%$) without a statistically significant degradation in out-of-sample log-loss relative to Ridge (a non-inferiority-style parsimony claim).

### Contributions
1. A leakage-audited feature pipeline with an explicit warm-up policy;
2. A controlled three-way comparison on a shared loss surface;
3. A block-bootstrap treatment of coefficient stability and selection frequency that respects matchweek clustering;
4. Benchmarks that let readers judge absolute forecast quality; and
5. A plain statement of what Lasso-based inference can and cannot support.

---

## 2. Background and Related Work

### Football Outcome Models
Goal-based Poisson models (Maher, 1982) and their dependence-corrected extension (Dixon and Coles, 1997) dominate the statistical literature, alongside Bayesian network approaches such as Constantinou et al. (2012). Direct models of the three-way result with a multinomial logit are simpler but share the central difficulty that informative covariates are correlated.

### Regularised Generalised Linear Models
Ridge regression (Hoerl and Kennard, 1970) and the Lasso (Tibshirani, 1996) are treated comprehensively in Hastie et al. (2009). Friedman et al. (2010) describe coordinate-descent path algorithms for penalised generalised linear models including the multinomial case. The SAGA algorithm (Defazio et al., 2014) supports the non-smooth $L_1$ penalty and is the solver used here.

### Evaluating Probabilistic Forecasts
Strictly proper scoring rules reward honest probability statements (Gneiting and Raftery, 2007); the Brier score (Brier, 1950) and the logarithmic score are the two used here. For football, the choice of scoring rule matters: Wheatcroft (2021) argues against the ranked probability score on the grounds that it may favour forecasts that ignore information about outcomes. Calibration diagnostics follow the reliability-diagram and expected-calibration-error tradition (Guo et al., 2017).

### Post-Selection Inference
Standard errors and $p$-values computed after data-driven variable selection are generally invalid (Berk et al., 2013). This is relevant to any claim based on the set of Lasso-retained predictors and is revisited in Section 7.

### Further Football Forecasting Work
Goal-count models have been extended with bivariate Poisson (Karlis and Ntzoufras, 2003) and bivariate Weibull count (Boshnakov et al., 2017) specifications, with Bayesian hierarchical structure (Baio and Blangiardo, 2010), and with dynamically evolving team strengths (Owen, 2011). Regression models that use lagged team performance as covariates, for goals and for results, were compared by Goddard (2005), which is the closest antecedent of the covariate-based approach taken here. Machine-learning studies of English and European leagues use richer feature sets (Baboota and Kaur, 2019; Stübinger et al., 2020). Rating-based approaches have been compared through maximum likelihood estimation of team strength (Ley et al., 2019), and bookmaker odds have been combined with historical data in a statistical model (Egidi et al., 2018). Because bookmaker odds aggregate information from many sources, they are a demanding benchmark, and the efficiency of online football betting markets has been examined by Angelini and De Angelis (2019). The present study does not claim a new model class. Its aim is narrower: to attribute differences in forecasts and coefficients to the penalty alone, under a common loss and common data splits.

---

## 3. Data

### 3.1 Sources and Sample
Fixtures, kickoff timestamps, venues and scores were obtained from FBref (primary source) and matched to Football-Data.co.uk (secondary source), which supplies shots, shots on target, corners, fouls, cards and bookmaker odds. The sample covers the English Premier League for 2019/20 to 2023/24: 380 matches per season, $N = 1,900$ in total. All 1,900 fixtures were reconciled across the two sources with full agreement on final scores, and no duplicate fixtures were found. Raw files are treated as immutable; all derived data are produced by version-controlled scripts.

### 3.2 Pre-Match Predictors and Leakage Control
For a match at kickoff time $t$, features use only matches strictly before $t$. For team $i$, the rolling mean over window $k$ is:
$$\bar{X}_{i,t}^{(k)} = \frac{1}{k} \sum_{j=1}^k X_{i,\text{match}(t-j)}, \tag{1}$$
so the summation never includes the match being predicted ($j \ge 1$). Fixtures are ordered chronologically by kickoff timestamp, not nominal matchweek. The home–away differential is $\Delta X_t^{(k)} = \bar{X}_{H,t}^{(k)} - \bar{X}_{A,t}^{(k)}$. Sixteen differentials are used (Table 1); automated unit tests confirm that mutating current or future match values leaves historical feature vectors strictly unchanged.

| # | Predictor | Rolling Window ($k$) | Tactical Domain |
| :---: | :--- | :---: | :--- |
| 1–2 | Shots on target for / against (`diff_sot_k5`, `diff_sotA_k5`) | 5 | Shot quality & precision |
| 3–4 | Total shots for / against (`diff_shots_k5`, `diff_shotsA_k5`) | 5 | Shot volume & dominance |
| 5 | Shot accuracy ratio (SoT/shots) (`diff_sot_ratio_k5`) | 5 | Shot quality & precision |
| 6 | Shot dominance share (`diff_shot_dominance_k5`) | 5 | Shot volume & dominance |
| 7–8 | Corners won / conceded (`diff_corners_k5`, `diff_cornersA_k5`) | 5 | Territorial / set pieces |
| 9–10 | Fouls committed; yellow cards (`diff_fouls_k5`, `diff_yellows_k5`) | 5 | Disciplinary / physical contest |
| 11–12| Goal difference (`diff_goal_diff_k5`, `diff_goal_diff_k10`) | 5, 10 | Form & goal production |
| 13 | Clean-sheet rate (`diff_clean_sheets_k10`) | 10 | Defensive solidity |
| 14 | Points per game (`diff_ppg_k5`) | 5 | Form & points |
| 15 | Venue-specific points per game (`diff_venue_ppg_k5`) | 5 (venue) | Venue-specific form |
| 16 | Rest days differential (`diff_rest_days`) | — | Schedule congestion / fatigue |

*Table 1: The 16 differential candidate predictors (home team minus away team).*

### 3.3 Warm-Up Policy and Modelling Sample
Fixtures for which a club lacks $k \ge 10$ prior matches overall or $k \ge 5$ at the relevant venue are excluded from modelling but retained in the history used to build later features. This isolates 159 warm-up fixtures and leaves $N = 1,741$ modelling-ready matches. The warm-up exclusions fall mainly in 2019/20 (100) and 2020/21 (28), with 10, 10 and 11 in the three later seasons. The chronological partitions are therefore:

| Partition | Contiguous Seasons | Gross Fixtures | Warm-Up Excluded | Modelling-Ready ($N$) |
| :--- | :--- | :---: | :---: | :---: |
| **Training** | 2019/20–2021/22 | 1,140 | 138 | **1,002** |
| **Validation** | 2022/23 | 380 | 10 | **370** |
| **Test** | 2023/24 | 380 | 11 | **369** |
| **Total** | 5 Seasons | 1,900 | 159 | **1,741** |

Among the 1,741 modelling-ready matches, 774 (44.5%) are home wins, 389 (22.3%) draws and 578 (33.2%) away wins. In the training split ($N = 1,002$) the counts are 424 home wins (42.32%), 225 draws (22.46%) and 353 away wins (35.23%); these training proportions define the frequency-prior benchmark. The 2019/20 and 2020/21 seasons were played largely behind closed doors; stadium attendance is excluded as a predictor, and this structural change is acknowledged as a source of non-stationarity.

### 3.4 Collinearity Structure
Figure 1 displays the empirical correlation matrix of the 16 predictors. Pairs with $|r| \ge 0.84$ include shots vs. shot dominance ($r = 0.88$), conceded shots vs. shot dominance ($r = -0.89$), goal difference vs. points per game ($r = 0.89$), and 5- vs. 10-match goal difference ($r = 0.84$). Rest days are nearly uncorrelated with all other predictors.

Variance inflation factors (VIF) and condition numbers computed on the training split ($N = 1,002$) are reported in Table 2.

| # | Predictor | Cluster | VIF | Tolerance ($1/\text{VIF}$) | $R_j^2$ | Strongest Correlate | $r$ |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| 1 | `diff_shot_dominance_k5` | Shot volume & dominance | 74.14 | 0.0135 | 0.9865 | `diff_shotsA_k5` | -0.89 |
| 2 | `diff_shots_k5` | Shot volume & dominance | 48.23 | 0.0207 | 0.9793 | `diff_shot_dominance_k5` | +0.89 |
| 3 | `diff_sot_k5` | Shot quality & precision | 45.06 | 0.0222 | 0.9778 | `diff_shots_k5` | +0.79 |
| 4 | `diff_shotsA_k5` | Shot volume & dominance | 26.24 | 0.0381 | 0.9619 | `diff_shot_dominance_k5` | -0.89 |
| 5 | `diff_sot_ratio_k5` | Shot quality & precision | 16.75 | 0.0597 | 0.9403 | `diff_sot_k5` | +0.64 |
| 6 | `diff_goal_diff_k5` | Form & goal difference | 8.40 | 0.1190 | 0.8810 | `diff_ppg_k5` | +0.89 |
| 7 | `diff_goal_diff_k10` | Form & goal difference | 6.33 | 0.1580 | 0.8420 | `diff_goal_diff_k5` | +0.84 |
| 8 | `diff_ppg_k5` | Form & goal difference | 5.46 | 0.1832 | 0.8168 | `diff_goal_diff_k5` | +0.89 |
| 9 | `diff_sotA_k5` | Shot quality & precision | 3.24 | 0.3082 | 0.6918 | `diff_shotsA_k5` | +0.78 |
| 10 | `diff_venue_ppg_k5` | Form & goal difference | 2.36 | 0.4245 | 0.5755 | `diff_goal_diff_k10` | +0.72 |
| 11 | `diff_cornersA_k5` | Territorial / set pieces | 2.21 | 0.4519 | 0.5481 | `diff_shot_dominance_k5` | -0.70 |
| 12 | `diff_clean_sheets_k10` | Defensive solidity | 2.16 | 0.4637 | 0.5363 | `diff_goal_diff_k10` | +0.69 |
| 13 | `diff_corners_k5` | Territorial / set pieces | 1.98 | 0.5062 | 0.4938 | `diff_shots_k5` | +0.68 |
| 14 | `diff_yellows_k5` | Disciplinary | 1.33 | 0.7524 | 0.2476 | `diff_fouls_k5` | +0.43 |
| 15 | `diff_fouls_k5` | Disciplinary | 1.25 | 0.7976 | 0.2024 | `diff_yellows_k5` | +0.43 |
| 16 | `diff_rest_days` | Schedule congestion | 1.03 | 0.9725 | 0.0275 | `diff_corners_k5` | -0.10 |

*Table 2: Training-split ($N = 1,002$) variance inflation factors, sorted in descending order. Eight predictors have $\text{VIF} \ge 5$ and five have $\text{VIF} \ge 10$.*

### 3.5 Benchmarks
Two reference forecasts are evaluated on the test season:
1. **Training-Frequency Prior:** Predicts the class proportions observed in the training split ($\hat{\mathbf{p}} = [0.3523, 0.2246, 0.4232]$).
2. **Market-Implied Odds:** Converts bookmaker decimal odds $O_{ic}$ to probabilities with bookmaker overround removed:
   $$\hat{p}_{ic} = \frac{1/O_{ic}}{\sum_{k=1}^3 1/O_{ik}}, \tag{2}$$
   using standard pre-match fixed odds columns (`B365H/D/A` for Bet365 and `AvgH/D/A` for Market Average) supplied by Football-Data.co.uk. These are pre-match opening/standard quotes, not closing odds (`B365CH` / `AvgCH`), making them an authentic, non-leaking benchmark.

---

## 4. Methods

### 4.1 Multinomial Model and Shared Loss
Let $\mathbf{x}_i = (1, \Delta X_{i1}, \dots, \Delta X_{i16})^T$ and let outcome classes be Away win ($0$), Draw ($1$), Home win ($2$) ($K = 3$). The symmetric softmax model is:
$$p_{ic} = \frac{\exp(\mathbf{x}_i^T \boldsymbol{\beta}_c)}{\sum_{r=1}^3 \exp(\mathbf{x}_i^T \boldsymbol{\beta}_r)}, \tag{3}$$
and all three estimators minimise the identical cross-entropy empirical risk:
$$J_0(\mathbf{B}) = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^3 y_{ic} \ln p_{ic}. \tag{4}$$
Predictors are standardised with training-split means and standard deviations frozen before validation and test data are transformed.

### 4.2 Estimators
$$\hat{\mathbf{B}}_{\text{MLE}} = \arg\min_{\mathbf{B}} J_0(\mathbf{B}), \tag{5}$$
$$\hat{\mathbf{B}}_{\text{Ridge}} = \arg\min_{\mathbf{B}} \left\{ J_0(\mathbf{B}) + \lambda \sum_{c=1}^3 \sum_{j=1}^{16} \beta_{jc}^2 \right\}, \tag{6}$$
$$\hat{\mathbf{B}}_{\text{Lasso}} = \arg\min_{\mathbf{B}} \left\{ J_0(\mathbf{B}) + \lambda \sum_{c=1}^3 \sum_{j=1}^{16} |\beta_{jc}| \right\}. \tag{7}$$

Intercepts are strictly unpenalised. The unpenalised Hessian block is $\mathbf{H}_{cc} = n^{-1} \mathbf{X}^T \mathbf{W}_c \mathbf{X}$ with $W_{c,ii} = p_{ic}(1 - p_{ic})$; under collinearity its smallest eigenvalue approaches zero, inflating $\mathbf{H}^{-1}$. Ridge adds $2\lambda \mathbf{I}$ to the penalised block, guaranteeing $\lambda_{\min} \ge 2\lambda$. The Lasso subgradient condition and soft-thresholding operator $S_\alpha(u) = \text{sign}(u)\max(|u|-\alpha, 0)$ generate exact parameter sparsity.

### 4.3 Solvers and Mapping
- **MLE:** L-BFGS (`penalty=None`), cross-checked with statsmodels `MNLogit`.
- **Ridge:** L-BFGS (`penalty='l2'`).
- **Lasso:** SAGA (`penalty='l1'`).
- **Tolerances:** $10^{-5}$ for MLE and Ridge, $10^{-4}$ for Lasso; maximum 5,000 iterations.
Because $J_0$ includes the factor $1/n$, the scikit-learn inverse-regularisation parameter $C$ relates to $\lambda$ through $n = N_{\text{train}} = 1,002$:
- Ridge: $C = \frac{1}{2n\lambda}$ (selected $\lambda^* = 0.0886 \implies C = 0.005633$).
- Lasso: $C = \frac{1}{n\lambda}$ (selected $\lambda^* = 0.0127 \implies C = 0.078319$).

### 4.4 Tuning Protocol & Degrees of Freedom
$\lambda$ is selected on a 20-point logarithmic grid over $[10^{-4}, 10^4]$ by minimising multi-class log-loss on the 2022/23 validation season ($N = 370$). The effective degrees of freedom of the Ridge fit are reported as:
$$\text{df}(\lambda) = \operatorname{tr}\left( (\mathbf{X}^T \mathbf{W} \mathbf{X} + 2n\lambda\mathbf{I})^{-1}\mathbf{X}^T \mathbf{W} \mathbf{X} \right) + K, \tag{8}$$
where $\mathbf{X}$ holds the 16 standardised predictors, $\mathbf{W}$ is a diagonal matrix of average binary IRLS weights across the three classes, and $K = 3$ counts unpenalised class intercepts. This local approximation evaluates over the 16 predictor columns directly (yielding a maximum parameter capacity of $16 + 3 = 19$), which at $\lambda^* = 0.0886$ evaluates to $\text{df}(\lambda^*) = 5.73 + 3 = 8.73$ (a $64.2\%$ shrinkage of the predictor basis). For comparison, evaluating $\text{df}(\lambda)$ over the full $32 \times 32$ identifiable reference-class contrast Hessian yields $\text{df}_{\text{ref}}(\lambda^*) = 10.86 + 2 = 12.86$ (a $66.1\%$ shrinkage of the 32 unregularised contrast parameters), demonstrating consistent shrinkage intensity across parametrisations. We retain the basis-level approximation $\text{df}(\lambda^*) = 8.73$ throughout the text to align directly with the 16 candidate predictor variables.

### 4.5 Evaluation Metrics & Hypothesis Testing Framework
- **Primary Probabilistic Scoring Rules:** Multi-class Log-Loss and Multi-class Brier Score $\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^3 (\hat{p}_{ic} - y_{ic})^2$.
- **Calibration:** Expected Calibration Error (ECE, 10 bins) and one-vs-rest binary calibration curves.
- **Paired Weekly Wilcoxon Signed-Rank Test ($H_1$):** Evaluated across the 38 matchweeks on weekly mean log-loss, with Bonferroni correction $\alpha = 0.025$.
- **Matchweek Block-Bootstrap ($H_2, H_3$):** $B = 1,000$ draws of whole matchweeks (104 training blocks) computing Empirical Bootstrap Standard Errors, Sign Stability Index ($\text{SSI}_j = B^{-1}\sum_{b=1}^B \mathbb{I}\{\text{sign}(\hat{\beta}_j^{(b)}) = \text{sign}(\hat{\beta}_j^{\text{full}})\}$) and Lasso Feature Selection Frequency ($\text{FSF}_j = B^{-1}\sum_{b=1}^B \mathbb{I}\{\hat{\beta}_j^{(b)} \ne 0\}$). Master seed is 42. Predictors with $\text{FSF}_j \ge 0.80$ define the Core Tactical Feature Set.
- **Parsimony & Non-Inferiority ($H_3$):** Testing non-inferiority margin $\delta = 0.005\text{ nats}$ on the log-loss difference $\mathcal{L}_{\text{Lasso}} - \mathcal{L}_{\text{Ridge}}$.

---

## 5. Results

### 5.1 Collinearity Diagnostics
On the training split ($N = 1,002$), the design matrix has condition number $\kappa(\mathbf{X}) = 30.63$, and the unpenalised Hessian has condition number $\kappa(\mathbf{H}_0) = 938.36$. Eight of the sixteen predictors (50.0%) have $\text{VIF} \ge 5$ and five (31.3%) have $\text{VIF} \ge 10$ (Table 2).

### 5.2 Tuning and Regularisation Paths
Table 3 reports validation-season performance ($N_{\text{val}} = 370$) across candidate models. Validation loss was minimised at $\lambda_{\text{Ridge}}^* = 0.0886$ ($C = 0.005633$) and $\lambda_{\text{Lasso}}^* = 0.0127$ ($C = 0.078319$).

| Model | Penalty $\lambda$ | $C$ Parameter | Val Log-Loss (nats) | Val Brier | Val Accuracy (%) | Model Complexity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Training-Frequency Prior | — | — | 1.0581 | 0.6391 | 48.38% | 0 predictors |
| Market Average Odds | — | — | 0.9684 | 0.5755 | 55.95% | Exogenous benchmark |
| Unregularized MLE | 0 | $\infty$ | 1.0146 | 0.6048 | 54.59% | 16 predictors (full basis) |
| Ridge ($L_2$) | 0.0886 | 0.005633 | 1.0016 | 0.5984 | 54.86% | $\text{df}(\lambda) = 8.73$ |
| Lasso ($L_1$) | 0.0127 | 0.078319 | 1.0003 | 0.5978 | 55.41% | 9 of 16 retained (56.2%) |

*Table 3: Validation season 2022/23 ($N = 370$) hyperparameter tuning results.*

The validation loss difference between the two penalised models is only 0.0013 nats, much smaller than the gap between either model and MLE ($-0.0130$ for Ridge, $-0.0143$ for Lasso).

Inspection of the full grid in `reports/hyperparameter_tuning.csv` reveals that at the immediate next grid point above the optimum ($\lambda = 0.0336$, $C = 0.0297$), the Lasso retains only **three predictors** (`diff_shotsA_k5`, `diff_shot_dominance_k5`, and `diff_goal_diff_k10`). At this point, the validation log-loss is **$1.000409\text{ nats}$**—differing from the global minimum by merely $+0.000119\text{ nats}$ ($+0.012\%$), while yielding an even slightly sharper Brier score of $0.597640$. At the second point ($\lambda = 0.0886$, $C = 0.0113$), the same three predictors are retained with validation log-loss $1.016714\text{ nats}$ ($+0.0164\text{ nats}$ above the minimum), before shrinking all coefficients to zero at $\lambda = 0.2336$.

At $\lambda_{\text{Lasso}}^* = 0.0127$, the 9 retained predictors are: `diff_shots_k5`, `diff_shotsA_k5`, `diff_shot_dominance_k5`, `diff_corners_k5`, `diff_fouls_k5`, `diff_yellows_k5`, `diff_goal_diff_k5`, `diff_goal_diff_k10`, and `diff_rest_days`. The 7 eliminated predictors are: `diff_sot_k5`, `diff_sotA_k5`, `diff_sot_ratio_k5`, `diff_cornersA_k5`, `diff_clean_sheets_k10`, `diff_ppg_k5`, and `diff_venue_ppg_k5`.

### 5.3 Out-of-Sample Forecast Quality
Table 4 presents the out-of-sample evaluation on the held-out Season 2023/24 test partition ($N = 369$, 38 matchweeks).

| Model / Benchmark | Log-Loss (nats) | Brier Score | Top ECE (10 Bins) | Accuracy (%) | Macro-F1 | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Training-Frequency Prior | 1.0548 | 0.6380 | 0.0403 | 46.34% | 0.2111 | 0.5000 |
| Market Average Odds | 0.9094 | 0.5332 | 0.0528 | 58.54% | 0.4312 | 0.7170 |
| Bet365 Closing Odds | 0.9091 | 0.5330 | 0.0584 | 58.81% | 0.4331 | 0.7185 |
| Unregularized MLE | 0.9846 | 0.5810 | 0.0568 | 56.10% | 0.4274 | 0.6403 |
| Ridge ($L_2$) ($\lambda^* = 0.0886$) | 0.9636 | 0.5703 | **0.0370** | 55.83% | 0.4091 | 0.6555 |
| Lasso ($L_1$) ($\lambda^* = 0.0127$) | **0.9573** | **0.5658** | 0.0467 | **56.37%** | 0.4144 | **0.6673** |

*Table 4: Out-of-sample forecast evaluation on Season 2023/24 test set ($N_{\text{test}} = 369$).*

Relative to unregularized MLE, Ridge lowers out-of-sample log-loss by $0.0210\text{ nats}$ and Brier score by $0.0107$; Lasso lowers them by $0.0273\text{ nats}$ and $0.0152$. The gap to the market benchmark is $0.0479\text{ nats}$ for Lasso, $0.0542\text{ nats}$ for Ridge, and $0.0752\text{ nats}$ for MLE. Accuracy differences remain within 1 percentage point (56.10%, 55.83%, 56.37%).

### 5.4 Paired Statistical Tests & Non-Inferiority
Table 5 reports the weekly paired statistical tests across the 38 matchweeks.

| Comparison | Statistical Test | Test Statistic | $p$-value | Mean Weekly Difference |
| :--- | :--- | :---: | :---: | :---: |
| **Log-loss** | | | | |
| MLE vs. Ridge | Wilcoxon signed-rank (one-sided) | $W = 489.0$ | $0.0436$ | $+0.0210\text{ nats}$ |
| MLE vs. Lasso | Wilcoxon signed-rank (one-sided) | $W = 495.0$ | $0.0359$ | $+0.0275\text{ nats}$ |
| Ridge vs. Lasso | Wilcoxon signed-rank (two-sided) | $W = 293.0$ | $0.2675$ | $0.0063\text{ nats}$ (Lasso lower) |
| **Brier score** | | | | |
| MLE vs. Ridge | Wilcoxon signed-rank (one-sided) | $W = 467.0$ | $0.0828$ | $+0.0108$ |
| MLE vs. Lasso | Wilcoxon signed-rank (one-sided) | $W = 472.0$ | $0.0722$ | $+0.0155$ |
| Ridge vs. Lasso | Wilcoxon signed-rank (two-sided) | $W = 310.0$ | $0.3885$ | $0.0047$ (Lasso lower) |
| **Discrete Classification**| | | | |
| MLE vs. Ridge | McNemar test ($\chi^2 = 0.00$) | — | $1.0000$ | 23 discordant matches |
| MLE vs. Lasso | McNemar test ($\chi^2 = 0.00$) | — | $1.0000$ | 23 discordant matches |
| Ridge vs. Lasso | McNemar test ($\chi^2 = 0.05$) | — | $0.8231$ | 20 discordant matches |

*Table 5: Paired tests on the 2023/24 test season (38 matchweeks). Bonferroni threshold for MLE comparisons is $\alpha = 0.025$.*

#### Non-Inferiority of Lasso to Ridge
- **Weekly Aggregation ($N = 38$ Gameweeks):** Mean difference $\bar{d} = -0.006547\text{ nats}$, $\text{SE} = 0.004796$. Two-sided 95% CI is $[-0.016264, +0.003170]\text{ nats}$; one-sided 95% upper confidence bound is **$+0.001544\text{ nats}$** (strictly below $\delta = 0.005\text{ nats}$; $t_{\text{NI}} = -2.408, p_{\text{NI}} = 0.0106$).
- **Match-Level Verification ($N = 369$):** Mean difference $\bar{d} = -0.006335\text{ nats}$, $\text{SE} = 0.004335$. Two-sided 95% CI is $[-0.014859, +0.002189]\text{ nats}$; one-sided 95% upper confidence bound is **$+0.000813\text{ nats}$** ($t_{\text{NI}} = -2.615, p_{\text{NI}} = 0.00465$). Bootstrap 95% CI ($B = 10,000$) is $[-0.015148, +0.001713]\text{ nats}$.
- **Conclusion:** Non-inferiority of Lasso to Ridge is formally established at margin $\delta = 0.005\text{ nats}$.

### 5.5 Coefficient Stability & Variable Selection ($B = 1,000$)
Table 6 summarizes the $B = 1,000$ matchweek block-bootstrap resamples for the Home Win vs. Draw contrast.

| Predictor Variable | $\text{SE}_{\text{MLE}}$ | $\text{SE}_{\text{Ridge}}$ | SE Reduction (%) | $\text{SSI}_{\text{MLE}}$ | $\text{SSI}_{\text{Ridge}}$ | $\text{FSF}_{\text{Lasso}}$ (%) | Core Feature Set |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `diff_shot_dominance_k5` | 0.7073 | 0.0256 | 96.4% | 0.910 | 1.000 | 55.9% | No |
| `diff_shots_k5` | 0.5598 | 0.0354 | 93.7% | 0.952 | 0.540 | 73.4% | No |
| `diff_sot_k5` | 0.4920 | 0.0313 | 93.6% | 0.821 | 0.714 | 29.2% | No |
| `diff_shotsA_k5` | 0.4102 | 0.0383 | 90.7% | 0.529 | 1.000 | 97.0% | **Yes** |
| `diff_goal_diff_k5` | 0.2740 | 0.0403 | 85.3% | 0.888 | 0.664 | 59.1% | No |
| `diff_sot_ratio_k5` | 0.3147 | 0.0467 | 85.1% | 0.823 | 0.625 | 52.8% | No |
| `diff_goal_diff_k10` | 0.2548 | 0.0470 | 81.5% | 0.991 | 0.997 | 99.9% | **Yes** |
| `diff_ppg_k5` | 0.2049 | 0.0418 | 79.6% | 0.846 | 0.773 | 22.1% | No |
| `diff_rest_days` | 0.1490 | 0.0318 | 78.6% | 0.893 | 0.880 | 50.1% | No |
| `diff_sotA_k5` | 0.1623 | 0.0489 | 69.9% | 0.922 | 0.633 | 41.2% | No |
| `diff_venue_ppg_k5` | 0.1206 | 0.0436 | 63.9% | 0.649 | 0.863 | 55.1% | No |
| `diff_clean_sheets_k10` | 0.1236 | 0.0467 | 62.2% | 0.688 | 0.786 | 57.1% | No |
| `diff_cornersA_k5` | 0.1276 | 0.0496 | 61.2% | 0.738 | 0.718 | 62.4% | No |
| `diff_corners_k5` | 0.1128 | 0.0475 | 57.9% | 0.516 | 0.546 | 70.9% | No |
| `diff_yellows_k5` | 0.1057 | 0.0531 | 49.7% | 0.872 | 0.762 | 72.3% | No |
| `diff_fouls_k5` | 0.1024 | 0.0539 | 47.4% | 0.891 | 0.893 | 86.6% | **Yes** |
| **Mean Across Predictors** | **0.2639** | **0.0426** | **74.8%** | **0.808** | **0.775** | **61.6%** | **3 Core** |

*Table 6: Matchweek-block bootstrap ($B = 1,000$) coefficient stability results for the Home Win vs. Draw contrast.*

A Lasso selection frequency $\text{FSF} \ge 0.80$ is achieved by only three predictors: `diff_goal_diff_k10` (99.9%), `diff_shotsA_k5` (97.0%), and `diff_fouls_k5` (86.6%). Within the shot cluster, frequencies are unstable (`diff_shots_k5` 73.4%, `diff_shot_dominance_k5` 55.9%, `diff_sot_k5` 29.2%).

### 5.6 Probability Calibration
Table 7 reports the per-class calibration error across 10 bins for each outcome class on the test set ($N = 369$: 119 away wins, 79 draws, 171 home wins).

| Model / Benchmark | Mean Predicted Draw | $\text{ECE}_{\text{Draw}}$ | $\text{Brier}_{\text{Draw}}$ | $\text{ECE}_{\text{Home}}$ | $\text{ECE}_{\text{Away}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Training-Frequency Prior | 0.2246 | 0.0105 | 0.1684 | 0.0403 | 0.0298 |
| Market Average Odds | 0.2237 | 0.0201 | 0.1659 | 0.0692 | 0.0746 |
| Bet365 Closing Odds | 0.2253 | 0.0224 | 0.1662 | 0.0679 | 0.0796 |
| Unregularized MLE | 0.2244 | 0.0547 | 0.1738 | 0.0652 | 0.0620 |
| Ridge ($L_2$) ($\lambda^* = 0.0886$) | 0.2221 | **0.0080** | 0.1680 | **0.0564** | 0.0444 |
| Lasso ($L_1$) ($\lambda^* = 0.0127$) | 0.2226 | **0.0085** | **0.1677** | 0.0615 | **0.0364** |

*Table 7: Per-class calibration on 2023/24 test season ($N = 369$). Observed draw frequency is 0.2141.*

The overall ECE in Table 4 is computed as the top-label (maximum-confidence) Expected Calibration Error (Guo et al., 2017) partitioned across ten equal-width confidence bins:
$$\text{ECE}_{\text{top}} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|,$$
where for match $i$, $\hat{y}_i = \arg\max_c \hat{p}_{ic}$ and $\hat{c}_i = \max_c \hat{p}_{ic}$. For the constant training-frequency prior, every forecast assigns its maximum confidence to Home Win ($\hat{c}_i = 0.4232$), all falling into bin $(0.4, 0.5]$; because the realised test home-win frequency is $171/369 = 0.4634$, the prior's top-label error is identically $|0.4634 - 0.4232| = 0.0403$. Table 7 complements this by reporting the one-vs-rest binary calibration error evaluated individually for each outcome class.

### 5.7 Summary of Hypothesis Decisions
- **$H_1$ (Lower loss than MLE):** *Direction supported; significance not established.* Both regularised models lower log-loss and Brier score on test and validation partitions, but the paired Wilcoxon tests ($p = 0.044$ and $0.036$) do not clear the Bonferroni threshold of $0.025$.
- **$H_2$ (Variance and sign stability):** *Partly supported.* Ridge reduces bootstrap standard errors by an average of 74.8% (and $>90\%$ on shot features). Sign stability is not uniformly improved (mean SSI is 0.775 for Ridge vs. 0.808 for MLE), as Ridge shrinks substitutable coefficients close to zero where signs can oscillate.
- **$H_3$ (Parsimony without loss of accuracy):** *Not supported as specified.* Lasso retains 9 of 16 predictors (56.2%), above the pre-specified threshold of 8 ($\le 50\%$). The predictive non-inferiority claim holds firmly ($+0.0015\text{ nats}$ upper bound vs. margin $0.005\text{ nats}$).

---

## 6. Discussion

### Does Regularisation Help?
In the direction of the effect, yes. On both validation and test seasons, penalised models achieve lower log-loss and Brier scores than unregularised MLE, with Lasso slightly edging out Ridge. However, the magnitude of the improvement is modest: $0.021$ to $0.027\text{ nats}$ on the test season, against a gap of $0.05$ to $0.075\text{ nats}$ between statistical models and bookmaker odds. With a single test season of 369 matches, Wilcoxon tests yield $p \approx 0.04$ which do not survive a two-comparison Bonferroni correction ($\alpha = 0.025$).

### Variance, Conditioning, and Sign Stability
Ridge shrinkage reduces coefficient standard errors by 74.8% on average, providing massive numerical stabilisation for ill-conditioned Hessians ($\kappa(\mathbf{H}_0) = 938.36$). However, variance reduction does not translate into uniform sign stability. When features are near-substitutes (e.g., total shots vs. shots on target), Ridge splits the shared signal and shrinks both toward zero, where small sample fluctuations flip coefficient signs. Sign stability must therefore be interpreted predictor by predictor.

### What Lasso Does with Correlated Predictors
Lasso cleanly resolved the form cluster by retaining goal difference and eliminating points per game ($r = 0.89$). However, it failed to break up the shot cluster, retaining shots, conceded shots, and shot dominance. Bootstrapping indicates that this selection is sample-sensitive: only three predictors (`diff_goal_diff_k10`, `diff_shotsA_k5`, and `diff_fouls_k5`) persist in $\ge 80\%$ of resamples. The 9-variable model is one plausible sparse approximation, but only the 3-variable core set represents a stable structure.

### Calibration and Market Inefficiencies
Ridge and Lasso achieve an $85.4\%$ reduction in draw calibration error ($\text{ECE}_{\text{Draw}} = 0.0080$ vs. $0.0547$ for MLE). However, their draw probabilities remain tightly clustered around the empirical base rate ($\approx 0.22$), meaning the low ECE reflects conservative risk pooling rather than an ability to discriminate draws. Commercial bookmaker odds remain vastly superior overall, reflecting real-time market aggregation of team news and non-public information.

---

## 7. Limitations

1. **Dependence:** Matches are clustered within teams and autocorrelated over time; i.i.d. likelihood is an approximation.
2. **Post-Selection Inference:** Naive standard errors and $p$-values after Lasso selection are invalid (Berk et al., 2013); selection frequencies are descriptive.
3. **Predictor Set:** Tracking data, progressive passes, and expected goals (xG) were inaccessible; shots and corners act as proxies. Environmental factors (e.g., heat stress in Saudi Pro League, Nisbet et al., 2027) are omitted.
4. **Non-Stationarity:** The 2019/20 and 2020/21 seasons were played largely behind closed doors due to COVID-19.
5. **Sample Size:** A single test season of 369 matches provides limited statistical power.
6. **Tuning Grid Granularity:** The 20-point grid has adjacent points differing by a factor of 2.6; Lasso retains 3 predictors at the next grid point with virtually identical validation loss ($+0.00012\text{ nats}$).
7. **Multiple Comparisons:** Bonferroni correction covers only the primary MLE comparisons.
8. **Scope:** Single league (English Premier League) over five contiguous seasons.

---

## 8. Conclusion

Holding the model family, features, loss and chronological splits fixed, Ridge and Lasso multinomial logistic regression produce modestly better probabilistic forecasts than unpenalised maximum likelihood on English Premier League data (test log-loss 0.9636 and 0.9573 against 0.9846), and Ridge reduces bootstrap coefficient standard errors by about three quarters, although it does not improve average sign stability. The improvements in log-loss are in a consistent direction but are not statistically significant after Bonferroni correction on a single test season, and the Brier-score tests are weaker still. The Lasso retains nine of sixteen predictors, so the pre-specified parsimony threshold of eight is not met, although its accuracy is formally non-inferior to Ridge's; it does not fully resolve the most strongly collinear shot cluster, and only three predictors are selected in at least 80% of bootstrap resamples. Draw probabilities stay near the base rate for all models. All models are clearly less accurate than bookmaker-implied probabilities. Regularisation is therefore a sensible, low-cost default for correlated football covariates, but the evidence here supports it as a modest stabiliser rather than a decisive improvement.

---

## Reproducibility

Code, feature construction, automated test suites, and figure scripts are maintained in a version-controlled repository; every derived dataset is regenerated by a standalone script, and raw data are never modified. 
- **Pinned Dependencies:** `scikit-learn 1.9.0`, `statsmodels 0.15.0`, `NumPy 2.4.4`, `pandas 3.0.2`, `SciPy 1.18.0`, `Matplotlib 3.11.1` (Python 3.14.4).
- **Test Suite:** 36 automated unit tests across all 6 project quality gates pass with zero failures (`pytest tests/`).
- **Bootstrap Master Seed:** 42.
- **Repository URL:** [https://github.com/KhanalShubham/Lasso](https://github.com/KhanalShubham/Lasso)

---

## Declarations

- **Data Availability:** Match results, scores, and closing/opening odds are openly available from Football-Data.co.uk under standard non-commercial educational fair use. Detailed fixture schedules and timestamps originate from FBref (Sports Reference LLC; Opta-derived data). In accordance with Sports Reference terms of service, raw scraped HTML/CSV files are not redistributed in the public repository; instead, automated reproducible ingestion scripts (`src/data_collection.py`) and the leak-free derived feature tables (`data/features/dataset_features.csv`) are provided in the repository, enabling complete replication from primary web endpoints.
- **Funding and Competing Interests:** The authors declare that they have no competing financial or non-financial interests. This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
- **Use of AI Tools:** An AI coding assistant (Google DeepMind Antigravity) was used to implement data pipelines, maintain automated quality-gate unit test suites, and execute statistical computations. A large language model (Claude, Anthropic) was used to assist in drafting and refining manuscript prose. The authors have verified all mathematical derivations, computational algorithms, empirical results, and bibliographic citations, and accept sole responsibility for the contents of this manuscript.

---

## Appendix: Hyperparameter Tuning Grid Diagnostics

The complete 20-point logarithmic hyperparameter search evaluated on Season 2022/23 validation partition ($N_{\text{val}} = 370$):

| Step | Penalty $\lambda$ | $\log_{10}\lambda$ | $C_{\text{Ridge}}$ | $C_{\text{Lasso}}$ | Ridge Log-Loss | Ridge Brier | Ridge $\text{df}(\lambda)$ | Lasso Log-Loss | Lasso Brier | Lasso Active Features |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 0.000100 | -4.00 | 4.990020 | 9.980040 | 1.014282 | 0.604705 | 18.76 | 1.013960 | 0.604550 | 16 (100.0%) |
| 2 | 0.000264 | -3.58 | 1.892560 | 3.785120 | 1.013866 | 0.604531 | 18.44 | 1.012935 | 0.604061 | 16 (100.0%) |
| 3 | 0.000695 | -3.16 | 0.717789 | 1.435579 | 1.013041 | 0.604178 | 17.85 | 1.011082 | 0.603253 | 16 (100.0%) |
| 4 | 0.001833 | -2.74 | 0.272235 | 0.544471 | 1.011648 | 0.603552 | 16.97 | 1.007679 | 0.601724 | 15 (93.8%) |
| 5 | 0.004833 | -2.32 | 0.103250 | 0.206501 | 1.009303 | 0.602427 | 15.73 | 1.002691 | 0.599200 | 13 (81.3%) |
| 6 | **0.012743** | **-1.89** | **0.039160** | **0.078319** | 1.006095 | 0.600813 | 13.92 | **1.000290** | **0.597842** | **9 (56.2%)** |
| 7 | 0.033598 | -1.47 | 0.014852 | 0.029704 | 1.003160 | 0.599278 | 11.47 | 1.000409 | 0.597640 | 3 (18.8%) |
| 8 | **0.088587** | **-1.05** | **0.005633** | **0.011266** | **1.001601** | **0.598406** | **8.73** | 1.016714 | 0.608121 | 3 (18.8%) |
| 9 | 0.233572 | -0.63 | 0.002136 | 0.004273 | 1.001921 | 0.598445 | 6.38 | 1.057665 | 0.638790 | 0 (0.0%) |
| 10 | 0.615848 | -0.21 | 0.000810 | 0.001621 | 1.006232 | 0.600919 | 4.77 | 1.057739 | 0.638849 | 0 (0.0%) |
| 11 | 1.623777 | 0.21 | 0.000307 | 0.000615 | 1.017586 | 0.608533 | 3.84 | 1.057637 | 0.638875 | 0 (0.0%) |
| 12 | 4.281332 | 0.63 | 0.000117 | 0.000233 | 1.033390 | 0.620127 | 3.37 | 1.057636 | 0.638875 | 0 (0.0%) |
| 13 | 11.288379 | 1.05 | 0.000044 | 0.000088 | 1.046091 | 0.629833 | 3.15 | 1.057637 | 0.638876 | 0 (0.0%) |
| 14 | 29.763514 | 1.47 | 0.000017 | 0.000034 | 1.053015 | 0.635193 | 3.06 | 1.057637 | 0.638876 | 0 (0.0%) |
| 15 | 78.475997 | 1.89 | 0.000006 | 0.000013 | 1.056079 | 0.637574 | 3.02 | 1.057637 | 0.638876 | 0 (0.0%) |
| 16 | 206.913808 | 2.32 | 0.000002 | 0.000005 | 1.057313 | 0.638534 | 3.01 | 1.057637 | 0.638876 | 0 (0.0%) |
| 17 | 545.559478 | 2.74 | 9.15e-07 | 1.83e-06 | 1.057791 | 0.638906 | 3.00 | 1.057637 | 0.638876 | 0 (0.0%) |
| 18 | 1438.449888 | 3.16 | 3.47e-07 | 6.94e-07 | 1.057975 | 0.639049 | 3.00 | 1.057637 | 0.638876 | 0 (0.0%) |
| 19 | 3792.690191 | 3.58 | 1.32e-07 | 2.63e-07 | 1.058042 | 0.639101 | 3.00 | 1.057637 | 0.638876 | 0 (0.0%) |
| 20 | 10000.000000 | 4.00 | 4.99e-08 | 9.98e-08 | 1.058071 | 0.639124 | 3.00 | 1.057637 | 0.638876 | 0 (0.0%) |

---

## References

1. Angelini, G., & De Angelis, L. (2019). Efficiency of online football betting markets. *International Journal of Forecasting*, 35(2), 712–721.
2. Baboota, R., & Kaur, H. (2019). Predictive analysis and modelling football results using machine learning approach for English Premier League. *International Journal of Forecasting*, 35(2), 741–755.
3. Baio, G., & Blangiardo, M. (2010). Bayesian hierarchical model for the prediction of football results. *Journal of Applied Statistics*, 37(2), 253–264.
4. Berk, R., Brown, L., Buja, A., Zhang, K., & Zhao, L. (2013). Valid post-selection inference. *The Annals of Statistics*, 41(2), 802–837.
5. Boshnakov, G., Kharrat, T., & McHale, I. G. (2017). A bivariate Weibull count model for forecasting association football scores. *International Journal of Forecasting*, 33(2), 458–466.
6. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
7. Constantinou, A. C., Fenton, N. E., & Neil, M. (2012). pi-football: A Bayesian network model for forecasting association football match outcomes. *Knowledge-Based Systems*, 36, 322–339.
8. Defazio, A., Bach, F., & Lacoste-Julien, S. (2014). SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. *Advances in Neural Information Processing Systems*, 27.
9. Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.
10. Egidi, L., Pauli, F., & Torelli, N. (2018). Combining historical data and bookmakers’ odds in modelling football scores. *Statistical Modelling*, 18(5–6), 436–459.
11. Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1–22.
12. Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
13. Goddard, J. (2005). Regression models for forecasting goals and match results in association football. *International Journal of Forecasting*, 21(2), 331–340.
14. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*.
15. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
16. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
17. Karlis, D., & Ntzoufras, I. (2003). Analysis of sports data by using bivariate Poisson models. *Journal of the Royal Statistical Society: Series D (The Statistician)*, 52(3), 381–393.
18. Ley, C., Van de Wiele, T., & Van Eetvelde, H. (2019). Ranking soccer teams on the basis of their current strength: A comparison of maximum likelihood approaches. *Statistical Modelling*, 19(1), 55–73.
19. Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.
20. Nisbet, D., Mandorino, M., Żmijewski, P., & Morgans, R. (2027). Context-dependent effects of environmental conditions on match running performance in elite soccer. *Biology of Sport*, 44(1), 59–73.
21. Owen, A. (2011). Dynamic Bayesian forecasting models of football match outcomes with estimation of the evolution variance parameter. *IMA Journal of Management Mathematics*, 22(2), 99–113.
22. Stübinger, J., Mangold, B., & Knoll, J. (2020). Machine learning in football betting: Prediction of match results based on player characteristics. *Applied Sciences*, 10(1), 46.
23. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
24. Wheatcroft, E. (2021). Evaluating probabilistic forecasts of football matches: The case against the ranked probability score. *Journal of Quantitative Analysis in Sports*, 17(4), 273–287.
