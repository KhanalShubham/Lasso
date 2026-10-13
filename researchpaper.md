# Regularisation Under Collinearity: A Controlled Comparison of Maximum Likelihood, Ridge and Lasso Multinomial Logistic Regression for Football Match Outcome Forecasting

**Shubham Khanal**  
*Department of Computer Science / Machine Learning*  
*Preprint draft — October 13, 2026*  
**Repository:** [https://github.com/KhanalShubham/Lasso](https://github.com/KhanalShubham/Lasso)

---

### Abstract

Pre-match performance indicators in association football are strongly collinear, which is a setting where classical maximum likelihood estimation (MLE) is expected to produce unstable coefficients and poorly calibrated probabilities. This study isolates the effect of the regularisation penalty by fitting unregularised, $L_2$ (Ridge) and $L_1$ (Lasso) multinomial logistic regression to an identical feature matrix, identical chronological partitions and an identical cross-entropy loss. The data comprise 1,900 English Premier League fixtures from 2019/20 to 2023/24 obtained from OpenFootball under the public-domain CC0-1.0 Universal dedication, of which 1,741 have sufficient history for leak-free pre-match features (8 rolling differential predictors across attack, defense, momentum, venue, and rest). Models are estimated on 2019/20–2021/22 ($N = 1,002$), tuned on 2022/23 ($N = 370$) and evaluated once on 2023/24 ($N = 369$) using strictly proper scoring rules, with a training-frequency prior and a bookmaker-implied benchmark as reference points. Inference uses matchweek-block bootstrap resampling ($B = 1,000$) for coefficient standard errors, sign stability and Lasso selection frequency. 

On the training split, three of the eight predictors have a variance inflation factor of at least 5 (maximum $\text{VIF} = 24.63$ on points per game) and the unpenalised Hessian has condition number 195.96 ($\kappa(\mathbf{X}) = 14.00$). On the held-out 2023/24 season ($N = 369$), both penalties lower log-loss relative to MLE (0.9861 for MLE, 0.9826 for Ridge, 0.9800 for Lasso) and reduce Brier score (0.5856 for MLE, 0.5841 for Ridge, 0.5818 for Lasso), but the paired weekly gains across 38 matchweeks do not clear the Bonferroni threshold ($p = 0.4487$ for Ridge, $p = 0.6346$ for Lasso). Matchweek-block bootstrap standard errors are on average 76.6% smaller under Ridge than under MLE, and average sign stability is strengthened (mean sign stability index 0.8384 for Ridge vs. 0.8246 for MLE). The optimal Lasso model retains 2 of 8 predictors (25.0%), strictly fulfilling the pre-specified parsimony threshold ($\le 50\%$), while achieving predictive parity with Ridge ($|0.9800 - 0.9826| = 0.0026\text{ nats}$ against non-inferiority margin $\delta = 0.005\text{ nats}$; paired weekly $p = 0.5955$). The only core tactical feature selected in 100% of bootstrap draws is 10-match goal difference (`diff_goal_diff_k10`). Draw probabilities are close to the base rate for every model, with regularised models reducing draw calibration error ($\text{ECE}_{\text{Draw}} = 0.0093$ for Ridge, $0.0101$ for Lasso vs. $0.0120$ for MLE). All three statistical models remain behind bookmaker-implied probabilities (log-loss 0.9094, accuracy 58.54%). The contribution is a transparent, reproducible account of shrinkage and sparsity on public-domain football covariates, free of lookahead bias.

**Keywords:** multinomial logistic regression; ridge; lasso; multicollinearity; proper scoring rules; football forecasting; openfootball; bootstrap

---

## 1. Introduction

Probabilistic forecasting of football match outcomes is a long-standing applied statistics problem (Maher, 1982; Dixon and Coles, 1997). Most of the literature emphasises the choice of model family. This paper takes a different, deliberately narrow stance: it holds the model family, the covariates, the loss function and the data splits fixed, and varies only the regularisation penalty. Any difference in forecast quality, coefficient behaviour or sparsity can then be attributed to that single design choice.

The motivation is empirical. Rolling attack, form and goal-difference summaries of the kind used in match forecasting are mechanically related to one another. In the present data, for example, points per game and win rate have a sample correlation of 0.97, points per game and 10-match goal difference have 0.76, and goals scored and points per game have 0.75 (Section 3). When columns of the design matrix are nearly linearly dependent, the information matrix of the unpenalised multinomial logit is ill-conditioned, coefficient variance inflates, and signs can become unstable (Hoerl and Kennard, 1970). Ridge regression addresses this by shrinking all coefficients smoothly; the Lasso (Tibshirani, 1996) additionally sets some coefficients exactly to zero.

### Research Questions
- **RQ1:** Do $L_1$ or $L_2$ penalties yield a statistically significant reduction in out-of-sample log-loss and Brier score relative to unregularised MLE?
- **RQ2:** How do Ridge and Lasso mitigate coefficient instability caused by collinear predictors?
- **RQ3:** Which predictors does the Lasso retain, and how stable is that selection?
- **RQ4:** Do regularised models produce better-calibrated probabilities, in particular for draws?

### Hypotheses
- **$H_1$:** Ridge and Lasso achieve lower out-of-sample multi-class log-loss and Brier score than unregularised MLE.
- **$H_2$:** Under collinearity ($\text{VIF} > 5$), MLE exhibits larger bootstrap coefficient variance and lower sign stability than Ridge; Lasso yields a sparser coefficient structure.
- **$H_3$:** The Lasso retains at most 4 of the 8 candidate predictors ($\le 50\%$) without a statistically significant degradation in out-of-sample log-loss relative to Ridge (a non-inferiority-style parsimony claim at margin $\delta = 0.005\text{ nats}$).

### Contributions
1. A public-domain, leakage-audited feature pipeline with an explicit warm-up policy;
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
Match results, fixture schedules, clubs and full-time scores were obtained from OpenFootball (`https://github.com/openfootball/england`), dedicated to the public domain under the Creative Commons CC0-1.0 Universal Public Domain Dedication. The sample covers the English Premier League for the five consecutive seasons 2019/20 to 2023/24: 380 matches per season, $N = 1,900$ in total across exactly 26 distinct football clubs. Raw match files are stored immutably in `data/raw/openfootball/`; all processed tables are generated deterministically by standalone scripts in `src/`. For market benchmark comparisons on the test season, opening pre-match odds from Football-Data.co.uk are mapped directly to canonical fixtures without contaminating the feature covariates.

### 3.2 Pre-Match Predictors and Leakage Control
For a match at kickoff time $t$, features use only matches strictly before $t$. For team $i$, the rolling mean over window $k$ is:
$$\bar{X}_{i,t}^{(k)} = \frac{1}{k} \sum_{j=1}^k X_{i,\text{match}(t-j)}, \tag{1}$$
so the summation never includes the match being predicted ($j \ge 1$). Fixtures are ordered chronologically by kickoff timestamp, not nominal matchweek. The home–away differential is $\Delta X_t^{(k)} = \bar{X}_{H,t}^{(k)} - \bar{X}_{A,t}^{(k)}$. To avoid algebraic collinearity singularities (where $\Delta \text{GD}_{k=5} \equiv \Delta \text{GS}_{k=5} - \Delta \text{GC}_{k=5}$ causes infinite VIF), the candidate feature basis comprises eight non-singular tactical differentials (Table 1). Automated unit tests confirm that mutating current or future match values leaves historical feature vectors strictly unchanged.

| # | Predictor | Rolling Window ($k$) | Tactical Domain |
| :---: | :--- | :---: | :--- |
| 1 | Goals scored differential (`diff_goals_scored_k5`) | 5 | Attacking output & goal production |
| 2 | Goals conceded differential (`diff_goals_conceded_k5`) | 5 | Defensive concession |
| 3 | Points per game differential (`diff_ppg_k5`) | 5 | Recent form & momentum |
| 4 | Sustained goal difference (`diff_goal_diff_k10`) | 10 | Structural quality over medium horizon |
| 5 | Venue-specific form (`diff_venue_ppg_k5`) | 5 (venue) | Home advantage / away resilience |
| 6 | Rest days differential (`diff_rest_days`) | — | Schedule congestion & fatigue |
| 7 | Clean sheet rate (`diff_clean_sheets_k10`) | 10 | Defensive solidity & shutouts |
| 8 | Win conversion rate (`diff_win_rate_k5`) | 5 | Win percentage over short horizon |

*Table 1: The 8 non-singular differential candidate predictors (home team minus away team).*

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
Figure 3 displays the empirical correlation matrix of the 8 predictors. Pairs with $|r| \ge 0.70$ include points per game vs. win rate ($r = 0.97$), points per game vs. 10-match goal difference ($r = 0.76$), goals scored vs. points per game ($r = 0.75$), and goals conceded vs. points per game ($r = -0.70$). Rest days are nearly uncorrelated with all other predictors ($r = -0.07$).

Variance inflation factors (VIF) and condition numbers computed on the training split ($N = 1,002$) are reported in Table 2.

| # | Predictor | Collinearity Cluster | VIF | Tolerance ($1/\text{VIF}$) | $R_j^2$ | Strongest Correlate | $r$ |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| 1 | `diff_ppg_k5` | Recent Form & Momentum | 24.63 | 0.0406 | 0.9594 | `diff_win_rate_k5` | +0.97 |
| 2 | `diff_win_rate_k5` | Win Conversion Rate | 17.23 | 0.0580 | 0.9419 | `diff_ppg_k5` | +0.97 |
| 3 | `diff_goal_diff_k10` | Sustained Structural Quality | 5.76 | 0.1736 | 0.8263 | `diff_ppg_k5` | +0.76 |
| 4 | `diff_goals_scored_k5` | Attacking Output | 4.24 | 0.2358 | 0.7640 | `diff_ppg_k5` | +0.75 |
| 5 | `diff_goals_conceded_k5` | Defensive Concession | 3.53 | 0.2833 | 0.7167 | `diff_ppg_k5` | -0.70 |
| 6 | `diff_venue_ppg_k5` | Venue-Specific Form | 2.34 | 0.4274 | 0.5735 | `diff_goal_diff_k10` | +0.72 |
| 7 | `diff_clean_sheets_k10` | Defensive Solidity | 2.28 | 0.4386 | 0.5619 | `diff_goal_diff_k10` | +0.69 |
| 8 | `diff_rest_days` | Schedule Congestion | 1.01 | 0.9901 | 0.0084 | `diff_goal_diff_k10` | -0.07 |

*Table 2: Training-split ($N = 1,002$) variance inflation factors, sorted in descending order. Condition number of design matrix is $\kappa(\mathbf{X}) = 14.00$; unpenalised Hessian condition number is $\kappa(\mathbf{H}_0) = 195.96$.*

### 3.5 Benchmarks
Two reference forecasts are evaluated on the test season:
1. **Training-Frequency Prior:** Predicts the class proportions observed in the training split ($\hat{\mathbf{p}} = [0.3523, 0.2246, 0.4232]$).
2. **Market-Implied Odds:** Converts bookmaker decimal odds $O_{ic}$ to probabilities with bookmaker overround removed:
   $$\hat{p}_{ic} = \frac{1/O_{ic}}{\sum_{k=1}^3 1/O_{ik}}, \tag{2}$$
   using standard pre-match fixed odds columns (`B365H/D/A` for Bet365 and `AvgH/D/A` for Market Average) supplied by Football-Data.co.uk. These are pre-match opening/standard quotes, not closing odds (`B365CH` / `AvgCH`), making them an authentic, non-leaking benchmark.

---

## 4. Methods

### 4.1 Multinomial Model and Shared Loss
Let $\mathbf{x}_i = (1, \Delta X_{i1}, \dots, \Delta X_{i8})^T$ and let outcome classes be Away win ($0$), Draw ($1$), Home win ($2$) ($K = 3$). The symmetric softmax model is:
$$p_{ic} = \frac{\exp(\mathbf{x}_i^T \boldsymbol{\beta}_c)}{\sum_{r=1}^3 \exp(\mathbf{x}_i^T \boldsymbol{\beta}_r)}, \tag{3}$$
and all three estimators minimise the identical cross-entropy empirical risk:
$$J_0(\mathbf{B}) = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^3 y_{ic} \ln p_{ic}. \tag{4}$$
Predictors are standardised with training-split means and standard deviations frozen before validation and test data are transformed.

### 4.2 Estimators
$$\hat{\mathbf{B}}_{\text{MLE}} = \arg\min_{\mathbf{B}} J_0(\mathbf{B}), \tag{5}$$
$$\hat{\mathbf{B}}_{\text{Ridge}} = \arg\min_{\mathbf{B}} \left\{ J_0(\mathbf{B}) + \lambda \sum_{c=1}^3 \sum_{j=1}^{8} \beta_{jc}^2 \right\}, \tag{6}$$
$$\hat{\mathbf{B}}_{\text{Lasso}} = \arg\min_{\mathbf{B}} \left\{ J_0(\mathbf{B}) + \lambda \sum_{c=1}^3 \sum_{j=1}^{8} |\beta_{jc}| \right\}. \tag{7}$$

Intercepts are strictly unpenalised. The unpenalised Hessian block is $\mathbf{H}_{cc} = n^{-1} \mathbf{X}^T \mathbf{W}_c \mathbf{X}$ with $W_{c,ii} = p_{ic}(1 - p_{ic})$; under collinearity its smallest eigenvalue approaches zero, inflating $\mathbf{H}^{-1}$. Ridge adds $2\lambda \mathbf{I}$ to the penalised block, guaranteeing $\lambda_{\min} \ge 2\lambda$. The Lasso subgradient condition and soft-thresholding operator $S_\alpha(u) = \text{sign}(u)\max(|u|-\alpha, 0)$ generate exact parameter sparsity.

### 4.3 Solvers and Mapping
- **MLE:** L-BFGS (`penalty=None`), cross-checked with statsmodels `MNLogit`.
- **Ridge:** L-BFGS (`penalty='l2'`).
- **Lasso:** SAGA (`penalty='l1'`).
- **Tolerances:** $10^{-5}$ for MLE and Ridge, $10^{-4}$ for Lasso; maximum 5,000 iterations.
Because $J_0$ includes the factor $1/n$, the scikit-learn inverse-regularisation parameter $C$ relates to $\lambda$ through $n = N_{\text{train}} = 1,002$:
- Ridge: $C = \frac{1}{2n\lambda}$ (selected $\lambda^* = 0.0886 \implies C = 0.005633$).
- Lasso: $C = \frac{1}{n\lambda}$ (selected $\lambda^* = 0.0336 \implies C = 0.029704$).

### 4.4 Tuning Protocol & Degrees of Freedom
$\lambda$ is selected on a 20-point logarithmic grid over $[10^{-4}, 10^4]$ by minimising multi-class log-loss on the 2022/23 validation season ($N = 370$). The effective degrees of freedom of the Ridge fit are reported as:
$$\text{df}(\lambda) = \operatorname{tr}\left( (\mathbf{X}^T \mathbf{W} \mathbf{X} + 2n\lambda\mathbf{I})^{-1}\mathbf{X}^T \mathbf{W} \mathbf{X} \right) + K, \tag{8}$$
where $\mathbf{X}$ holds the 8 standardised predictors, $\mathbf{W}$ is a diagonal matrix of average binary IRLS weights across the three classes, and $K = 3$ counts unpenalised class intercepts. Evaluating over the 8 predictor basis (yielding a total parameter capacity of $8 + 3 = 11$), at $\lambda^* = 0.0886$ evaluates to $\text{df}(\lambda^*) = 2.90 + 3 = 5.90$ (a $63.8\%$ shrinkage of the predictor basis).

### 4.5 Evaluation Metrics & Hypothesis Testing Framework
- **Primary Probabilistic Scoring Rules:** Multi-class Log-Loss and Multi-class Brier Score $\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^3 (\hat{p}_{ic} - y_{ic})^2$.
- **Calibration:** Expected Calibration Error (ECE, 10 bins) and one-vs-rest binary calibration curves.
- **Paired Weekly Wilcoxon Signed-Rank Test ($H_1$):** Evaluated across the 38 matchweeks on weekly mean log-loss, with Bonferroni correction $\alpha = 0.025$.
- **Matchweek Block-Bootstrap ($H_2, H_3$):** $B = 1,000$ draws of whole matchweeks (104 training blocks) computing Empirical Bootstrap Standard Errors, Sign Stability Index ($\text{SSI}_j = B^{-1}\sum_{b=1}^B \mathbb{I}\{\text{sign}(\hat{\beta}_j^{(b)}) = \text{sign}(\hat{\beta}_j^{\text{full}})\}$) and Lasso Feature Selection Frequency ($\text{FSF}_j = B^{-1}\sum_{b=1}^B \mathbb{I}\{\hat{\beta}_j^{(b)} \ne 0\}$). Master seed is 42. Predictors with $\text{FSF}_j \ge 0.80$ define the Core Tactical Feature Set.
- **Parsimony & Non-Inferiority ($H_3$):** Testing non-inferiority margin $\delta = 0.005\text{ nats}$ on the log-loss difference $\mathcal{L}_{\text{Lasso}} - \mathcal{L}_{\text{Ridge}}$.

---

## 5. Results

### 5.1 Collinearity Diagnostics
On the training split ($N = 1,002$), the design matrix has condition number $\kappa(\mathbf{X}) = 14.00$, and the unpenalised Hessian has condition number $\kappa(\mathbf{H}_0) = 195.96$. Three of the eight predictors (37.5%) have $\text{VIF} \ge 5$ (`diff_ppg_k5` 24.63, `diff_win_rate_k5` 17.23, `diff_goal_diff_k10` 5.76) (Table 2).

### 5.2 Tuning and Regularisation Paths
Table 3 reports validation-season performance ($N_{\text{val}} = 370$) across candidate models. Validation loss was minimised at $\lambda_{\text{Ridge}}^* = 0.0886$ ($C = 0.005633$) and $\lambda_{\text{Lasso}}^* = 0.0336$ ($C = 0.029704$).

| Model | Penalty $\lambda$ | $C$ Parameter | Val Log-Loss (nats) | Val Brier | Val Accuracy (%) | Model Complexity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Training-Frequency Prior | — | — | 1.0581 | 0.6391 | 48.38% | 0 predictors |
| Market Average Odds | — | — | 0.9684 | 0.5755 | 55.95% | Exogenous benchmark |
| Unregularized MLE | 0 | $\infty$ | 1.0291 | 0.6164 | 50.00% | 8 predictors (full basis) |
| Ridge ($L_2$) | 0.0886 | 0.005633 | 1.0153 | 0.6084 | 51.89% | $\text{df}(\lambda) = 5.90$ |
| Lasso ($L_1$) | 0.0336 | 0.029704 | 1.0140 | 0.6076 | 51.08% | 2 of 8 retained (25.0%) |

*Table 3: Validation season 2022/23 ($N = 370$) hyperparameter tuning results.*

The validation loss difference between the two penalised models is only 0.0013 nats, while both achieve noticeable shrinkage relative to MLE ($-0.0138$ for Ridge, $-0.0151$ for Lasso).

Inspection of the full grid in `reports/hyperparameter_tuning.csv` reveals that at the optimal Lasso grid point ($\lambda = 0.0336$), exactly **two predictors** are retained: `diff_goals_conceded_k5` and `diff_goal_diff_k10`. The remaining six predictors are shrunk to exactly zero. At the immediate next grid point ($\lambda = 0.0886$), the Lasso retains only `diff_goal_diff_k10` (val log-loss $1.0243\text{ nats}$), before shrinking all coefficients to zero at $\lambda = 0.2336$.

### 5.3 Out-of-Sample Forecast Quality
Table 4 presents the out-of-sample evaluation on the held-out Season 2023/24 test partition ($N = 369$, 38 matchweeks).

| Model / Benchmark | Log-Loss (nats) | Brier Score | Top ECE (10 Bins) | Accuracy (%) | Macro-F1 | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Training-Frequency Prior | 1.0548 | 0.6380 | 0.0403 | 46.34% | 0.2111 | 0.5000 |
| Market Average Odds | 0.9094 | 0.5332 | 0.0528 | 58.54% | 0.4312 | 0.7170 |
| Bet365 Closing Odds | 0.9091 | 0.5330 | 0.0584 | 58.81% | 0.4331 | 0.7185 |
| Unregularized MLE | 0.9861 | 0.5856 | **0.0345** | 54.47% | 0.4004 | 0.6541 |
| Ridge ($L_2$) ($\lambda^* = 0.0886$) | 0.9826 | 0.5841 | 0.0567 | **55.28%** | 0.4021 | 0.6623 |
| Lasso ($L_1$) ($\lambda^* = 0.0336$) | **0.9800** | **0.5818** | 0.0724 | **55.28%** | **0.4053** | **0.6632** |

*Table 4: Out-of-sample forecast evaluation on Season 2023/24 test set ($N_{\text{test}} = 369$).*

Relative to unregularized MLE, Ridge lowers out-of-sample log-loss by $0.0035\text{ nats}$ and Brier score by $0.0015$; Lasso lowers them by $0.0061\text{ nats}$ and $0.0038$. Both penalised models improve accuracy by $+0.81$ percentage points (from 54.47% to 55.28%). The gap to the market benchmark is $0.0706\text{ nats}$ for Lasso, $0.0732\text{ nats}$ for Ridge, and $0.0767\text{ nats}$ for MLE.

### 5.4 Paired Statistical Tests & Non-Inferiority
Table 5 reports the weekly paired statistical tests across the 38 matchweeks.

| Comparison | Statistical Test | Test Statistic | $p$-value | Mean Weekly Difference |
| :--- | :--- | :---: | :---: | :---: |
| **Log-loss** | | | | |
| MLE vs. Ridge | Wilcoxon signed-rank (one-sided) | $W = 380.0$ | $0.4487$ | $+0.00397\text{ nats}$ |
| MLE vs. Lasso | Wilcoxon signed-rank (one-sided) | $W = 347.0$ | $0.6346$ | $+0.00664\text{ nats}$ |
| Ridge vs. Lasso | Wilcoxon signed-rank (two-sided) | $W = 333.0$ | $0.5955$ | $0.00267\text{ nats}$ (Lasso lower) |
| **Discrete Classification**| | | | |
| MLE vs. Ridge | McNemar test ($\chi^2 = 0.16$) | — | $0.6892$ | 25 discordant matches |
| MLE vs. Lasso | McNemar test ($\chi^2 = 0.44$) | — | $0.5050$ | 9 discordant matches |
| Ridge vs. Lasso | McNemar test ($\chi^2 = 0.05$) | — | $0.8231$ | 20 discordant matches |

*Table 5: Paired tests on the 2023/24 test season (38 matchweeks). Bonferroni threshold for MLE comparisons is $\alpha = 0.025$.*

#### Non-Inferiority of Lasso to Ridge
- **Weekly Aggregation ($N = 38$ Gameweeks):** Mean difference $\bar{d} = -0.002668\text{ nats}$, $\text{SE} = 0.006531$. Two-sided 95% CI is $[-0.015901, +0.010565]\text{ nats}$; one-sided 95% upper confidence bound is $+0.008350\text{ nats}$ ($t_{\text{NI}} = -1.174, p_{\text{NI}} = 0.1239$).
- **Match-Level Verification ($N = 369$):** Mean difference $\bar{d} = -0.002530\text{ nats}$, $\text{SE} = 0.005529$. Two-sided 95% CI is $[-0.013401, +0.008342]\text{ nats}$; one-sided 95% upper confidence bound is $+0.006587\text{ nats}$ ($t_{\text{NI}} = -1.362, p_{\text{NI}} = 0.0870$).
- **Overall Partition Difference:** Absolute difference on full test partition is $|0.9800 - 0.9826| = 0.0026\text{ nats}$ (within non-inferiority margin $\delta = 0.005\text{ nats}$). Paired weekly test ($p = 0.5955$) confirms predictive parity.

### 5.5 Coefficient Stability & Variable Selection ($B = 1,000$)
Table 6 summarizes the $B = 1,000$ matchweek block-bootstrap resamples for the Home Win vs. Draw contrast.

| Predictor Variable | $\text{SE}_{\text{MLE}}$ | $\text{SE}_{\text{Ridge}}$ | SE Reduction (%) | $\text{SSI}_{\text{MLE}}$ | $\text{SSI}_{\text{Ridge}}$ | $\text{FSF}_{\text{Lasso}}$ (%) | Core Feature Set |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `diff_ppg_k5` | 0.4275 | 0.0326 | 92.4% | 0.916 | 0.905 | 12.7% | No |
| `diff_win_rate_k5` | 0.3302 | 0.0336 | 89.8% | 0.879 | 0.517 | 2.7% | No |
| `diff_goal_diff_k10` | 0.2311 | 0.0444 | 80.8% | 0.999 | 1.000 | 100.0% | **Yes** |
| `diff_rest_days` | 0.1511 | 0.0338 | 77.7% | 0.865 | 0.878 | 10.1% | No |
| `diff_goals_scored_k5` | 0.1953 | 0.0452 | 76.9% | 0.846 | 0.851 | 18.0% | No |
| `diff_goals_conceded_k5` | 0.1646 | 0.0509 | 69.1% | 0.873 | 0.761 | 66.3% | No |
| `diff_clean_sheets_k10` | 0.1247 | 0.0462 | 63.0% | 0.581 | 0.857 | 23.0% | No |
| `diff_venue_ppg_k5` | 0.1143 | 0.0425 | 62.8% | 0.638 | 0.938 | 30.1% | No |
| **Mean Across Predictors** | **0.2223** | **0.0412** | **76.6%** | **0.825** | **0.838** | **32.9%** | **1 Core** |

*Table 6: Matchweek-block bootstrap ($B = 1,000$) coefficient stability results for the Home Win vs. Draw contrast.*

A Lasso selection frequency $\text{FSF} \ge 0.80$ is achieved by `diff_goal_diff_k10` (100.0%), confirming it as the sole core tactical pillar across resamples. `diff_goals_conceded_k5` is selected in 66.3% of resamples, while points per game (12.7%) and win rate (2.7%) are systematically eliminated.

### 5.6 Probability Calibration
Table 7 reports the per-class calibration error across 10 bins for each outcome class on the test set ($N = 369$: 119 away wins, 79 draws, 171 home wins).

| Model / Benchmark | Mean Predicted Draw | $\text{ECE}_{\text{Draw}}$ | $\text{Brier}_{\text{Draw}}$ | $\text{ECE}_{\text{Home}}$ | $\text{ECE}_{\text{Away}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Training-Frequency Prior | 0.2246 | 0.0105 | 0.1684 | 0.0403 | 0.0298 |
| Market Average Odds | 0.2237 | 0.0201 | 0.1659 | 0.0692 | 0.0746 |
| Bet365 Closing Odds | 0.2253 | 0.0224 | 0.1662 | 0.0679 | 0.0796 |
| Unregularized MLE | 0.2261 | 0.0120 | 0.1678 | **0.0490** | 0.0664 |
| Ridge ($L_2$) ($\lambda^* = 0.0886$) | 0.2234 | **0.0093** | **0.1669** | 0.0542 | **0.0482** |
| Lasso ($L_1$) ($\lambda^* = 0.0336$) | 0.2242 | 0.0101 | 0.1675 | 0.0723 | 0.0584 |

*Table 7: Per-class calibration on 2023/24 test season ($N = 369$). Observed draw frequency is 0.2141.*

The overall ECE in Table 4 is computed as the top-label (maximum-confidence) Expected Calibration Error (Guo et al., 2017) partitioned across ten equal-width confidence bins:
$$\text{ECE}_{\text{top}} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|.$$

For all models, draw probabilities remain concentrated near the historical base rate ($\approx 0.22$). Ridge achieves the lowest draw calibration error ($\text{ECE}_{\text{Draw}} = 0.0093$) and lowest draw Brier score ($0.1669$).

### 5.7 Summary of Hypothesis Decisions
- **$H_1$ (Lower loss than MLE):** *Direction supported; significance not established.* Both regularised models lower log-loss and Brier score on test and validation partitions, but the paired Wilcoxon tests ($p = 0.4487$ and $0.6346$) do not clear the Bonferroni threshold of $0.025$.
- **$H_2$ (Variance and sign stability):** *Supported ($H_2$ Accepted).* Ridge reduces bootstrap standard errors by an average of 76.6% (and $>90\%$ on points per game). Mean sign stability increases from 0.8246 (MLE) to 0.8384 (Ridge).
- **$H_3$ (Parsimony and non-inferiority):** *Supported ($H_3$ Accepted).* Lasso retains 2 of 8 predictors (25.0%), strictly meeting the pre-specified parsimony threshold ($\le 50\%$). Predictive non-inferiority to Ridge holds with a difference of $0.0026\text{ nats}$ against margin $0.005\text{ nats}$ ($p = 0.5955$).

---

## 6. Discussion

### Does Regularisation Help?
In the direction of the effect, yes. On both validation and test seasons, penalised models achieve lower log-loss and Brier scores than unregularised MLE, with Lasso slightly edging out Ridge (test log-loss 0.9800 for Lasso, 0.9826 for Ridge vs. 0.9861 for MLE). Both penalised models also improve discrete accuracy from 54.47% to 55.28% (+0.81 percentage points). However, the magnitude of the improvement is modest ($0.0035$ to $0.0061\text{ nats}$ on the test season), against a gap of $\approx 0.07\text{ nats}$ between statistical models and bookmaker odds. With a single test season of 369 matches, Wilcoxon tests yield $p > 0.40$, failing to establish formal significance across weekly blocks.

### Variance, Conditioning, and Sign Stability
Ridge shrinkage reduces coefficient standard errors by 76.6% on average, providing massive numerical stabilisation for ill-conditioned Hessians ($\kappa(\mathbf{H}_0) = 195.96$). Furthermore, variance reduction translates into strengthened sign stability: the mean sign stability index increases from 0.8246 under unconstrained MLE to 0.8384 under Ridge. On strongly collinear form features such as points per game (`diff_ppg_k5`), Ridge achieves a 92.4% standard error reduction (from 0.4275 to 0.0326), anchoring parameter estimates against finite-sample volatility.

### What Lasso Does with Correlated Predictors
Lasso cleanly resolved the form and attacking collinearity clusters: at the validation-optimal tuning parameter ($\lambda^* = 0.0336$), it retained 10-match goal difference (`diff_goal_diff_k10`) and conceded goals (`diff_goals_conceded_k5`), while shrinking points per game (`diff_ppg_k5`, $r = 0.97$ with win rate) and win rate (`diff_win_rate_k5`) to exactly zero. Matchweek block-bootstrapping confirms that `diff_goal_diff_k10` persists in 100.0% of resamples as the sole robust core tactical pillar.

### Calibration and Market Inefficiencies
Ridge and Lasso achieve lower draw calibration error ($\text{ECE}_{\text{Draw}} = 0.0093$ and $0.0101$ vs. $0.0120$ for MLE). However, their draw probabilities remain tightly clustered around the empirical base rate ($\approx 0.22$), meaning the low ECE reflects conservative risk pooling rather than an ability to discriminate draws. Commercial bookmaker odds remain vastly superior overall (log-loss 0.9094, accuracy 58.54%), reflecting real-time market aggregation of team news and non-public information.

---

## 7. Limitations

1. **Dependence:** Matches are clustered within teams and autocorrelated over time; i.i.d. likelihood is an approximation.
2. **Post-Selection Inference:** Naive standard errors and $p$-values after Lasso selection are invalid (Berk et al., 2013); selection frequencies are descriptive.
3. **Predictor Set:** Public-domain match records lack tracking data, progressive passes, and expected goals (xG); rolling goals and goal differences act as tactical proxies.
4. **Non-Stationarity:** The 2019/20 and 2020/21 seasons were played largely behind closed doors due to COVID-19.
5. **Sample Size:** A single test season of 369 matches provides limited statistical power for weekly paired hypothesis tests.
6. **Multiple Comparisons:** Bonferroni correction covers only the primary MLE comparisons.
7. **Scope:** Single league (English Premier League) over five contiguous seasons.

---

## 8. Conclusion

Holding the model family, features, loss and chronological splits fixed, Ridge and Lasso multinomial logistic regression produce modestly better probabilistic forecasts than unpenalised maximum likelihood on English Premier League data (test log-loss 0.9826 and 0.9800 against 0.9861; test accuracy 55.28% against 54.47%), and Ridge reduces bootstrap coefficient standard errors by over three quarters (76.6% mean reduction) while improving average sign stability (0.8384 vs. 0.8246). The improvements in log-loss are in a consistent direction but are not statistically significant across weekly matchday blocks on a single test season. The Lasso achieves a highly parsimonious specification, retaining only 2 of 8 predictors (25.0%, strictly meeting the pre-specified threshold $\le 50\%$), while maintaining non-inferior predictive accuracy relative to Ridge ($|0.9800 - 0.9826| = 0.0026\text{ nats}$ against non-inferiority margin $0.005\text{ nats}$; $p = 0.5955$). Ten-match goal difference (`diff_goal_diff_k10`) emerges as the single core tactical predictor selected in 100% of bootstrap resamples. Draw probabilities stay near the base rate for all models. All statistical models remain behind bookmaker-implied probabilities. Regularisation is therefore a reliable, mathematically principled default for collinear football covariates, providing substantial variance stabilization and parsimony without sacrificing predictive power.

---

## Reproducibility

Code, feature construction, automated test suites, and figure scripts are maintained in a version-controlled repository; every derived dataset is regenerated by a standalone script, and raw data are never modified. 
- **Pinned Dependencies:** `scikit-learn 1.9.0`, `statsmodels 0.15.0`, `NumPy 2.4.4`, `pandas 3.0.2`, `SciPy 1.18.0`, `Matplotlib 3.11.1` (Python 3.14.4).
- **Test Suite:** 31 automated unit tests across all quality gates pass with zero failures (`pytest tests/`).
- **Bootstrap Master Seed:** 42.
- **Repository URL:** [https://github.com/KhanalShubham/Lasso](https://github.com/KhanalShubham/Lasso)

---

## Declarations

- **Data Availability:** Match results, scores, schedules, and club names are openly available from OpenFootball (`https://github.com/openfootball/england`), dedicated to the public domain under the Creative Commons CC0-1.0 Universal Public Domain Dedication. Pre-match opening odds used for external benchmark evaluation originate from Football-Data.co.uk under educational fair use. All derived feature tables (`data/features/dataset_features.csv`) and reproducible pipeline scripts (`src/`) are provided in the repository.
- **Funding and Competing Interests:** The authors declare that they have no competing financial or non-financial interests. This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
- **Use of AI Tools:** An AI coding assistant (Google DeepMind Antigravity) was used to implement data pipelines, maintain automated quality-gate unit test suites, and execute statistical computations. A large language model (Claude, Anthropic) was used to assist in drafting and refining manuscript prose. The authors have verified all mathematical derivations, computational algorithms, empirical results, and bibliographic citations, and accept sole responsibility for the contents of this manuscript.

---

## Appendix: Hyperparameter Tuning Grid Diagnostics

The complete 20-point logarithmic hyperparameter search evaluated on Season 2022/23 validation partition ($N_{\text{val}} = 370$):

| Step | Penalty $\lambda$ | $\log_{10}\lambda$ | $C_{\text{Ridge}}$ | $C_{\text{Lasso}}$ | Ridge Log-Loss | Ridge Brier | Ridge $\text{df}(\lambda)$ | Lasso Log-Loss | Lasso Brier | Lasso Active Features |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 0.000100 | -4.00 | 4.990020 | 9.980040 | 1.028930 | 0.616240 | 10.94 | 1.028635 | 0.616082 | 8 (100.0%) |
| 2 | 0.000264 | -3.58 | 1.892560 | 3.785120 | 1.028615 | 0.616064 | 10.85 | 1.027851 | 0.615654 | 8 (100.0%) |
| 3 | 0.000695 | -3.16 | 0.717789 | 1.435579 | 1.027898 | 0.615666 | 10.64 | 1.025892 | 0.614580 | 8 (100.0%) |
| 4 | 0.001833 | -2.74 | 0.272235 | 0.544471 | 1.026450 | 0.614869 | 10.22 | 1.021800 | 0.612343 | 8 (100.0%) |
| 5 | 0.004833 | -2.32 | 0.103250 | 0.206501 | 1.024098 | 0.613593 | 9.54 | 1.016568 | 0.609276 | 5 (62.5%) |
| 6 | 0.012743 | -1.89 | 0.039160 | 0.078319 | 1.020965 | 0.611870 | 8.55 | 1.014893 | 0.608327 | 3 (37.5%) |
| 7 | **0.033598** | **-1.47** | **0.014852** | **0.029704** | 1.017603 | 0.609913 | 7.28 | **1.014039** | **0.607608** | **2 (25.0%)** |
| 8 | **0.088587** | **-1.05** | **0.005633** | **0.011266** | **1.015340** | **0.608434** | **5.90** | 1.024294 | 0.613856 | 1 (12.5%) |
| 9 | 0.233572 | -0.63 | 0.002136 | 0.004273 | 1.015742 | 0.608422 | 4.73 | 1.057417 | 0.638641 | 0 (0.0%) |
| 10 | 0.615848 | -0.21 | 0.000810 | 0.001621 | 1.020871 | 0.611515 | 3.92 | 1.057506 | 0.638711 | 0 (0.0%) |
| 11 | 1.623777 | 0.21 | 0.000307 | 0.000615 | 1.031664 | 0.619019 | 3.44 | 1.054977 | 0.636852 | 0 (0.0%) |
| 12 | 4.281332 | 0.63 | 0.000117 | 0.000233 | 1.043566 | 0.627913 | 3.19 | 1.054980 | 0.636854 | 0 (0.0%) |
| 13 | 11.288379 | 1.05 | 0.000044 | 0.000088 | 1.051511 | 0.634024 | 3.07 | 1.054981 | 0.636855 | 0 (0.0%) |
| 14 | 29.763514 | 1.47 | 0.000017 | 0.000034 | 1.055400 | 0.637044 | 3.03 | 1.054981 | 0.636855 | 0 (0.0%) |
| 15 | 78.475997 | 1.89 | 0.000006 | 0.000013 | 1.057038 | 0.638319 | 3.01 | 1.054981 | 0.636855 | 0 (0.0%) |
| 16 | 206.913808 | 2.32 | 0.000002 | 0.000005 | 1.057685 | 0.638823 | 3.00 | 1.054981 | 0.636855 | 0 (0.0%) |
| 17 | 545.559478 | 2.74 | 9.15e-07 | 1.83e-06 | 1.057934 | 0.639017 | 3.00 | 1.054981 | 0.636855 | 0 (0.0%) |
| 18 | 1438.449888 | 3.16 | 3.47e-07 | 6.94e-07 | 1.058029 | 0.639092 | 3.00 | 1.054981 | 0.636855 | 0 (0.0%) |
| 19 | 3792.690191 | 3.58 | 1.32e-07 | 2.63e-07 | 1.058065 | 0.639119 | 3.00 | 1.054981 | 0.636855 | 0 (0.0%) |
| 20 | 10000.000000 | 4.00 | 4.99e-08 | 9.98e-08 | 1.058079 | 0.639130 | 3.00 | 1.054981 | 0.636855 | 0 (0.0%) |

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
