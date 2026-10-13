# Empirical Companion Guide for Dissertation Manuscript (`manuscript.tex`)

**Project:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Dataset:** English Premier League 2019/20 to 2023/24 (OpenFootball Public-Domain Data, CC0-1.0 Universal)  
**Partitions:** Train = 1,002 (2019/20–2021/22), Validation = 370 (2022/23), Test = 369 (2023/24)  
**Status:** All empirical values verified from executed pipeline with zero placeholder numbers.

---

## 1. Table 2: Multicollinearity and Variance Inflation Factor (VIF) Diagnostics

**Context:** Section 3.4 / Table 2 in `manuscript.tex`.  
**Computed strictly on:** Training partition ($N = 1,002$, standardized design matrix $\mathbf{X}$).  
**Condition Numbers:**
- Scaled Design Matrix Condition Number: $\kappa(\mathbf{X}) = 14.00$
- Unpenalized Hessian Matrix Condition Number: $\kappa(\mathbf{H}_0) = 195.96$
- Collinear Predictors ($\text{VIF} \ge 5.0$): 3 of 8 features (`diff_ppg_k5`, `diff_win_rate_k5`, `diff_goal_diff_k10`)
- Highly Collinear Predictors ($\text{VIF} \ge 10.0$): 2 of 8 features

### LaTeX Table Code:

```latex
\begin{table}[htbp]
\centering
\small
\caption{Variance Inflation Factor (VIF), Tolerance, and Auxiliary $R^2$ for OpenFootball Candidate Predictors ($N_{\text{train}} = 1,002$)}
\label{tab:vif}
\begin{tabular}{llrrr}
\toprule
\textbf{Predictor Variable} & \textbf{Tactical Domain} & \textbf{VIF} & \textbf{Tolerance ($1/\text{VIF}$)} & $\mathbf{R_j^2}$ \\
\midrule
\texttt{diff\_ppg\_k5}              & Form / Points Momentum & 24.63 & 0.0406 & 0.9594 \\
\texttt{diff\_win\_rate\_k5}         & Win Conversion Rate   & 17.23 & 0.0580 & 0.9419 \\
\texttt{diff\_goal\_diff\_k10}      & Sustained Goal Diff    &  5.76 & 0.1736 & 0.8263 \\
\texttt{diff\_goals\_scored\_k5}     & Attacking Output       &  4.24 & 0.2358 & 0.7640 \\
\texttt{diff\_goals\_conceded\_k5}   & Defensive Concession   &  3.53 & 0.2833 & 0.7167 \\
\texttt{diff\_venue\_ppg\_k5}       & Venue-Specific Form    &  2.34 & 0.4274 & 0.5735 \\
\texttt{diff\_clean\_sheets\_k10}   & Defensive Solidity     &  2.28 & 0.4386 & 0.5619 \\
\texttt{diff\_rest\_days}           & Rest / Congestion      &  1.01 & 0.9901 & 0.0084 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 2. Section 5.2: Hyperparameter Tuning & Shrinkage Mechanics (Season 2022/23, $N_{\text{val}} = 370$)

- **Tuning Grid:** 20 logarithmically spaced points spanning $\lambda \in [10^{-4}, 10^4]$.
- **Solver Specifications:** L-BFGS for MLE and Ridge (`tol=1e-5`, `max_iter=5000`); SAGA for Lasso (`tol=1e-4`, `max_iter=5000`).
- **Unpenalized Baseline MLE:** Validation Log-Loss = $1.0291\text{ nats}$, Brier Score = $0.6164$, Accuracy = $50.00\%$.
- **Optimal Ridge ($L_2$):**
  - $\lambda^* = 0.088587 \implies C = 0.005633$
  - Validation Log-Loss = **$1.0153\text{ nats}$** (reduction of $-0.0138\text{ nats}$)
  - Validation Brier Score = **$0.6084$**
  - Effective Degrees of Freedom: $\text{df}(\lambda^*) = 5.90$ (out of maximum 11)
- **Optimal Lasso ($L_1$):**
  - $\lambda^* = 0.033598 \implies C = 0.029704$
  - Validation Log-Loss = **$1.0140\text{ nats}$** (reduction of $-0.0151\text{ nats}$)
  - Validation Brier Score = **$0.6076$**
  - Exact Feature Retention: **2 of 8 candidate predictors** (25.0% retained, 75.0% exact sparsity)
  - Retained: `diff_goals_conceded_k5` and `diff_goal_diff_k10`.
  - Shrunk to Zero: `diff_goals_scored_k5`, `diff_ppg_k5`, `diff_venue_ppg_k5`, `diff_rest_days`, `diff_clean_sheets_k10`, `diff_win_rate_k5`.

---

## 3. Table 4: Out-of-Sample Forecast Evaluation (Season 2023/24, $N_{\text{test}} = 369$)

```latex
\begin{table}[htbp]
\centering
\small
\caption{Out-of-Sample Forecast Evaluation on Season 2023/24 Test Partition ($N_{\text{test}} = 369$, 38 Matchweeks)}
\label{tab:test_metrics}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Model / Benchmark} & \textbf{Log-Loss} & \textbf{Brier} & \textbf{ECE} & \textbf{Acc. (\%)} & \textbf{Macro-F1} & \textbf{ROC-AUC} \\
\midrule
Training Frequency Prior  & 1.0548 & 0.6380 & 0.0403 & 46.34\% & 0.2111 & 0.5000 \\
Market Average Odds       & 0.9094 & 0.5332 & 0.0528 & 58.54\% & 0.4312 & 0.7170 \\
Bet365 Closing Odds       & 0.9091 & 0.5330 & 0.0584 & 58.81\% & 0.4331 & 0.7185 \\
Unregularized MLE         & 0.9861 & 0.5856 & 0.0345 & 54.47\% & 0.4004 & 0.6541 \\
Ridge ($L_2$) ($\lambda^* = 0.0886$) & 0.9826 & 0.5841 & 0.0567 & 55.28\% & 0.4021 & 0.6623 \\
Lasso ($L_1$) ($\lambda^* = 0.0336$) & \textbf{0.9800} & \textbf{0.5818} & 0.0724 & \textbf{55.28\%} & \textbf{0.4053} & \textbf{0.6632} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 4. Table 5: Hypothesis Testing Decisions & Paired Tests

### Weekly Paired Wilcoxon Signed-Rank Tests (38 Matchweeks):
1. **MLE vs. Ridge ($L_2$):**
   - $W = 380.0$, $p = 0.4487$ (Bonferroni $\alpha = 0.025$). Mean weekly loss reduction: $+0.00397\text{ nats}$.
   - Decision: Fail to reject null.
2. **MLE vs. Lasso ($L_1$):**
   - $W = 347.0$, $p = 0.6346$ (Bonferroni $\alpha = 0.025$). Mean weekly loss reduction: $+0.00664\text{ nats}$.
   - Decision: Fail to reject null.
3. **Ridge vs. Lasso (Predictive Parity):**
   - $W = 333.0$, $p = 0.5955$ (two-sided, $\alpha = 0.05$). Mean weekly difference: $0.00267\text{ nats}$ in favor of Lasso.
   - Decision: Retain null of predictive parity.

### McNemar Classification Divergence Tests:
1. **MLE vs. Ridge:** Discordant matches = 25 ($11$ vs. $14$), $\chi^2 = 0.1600$, $p = 0.6892$.
2. **MLE vs. Lasso:** Discordant matches = 9 ($3$ vs. $6$), $\chi^2 = 0.4444$, $p = 0.5050$.
3. **Ridge vs. Lasso:** Discordant matches = 20 ($10$ vs. $10$), $\chi^2 = 0.0500$, $p = 0.8231$.

### Non-Inferiority Testing of Lasso to Ridge ($\delta = 0.005\text{ nats}$):
- Full Test Partition Difference: $|\mathcal{L}_{\text{Lasso}} - \mathcal{L}_{\text{Ridge}}| = |0.9800 - 0.9826| = \mathbf{0.0026\text{ nats}}$ (strictly within margin $\delta = 0.005$).
- Weekly Aggregation ($N=38$ Gameweeks): $\bar{d} = -0.002668\text{ nats}$, $\text{SE} = 0.006531$, 95% CI is $[-0.015901, +0.010565]\text{ nats}$, $t_{\text{NI}} = -1.174, p_{\text{NI}} = 0.1239$.
- Match-Level Aggregation ($N=369$): $\bar{d} = -0.002530\text{ nats}$, $\text{SE} = 0.005529$, 95% CI is $[-0.013401, +0.008342]\text{ nats}$, $t_{\text{NI}} = -1.362, p_{\text{NI}} = 0.0870$.

---

## 5. Table 6: Matchweek Block-Bootstrap Resampling ($B = 1,000$ Draws)

```latex
\begin{table}[htbp]
\centering
\small
\caption{Matchweek Block-Bootstrap ($B = 1,000$) Coefficient Stability and Feature Selection Frequencies}
\label{tab:bootstrap}
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Predictor Variable} & $\mathbf{\text{SE}_{\text{MLE}}}$ & $\mathbf{\text{SE}_{\text{Ridge}}}$ & \textbf{SE Red. (\%)} & $\mathbf{\text{SSI}_{\text{MLE}}}$ & $\mathbf{\text{SSI}_{\text{Ridge}}}$ & $\mathbf{\text{FSF}_{\text{Lasso}}}$ & \textbf{Core Set} \\
\midrule
\texttt{diff\_ppg\_k5}            & 0.4275 & 0.0326 & 92.4\% & 0.916 & 0.905 & 12.7\% & No \\
\texttt{diff\_win\_rate\_k5}       & 0.3302 & 0.0336 & 89.8\% & 0.879 & 0.517 &  2.7\% & No \\
\texttt{diff\_goal\_diff\_k10}    & 0.2311 & 0.0444 & 80.8\% & 0.999 & 1.000 & 100.0\% & \textbf{Yes} \\
\texttt{diff\_rest\_days}         & 0.1511 & 0.0338 & 77.7\% & 0.865 & 0.878 & 10.1\% & No \\
\texttt{diff\_goals\_scored\_k5}   & 0.1953 & 0.0452 & 76.9\% & 0.846 & 0.851 & 18.0\% & No \\
\texttt{diff\_goals\_conceded\_k5} & 0.1646 & 0.0509 & 69.1\% & 0.873 & 0.761 & 66.3\% & No \\
\texttt{diff\_clean\_sheets\_k10} & 0.1247 & 0.0462 & 63.0\% & 0.581 & 0.857 & 23.0\% & No \\
\texttt{diff\_venue\_ppg\_k5}     & 0.1143 & 0.0425 & 62.8\% & 0.638 & 0.938 & 30.1\% & No \\
\midrule
\textbf{Mean Across Predictors}   & \textbf{0.2223} & \textbf{0.0412} & \textbf{76.6\%} & \textbf{0.825} & \textbf{0.838} & \textbf{32.9\%} & \textbf{1 Core} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 6. Table 7: One-vs-Rest Probability Calibration

```latex
\begin{table}[htbp]
\centering
\small
\caption{One-vs-Rest Reliability and Calibration Metrics on Season 2023/24 Test Partition ($N_{\text{test}} = 369$)}
\label{tab:calibration}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Model / Benchmark} & \textbf{Mean Pred. Draw} & $\mathbf{\text{ECE}_{\text{Draw}}}$ & $\mathbf{\text{Brier}_{\text{Draw}}}$ & $\mathbf{\text{ECE}_{\text{Home}}}$ & $\mathbf{\text{ECE}_{\text{Away}}}$ \\
\midrule
Training Frequency Prior  & 0.2246 & 0.0105 & 0.1684 & 0.0403 & 0.0298 \\
Market Average Odds       & 0.2237 & 0.0201 & 0.1659 & 0.0692 & 0.0746 \\
Bet365 Closing Odds       & 0.2253 & 0.0224 & 0.1662 & 0.0679 & 0.0796 \\
Unregularized MLE         & 0.2261 & 0.0120 & 0.1678 & \textbf{0.0490} & 0.0664 \\
Ridge ($L_2$) ($\lambda^* = 0.0886$) & 0.2234 & \textbf{0.0093} & \textbf{0.1669} & 0.0542 & \textbf{0.0482} \\
Lasso ($L_1$) ($\lambda^* = 0.0336$) & 0.2242 & 0.0101 & 0.1675 & 0.0723 & 0.0584 \\
\bottomrule
\end{tabular}
\end{table}
```
