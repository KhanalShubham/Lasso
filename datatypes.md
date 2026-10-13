# Data Types & Schema Contracts (datatypes.md)

**Project:** MLE vs Ridge vs Lasso Logistic Regression for Football Match Outcome Prediction  
**Status:** Canonical Data Contract  
**Authority:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md) | [RESEARCH_METHODOLOGY_GUIDE.md](file:///d:/MLE-Lasso/RESEARCH_METHODOLOGY_GUIDE.md#c8--final-modelling-dataset-structure-dataset_finalcsv)  

---

## 1. Feature Dataset Schema (`data/features/dataset_features.csv`)

The compiled tabular dataset contains exactly $N = 1,900$ rows ($N = 1,741$ modelling-ready, $N = 159$ warm-up). All models ingest the identical feature matrix and target columns:

### 1.1 Match Metadata & Identifiers

| Column Name | Data Type | Nullable | Example | Description |
| :--- | :---: | :---: | :--- | :--- |
| `match_id` | `string` | No | `"EPL_2023_2024_20230811_BUR_MAN"` | Unique deterministic match key. |
| `season` | `string` | No | `"2023-2024"` | Season string identifier. |
| `date` | `string` (ISO-8601) | No | `"2023-08-11"` | Match date (`YYYY-MM-DD`). |
| `kickoff_time` | `string` | No | `"20:00"` | Match kickoff time (`HH:MM`). |
| `home_team` | `string` | No | `"Burnley"` | Normalized home club name. |
| `away_team` | `string` | No | `"Manchester City"` | Normalized away club name. |
| `is_modelling_ready` | `bool` | No | `True` | Flag: True if k >= 10 overall & k >= 5 venue for both clubs. |

### 1.2 Ground Truth Targets (Dependent Variables)

| Column Name | Data Type | Nullable | Domain / Values | Description |
| :--- | :---: | :---: | :---: | :--- |
| `home_goals` | `int64` | No | $[0, \infty)$ | Actual full-time goals scored by home club. |
| `away_goals` | `int64` | No | $[0, \infty)$ | Actual full-time goals scored by away club. |
| `target_multinomial` | `int64` | No | $\{0, 1, 2\}$ | Primary target: `0`: Away Win, `1`: Draw, `2`: Home Win. |
| `target_binary` | `int64` | No | $\{0, 1\}$ | Benchmark target: `1`: Home Win, `0`: Non-Home Win. |

### 1.3 Candidate Pre-Match Predictors ($p = 16$ Standardized Differentials)

All differential features represent $\Delta X = X_{\text{Home}} - X_{\text{Away}}$, standardized to zero mean and unit variance ($z = (x - \mu)/\sigma$):

| Column Name | Data Type | Nullable | Window ($k$) | Tactical Role |
| :--- | :---: | :---: | :---: | :--- |
| `diff_sot_k5` | `float64` | No | 5 matches | Differential in rolling shots on target. |
| `diff_sotA_k5` | `float64` | No | 5 matches | Differential in rolling shots on target conceded. |
| `diff_shots_k5` | `float64` | No | 5 matches | Differential in rolling total shots attempted. |
| `diff_shotsA_k5` | `float64` | No | 5 matches | Differential in rolling total shots conceded. |
| `diff_sot_ratio_k5` | `float64` | No | 5 matches | Differential in shot accuracy ratio (`SoT / Shots`). |
| `diff_shot_dominance_k5` | `float64` | No | 5 matches | Differential in shot share (`Shots / (Shots + OppShots)`). |
| `diff_corners_k5` | `float64` | No | 5 matches | Differential in corner kicks won (Attacking 3rd pressure). |
| `diff_cornersA_k5` | `float64` | No | 5 matches | Differential in corner kicks conceded. |
| `diff_fouls_k5` | `float64` | No | 5 matches | Differential in fouls committed (Defensive aggression). |
| `diff_yellows_k5` | `float64` | No | 5 matches | Differential in yellow cards received (Disciplinary risk). |
| `diff_goal_diff_k5` | `float64` | No | 5 matches | Differential in short-term goal difference (`GF - GA`). |
| `diff_goal_diff_k10` | `float64` | No | 10 matches | Differential in sustained goal difference. |
| `diff_clean_sheets_k10` | `float64` | No | 10 matches | Differential in sustained clean sheet rate. |
| `diff_ppg_k5` | `float64` | No | 5 matches | Differential in rolling points per game (Form). |
| `diff_venue_ppg_k5` | `float64` | No | 5 venue games | Home team home PPG minus Away team road PPG. |
| `diff_rest_days` | `float64` | No | Continuous | Inter-match rest days differential (Fatigue). |

*(Note: Formally locked as Option A per ADR-007 approval on 12 October 2026).*

---

## 2. Invariant Type Constraints

1. **Non-Null Invariant:** All 16 feature columns and target columns must have 0 null entries across all modelling-ready rows ($N = 1,741$) in `dataset_features.csv`. Warm-up excluded rows ($N = 159$) are explicitly flagged via `is_modelling_ready = False`.
2. **Standardization Invariant:** Mean and standard deviation parameters used to standardize features must be estimated strictly on the training partition (`2019/20`–`2021/22`) and frozen before applying to validation and test partitions.
3. **Target Invariant:** `target_multinomial` must sum to exactly $N = 1,900$ when one-hot encoded ($\sum_{c=0}^2 y_{ic} = 1$).
