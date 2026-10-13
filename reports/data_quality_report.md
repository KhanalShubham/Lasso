# Data Quality & Integrity Audit Report (reports/data_quality_report.md)

**Project Title:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Data Provider:** OpenFootball England Repository (CC0-1.0 Public Domain Dedication)  
**Date of Audit:** 13 October 2026  
**Audited Artifacts:**
- Raw source files: `data/raw/openfootball/{season}/1-premierleague.txt` (5 seasons)
- Canonical match table: [`data/processed/epl_matches.csv`](file:///d:/MLE-Lasso/data/processed/epl_matches.csv)
- Feature matrix: [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv)

---

## 1. Raw and Processed Row Counts by Season

Every season in the 5-year study period (2019/20 through 2023/24) was independently parsed from the raw OpenFootball Football.TXT source files. The row counts across all stages are completely reconciled:

| Season | Raw Text Fixtures Parsed | Canonical Match Rows | Warm-Up Excluded Rows | Modelling-Ready Rows | Expected Season Total | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2019/20** | 380 | 380 | 100 | 280 | 380 | **100.0% COMPLETE** |
| **2020/21** | 380 | 380 | 28 | 352 | 380 | **100.0% COMPLETE** |
| **2021/22** | 380 | 380 | 10 | 370 | 380 | **100.0% COMPLETE** |
| **2022/23** | 380 | 380 | 10 | 370 | 380 | **100.0% COMPLETE** |
| **2023/24** | 380 | 380 | 11 | 369 | 380 | **100.0% COMPLETE** |
| **Total** | **1,900** | **1,900** | **159** | **1,741** | **1,900** | **100.0% COMPLETE** |

---

## 2. Excluded Fixtures and Justifications

No fixtures are dropped or discarded from the canonical match table (`epl_matches.csv`). However, in accordance with the project's **Warm-Up Cold-Start Isolation Protocol (ADR-002, ADR-007, ADR-013)**, fixtures with insufficient empirical history are excluded from model training and evaluation partitions:

- **Total Excluded Warm-Up Fixtures:** Exactly **159 fixtures** (8.37% of the gross 1,900 matches).
- **Reason for Exclusion:** Either the home club or the away club had completed fewer than 10 top-flight matches overall ($N < 10$) or fewer than 5 venue-specific matches ($N_{\text{venue}} < 5$) in the dataset.
- **Season 2019/20 ($N=100$ warm-up):** As the initial season in the dataset, all 20 clubs start with zero top-flight history in this repository. Warm-up is completed by Gameweek 10.
- **Subsequent Seasons ($N=59$ warm-up across 4 seasons):** Newly promoted clubs (3 clubs per season: e.g. Leeds, West Brom, Fulham in 2020/21; Brentford, Norwich, Watford in 2021/22; Bournemouth, Fulham, Nottingham Forest in 2022/23; Burnley, Sheffield United, Luton Town in 2023/24) require warm-up fixtures until $N \ge 10$ overall and $N \ge 5$ venue games are observed.
- **Methodological Protection:** These 159 rows have their predictor columns set to `NaN`. They are preserved in `dataset_features.csv` for full historical auditability, but flagged `is_modelling_ready = False`.

---

## 3. Duplicate Records Audit

- **Audit Query:** Evaluated whether any pair of `(season, home_team, away_team)` occurs more than once in `data/processed/epl_matches.csv`.
- **Result:** **0 duplicate records detected**. In each of the five seasons, each pair of clubs plays exactly once at each venue (380 distinct pairings per season).
- **Matchday Uniqueness:** Every fixture possesses a unique deterministic key (`{season}_GW{matchday:02d}_{home}_v_{away}`).

---

## 4. Missing Values Audit

| Table / Partition | Total Rows | Target Missing | Metadata Missing | Predictor Missing | Missingness Policy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `data/processed/epl_matches.csv` | 1,900 | 0 (0.0%) | 0 (0.0%) | N/A | Non-null canonical table |
| `data/features/dataset_features.csv` (All) | 1,900 | 0 (0.0%) | 0 (0.0%) | 4,293 cells (in warm-up only) | Strict NaN preservation |
| `dataset_features.csv` (`is_modelling_ready == True`) | 1,741 | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | Zero nulls across all 27 predictors |
| `dataset_features.csv` (`is_modelling_ready == False`) | 159 | 0 (0.0%) | 0 (0.0%) | 4,293 cells (100.0% NaN) | Zero-imputation prohibited |

---

## 5. Team-Name Normalisation Audit

OpenFootball source files contain 38 distinct textual representations of clubs across the five seasons due to occasional `" FC"` suffixes. A deterministic regular expression normalizer (`re.sub(r'\s+FC$', '', name)`) successfully maps all variants to exactly **26 canonical Premier League clubs**:

| Canonical Club Name | Raw Variations Mapped | Total Appearances (Matches) |
| :--- | :--- | :---: |
| Arsenal | `Arsenal`, `Arsenal FC` | 190 |
| Aston Villa | `Aston Villa`, `Aston Villa FC` | 190 |
| AFC Bournemouth | `AFC Bournemouth` | 114 |
| Brentford | `Brentford`, `Brentford FC` | 114 |
| Brighton & Hove Albion | `Brighton & Hove Albion`, `Brighton & Hove Albion FC` | 190 |
| Burnley | `Burnley`, `Burnley FC` | 152 |
| Chelsea | `Chelsea`, `Chelsea FC` | 190 |
| Crystal Palace | `Crystal Palace`, `Crystal Palace FC` | 190 |
| Everton | `Everton`, `Everton FC` | 190 |
| Fulham | `Fulham`, `Fulham FC` | 114 |
| Leeds United | `Leeds United`, `Leeds United FC` | 114 |
| Leicester City | `Leicester City`, `Leicester City FC` | 152 |
| Liverpool | `Liverpool`, `Liverpool FC` | 190 |
| Luton Town | `Luton Town`, `Luton Town FC` | 38 |
| Manchester City | `Manchester City`, `Manchester City FC` | 190 |
| Manchester United | `Manchester United`, `Manchester United FC` | 190 |
| Newcastle United | `Newcastle United`, `Newcastle United FC` | 190 |
| Norwich City | `Norwich City`, `Norwich City FC` | 76 |
| Nottingham Forest | `Nottingham Forest`, `Nottingham Forest FC` | 76 |
| Sheffield United | `Sheffield United`, `Sheffield United FC` | 114 |
| Southampton | `Southampton`, `Southampton FC` | 152 |
| Tottenham Hotspur | `Tottenham Hotspur`, `Tottenham Hotspur FC` | 190 |
| Watford | `Watford`, `Watford FC` | 76 |
| West Bromwich Albion | `West Bromwich Albion`, `West Bromwich Albion FC` | 38 |
| West Ham United | `West Ham United`, `West Ham United FC` | 190 |
| Wolverhampton Wanderers | `Wolverhampton Wanderers`, `Wolverhampton Wanderers FC` | 190 |

**Result:** Zero orphaned clubs, zero merged clubs, and zero lost fixtures.

---

## 6. Target-Class Distribution by Season

The ground truth targets derived from full-time scorelines (`home_goals` vs `away_goals`) demonstrate full historical accuracy against official Premier League records:

| Season | Home Wins ($y=2$) | Draws ($y=1$) | Away Wins ($y=0$) | Total Fixtures | Home Win Rate | Draw Rate | Away Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2019/20** | 172 | 92 | 116 | 380 | 45.26% | 24.21% | 30.53% |
| **2020/21** | 144 | 83 | 153 | 380 | 37.89% | 21.84% | 40.26% |
| **2021/22** | 163 | 88 | 129 | 380 | 42.89% | 23.16% | 33.95% |
| **2022/23** | 184 | 87 | 109 | 380 | 48.42% | 22.89% | 28.68% |
| **2023/24** | 175 | 82 | 123 | 380 | 46.05% | 21.58% | 32.37% |
| **Overall** | **838** | **432** | **630** | **1,900** | **44.11%** | **22.74%** | **33.16%** |

*Historical Invariant Check:* The notable inversion in Season 2020/21 (153 Away Wins vs 144 Home Wins) accurately captures the documented empirical anomaly of the COVID-19 empty-stadium season behind closed doors.

---

## 7. Feature Availability Audit

Under the OpenFootball source architecture:
- In-game micro-event metrics (shots, shots on target, corner kicks, fouls, yellow cards, red cards, possession share, xG) are **unavailable** in the public-domain raw data.
- All candidate predictors derived from historical results, scorelines, and match timing are **100% available** with zero missingness for all 1,741 modelling-ready observations.

---

## 8. Unresolved Data Issues & Limitations

1. **Absence of In-Game Shot/Card Metrics:** Feature set is confined to result-derived historical performance metrics.
2. **Absence of Betting Odds in OpenFootball:** Bookmaker market odds (Bet365, Market Average) are not provided in the public-domain OpenFootball repository.
3. **Summer Inter-Season Rest Days:** Rest days between the final match of season $S$ and the opening match of season $S+1$ reflects the ~75–90 calendar day summer interval. Because both competing clubs experience comparable summer breaks, the differential `diff_rest_days` remains balanced.
