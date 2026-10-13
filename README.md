# A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction

**Document Type:** Project Documentation & Data Pipeline Specification  
**Domain:** Sports Analytics & Statistical Machine Learning  
**Target League:** English Premier League (EPL)  
**Seasons Covered:** 2019/20, 2020/21, 2021/22, 2022/23, and 2023/24 (5 Contiguous Seasons)  
**Sample Size:** Exactly 1,900 Completed Matches (380 per season)  

---

## 1. Research Overview

This repository houses the empirical data engineering, statistical modeling, and experimental evaluation for a study comparing classical unregularized Maximum Likelihood Estimation (MLE), $L_2$-regularized (Ridge), and $L_1$-regularized (Lasso) logistic regression in forecasting English Premier League match outcomes ($\mathcal{Y} \in \{\text{Home Win}, \text{Draw}, \text{Away Win}\}$).

The research focuses on the behavior of shrinkage penalties and variable sparsity in the presence of highly collinear pre-match tactical indicators derived from historical match data.

---

## 2. Directory Structure

```text
d:\MLE-Lasso\
├── data\
│   ├── raw\
│   │   ├── fbref\
│   │   │   ├── 2019-2020\schedule.csv
│   │   │   ├── 2020-2021\schedule.csv
│   │   │   ├── 2021-2022\schedule.csv
│   │   │   ├── 2022-2023\schedule.csv
│   │   │   ├── 2023-2024\schedule.csv
│   │   │   ├── 2023-2024\sample_shooting_matchlogs_Arsenal.csv
│   │   │   ├── 2023-2024\sample_keeper_matchlogs_Arsenal.csv
│   │   │   └── fbref_premier_league_2019_2024_schedule.csv
│   │   └── football_data\
│   │       ├── premier_league_2019-2020.csv
│   │       ├── premier_league_2020-2021.csv
│   │       ├── premier_league_2021-2022.csv
│   │       ├── premier_league_2022-2023.csv
│   │       └── premier_league_2023-2024.csv
│   ├── processed\               # Destination for cleaned, chronologically sorted data
│   └── audit\
│       ├── data_audit_summary.csv
│       ├── seasonal_integrity_audit.csv
│       ├── fixture_matching_audit.csv
│       ├── feature_availability_audit.csv
│       └── data_audit_report.md
├── src\
│   ├── __init__.py
│   ├── data_collection.py       # Ingests raw schedules & validation CSVs
│   └── data_audit.py            # Automated audit, cross-matching & predictor check
├── RESEARCH_METHODOLOGY_GUIDE.md # Theoretical formulation & experimental plan
├── README.md                    # Project blueprint & reproduction instructions
└── requirements.txt             # Pinned library dependencies
```

---

## 3. Data Sources & Metadata

### Primary Research Source: FBref
- **Provider:** Sports Reference LLC / FBref (Stats powered by Opta)
- **Target URL:** `https://fbref.com/en/comps/9/Premier-League-Stats`
- **Schedule URL Pattern:** `https://fbref.com/en/comps/9/{Season}/schedule/{Season}-Premier-League-Scores-and-Fixtures`
- **Acquisition Tool:** `soccerdata` (version 1.9.1) with local caching to comply with FBref's 10 requests/min rate limit.
- **Coverage Collected:** All 1,900 completed match fixtures with date, kickoff time, round/matchweek, venue, referee, home/away clubs, full-time score, and Opta `game_id`.

### Secondary Validation Source: Football-Data.co.uk
- **Provider:** Football-Data.co.uk (Joseph Buchdahl)
- **Target Portal:** `https://www.football-data.co.uk/englandm.php`
- **File URLs:**
  - `2019/20`: `https://www.football-data.co.uk/mmz4281/1920/E0.csv`
  - `2020/21`: `https://www.football-data.co.uk/mmz4281/2021/E0.csv`
  - `2021/22`: `https://www.football-data.co.uk/mmz4281/2122/E0.csv`
  - `2022/23`: `https://www.football-data.co.uk/mmz4281/2223/E0.csv`
  - `2023/24`: `https://www.football-data.co.uk/mmz4281/2324/E0.csv`
- **Coverage Collected:** All 1,900 completed matches across 106 attributes, including half-time and full-time scores, shots (`HS`, `AS`), shots on target (`HST`, `AST`), fouls, corners, cards, and betting odds.

---

## 4. Environment Setup & Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.14.4)
- Google Chrome browser (for initial SeleniumBase driver handshake with FBref)

### Installation
Clone or navigate to the repository directory and install the required dependencies:
```bash
python -m pip install -r requirements.txt
```

### Execution Commands

1. **Run the Data Collection Pipeline:**
```bash
python src/data_collection.py
```
*Outputs:* Downloads raw Football-Data CSV files into `data/raw/football_data/`, extracts FBref seasonal schedules into `data/raw/fbref/{season}/schedule.csv`, and creates the consolidated `data/raw/fbref/fbref_premier_league_2019_2024_schedule.csv`.

2. **Run the Automated Data Audit:**
```bash
python src/data_audit.py
```
*Outputs:* Performs row count verification, missing value scans, duplicate fixture checks, cross-source fixture reconciliation, and predictor availability auditing. Generates CSV summaries and `data/audit/data_audit_report.md`.

---

## 5. Audit Results Summary

| Diagnostic Dimension | Target / Baseline | Audited Result | Verification Status |
| :--- | :---: | :---: | :---: |
| **Total Match Fixtures** | 1,900 observations | **1,900 observations** | **PASS** (100.0%) |
| **Per-Season Fixture Count** | 380 matches / season | **380 matches / season** | **PASS** (100.0%) |
| **Duplicate Fixtures** | 0 duplicates | **0 duplicates** | **PASS** |
| **Cross-Source Matched Fixtures** | 1,900 matches | **1,900 matches** | **PASS** (100.0%) |
| **Full-Time Score Concordance** | 100% agreement | **1,900 / 1,900 (100.0%)** | **PASS** |
| **Missing Core Values** | 0 in score/teams/date | **0 missing** | **PASS** |

---

## 6. Known Technical Limitations & Next Steps

1. **Soccerdata API Scope:**
   `soccerdata.FBref.read_team_match_stats` only implements `['schedule', 'shooting', 'keeper', 'misc']`. Passing logs (`PrgP`, `1/3`), possession logs (`PrgC`, `Att Pen`), and advanced goalkeeping (`PSxG+/-`) are not exposed by the package's match-level API.
2. **FBref All-Competitions View:**
   Team match logs fetched for `all_comps` omit `xG` and `npxG` due to cup matches lacking Opta tracking data.
3. **Feature Engineering Alignment:**
   A 16-predictor candidate set utilizing 100% complete, verified match statistics (shots, shots on target, shot dominance, corners, disciplinary actions, goal differentials, points-per-game, venue-specific form, and rest days) is fully feasible and avoids data leakage. See [data_audit_report.md](file:///d:/MLE-Lasso/data/audit/data_audit_report.md) for full specifications.
