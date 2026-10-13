# Dataset Overview (dataoverview.md)

**Project:** MLE vs Ridge vs Lasso Logistic Regression for Football Match Outcome Prediction  
**Status:** Ingested & Audited  
**Authority:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md) | [DATA_PROVENANCE.md](file:///d:/MLE-Lasso/DATA_PROVENANCE.md)  

---

## 1. Scope & Observation Horizon

The dataset represents completed English Premier League matches across five consecutive seasons:
- **Total Completed Fixtures:** Exactly $N = 1,900$ observations (380 matches per season).
- **Temporal Span:** August 2019 to May 2024.
- **Participating Clubs:** 26 unique clubs over the five seasons.
- **Outcome Balance (Full Sample, N = 1,900):**
  - Home Wins: 838 matches (44.11%)
  - Draws: 432 matches (22.74%)
  - Away Wins: 630 matches (33.16%)
- **Modelling Partition Balance (N = 1,741 Modelling-Ready Rows):**
  - Home Wins: 774 matches (44.46%)
  - Draws: 389 matches (22.34%)
  - Away Wins: 578 matches (33.20%)
  - Warm-up excluded rows: 159 matches (8.37%)
- **Feature Dataset:** [`data/features/dataset_features.csv`](file:///d:/MLE-Lasso/data/features/dataset_features.csv) containing 16 pre-match differential predictors.

---

## 2. Ingested Datasets

### A. Primary Research Source: OpenFootball England (`data/raw/openfootball/`)
- **Origin:** OpenFootball project (`https://github.com/openfootball/england`).
- **License:** CC0-1.0 Universal Public Domain Dedication.
- **Format:** Plaintext `Football.TXT` structured fixtures per season (`1-premierleague.txt`).
- **Verified Row Count:** Exactly 380 rows per season; 1,900 rows combined.
- **Key Attributes:** Match date (`YYYY-MM-DD`), kickoff time (`HH:MM`), nominal matchday round (1–38), home club, away club, and full-time scoreline.
- **Processed Canonical Output:** `data/processed/epl_matches.csv` (1,900 rows, 11 columns, 0 missing).

### B. Legacy Archives: Football-Data.co.uk & FBref (`data/raw/football_data/`, `data/raw/fbref/`)
- **Status:** Deprecated legacy archives, preserved strictly for historical provenance and audit traceability. Excluded from active modeling per ADR-013.

---

## 3. Seasonal Breakdown & Characteristics

| Season | Matches | Earliest Match | Latest Match | Key Domain Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **2019/20** | 380 | 2019-08-09 | 2020-07-26 | Season suspended in March 2020; resumed behind closed doors (Project Restart). |
| **2020/21** | 380 | 2020-09-12 | 2021-05-23 | Played almost entirely behind closed doors due to COVID-19; marked drop in home advantage. |
| **2021/22** | 380 | 2021-08-13 | 2022-05-22 | Full stadium capacities returned; baseline home advantage restored. |
| **2022/23** | 380 | 2022-08-05 | 2023-05-28 | Mid-season winter FIFA World Cup break (Nov–Dec 2022); fixture congestion in spring. |
| **2023/24** | 380 | 2023-08-11 | 2024-05-19 | Out-of-sample final test season; highest goals-per-game in modern EPL history. |
