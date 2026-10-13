# Data Provenance and Lineage Record (DATA_PROVENANCE.md)

**Project:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Last Updated:** October 2026  
**Status:** Verified  
**Authority:** [PROJECT_CONSTITUTION.md](file:///d:/MLE-Lasso/PROJECT_CONSTITUTION.md)  

---

## 1. Principles of Data Provenance

1. **Raw Immutability:** Files in `data/raw/` are immutable primary records. They are never edited in place, cleaned, imputed, or altered.
2. **Deterministic Reproducibility:** Every transformed dataset in `data/processed/` must be generated entirely by running version-controlled scripts in `src/`.
3. **Audited Verification:** Every dataset must record its origin URL, retrieval timestamp, acquisition method, schema definition, and row/column count.

---

## 2. Ingested Datasets Inventory

### 2.1 Primary Active Research Source: OpenFootball England (Public Domain)

- **Provider:** OpenFootball / Gerald Bauer & Contributors
- **Repository URL:** `https://github.com/openfootball/england`
- **Acquisition Script:** [`src/data_collection_openfootball.py`](file:///d:/MLE-Lasso/src/data_collection_openfootball.py)
- **HTTP Method:** Direct HTTPS GET via `requests==2.34.2` from GitHub raw repository
- **Collection Timestamp:** 2026-10-13T10:50:54Z
- **License / Dedication:** **CC0-1.0 Universal Public Domain Dedication** (`LICENSE.md`; SHA-256: `36ffd9dc085d529a7e60e1276d73ae5a030b020313e6c5408593a6ae2af39673`)
- **Ingestion Manifest:** [`data/audit/openfootball_ingestion_manifest.json`](file:///d:/MLE-Lasso/data/audit/openfootball_ingestion_manifest.json)
- **Status:** **ACTIVE PRIMARY AND SOLE SOURCE OF TRUTH FOR MATCH FIXTURES & OUTCOMES**

| Season | Download URL | Local Target Path | File Size | Row Count | Format | Missing Fields |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **2019-20** | `https://raw.githubusercontent.com/openfootball/england/master/2019-20/1-premierleague.txt` | `data/raw/openfootball/2019-20/1-premierleague.txt` | 25.42 KB | 380 | Football.TXT | 0 |
| **2020-21** | `https://raw.githubusercontent.com/openfootball/england/master/2020-21/1-premierleague.txt` | `data/raw/openfootball/2020-21/1-premierleague.txt` | 27.07 KB | 380 | Football.TXT | 0 |
| **2021-22** | `https://raw.githubusercontent.com/openfootball/england/master/2021-22/1-premierleague.txt` | `data/raw/openfootball/2021-22/1-premierleague.txt` | 26.95 KB | 380 | Football.TXT | 0 |
| **2022-23** | `https://raw.githubusercontent.com/openfootball/england/master/2022-23/1-premierleague.txt` | `data/raw/openfootball/2022-23/1-premierleague.txt` | 26.50 KB | 380 | Football.TXT | 0 |
| **2023-24** | `https://raw.githubusercontent.com/openfootball/england/master/2023-24/1-premierleague.txt` | `data/raw/openfootball/2023-24/1-premierleague.txt` | 26.36 KB | 380 | Football.TXT | 0 |

**Core Variables Extracted & Canonicalized:**
- `match_id`: Deterministic fixture key (`{season}_GW{matchday:02d}_{home}_v_{away}`)
- `season`: Season string (`YYYY-YY`)
- `date`: Official match date (`YYYY-MM-DD`)
- `time`: Kickoff time (`HH:MM`)
- `matchday`: Nominal Gameweek round (1 to 38)
- `home_team`, `away_team`: Canonical normalized club names (26 distinct clubs)
- `home_goals`, `away_goals`: Integer final scoreline
- `target`: Categorical outcome label (`HOME_WIN`, `DRAW`, `AWAY_WIN`)
- `target_multinomial`: Numeric label (`2`: Home Win, `1`: Draw, `0`: Away Win)

---

### 2.2 Legacy Archive: Football-Data.co.uk (EXCLUDED FROM ACTIVE PIPELINE)

- **Provider:** Football-Data.co.uk (Joseph Buchdahl)
- **Status:** **LEGACY ARCHIVE ONLY (DEPRECATED)**
- **Preservation Reason:** Preserved in `data/raw/football_data/` for traceability and prior audit reconciliation; strictly excluded from the active research modeling pipeline per ADR-013.

---

### 2.3 Legacy Archive: FBref (EXCLUDED FROM ACTIVE PIPELINE)

- **Provider:** Sports Reference LLC / FBref (Opta Sports)
- **Status:** **LEGACY ARCHIVE ONLY (DEPRECATED)**
- **Preservation Reason:** Preserved in `data/raw/fbref/` for traceability of earlier experimental drafts; strictly excluded from the active research modeling pipeline per ADR-013.

---

## 3. Transformations & Derived Data Tracking

| Stage | Script | Input Files | Output File | Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | [`src/data_collection_openfootball.py`](file:///d:/MLE-Lasso/src/data_collection_openfootball.py) | OpenFootball GitHub Raw API | `data/raw/openfootball/{season}/1-premierleague.txt`, `LICENSE.md` | Pure download of CC0-1.0 source files, SHA-256 manifest recording. |
| **Processing** | [`src/data_processing_openfootball.py`](file:///d:/MLE-Lasso/src/data_processing_openfootball.py) | `data/raw/openfootball/` | `data/processed/epl_matches.csv` | Football.TXT parsing, date resolution, team name normalization, final scores, 3-class target creation, chronological sorting. |
| **Features (Gate 3)** | [`src/features_openfootball.py`](file:///d:/MLE-Lasso/src/features_openfootball.py) | `data/processed/epl_matches.csv` | `data/features/dataset_features.csv`, `data/features/feature_dictionary.md` | Chronological rolling $k$-match differential calculation strictly excluding match $t$ ($j \ge 1$). Result-derived features (goals scored/conceded, PPG, GD, venue PPG, rest days). Exactly 1,900 rows ($N=1,741$ modelling ready, $N=159$ warm-up). |
| **Quality Audit** | Automated Pipeline Tests | `data/processed/epl_matches.csv`, `data/features/` | `reports/data_quality_report.md`, `reports/openfootball_migration_report.md`, `reports/feature_engineering_report.md` | Automated verification of row counts, chronological monotonic sorting, score validity, zero lookahead leakage, and warm-up isolation. |

