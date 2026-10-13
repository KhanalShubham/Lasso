# OpenFootball Migration Report (reports/openfootball_migration_report.md)

**Project Title:** A Comparative Study of Maximum Likelihood, Ridge, and Lasso Logistic Regression for Football Match Outcome Prediction  
**Date of Audit & Migration:** 13 October 2026  
**Governing Architecture Decision Record:** [ADR-013 (DECISIONS.md)](file:///d:/MLE-Lasso/DECISIONS.md)  
**Primary Source Repository:** [openfootball/england](https://github.com/openfootball/england)  
**License Status:** CC0-1.0 Universal Public Domain Dedication (Verified)  

---

## 1. Executive Summary

This report documents the end-to-end migration of the `MLE-Lasso` research project from its previous dual-source architecture (FBref schedules and Football-Data.co.uk match event statistics) to the public-domain **OpenFootball** data repository (`https://github.com/openfootball/england`). 

The migration replaces proprietary and scraper-dependent sources with an open-access, public-domain dataset dedicated under **Creative Commons CC0 1.0 Universal**, establishing complete academic reproducibility and eliminating third-party licensing risks. All 1,900 historical English Premier League fixtures across the five target seasons (2019/20 to 2023/24) have been ingested, parsed, canonicalized, feature-engineered, and validated through automated unit and leakage test suites.

---

## 2. Rationale & Licensing Verification

### 2.1 Limitations of Previous Data Sources
- **FBref (Sports Reference LLC / Opta):** Data collection relied on browser automation (`soccerdata`) subject to stringent rate-limiting (10 requests/min, 7-second inter-request delay) and terms of service that restrict redistribution and commercial/unrestricted open access. Furthermore, match-level passing coordinates, defensive actions, and xG in `all_comps` logs were unavailable or stripped.
- **Football-Data.co.uk:** Compiles proprietary match sheets and commercial betting odds. While widely used in academic sports analytics, it lacks a formal Creative Commons open data license.

### 2.2 OpenFootball License Audit
The OpenFootball repository provides complete public domain dedication:
- **Repository URL:** `https://github.com/openfootball/england`
- **License File:** `https://raw.githubusercontent.com/openfootball/england/master/LICENSE.md`
- **License Declaration:** **CC0 1.0 Universal (CC0 1.0) Public Domain Dedication**
- **Verification Date:** 13 October 2026
- **Cryptographic Hash (SHA-256):** `36ffd9dc085d529a7e60e1276d73ae5a030b020313e6c5408593a6ae2af39673`
- **Attribution & Rights Statement:** To the extent possible under law, Gerald Bauer and the openfootball project contributors have dedicated all copyright and related neighboring rights to the public domain worldwide.

---

## 3. Data Ingestion & Seasonal Integrity Audit

Raw source text files were acquired reproducibly via direct HTTPS GET without web scraping or rate-limit violations, and stored immutably in `data/raw/openfootball/`.

### 3.1 Raw Ingestion Manifest

| Season | Relative Path | File Size | SHA-256 Hash | Ingested Fixtures |
| :--- | :--- | :---: | :--- | :---: |
| `2019-20` | `data/raw/openfootball/2019-20/1-premierleague.txt` | 25,416 B | `fdc8300c09e0700faa0c517ff408bc630b8b6a77b6f1aa68e5858f9adae6f72d` | 380 |
| `2020-21` | `data/raw/openfootball/2020-21/1-premierleague.txt` | 27,068 B | `2123fa19c676d0194c69380dceb61f9413bfe6b89eb6c538cd269d8d856ba255` | 380 |
| `2021-22` | `data/raw/openfootball/2021-22/1-premierleague.txt` | 26,953 B | `592df232fbc7c2f36ec3e643eef608f77fcbf7d8da0706c94c99248d60b932b3` | 380 |
| `2022-23` | `data/raw/openfootball/2022-23/1-premierleague.txt` | 26,495 B | `c85a2f8ac64975435b8359eb923eaa09e48f33d47672c8b32d2c6e99a8cec5f0` | 380 |
| `2023-24` | `data/raw/openfootball/2023-24/1-premierleague.txt` | 26,357 B | `0bfb4a7fffe0b1bf7318955686eafeb28fc9676346ae433259e7409d69551450` | 380 |
| **Total** | **5 Contiguous Seasons** | **132,289 B** | **Verified Checksums** | **1,900** |

Complete cryptographic provenance is serialized in [`data/audit/openfootball_ingestion_manifest.json`](file:///d:/MLE-Lasso/data/audit/openfootball_ingestion_manifest.json).

---

## 4. Source Format & Canonical Processing Pipeline

### 4.1 Football.TXT Parsing
OpenFootball stores match fixtures in human-readable plain text blocks under the `Football.TXT` syntax:
- Headers denote matchdays: `▪ Matchday 1`, `Matchday 2`, etc.
- Date lines denote calendar days: `[Fri Aug 9]`, `[Sat Aug 10]`, etc.
- Match lines contain kickoff times, participating clubs, and full-time scores:
  `20.00  Liverpool FC  4-1 (4-0)  Norwich City FC`
  `15.00  Burnley FC    3-0 (0-0)  Southampton FC`

### 4.2 Calendar Year Resolution
Because Football.TXT date lines omit the 4-digit calendar year (e.g. `Fri Aug 9`), calendar years are deterministically resolved using season boundaries:
- Months from August through December ($m \ge 8$) are assigned to `start_year` (e.g., 2019 for `2019-20`).
- Months from January through July ($m \le 7$) are assigned to `end_year` (e.g., 2020 for `2019-20`).
- This correctly handles COVID-19 delayed fixtures played during Project Restart in June and July 2020.

### 4.3 Club Name Normalization
OpenFootball files contain 38 text variants for participating clubs (e.g. `Aston Villa` vs `Aston Villa FC`, `Wolverhampton Wanderers` vs `Wolverhampton Wanderers FC`). The pipeline applies a deterministic regex stripper (`re.sub(r'\s+FC$', '', name)`) that resolves all strings into exactly 26 canonical Premier League clubs without merging or losing any club entities.

### 4.4 Canonical Output Table
The processed table is stored at [`data/processed/epl_matches.csv`](file:///d:/MLE-Lasso/data/processed/epl_matches.csv):
- Total rows: exactly 1,900 fixtures (380 per season).
- Null count: 0 across all 11 columns.
- Ground truth target breakdown:
  - **Home Wins:** 838 (44.11%)
  - **Draws:** 432 (22.74%)
  - **Away Wins:** 630 (33.16%)
  - Exactly matches official English Premier League historical records.

---

## 5. Methodological & Feature Set Revision

### 5.1 Data Constraints of OpenFootball
OpenFootball provides match dates, kickoff times, clubs, and final scores, but does **not** record in-game event statistics (shots, shots on target, corner kicks, fouls, yellow/red cards, possession, or betting odds).

### 5.2 Research Continuity
- **Research Question:** Unchanged. Compares unregularized MLE, Ridge ($L_2$), and Lasso ($L_1$) multinomial logistic regression.
- **Target Variable:** Unchanged. Symmetrical 3-class outcome: Home Win (`2`), Draw (`1`), Away Win (`0`).
- **Modelling Framework:** Unchanged. Symmetric Softmax link with identical penalization across parameter matrices.
- **Validation Protocol:** Unchanged. Strict chronological partitioning:
  - **Train:** Seasons 2019/20, 2020/21, 2021/22 ($N = 1,002$ modelling-ready fixtures).
  - **Validation:** Season 2022/23 ($N = 370$ modelling-ready fixtures).
  - **Test:** Season 2023/24 ($N = 369$ modelling-ready fixtures).
  - **Warm-Up Excluded:** 159 fixtures across 5 seasons.

### 5.3 Revised Feature Set Specification
Per ADR-013, the candidate predictor set is revised from shot/card metrics to **result-derived historical performance metrics** computed strictly over prior matches ($j \ge 1$):
1. `diff_goals_scored_k5`: Differential in rolling 5-match goals scored.
2. `diff_goals_conceded_k5`: Differential in rolling 5-match goals conceded.
3. `diff_ppg_k5`: Differential in rolling 5-match points per match ($W=3, D=1, L=0$).
4. `diff_goal_diff_k10`: Differential in rolling 10-match sustained goal difference per match.
5. `diff_venue_ppg_k5`: Differential in 5-match venue form (Home team at home minus Away team away).
6. `diff_rest_days`: Differential in inter-match rest days ($\Delta \tau_{\text{Home}} - \Delta \tau_{\text{Away}}$).
7. `diff_goal_diff_k5`: Differential in rolling 5-match goal difference per match.
8. `diff_clean_sheets_k10`: Differential in rolling 10-match clean sheet rate.
9. `diff_win_rate_k5`: Differential in rolling 5-match win rate.

---

## 6. Legacy Data Policy

In strict accordance with project rules:
1. Files in `data/raw/fbref/` and `data/raw/football_data/` remain untouched and are marked as **LEGACY ARCHIVES**.
2. They are strictly excluded from the active research pipeline.
3. No fallback scraping or silent combining of FBref and Football-Data is permitted.

---

## 7. Migration Verification Summary

- [x] All 5 target seasons retrieved from OpenFootball and checksums verified.
- [x] License verified as CC0-1.0 Universal Public Domain Dedication.
- [x] Canonical match table created (`data/processed/epl_matches.csv`, 1,900 rows, 0 nulls).
- [x] Feature dataset generated (`data/features/dataset_features.csv`, 1,900 rows, 1,741 modelling-ready, 159 warm-up).
- [x] Automated test suite passed (15/15 unit and leakage tests pass).
- [x] ADR-013 logged in `DECISIONS.md`.
- [x] Data provenance documented in `DATA_PROVENANCE.md`.
- [x] Zero models trained; Gate 3 held for formal review.
