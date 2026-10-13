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

### 2.1 Secondary Validation Source: Football-Data.co.uk

- **Provider:** Football-Data.co.uk (Joseph Buchdahl)
- **Base Portal:** `https://www.football-data.co.uk/englandm.php`
- **Acquisition Script:** [`src/data_collection.py`](file:///d:/MLE-Lasso/src/data_collection.py)
- **HTTP Method:** Direct HTTPS GET via `requests==2.34.2`
- **Collection Timestamp:** 2026-10-12
- **License / Usage:** Public historical match results and betting odds dataset for academic/research modeling.

| Season | Download URL | Local Target Path | File Size | Row Count | Col Count | Missing Core Fields |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **2019-2020** | `https://www.football-data.co.uk/mmz4281/1920/E0.csv` | `data/raw/football_data/premier_league_2019-2020.csv` | 171.36 KB | 380 | 106 | 0 |
| **2020-2021** | `https://www.football-data.co.uk/mmz4281/2021/E0.csv` | `data/raw/football_data/premier_league_2020-2021.csv` | 171.69 KB | 380 | 106 | 0 |
| **2021-2022** | `https://www.football-data.co.uk/mmz4281/2122/E0.csv` | `data/raw/football_data/premier_league_2021-2022.csv` | 171.25 KB | 380 | 106 | 0 |
| **2022-2023** | `https://www.football-data.co.uk/mmz4281/2223/E0.csv` | `data/raw/football_data/premier_league_2022-2023.csv` | 172.05 KB | 380 | 106 | 0 |
| **2023-2024** | `https://www.football-data.co.uk/mmz4281/2324/E0.csv` | `data/raw/football_data/premier_league_2023-2024.csv` | 168.16 KB | 380 | 106 | 0 |

**Core Variables Extracted & Validated:**
- `Div`: Division identifier (`E0` for EPL)
- `Date`: Match date (`DD/MM/YYYY`)
- `Time`: Kickoff time
- `HomeTeam`, `AwayTeam`: Participating clubs
- `FTHG`, `FTAG`: Full-time goals (Home / Away)
- `FTR`: Full-time result (`H`, `D`, `A`)
- `HTHG`, `HTAG`, `HTR`: Half-time scoreline and result
- `HS`, `AS`: Total shots taken
- `HST`, `AST`: Shots on target
- `HF`, `AF`: Fouls committed
- `HC`, `AC`: Corner kicks won
- `HY`, `AY`: Yellow cards received
- `HR`, `AR`: Red cards received
- Full market odds: `B365H`, `B365D`, `B365A`, `AvgH`, `AvgD`, `AvgA`, `MaxH`, `MaxD`, `MaxA`

---

### 2.2 Primary Research Source: FBref

- **Provider:** Sports Reference LLC / FBref (Advanced stats powered by Opta)
- **Base Portal:** `https://fbref.com/en/comps/9/Premier-League-Stats`
- **Acquisition Tool:** `soccerdata` (version 1.9.1) backed by SeleniumBase UC browser driver and local cache
- **Collection Timestamp:** 2026-10-12
- **Rate-Limiting Compliance:** 7-second inter-request delay conforming strictly to FBref's 10 requests/min terms of service
- **Local Cache Location:** `C:\Users\shubh\soccerdata\data\FBref`

| Season | FBref Source Endpoint | Local Target Path | File Size | Row Count | Col Count | Missing Core Fields |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **2019-2020** | `/en/comps/9/2019-2020/schedule/` | `data/raw/fbref/2019-2020/schedule.csv` | 90.31 KB | 380 | 16 | 0* |
| **2020-2021** | `/en/comps/9/2020-2021/schedule/` | `data/raw/fbref/2020-2021/schedule.csv` | 88.92 KB | 380 | 16 | 0* |
| **2021-2022** | `/en/comps/9/2021-2022/schedule/` | `data/raw/fbref/2021-2022/schedule.csv` | 90.30 KB | 380 | 16 | 0 |
| **2022-2023** | `/en/comps/9/2022-2023/schedule/` | `data/raw/fbref/2022-2023/schedule.csv` | 90.61 KB | 380 | 16 | 0 |
| **2023-2024** | `/en/comps/9/2023-2024/schedule/` | `data/raw/fbref/2023-2024/schedule.csv` | 90.23 KB | 380 | 16 | 0 |
| **Combined** | All 5 seasons concatenated | `data/raw/fbref/fbref_premier_league_2019_2024_schedule.csv` | 449.91 KB | 1,900 | 16 | 0* |

*\*Note on Missing Fields:* The `attendance` column contains 92 missing entries in 2019/20 and 348 missing entries in 2020/21 due to COVID-19 empty-stadium protocols. All fixture metadata (dates, kickoff times, teams, scores, referee, game_id) have zero missing values across all 1,900 fixtures.

---

## 3. Transformations & Derived Data Tracking

| Stage | Script | Input Files | Output File | Transformation Description |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | [`src/data_collection.py`](file:///d:/MLE-Lasso/src/data_collection.py) | Remote HTTP / Cache | `data/raw/football_data/*.csv`, `data/raw/fbref/*/schedule.csv` | Pure extraction, format to CSV, zero alterations. |
| **Audit** | [`src/data_audit.py`](file:///d:/MLE-Lasso/src/data_audit.py) | All raw CSVs | `data/audit/*.csv`, `data/audit/data_audit_report.md` | Verification of counts, duplicate detection, cross-source fixture reconciliation. |
| **Features (Gate 3)** | [`src/features.py`](file:///d:/MLE-Lasso/src/features.py) | `data/raw/football_data/*.csv`, `data/raw/fbref/.../schedule.csv` | `data/features/dataset_features.csv`, `data/audit/feature_*.csv` | Chronological rolling $k$-match differential calculation strictly excluding match $t$ ($j \ge 1$). Exactly 1,900 rows ($N=1,741$ modelling ready, $N=159$ warm-up). |
| **Preprocessing (Gate 4)** | [`src/data_split.py`](file:///d:/MLE-Lasso/src/data_split.py) | `data/features/dataset_features.csv` | `data/processed/train_data.csv`, `val_data.csv`, `test_data.csv`, `scalers.joblib`, `reports/vif_summary.csv` | Chronological train/val/test split (1002/370/369), frozen StandardScaler fitted strictly on train partition. Multicollinearity (VIF and Condition Number) diagnostics. |
| **Model Calibration (Gate 5)** | [`src/models/train_and_tune.py`](file:///d:/MLE-Lasso/src/models/train_and_tune.py) | `data/processed/train_data.csv`, `val_data.csv` | `data/processed/fitted_models.joblib`, `reports/hyperparameter_tuning.csv`, `reports/figures/fig6_hyperparameter_tuning.png` | 20-point logarithmic grid search over $\lambda \in [10^{-4}, 10^4]$. Model fitting for unregularized MLE, Ridge ($L_2$), and Lasso ($L_1$). Optimal $\lambda^*$ chosen by minimum validation log-loss. |
| **Evaluation & Hypothesis Testing (Gate 6)** | [`src/evaluation.py`](file:///d:/MLE-Lasso/src/evaluation.py) | `data/processed/test_data.csv`, `data/processed/fitted_models.joblib` | `reports/test_metrics.csv`, `reports/wilcoxon_results.csv`, `reports/mcnemar_results.csv`, `reports/bootstrap_coefficients.csv`, `reports/figures/calibration_curves.png`, `reports/figures/bootstrap_stability.png`, `reports/final_evaluation_report.md` | Out-of-sample evaluation on Season 2023/24 test set ($N=369$). Paired Wilcoxon signed-rank test across 38 gameweeks, McNemar classification test, matchweek block-bootstrap ($B=1,000$), calibration curves, and feature stability. |
