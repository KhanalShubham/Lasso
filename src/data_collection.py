"""Data Collection Script for EPL Match Outcome Prediction Study.

Research Study: A Comparative Study of Maximum Likelihood, Ridge,
and Lasso Logistic Regression for Football Match Outcome Prediction.

Target: English Premier League (EPL)
Seasons: 2019/20, 2020/21, 2021/22, 2022/23, 2023/24 (1,900 matches)
Sources:
  1. FBref (Primary Research Source) via soccerdata cached files & parser
  2. Football-Data.co.uk (Secondary Validation Source)
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Base Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FBREF_DIR = RAW_DIR / "fbref"
FOOTBALL_DATA_DIR = RAW_DIR / "football_data"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIT_DIR = DATA_DIR / "audit"

SEASONS = [
    {"label": "2019-2020", "fbref_code": "1920", "fd_code": "1920"},
    {"label": "2020-2021", "fbref_code": "2021", "fd_code": "2021"},
    {"label": "2021-2022", "fbref_code": "2122", "fd_code": "2122"},
    {"label": "2022-2023", "fbref_code": "2223", "fd_code": "2223"},
    {"label": "2023-2024", "fbref_code": "2324", "fd_code": "2324"},
]

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"


def setup_directory_structure():
    """Create all required folders for raw, processed, and audited data."""
    logger.info("Initializing directory structure...")
    for s in SEASONS:
        (FBREF_DIR / s["label"]).mkdir(parents=True, exist_ok=True)
    FOOTBALL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure successfully verified.")


def download_football_data():
    """Download raw Premier League CSV files from Football-Data.co.uk."""
    logger.info("--- Collecting Secondary Validation Data (Football-Data.co.uk) ---")
    results = {}
    for season in SEASONS:
        label = season["label"]
        code = season["fd_code"]
        url = f"{FOOTBALL_DATA_BASE_URL}/{code}/E0.csv"
        target_path = FOOTBALL_DATA_DIR / f"premier_league_{label}.csv"

        logger.info("Fetching %s from %s...", label, url)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(resp.content)
            df = pd.read_csv(target_path)
            logger.info("Saved %s: %d rows, %d columns -> %s", label, len(df), len(df.columns), target_path)
            results[label] = {
                "status": "SUCCESS",
                "rows": len(df),
                "cols": len(df.columns),
                "file": str(target_path),
            }
        except Exception as e:
            logger.error("Failed to download Football-Data for %s: %s", label, e)
            results[label] = {"status": "FAILED", "error": str(e)}

    return results


def parse_fbref_matchlog_html(filepath: Path) -> pd.DataFrame:
    """Parse FBref matchlog HTML file into a clean DataFrame."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    table = soup.find("table", {"id": "matchlogs_for"})
    if not table:
        return pd.DataFrame()

    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows_data = []
    for tr in tbody.find_all("tr"):
        row = {}
        for td in tr.find_all(["th", "td"]):
            stat = td.get("data-stat")
            if stat:
                row[stat] = td.text.strip()
        if row:
            rows_data.append(row)

    return pd.DataFrame(rows_data)


def collect_fbref_data():
    """Collect FBref schedule and match metadata across the 5 seasons."""
    logger.info("--- Collecting Primary Research Data (FBref via soccerdata) ---")
    results = {}
    try:
        import soccerdata as sd

        season_labels = [s["label"] for s in SEASONS]
        fb = sd.FBref(leagues="ENG-Premier League", seasons=season_labels, no_cache=False)
        logger.info("Reading FBref schedule for seasons: %s", season_labels)
        df_all_schedule = fb.read_schedule().reset_index()

        logger.info("Total FBref matches retrieved: %d", len(df_all_schedule))

        # Save overall schedule
        combined_path = FBREF_DIR / "fbref_premier_league_2019_2024_schedule.csv"
        df_all_schedule.to_csv(combined_path, index=False)
        logger.info("Saved combined schedule to %s", combined_path)

        # Save per-season schedule files
        for season in SEASONS:
            label = season["label"]
            code = season["fbref_code"]
            df_season = df_all_schedule[df_all_schedule["season"] == code].copy()
            season_path = FBREF_DIR / label / "schedule.csv"
            df_season.to_csv(season_path, index=False)
            logger.info("Saved season %s (%s): %d matches -> %s", label, code, len(df_season), season_path)
            results[label] = {
                "status": "SUCCESS",
                "matches": len(df_season),
                "cols": len(df_season.columns),
                "file": str(season_path),
            }

        # Parse and save sample matchlog files from cache if present
        cache_dir = Path.home() / "soccerdata" / "data" / "FBref"
        arsenal_shooting = cache_dir / "matchlogs_Arsenal_2324_shooting.html"
        if arsenal_shooting.exists():
            df_shoot = parse_fbref_matchlog_html(arsenal_shooting)
            if not df_shoot.empty:
                out_path = FBREF_DIR / "2023-2024" / "sample_shooting_matchlogs_Arsenal.csv"
                df_shoot.to_csv(out_path, index=False)
                logger.info("Saved sample shooting matchlog for Arsenal -> %s (%d rows)", out_path, len(df_shoot))

        arsenal_keeper = cache_dir / "matchlogs_Arsenal_2324_keeper.html"
        if arsenal_keeper.exists():
            df_keep = parse_fbref_matchlog_html(arsenal_keeper)
            if not df_keep.empty:
                out_path = FBREF_DIR / "2023-2024" / "sample_keeper_matchlogs_Arsenal.csv"
                df_keep.to_csv(out_path, index=False)
                logger.info("Saved sample keeper matchlog for Arsenal -> %s (%d rows)", out_path, len(df_keep))

    except Exception as e:
        logger.error("Error in FBref collection: %s", e)
        results["overall_error"] = str(e)

    return results


def main():
    logger.info("=============================================================")
    logger.info("Starting EPL Data Collection Pipeline (2019/20 - 2023/24)")
    logger.info("=============================================================")

    setup_directory_structure()
    fd_results = download_football_data()
    fb_results = collect_fbref_data()

    logger.info("=============================================================")
    logger.info("Data Collection Summary:")
    logger.info("Football-Data: %d seasons downloaded", len([k for k, v in fd_results.items() if v.get("status") == "SUCCESS"]))
    logger.info("FBref Schedules: %d seasons parsed", len([k for k, v in fb_results.items() if v.get("status") == "SUCCESS"]))
    logger.info("Raw files successfully saved in data/raw/")
    logger.info("=============================================================")


if __name__ == "__main__":
    main()
