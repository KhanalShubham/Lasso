"""OpenFootball Canonical Match Table Parser and Normalizer.

Transforms raw OpenFootball Football.TXT match files into a standardized,
chronologically sorted match-level dataset:
`data/processed/epl_matches.csv`

Fields:
- `match_id`: Deterministic unique identifier (`season_round_home_away`)
- `date`: Kickoff date (YYYY-MM-DD)
- `time`: Kickoff time (HH:MM or empty)
- `season`: Season string (e.g., '2019-20')
- `matchday`: Integer round/matchday (1 to 38)
- `home_team`: Normalized canonical club name
- `away_team`: Normalized canonical club name
- `home_goals`: Full-time home score (integer)
- `away_goals`: Full-time away score (integer)
- `target`: Categorical outcome ('HOME_WIN', 'DRAW', 'AWAY_WIN')
- `target_multinomial`: Discrete outcome code (2: HOME_WIN, 1: DRAW, 0: AWAY_WIN)
"""

from datetime import date
from pathlib import Path
import re
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = WORKSPACE_ROOT / "data" / "raw" / "openfootball"
PROCESSED_DIR = WORKSPACE_ROOT / "data" / "processed"

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

# Canonical 26 Premier League Clubs (2019/20 - 2023/24)
# Mapping raw names from openfootball to canonical club names
TEAM_CANONICAL_MAPPING = {
    "AFC Bournemouth": "AFC Bournemouth",
    "Arsenal": "Arsenal",
    "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Aston Villa FC": "Aston Villa",
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Burnley FC": "Burnley",
    "Chelsea": "Chelsea",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Crystal Palace FC": "Crystal Palace",
    "Everton": "Everton",
    "Everton FC": "Everton",
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",
    "Leeds United": "Leeds United",
    "Leeds United FC": "Leeds United",
    "Leicester City": "Leicester City",
    "Leicester City FC": "Leicester City",
    "Liverpool": "Liverpool",
    "Liverpool FC": "Liverpool",
    "Luton Town": "Luton Town",
    "Luton Town FC": "Luton Town",
    "Manchester City": "Manchester City",
    "Manchester City FC": "Manchester City",
    "Manchester United": "Manchester United",
    "Manchester United FC": "Manchester United",
    "Newcastle United": "Newcastle United",
    "Newcastle United FC": "Newcastle United",
    "Norwich City": "Norwich City",
    "Norwich City FC": "Norwich City",
    "Nottingham Forest": "Nottingham Forest",
    "Nottingham Forest FC": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield United FC": "Sheffield United",
    "Southampton": "Southampton",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur": "Tottenham Hotspur",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Watford": "Watford",
    "Watford FC": "Watford",
    "West Bromwich Albion": "West Bromwich Albion",
    "West Bromwich Albion FC": "West Bromwich Albion",
    "West Ham United": "West Ham United",
    "West Ham United FC": "West Ham United",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
}

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

DATE_REGEX = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]+)\s+(\d{1,2})(?:\s+(\d{4}))?$")
MATCH_REGEX = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?([A-Za-z0-9&.\'\-\s]+?)\s+(\d+)-(\d+)(?:\s*\([^)]*\))?\s+([A-Za-z0-9&.\'\-\s]+?)\s*$"
)
ROUND_REGEX = re.compile(r"(?:Matchday|Round)\s+(\d+)", re.IGNORECASE)


def normalize_team_name(raw_name: str) -> str:
    """Map raw openfootball team string to canonical club name."""
    clean_name = raw_name.strip()
    if clean_name in TEAM_CANONICAL_MAPPING:
        return TEAM_CANONICAL_MAPPING[clean_name]
    # Fallback: strip trailing ' FC' or ' AFC'
    stripped = re.sub(r"\s+(?:FC|AFC)$", "", clean_name).strip()
    if stripped in TEAM_CANONICAL_MAPPING:
        return TEAM_CANONICAL_MAPPING[stripped]
    raise ValueError(f"Unknown team name encountered in source: {raw_name}")


def parse_season_file(season: str, file_path: Path) -> list[dict]:
    """Parse a single season Football.TXT file into structured match dicts."""
    start_year = int(season.split("-")[0])
    end_year = start_year + 1

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    matches = []
    current_date = None
    current_round = 0

    for line_idx, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("=") or line_clean.startswith("#"):
            continue

        # Check round header
        round_match = ROUND_REGEX.search(line_clean)
        if round_match:
            current_round = int(round_match.group(1))
            continue

        # Check date header
        date_match = DATE_REGEX.match(line_clean)
        if date_match:
            dow, month_str, day_str, year_str = date_match.groups()
            month_num = MONTH_MAP[month_str]
            day_num = int(day_str)
            # EPL seasons start in August/September and end in May/July
            if year_str:
                yr = int(year_str)
            else:
                yr = start_year if month_num >= 8 else end_year
            current_date = date(yr, month_num, day_num)
            continue

        # Check match score line
        m_match = MATCH_REGEX.match(line)
        if m_match:
            time_str, home_raw, h_goals_str, a_goals_str, away_raw = m_match.groups()
            home_team = normalize_team_name(home_raw)
            away_team = normalize_team_name(away_raw)
            home_goals = int(h_goals_str)
            away_goals = int(a_goals_str)

            if home_goals > away_goals:
                target = "HOME_WIN"
                target_code = 2
            elif home_goals == away_goals:
                target = "DRAW"
                target_code = 1
            else:
                target = "AWAY_WIN"
                target_code = 0

            match_id = f"{season}_GW{current_round:02d}_{home_team.replace(' ', '')}_v_{away_team.replace(' ', '')}"

            matches.append({
                "match_id": match_id,
                "season": season,
                "date": current_date.isoformat(),
                "time": time_str if time_str else "15:00",
                "matchday": current_round,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "target": target,
                "target_multinomial": target_code,
            })
        else:
            # Report unexpected lines
            raise ValueError(f"Line {line_idx+1} in {season} could not be parsed: {repr(line)}")

    return matches


def build_canonical_match_table() -> pd.DataFrame:
    """Parse all 5 seasons, validate integrity, and save canonical match table."""
    print("=" * 70)
    print("PARSING OPENFOOTBALL SEASONS INTO CANONICAL MATCH TABLE")
    print("=" * 70)

    all_matches = []
    for season in SEASONS:
        season_file = RAW_DIR / season / "1-premierleague.txt"
        if not season_file.exists():
            raise FileNotFoundError(f"Missing raw file: {season_file}")
        season_matches = parse_season_file(season, season_file)
        print(f"Season {season}: Parsed {len(season_matches)} fixtures.")
        if len(season_matches) != 380:
            raise ValueError(f"Season {season} expected 380 fixtures, got {len(season_matches)}!")
        all_matches.extend(season_matches)

    df = pd.DataFrame(all_matches)

    # 1. Validation: Exact count
    assert len(df) == 1900, f"Expected 1,900 matches, got {len(df)}"

    # 2. Validation: Zero nulls
    assert df.isnull().sum().sum() == 0, "Null values detected in canonical match table!"

    # 3. Validation: Unique match IDs and fixtures per season
    assert df["match_id"].nunique() == 1900, "Duplicate match IDs found!"
    for s, g in df.groupby("season"):
        pair_counts = g.groupby(["home_team", "away_team"]).size()
        assert (pair_counts == 1).all(), f"Duplicate fixture in season {s}!"

    # 4. Chronological Sorting: primary by date, secondary by time, tertiary by matchday
    df["datetime_sort"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values(by=["datetime_sort", "matchday", "match_id"]).reset_index(drop=True)
    df = df.drop(columns=["datetime_sort"])

    # 5. Save canonical table
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DIR / "epl_matches.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved canonical match table to: {out_csv}")
    print(f"Total Rows: {len(df)}")
    print(f"Target distribution:\n{df['target'].value_counts().to_string()}")
    print("=" * 70)
    return df


if __name__ == "__main__":
    build_canonical_match_table()
