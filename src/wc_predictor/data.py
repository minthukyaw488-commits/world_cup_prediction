"""Loading and normalising international match data.

Two supported sources, checked in order:

1. ``data/full_international_results.csv`` — the full historical dataset of
   international matches (1872-present) in the well-known ``results.csv``
   format (columns: date, home_team, away_team, home_score, away_score,
   tournament, city, country, neutral). Fetch it with
   ``scripts/download_full_data.py`` on a machine with internet access.
2. ``data/wc_matches.csv`` — a bundled fallback covering every FIFA World Cup
   finals match from 1990 through 2022, so the pipeline works offline.

Additionally, ``data/extra_matches.csv`` (same columns) is appended to
whichever source is used — put recent results there (e.g. 2026 fixtures as
they are played) and retrain to refresh the ratings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FULL_DATA_FILE = "full_international_results.csv"
BUNDLED_DATA_FILE = "wc_matches.csv"
EXTRA_DATA_FILE = "extra_matches.csv"

# Common aliases -> canonical names used in the datasets.
TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Serbia and Montenegro": "Serbia",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "UAE": "United Arab Emirates",
    "Holland": "Netherlands",
}

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
]


def canonical_team(name: str) -> str:
    """Map a team name to its canonical spelling."""
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def data_dir_default() -> Path:
    """The repository's data directory."""
    return Path(__file__).resolve().parents[2] / "data"


def load_matches(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load match data, preferring the full dataset when present.

    Returns a DataFrame sorted by date with the columns in
    ``REQUIRED_COLUMNS``; ``date`` is a datetime, ``neutral`` a bool, and
    scores are ints. Matches with missing scores are dropped.
    """
    directory = Path(data_dir) if data_dir is not None else data_dir_default()
    full = directory / FULL_DATA_FILE
    bundled = directory / BUNDLED_DATA_FILE
    path = full if full.exists() else bundled
    if not path.exists():
        raise FileNotFoundError(
            f"No match data found in {directory}. Expected {FULL_DATA_FILE} "
            f"or {BUNDLED_DATA_FILE}."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    extra = directory / EXTRA_DATA_FILE
    if extra.exists():
        extra_df = pd.read_csv(extra, comment="#")
        missing = [c for c in REQUIRED_COLUMNS if c not in extra_df.columns]
        if missing:
            raise ValueError(f"{extra} is missing columns: {missing}")
        if len(extra_df):
            df = pd.concat([df[REQUIRED_COLUMNS], extra_df[REQUIRED_COLUMNS]])

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].map(canonical_team)
    df["away_team"] = df["away_team"].map(canonical_team)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if df["neutral"].dtype != bool:
        df["neutral"] = (
            df["neutral"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        )
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df[REQUIRED_COLUMNS]


def load_teams(path: str | Path) -> pd.DataFrame:
    """Load a tournament teams file (team, group, confederation[, rating_override])."""
    teams = pd.read_csv(path, comment="#")
    for col in ("team", "group", "confederation"):
        if col not in teams.columns:
            raise ValueError(f"{path} must have a '{col}' column")
    teams["team"] = teams["team"].map(canonical_team)
    if "rating_override" not in teams.columns:
        teams["rating_override"] = pd.NA
    dupes = teams["team"][teams["team"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate teams in {path}: {dupes}")
    return teams
