"""A football Elo rating engine (eloratings.net-style).

Ratings start from a confederation-based prior and are updated match by
match with an importance-weighted K factor and a goal-difference multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

BASE_RATING = 1500.0

# Priors for teams never seen before. International football strength differs
# systematically by confederation; without this, minnows debut overrated.
CONFEDERATION_PRIORS = {
    "UEFA": 1600.0,
    "CONMEBOL": 1600.0,
    "CONCACAF": 1450.0,
    "CAF": 1450.0,
    "AFC": 1400.0,
    "OFC": 1300.0,
}

HOME_ADVANTAGE = 100.0  # rating points added to the home side when not neutral


def k_factor(tournament: str) -> float:
    """Match importance weight, keyed on the tournament name."""
    t = str(tournament).lower()
    if "world cup" in t and "qualification" not in t:
        return 60.0
    if any(
        name in t
        for name in (
            "euro",
            "copa américa",
            "copa america",
            "african cup",
            "africa cup",
            "asian cup",
            "gold cup",
            "confederations cup",
        )
    ) and "qualification" not in t:
        return 50.0
    if "qualification" in t or "nations league" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def goal_diff_multiplier(margin: int) -> float:
    """Bigger wins move ratings more, with diminishing returns."""
    margin = abs(int(margin))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected match score (win=1, draw=0.5, loss=0) for side A."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass
class EloRatings:
    """Mutable table of team ratings."""

    initial: dict[str, float] = field(default_factory=dict)
    home_advantage: float = HOME_ADVANTAGE
    ratings: dict[str, float] = field(init=False, default_factory=dict)

    def get(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.initial.get(team, BASE_RATING)
        return self.ratings[team]

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool,
    ) -> tuple[float, float]:
        """Update both teams for one match; returns their pre-match ratings."""
        home_pre = self.get(home_team)
        away_pre = self.get(away_team)

        adv = 0.0 if neutral else self.home_advantage
        exp_home = expected_score(home_pre + adv, away_pre)
        if home_score > away_score:
            actual = 1.0
        elif home_score < away_score:
            actual = 0.0
        else:
            actual = 0.5

        k = k_factor(tournament) * goal_diff_multiplier(home_score - away_score)
        delta = k * (actual - exp_home)
        self.ratings[home_team] = home_pre + delta
        self.ratings[away_team] = away_pre - delta
        return home_pre, away_pre


def make_initial_ratings(confederations: dict[str, str]) -> dict[str, float]:
    """Build per-team starting ratings from a team -> confederation map."""
    return {
        team: CONFEDERATION_PRIORS.get(conf, BASE_RATING)
        for team, conf in confederations.items()
    }


def run_elo(
    matches: pd.DataFrame,
    initial: dict[str, float] | None = None,
    home_advantage: float = HOME_ADVANTAGE,
) -> tuple[pd.DataFrame, EloRatings]:
    """Replay matches chronologically, recording pre-match ratings.

    Returns (copy of ``matches`` with elo_home/elo_away columns, final ratings).
    """
    elo = EloRatings(initial=initial or {}, home_advantage=home_advantage)
    home_pres: list[float] = []
    away_pres: list[float] = []
    for row in matches.itertuples(index=False):
        home_pre, away_pre = elo.update_match(
            row.home_team,
            row.away_team,
            row.home_score,
            row.away_score,
            row.tournament,
            bool(row.neutral),
        )
        home_pres.append(home_pre)
        away_pres.append(away_pre)
    out = matches.copy()
    out["elo_home"] = home_pres
    out["elo_away"] = away_pres
    return out, elo
