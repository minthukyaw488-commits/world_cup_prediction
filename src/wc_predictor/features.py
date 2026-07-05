"""Feature engineering for the match outcome model.

All features are computed strictly from information available *before* each
match (pre-match Elo ratings), so there is no target leakage.

The feature set is deliberately small: backtesting over the 2010-2022
tournaments showed that rolling-form features degrade out-of-sample log loss
at this data size, while the Elo gap generalises across tournaments.
``abs_elo_diff`` lets the model lower the draw probability in mismatched
games.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .elo import run_elo

FEATURE_COLUMNS = [
    "elo_diff",
    "abs_elo_diff",
]

# Outcome labels, from the home side's perspective.
HOME_WIN, DRAW, AWAY_WIN = 0, 1, 2
LABEL_NAMES = {HOME_WIN: "home win", DRAW: "draw", AWAY_WIN: "away win"}


def outcome_label(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return HOME_WIN
    if home_score < away_score:
        return AWAY_WIN
    return DRAW


def build_features(
    matches: pd.DataFrame,
    initial_ratings: dict[str, float] | None = None,
    home_advantage: float | None = None,
) -> tuple[pd.DataFrame, "FeatureState"]:
    """Compute model features for every match, chronologically.

    Returns the feature frame (matches + features + ``label`` and
    ``prior_matches`` columns) and a :class:`FeatureState` snapshot that can
    generate features for future fixtures.
    """
    kwargs = {} if home_advantage is None else {"home_advantage": home_advantage}
    with_elo, elo = run_elo(matches, initial=initial_ratings, **kwargs)

    played: dict[str, int] = defaultdict(int)
    rows = []
    for row in with_elo.itertuples(index=False):
        adv = 0.0 if row.neutral else elo.home_advantage
        eff_home = row.elo_home + adv
        rows.append(
            {
                "elo_diff": eff_home - row.elo_away,
                "abs_elo_diff": abs(eff_home - row.elo_away),
                "label": outcome_label(row.home_score, row.away_score),
                "prior_matches": min(played[row.home_team], played[row.away_team]),
            }
        )
        played[row.home_team] += 1
        played[row.away_team] += 1

    features = pd.concat(
        [with_elo.reset_index(drop=True), pd.DataFrame(rows)], axis=1
    )
    state = FeatureState(
        ratings=dict(elo.ratings),
        home_advantage=elo.home_advantage,
    )
    return features, state


class FeatureState:
    """Snapshot of ratings after replaying history.

    Used to build the feature vector for a hypothetical future fixture.
    """

    def __init__(self, ratings: dict[str, float], home_advantage: float):
        self.ratings = ratings
        self.home_advantage = home_advantage

    def rating(self, team: str, default: float = 1500.0) -> float:
        return self.ratings.get(team, default)

    def fixture_features(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        rating_overrides: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        overrides = rating_overrides or {}
        home = overrides.get(home_team, self.rating(home_team))
        away = overrides.get(away_team, self.rating(away_team))
        eff_home = home + (0.0 if neutral else self.home_advantage)
        return pd.DataFrame(
            [
                {
                    "elo_diff": eff_home - away,
                    "abs_elo_diff": abs(eff_home - away),
                }
            ]
        )[FEATURE_COLUMNS]
