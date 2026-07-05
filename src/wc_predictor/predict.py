"""Predict a single fixture.

Usage:
    python -m wc_predictor.predict "Argentina" "France"
    python -m wc_predictor.predict "United States" "Mexico" --home
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np

from .data import canonical_team
from .features import LABEL_NAMES


def predict_fixture(model: dict, home_team: str, away_team: str, neutral: bool = True) -> dict:
    state = model["state"]
    home_team = canonical_team(home_team)
    away_team = canonical_team(away_team)
    for team in (home_team, away_team):
        if team not in state.ratings:
            print(f"note: no match history for '{team}'; using default rating 1500")
    X = state.fixture_features(home_team, away_team, neutral=neutral)
    proba = model["classifier"].predict_proba(X)[0]
    lam_home = float(np.exp(model["poisson_home"].intercept_ + model["poisson_home"].coef_[0] * X["elo_diff"].iloc[0]))
    lam_away = float(np.exp(model["poisson_away"].intercept_ + model["poisson_away"].coef_[0] * X["elo_diff"].iloc[0]))
    return {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,
        "elo_home": state.rating(home_team),
        "elo_away": state.rating(away_team),
        "probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
        "expected_goals": {home_team: lam_home, away_team: lam_away},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument(
        "--home", action="store_true",
        help="first team plays at home (default: neutral venue)",
    )
    args = parser.parse_args(argv)

    model = joblib.load(args.model)
    result = predict_fixture(model, args.home_team, args.away_team, neutral=not args.home)

    print(f"\n{result['home_team']} vs {result['away_team']}"
          f" ({'home' if not result['neutral'] else 'neutral venue'})")
    print(f"Elo: {result['elo_home']:.0f} vs {result['elo_away']:.0f}")
    for name, p in result["probabilities"].items():
        print(f"  {name:<9} {p:6.1%}")
    eg = result["expected_goals"]
    print("Expected goals: "
          + ", ".join(f"{team} {lam:.2f}" for team, lam in eg.items()))


if __name__ == "__main__":
    main()
