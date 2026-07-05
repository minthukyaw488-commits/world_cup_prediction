"""Monte Carlo simulation of the 2026 World Cup (48-team format).

Usage:
    python -m wc_predictor.simulate [--runs 10000] [--teams data/wc2026_teams.csv]

Format modelled: 12 groups of 4; group winners, runners-up and the 8 best
third-placed teams advance to a round of 32, then straight knockout. The
round-of-32 bracket uses a fixed seeded template (winners vs thirds/
runners-up, avoiding same-group rematches where possible) — a simplification
of FIFA's official allocation tables.

Scorelines are sampled from the trained Poisson models; drawn knockout
matches go to extra time (goal rate scaled by 1/3) and, if still level, to a
penalty shootout whose odds tilt slightly with the rating gap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from .data import load_teams
from .elo import CONFEDERATION_PRIORS, BASE_RATING

STAGES = ["R32", "R16", "QF", "SF", "final", "champion"]

# Round-of-32 template. W=group winner, R=runner-up, T=third (ranked 1-8 by
# group-stage record). Winners paired with thirds; runners-up paired together.
R32_TEMPLATE = [
    ("W", 0, "T", 7), ("R", 2, "R", 5),
    ("W", 4, "T", 6), ("W", 8, "R", 10),
    ("W", 2, "T", 5), ("R", 0, "R", 3),
    ("W", 6, "T", 4), ("W", 10, "R", 8),
    ("W", 1, "T", 3), ("R", 4, "R", 7),
    ("W", 5, "T", 2), ("W", 9, "R", 11),
    ("W", 3, "T", 1), ("R", 1, "R", 6),
    ("W", 7, "T", 0), ("W", 11, "R", 9),
]


class MatchSampler:
    """Samples scorelines from the trained Poisson goal models."""

    def __init__(self, model: dict, rng: np.random.Generator):
        self.rng = rng
        self.state = model["state"]
        ph, pa = model["poisson_home"], model["poisson_away"]
        self.home_params = (float(ph.intercept_), float(ph.coef_[0]))
        self.away_params = (float(pa.intercept_), float(pa.coef_[0]))

    def goal_rates(self, elo_a: float, elo_b: float) -> tuple[float, float]:
        diff = elo_a - elo_b
        lam_a = np.exp(self.home_params[0] + self.home_params[1] * diff)
        lam_b = np.exp(self.away_params[0] + self.away_params[1] * diff)
        return float(np.clip(lam_a, 0.05, 8.0)), float(np.clip(lam_b, 0.05, 8.0))

    def group_match(self, elo_a: float, elo_b: float) -> tuple[int, int]:
        lam_a, lam_b = self.goal_rates(elo_a, elo_b)
        return int(self.rng.poisson(lam_a)), int(self.rng.poisson(lam_b))

    def knockout_winner(self, team_a: str, team_b: str, elo: dict[str, float]) -> str:
        elo_a, elo_b = elo[team_a], elo[team_b]
        ga, gb = self.group_match(elo_a, elo_b)
        if ga == gb:  # extra time at a third of the regulation goal rate
            lam_a, lam_b = self.goal_rates(elo_a, elo_b)
            ga += int(self.rng.poisson(lam_a / 3.0))
            gb += int(self.rng.poisson(lam_b / 3.0))
        if ga == gb:  # penalties: near coin flip, tilted by rating gap
            p_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 2000.0))
            return team_a if self.rng.random() < p_a else team_b
        return team_a if ga > gb else team_b


def simulate_group(teams: list[str], elo: dict[str, float], sampler: MatchSampler):
    """Round-robin; returns teams ordered by points, goal difference, goals."""
    stats = {t: [0, 0, 0] for t in teams}  # points, gd, gf
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            ga, gb = sampler.group_match(elo[a], elo[b])
            stats[a][1] += ga - gb
            stats[b][1] += gb - ga
            stats[a][2] += ga
            stats[b][2] += gb
            if ga > gb:
                stats[a][0] += 3
            elif gb > ga:
                stats[b][0] += 3
            else:
                stats[a][0] += 1
                stats[b][0] += 1
    order = sorted(
        teams,
        key=lambda t: (stats[t][0], stats[t][1], stats[t][2], sampler.rng.random()),
        reverse=True,
    )
    return order, stats


def simulate_tournament(
    groups: dict[str, list[str]], elo: dict[str, float], sampler: MatchSampler
) -> dict[str, str]:
    """One tournament run; returns team -> furthest stage reached."""
    reached: dict[str, str] = {}
    group_names = sorted(groups)
    winners, runners, thirds = [], [], []
    for name in group_names:
        order, stats = simulate_group(groups[name], elo, sampler)
        winners.append(order[0])
        runners.append(order[1])
        thirds.append((order[2], tuple(stats[order[2]])))

    # Best 8 thirds by group record.
    thirds_sorted = sorted(thirds, key=lambda x: (x[1], sampler.rng.random()), reverse=True)
    best_thirds = [t for t, _ in thirds_sorted[:8]]

    pools = {"W": winners, "R": runners, "T": best_thirds}
    field32 = []
    for kind_a, ia, kind_b, ib in R32_TEMPLATE:
        field32.append((pools[kind_a][ia % len(pools[kind_a])],
                        pools[kind_b][ib % len(pools[kind_b])]))

    for pair in field32:
        for t in pair:
            reached[t] = "R32"

    rnd = [sampler.knockout_winner(a, b, elo) for a, b in field32]
    for stage in ("R16", "QF", "SF"):
        for t in rnd:
            reached[t] = stage
        rnd = [
            sampler.knockout_winner(rnd[i], rnd[i + 1], elo)
            for i in range(0, len(rnd), 2)
        ]
    finalists = rnd  # two teams
    for t in finalists:
        reached[t] = "final"
    champion = sampler.knockout_winner(finalists[0], finalists[1], elo)
    reached[champion] = "champion"
    return reached


def team_ratings(teams: pd.DataFrame, model: dict) -> dict[str, float]:
    """Rating per team: explicit override > model Elo > confederation prior."""
    state = model["state"]
    ratings = {}
    for row in teams.itertuples(index=False):
        if pd.notna(row.rating_override):
            ratings[row.team] = float(row.rating_override)
        elif row.team in state.ratings:
            ratings[row.team] = state.ratings[row.team]
        else:
            ratings[row.team] = CONFEDERATION_PRIORS.get(row.confederation, BASE_RATING)
    return ratings


def run_simulation(
    teams_csv: str, model_path: str, runs: int = 10000, seed: int = 42
) -> pd.DataFrame:
    teams = load_teams(teams_csv)
    model = joblib.load(model_path)
    rng = np.random.default_rng(seed)
    sampler = MatchSampler(model, rng)
    elo = team_ratings(teams, model)

    groups: dict[str, list[str]] = defaultdict(list)
    for row in teams.itertuples(index=False):
        groups[row.group].append(row.team)
    for name, members in groups.items():
        if len(members) != 4:
            raise ValueError(f"Group {name} has {len(members)} teams; expected 4")
    if len(groups) != 12:
        raise ValueError(f"Expected 12 groups, found {len(groups)}")

    counts = {t: dict.fromkeys(STAGES, 0) for t in elo}
    stage_rank = {s: i for i, s in enumerate(STAGES)}
    for _ in range(runs):
        reached = simulate_tournament(groups, elo, sampler)
        for team, stage in reached.items():
            for s in STAGES[: stage_rank[stage] + 1]:
                counts[team][s] += 1

    result = pd.DataFrame(
        [
            {
                "team": t,
                "group": teams.loc[teams["team"] == t, "group"].iloc[0],
                "rating": round(elo[t], 1),
                **{f"P({s})": counts[t][s] / runs for s in STAGES},
            }
            for t in elo
        ]
    ).sort_values("P(champion)", ascending=False).reset_index(drop=True)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", default="data/wc2026_teams.csv")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--runs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="optional CSV path for full results")
    args = parser.parse_args(argv)

    result = run_simulation(args.teams, args.model, args.runs, args.seed)
    pd.set_option("display.width", 140)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(result.head(20).to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
