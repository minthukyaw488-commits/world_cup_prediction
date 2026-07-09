"""Print the current Elo rating table from a trained model.

Usage:
    python -m wc_predictor.ratings [--model models/model.pkl] [--top 30]
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd


def ratings_table(model: dict) -> pd.DataFrame:
    state = model["state"]
    table = pd.DataFrame(
        sorted(state.ratings.items(), key=lambda kv: -kv[1]),
        columns=["team", "elo"],
    )
    table.index = range(1, len(table) + 1)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)

    table = ratings_table(joblib.load(args.model))
    with pd.option_context("display.float_format", "{:.0f}".format):
        print(table.head(args.top).to_string())


if __name__ == "__main__":
    main()
