"""Train the match outcome model and the scoreline (Poisson) models.

Usage:
    python -m wc_predictor.train [--data-dir data] [--model-out models/model.pkl]

Evaluation uses an expanding-window backtest: for each of the last few
tournament years, the model is trained only on earlier matches and scored on
that year. With a few hundred matches, a single chronological split is
dominated by one tournament's quirks (2022 was extremely upset-heavy); the
backtest average is a much more stable estimate. The candidate with the best
mean backtest log loss is then refit on all data and saved, together with
two Poisson goal models used by the tournament simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_matches
from .features import FEATURE_COLUMNS, build_features

MIN_PRIOR_MATCHES = 3  # both teams need this much history before a match is trainable
BACKTEST_YEARS = 4  # hold out each of the last N tournament years in turn


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Swap the two sides of every match (features and label).

    Which team is 'home' in a neutral-venue row is an artifact of listing
    order (seeded teams tend to be listed first), and true home advantage is
    already folded into ``elo_diff``. Training on ``df + mirror(df)`` removes
    the listing-order bias and makes the fitted model exactly symmetric.
    """
    out = df.copy()
    out["elo_diff"] = -df["elo_diff"]
    out["label"] = df["label"].map({0: 2, 1: 1, 2: 0})
    out["home_score"], out["away_score"] = (
        df["away_score"].to_numpy(),
        df["home_score"].to_numpy(),
    )
    return out


def symmetrized(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, mirror(df)], ignore_index=True)


def make_candidates(seed: int) -> dict:
    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.1, max_iter=60, random_state=seed
        ),
    }


def backtest(usable: pd.DataFrame, candidates: dict) -> dict:
    """Expanding-window backtest by year; returns per-candidate mean metrics."""
    years = sorted(usable["date"].dt.year.unique())
    test_years = years[-min(BACKTEST_YEARS, len(years) - 1):]

    scores: dict[str, dict] = {
        name: {"log_loss": [], "accuracy": []} for name in candidates
    }
    scores["baseline"] = {"log_loss": [], "accuracy": []}

    for year in test_years:
        train_df = usable[usable["date"].dt.year < year]
        test_df = usable[usable["date"].dt.year == year]
        if len(train_df) < 50 or len(test_df) == 0:
            continue
        train_sym = symmetrized(train_df)
        X_tr, y_tr = train_sym[FEATURE_COLUMNS], train_sym["label"]
        X_te, y_te = test_df[FEATURE_COLUMNS], test_df["label"]
        for name, model in candidates.items():
            fitted = clone(model).fit(X_tr, y_tr)
            proba = fitted.predict_proba(X_te)
            scores[name]["log_loss"].append(log_loss(y_te, proba, labels=[0, 1, 2]))
            scores[name]["accuracy"].append(
                accuracy_score(y_te, proba.argmax(axis=1))
            )
        class_freq = np.bincount(y_tr, minlength=3) / len(y_tr)
        proba = np.tile(class_freq, (len(y_te), 1))
        scores["baseline"]["log_loss"].append(log_loss(y_te, proba, labels=[0, 1, 2]))
        scores["baseline"]["accuracy"].append(
            accuracy_score(y_te, proba.argmax(axis=1))
        )

    return {
        "test_years": [int(y) for y in test_years],
        "results": {
            name: {
                "mean_log_loss": float(np.mean(s["log_loss"])),
                "mean_accuracy": float(np.mean(s["accuracy"])),
                "per_year_log_loss": [round(float(x), 4) for x in s["log_loss"]],
            }
            for name, s in scores.items()
        },
    }


def train(data_dir: str | None, model_out: str, seed: int = 42) -> dict:
    matches = load_matches(data_dir)
    features, state = build_features(matches)

    usable = features[features["prior_matches"] >= MIN_PRIOR_MATCHES].reset_index(drop=True)
    if len(usable) < 100:
        raise RuntimeError(
            f"Only {len(usable)} usable matches after burn-in; need at least 100."
        )

    candidates = make_candidates(seed)
    bt = backtest(usable, candidates)
    best_name = min(
        (n for n in candidates),
        key=lambda n: bt["results"][n]["mean_log_loss"],
    )

    # Refit the winning candidate on everything (symmetrized) before saving.
    usable_sym = symmetrized(usable)
    best = clone(candidates[best_name]).fit(
        usable_sym[FEATURE_COLUMNS], usable_sym["label"]
    )

    # Scoreline models for the simulator (goals as a function of rating gap).
    goal_X = usable_sym[["elo_diff"]]
    poisson_home = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(
        goal_X, usable_sym["home_score"]
    )
    poisson_away = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(
        goal_X, usable_sym["away_score"]
    )

    metrics = {
        "n_matches_total": int(len(matches)),
        "n_matches_trainable": int(len(usable)),
        "date_range": [str(usable["date"].min().date()), str(usable["date"].max().date())],
        "backtest": bt,
        "selected_model": best_name,
    }

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": best,
            "classifier_name": best_name,
            "poisson_home": poisson_home,
            "poisson_away": poisson_away,
            "feature_columns": FEATURE_COLUMNS,
            "state": state,
            "metrics": metrics,
        },
        out_path,
    )
    metrics["model_path"] = str(out_path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="directory containing match CSVs")
    parser.add_argument("--model-out", default="models/model.pkl")
    args = parser.parse_args(argv)

    metrics = train(args.data_dir, args.model_out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
