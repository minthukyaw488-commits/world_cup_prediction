import numpy as np
import pytest

import joblib

from wc_predictor import train as train_mod
from wc_predictor.data import data_dir_default
from wc_predictor.predict import predict_fixture
from wc_predictor.simulate import STAGES, run_simulation


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "model.pkl"
    metrics = train_mod.train(None, str(out))
    return out, metrics


def test_training_beats_baseline(trained_model):
    _, metrics = trained_model
    results = metrics["backtest"]["results"]
    selected = metrics["selected_model"]
    assert results[selected]["mean_log_loss"] < results["baseline"]["mean_log_loss"]
    assert 0 < results[selected]["mean_accuracy"] <= 1


def test_predict_probabilities_sum_to_one(trained_model):
    out, _ = trained_model
    model = joblib.load(out)
    result = predict_fixture(model, "Brazil", "Germany", neutral=True)
    probs = list(result["probabilities"].values())
    assert sum(probs) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in probs)


def test_stronger_team_favoured(trained_model):
    out, _ = trained_model
    model = joblib.load(out)
    # France (multiple deep runs) vs a team with far less success.
    result = predict_fixture(model, "France", "Saudi Arabia", neutral=True)
    assert result["probabilities"]["home win"] > result["probabilities"]["away win"]


def test_simulation_probabilities(trained_model):
    out, _ = trained_model
    teams_csv = data_dir_default() / "wc2026_teams.csv"
    result = run_simulation(str(teams_csv), str(out), runs=300, seed=7)

    assert len(result) == 48
    # Exactly one champion per run.
    assert result["P(champion)"].sum() == pytest.approx(1.0)
    # Two finalists, four semifinalists, ... per run.
    assert result["P(final)"].sum() == pytest.approx(2.0)
    assert result["P(SF)"].sum() == pytest.approx(4.0)
    assert result["P(R32)"].sum() == pytest.approx(32.0)
    # Stage probabilities must be monotonically non-increasing per team.
    for _, row in result.iterrows():
        probs = [row[f"P({s})"] for s in STAGES]
        assert all(a >= b - 1e-9 for a, b in zip(probs, probs[1:]))
