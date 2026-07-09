import numpy as np
import pytest

import joblib

from wc_predictor import train as train_mod
from wc_predictor.data import data_dir_default
from wc_predictor.predict import predict_fixture
from wc_predictor.simulate import STAGES, run_knockout_simulation, run_simulation


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


def test_knockout_simulation(trained_model):
    out, _ = trained_model
    bracket = data_dir_default() / "example_r16_bracket.csv"
    result = run_knockout_simulation(str(bracket), str(out), runs=400, seed=3)
    assert len(result) == 16
    assert result["P(champion)"].sum() == pytest.approx(1.0)
    assert result["P(final)"].sum() == pytest.approx(2.0)
    assert result["P(QF)"].sum() == pytest.approx(8.0)
    # A first-round pair can produce at most one quarter-finalist:
    # P(QF) of bracket neighbours must sum to exactly 1.
    by_team = result.set_index("team")
    import pandas as pd
    teams_in_order = pd.read_csv(bracket, comment="#")["team"].tolist()
    for i in range(0, 16, 2):
        pair = by_team.loc[[teams_in_order[i], teams_in_order[i + 1]], "P(QF)"]
        assert pair.sum() == pytest.approx(1.0)


def test_sampler_outcomes_follow_ratings(trained_model):
    out, _ = trained_model
    model = joblib.load(out)
    from wc_predictor.simulate import MatchSampler

    sampler = MatchSampler(model, np.random.default_rng(0))
    p_even = sampler.outcome_probs(1600.0, 1600.0)
    p_big = sampler.outcome_probs(1900.0, 1400.0)
    assert p_even.sum() == pytest.approx(1.0)
    assert p_big.sum() == pytest.approx(1.0)
    assert p_even[0] == pytest.approx(p_even[2], abs=5e-3)  # symmetric when equal
    assert p_big[0] > 0.6  # heavy favourite
    assert p_big[1] < p_even[1]  # mismatches draw less
    # Sampled group scorelines agree with the sampled outcome by construction:
    # over many samples, win share should approximate the classifier's P(win).
    wins = draws = 0
    n = 3000
    for _ in range(n):
        ga, gb = sampler.group_match(1900.0, 1400.0)
        wins += ga > gb
        draws += ga == gb
    assert wins / n == pytest.approx(p_big[0], abs=0.03)
    assert draws / n == pytest.approx(p_big[1], abs=0.03)


def test_extra_matches_are_appended(tmp_path):
    import shutil

    from wc_predictor.data import load_matches

    src = data_dir_default()
    shutil.copy(src / "wc_matches.csv", tmp_path / "wc_matches.csv")
    base = load_matches(tmp_path)
    (tmp_path / "extra_matches.csv").write_text(
        "# comment line\n"
        "date,home_team,away_team,home_score,away_score,tournament,neutral\n"
        "2026-06-11,Mexico,South Africa,2,0,FIFA World Cup,FALSE\n"
    )
    combined = load_matches(tmp_path)
    assert len(combined) == len(base) + 1
    last = combined.iloc[-1]
    assert last["home_team"] == "Mexico" and last["home_score"] == 2
    assert not last["neutral"]


def test_rating_noise_flattens_probabilities(trained_model):
    out, _ = trained_model
    bracket = data_dir_default() / "example_r16_bracket.csv"
    sharp = run_knockout_simulation(str(bracket), str(out), runs=2000, seed=5)
    noisy = run_knockout_simulation(
        str(bracket), str(out), runs=2000, seed=5, rating_noise=150.0
    )
    # Heavy rating uncertainty should compress the favourite's edge.
    assert noisy["P(champion)"].max() < sharp["P(champion)"].max()
