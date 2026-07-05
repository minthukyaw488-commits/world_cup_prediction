import pytest

from wc_predictor.elo import (
    EloRatings,
    expected_score,
    goal_diff_multiplier,
    k_factor,
)


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == pytest.approx(1.0)
    assert expected_score(1800, 1400) > 0.9


def test_goal_diff_multiplier():
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(3) == pytest.approx(14 / 8)
    assert goal_diff_multiplier(-3) == pytest.approx(14 / 8)
    assert goal_diff_multiplier(7) > goal_diff_multiplier(3)


def test_k_factor_ordering():
    assert k_factor("FIFA World Cup") > k_factor("FIFA World Cup qualification")
    assert k_factor("FIFA World Cup qualification") > k_factor("Friendly")


def test_update_moves_ratings_towards_winner():
    elo = EloRatings()
    elo.update_match("A", "B", 2, 0, "FIFA World Cup", neutral=True)
    assert elo.get("A") > 1500 > elo.get("B")
    # Zero-sum update.
    assert elo.get("A") + elo.get("B") == pytest.approx(3000)


def test_upset_moves_more_than_expected_win():
    strong_wins = EloRatings(initial={"S": 1800, "W": 1400})
    strong_wins.update_match("S", "W", 1, 0, "FIFA World Cup", neutral=True)
    gain_expected = strong_wins.get("S") - 1800

    upset = EloRatings(initial={"S": 1800, "W": 1400})
    upset.update_match("W", "S", 1, 0, "FIFA World Cup", neutral=True)
    gain_upset = upset.get("W") - 1400

    assert gain_upset > gain_expected > 0


def test_draw_favours_underdog():
    elo = EloRatings(initial={"S": 1800, "W": 1400})
    elo.update_match("S", "W", 1, 1, "FIFA World Cup", neutral=True)
    assert elo.get("S") < 1800
    assert elo.get("W") > 1400


def test_home_advantage_reduces_home_gain():
    neutral = EloRatings()
    neutral.update_match("A", "B", 1, 0, "Friendly", neutral=True)
    at_home = EloRatings()
    at_home.update_match("A", "B", 1, 0, "Friendly", neutral=False)
    assert at_home.get("A") < neutral.get("A")
