import pandas as pd
import pytest

from wc_predictor.data import canonical_team, load_matches
from wc_predictor.features import FEATURE_COLUMNS, build_features, outcome_label


@pytest.fixture(scope="module")
def matches():
    return load_matches()


def test_bundled_data_loads(matches):
    assert len(matches) >= 400
    assert matches["date"].is_monotonic_increasing
    assert (matches["home_score"] >= 0).all()
    assert (matches["away_score"] >= 0).all()
    # 7 tournaments of 64 matches each
    assert matches["date"].dt.year.nunique() >= 7


def test_finals_are_present(matches):
    final_2022 = matches[
        (matches["date"] == "2022-12-18")
        & (matches["home_team"] == "Argentina")
        & (matches["away_team"] == "France")
    ]
    assert len(final_2022) == 1


def test_canonical_team():
    assert canonical_team("USA") == "United States"
    assert canonical_team("Korea Republic") == "South Korea"
    assert canonical_team("Brazil") == "Brazil"


def test_outcome_label():
    assert outcome_label(2, 0) == 0
    assert outcome_label(1, 1) == 1
    assert outcome_label(0, 3) == 2


def test_features_no_leakage():
    """A match's own result must not influence its own features."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_score": [3, 0],
            "away_score": [0, 3],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )
    feats, _ = build_features(df)
    flipped = df.copy()
    flipped.loc[0, ["home_score", "away_score"]] = [0, 3]
    feats_flipped, _ = build_features(flipped)
    # First match features identical regardless of its own outcome.
    for col in FEATURE_COLUMNS:
        assert feats.loc[0, col] == feats_flipped.loc[0, col]
    # But the second match's features must differ (they see match 1's result).
    assert feats.loc[1, "elo_diff"] != feats_flipped.loc[1, "elo_diff"]


def test_feature_state_fixture(matches):
    _, state = build_features(matches)
    X = state.fixture_features("Argentina", "France", neutral=True)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 1
    # Neutral fixture between the same two teams reverses the sign.
    X_rev = state.fixture_features("France", "Argentina", neutral=True)
    assert X["elo_diff"].iloc[0] == pytest.approx(-X_rev["elo_diff"].iloc[0])
