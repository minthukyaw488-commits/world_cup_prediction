# World Cup Prediction

[![CI](https://github.com/minthukyaw488-commits/world_cup_prediction/actions/workflows/ci.yml/badge.svg?branch=claude%2Fworld-cup-prediction-model-ezv8yf)](https://github.com/minthukyaw488-commits/world_cup_prediction/actions/workflows/ci.yml)

A complete, offline-capable pipeline for predicting international football
matches and simulating the FIFA World Cup 2026:

- **Elo rating engine** (eloratings.net-style): importance-weighted K factor,
  goal-difference multiplier, home advantage, confederation-based priors.
- **Match outcome model**: win/draw/loss classifier trained on pre-match Elo
  features (multinomial logistic regression vs. gradient boosting, selected by
  expanding-window backtest), plus Poisson models for scorelines.
- **Tournament simulator**: Monte Carlo simulation of the 48-team 2026 format
  (12 groups of 4, best thirds, round of 32 → final).

## Quick start

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m pytest            # run the test suite

python -m wc_predictor.train                      # train, prints backtest metrics
python -m wc_predictor.predict "Argentina" "France"
python -m wc_predictor.simulate --runs 10000 --out output/wc2026_forecast.csv

# Mid-tournament? Simulate just a knockout bracket (2/4/8/16/32 teams,
# listed in bracket order — adjacent rows meet in round one):
python -m wc_predictor.simulate --knockout data/example_r16_bracket.csv

# Express uncertainty about team strength (Elo points of per-run noise;
# flattens overconfident probabilities):
python -m wc_predictor.simulate --runs 10000 --rating-noise 50
```

## Results (bundled data)

Expanding-window backtest — for each World Cup year the model is trained only
on earlier tournaments and scored on that year:

| Model                         | Mean log loss | Mean accuracy |
|-------------------------------|---------------|---------------|
| Logistic regression (selected)| **1.034**     | **53.3%**     |
| Gradient boosting             | 1.121         | 48.3%         |
| Class-frequency baseline      | 1.075         | 41.5%         |

Training rows are *symmetrized* (each match is also included with the two
sides swapped): which team is listed "home" at a neutral venue is an artifact
of fixture listing order, and mirroring both removes that bias and makes the
model exactly symmetric.

Test years: 2010, 2014, 2018, 2022. The model beats the baseline in three of
the four years; 2022 (Saudi Arabia over Argentina, Japan over Germany and
Spain, Morocco to the semis) is the exception — no rating-based model looked
good in Qatar.

A negative result worth knowing: regressing idle teams' Elo toward the mean
between tournaments (`mean_reversion_rate` in `elo.py`) *hurt* backtest log
loss monotonically (0.0 → 1.045, 0.1 → 1.057, 0.3 → 1.080), so it ships
disabled — even a four-year-old rating carries real signal. Rolling-form
features were dropped for the same reason.

## Data

Two data sources, checked in this order:

1. **`data/full_international_results.csv`** *(recommended, not bundled)* —
   the community-maintained dataset of ~47,000 international matches since
   1872 (`martj42/international_results`, CC0). Fetch it on a machine with
   internet access:

   ```bash
   python scripts/download_full_data.py
   ```

   Training automatically switches to it when present, which gives every team
   a proper rating history (qualifiers, continental cups, friendlies).

2. **`data/wc_matches.csv`** *(bundled fallback)* — all 552 World Cup finals
   matches 1990–2022, so everything works offline out of the box.

   ⚠️ **Provenance**: the bundled file was compiled from public match records
   by an AI assistant working offline. Spot-checks are encouraged; isolated
   scoreline errors are possible (they have negligible effect on aggregate
   ratings, but don't cite this file as an authoritative record). Scores
   include extra time (penalty shootouts count as draws, matching the
   convention of the full dataset). Matches involving the host country are
   listed with the host as the home team and `neutral=FALSE`.

   Known artifact of training on World Cup finals only: teams that skipped
   tournaments or exited via penalty shootouts (recorded as draws) can end up
   over-rated — e.g. the Netherlands ranks #1 because it never lost a
   90-minute knockout match in this window. The full dataset fixes this.

### 2026 teams file

`data/wc2026_teams.csv` lists the 48 qualified teams with **illustrative,
pot-based group assignments — not the official draw** — and placeholder rows
for the playoff spots decided in March 2026. Edit it to match reality before
taking a simulation seriously. You can pin any team's strength with the
`rating_override` column (this project's Elo scale, roughly 1300–1800).

## How it works

```
data/*.csv ──► data.load_matches ──► elo.run_elo ──► features.build_features
                                                          │
                            ┌─────────────────────────────┤
                            ▼                             ▼
                 train: W/D/L classifier         train: Poisson goal models
                            └──────────────┬──────────────┘
                                           ▼
                                   models/model.pkl
                                    │            │
                       predict (single match)   simulate (Monte Carlo 2026)
```

- **Features**: `elo_diff` (with home advantage folded in) and
  `abs_elo_diff` (lets the model reduce draw probability in mismatches).
  Rolling-form features were tried and *removed*: they degraded backtest
  log loss at this data size.
- **No leakage**: every feature uses strictly pre-match information; there is
  a test asserting a match's own result cannot influence its own features.
- **Simulation is classifier-driven**: win/draw/loss is sampled from the
  backtested classifier (via a precomputed probability grid), and the Poisson
  models then supply a scoreline consistent with that outcome (needed for
  group tiebreakers). Drawn knockout games go to extra time at ⅓ goal rate,
  then a penalty shootout that tilts slightly with the Elo gap.
- **Keeping ratings current**: append real results (e.g. 2026 fixtures as
  they are played) to `data/extra_matches.csv` and re-run
  `python -m wc_predictor.train`. `python -m wc_predictor.ratings` prints the
  resulting Elo table.
- **Round of 32**: a fixed seeded template pairing winners with best thirds
  and runners-up with each other — a documented simplification of FIFA's
  official third-place allocation tables.

## Project layout

```
data/                  bundled matches, 2026 teams file, example bracket
scripts/               download_full_data.py
src/wc_predictor/      data, elo, features, train, predict, simulate
tests/                 20 tests (Elo math, leakage, training, simulation)
.github/workflows/     CI: pytest + train/simulate smoke tests
```

## Limitations

- Trained on 552 matches out of the box — upgrade to the full dataset for
  serious use.
- The 2026 group draw and playoff qualifiers in `wc2026_teams.csv` are
  placeholders to edit.
- No player-level information (injuries, squad strength, age curves), no
  bookmaker odds; this is a team-rating model.
