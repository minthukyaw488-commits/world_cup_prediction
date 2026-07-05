# World Cup Prediction

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
```

## Results (bundled data)

Expanding-window backtest — for each World Cup year the model is trained only
on earlier tournaments and scored on that year:

| Model                         | Mean log loss | Mean accuracy |
|-------------------------------|---------------|---------------|
| Logistic regression (selected)| **1.041**     | **52.9%**     |
| Gradient boosting             | 1.123         | 45.2%         |
| Class-frequency baseline      | 1.079         | 40.7%         |

Test years: 2010, 2014, 2018, 2022. The model beats the baseline in three of
the four years; 2022 (Saudi Arabia over Argentina, Japan over Germany and
Spain, Morocco to the semis) is the exception — no rating-based model looked
good in Qatar.

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

2. **`data/wc_matches.csv`** *(bundled fallback)* — all 448 World Cup finals
   matches 1998–2022, so everything works offline out of the box.

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
- **Knockout matches** in the simulator: 90-minute score sampled from the
  Poisson models; if level, extra time at ⅓ goal rate; if still level, a
  penalty shootout that tilts slightly with the Elo gap.
- **Round of 32**: a fixed seeded template pairing winners with best thirds
  and runners-up with each other — a documented simplification of FIFA's
  official third-place allocation tables.

## Project layout

```
data/                  bundled matches, 2026 teams file
scripts/               download_full_data.py
src/wc_predictor/      data, elo, features, train, predict, simulate
tests/                 17 tests (Elo math, leakage, training, simulation)
```

## Limitations

- Trained on 448 matches out of the box — upgrade to the full dataset for
  serious use.
- The 2026 group draw and playoff qualifiers in `wc2026_teams.csv` are
  placeholders to edit.
- No player-level information (injuries, squad strength, age curves), no
  bookmaker odds; this is a team-rating model.
