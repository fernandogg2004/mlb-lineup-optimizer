# MLB Lineup Optimizer — plate-appearance model, Monte Carlo game engine and batting-order search

Predicts the outcome distribution of every plate appearance (PA), simulates games with a
base-out Markov chain and searches batting orders with a genetic algorithm to maximise expected
runs — with an honest, bootstrap-based evaluation of what the system can and cannot do.

## The problem

A manager chooses one of 9! ≈ 363,000 batting orders every day. Sabermetric research suggests the
choice is worth only a handful of runs per season, which makes it a good test of a modelling
stack: the PA model must be **calibrated** (its probabilities feed a simulator), the simulator
must be **unbiased** in runs, and the evaluation must separate genuine value from Monte Carlo
noise. A secondary question is whether the same stack predicts game winners — a much harder,
almost coin-flip problem in MLB.

## At a glance

| | |
|---|---|
| **Approach** | 8-class LightGBM PA model (out, K, BB/HBP, 1B, 2B, 3B, HR, DP) with isotonic calibration → 24-state base-out Monte Carlo engine with probabilistic runner-advance tables → 3-stage genetic algorithm over batting orders |
| **Data** | Public Statcast data via `pybaseball` and the MLB Stats API, aggregated to one row per PA (2015–2026 seasons) |
| **Evaluation** | Strict temporal split (train ≤ 2024, calibrate on 2025, evaluate on 2026 PAs); game-level backtest with 90% bootstrap intervals; counterfactual lineup backtest |
| **Main results** | PA model is **well calibrated but weakly discriminative** (ECE 0.002, weighted one-vs-rest AUC 0.57). Game winners: **no measurable edge** over a 50/50 baseline. Batting order: optimised orders score **+0.05 runs/game over posted orders in the model's own simulator** (reproduced), shrinking to about +0.02 on realistic lineups |
| **Status** | Functional prototype with CLI, FastAPI + React dashboard and Airflow/Docker automation. Not used in production |

## Results

Sources: committed artifacts in [`reports/diagnostics/`](reports/diagnostics/) and
[`reports/backtest/`](reports/backtest/) (generated 2026-06-24). The lineup backtest was
**re-run independently** for this README; the other figures are read from the committed JSON.

### 1. Plate-appearance model — out-of-sample 2026 season (82,574 PAs)

| Metric | Result | Reference |
|---|---|---|
| Expected calibration error (overall) | **0.0021** | target ≤ 0.035 |
| Bias in expected run value per PA | −0.0006 runs | predicted 0.1682 vs observed 0.1688 |
| Multiclass log-loss | 1.520 | 1.538 for league base rates (computed for this README from the observed class frequencies) |
| Weighted one-vs-rest AUC | **0.571** | per class 0.53 (2B) – 0.61 (HR) |

Calibration is the design goal and it is met. Discrimination is close to the base rates; a
controlled ceiling test (`reports/diagnostics/ceiling_test.json`) showed that adding player
identity does not help and that most of the remaining signal is in the PA's own contact quality,
which is not knowable before the pitch.

Source: [`reports/diagnostics/pipeline_diagnosis.json`](reports/diagnostics/pipeline_diagnosis.json).

### 2. Game winner — 104 games with valid two-sided simulations (2 – 23 June 2026)

| Metric | Model | Baseline |
|---|---|---|
| Log-loss | 0.699, 90% CI [0.665, 0.732] | 0.693 (50/50) · 0.699 (Elo) |
| AUC | 0.538, 90% CI [0.446, 0.631] | 0.5 |
| P(model better than 50/50), bootstrap | 0.41 | — |

**No measurable edge**; the 50/50 baseline lies inside every interval. 115 further games were
excluded because one side had not been simulated (a data-hygiene bug, since fixed in serving).
Source: [`reports/backtest/backtest_clean.json`](reports/backtest/backtest_clean.json).

### 3. Batting-order optimisation — counterfactual, 40 team-games (28 May and 2 June 2026)

For each saved lineup, the expected runs of the order actually posted, the optimiser's order and
30 random permutations are evaluated with the same simulation engine (60k games each).

| Metric | Committed run | Independent re-run (2026-10-26) |
|---|---|---|
| Uplift, optimal vs posted order | +0.052 runs/game, 90% CI [+0.033, +0.072] | **+0.049 runs/game, 90% CI [+0.032, +0.067]** |
| Uplift, optimal vs random order | +0.103 runs/game | +0.096 runs/game |
| Posted order, percentile vs random orders | 81st | 77th |
| Games where the winning order beats the runner-up beyond MC noise | 5% | 0% |

Source: [`reports/diagnostics/lineup_backtest.json`](reports/diagnostics/lineup_backtest.json),
`scripts/backtest_lineups.py`. The re-run used the same command, the same 40 saved lineups and
Python 3.11 with `requirements_local.txt` (17 min on CPU). The Monte Carlo engine is not seeded, so
small differences between the two runs are expected.

**How to read this.** The uplift is measured *inside the model's own simulator*, not on real game
outcomes — a 0.05 runs/game effect is far too small to observe in a few dozen real games. It is
therefore evidence that the optimiser finds better orders **according to the model**, not
evidence of real runs gained. In addition:

- The sample is small (40 team-games from two dates) and the 90% interval is a normal
  approximation over those games.
- Most of the uplift comes from the 14 lineups whose simulated expected runs are implausibly high
  (5.5–8.6 runs/game, against a league average near 4.4), all from 2 June. On the 26 lineups with
  realistic run expectations the mean uplift is **+0.021** (committed run) and **+0.024**
  (re-run) runs/game, about 3–4 runs per 162 games. Saved predictions from
  2–6 June contain PA probability vectors with P(out) as low as 0.07, which points to an upstream
  feature or serving issue on those dates.
- Extrapolating to "+8.5 runs per season" assumes the per-game uplift transfers to real games and
  persists across 162 games; neither is established.

## How it works

```
Statcast (pybaseball) + MLB Stats API
   │ build_silver.py            one row per plate appearance, partitioned by season
   ▼
data/silver/ ──► scripts/build_gold_v3.py   51 features; rolling stats use shift(1) (no same-PA info)
   ▼
train_v3.py      LightGBM (8 classes) + isotonic calibration; temporal split; deployment gate
   ▼
models/at_bat_predictor.pkl
   ├─ predict_tonight.py   today's lineups → 9×8 probability matrix per team
   ├─ src/simulation/      base-out Markov chain Monte Carlo → E[runs], P(win), percentiles
   └─ src/optimizer/       GA: sabermetric seeding → evolution → high-precision refinement of top-k
```

Design decisions that matter:

- **Calibration over rebalancing.** No class weights; probabilities are used as simulator inputs.
- **Single feature contract.** `src/features/shared_features.py` is the only feature definition
  for both training and serving; `tests/test_feature_parity.py` guards against train/serve skew.
- **Leakage exclusions.** Contact-quality fields known only after the PA ends (`xwoba`, launch
  speed/angle, pitch count) are excluded from model inputs.
- **Deployment gate.** `train_v3.py` refuses to promote a model whose per-class drift or
  run-value bias exceeds a threshold.
- **Simulator check.** With a league-average lineup and neutral park, the 9-inning engine produces
  4.381 runs/game vs 4.373 observed (`reports/diagnostics/FASE2_FINDINGS.md`).

## Data and provenance

- **Statcast** pitch-level data via [`pybaseball`](https://github.com/jldbc/pybaseball), aggregated
  to plate appearances; **MLB Stats API** (`statsapi.mlb.com`) for schedules, lineups and rosters.
- Processed Silver/Gold parquet files and model pickles are stored with **Git LFS** in this
  repository (~360 MB in total). The data originate from MLB Advanced Media; see
  [`docs/data-rights.md`](docs/data-rights.md) — non-commercial use only without a licence.
- `results/<date>/` contains the model's saved daily predictions (May–June 2026), which the
  backtests consume.

## Limitations

- **Lineup value is a simulated quantity** (see Results §3); it has not been validated against
  real runs.
- **Game-level prediction ≈ coin flip**, consistent with how hard pregame MLB prediction is.
- **PA model discrimination is saturated** with the current features (screened additions changed
  AUC by < 0.002); more signal would need new data such as pitch "stuff" or historical weather.
- **Some saved predictions look inflated** (E[runs] up to 8.6 per game on 2–6 June 2026); the
  cause has not been diagnosed.
- **Monte Carlo runs are not seeded**, so re-running the backtests reproduces results only up to
  simulation noise.
- **Bullpen modelling in the optimiser is partial**; the GA uses the opposing starter's profile.

## Reproduction

Python **3.11** is required for the pinned stack (`ray==2.20.0` has no wheels for 3.12+);
Node.js 18+ for the front end; [Git LFS](https://git-lfs.com) for data and models.

```bash
git lfs install && git clone https://github.com/fernandogg2004/mlb-lineup-optimizer.git
cd mlb-lineup-optimizer
python3.11 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements_local.txt "setuptools<70"    # mlflow 2.13 imports pkg_resources
pytest
```

Checked on 2026-10-26 (Python 3.11, Linux): **207 tests pass, 5 skipped**; two test modules
(`tests/mlops/test_mlb_stats_monitor.py`, `tests/rag/test_lambda_scouting_trigger.py`) fail to
import because `redis` and `openai` are not listed in `requirements_local.txt`.

Reproduce the lineup backtest (uses the saved predictions in `results/`, no network, ~17 min on CPU):

```bash
python scripts/backtest_lineups.py --max-games 40 --n-random 30 --out /tmp/lineup_backtest.json
```

Other evaluation entry points (need the LFS data): `python backtest.py`,
`python scripts/diagnose_pipeline.py`, `python scripts/ceiling_test.py`,
`python scripts/feature_screen.py`.

Daily use and serving:

```bash
python predict_tonight.py --team NYY --optimize     # today's game: probabilities, simulation, best order
python -m api.main                                   # FastAPI backend on :8000
cd frontend && npm install && npm run dev            # React dashboard
```

Retraining (rebuilds data from Statcast; network and time intensive):

```bash
python build_silver.py --years 2021 2022 2023 2024 2025 2026
python scripts/build_gold_v3.py
python train_v3.py
```

`ANTHROPIC_API_KEY` is needed only for the optional LLM explanation features (`src/rag/`).
Docker/Airflow automation is described in [`SETUP.md`](SETUP.md) and `docker-compose.airflow.yml`.

## Repository structure

```
build_silver.py  train_v3.py  predict_tonight.py  backtest.py  morning.py   main entry points
scripts/          gold features, diagnostics, ceiling test, feature screen, lineup backtest
src/features/     shared feature contract, rolling/platoon/pitcher features
src/models/       AtBatPredictor (LightGBM + isotonic + ECE)
src/simulation/   Monte Carlo engine, runner-advance tables, bullpen model
src/optimizer/    genetic lineup optimiser
src/rag/          optional LLM explanations (Claude)
api/  app/  frontend/   FastAPI services and React dashboard
reports/          diagnostics, backtests, drift reports (evidence for the numbers above)
results/          saved daily predictions
tests/            pytest suite
```

## Technologies

Python, LightGBM, scikit-learn (isotonic calibration), NumPy/Numba, Polars, DEAP (genetic
algorithm), MLflow, FastAPI, React/TypeScript, Docker, Apache Airflow, pytest, Git LFS.

## Author

**Fernando González González** — BSc Mathematics, Universidade de Santiago de Compostela.
[LinkedIn](https://www.linkedin.com/in/fernando-gonzalez-gonzalez-math/) ·
[GitHub](https://github.com/fernandogg2004)
