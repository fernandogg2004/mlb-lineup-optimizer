"""
test_simulation_engine.py — Comportamiento del motor Monte Carlo.

Audit B2: regresión para las invariantes del simulador y para los arreglos
D1 (DP sin corredor en 1B = 1 out) y B4 (kernel solo-ofensiva == juego completo).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.constants import LEAGUE_AVG_PA_PROBS
from src.simulation.simulation_engine import (
    MonteCarloConfig,
    MonteCarloEngine,
    _NEW_BASES_TBL_P,
    _RUNS_TBL_P,
    _SCENARIO_PROBS,
    _simulate_half_inning_prob,
    _simulate_n_games_prob,
    _simulate_n_runs_only,
    compute_run_expectancy_matrix,
)

N_OUT = 8


def _onehot(idx: int, n: int = 9) -> np.ndarray:
    m = np.zeros((n, N_OUT), dtype=np.float32)
    m[:, idx] = 1.0
    return m


def _league() -> np.ndarray:
    return np.tile(LEAGUE_AVG_PA_PROBS, (9, 1)).astype(np.float32)


@pytest.fixture(scope="module")
def engine() -> MonteCarloEngine:
    return MonteCarloEngine(MonteCarloConfig(
        use_ray=False, n_simulations=20_000, fast_mode_n_sims=10_000,
        use_extra_innings=True,
    ))


def test_win_prob_and_opp_sum_to_one(engine):
    league = _league()
    res = engine.run(league, league)
    assert abs(res.win_probability + res.opponent_win_probability - 1.0) < 1e-9


def test_symmetric_lineups_are_a_cointoss(engine):
    league = _league()
    res = engine.run(league, league)
    assert 0.45 < res.win_probability < 0.55


def test_all_outs_score_zero(engine):
    res = engine.run(_onehot(0), _league())
    assert res.expected_runs_scored == 0.0
    assert res.percentile_10 == 0.0
    assert res.percentile_90 == 0.0


def test_better_lineup_scores_more(engine):
    league = _league()
    elite = np.array([0.30, 0.18, 0.12, 0.18, 0.08, 0.01, 0.11, 0.02], dtype=np.float32)
    elite = elite / elite.sum()
    er_league = engine.run_fast(league, league)
    er_elite = engine.run_fast(np.tile(elite, (9, 1)).astype(np.float32), league)
    assert er_elite > er_league + 0.5


def test_dp_empty_bases_is_a_single_out():
    # audit D1: un outcome DP con bases vacías no puede ser doble matanza.
    dp = _onehot(7)
    runs, pos = _simulate_half_inning_prob(
        dp, 0, _NEW_BASES_TBL_P, _RUNS_TBL_P, _SCENARIO_PROBS, 1.0, 1.0, 0,
    )
    assert runs == 0
    assert pos == 3   # 3 bateadores, 1 out cada uno


def test_dp_with_runner_on_first_is_a_real_double_play():
    # audit D1: con corredor en 1B (initial_bases=1) el primer DP elimina 2.
    dp = _onehot(7)
    runs, pos = _simulate_half_inning_prob(
        dp, 0, _NEW_BASES_TBL_P, _RUNS_TBL_P, _SCENARIO_PROBS, 1.0, 1.0, 1,
    )
    assert runs == 0
    assert pos == 2   # bateador1 = GIDP (2 outs), bateador2 = 1 out


def test_runs_only_kernel_matches_full_game():
    # audit B4: el kernel solo-ofensiva debe dar el mismo E[R] que el juego
    # completo (que además simulaba al rival y lo descartaba).
    league = _league()
    n = 150_000
    ro = _simulate_n_runs_only(
        league, _NEW_BASES_TBL_P, _RUNS_TBL_P, _SCENARIO_PROBS, 1.0, 1.0, n, 9,
    )
    mr, _, _ = _simulate_n_games_prob(
        league, league, _NEW_BASES_TBL_P, _RUNS_TBL_P, _SCENARIO_PROBS,
        1.0, 1.0, n, 9, 0,
    )
    assert abs(float(ro.mean()) - float(mr.mean())) < 0.05


def test_run_expectancy_matrix_is_monotonic():
    re = compute_run_expectancy_matrix(_league(), n_simulations=8_000)
    # más corredores con los mismos outs => más carreras esperadas
    assert re[0, 0b111] > re[0, 0b001] > re[0, 0]
    # menos outs => más carreras esperadas (bases vacías)
    assert re[0, 0] > re[1, 0] > re[2, 0]


def test_park_factor_increases_scoring(engine):
    league = _league()
    er_neutral = engine.run_fast(league, league, park_factor_hr=1.0, park_factor_xb=1.0)
    er_coors   = engine.run_fast(league, league, park_factor_hr=1.30, park_factor_xb=1.10)
    assert er_coors > er_neutral
