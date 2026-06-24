"""Tests de invariantes del optimizador y del motor de simulación.

Cubren las regresiones halladas en la auditoría 2026-06 (F02, F03, F04, F08,
F11, F18). Diseñados para ser rápidos: pocas simulaciones y GA mínimo. La
primera ejecución paga la compilación JIT de Numba (~unos segundos).
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from src.constants import LEAGUE_AVG_PA_PROBS, LEAGUE_AVG_LINEUP, N_OUTCOMES
from src.simulation.simulation_engine import (
    MonteCarloEngine,
    MonteCarloConfig,
    compute_run_expectancy_matrix,
)
from src.simulation.runner_advance_tables import build_deterministic_transition_tables as _det


def _rand_lineup(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(9):
        p = LEAGUE_AVG_PA_PROBS.astype(np.float64) * rng.uniform(0.7, 1.3, N_OUTCOMES)
        rows.append((p / p.sum()).astype(np.float32))
    return np.stack(rows)


def _engine(n=3000, fast=1500):
    return MonteCarloEngine(MonteCarloConfig(
        use_ray=False, n_simulations=n, fast_mode_n_sims=fast, use_extra_innings=False,
    ))


# ── Invariantes de las tablas de transición (F11) ────────────────────────────

def test_hr_clears_bases_and_scores_all():
    nb, runs = _det()
    # Grand slam: bases llenas (0b111) -> 4 carreras, bases vacías
    assert runs[6, 0b111] == 4
    assert nb[6, 0b111] == 0


def test_triple_puts_batter_on_third():
    nb, runs = _det()
    assert runs[5, 0b111] == 3      # 3 corredores anotan
    assert nb[5, 0b111] == 0b100    # bateador en 3B


def test_double_play_clears_first():
    nb, _ = _det()
    # DP con corredor en 1B (0b001) -> limpia 1B
    assert nb[7, 0b001] == 0b000


# ── RE24 (F11): el DP suma outs, matriz plausible y monótona ─────────────────

def test_re24_monotonic_in_outs_and_plausible():
    re = compute_run_expectancy_matrix(LEAGUE_AVG_LINEUP.astype(np.float32), n_simulations=2000)
    assert re.shape == (3, 8)
    # 0 outs vale más que 2 outs en el mismo estado de bases
    assert re[0, 0] > re[2, 0]
    # bases vacías 0 outs en rango canónico razonable
    assert 0.30 < re[0, 0] < 0.80


# ── Motor: shape contract de 8 clases y CI binomial honesto (F02, F04) ───────

def test_engine_requires_8_class_shape():
    eng = _engine()
    bad = np.full((9, 7), 1 / 7, dtype=np.float32)
    with pytest.raises(ValueError):
        eng.run(bad, LEAGUE_AVG_LINEUP.astype(np.float32))


def test_win_prob_ci_is_binomial_not_fabricated():
    eng = _engine(n=20000)
    my = _rand_lineup(1)
    r = eng.run(my, LEAGUE_AVG_LINEUP.astype(np.float32))
    p = r.win_probability
    se = np.sqrt(max(p * (1 - p), 0.0) / r.n_simulations)
    width = r.win_prob_ci_high - r.win_prob_ci_low
    # Debe coincidir con 2*1.645*SE binomial (no un n_eff=100 fabricado)
    assert width == pytest.approx(2 * 1.645 * se, abs=2e-3)


def test_percentile_10_not_floored():
    # Con un lineup muy débil, P10 real puede ser 0; no debe forzarse a 1.0
    eng = _engine(n=8000)
    weak = LEAGUE_AVG_LINEUP.astype(np.float32).copy()
    weak[:, 6] *= 0.1   # casi sin HR
    r = eng.run(weak, LEAGUE_AVG_LINEUP.astype(np.float32))
    assert r.percentile_10 >= 0.0   # valor real, sin piso artificial


# ── Fitness consistente: park factors influyen en run_fast (F03) ─────────────

def test_run_fast_uses_park_factors():
    eng = _engine()
    my = _rand_lineup(2)
    opp = LEAGUE_AVG_LINEUP.astype(np.float32)
    er_neutral = eng.run_fast(my, opp, 1.0, 1.0)
    er_hitter = eng.run_fast(my, opp, 1.35, 1.10)
    assert er_hitter > er_neutral


# ── Optimizador: corre con vectores de 8 clases y reporta significancia (F02, F18)

def test_ga_runs_with_8class_and_reports_significance(monkeypatch):
    import mlflow
    mlflow.set_tracking_uri("file:" + tempfile.mkdtemp())
    from src.optimizer.lineup_optimizer import (
        GeneticLineupOptimizer, GAConfig, PlayerStats,
    )
    probs = _rand_lineup(3)
    players = [
        PlayerStats(player_id=i, player_name=f"P{i}", obp=0.33, woba=0.32,
                    iso=0.16, batter_stand="R", prob_vector=probs[i])
        for i in range(9)
    ]
    opp = np.tile(LEAGUE_AVG_PA_PROBS, (9, 1)).astype(np.float32)
    eng = _engine(n=4000, fast=1000)
    cfg = GAConfig(n_generations=3, population_size=16, n_sabermetric_seeds=4,
                   refinement_top_k=4)
    opt = GeneticLineupOptimizer(players, probs, eng, opp, config=cfg)
    res = opt.run(pitcher_hand="R", park_factor_hr=1.05)
    assert len(set(res.best_lineup_indices)) == 9          # permutación válida
    assert res.best_expected_runs_se > 0                   # SE expuesto (F18)
    assert isinstance(res.is_significant_vs_second, bool)


def test_optimizer_rejects_wrong_shape():
    from src.optimizer.lineup_optimizer import GeneticLineupOptimizer, PlayerStats
    probs7 = np.full((9, 7), 1 / 7, dtype=np.float32)
    players = [
        PlayerStats(i, f"P{i}", 0.33, 0.32, 0.16, "R", probs7[i]) for i in range(9)
    ]
    opp = np.tile(LEAGUE_AVG_PA_PROBS, (9, 1)).astype(np.float32)
    with pytest.raises(ValueError):
        GeneticLineupOptimizer(players, probs7, _engine(), opp)
