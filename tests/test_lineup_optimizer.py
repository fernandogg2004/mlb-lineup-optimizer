"""
test_lineup_optimizer.py — Operadores del GA, seeder y end-to-end.

Audit B2: los operadores de permutación (OX, swap), el seeder sabermétrico y el
flujo completo del optimizador no tenían regresión. Un operador que produzca una
permutación inválida (un bateador repetido) corrompe silenciosamente el orden.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.constants import LEAGUE_AVG_PA_PROBS
from src.optimizer.lineup_optimizer import (
    GAConfig,
    GeneticLineupOptimizer,
    PlayerStats,
    SabermetricSeeder,
    cxOrdered,
    mutSwapPositions,
)


def _valid_perm(p) -> bool:
    return sorted(p) == list(range(9))


def _players(obps: list[float]) -> list[PlayerStats]:
    return [
        PlayerStats(player_id=i, player_name=f"P{i}",
                    obp=o, woba=o - 0.05, iso=0.15, batter_stand="R")
        for i, o in enumerate(obps)
    ]


def test_cx_ordered_always_valid_permutations():
    rng = random.Random(0)
    for _ in range(5_000):
        a = rng.sample(range(9), 9)
        b = rng.sample(range(9), 9)
        c1, c2 = cxOrdered(list(a), list(b))
        assert _valid_perm(c1), c1
        assert _valid_perm(c2), c2


def test_mut_swap_always_valid_permutation():
    rng = random.Random(1)
    for _ in range(5_000):
        ind = rng.sample(range(9), 9)
        (out,) = mutSwapPositions(list(ind), indpb=0.5)
        assert _valid_perm(out), out


def test_seeder_returns_valid_permutation_with_high_obp_leadoff():
    obps = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.31, 0.33, 0.29]
    seed = SabermetricSeeder(_players(obps)).canonical_seed("R")
    assert _valid_perm(seed)
    assert obps[seed[0]] == max(obps)   # el leadoff es el de mayor OBP


def test_seeder_rejects_wrong_player_count():
    with pytest.raises(ValueError):
        SabermetricSeeder(_players([0.3] * 8))   # solo 8 jugadores


def test_diverse_seeds_are_all_valid_permutations():
    obps = [0.30 + 0.01 * i for i in range(9)]
    seeds = SabermetricSeeder(_players(obps)).diverse_seeds(40, "R")
    assert len(seeds) == 40
    assert all(_valid_perm(s) for s in seeds)


@pytest.mark.slow
def test_optimizer_end_to_end_returns_valid_lineup(tmp_path, monkeypatch):
    # Aísla MLflow a un directorio temporal (no contamina ./mlruns).
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    from src.simulation.simulation_engine import MonteCarloConfig, MonteCarloEngine

    obps = [0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.31, 0.33, 0.29]
    players = _players(obps)
    probs = np.tile(LEAGUE_AVG_PA_PROBS, (9, 1)).astype(np.float32)
    opp = probs.copy()
    eng = MonteCarloEngine(MonteCarloConfig(
        use_ray=False, n_simulations=2_000, fast_mode_n_sims=800,
        use_extra_innings=False,
    ))
    cfg = GAConfig(n_generations=5, population_size=20,
                   n_sabermetric_seeds=8, refinement_top_k=4)
    opt = GeneticLineupOptimizer(players, probs, eng, opp, config=cfg)
    res = opt.run(pitcher_hand="R")

    assert _valid_perm(res.best_lineup_indices)
    assert len(res.best_lineup_ids) == 9
    assert np.isfinite(res.best_expected_runs)
    assert 0.0 <= res.best_win_probability <= 1.0


@pytest.mark.slow
def test_optimizer_winprob_objective_runs(tmp_path, monkeypatch):
    # audit A2: el objetivo "win_probability" debe ejecutar y devolver un orden válido.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    from src.simulation.simulation_engine import MonteCarloConfig, MonteCarloEngine

    players = _players([0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.31, 0.33, 0.29])
    probs = np.tile(LEAGUE_AVG_PA_PROBS, (9, 1)).astype(np.float32)
    eng = MonteCarloEngine(MonteCarloConfig(
        use_ray=False, n_simulations=2_000, fast_mode_n_sims=800,
        use_extra_innings=False,
    ))
    cfg = GAConfig(n_generations=4, population_size=16, n_sabermetric_seeds=6,
                   refinement_top_k=4, objective="win_probability")
    res = GeneticLineupOptimizer(players, probs, eng, probs.copy(), config=cfg).run()
    assert _valid_perm(res.best_lineup_indices)
