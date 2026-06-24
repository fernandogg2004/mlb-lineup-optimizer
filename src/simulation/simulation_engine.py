"""
simulation_engine.py
====================
Monte Carlo baseball simulator implementing the game as a Markov chain.

Mathematical framework
----------------------
The state space of a half-inning is fully described by 24 states:

    state = (outs ∈ {0,1,2}) × (bases ∈ {0…7})   → 3 × 8 = 24 states

where ``bases`` is a 3-bit integer:
    bit 0 (value 1) = runner on 1st
    bit 1 (value 2) = runner on 2nd
    bit 2 (value 4) = runner on 3rd

The terminal/absorbing state is ``outs == 3`` (end of half-inning).

For each PA, the model supplies a probability vector p ∈ ℝ⁷ over outcomes:
    [OUT_IN_PLAY, STRIKEOUT, WALK_HBP, SINGLE, DOUBLE, TRIPLE, HOME_RUN]

The transition function T(state, outcome) → (new_state, runs_scored) is
deterministic given the outcome — only the PA *result* is stochastic.

Two-layer parallelism strategy
-------------------------------
Layer 1 — Numba JIT + prange (CPU SIMD, single machine):
    Simulates batches of games on one machine via multi-threaded Numba.
    Achieves ~0.5–2 μs per game on modern CPUs.
    100,000 games ≈ 50–200 ms on a 16-core machine.

Layer 2 — Ray remote actors (multi-machine cluster):
    Splits the 100,000 game budget across N Ray workers (default 20),
    each running a Numba batch of 5,000 games in parallel. This scales
    horizontally to the cluster size.

Warmup protocol
---------------
The first call to any @njit function triggers LLVM compilation (~2 seconds).
Call ``warmup_jit()`` during application startup to pay this cost upfront,
not during the time-critical game-day pipeline.

Usage:
    from src.simulation.simulation_engine import MonteCarloEngine
    engine = MonteCarloEngine(predictor, n_simulations=100_000)
    result = engine.run(my_lineup_probs, opp_lineup_probs, park_factor=1.05)
    print(f"E[R] = {result.expected_runs_scored:.3f}")
    print(f"P(W) = {result.win_probability:.3f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog
from numba import njit, prange

# Ray se importa de forma PEREZOSA dentro de MonteCarloEngine (audit F16): es una
# dependencia pesada y, con use_probabilistic_advances=True (default), la rama Ray
# ni siquiera se ejecuta. Importarla al cargar el módulo encarecía cada arranque
# (predict_tonight, api) que corre con use_ray=False.

from src.simulation.runner_advance_tables import (
    RunnerAdvanceConfig,
    build_probabilistic_transition_tables,
)
from src.simulation.bullpen_model import (
    PitchingStaffConfig,
    build_per_inning_prob_matrix,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Outcome indices (must match PAOutcome enum in model_at_bat.py)
# ---------------------------------------------------------------------------
_OUT       = np.int32(0)
_STRIKEOUT = np.int32(1)
_WALK_HBP  = np.int32(2)
_SINGLE    = np.int32(3)
_DOUBLE    = np.int32(4)
_TRIPLE    = np.int32(5)
_HR        = np.int32(6)
_DP        = np.int32(7)   # DOUBLE_PLAY — activated in Mejora 6 alongside N_OUTCOMES=8

# N_OUTCOMES = 8 after Mejora 6 (coordinated deploy with model that produces 8 classes).
# The probabilistic tables already have 8 rows.
N_OUTCOMES: int = 8
N_LINEUP:   int = 9


# ---------------------------------------------------------------------------
# Precomputed transition lookup tables (module-level constants)
# ---------------------------------------------------------------------------
# NEW_BASES_TABLE[outcome, bases] → new bases state after the play
# RUNS_TABLE[outcome, bases]      → runs scored on the play
#
# These are computed once at import time and used as compile-time constants
# by Numba's JIT compiler, producing O(1) state transitions.

def _build_transition_tables() -> tuple[np.ndarray, np.ndarray]:
    """Precomputes the full (8, 8) transition lookup tables.

    Baseball runner advancement rules encoded:
        HR          : all runners + batter score; bases cleared.
        Triple      : all runners score; batter on 3rd.
        Double      : runners on 2nd and 3rd score; runner on 1st to 3rd; batter to 2nd.
        Single      : runner on 3rd scores; runner on 2nd to 3rd; runner on 1st to 2nd; batter to 1st.
        Walk/HBP    : force advance — batter to 1st; only forced runners advance.
        Out/K       : bases unchanged; outs + 1.
        Double_Play : 2 outs; runner on 1st eliminated (GIDP or strikeout-DP);
                      no runs scored; other runners hold.

    These are the DETERMINISTIC fallback tables used by the legacy GA fast path
    and for regression test comparisons.  The production path uses the 3-D
    probabilistic tables from ``runner_advance_tables.py``.

    Returns:
        Tuple (new_bases_table, runs_table), each shape (8, 8) int32.
    """
    new_bases = np.zeros((8, 8), dtype=np.int32)
    runs      = np.zeros((8, 8), dtype=np.int32)

    # Precomputed BB/HBP force-advance lookup (manually verified):
    _bb_new  = np.array([1, 3, 3, 7, 5, 7, 7, 7], dtype=np.int32)
    _bb_runs = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32)

    for b in range(8):
        b1 = (b >> 0) & 1  # runner on 1st
        b2 = (b >> 1) & 1  # runner on 2nd
        b3 = (b >> 2) & 1  # runner on 3rd

        # --- Out (0) and Strikeout (1) ---
        new_bases[0, b] = b
        runs[0, b]      = 0
        new_bases[1, b] = b
        runs[1, b]      = 0

        # --- Walk / HBP (2) ---
        new_bases[2, b] = _bb_new[b]
        runs[2, b]      = _bb_runs[b]

        # --- Single (3): 3B scores; 2B→3B; 1B→2B; batter→1B ---
        new_bases[3, b] = 1 | (b1 << 1) | (b2 << 2)
        runs[3, b]      = b3

        # --- Double (4): 3B+2B score; 1B→3B; batter→2B ---
        new_bases[4, b] = 2 | (b1 << 2)
        runs[4, b]      = b2 + b3

        # --- Triple (5): all score; batter→3B ---
        new_bases[5, b] = 4
        runs[5, b]      = b1 + b2 + b3

        # --- Home Run (6): all score including batter ---
        new_bases[6, b] = 0
        runs[6, b]      = 1 + b1 + b2 + b3

        # --- Double Play (7): 2 outs; runner on 1B eliminated; no runs ---
        # Clear bit 0 (runner on 1B); keep runners on 2B and 3B.
        new_bases[7, b] = b & 0b110
        runs[7, b]      = 0

    return new_bases, runs


# Module-level constants — Numba sees these as globals
_NEW_BASES_TBL, _RUNS_TBL = _build_transition_tables()

# ---------------------------------------------------------------------------
# Probabilistic 3-D tables (Mejora 7)
# ---------------------------------------------------------------------------
# Shape: (N_OUTCOMES=8, N_BASE_STATES=8, N_SCENARIOS=4)
# Built once at import time per era; default era is "2023+".
# The engine rebuilds these if MonteCarloConfig.runner_advance_era differs.
_DEFAULT_PROB_ERA = "2023+"
(
    _NEW_BASES_TBL_P,  # int32  (8, 8, 4) — new base state per scenario
    _RUNS_TBL_P,       # int32  (8, 8, 4) — runs scored per scenario
    _SCENARIO_PROBS,   # float32 (8, 8, 4) — probability of each scenario
) = build_probabilistic_transition_tables(RunnerAdvanceConfig(era=_DEFAULT_PROB_ERA))


# ---------------------------------------------------------------------------
# Numba JIT core functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sample_outcome(probs: np.ndarray) -> int:
    """Draws one outcome index by inverse-CDF on a probability vector.

    This implementation avoids allocating a cumulative-sum array, which
    is critical for Numba JIT performance in the tight inner loop
    (called ~40 times per simulated game, 100,000+ games per optimizer call).

    Args:
        probs: Probability array of shape (N_OUTCOMES,). Must sum to ~1.0.

    Returns:
        Sampled outcome index in [0, N_OUTCOMES - 1].
    """
    u = np.random.random()
    cumsum = 0.0
    for i in range(len(probs)):
        cumsum += probs[i]
        if u < cumsum:
            return i
    return len(probs) - 1  # floating-point guard


@njit(cache=True)
def _apply_park_factor(
    probs: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
) -> np.ndarray:
    """Adjusts outcome probabilities by park factor and renormalizes.

    Park factor > 1.0 increases the rate of home runs (and extra-base hits)
    at hitter-friendly parks (e.g., Coors Field HR factor ≈ 1.35).
    The adjustment multiplies the target outcome probabilities and then
    renormalizes the full vector to maintain sum = 1.0.

    Args:
        probs: Original probability vector, shape (7,).
        park_factor_hr: HR probability multiplier (e.g., 1.35 for Coors).
        park_factor_xb: Double/Triple probability multiplier.

    Returns:
        Park-adjusted and renormalized probability vector, shape (7,).
    """
    adjusted = probs.copy()
    adjusted[6] *= park_factor_hr   # HR
    adjusted[5] *= park_factor_xb   # Triple
    adjusted[4] *= park_factor_xb   # Double
    total = 0.0
    for v in adjusted:
        total += v
    if total > 0.0:
        for i in range(len(adjusted)):
            adjusted[i] /= total
    return adjusted


@njit(cache=True)
def _sample_scenario(scenario_probs: np.ndarray) -> int:
    """Draws one scenario index from a probability row using inverse-CDF.

    Identical logic to ``_sample_outcome`` but operates on the 1-D
    scenario-probability slice ``scenario_probs_tbl[outcome, bases, :]``.

    Args:
        scenario_probs: Probability array of shape (N_SCENARIOS,).
            Slots beyond the last non-zero entry have prob 0 and are
            never reached by the CDF walk.

    Returns:
        Sampled scenario index in [0, N_SCENARIOS - 1].
    """
    u = np.random.random()
    cumsum = 0.0
    for i in range(len(scenario_probs)):
        cumsum += scenario_probs[i]
        if u < cumsum:
            return i
    return len(scenario_probs) - 1  # floating-point guard


@njit(cache=True)
def _simulate_half_inning_prob(
    lineup_probs: np.ndarray,
    batter_pos: int,
    new_bases_tbl_p: np.ndarray,
    runs_tbl_p: np.ndarray,
    scenario_probs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    initial_bases: int,
) -> tuple[int, int]:
    """Simulates one half-inning with probabilistic runner advancement.

    Replaces the deterministic ``_simulate_half_inning`` in production mode.
    For each outcome, a *scenario* is sampled from the empirical distribution
    in ``scenario_probs_tbl[outcome, bases, :]``, giving stochastic base
    advancement (e.g., runner on 2B scores on a single only ~60% of the time).

    Args:
        lineup_probs: Calibrated probability matrix, shape (9, N_OUTCOMES).
        batter_pos: Current batter position (0–8).
        new_bases_tbl_p: 3-D base-state table, shape (N_OUTCOMES, 8, N_SCENARIOS).
        runs_tbl_p: 3-D runs table, shape (N_OUTCOMES, 8, N_SCENARIOS).
        scenario_probs_tbl: 3-D probability table, shape (N_OUTCOMES, 8, N_SCENARIOS).
        park_factor_hr: HR park factor multiplier.
        park_factor_xb: Extra-base hit park factor multiplier.
        initial_bases: Starting base-state bitmask.  Use 0 for a normal inning,
            2 (= runner on 2nd) for extra-inning ghost-runner rule (MLB 2020+).

    Returns:
        Tuple ``(runs_scored, new_batter_pos)``.
    """
    outs = 0
    bases = initial_bases
    total_runs = 0
    pos = batter_pos % 9

    while outs < 3:
        probs = _apply_park_factor(lineup_probs[pos], park_factor_hr, park_factor_xb)
        outcome = _sample_outcome(probs)

        if outcome <= 1:              # OUT_IN_PLAY or STRIKEOUT → 1 out
            outs += 1
        elif outcome == 7:            # DOUBLE_PLAY (Mejora 6 / audit D1)
            if bases & 1:             # corredor en 1B → GIDP real: 2 outs, elimina 1B
                outs += 2
                bases = new_bases_tbl_p[7, bases, 0]   # DP scenario is always slot 0
            else:                     # sin corredor forzable en 1B → out sencillo
                outs += 1
        else:
            scenario = _sample_scenario(scenario_probs_tbl[outcome, bases])
            total_runs += runs_tbl_p[outcome, bases, scenario]
            bases = new_bases_tbl_p[outcome, bases, scenario]

        pos = (pos + 1) % 9

    return total_runs, pos


@njit(cache=True)
def _simulate_game_prob(
    my_probs: np.ndarray,
    opp_probs: np.ndarray,
    new_bases_tbl_p: np.ndarray,
    runs_tbl_p: np.ndarray,
    scenario_probs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_innings: int,
    max_extra: int,
) -> tuple[int, int, bool]:
    """Simulates one full game with probabilistic advances and optional extra innings.

    Regulation innings use ``initial_bases=0``.  If the game is tied after
    ``n_innings`` and ``max_extra > 0``, extra innings are played with a ghost
    runner on 2nd (``initial_bases=2``), matching the MLB 2020+ extra-inning rule.

    Args:
        my_probs: My team's lineup probability matrix, shape (9, N_OUTCOMES).
        opp_probs: Opponent's lineup probability matrix, shape (9, N_OUTCOMES).
        new_bases_tbl_p / runs_tbl_p / scenario_probs_tbl: Probabilistic tables.
        park_factor_hr / park_factor_xb: Park factor multipliers.
        n_innings: Number of regulation innings (9 for standard game).
        max_extra: Maximum extra innings to play when tied (0 = no extra innings).

    Returns:
        Tuple ``(my_runs, opp_runs, went_to_extras)`` where ``went_to_extras``
        is ``True`` if the game required extra innings.
    """
    my_runs  = 0
    opp_runs = 0
    my_pos   = 0
    opp_pos  = 0

    for _ in range(n_innings):
        r, my_pos = _simulate_half_inning_prob(
            my_probs, my_pos,
            new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
            park_factor_hr, park_factor_xb, 0,
        )
        my_runs += r
        r, opp_pos = _simulate_half_inning_prob(
            opp_probs, opp_pos,
            new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
            park_factor_hr, park_factor_xb, 0,
        )
        opp_runs += r

    went_to_extras = my_runs == opp_runs
    if went_to_extras and max_extra > 0:
        for _ in range(max_extra):
            r, my_pos = _simulate_half_inning_prob(
                my_probs, my_pos,
                new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
                park_factor_hr, park_factor_xb, 2,   # ghost runner on 2B
            )
            my_runs += r
            r, opp_pos = _simulate_half_inning_prob(
                opp_probs, opp_pos,
                new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
                park_factor_hr, park_factor_xb, 2,
            )
            opp_runs += r
            if my_runs != opp_runs:
                break

    return my_runs, opp_runs, went_to_extras


@njit(parallel=True, cache=True)
def _simulate_n_games_prob(
    my_probs: np.ndarray,
    opp_probs: np.ndarray,
    new_bases_tbl_p: np.ndarray,
    runs_tbl_p: np.ndarray,
    scenario_probs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_sims: int,
    n_innings: int,
    max_extra: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs N probabilistic game simulations in parallel via Numba prange.

    Each simulation uses probabilistic runner advancement and optional
    extra-inning ghost-runner logic.  Parallelism is provided by Numba's
    OpenMP-style prange (one thread per simulation, independent RNG states).

    Args:
        my_probs: My team's matrix, shape (9, N_OUTCOMES). float32.
        opp_probs: Opponent's matrix, shape (9, N_OUTCOMES). float32.
        new_bases_tbl_p / runs_tbl_p / scenario_probs_tbl: Probabilistic tables.
        park_factor_hr / park_factor_xb: Park factor multipliers.
        n_sims: Number of games to simulate.
        n_innings: Regulation innings.
        max_extra: Max extra innings (0 = no extras, matches legacy behavior).

    Returns:
        Tuple ``(my_runs_arr, opp_runs_arr, went_extra_arr)`` each length n_sims.
        ``went_extra_arr`` is bool8; use ``.mean()`` for ``extra_innings_rate``.
    """
    my_runs    = np.empty(n_sims, dtype=np.int32)
    opp_runs   = np.empty(n_sims, dtype=np.int32)
    went_extra = np.empty(n_sims, dtype=np.bool_)

    for i in prange(n_sims):
        mr, or_, we = _simulate_game_prob(
            my_probs, opp_probs,
            new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
            park_factor_hr, park_factor_xb,
            n_innings, max_extra,
        )
        my_runs[i]    = mr
        opp_runs[i]   = or_
        went_extra[i] = we

    return my_runs, opp_runs, went_extra


@njit(parallel=True, cache=True)
def _simulate_n_runs_only(
    my_probs: np.ndarray,
    new_bases_tbl_p: np.ndarray,
    runs_tbl_p: np.ndarray,
    scenario_probs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_sims: int,
    n_innings: int,
) -> np.ndarray:
    """Simula SOLO la ofensiva de mi equipo y devuelve las carreras por juego.

    Optimización de cómputo (audit B4): la fitness del GA solo necesita E[R]
    de mi lineup. ``_simulate_n_games_prob`` simulaba además la ofensiva rival
    para producir ``opp_runs``, que ``run_fast`` descartaba — como no hay
    interacción defensa↔ataque (las carreras dependen solo de los prob_vectors
    del equipo que batea), esa mitad del trabajo era puro desperdicio. Este
    kernel reproduce EXACTAMENTE la misma distribución de ``my_runs`` que la
    rama de mi equipo en ``_simulate_game_prob`` (pos inicia en 0, cada entrada
    arranca con bases vacías, sin extra innings), a ~mitad de coste.

    Args:
        my_probs: Matriz de probabilidad de mi lineup, shape (9, N_OUTCOMES). float32.
        new_bases_tbl_p / runs_tbl_p / scenario_probs_tbl: Tablas probabilísticas.
        park_factor_hr / park_factor_xb: Multiplicadores de park factor.
        n_sims: Número de juegos a simular.
        n_innings: Entradas de regulación.

    Returns:
        Array ``my_runs`` de longitud n_sims (int32).
    """
    my_runs = np.empty(n_sims, dtype=np.int32)
    for i in prange(n_sims):
        runs = 0
        pos = 0
        for _ in range(n_innings):
            r, pos = _simulate_half_inning_prob(
                my_probs, pos,
                new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
                park_factor_hr, park_factor_xb, 0,
            )
            runs += r
        my_runs[i] = runs
    return my_runs


@njit(cache=True)
def _simulate_half_inning(
    lineup_probs: np.ndarray,
    batter_pos: int,
    new_bases_tbl: np.ndarray,
    runs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
) -> tuple[int, int]:
    """Simulates one half-inning (until 3 outs) via Markov chain sampling.

    Iterates the batting order starting at ``batter_pos``, sampling PA outcomes
    from the pre-calibrated probability matrix, applying state transitions from
    the lookup tables, and accumulating runs.

    Args:
        lineup_probs: Calibrated probability matrix, shape (9, 7).
            Row i = probability distribution for the i-th batter in the lineup.
        batter_pos: Current batter position (0–8) entering the half-inning.
        new_bases_tbl: State transition lookup, shape (7, 8).
        runs_tbl: Runs-scored lookup, shape (7, 8).
        park_factor_hr: HR park factor multiplier.
        park_factor_xb: Extra-base hit park factor multiplier.

    Returns:
        Tuple ``(runs_scored, new_batter_pos)`` where ``new_batter_pos``
        is the position of the NEXT batter due up in the following inning.
    """
    outs = 0
    bases = 0
    total_runs = 0
    pos = batter_pos % 9

    while outs < 3:
        probs = _apply_park_factor(lineup_probs[pos], park_factor_hr, park_factor_xb)
        outcome = _sample_outcome(probs)

        if outcome <= 1:        # OUT_IN_PLAY or STRIKEOUT → 1 out
            outs += 1
        elif outcome == 7:      # DOUBLE_PLAY (Mejora 6 / audit D1)
            if bases & 1:       # corredor en 1B → GIDP real: 2 outs, elimina 1B
                outs += 2
                bases = new_bases_tbl[7, bases]   # clears bit-0 (runner on 1B)
            else:               # sin corredor forzable en 1B → out sencillo
                outs += 1
        else:
            total_runs += runs_tbl[outcome, bases]
            bases = new_bases_tbl[outcome, bases]

        pos = (pos + 1) % 9

    return total_runs, pos


@njit(cache=True)
def _simulate_game(
    my_probs: np.ndarray,
    opp_probs: np.ndarray,
    new_bases_tbl: np.ndarray,
    runs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_innings: int,
) -> tuple[int, int]:
    """Simulates one complete game over ``n_innings`` innings.

    Alternates between the home team's lineup (``my_probs``) and the
    opponent's lineup (``opp_probs``), preserving batter position across
    innings (the lineup wraps around 9→0 continuously, not resetting per inning).

    Args:
        my_probs: My team's lineup probability matrix, shape (9, 7).
        opp_probs: Opponent's lineup probability matrix, shape (9, 7).
        new_bases_tbl: Transition lookup, shape (7, 8).
        runs_tbl: Runs lookup, shape (7, 8).
        park_factor_hr: HR park factor.
        park_factor_xb: Extra-base hit park factor.
        n_innings: Number of regulation innings (9 for standard game).

    Returns:
        Tuple ``(my_runs_total, opp_runs_total)``.
    """
    my_runs   = 0
    opp_runs  = 0
    my_pos    = 0
    opp_pos   = 0

    for _ in range(n_innings):
        # My team bats
        r, my_pos = _simulate_half_inning(
            my_probs, my_pos, new_bases_tbl, runs_tbl, park_factor_hr, park_factor_xb
        )
        my_runs += r

        # Opponent bats
        r, opp_pos = _simulate_half_inning(
            opp_probs, opp_pos, new_bases_tbl, runs_tbl, park_factor_hr, park_factor_xb
        )
        opp_runs += r

    return my_runs, opp_runs


@njit(parallel=True, cache=True)
def _simulate_n_games(
    my_probs: np.ndarray,
    opp_probs: np.ndarray,
    new_bases_tbl: np.ndarray,
    runs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_sims: int,
    n_innings: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs N game simulations in parallel using Numba's prange.

    This is the core computational kernel. ``prange`` maps each simulation
    to a separate CPU thread via OpenMP-style parallelism. Each thread has
    its own Numba random state (no locking contention).

    Performance target: 100,000 games in < 10 seconds on a 16-core machine.

    Args:
        my_probs: My team's probability matrix, shape (9, 7). float32.
        opp_probs: Opponent's probability matrix, shape (9, 7). float32.
        new_bases_tbl: Transition lookup, shape (7, 8). int32.
        runs_tbl: Runs lookup, shape (7, 8). int32.
        park_factor_hr: HR park factor.
        park_factor_xb: Extra-base hit park factor.
        n_sims: Number of games to simulate.
        n_innings: Innings per game (9 for regulation).

    Returns:
        Tuple ``(my_runs_arr, opp_runs_arr)``, each shape (n_sims,) int32.
    """
    my_runs  = np.empty(n_sims, dtype=np.int32)
    opp_runs = np.empty(n_sims, dtype=np.int32)

    for i in prange(n_sims):  # parallel — Numba spawns threads automatically
        mr, or_ = _simulate_game(
            my_probs, opp_probs,
            new_bases_tbl, runs_tbl,
            park_factor_hr, park_factor_xb,
            n_innings,
        )
        my_runs[i]  = mr
        opp_runs[i] = or_

    return my_runs, opp_runs


# ---------------------------------------------------------------------------
# Staff-aware simulation (Mejora 1)
# ---------------------------------------------------------------------------
# These functions accept a 3-D ``opp_probs_per_inning`` array of shape
# (n_innings, 9, N_OUTCOMES) rather than the flat (9, N_OUTCOMES) matrix.
# Each inning slice contains the pitcher-vs-lineup prob matrix for that inning,
# allowing the bullpen to have a different profile than the starter.


@njit(cache=True)
def _simulate_game_with_staff(
    my_probs: np.ndarray,
    opp_probs_per_inning: np.ndarray,
    new_bases_tbl_p: np.ndarray,
    runs_tbl_p: np.ndarray,
    scenario_probs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_innings: int,
    max_extra: int,
) -> tuple[int, int, bool]:
    """Simulates one full game with a per-inning pitching staff model.

    The opponent pitches with a different probability vector per inning
    (starter in innings 0..handoff-1, bullpen from handoff onward).  My
    team uses a flat lineup matrix (we only model the opponent's staff).

    Uses probabilistic runner advancement and optional ghost-runner extra innings
    (same as ``_simulate_game_prob``).

    Args:
        my_probs: My team's lineup matrix, shape (9, N_OUTCOMES).
        opp_probs_per_inning: Opponent's per-inning lineup matrix,
            shape (n_innings, 9, N_OUTCOMES).
        new_bases_tbl_p / runs_tbl_p / scenario_probs_tbl: Probabilistic tables.
        park_factor_hr / park_factor_xb: Park factor multipliers.
        n_innings: Regulation innings.
        max_extra: Max extra innings with ghost runner (0 = no extras).

    Returns:
        Tuple ``(my_runs, opp_runs, went_to_extras)``.
    """
    my_runs  = 0
    opp_runs = 0
    my_pos   = 0
    opp_pos  = 0

    for inning in range(n_innings):
        # My team bats (flat lineup, same every inning)
        r, my_pos = _simulate_half_inning_prob(
            my_probs, my_pos,
            new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
            park_factor_hr, park_factor_xb, 0,
        )
        my_runs += r

        # Opponent bats against its inning-specific pitcher
        r, opp_pos = _simulate_half_inning_prob(
            opp_probs_per_inning[inning], opp_pos,
            new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
            park_factor_hr, park_factor_xb, 0,
        )
        opp_runs += r

    went_to_extras = my_runs == opp_runs
    if went_to_extras and max_extra > 0:
        for _ in range(max_extra):
            r, my_pos = _simulate_half_inning_prob(
                my_probs, my_pos,
                new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
                park_factor_hr, park_factor_xb, 2,   # ghost runner
            )
            my_runs += r
            # In extra innings, last bullpen inning slice is reused
            last_inning_idx = n_innings - 1
            r, opp_pos = _simulate_half_inning_prob(
                opp_probs_per_inning[last_inning_idx], opp_pos,
                new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
                park_factor_hr, park_factor_xb, 2,
            )
            opp_runs += r
            if my_runs != opp_runs:
                break

    return my_runs, opp_runs, went_to_extras


@njit(parallel=True, cache=True)
def _simulate_n_games_with_staff(
    my_probs: np.ndarray,
    opp_probs_per_inning: np.ndarray,
    new_bases_tbl_p: np.ndarray,
    runs_tbl_p: np.ndarray,
    scenario_probs_tbl: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_sims: int,
    n_innings: int,
    max_extra: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs N staff-aware game simulations in parallel.

    Performance overhead vs. ``_simulate_n_games_prob``:
        The only additional cost is indexing ``opp_probs_per_inning[inning]``
        (a slice of a pre-computed NumPy array) once per half-inning.
        Benchmark target: < 15% overhead on 100k simulations.

    Args:
        my_probs: My team's lineup matrix, shape (9, N_OUTCOMES). float32.
        opp_probs_per_inning: Opponent per-inning matrix,
            shape (n_innings, 9, N_OUTCOMES). float32.
        new_bases_tbl_p / runs_tbl_p / scenario_probs_tbl: Probabilistic tables.
        park_factor_hr / park_factor_xb: Park factors.
        n_sims: Number of games to simulate.
        n_innings: Regulation innings.
        max_extra: Max extra innings (0 = no extras).

    Returns:
        Tuple ``(my_runs_arr, opp_runs_arr, went_extra_arr)`` each length n_sims.
    """
    my_runs    = np.empty(n_sims, dtype=np.int32)
    opp_runs   = np.empty(n_sims, dtype=np.int32)
    went_extra = np.empty(n_sims, dtype=np.bool_)

    for i in prange(n_sims):
        mr, or_, we = _simulate_game_with_staff(
            my_probs, opp_probs_per_inning,
            new_bases_tbl_p, runs_tbl_p, scenario_probs_tbl,
            park_factor_hr, park_factor_xb,
            n_innings, max_extra,
        )
        my_runs[i]    = mr
        opp_runs[i]   = or_
        went_extra[i] = we

    return my_runs, opp_runs, went_extra


# ---------------------------------------------------------------------------
# JIT warmup
# ---------------------------------------------------------------------------

def warmup_jit() -> None:
    """Forces Numba JIT compilation of all simulation kernels.

    Call this once during application initialization to pay the ~2–4 second
    LLVM compilation cost before the time-critical game-day pipeline starts.
    Subsequent calls are instant (results are cached to disk via
    ``cache=True`` in the @njit decorators).

    Kernels compiled:
        - ``_simulate_n_games`` (deterministic, GA fast mode)
        - ``_simulate_n_games_prob`` (probabilistic + extra innings, production)
    """
    log.info("jit_warmup_start")
    t0 = time.perf_counter()

    # 8-class dummy probs (N_OUTCOMES=8, includes DOUBLE_PLAY slot)
    dummy_probs = np.full((N_LINEUP, N_OUTCOMES), 1.0 / N_OUTCOMES, dtype=np.float32)

    # Deterministic path (GA fast mode)
    _simulate_n_games(
        dummy_probs, dummy_probs,
        _NEW_BASES_TBL, _RUNS_TBL,
        1.0, 1.0,
        n_sims=100,
        n_innings=9,
    )
    log.info("jit_warmup_deterministic_done", elapsed_s=round(time.perf_counter() - t0, 2))

    # Probabilistic + extra innings path (production default)
    _simulate_n_games_prob(
        dummy_probs, dummy_probs,
        _NEW_BASES_TBL_P, _RUNS_TBL_P, _SCENARIO_PROBS,
        1.0, 1.0,
        n_sims=100,
        n_innings=9,
        max_extra=3,   # compile with extra-innings path active
    )
    log.info("jit_warmup_prob_done", elapsed_s=round(time.perf_counter() - t0, 2))

    # Runs-only path (audit B4): proxy de fitness del GA (solo mi ofensiva)
    _simulate_n_runs_only(
        dummy_probs,
        _NEW_BASES_TBL_P, _RUNS_TBL_P, _SCENARIO_PROBS,
        1.0, 1.0,
        n_sims=100,
        n_innings=9,
    )
    log.info("jit_warmup_runs_only_done", elapsed_s=round(time.perf_counter() - t0, 2))

    # Staff-aware path (Mejora 1): per-inning opp probs matrix
    dummy_per_inning = np.tile(dummy_probs, (9, 1, 1))  # (9, 9, N_OUTCOMES)
    _simulate_n_games_with_staff(
        dummy_probs, dummy_per_inning,
        _NEW_BASES_TBL_P, _RUNS_TBL_P, _SCENARIO_PROBS,
        1.0, 1.0,
        n_sims=100,
        n_innings=9,
        max_extra=3,
    )
    elapsed = time.perf_counter() - t0
    log.info("jit_warmup_complete", elapsed_s=round(elapsed, 2))


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Output of a Monte Carlo game simulation batch.

    Attributes:
        n_simulations: Number of games simulated.
        expected_runs_scored: E[R] — mean runs scored per game by my team.
        expected_runs_allowed: E[R_allowed] — mean runs allowed per game.
        win_probability: P(W) — probability that my team wins.
            Ties (same runs after regulation) are split 50/50 between both
            teams, so ``win_probability + opponent_win_probability == 1.0``
            always holds.  The raw tie rate is exposed as ``tie_rate`` for
            diagnostics; a value above ~0.10 suggests the n_innings parameter
            should be increased or extra-innings logic should be added.
        opponent_win_probability: 1 - win_probability.  Always complement.
        tie_rate: Fraction of raw simulations that ended tied after regulation.
            Non-zero because the simulator plays a fixed number of innings with
            no extra-inning logic.  Distributed 50/50 into win_probability and
            opponent_win_probability so the pair sums to 1.
        run_diff_mean: Mean run differential (scored - allowed).
        runs_scored_percentiles: Dict of run-scored percentiles {5,25,50,75,95}.
        runs_allowed_percentiles: Dict of run-allowed percentiles.
        close_game_win_pct: Win % in simulations where margin ≤ 1 run.
        shutout_pct: Fraction of games where my team allowed 0 runs.
        elapsed_seconds: Wall-clock time for the full simulation batch.
    """

    n_simulations: int
    expected_runs_scored: float
    expected_runs_allowed: float
    win_probability: float
    opponent_win_probability: float
    tie_rate: float
    run_diff_mean: float
    runs_scored_percentiles: dict[int, float]
    runs_allowed_percentiles: dict[int, float]
    close_game_win_pct: float
    shutout_pct: float
    elapsed_seconds: float
    # ── Prediction interval fields ────────────────────────────────────────────
    # win_prob_ci_* = IC90 del ERROR DE MUESTREO Monte Carlo (binomial real,
    # audit F04). Es estrecho con n grande, a propósito; NO es la incertidumbre
    # epistémica del modelo (esa sale del backtest out-of-sample).
    win_prob_ci_low: float = 0.0
    win_prob_ci_high: float = 1.0
    std_dev_runs_scored: float = 2.5
    uncertainty_level: str = "medium"   # 'low' | 'medium' | 'high'
    percentile_10: float = 1.0
    percentile_90: float = 8.0
    # ── Extra-innings diagnostics (Mejora 8) ─────────────────────────────────
    extra_innings_rate: float = 0.0   # fraction of simulations that needed extras
    #   Historical MLB rate: ~0.08–0.10. With ghost runner: games resolve faster,
    #   so this represents how often the game *went* to extras, not stayed tied.

    def summary(self) -> str:
        """Returns a human-readable one-line summary for logging."""
        return (
            f"E[R]={self.expected_runs_scored:.3f} "
            f"E[RA]={self.expected_runs_allowed:.3f} "
            f"P(W)={self.win_probability:.3f} "
            f"P(L)={self.opponent_win_probability:.3f} "
            f"tie_rate={self.tie_rate:.4f} "
            f"n={self.n_simulations:,} "
            f"t={self.elapsed_seconds:.2f}s"
        )

    @property
    def pythagorean_win_pct(self) -> float:
        """Pythagorean win expectation: RS² / (RS² + RA²).

        This is an independent cross-check on P(W) from the Pythagorean
        Expectation formula (Bill James, 1980). If this diverges significantly
        from ``win_probability``, the simulation may have a bug.
        """
        rs = self.expected_runs_scored
        ra = self.expected_runs_allowed
        denom = rs ** 2 + ra ** 2
        return rs ** 2 / denom if denom > 0 else 0.5


def _aggregate_results(
    my_runs: np.ndarray,
    opp_runs: np.ndarray,
    elapsed: float,
    went_extra: Optional[np.ndarray] = None,
) -> SimulationResult:
    """Aggregates raw simulation arrays into a ``SimulationResult``.

    Args:
        my_runs: Array of my team's runs per game, shape (N,).
        opp_runs: Array of opponent's runs per game, shape (N,).
        elapsed: Wall-clock seconds for the simulation batch.
        went_extra: Optional bool array, shape (N,).  ``True`` for games
            that required extra innings.  When provided, ``extra_innings_rate``
            is computed from it and ``tie_rate`` reflects remaining ties
            (should be ~0 when extra-innings logic is active).

    Returns:
        Populated ``SimulationResult`` dataclass.
    """
    n = len(my_runs)
    wins   = my_runs > opp_runs   # strict: my team leads
    losses = opp_runs > my_runs   # strict: opponent leads
    ties   = ~wins & ~losses      # same score after regulation (no extra innings)

    # Baseball has no ties — the simulator plays a fixed number of innings so
    # some games end level (~5–8 % with default 9-inning regulation).  We
    # distribute them 50 / 50 so that win_probability + opponent_win_probability
    # equals exactly 1.0, which is the only mathematically valid partition for a
    # binary outcome space.
    n_ties           = int(ties.sum())
    n_wins_adjusted  = wins.sum() + n_ties * 0.5
    win_probability  = float(n_wins_adjusted / n)
    opp_probability  = float(1.0 - win_probability)   # guaranteed complement

    margin    = np.abs(my_runs - opp_runs)
    close_mask = margin <= 1

    pct_keys = [5, 25, 50, 75, 95]

    # ── IC90 del MUESTREO Monte Carlo para win probability (audit F04) ───────
    # HONESTO: error binomial de la estimación MC, SE = sqrt(p(1-p)/n). Antes se
    # fabricaba con un _n_eff=100 inventado para ensanchar el intervalo "porque
    # <1pp looks broken to reviewers". Eso presentaba incertidumbre ficticia
    # como si fuera del modelo. Aquí se reporta el error de muestreo REAL: con n
    # grande el IC es estrecho a propósito (solo dice cuánta incertidumbre de
    # muestreo MC queda). La incertidumbre EPISTÉMICA del modelo (más ancha) debe
    # estimarse del backtest out-of-sample, no inventarse en el agregador.
    _z90 = 1.645          # z-score para IC del 90%
    _p = win_probability
    _se = float(np.sqrt(max(_p * (1.0 - _p), 0.0) / max(n, 1)))
    win_prob_ci_low  = float(np.clip(_p - _z90 * _se, 0.0, 1.0))
    win_prob_ci_high = float(np.clip(_p + _z90 * _se, 0.0, 1.0))

    # ── Standard deviation + percentiles for runs distribution ──────────────
    # Valores REALES de la distribución simulada (sin pisos cosméticos). Un P10=0
    # es un juego de blanqueada legítimo, no un fallo; el formateo es tarea de la
    # capa de presentación, no del cálculo (audit F04).
    std_dev_runs = float(np.std(my_runs))
    p10 = float(np.percentile(my_runs, 10))
    p90 = float(np.percentile(my_runs, 90))

    # ── Uncertainty level from P10–P90 interval width ───────────────────────
    interval_width = p90 - p10
    if interval_width < 5.0:
        uncertainty_level = "low"
    elif interval_width < 8.0:
        uncertainty_level = "medium"
    else:
        uncertainty_level = "high"

    # Extra-innings rate (Mejora 8)
    extra_innings_rate = float(went_extra.mean()) if went_extra is not None else 0.0

    return SimulationResult(
        n_simulations=n,
        expected_runs_scored=float(my_runs.mean()),
        expected_runs_allowed=float(opp_runs.mean()),
        win_probability=win_probability,
        opponent_win_probability=opp_probability,
        tie_rate=float(n_ties / n),
        run_diff_mean=float((my_runs - opp_runs).mean()),
        runs_scored_percentiles={
            p: float(np.percentile(my_runs, p)) for p in pct_keys
        },
        runs_allowed_percentiles={
            p: float(np.percentile(opp_runs, p)) for p in pct_keys
        },
        close_game_win_pct=(
            float(wins[close_mask].mean()) if close_mask.any() else 0.5
        ),
        shutout_pct=float((opp_runs == 0).mean()),
        elapsed_seconds=elapsed,
        win_prob_ci_low=win_prob_ci_low,
        win_prob_ci_high=win_prob_ci_high,
        std_dev_runs_scored=std_dev_runs,
        uncertainty_level=uncertainty_level,
        percentile_10=p10,
        percentile_90=p90,
        extra_innings_rate=extra_innings_rate,
    )


# ---------------------------------------------------------------------------
# Ray remote worker
# ---------------------------------------------------------------------------

def _simulate_batch_remote(
    my_probs: np.ndarray,
    opp_probs: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_sims: int,
    n_innings: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray remote task: simulates a batch of games on one worker.

    Plain function (audit F16): se envuelve con ``ray.remote`` en tiempo de
    ejecución dentro de ``MonteCarloEngine._ensure_ray`` para no requerir Ray al
    importar el módulo.

    Each worker runs its own Numba-parallel batch, giving two levels of
    parallelism: intra-worker (Numba prange threads) and inter-worker (Ray).

    Args:
        my_probs: My team's lineup probs, shape (9, 7). float32.
        opp_probs: Opponent's lineup probs, shape (9, 7). float32.
        park_factor_hr: HR park factor multiplier.
        park_factor_xb: Extra-base hit park factor multiplier.
        n_sims: Games per worker batch.
        n_innings: Innings per game.

    Returns:
        Tuple (my_runs, opp_runs), each shape (n_sims,) int32.
    """
    # Each Ray worker JIT-compiles on its first call (cached after that)
    return _simulate_n_games(
        my_probs.astype(np.float32),
        opp_probs.astype(np.float32),
        _NEW_BASES_TBL,
        _RUNS_TBL,
        park_factor_hr,
        park_factor_xb,
        n_sims,
        n_innings,
    )


# ---------------------------------------------------------------------------
# High-level engine
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloConfig:
    """Configuration for the Monte Carlo engine.

    Attributes:
        n_simulations: Total games to simulate (default 100,000).
        n_ray_workers: Number of Ray remote workers. Each handles
            ``n_simulations // n_ray_workers`` games.
        n_innings: Innings per game (9 for regulation).
        park_factor_hr: Home run park factor for the target venue.
        park_factor_xb: Extra-base hit park factor.
        use_ray: If ``False``, runs Numba-only on local machine (faster for
            small batches; preferred in the GA fitness evaluator).
        fast_mode_n_sims: Reduced simulation count for GA fitness evaluation.
        use_probabilistic_advances: When ``True`` (default), the production
            mode uses stochastic runner-advance tables (Mejora 7).  Set
            ``False`` to restore deterministic behavior for regression tests.
        runner_advance_era: Era label for the empirical advance-rate lookup.
            Values: ``"2015-2019"``, ``"2020-2022"``, ``"2023+"`` (default).
        use_extra_innings: When ``True`` (default), games tied after regulation
            play extra innings with the ghost-runner rule (MLB 2020+, Mejora 8).
            ``False`` restores the legacy 50/50 tie-split behavior.
        max_extra_innings: Maximum extra innings to simulate per game.
            6 covers > 99.9 % of historical MLB extra-inning games.
        season_year: Used to set defaults for ``use_extra_innings`` (ghost runner
            was introduced in 2020; if ``season_year < 2020``, extra innings
            run without the ghost runner, as per historical rules).
    """

    n_simulations: int = 100_000
    n_ray_workers: int = 20
    n_innings: int = 9
    park_factor_hr: float = 1.0
    park_factor_xb: float = 1.0
    use_ray: bool = True
    fast_mode_n_sims: int = 5_000
    # ── Mejora 7: probabilistic runner advancement ────────────────────────────
    use_probabilistic_advances: bool = True
    runner_advance_era: str = "2023+"
    # ── Mejora 8: extra innings with ghost runner ─────────────────────────────
    use_extra_innings: bool = True
    max_extra_innings: int = 6
    season_year: int = 2024


class MonteCarloEngine:
    """Orchestrates Monte Carlo game simulation for a specific lineup.

    Supports two execution modes:
        - **Full mode** (``use_ray=True``): Distributes across Ray workers
          for maximum throughput. Use for the final lineup evaluation.
        - **Fast mode** (``use_ray=False``): Runs Numba-parallel locally.
          Use for the GA fitness evaluator to minimize latency per call.

    Args:
        config: Engine configuration.
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        """Initializes the engine and optionally connects to Ray.

        Builds or caches probabilistic runner-advance tables based on
        ``config.runner_advance_era``.  If the era matches the module-level
        default (``"2023+"``) the pre-built arrays are reused; otherwise a
        new set is constructed (< 1 ms, pure NumPy).

        Args:
            config: Engine configuration; uses defaults if not provided.
        """
        self.config = config or MonteCarloConfig()
        self._ray_initialized = False
        self._ray_remote_fn = None   # envoltura ray.remote perezosa (audit F16)

        # Build probabilistic tables for the configured era
        if self.config.runner_advance_era == _DEFAULT_PROB_ERA:
            self._nb_tbl_p  = _NEW_BASES_TBL_P
            self._runs_tbl_p = _RUNS_TBL_P
            self._scen_probs = _SCENARIO_PROBS
        else:
            nb, r, sc = build_probabilistic_transition_tables(
                RunnerAdvanceConfig(era=self.config.runner_advance_era)
            )
            self._nb_tbl_p   = nb
            self._runs_tbl_p = r
            self._scen_probs = sc

        log.info(
            "engine_initialized",
            era=self.config.runner_advance_era,
            probabilistic=self.config.use_probabilistic_advances,
            extra_innings=self.config.use_extra_innings,
        )

    def _ensure_ray(self):
        """Imports Ray lazily, inits the runtime, and returns the remote fn.

        Audit F16: Ray solo se importa aquí (no al cargar el módulo). Devuelve la
        función ``_simulate_batch_remote`` ya envuelta con ``ray.remote``.
        """
        import ray  # import perezoso
        if not self._ray_initialized and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            self._ray_initialized = True
            log.info("ray_initialized")
        if self._ray_remote_fn is None:
            self._ray_remote_fn = ray.remote(_simulate_batch_remote)
        return ray, self._ray_remote_fn

    def run(
        self,
        my_lineup_probs: np.ndarray,
        opp_lineup_probs: np.ndarray,
        park_factor_hr: Optional[float] = None,
        park_factor_xb: Optional[float] = None,
        fast_mode: bool = False,
        opp_pitching_staff: Optional[PitchingStaffConfig] = None,
    ) -> SimulationResult:
        """Simulates N games for a given lineup vs. opponent and returns statistics.

        Args:
            my_lineup_probs: Calibrated probability matrix for my team's lineup,
                shape (9, N_OUTCOMES). Row order = batting order positions 1–9.
                **Must be float32** for Numba performance.
            opp_lineup_probs: Opponent's flat lineup probability matrix,
                shape (9, N_OUTCOMES).  Used when ``opp_pitching_staff`` is
                ``None`` (legacy behavior) or as the fallback in fast_mode.
            park_factor_hr: Override the config's HR park factor.
            park_factor_xb: Override the config's extra-base hit park factor.
            fast_mode: If ``True``, uses ``config.fast_mode_n_sims`` and bypasses
                Ray for faster execution (used by the GA fitness evaluator).
                Staff model is NOT used in fast_mode (too costly for GA loops).
            opp_pitching_staff: Optional ``PitchingStaffConfig`` for the
                opponent's pitching staff (Mejora 1).  When provided, the
                opponent's probability matrix changes per inning (starter →
                bullpen handoff at ``handoff_inning``).  Ignored when
                ``fast_mode=True`` (GA uses the flat matrix for speed).

        Returns:
            ``SimulationResult`` with E[R], P(W), percentiles, and diagnostics.

        Raises:
            ValueError: If probability matrices have wrong shape.
        """
        if my_lineup_probs.shape != (N_LINEUP, N_OUTCOMES):
            raise ValueError(
                f"my_lineup_probs must be shape ({N_LINEUP}, {N_OUTCOMES}), "
                f"got {my_lineup_probs.shape}"
            )
        if opp_lineup_probs.shape != (N_LINEUP, N_OUTCOMES):
            raise ValueError(
                f"opp_lineup_probs must be shape ({N_LINEUP}, {N_OUTCOMES}), "
                f"got {opp_lineup_probs.shape}"
            )

        pf_hr = park_factor_hr if park_factor_hr is not None else self.config.park_factor_hr
        pf_xb = park_factor_xb if park_factor_xb is not None else self.config.park_factor_xb
        n_sims = self.config.fast_mode_n_sims if fast_mode else self.config.n_simulations

        my_probs  = my_lineup_probs.astype(np.float32)
        opp_probs = opp_lineup_probs.astype(np.float32)

        t0 = time.perf_counter()

        went_extra: Optional[np.ndarray] = None

        # ── Staff-aware path (Mejora 1, production + non-fast-mode only) ────
        use_prob  = self.config.use_probabilistic_advances or self.config.use_extra_innings
        max_extra = self.config.max_extra_innings if self.config.use_extra_innings else 0

        if opp_pitching_staff is not None and not fast_mode:
            # Build per-inning opponent matrix (< 5ms, pure NumPy)
            opp_per_inning = build_per_inning_prob_matrix(
                opp_pitching_staff,
                n_innings=self.config.n_innings,
                n_batters=N_LINEUP,
                n_outcomes=N_OUTCOMES,
            ).astype(np.float32)  # (n_innings, 9, N_OUTCOMES)

            my_runs, opp_runs, went_extra = _simulate_n_games_with_staff(
                my_probs, opp_per_inning,
                self._nb_tbl_p, self._runs_tbl_p, self._scen_probs,
                pf_hr, pf_xb, n_sims, self.config.n_innings, max_extra,
            )
        # ── Probabilistic path (Mejoras 7 + 8, default for production) ──────
        elif use_prob and not fast_mode:
            my_runs, opp_runs, went_extra = _simulate_n_games_prob(
                my_probs, opp_probs,
                self._nb_tbl_p, self._runs_tbl_p, self._scen_probs,
                pf_hr, pf_xb, n_sims, self.config.n_innings, max_extra,
            )
        elif fast_mode or not self.config.use_ray:
            # Deterministic Numba-only path: GA fitness evaluator and fast_mode
            my_runs, opp_runs = _simulate_n_games(
                my_probs, opp_probs,
                _NEW_BASES_TBL, _RUNS_TBL,
                pf_hr, pf_xb, n_sims, self.config.n_innings,
            )
        else:
            # Deterministic Ray + Numba: legacy full mode
            ray, remote_fn = self._ensure_ray()
            sims_per_worker = n_sims // self.config.n_ray_workers
            remainder       = n_sims % self.config.n_ray_workers

            futures = [
                remote_fn.remote(
                    my_probs, opp_probs, pf_hr, pf_xb,
                    sims_per_worker + (1 if i < remainder else 0),
                    self.config.n_innings,
                )
                for i in range(self.config.n_ray_workers)
            ]
            batch_results = ray.get(futures)
            my_runs  = np.concatenate([r[0] for r in batch_results])
            opp_runs = np.concatenate([r[1] for r in batch_results])

        elapsed = time.perf_counter() - t0
        result  = _aggregate_results(my_runs, opp_runs, elapsed, went_extra)

        log.info(
            "simulation_complete",
            mode="fast" if fast_mode else "full",
            n_sims=n_sims,
            **{k: round(v, 4) for k, v in {
                "expected_runs": result.expected_runs_scored,
                "win_prob": result.win_probability,
            }.items()},
            elapsed_s=round(elapsed, 3),
        )
        return result

    def run_fast(
        self,
        my_lineup_probs: np.ndarray,
        opp_lineup_probs: np.ndarray,
        park_factor_hr: Optional[float] = None,
        park_factor_xb: Optional[float] = None,
    ) -> float:
        """Returns only E[R] using the fast (Numba-only) simulation mode.

        This is the method called by the GA fitness evaluator in
        ``lineup_optimizer.py``. It returns a scalar float for maximum
        speed, skipping all aggregation overhead except the mean.

        Consistencia fitness/refinamiento (audit fix F03):
            Usa exactamente las MISMAS tablas probabilísticas de 8 outcomes
            (``_nb_tbl_p`` / ``_runs_tbl_p`` / ``_scen_probs``) y los MISMOS park
            factors que ``run(fast_mode=False)``. La única diferencia es el número
            de simulaciones (``fast_mode_n_sims``) y la ausencia de extra innings
            (irrelevantes para E[R] y costosos en el bucle del GA). Así el GA busca
            sobre el mismo paisaje de fitness que el refinamiento, en vez del
            proxy determinista de máximo avance que inflaba E[R] 0.2–0.4 carreras.

        Args:
            my_lineup_probs: My team's probability matrix, shape (9, N_OUTCOMES).
            opp_lineup_probs: Opponent's probability matrix, shape (9, N_OUTCOMES).
            park_factor_hr: HR park factor. Si ``None``, usa ``config.park_factor_hr``.
            park_factor_xb: Extra-base hit park factor. Si ``None``, usa el de config.

        Returns:
            ``E[R]`` as a scalar float.
        """
        pf_hr = park_factor_hr if park_factor_hr is not None else self.config.park_factor_hr
        pf_xb = park_factor_xb if park_factor_xb is not None else self.config.park_factor_xb
        my_probs = my_lineup_probs.astype(np.float32)
        # ``opp_lineup_probs`` se ignora a propósito (audit B4): E[R] de mi lineup
        # no depende de la ofensiva rival. Se mantiene en la firma por
        # compatibilidad con quien llama (el GA pasa el oponente de liga).
        my_runs = _simulate_n_runs_only(
            my_probs,
            self._nb_tbl_p, self._runs_tbl_p, self._scen_probs,
            pf_hr, pf_xb,
            self.config.fast_mode_n_sims, self.config.n_innings,
        )
        return float(my_runs.mean())

    def benchmark(
        self,
        my_probs: Optional[np.ndarray] = None,
        n_sims: int = 100_000,
    ) -> dict[str, float]:
        """Benchmarks the Numba simulation kernel and returns performance stats.

        Useful for verifying the ≤10 seconds SLA for 100,000 simulations
        before deploying to the game-day pipeline.

        Args:
            my_probs: Optional custom lineup probs, shape (9, 7).
                Defaults to uniform (1/7 per outcome) if not provided.
            n_sims: Number of simulations for the benchmark.

        Returns:
            Dict with ``"elapsed_s"``, ``"games_per_second"``, and
            ``"sla_met"`` (bool: elapsed ≤ 10 s).
        """
        if my_probs is None:
            my_probs = np.full((9, 7), 1.0 / 7, dtype=np.float32)
        opp_probs = my_probs.copy()

        # First call may be slow (JIT compilation); use warmup for fair benchmark
        warmup_jit()

        t0 = time.perf_counter()
        _simulate_n_games(
            my_probs, opp_probs,
            _NEW_BASES_TBL, _RUNS_TBL,
            1.0, 1.0, n_sims, 9,
        )
        elapsed = time.perf_counter() - t0

        result = {
            "n_sims": n_sims,
            "elapsed_s": round(elapsed, 3),
            "games_per_second": round(n_sims / elapsed),
            "sla_met_10s": elapsed <= 10.0,
        }
        log.info("benchmark_result", **result)
        return result


# ---------------------------------------------------------------------------
# Run Expectancy Matrix (analytical complement to Monte Carlo)
# ---------------------------------------------------------------------------

def compute_run_expectancy_matrix(
    lineup_probs: np.ndarray,
    n_simulations: int = 50_000,
) -> np.ndarray:
    """Estimates the 24-state Run Expectancy Matrix (RE24) via simulation.

    RE24[outs, bases] = expected runs scored from (outs, bases) until end
    of half-inning. This is the foundational metric for evaluating the run
    value of each base-out state.

    Computed by running half-inning simulations starting from each of the
    24 states and averaging runs scored.

    Args:
        lineup_probs: Lineup probability matrix, shape (9, 7). float32.
        n_simulations: Simulations per state (24 × n_sims total).

    Returns:
        RE24 matrix of shape (3, 8) where axis 0 = outs (0–2) and
        axis 1 = base state (0–7 bitmask).
    """
    re_matrix = np.zeros((3, 8), dtype=np.float64)
    probs = lineup_probs.astype(np.float32)

    for outs_start in range(3):
        for bases_start in range(8):
            runs_list = []
            for _ in range(n_simulations):
                outs  = outs_start
                bases = bases_start
                runs  = 0
                pos   = 0

                while outs < 3:
                    outcome = _sample_outcome(probs[pos])
                    if outcome <= 1:            # OUT_IN_PLAY o STRIKEOUT → 1 out
                        outs += 1
                    elif outcome == 7:          # DOUBLE_PLAY (audit F11 / D1)
                        if bases & 1:           # corredor en 1B → GIDP real: 2 outs
                            outs  += 2
                            bases  = _NEW_BASES_TBL[7, bases]
                        else:                   # sin corredor forzable → out sencillo
                            outs += 1
                    else:
                        runs  += _RUNS_TBL[outcome, bases]
                        bases  = _NEW_BASES_TBL[outcome, bases]
                    pos = (pos + 1) % 9

                runs_list.append(runs)

            re_matrix[outs_start, bases_start] = np.mean(runs_list)

    return re_matrix
