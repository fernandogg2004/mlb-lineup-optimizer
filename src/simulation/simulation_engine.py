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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import ray
import structlog
from numba import njit, prange

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

N_OUTCOMES: int = 7
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
    """Precomputes the full (7, 8) transition lookup tables.

    Baseball runner advancement rules encoded:
        HR     : all runners + batter score; bases cleared.
        Triple : all runners score; batter on 3rd.
        Double : runners on 2nd and 3rd score; runner on 1st to 3rd; batter to 2nd.
        Single : runner on 3rd scores; runner on 2nd to 3rd; runner on 1st to 2nd; batter to 1st.
        Walk/HBP: force advance — batter to 1st; only forced runners advance.
        Out/K  : bases unchanged; outs + 1.

    Returns:
        Tuple (new_bases_table, runs_table), each shape (7, 8) int32.
    """
    new_bases = np.zeros((7, 8), dtype=np.int32)
    runs      = np.zeros((7, 8), dtype=np.int32)

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

    return new_bases, runs


# Module-level constants — Numba sees these as globals
_NEW_BASES_TBL, _RUNS_TBL = _build_transition_tables()


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

        # Out or Strikeout → increment outs
        if outcome <= 1:
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
# JIT warmup
# ---------------------------------------------------------------------------

def warmup_jit() -> None:
    """Forces Numba JIT compilation of all simulation kernels.

    Call this once during application initialization to pay the ~2 second
    LLVM compilation cost before the time-critical game-day pipeline starts.
    Subsequent calls are instant (results are cached to disk via
    ``cache=True`` in the @njit decorators).
    """
    log.info("jit_warmup_start")
    t0 = time.perf_counter()
    dummy_probs = np.full((9, 7), 1.0 / 7, dtype=np.float32)
    _simulate_n_games(
        dummy_probs, dummy_probs,
        _NEW_BASES_TBL, _RUNS_TBL,
        1.0, 1.0,
        n_sims=100,
        n_innings=9,
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
        win_probability: P(W) = fraction of simulations where my team won.
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
    run_diff_mean: float
    runs_scored_percentiles: dict[int, float]
    runs_allowed_percentiles: dict[int, float]
    close_game_win_pct: float
    shutout_pct: float
    elapsed_seconds: float

    def summary(self) -> str:
        """Returns a human-readable one-line summary for logging."""
        return (
            f"E[R]={self.expected_runs_scored:.3f} "
            f"E[RA]={self.expected_runs_allowed:.3f} "
            f"P(W)={self.win_probability:.3f} "
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
) -> SimulationResult:
    """Aggregates raw simulation arrays into a ``SimulationResult``.

    Args:
        my_runs: Array of my team's runs per game, shape (N,).
        opp_runs: Array of opponent's runs per game, shape (N,).
        elapsed: Wall-clock seconds for the simulation batch.

    Returns:
        Populated ``SimulationResult`` dataclass.
    """
    n = len(my_runs)
    wins = my_runs > opp_runs
    margin = np.abs(my_runs - opp_runs)
    close_mask = margin <= 1

    pct_keys = [5, 25, 50, 75, 95]

    return SimulationResult(
        n_simulations=n,
        expected_runs_scored=float(my_runs.mean()),
        expected_runs_allowed=float(opp_runs.mean()),
        win_probability=float(wins.mean()),
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
    )


# ---------------------------------------------------------------------------
# Ray remote worker
# ---------------------------------------------------------------------------

@ray.remote
def _simulate_batch_remote(
    my_probs: np.ndarray,
    opp_probs: np.ndarray,
    park_factor_hr: float,
    park_factor_xb: float,
    n_sims: int,
    n_innings: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray remote task: simulates a batch of games on one worker.

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
    """

    n_simulations: int = 100_000
    n_ray_workers: int = 20
    n_innings: int = 9
    park_factor_hr: float = 1.0
    park_factor_xb: float = 1.0
    use_ray: bool = True
    fast_mode_n_sims: int = 5_000


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

        Args:
            config: Engine configuration; uses defaults if not provided.
        """
        self.config = config or MonteCarloConfig()
        self._ray_initialized = False

    def _ensure_ray(self) -> None:
        """Initializes the Ray runtime if it is not already running."""
        if not self._ray_initialized and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            self._ray_initialized = True
            log.info("ray_initialized")

    def run(
        self,
        my_lineup_probs: np.ndarray,
        opp_lineup_probs: np.ndarray,
        park_factor_hr: Optional[float] = None,
        park_factor_xb: Optional[float] = None,
        fast_mode: bool = False,
    ) -> SimulationResult:
        """Simulates N games for a given lineup vs. opponent and returns statistics.

        Args:
            my_lineup_probs: Calibrated probability matrix for my team's lineup,
                shape (9, 7). Row order = batting order positions 1–9.
                **Must be float32** for Numba performance.
            opp_lineup_probs: Opponent's probability matrix, shape (9, 7).
            park_factor_hr: Override the config's HR park factor.
            park_factor_xb: Override the config's extra-base hit park factor.
            fast_mode: If ``True``, uses ``config.fast_mode_n_sims`` and bypasses
                Ray for faster execution (used by the GA fitness evaluator).

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

        if fast_mode or not self.config.use_ray:
            # Numba-only: fastest for small batches and GA fitness loops
            my_runs, opp_runs = _simulate_n_games(
                my_probs, opp_probs,
                _NEW_BASES_TBL, _RUNS_TBL,
                pf_hr, pf_xb, n_sims, self.config.n_innings,
            )
        else:
            # Ray + Numba: distribute across workers for large batches
            self._ensure_ray()
            sims_per_worker = n_sims // self.config.n_ray_workers
            remainder       = n_sims % self.config.n_ray_workers

            futures = [
                _simulate_batch_remote.remote(
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
        result  = _aggregate_results(my_runs, opp_runs, elapsed)

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
        park_factor_hr: float = 1.0,
        park_factor_xb: float = 1.0,
    ) -> float:
        """Returns only E[R] using the fast (Numba-only) simulation mode.

        This is the method called by the GA fitness evaluator in
        ``lineup_optimizer.py``. It returns a scalar float for maximum
        speed, skipping all aggregation overhead except the mean.

        Args:
            my_lineup_probs: My team's probability matrix, shape (9, 7).
            opp_lineup_probs: Opponent's probability matrix, shape (9, 7).
            park_factor_hr: HR park factor.
            park_factor_xb: Extra-base hit park factor.

        Returns:
            ``E[R]`` as a scalar float.
        """
        my_probs  = my_lineup_probs.astype(np.float32)
        opp_probs = opp_lineup_probs.astype(np.float32)
        my_runs, _ = _simulate_n_games(
            my_probs, opp_probs,
            _NEW_BASES_TBL, _RUNS_TBL,
            park_factor_hr, park_factor_xb,
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
                    if outcome <= 1:
                        outs += 1
                    else:
                        runs  += _RUNS_TBL[outcome, bases]
                        bases  = _NEW_BASES_TBL[outcome, bases]
                    pos = (pos + 1) % 9

                runs_list.append(runs)

            re_matrix[outs_start, bases_start] = np.mean(runs_list)

    return re_matrix
