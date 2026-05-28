"""
bullpen_model.py — Pitching staff model for mid-game lineup simulation.

The problem this solves
-----------------------
The Monte Carlo engine previously used a single fixed ``opp_lineup_probs``
matrix (9 batters × N_OUTCOMES) for all 9 innings.  In reality the starting
pitcher is replaced around innings 6–7 by the bullpen, which typically has a
significantly different — often *better* — ERA profile:

    MLB average 2023+:
        Starting pitcher innings 1–5:  ERA ≈ 4.40
        Bullpen innings 6–9:           ERA ≈ 3.80

Using a flat SP matrix over-estimates E[RA] in innings 6–9 by ≈ 0.15–0.25
runs per game.

Design
------
``PitcherProfile``
    Lightweight dataclass holding the probability vector for one pitcher.
    Supports a ``fatigue_factor`` multiplier from ``api.fatigue_model``.

``PitchingStaffConfig``
    Configures the full staff (starter + bullpen order).  Two modes:

    *simple_mode=True* (default):
        Innings 0 .. handoff_inning-1 → starter probs (scaled by fatigue_factor)
        Innings handoff_inning .. N-1 → mean of available bullpen probs

    *simple_mode=False* (phase 2, future):
        Matchup-specific probs per pitcher × batter combination.  Requires
        the pitcher arsenal embeddings to be pre-computed per hitter.

``build_per_inning_prob_matrix``
    Converts a ``PitchingStaffConfig`` into the ``(n_innings, 9, N_OUTCOMES)``
    array consumed by ``_simulate_n_games_with_staff`` in ``simulation_engine``.

Backwards compatibility
-----------------------
``PitchingStaffConfig.single_pitcher_legacy(prob_vector)`` factory returns a
config that exactly reproduces the pre-Mejora-1 behavior (flat matrix, all
innings identical to the starter).

Usage:
    from src.simulation.bullpen_model import (
        PitcherProfile,
        PitchingStaffConfig,
        build_per_inning_prob_matrix,
    )

    starter = PitcherProfile(
        pitcher_id=543_305,
        prob_vector=np.array([0.28, 0.22, 0.09, 0.18, 0.08, 0.01, 0.04, 0.10],
                              dtype=np.float32),
        role="SP",
        fatigue_factor=0.94,
    )
    bullpen = [
        PitcherProfile(pitcher_id=622_567, prob_vector=..., role="MRP"),
        PitcherProfile(pitcher_id=669_016, prob_vector=..., role="SU"),
        PitcherProfile(pitcher_id=543_302, prob_vector=..., role="CL"),
    ]
    staff = PitchingStaffConfig(starter=starter, bullpen=bullpen)
    opp_probs_per_inning = build_per_inning_prob_matrix(staff, n_innings=9)
    # shape: (9, 9, 8) — (innings, batters, outcomes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# N_OUTCOMES mirrors simulation_engine.N_OUTCOMES (8 after Mejora 6).
# Import lazily to avoid circular dependency; use constant here.
_N_OUTCOMES_DEFAULT: int = 8


# ---------------------------------------------------------------------------
# Pitcher profile
# ---------------------------------------------------------------------------

@dataclass
class PitcherProfile:
    """Probability profile for a single pitcher.

    Attributes:
        pitcher_id: MLB pitcher identifier.
        prob_vector: Mean matchup probability vector, shape (N_OUTCOMES,).
            Represents the average PA outcome distribution when this pitcher
            faces the opposing lineup (averaged across all batter matchups).
        role: Pitching role — "SP", "MRP", "SU", or "CL".
        available: Whether the pitcher is available for tonight's game.
            Unavailable pitchers are skipped when building the bullpen matrix.
        fatigue_factor: Multiplicative adjustment from ``api.fatigue_model``.
            Range [0.85, 1.0]; values < 1 shift probability mass from
            ``OUT_IN_PLAY`` toward ``WALK_HBP`` and ``HOME_RUN``.
        max_innings: Maximum innings this pitcher can be asked to throw.
    """

    pitcher_id: int
    prob_vector: np.ndarray  # shape (N_OUTCOMES,), dtype float32
    role: str = "SP"         # "SP" | "MRP" | "SU" | "CL"
    available: bool = True
    fatigue_factor: float = 1.0   # from api.fatigue_model; [0.85, 1.0]
    max_innings: int = 3


# ---------------------------------------------------------------------------
# Staff configuration
# ---------------------------------------------------------------------------

@dataclass
class PitchingStaffConfig:
    """Full pitching staff configuration for one game.

    Attributes:
        starter: The starting pitcher profile.
        bullpen: Bullpen pitchers in expected deployment order
            (MRP → SU → CL).  Can be empty (starter goes the distance).
        starter_max_pitches: Pitch count threshold for starting pitcher.
            Not used in simple_mode (inning threshold only).
        handoff_inning: Zero-indexed inning at which the first bullpen pitcher
            enters.  Default 6 = typical SP through 5 innings.
        simple_mode: When ``True``, uses inning-threshold handoff and averages
            all available bullpen pitchers.  When ``False``, future matchup-
            specific per-pitcher probs are used (Phase 2).
    """

    starter: PitcherProfile
    bullpen: list[PitcherProfile] = field(default_factory=list)
    starter_max_pitches: int = 100
    handoff_inning: int = 6    # inning index (0-based): default = entry into inning 7
    simple_mode: bool = True

    @classmethod
    def single_pitcher_legacy(cls, prob_vector: np.ndarray) -> "PitchingStaffConfig":
        """Creates a staff that exactly reproduces pre-Mejora-1 behavior.

        All 9 innings use the same probability vector (flat, no bullpen).
        This is the backwards-compatible fallback when ``opp_pitching_staff``
        is ``None`` in ``MonteCarloEngine.run()``.

        Args:
            prob_vector: Lineup probability vector, shape (N_OUTCOMES,).

        Returns:
            A ``PitchingStaffConfig`` in legacy mode.
        """
        return cls(
            starter=PitcherProfile(
                pitcher_id=0,
                prob_vector=prob_vector.astype(np.float32),
                role="SP",
                fatigue_factor=1.0,
            ),
            bullpen=[],
            handoff_inning=999,   # never hand off → starter pitches forever
            simple_mode=True,
        )


# ---------------------------------------------------------------------------
# Apply fatigue factor to a probability vector
# ---------------------------------------------------------------------------

def _apply_fatigue(prob_vector: np.ndarray, fatigue_factor: float) -> np.ndarray:
    """Adjusts a pitcher's probability vector by fatigue factor.

    Fatigue shifts probability mass from ``OUT_IN_PLAY`` (index 0) toward
    ``WALK_HBP`` (index 2) and ``HOME_RUN`` (index 6), reflecting reduced
    stuff and command.  The vector is renormalized to sum to 1.0.

    Args:
        prob_vector: Original (N_OUTCOMES,) float32 probability vector.
        fatigue_factor: Value in [0.85, 1.0].  1.0 = no adjustment.

    Returns:
        Adjusted and renormalized (N_OUTCOMES,) float32 probability vector.
    """
    if fatigue_factor >= 1.0:
        return prob_vector.astype(np.float32)

    adjusted = prob_vector.astype(np.float64).copy()

    # Determine how much probability to shift away from OUT_IN_PLAY
    reduction = adjusted[0] * (1.0 - fatigue_factor)

    # Shift OUT_IN_PLAY mass → WALK_HBP (70%) and HOME_RUN (30%)
    adjusted[0] = max(0.0, adjusted[0] - reduction)
    adjusted[2] += reduction * 0.70   # WALK_HBP
    adjusted[6] += reduction * 0.30   # HOME_RUN

    # Renormalize
    total = adjusted.sum()
    if total > 0:
        adjusted /= total

    return adjusted.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-inning matrix builder
# ---------------------------------------------------------------------------

def build_per_inning_prob_matrix(
    staff: PitchingStaffConfig,
    n_innings: int = 9,
    n_batters: int = 9,
    n_outcomes: int = _N_OUTCOMES_DEFAULT,
) -> np.ndarray:
    """Builds the per-inning probability matrix for a pitching staff.

    Returns a 3-D array where each inning slice contains the pitcher-vs-lineup
    probability matrix for that inning.

    Simple mode (default):
        Innings [0, handoff_inning):
            Starter probability vector (fatigue-adjusted), tiled across batters.
        Innings [handoff_inning, n_innings):
            Mean of available bullpen prob vectors (fatigue-adjusted per pitcher),
            tiled across batters.

    The "tiling" (same vector for all 9 batters) is the initial approximation.
    Phase 2 (simple_mode=False) would use matchup-specific probs per batter.

    Args:
        staff: Full pitching staff configuration.
        n_innings: Number of innings to generate (default 9 for regulation).
        n_batters: Number of batters in the lineup (always 9 in MLB).
        n_outcomes: Number of PA outcome classes (default 8 after Mejora 6).

    Returns:
        ndarray of shape ``(n_innings, n_batters, n_outcomes)``, dtype float32.
        Index ``[inning, batter, outcome]`` gives the probability of that outcome
        for that batter facing the inning's pitcher.

    Notes:
        - This function is pure NumPy (no Numba) — overhead < 5ms.
        - Called once per game before the simulation batch, not inside the loop.
    """
    result = np.empty((n_innings, n_batters, n_outcomes), dtype=np.float32)

    # --- Starter probs (fatigue-adjusted) ---
    sp_probs = _apply_fatigue(staff.starter.prob_vector, staff.starter.fatigue_factor)
    sp_tile  = np.tile(sp_probs, (n_batters, 1))  # (9, N_OUTCOMES)

    # --- Bullpen probs (mean of available pitchers) ---
    available_bp = [p for p in staff.bullpen if p.available]
    if available_bp:
        bp_vecs = np.stack(
            [_apply_fatigue(p.prob_vector, p.fatigue_factor) for p in available_bp],
            axis=0,
        )  # (n_bp, N_OUTCOMES)
        bp_mean  = bp_vecs.mean(axis=0).astype(np.float32)
        bp_tile  = np.tile(bp_mean, (n_batters, 1))  # (9, N_OUTCOMES)
    else:
        # No bullpen available (e.g., single-pitcher legacy mode)
        bp_tile = sp_tile

    # Fill per-inning slices
    for inning in range(n_innings):
        if inning < staff.handoff_inning:
            result[inning] = sp_tile
        else:
            result[inning] = bp_tile

    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_staff_config(staff: PitchingStaffConfig, n_outcomes: int = _N_OUTCOMES_DEFAULT) -> list[str]:
    """Validates a ``PitchingStaffConfig`` and returns a list of warning strings.

    Args:
        staff: Staff configuration to validate.
        n_outcomes: Expected number of outcome classes.

    Returns:
        List of warning messages (empty = valid).
    """
    warnings: list[str] = []

    # Check starter
    sp = staff.starter
    if sp.prob_vector.shape != (n_outcomes,):
        warnings.append(
            f"Starter (id={sp.pitcher_id}) prob_vector shape {sp.prob_vector.shape} "
            f"!= ({n_outcomes},)"
        )
    elif abs(sp.prob_vector.sum() - 1.0) > 0.01:
        warnings.append(
            f"Starter (id={sp.pitcher_id}) prob_vector sums to "
            f"{sp.prob_vector.sum():.4f} (expected ~1.0)"
        )

    if not (0.8 <= sp.fatigue_factor <= 1.0):
        warnings.append(
            f"Starter fatigue_factor {sp.fatigue_factor} outside [0.8, 1.0]"
        )

    # Check bullpen
    for i, bp in enumerate(staff.bullpen):
        if bp.prob_vector.shape != (n_outcomes,):
            warnings.append(
                f"Bullpen[{i}] (id={bp.pitcher_id}) prob_vector shape "
                f"{bp.prob_vector.shape} != ({n_outcomes},)"
            )
        if not (0.8 <= bp.fatigue_factor <= 1.0):
            warnings.append(
                f"Bullpen[{i}] (id={bp.pitcher_id}) fatigue_factor "
                f"{bp.fatigue_factor} outside [0.8, 1.0]"
            )

    if staff.handoff_inning < 1:
        warnings.append(
            f"handoff_inning={staff.handoff_inning} means bullpen enters inning 0 "
            "(starter never pitches; unusual)"
        )

    return warnings
