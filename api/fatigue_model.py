"""
Fatigue / Workload Model — Roadmap 3.1

Defines a dynamic penalization factor over the player's marginal E[R] contribution
based on publicly available signals: consecutive games, plate appearances seen,
rest days, and travel (timezone changes).

Design goals:
  - Use ONLY public signals (no biometrics required)
  - Return a multiplicative factor fatigue_factor ∈ [0.80, 1.00]
  - factor < 1.0 means the player is likely to under-perform their season average
  - Validated against backtest (compare model WITH vs WITHOUT fatigue; factor must
    improve Log-Loss to be kept)

Usage:
    from api.fatigue_model import compute_fatigue_factor, FatigueSignals

    signals = FatigueSignals(
        consecutive_games_played=6,
        rest_days_last_7=1,
        pa_last_7_days=28,
        timezone_changes_last_3_days=2,
    )
    factor = compute_fatigue_factor(signals)
    # factor = 0.91 → player expected at ~91% of his season wOBA tonight
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ── Public signal constants ────────────────────────────────────────────────────
# All coefficients are conservative estimates pending backtest validation.
# None of these should change the model output by more than ±5% without
# a confirmed improvement in out-of-sample Log-Loss.

# Typical season PA total ≈ 620; typical PA per 7 days ≈ ~28
_TYPICAL_PA_7D = 28.0
_HIGH_PA_7D_THRESHOLD = 36.0    # ~3 above average → mild fatigue signal
_VERY_HIGH_PA_7D_THRESHOLD = 44.0  # heavy workload

# Rest days: 0 = played yesterday, 1 = one day rest, etc.
_OPTIMAL_REST_DAYS_7D = 2       # 2 days off in past 7 → good recovery

# Consecutive games: 5+ starts accumulating fatigue
_CONSECUTIVE_GAMES_MILD = 5
_CONSECUTIVE_GAMES_HEAVY = 8

# Timezone penalty: each cross-country flight is ≈ 2–3h shift
_TZ_CHANGE_PENALTY_PER_CHANGE = 0.008  # 0.8% per timezone change

# Hard bounds
_FACTOR_MIN = 0.82
_FACTOR_MAX = 1.00


@dataclass
class FatigueSignals:
    """
    Public signals used to estimate player fatigue.

    Attributes:
        consecutive_games_played: Number of consecutive games started without a day off.
        rest_days_last_7: Days the player had off in the last 7 calendar days.
        pa_last_7_days: Plate appearances logged in the last 7 days.
        timezone_changes_last_3_days: Number of significant timezone transitions
            (≥2h change) in the last 3 days. Proxy for travel load.
        pitcher_appearances_7d: For pitchers — appearances in last 7 days
            (not currently used in lineup optimization, kept for completeness).
    """
    consecutive_games_played: int = 0
    rest_days_last_7: int = 2
    pa_last_7_days: float = _TYPICAL_PA_7D
    timezone_changes_last_3_days: int = 0
    pitcher_appearances_7d: int = 0

    # Future signals (not yet used; reserved for v2 of the model)
    sprint_speed_zscore: Optional[float] = None  # from public Statcast sprint speed
    hard_hit_rate_3d_delta: Optional[float] = None  # Delta from 3d vs 30d avg


def compute_fatigue_factor(signals: FatigueSignals) -> float:
    """
    Compute a multiplicative fatigue factor ∈ [FACTOR_MIN, FACTOR_MAX].

    Returns 1.0 if the player is well rested.
    Returns < 1.0 if signals indicate accumulated fatigue.

    All components are additive deductions from 1.0 before clamping.
    """
    deduction = 0.0

    # Component 1: Consecutive games penalty
    consec = signals.consecutive_games_played
    if consec >= _CONSECUTIVE_GAMES_HEAVY:
        deduction += 0.060
    elif consec >= _CONSECUTIVE_GAMES_MILD:
        # Linear: 0.012 per game beyond the 5-game threshold
        excess = consec - _CONSECUTIVE_GAMES_MILD + 1
        deduction += 0.012 * min(excess, 3)  # cap at 3 games excess = 0.036

    # Component 2: Rest days (fewer rest = more fatigue)
    rest = signals.rest_days_last_7
    if rest == 0:
        deduction += 0.040  # zero rest days: significant fatigue risk
    elif rest == 1:
        deduction += 0.020
    # rest >= 2: no deduction

    # Component 3: Plate appearances workload
    pa = signals.pa_last_7_days
    if pa >= _VERY_HIGH_PA_7D_THRESHOLD:
        deduction += 0.030
    elif pa >= _HIGH_PA_7D_THRESHOLD:
        deduction += 0.015

    # Component 4: Travel / timezone changes
    tz_changes = max(0, signals.timezone_changes_last_3_days)
    deduction += tz_changes * _TZ_CHANGE_PENALTY_PER_CHANGE

    # Apply deduction and clamp
    factor = 1.0 - deduction
    return round(max(_FACTOR_MIN, min(_FACTOR_MAX, factor)), 4)


def apply_fatigue_to_woba(woba: float, fatigue_factor: float) -> float:
    """
    Apply fatigue factor to a player's season wOBA to get expected tonight's wOBA.

    woba_tonight ≈ woba_season × fatigue_factor

    Example: wOBA 0.380, factor 0.91 → expected 0.346 tonight
    """
    return round(woba * fatigue_factor, 4)


def describe_fatigue(signals: FatigueSignals) -> dict:
    """
    Return a human-readable dict describing the fatigue assessment for a player.
    Used in the War Room tactical explanation.
    """
    factor = compute_fatigue_factor(signals)
    pct_below = round((1.0 - factor) * 100, 1)

    if factor >= 0.97:
        level = "fresh"
        label = "Descansado"
    elif factor >= 0.93:
        level = "mild"
        label = "Fatiga leve"
    elif factor >= 0.88:
        level = "moderate"
        label = "Fatiga moderada"
    else:
        level = "high"
        label = "Alta carga acumulada"

    signals_summary = []
    if signals.consecutive_games_played >= _CONSECUTIVE_GAMES_MILD:
        signals_summary.append(f"{signals.consecutive_games_played} partidos consecutivos sin descanso")
    if signals.rest_days_last_7 == 0:
        signals_summary.append("ningún día de descanso esta semana")
    elif signals.rest_days_last_7 == 1:
        signals_summary.append("solo 1 día de descanso esta semana")
    if signals.pa_last_7_days >= _HIGH_PA_7D_THRESHOLD:
        signals_summary.append(f"{int(signals.pa_last_7_days)} turnos al bate esta semana (carga alta)")
    if signals.timezone_changes_last_3_days >= 1:
        signals_summary.append(f"{signals.timezone_changes_last_3_days} viaje(s) de zona horaria en 3 días")

    return {
        "fatigue_factor": factor,
        "fatigue_level": level,
        "fatigue_label": label,
        "pct_below_baseline": pct_below,
        "signals": signals_summary,
    }


# ── Fetch public signals from MLB Stats API ────────────────────────────────────

def _load_pitcher_rolling_features(
    pitcher_id: int,
    game_date: str,
    gold_pitcher_rolling_path: Optional[str] = None,
) -> Optional[dict]:
    """Loads pitcher rolling features from the Gold layer (Mejora 3).

    Args:
        pitcher_id: MLB pitcher identifier.
        game_date: Target date in "YYYY-MM-DD" format.
        gold_pitcher_rolling_path: Path to the Gold pitcher rolling Parquet.
            When ``None``, returns ``None`` (fallback to neutral signals).

    Returns:
        Dict with pitcher rolling feature values, or ``None`` if unavailable.
    """
    if gold_pitcher_rolling_path is None:
        return None

    try:
        import polars as pl
        df = (
            pl.scan_parquet(f"{gold_pitcher_rolling_path}/feature_date={game_date}/*.parquet")
            .filter(pl.col("pitcher_id") == pitcher_id)
            .collect()
        )
        if df.is_empty():
            return None
        row = df.row(0, named=True)
        return row
    except Exception:
        return None


async def fetch_player_fatigue_signals(
    player_id: int,
    game_date: str,
    http_client=None,
    gold_pitcher_rolling_path: Optional[str] = None,
) -> FatigueSignals:
    """Fetch fatigue signals for a player.

    For pitchers, this now uses the Gold pitcher rolling feature store
    (Mejora 3) when available, providing data-driven fatigue estimates
    derived from recent workload and velocity degradation signals.

    For batters (or when pitcher rolling data is unavailable), falls back
    to the MLB Stats API-derived signals.

    Args:
        player_id: MLB player identifier.
        game_date: Target date "YYYY-MM-DD".
        http_client: Optional async HTTP client (for batter game-log lookup).
        gold_pitcher_rolling_path: Path to the Gold pitcher rolling store.
            When set, enables data-driven pitcher fatigue.  When ``None``,
            returns neutral signals (factor = 1.0, backwards-compatible).

    Signals derived from pitcher rolling store (Mejora 3):
        - ``days_since_last_app`` → ``rest_days_last_7``
        - ``pitches_7d`` → ``pitcher_appearances_7d`` (pitches / 15 ≈ outings)
        - ``velo_ff_delta_7d_vs_30d`` → if < -0.5 mph, signal extra fatigue

    Validation requirement (Roadmap 3.1): the model WITH fatigue must beat
    the model WITHOUT fatigue in out-of-sample Log-Loss before enabling in
    production.  Until validated, returns neutral signals (factor = 1.0)
    when the rolling store is unavailable.
    """
    # Try pitcher rolling store first (Mejora 3)
    rolling = _load_pitcher_rolling_features(
        player_id, game_date, gold_pitcher_rolling_path
    )

    if rolling is not None:
        # Map pitcher rolling features → FatigueSignals fields
        rest_days = int(rolling.get("days_since_last_app") or 3)
        pitches_7d = int(rolling.get("pitches_7d") or 0)
        # Approximate appearances as pitches / 15 (avg ~15 pitches per outing)
        appearances_7d = max(0, pitches_7d // 15)

        # Velocity degradation penalty: convert to rest_days equivalent
        # A -1 mph velocity drop vs. 30d baseline ≈ equivalent of 1 fewer rest day
        velo_delta = rolling.get("velo_ff_delta_7d_vs_30d")
        effective_rest = rest_days
        if velo_delta is not None and velo_delta < -0.5:
            # Penalize: large velocity drop signals fatigue beyond rest-day count
            effective_rest = max(0, rest_days - 1)

        return FatigueSignals(
            consecutive_games_played=max(0, 7 - max(rest_days, 1)),  # approximate
            rest_days_last_7=effective_rest,
            pa_last_7_days=_TYPICAL_PA_7D,   # batter PA; not applicable for pitchers
            pitcher_appearances_7d=appearances_7d,
        )

    # Fallback: neutral signals (backwards-compatible, no penalty)
    return FatigueSignals(
        consecutive_games_played=0,
        rest_days_last_7=2,
        pa_last_7_days=_TYPICAL_PA_7D,
        timezone_changes_last_3_days=0,
    )


# ── Validation harness ─────────────────────────────────────────────────────────

def validate_fatigue_model_improvement(
    predictions_with_fatigue: list[dict],
    predictions_without_fatigue: list[dict],
) -> dict:
    """
    Compare Log-Loss of predictions WITH vs WITHOUT fatigue factor.
    Both lists must have 'win_probability' and 'home_won' fields.

    Returns:
        {
          "log_loss_with_fatigue": float,
          "log_loss_without_fatigue": float,
          "delta": float,  # negative = fatigue model is BETTER (lower loss)
          "improvement_pct": float,
          "n_games": int,
          "verdict": "improved" | "no_improvement" | "insufficient_data",
        }

    The fatigue model should ONLY be enabled in production when delta < 0
    (i.e., it actually improves out-of-sample performance).
    """
    import math

    def log_loss(preds: list[dict]) -> float:
        eps = 1e-7
        total = 0.0
        n = 0
        for p in preds:
            pw = p.get("win_probability")
            hw = p.get("home_won")
            if pw is None or hw is None:
                continue
            pw = max(eps, min(1 - eps, float(pw)))
            y = int(hw)
            total += -(y * math.log(pw) + (1 - y) * math.log(1 - pw))
            n += 1
        return total / n if n > 0 else float("nan")

    ll_with = log_loss(predictions_with_fatigue)
    ll_without = log_loss(predictions_without_fatigue)
    n = min(len(predictions_with_fatigue), len(predictions_without_fatigue))

    if n < 30:
        verdict = "insufficient_data"
    elif ll_with < ll_without:
        verdict = "improved"
    else:
        verdict = "no_improvement"

    delta = ll_with - ll_without
    improvement_pct = round(-delta / ll_without * 100, 2) if ll_without > 0 else 0.0

    return {
        "log_loss_with_fatigue": round(ll_with, 6),
        "log_loss_without_fatigue": round(ll_without, 6),
        "delta": round(delta, 6),
        "improvement_pct": improvement_pct,
        "n_games": n,
        "verdict": verdict,
    }
