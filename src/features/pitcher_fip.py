"""
pitcher_fip.py
==============
Computes Fielding Independent Pitching (FIP) for a pitcher from Silver
plate-appearance data.

FIP formula (Tangotiger):
    FIP = (13*HR + 3*(BB + HBP) - 2*K) / IP + cFIP

Where:
    HR  = Home Runs allowed
    BB  = Walks allowed (approximated as all WALK_HBP events)
    HBP = Hit-by-Pitch (included in WALK_HBP class)
    K   = Strikeouts recorded
    IP  = Innings Pitched (estimated from out events in Silver data)
    cFIP = FIP constant ≈ league_ERA - raw_FIP_numerator/IP_total
           Calibrated at 3.10 for 2015-2024 MLB data.

IP estimation from Silver:
    Each out-generating PA contributes:
        STRIKEOUT     → 1 out (1/3 IP)
        OUT_IN_PLAY   → 1 out (1/3 IP)
        DOUBLE_PLAY   → 2 outs (2/3 IP)
    Other outcomes (hits, walks) contribute 0 outs.
    IP ≈ total_outs / 3.0

Usage:
    from src.features.pitcher_fip import compute_pitcher_fip

    # Rolling FIP for the last 30 days
    fip = compute_pitcher_fip(
        pitcher_id=592789,
        silver=silver_df,
        n_games=30,
    )
    print(f"FIP (last 30 games): {fip.fip:.2f}  K/9: {fip.k9:.1f}  BB/9: {fip.bb9:.1f}")
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# pa_outcome_int encoding (must match PAOutcome in model_at_bat.py)
_OUT_IN_PLAY  = 0
_STRIKEOUT    = 1
_WALK_HBP     = 2
_SINGLE       = 3
_DOUBLE       = 4
_TRIPLE       = 5
_HOME_RUN     = 6
_DOUBLE_PLAY  = 7

# FIP constants — importados de src.constants (fuente única de verdad).
from src.constants import (  # noqa: E402
    FIP_CONSTANT as _CFIP_DEFAULT,
    FIP_NEUTRAL  as _FIP_NEUTRAL,
    FIP_FLOOR    as _FIP_FLOOR,
    FIP_CEIL     as _FIP_CEIL,
    FIP_MIN_PA,
    LEAGUE_AVG   as _LEAGUE_AVG_FIP,
)


@dataclass
class PitcherFIPResult:
    """FIP and derived sabermetric rates for one pitcher.

    Attributes:
        pitcher_id: MLB pitcher identifier.
        fip:        Fielding Independent Pitching (ERA-scale).
        ip:         Innings pitched in the sample.
        k9:         Strikeouts per 9 innings.
        bb9:        Walks + HBP per 9 innings.
        hr9:        Home runs per 9 innings.
        k_pct:      Strikeout rate (K / PA).
        bb_pct:     Walk+HBP rate (BB+HBP / PA).
        hr_pct:     Home run rate (HR / PA).
        n_pa:       Total plate appearances in the sample.
        is_neutral: True when n_pa < min_pa_threshold — values are league-average.
    """

    pitcher_id: int
    fip:        float
    ip:         float
    k9:         float
    bb9:        float
    hr9:        float
    k_pct:      float
    bb_pct:     float
    hr_pct:     float
    n_pa:       int
    is_neutral: bool


def compute_pitcher_fip(
    pitcher_id: int,
    silver: pl.DataFrame,
    n_games: int = 30,
    cfip: float = _CFIP_DEFAULT,
    min_pa: int = 50,
) -> PitcherFIPResult:
    """Computes rolling FIP for a pitcher from Silver plate-appearance data.

    Args:
        pitcher_id: MLB pitcher identifier.
        silver:     Full Silver DataFrame (all seasons, all pitchers).
        n_games:    Number of most-recent game appearances to include.
                    Uses game-count windows (consistent with batter rolling features).
        cfip:       FIP constant (default 3.10 for 2015-2024 MLB).
        min_pa:     Minimum PAs required to return real values.
                    Below this threshold, returns neutral league-average FIP.

    Returns:
        PitcherFIPResult with FIP and secondary rate stats.
    """
    neutral = PitcherFIPResult(
        pitcher_id=pitcher_id,
        fip=_FIP_NEUTRAL, ip=0.0,
        k9=7.5, bb9=3.0, hr9=1.2,
        k_pct=0.224, bb_pct=0.083, hr_pct=0.033,
        n_pa=0, is_neutral=True,
    )

    if "pitcher_id" not in silver.columns or "pa_outcome_int" not in silver.columns:
        return neutral

    rows = silver.filter(pl.col("pitcher_id") == pitcher_id)
    if rows.is_empty():
        return neutral

    # Aggregate to game level and take last n_games
    if "game_date" in rows.columns:
        game_dates = (
            rows.select("game_date")
            .unique()
            .sort("game_date", descending=True)
            .head(n_games)["game_date"]
            .to_list()
        )
        rows = rows.filter(pl.col("game_date").is_in(game_dates))

    n_pa = len(rows)
    if n_pa < min_pa:
        return neutral

    outcomes = rows["pa_outcome_int"]
    n_k   = int((outcomes == _STRIKEOUT).sum())
    n_bb  = int((outcomes == _WALK_HBP).sum())
    n_hr  = int((outcomes == _HOME_RUN).sum())
    n_out = int((outcomes == _OUT_IN_PLAY).sum())
    n_dp  = int((outcomes == _DOUBLE_PLAY).sum())

    total_outs = n_k + n_out + 2 * n_dp
    ip = total_outs / 3.0

    if ip < 1.0:
        return neutral

    raw_fip = (13 * n_hr + 3 * n_bb - 2 * n_k) / ip
    fip = float(raw_fip + cfip)
    fip = float(max(_FIP_FLOOR, min(_FIP_CEIL, fip)))

    per_nine = 9.0 / ip
    return PitcherFIPResult(
        pitcher_id=pitcher_id,
        fip=round(fip, 2),
        ip=round(ip, 1),
        k9=round(n_k * per_nine, 2),
        bb9=round(n_bb * per_nine, 2),
        hr9=round(n_hr * per_nine, 2),
        k_pct=round(n_k / n_pa, 4),
        bb_pct=round(n_bb / n_pa, 4),
        hr_pct=round(n_hr / n_pa, 4),
        n_pa=n_pa,
        is_neutral=False,
    )


def compute_pitcher_fip_for_inference(
    pitcher_id: int,
    silver: pl.DataFrame,
    n_games: int = 30,
) -> dict[str, float]:
    """Returns FIP metrics as a flat dict for display / future feature inclusion.

    This dict can be included in the prediction output JSON so managers see
    FIP context alongside E[R] predictions. The values are NOT fed to the
    current model (which was trained without pitcher FIP); they will be included
    in the feature vector at the next model retrain cycle.

    Args:
        pitcher_id: MLB pitcher identifier (may be None for TBD pitchers).
        silver:     Full Silver DataFrame.
        n_games:    Rolling window in game appearances.

    Returns:
        Dict with keys: fip, k9, bb9, hr9, k_pct, bb_pct, hr_pct, fip_sample_pa,
        fip_is_estimated (True when data is insufficient and league-avg used).
    """
    if not pitcher_id:
        return {
            "fip": _FIP_NEUTRAL, "k9": 7.5, "bb9": 3.0, "hr9": 1.2,
            "k_pct": 0.224, "bb_pct": 0.083, "hr_pct": 0.033,
            "fip_sample_pa": 0, "fip_is_estimated": True,
        }
    result = compute_pitcher_fip(pitcher_id, silver, n_games=n_games)
    return {
        "fip":              result.fip,
        "k9":               result.k9,
        "bb9":              result.bb9,
        "hr9":              result.hr9,
        "k_pct":            result.k_pct,
        "bb_pct":           result.bb_pct,
        "hr_pct":           result.hr_pct,
        "fip_sample_pa":    result.n_pa,
        "fip_is_estimated": result.is_neutral,
    }
