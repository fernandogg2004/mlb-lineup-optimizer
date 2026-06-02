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
        pitcher_id:  MLB pitcher identifier.
        fip:         Fielding Independent Pitching (ERA-scale), shrunk toward
                     league average when n_pa < FIP_MIN_PA.
        ip:          Innings pitched in the sample.
        k9:          Strikeouts per 9 innings (shrunk).
        bb9:         Walks + HBP per 9 innings (shrunk).
        hr9:         Home runs per 9 innings (shrunk).
        k_pct:       Strikeout rate (K / PA, shrunk).
        bb_pct:      Walk+HBP rate (BB+HBP / PA, shrunk).
        hr_pct:      Home run rate (HR / PA, shrunk).
        n_pa:        Total plate appearances in the sample.
        is_neutral:  True only when n_pa < 5 (hard minimum — no data at all).
        data_weight: Fraction of real data used (0.0–1.0). 1.0 = full confidence.
    """

    pitcher_id:  int
    fip:         float
    ip:          float
    k9:          float
    bb9:         float
    hr9:         float
    k_pct:       float
    bb_pct:      float
    hr_pct:      float
    n_pa:        int
    is_neutral:  bool
    data_weight: float = 1.0   # shrinkage weight: n_pa / FIP_MIN_PA, capped at 1.0


def compute_pitcher_fip(
    pitcher_id: int,
    silver: pl.DataFrame,
    n_games: int = 30,
    cfip: float = _CFIP_DEFAULT,
    min_pa: int = FIP_MIN_PA,
) -> PitcherFIPResult:
    """Computes rolling FIP for a pitcher from Silver plate-appearance data.

    Uses gradual shrinkage toward league-average when PA count is below min_pa,
    eliminating the cliff-edge that previously returned neutral=4.20 for any
    pitcher with < 50 PA regardless of actual performance.

    Shrinkage weight = min(n_pa / min_pa, 1.0):
      - n_pa=0   → pure league average (is_neutral=True)
      - n_pa=25  → 50% real, 50% league avg (data_weight=0.5)
      - n_pa≥50  → 100% real data (data_weight=1.0)

    Args:
        pitcher_id: MLB pitcher identifier.
        silver:     Full Silver DataFrame (all seasons, all pitchers).
        n_games:    Number of most-recent game appearances to include.
        cfip:       FIP constant (default from src.constants.FIP_CONSTANT).
        min_pa:     PA count for full confidence (default FIP_MIN_PA=50).
                    Below this, values are shrunk toward league average.

    Returns:
        PitcherFIPResult with FIP and secondary rate stats.
    """
    _la = _LEAGUE_AVG_FIP   # league average priors
    neutral = PitcherFIPResult(
        pitcher_id=pitcher_id,
        fip=_FIP_NEUTRAL, ip=0.0,
        k9=round(_la["k_rate"] * 27, 2),
        bb9=round(_la["bb_rate"] * 27, 2),
        hr9=0.9,
        k_pct=_la["k_rate"], bb_pct=_la["bb_rate"], hr_pct=0.033,
        n_pa=0, is_neutral=True, data_weight=0.0,
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

    # Hard minimum: fewer than 5 PA → no signal at all, return pure neutral
    if n_pa < 5:
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
        neutral.n_pa = n_pa
        return neutral

    # Gradual shrinkage: weight blends real data toward league average
    w = min(n_pa / min_pa, 1.0)   # 0.0 → 1.0 as n_pa grows to min_pa

    raw_fip = (13 * n_hr + 3 * n_bb - 2 * n_k) / ip
    fip_real = float(max(_FIP_FLOOR, min(_FIP_CEIL, raw_fip + cfip)))
    fip = w * fip_real + (1 - w) * _FIP_NEUTRAL

    k_pct_real  = n_k  / n_pa
    bb_pct_real = n_bb / n_pa
    hr_pct_real = n_hr / n_pa
    k_pct  = w * k_pct_real  + (1 - w) * _la["k_rate"]
    bb_pct = w * bb_pct_real + (1 - w) * _la["bb_rate"]
    hr_pct = w * hr_pct_real + (1 - w) * 0.033

    per_nine = 9.0 / ip
    return PitcherFIPResult(
        pitcher_id=pitcher_id,
        fip=round(fip, 2),
        ip=round(ip, 1),
        k9=round(k_pct * 27, 2),
        bb9=round(bb_pct * 27, 2),
        hr9=round(hr_pct * 27, 2),
        k_pct=round(k_pct, 4),
        bb_pct=round(bb_pct, 4),
        hr_pct=round(hr_pct, 4),
        n_pa=n_pa,
        is_neutral=(w < 1.0),   # partially shrunk counts as not fully real
        data_weight=round(w, 3),
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
            "fip_sample_pa": 0, "fip_is_estimated": True, "fip_data_weight": 0.0,
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
        "fip_data_weight":  result.data_weight,
    }
