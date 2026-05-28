"""
Economic Layer — Roadmap 3.2 (Apuestas/DFS beachhead)

Implements EV (Expected Value) vs. market odds and CLV (Closing Line Value)
as the primary economic metrics for the betting/DFS client segment.

Why CLV (not just accuracy):
  - A prediction that beats the CLOSING line is proven to have edge,
    regardless of the outcome of the individual game.
  - Professional bettors are evaluated by CLV, not win rate.
  - CLV > 0 consistently = market is wrong → model has real alpha.

Usage:
    from api.economic_layer import compute_ev, compute_clv, OddsFormat

    # Convert American odds -110 to implied probability
    prob = american_to_implied(odds=-110)  # 0.5238

    # Compute EV given our model's win probability
    ev = compute_ev(model_win_prob=0.60, odds_american=-110, stake=100)
    # ev = 4.55 (positive → value bet)

    # Compute CLV
    clv = compute_clv(open_odds=-110, closing_odds=-130, side='home')
    # clv = 3.5% → we got 14.5% at opening, closing says 11% → edge confirmed
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import math


# ── Odds conversion ────────────────────────────────────────────────────────────

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 100:
        return (odds / 100) + 1.0
    else:
        return (100 / abs(odds)) + 1.0


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (no vig removed)."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability (raw, with vig)."""
    return decimal_to_implied_prob(american_to_decimal(odds))


def remove_vig(implied_home: float, implied_away: float) -> tuple[float, float]:
    """
    Remove vig (overround) from a two-way market.
    Returns fair probabilities that sum to 1.0.
    """
    total = implied_home + implied_away
    return implied_home / total, implied_away / total


def implied_to_american(prob: float) -> float:
    """Convert probability to American odds (no vig)."""
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    else:
        return round(((1 - prob) / prob) * 100)


# ── Expected Value ─────────────────────────────────────────────────────────────

def compute_ev(
    model_win_prob: float,
    odds_decimal: Optional[float] = None,
    odds_american: Optional[float] = None,
    stake: float = 100.0,
) -> dict:
    """
    Compute Expected Value of a bet.

    EV = (P_model × profit_if_win) - (P_model_loss × stake)
       = P × (decimal_odds - 1) × stake - (1 - P) × stake

    Args:
        model_win_prob: Our model's estimated win probability.
        odds_decimal: Decimal odds offered by the book (e.g. 1.91).
        odds_american: American odds (e.g. -110). Either this or decimal required.
        stake: Bet size in currency units.

    Returns:
        {
          "ev": float,              # Expected profit per unit staked
          "ev_pct": float,          # EV as % of stake
          "has_value": bool,        # True if EV > 0
          "model_prob": float,
          "book_implied_prob": float,
          "edge_pct": float,        # model_prob - book_implied_prob
        }
    """
    if odds_decimal is None and odds_american is not None:
        odds_decimal = american_to_decimal(odds_american)
    if odds_decimal is None:
        raise ValueError("Either odds_decimal or odds_american must be provided")

    book_implied = decimal_to_implied_prob(odds_decimal)
    profit_if_win = (odds_decimal - 1) * stake
    ev = model_win_prob * profit_if_win - (1 - model_win_prob) * stake
    edge_pct = round((model_win_prob - book_implied) * 100, 2)

    return {
        "ev": round(ev, 2),
        "ev_pct": round(ev / stake * 100, 2),
        "has_value": ev > 0,
        "model_prob": round(model_win_prob, 4),
        "book_implied_prob": round(book_implied, 4),
        "edge_pct": edge_pct,
    }


# ── Closing Line Value ─────────────────────────────────────────────────────────

def compute_clv(
    open_odds_decimal: Optional[float] = None,
    closing_odds_decimal: Optional[float] = None,
    open_odds_american: Optional[float] = None,
    closing_odds_american: Optional[float] = None,
    side: Literal['home', 'away'] = 'home',
) -> dict:
    """
    Compute Closing Line Value (CLV).

    CLV = implied probability at closing line - implied probability at your bet price.

    Positive CLV → you bet before the market moved against you → edge confirmed.
    Professional bettors maintain CLV > 2% long-term.

    A model with consistently positive CLV is beating sharp money.

    Returns:
        {
          "clv_pct": float,        # CLV in percentage points (positive = good)
          "open_implied": float,
          "closing_implied": float,
          "verdict": str,          # "value" / "break_even" / "no_value"
        }
    """
    if open_odds_decimal is None and open_odds_american is not None:
        open_odds_decimal = american_to_decimal(open_odds_american)
    if closing_odds_decimal is None and closing_odds_american is not None:
        closing_odds_decimal = american_to_decimal(closing_odds_american)
    if open_odds_decimal is None or closing_odds_decimal is None:
        raise ValueError("Opening and closing odds must be provided")

    open_implied = decimal_to_implied_prob(open_odds_decimal)
    closing_implied = decimal_to_implied_prob(closing_odds_decimal)

    # CLV = (closing line implies you should have less value) - (you got more value)
    # Positive = closing line moved in your favor → market agrees you had edge
    clv_pct = round((closing_implied - open_implied) * 100, 2)

    if clv_pct > 2.0:
        verdict = "value"
    elif clv_pct > -1.0:
        verdict = "break_even"
    else:
        verdict = "no_value"

    return {
        "clv_pct": clv_pct,
        "open_implied": round(open_implied, 4),
        "closing_implied": round(closing_implied, 4),
        "verdict": verdict,
    }


# ── Kelly Criterion (bet sizing) ──────────────────────────────────────────────

def kelly_fraction(model_win_prob: float, odds_decimal: float, fraction: float = 0.25) -> dict:
    """
    Compute fractional Kelly bet sizing.

    Full Kelly = (p × b - q) / b  where b = decimal_odds - 1, q = 1 - p
    Fractional Kelly = full_kelly × fraction (conservative; reduces variance)

    Recommended: use 0.20–0.25 fractional Kelly (quarter-Kelly).

    Returns:
        {
          "full_kelly_pct": float,  # % of bankroll to bet (full Kelly)
          "frac_kelly_pct": float,  # % of bankroll (fractional Kelly)
          "fraction": float,
          "has_edge": bool,
        }
    """
    b = odds_decimal - 1.0
    p = model_win_prob
    q = 1.0 - p

    if b <= 0:
        return {"full_kelly_pct": 0.0, "frac_kelly_pct": 0.0, "fraction": fraction, "has_edge": False}

    full_kelly = (p * b - q) / b

    if full_kelly <= 0:
        return {"full_kelly_pct": 0.0, "frac_kelly_pct": 0.0, "fraction": fraction, "has_edge": False}

    frac_kelly = full_kelly * fraction

    return {
        "full_kelly_pct": round(full_kelly * 100, 2),
        "frac_kelly_pct": round(frac_kelly * 100, 2),
        "fraction": fraction,
        "has_edge": True,
    }


# ── Season CLV tracker ────────────────────────────────────────────────────────

@dataclass
class CLVRecord:
    game_pk: int
    game_date: str
    team: str
    model_win_prob: float
    open_odds_american: Optional[float]
    closing_odds_american: Optional[float]
    home_won: Optional[bool]
    clv_pct: Optional[float] = None
    ev_pct: Optional[float] = None


def compute_season_clv_stats(records: list[CLVRecord]) -> dict:
    """
    Aggregate CLV and EV stats over a season.

    Returns key metrics:
      - mean_clv_pct: Average CLV across all bets
      - pct_positive_clv: % of bets with positive CLV
      - theoretical_roi: Sum of EVs / total staked (kelly-weighted or flat)
    """
    valid = [r for r in records if r.clv_pct is not None]
    if not valid:
        return {"n_bets": 0, "mean_clv_pct": None, "pct_positive_clv": None}

    clv_values = [r.clv_pct for r in valid]  # type: ignore
    n = len(clv_values)
    mean_clv = sum(clv_values) / n
    pct_positive = sum(1 for c in clv_values if c > 0) / n * 100

    return {
        "n_bets": n,
        "mean_clv_pct": round(mean_clv, 3),
        "pct_positive_clv": round(pct_positive, 1),
        "median_clv_pct": round(sorted(clv_values)[n // 2], 3),
    }
