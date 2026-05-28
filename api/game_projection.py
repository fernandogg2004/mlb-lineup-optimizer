"""
api/game_projection.py
======================
Utility: enrich Monte Carlo projection data with confidence intervals.

Called by GET /v1/optimize/{game_pk} in api/main.py.
Addresses Bug 2 — E[R] determinista sin intervalos de confianza.
"""
from __future__ import annotations

import math


def enrich_projection(
    runs_pct: dict,
    er: float,
    win_prob: float,
    n_sims: int,
) -> dict:
    """
    Add std_dev, percentile CI, and win_prob CI to a projection payload.

    Returns a dict of fields to merge (via **) into the /v1/optimize response.
    All new fields are optional extensions — no breaking change to the schema.

    Args:
        runs_pct:  Percentile dict from MonteCarloEngine (keys may be int or str).
        er:        Expected runs (mean).
        win_prob:  Win probability scalar (0–1).
        n_sims:    Number of Monte Carlo simulations performed.

    Returns:
        dict with keys: std_dev, percentile_10, percentile_25, percentile_75,
        percentile_90, win_prob_ci_low, win_prob_ci_high, simulation_n,
        uncertainty_level.
    """

    def _get(key: int | str) -> float | None:
        v = runs_pct.get(str(key)) if str(key) in runs_pct else runs_pct.get(key)
        return float(v) if v is not None else None

    # MC engine generates percentiles at [5,25,50,75,95]; fall back to nearest
    # available neighbour so the UI always has non-null CI bounds.
    p10 = _get(10) or _get(5)
    p25 = _get(25)
    p75 = _get(75)
    p90 = _get(90) or _get(95)

    # std_dev from IQR when available (robust to non-normality in run distributions).
    # IQR / 1.349 ≈ σ for a normal distribution; a good approximation for Poisson-ish data.
    if p25 is not None and p75 is not None:
        std_dev = (p75 - p25) / 1.349
    elif p10 is not None and p90 is not None:
        std_dev = (p90 - p10) / (2 * 1.282)   # P10–P90 span ≈ 2.564σ
    else:
        # Poisson approximation as last resort
        std_dev = math.sqrt(max(er, 0.5))

    std_dev = round(std_dev, 3)

    # Win probability CI via normal approximation of binomial (90% two-sided).
    n = max(n_sims, 1)
    z = 1.645  # z_{0.95} for 90% CI
    p = max(0.001, min(0.999, win_prob))
    margin = z * math.sqrt(p * (1.0 - p) / n)
    win_prob_ci_low  = round(max(0.01, p - margin), 4)
    win_prob_ci_high = round(min(0.99, p + margin), 4)

    # Uncertainty tier for UI colour-coding (WCAG AA: always accompanied by shape/label)
    if std_dev < 0.8:
        uncertainty_level = 'low'     # green  — proyección confiable
    elif std_dev < 1.5:
        uncertainty_level = 'medium'  # amber  — incertidumbre moderada
    else:
        uncertainty_level = 'high'    # red    — alta varianza

    return {
        'std_dev':           std_dev,
        'percentile_10':     p10,
        'percentile_25':     p25,
        'percentile_75':     p75,
        'percentile_90':     p90,
        'win_prob_ci_low':   win_prob_ci_low,
        'win_prob_ci_high':  win_prob_ci_high,
        'simulation_n':      n_sims,
        'uncertainty_level': uncertainty_level,
    }
