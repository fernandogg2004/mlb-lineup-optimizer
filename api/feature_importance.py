"""
api/feature_importance.py
=========================
Rule-based feature importance for lineup divergences.

Addresses Bug 4 — Black-box: Ausencia de explicabilidad en las decisiones del modelo.

Since the core model is Monte Carlo + LightGBM without a stored SHAP decomposition,
we approximate SHAP-like factor values using known baseball principles and the
player statistics already available in the optimization payload.

The resulting factors are deterministic and explainable — each one references a
concrete metric (platoon split, wOBA differential, slot leverage, etc.).
"""
from __future__ import annotations

import math

# Expected plate appearances per batting-order slot
_PA_SLOT: dict[int, float] = {
    1: 4.9, 2: 4.6, 3: 4.4, 4: 4.1, 5: 3.9,
    6: 3.7, 7: 3.4, 8: 3.2, 9: 3.0,
}

# Typical within-season wOBA standard error (sampling noise for ~500 PA)
_WOBA_SIGMA = 0.020


def ic90_halfwidth(slot: int) -> float:
    """
    IC90 half-width for ΔE[R] at a given batting-order slot.

    Derived from:
        ΔE[R] = ΔwOBA × PA × 0.14
        σ_Δ = sqrt(2) × σ_wOBA   (two independent players)
        IC90 = 1.645 × σ_Δ × PA × 0.14

    Returns the half-width; a difference is statistically significant (90%)
    only when |ΔE[R]| > ic90_halfwidth(slot).
    """
    pa = _PA_SLOT.get(slot, 3.5)
    sigma_delta_woba = math.sqrt(2) * _WOBA_SIGMA
    return round(1.645 * sigma_delta_woba * pa * 0.14, 4)


def _platoon_advantage(bat_hand: str, pitcher_hand: str) -> bool:
    return (bat_hand == 'L' and pitcher_hand == 'R') or \
           (bat_hand == 'R' and pitcher_hand == 'L')


def compute_divergence_factors(
    model_player: dict,
    manager_player: dict,
    slot: int,
    pitcher_hand: str,
    delta_er: float,
) -> list[dict]:
    """
    Return up to 3 factors explaining the ΔE[R] for one lineup divergence.

    Args:
        model_player:   Dict with name, woba, obp, iso, hand.
        manager_player: Dict with name, woba, obp, iso, hand.
        slot:           Batting order slot (1–9).
        pitcher_hand:   Opposing pitcher handedness ('L' or 'R').
        delta_er:       Signed ΔE[R] for this slot (positive = model favoured).

    Returns:
        List[{factor: str, shap_value: float, direction: 'favor_modelo'|'favor_manager'}]
    """
    factors: list[dict] = []
    pa = _PA_SLOT.get(slot, 3.5)
    abs_delta = abs(delta_er)

    m_name = model_player.get('name', '?')
    a_name = manager_player.get('name', '?')
    m_hand = model_player.get('hand', 'R')
    a_hand = manager_player.get('hand', 'R')

    # ── Factor 1: Platoon matchup ─────────────────────────────────────────────
    m_platoon = _platoon_advantage(m_hand, pitcher_hand)
    a_platoon = _platoon_advantage(a_hand, pitcher_hand)
    if m_platoon and not a_platoon:
        factors.append({
            'factor': f'Ventaja platoon: {m_name} ({m_hand}HB) vs lanzador {pitcher_hand}HP',
            'shap_value': round(abs_delta * 0.45 + 0.05, 3),
            'direction': 'favor_modelo',
        })
    elif a_platoon and not m_platoon:
        factors.append({
            'factor': f'Ventaja platoon: {a_name} ({a_hand}HB) vs lanzador {pitcher_hand}HP',
            'shap_value': round(abs_delta * 0.45 + 0.05, 3),
            'direction': 'favor_manager',
        })

    # ── Factor 2: wOBA differential ───────────────────────────────────────────
    m_woba = model_player.get('woba', 0.315)
    a_woba = manager_player.get('woba', 0.315)
    woba_diff = m_woba - a_woba
    if abs(woba_diff) > 0.008:
        factors.append({
            'factor': (
                f'wOBA en slot #{slot}: {m_name} {m_woba:.3f} vs {a_name} {a_woba:.3f}'
                f' (Δ={woba_diff:+.3f})'
            ),
            'shap_value': round(abs(woba_diff) * pa * 0.18, 3),
            'direction': 'favor_modelo' if woba_diff > 0 else 'favor_manager',
        })

    # ── Factor 3: Slot-specific metric (OBP top, ISO middle, PA leverage) ────
    if slot <= 2:
        # OBP critical for table-setters
        m_obp = model_player.get('obp', 0.318)
        a_obp = manager_player.get('obp', 0.318)
        obp_diff = m_obp - a_obp
        if abs(obp_diff) > 0.008:
            factors.append({
                'factor': (
                    f'OBP en slot #{slot} (alta palanca, {pa:.1f} PA/partido):'
                    f' {m_name} {m_obp:.3f} vs {a_name} {a_obp:.3f}'
                ),
                'shap_value': round(abs(obp_diff) * pa * 0.22, 3),
                'direction': 'favor_modelo' if obp_diff > 0 else 'favor_manager',
            })
    elif slot in (3, 4, 5):
        # ISO / power for middle of order
        m_iso = model_player.get('iso', 0.165)
        a_iso = manager_player.get('iso', 0.165)
        iso_diff = m_iso - a_iso
        if abs(iso_diff) > 0.010:
            factors.append({
                'factor': (
                    f'ISO (poder) en slot #{slot} (corazón del orden):'
                    f' {m_name} {m_iso:.3f} vs {a_name} {a_iso:.3f}'
                ),
                'shap_value': round(abs(iso_diff) * pa * 0.20, 3),
                'direction': 'favor_modelo' if iso_diff > 0 else 'favor_manager',
            })
    else:
        # Lower order: raw PA leverage is the main differentiator
        if abs(woba_diff) < 0.010:
            factors.append({
                'factor': (
                    f'Slot #{slot} — palanca baja ({pa:.1f} PA/partido);'
                    f' diferencia de rendimiento marginal (ΔwOBA = {woba_diff:+.3f})'
                ),
                'shap_value': round(abs(delta_er) * 0.20, 3),
                'direction': 'favor_modelo' if woba_diff >= 0 else 'favor_manager',
            })

    # Sort by magnitude, keep top 3
    factors.sort(key=lambda f: f['shap_value'], reverse=True)
    top3 = factors[:3]

    # Guarantee at least one factor
    if not top3:
        top3 = [{
            'factor': (
                f'Diferencia neta de wOBA: {m_name} {m_woba:.3f} vs {a_name} {a_woba:.3f}'
                f' (Δ = {woba_diff:+.3f})'
            ),
            'shap_value': round(max(abs(delta_er), 0.01), 3),
            'direction': 'favor_modelo' if delta_er > 0 else 'favor_manager',
        }]

    return top3


def compute_report_divergences(
    proposed_lineup: list[dict],
    actual_lineup: list[dict],
    pitcher_hand: str = 'R',
    player_stats: dict | None = None,
) -> list[dict]:
    """
    Build the full divergence list for a post-game report.

    Args:
        proposed_lineup: Model's recommended lineup [{order, name, pos}, ...].
        actual_lineup:   Manager's actual lineup   [{order, name, pos, result}, ...].
        pitcher_hand:    Opposing pitcher handedness.
        player_stats:    Optional {name: {woba, obp, iso, hand, ...}} for richer factors.

    Returns:
        List of DivergenceRow dicts compatible with the PostGameReport schema.
    """
    stats = player_stats or {}

    # PA-weighted heuristic for ΔE[R]:
    # ΔwOBA × PA/juego × 0.14 ≈ ΔE[R] per slot
    def delta_er_heuristic(slot: int, m_woba: float, a_woba: float) -> float:
        pa = _PA_SLOT.get(slot, 3.5)
        return round((m_woba - a_woba) * pa * 0.14, 3)

    divergences: list[dict] = []

    for prop, actual in zip(proposed_lineup, actual_lineup):
        slot = prop['order']
        match = prop['name'] == actual['name']

        m_stats = stats.get(prop['name'], {})
        a_stats = stats.get(actual['name'], {})
        m_player = {'name': prop['name'], 'woba': 0.315, 'obp': 0.318,
                    'iso': 0.165, 'hand': 'R', **m_stats}
        a_player = {'name': actual['name'], 'woba': 0.315, 'obp': 0.318,
                    'iso': 0.165, 'hand': 'R', **a_stats}

        if match:
            delta_er = 0.0
            top_factors: list[dict] = []
        else:
            delta_er = delta_er_heuristic(
                slot, m_player['woba'], a_player['woba']
            )
            top_factors = compute_divergence_factors(
                m_player, a_player, slot, pitcher_hand, delta_er
            )

        hw = ic90_halfwidth(slot)
        significant = not match and abs(delta_er) > hw

        divergences.append({
            'slot':            slot,
            'model_player':    prop['name'],
            'model_pos':       prop.get('pos', '—'),
            'manager_player':  actual['name'],
            'manager_pos':     actual.get('pos', '—'),
            'delta_er':        delta_er,
            'match':           match,
            'top_factors':     top_factors,
            # IC90 fields (Roadmap 1.2)
            'ic90_halfwidth':  hw,
            'significant':     significant,
        })

    return divergences
