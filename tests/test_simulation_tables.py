"""
test_simulation_tables.py — Invariantes de las tablas de transición base-out.

Audit B2: el corazón del sistema (reglas de avance de corredores y run values)
no tenía regresión. Estos tests fijan las transiciones deterministas conocidas
del baseball para que cualquier refactor que rompa E[R] falle de inmediato.

Índice de outcome (debe coincidir con PAOutcome / N_OUTCOMES):
    0=OUT 1=K 2=BB/HBP 3=1B 4=2B 5=3B 6=HR 7=DP
Estado de bases = bitmask de 3 bits: bit0=1B, bit1=2B, bit2=3B.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.constants import N_OUTCOMES, RUN_VALUES
from src.simulation.simulation_engine import (
    _NEW_BASES_TBL,
    _RUNS_TBL,
    _RUNS_TBL_P,
    _SCENARIO_PROBS,
)

OUT, K, BB, S1, S2, S3, HR, DP = range(8)


def _popcount(b: int) -> int:
    return bin(b).count("1")


def test_run_values_length_matches_n_outcomes():
    assert len(RUN_VALUES) == N_OUTCOMES == 8


@pytest.mark.parametrize("b", range(8))
def test_home_run_clears_bases_and_scores_all_plus_batter(b):
    assert _RUNS_TBL[HR, b] == 1 + _popcount(b)
    assert _NEW_BASES_TBL[HR, b] == 0


@pytest.mark.parametrize("b", range(8))
def test_triple_scores_all_runners_batter_to_third(b):
    assert _RUNS_TBL[S3, b] == _popcount(b)
    assert _NEW_BASES_TBL[S3, b] == 0b100   # solo corredor en 3B


@pytest.mark.parametrize("b", range(8))
def test_out_and_strikeout_preserve_state(b):
    for o in (OUT, K):
        assert _RUNS_TBL[o, b] == 0
        assert _NEW_BASES_TBL[o, b] == b


def test_single_bases_loaded():
    # 3B anota; 2B->3B; 1B->2B; bateador->1B
    assert _RUNS_TBL[S1, 0b111] == 1
    assert _NEW_BASES_TBL[S1, 0b111] == 0b111


def test_single_empty_bases_puts_batter_on_first():
    assert _RUNS_TBL[S1, 0] == 0
    assert _NEW_BASES_TBL[S1, 0] == 0b001


def test_double_runner_on_first_only():
    # 1B->3B; bateador->2B; nadie anota
    assert _RUNS_TBL[S2, 0b001] == 0
    assert _NEW_BASES_TBL[S2, 0b001] == 0b110


def test_double_scores_second_and_third():
    # corredores en 2B+3B anotan (2 carreras); bateador a 2B
    assert _RUNS_TBL[S2, 0b110] == 2
    assert _NEW_BASES_TBL[S2, 0b110] == 0b010


def test_walk_forces_run_only_when_loaded():
    assert _RUNS_TBL[BB, 0b111] == 1
    assert _NEW_BASES_TBL[BB, 0b111] == 0b111
    # bases vacías: el bateador va a 1B, nadie anota
    assert _RUNS_TBL[BB, 0] == 0
    assert _NEW_BASES_TBL[BB, 0] == 0b001
    # corredor solo en 2B: el walk NO lo fuerza
    assert _RUNS_TBL[BB, 0b010] == 0
    assert _NEW_BASES_TBL[BB, 0b010] == 0b011


@pytest.mark.parametrize("b", range(8))
def test_double_play_clears_first_runner_no_runs(b):
    assert _RUNS_TBL[DP, b] == 0
    assert _NEW_BASES_TBL[DP, b] == (b & 0b110)   # limpia bit-0 (corredor en 1B)


def test_probabilistic_scenario_probs_are_normalised():
    # Cada fila (outcome, bases) de escenarios suma ~1 (escenarios válidos) o ~0
    # (estado nunca alcanzado por ese outcome). Nunca un valor intermedio.
    sums = _SCENARIO_PROBS.sum(axis=2)
    ok = np.isclose(sums, 1.0, atol=1e-4) | np.isclose(sums, 0.0, atol=1e-4)
    assert ok.all(), f"filas con suma no normalizada: {sums[~ok]}"


def test_probabilistic_runs_are_nonnegative():
    assert (_RUNS_TBL_P >= 0).all()
