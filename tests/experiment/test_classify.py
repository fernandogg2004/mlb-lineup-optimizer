"""
Unit tests for classify_divergence().

Pure function — no I/O, no fixtures needed. Each test covers one branch of
the divergence taxonomy (order_only / player_swap / full_override) or verifies
the correctness of the slot/player counts.
"""
from src.experiment.experiment_store import classify_divergence

_AI = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# order_only — same 9 players, different batting order
# ---------------------------------------------------------------------------


def test_identical_lineups_are_order_only():
    dtype, slots, players = classify_divergence(_AI, list(_AI))
    assert dtype == "order_only"
    assert slots == 0
    assert players == 0


def test_adjacent_swap_is_order_only():
    manager = [2, 1, 3, 4, 5, 6, 7, 8, 9]  # positions 0 ↔ 1
    dtype, slots, players = classify_divergence(_AI, manager)
    assert dtype == "order_only"
    assert slots == 2
    assert players == 0


def test_cyclic_rotation_is_order_only():
    manager = [2, 3, 4, 5, 6, 7, 8, 9, 1]  # shift everyone up one slot
    dtype, slots, players = classify_divergence(_AI, manager)
    assert dtype == "order_only"
    assert slots == 9  # every position differs
    assert players == 0


def test_leadoff_moved_down_is_order_only():
    # Move player 5 from slot 4 to slot 0; shift 1-4 down by one
    manager = [5, 1, 2, 3, 4, 6, 7, 8, 9]
    dtype, slots, players = classify_divergence(_AI, manager)
    assert dtype == "order_only"
    assert players == 0
    assert slots == 5  # positions 0-4 all differ


# ---------------------------------------------------------------------------
# player_swap — exactly 1 player different
# ---------------------------------------------------------------------------


def test_single_player_replacement_is_player_swap():
    manager = [1, 2, 3, 4, 99, 6, 7, 8, 9]  # slot 4: id=5 → id=99
    dtype, slots, players = classify_divergence(_AI, manager)
    assert dtype == "player_swap"
    assert players == 1
    assert slots >= 1


def test_player_swap_in_same_slot_counts_one_slot():
    manager = [1, 2, 3, 4, 99, 6, 7, 8, 9]  # same position, different player
    _, slots, _ = classify_divergence(_AI, manager)
    assert slots == 1


# ---------------------------------------------------------------------------
# full_override — 2+ players different
# ---------------------------------------------------------------------------


def test_two_replacements_are_full_override():
    manager = [1, 2, 3, 4, 99, 6, 7, 88, 9]  # slots 4 and 7
    dtype, slots, players = classify_divergence(_AI, manager)
    assert dtype == "full_override"
    assert players == 2


def test_entirely_different_lineup_is_full_override():
    manager = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    dtype, slots, players = classify_divergence(_AI, manager)
    assert dtype == "full_override"
    assert players == 9
    assert slots == 9
