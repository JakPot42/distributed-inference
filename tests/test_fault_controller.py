"""
Tests for fault_controller.py — kill logic, state tracking, rate calculations.
Workers are MagicMocks with .kill() method; no threads started.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fault_controller import FaultController


def _mock_worker(i: int = 0) -> MagicMock:
    w = MagicMock()
    w.worker_id = i
    w.kill = MagicMock()
    w.alive = True
    return w


def _make_fc(n: int = 7) -> FaultController:
    return FaultController([_mock_worker(i) for i in range(n)])


# ── initial state ─────────────────────────────────────────────────────────────

def test_initial_n_alive_equals_n_workers():
    assert _make_fc(7).n_alive == 7


def test_initial_n_dead_is_zero():
    assert _make_fc(5).n_dead == 0


def test_initial_alive_indices():
    assert _make_fc(4).alive_indices == [0, 1, 2, 3]


def test_initial_dead_indices_empty():
    assert _make_fc(3).dead_indices == []


def test_n_workers_property():
    assert _make_fc(6).n_workers == 6


# ── kill_count_for_rate() ─────────────────────────────────────────────────────

def test_kill_count_zero_rate():
    assert _make_fc(7).kill_count_for_rate(0.0) == 0


def test_kill_count_10_percent():
    # round(7 * 0.10) = round(0.7) = 1
    assert _make_fc(7).kill_count_for_rate(0.10) == 1


def test_kill_count_30_percent():
    # round(7 * 0.30) = round(2.1) = 2
    assert _make_fc(7).kill_count_for_rate(0.30) == 2


def test_kill_count_100_percent():
    assert _make_fc(7).kill_count_for_rate(1.0) == 7


def test_kill_count_capped_by_alive():
    fc = _make_fc(7)
    fc.kill(5)  # 2 alive remain
    # Requesting 100% of 7 = 7, but only 2 alive → capped at 2
    assert fc.kill_count_for_rate(1.0) == 2


# ── kill() ────────────────────────────────────────────────────────────────────

def test_kill_reduces_alive_count():
    fc = _make_fc(7)
    fc.kill(2)
    assert fc.n_alive == 5


def test_kill_increases_dead_count():
    fc = _make_fc(7)
    fc.kill(3)
    assert fc.n_dead == 3


def test_kill_returns_sorted_list():
    fc = _make_fc(7)
    killed = fc.kill(3)
    assert killed == sorted(killed)


def test_kill_zero_kills_nothing():
    fc = _make_fc(7)
    assert fc.kill(0) == []
    assert fc.n_alive == 7


def test_kill_more_than_alive_capped():
    fc = _make_fc(3)
    killed = fc.kill(100)
    assert len(killed) == 3
    assert fc.n_alive == 0


def test_kill_calls_worker_kill_method():
    workers = [_mock_worker(i) for i in range(5)]
    fc = FaultController(workers)
    fc.kill(2)
    killed_count = sum(1 for w in workers if w.kill.called)
    assert killed_count == 2


def test_kill_updates_dead_indices():
    fc = _make_fc(5)
    killed = fc.kill(2)
    for idx in killed:
        assert idx in fc.dead_indices


def test_kill_updates_kill_log():
    fc = _make_fc(5)
    fc.kill(2)
    assert len(fc.kill_log) == 2
    for entry in fc.kill_log:
        assert "killed_idx" in entry
        assert "alive_after" in entry


def test_kill_sequential_accumulates():
    fc = _make_fc(7)
    fc.kill(2)
    fc.kill(2)
    assert fc.n_dead == 4
    assert fc.n_alive == 3


def test_kill_notifies_router():
    router = MagicMock()
    workers = [_mock_worker(i) for i in range(5)]
    fc = FaultController(workers, router=router)
    killed = fc.kill(2)
    assert router.mark_dead.call_count == len(killed)
    called_indices = {call.args[0] for call in router.mark_dead.call_args_list}
    assert called_indices == set(killed)


# ── kill_by_rate() ────────────────────────────────────────────────────────────

def test_kill_by_rate_zero():
    fc = _make_fc(7)
    assert fc.kill_by_rate(0.0) == []
    assert fc.n_alive == 7


def test_kill_by_rate_30_percent():
    fc = _make_fc(7)
    killed = fc.kill_by_rate(0.30)
    assert len(killed) == 2  # round(7 * 0.3) = 2


def test_kill_by_rate_matches_kill_count():
    fc = _make_fc(7)
    expected = fc.kill_count_for_rate(0.20)
    killed = fc.kill_by_rate(0.20)
    assert len(killed) == expected


# ── reset() ──────────────────────────────────────────────────────────────────

def test_reset_restores_alive_count():
    fc = _make_fc(7)
    fc.kill(4)
    fc.reset([_mock_worker(i) for i in range(7)])
    assert fc.n_alive == 7


def test_reset_clears_dead_set():
    fc = _make_fc(5)
    fc.kill(3)
    fc.reset([_mock_worker(i) for i in range(5)])
    assert fc.n_dead == 0


def test_reset_clears_kill_log():
    fc = _make_fc(5)
    fc.kill(2)
    fc.reset([_mock_worker(i) for i in range(5)])
    assert fc.kill_log == []


def test_reset_without_new_workers():
    fc = _make_fc(4)
    fc.kill(2)
    fc.reset()
    assert fc.n_alive == 4
    assert fc.n_dead == 0


def test_reset_notifies_router():
    router = MagicMock()
    workers = [_mock_worker(i) for i in range(4)]
    fc = FaultController(workers, router=router)
    fc.kill(2)
    fc.reset()
    router.reset_alive.assert_called()
