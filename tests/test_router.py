"""
Tests for router.py — state management and inference logic.
Workers are MagicMocks: .send(), .recv(), .alive, .kill() mocked in-process.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from router import InferenceRouter
from model import NUM_CLASSES


def _fake_result(cls: int, req_id: str = "r0", worker_id: int = 0) -> dict:
    probs = np.zeros(NUM_CLASSES)
    probs[cls] = 0.9
    probs[(cls + 1) % NUM_CLASSES] = 0.1
    return {"req_id": req_id, "class": cls, "probs": probs,
            "worker_id": worker_id, "latency_ms": 1.0}


def _mock_worker(i: int = 0, result_cls: int = 0) -> MagicMock:
    w = MagicMock()
    w.alive = True
    w.worker_id = i
    w.send = MagicMock()
    w.recv = MagicMock(return_value=_fake_result(result_cls, "r0", i))
    w.kill = MagicMock()
    return w


# ── alive state ───────────────────────────────────────────────────────────────

def test_initial_n_alive_equals_n_workers():
    workers = [_mock_worker(i) for i in range(5)]
    router = InferenceRouter(workers)
    assert router.n_alive == 5


def test_initial_alive_indices():
    workers = [_mock_worker(i) for i in range(4)]
    router = InferenceRouter(workers)
    assert router.alive_indices == [0, 1, 2, 3]


def test_mark_dead_removes_from_alive():
    workers = [_mock_worker(i) for i in range(5)]
    router = InferenceRouter(workers)
    router.mark_dead(2)
    assert 2 not in router.alive_indices
    assert router.n_alive == 4


def test_mark_dead_idempotent():
    workers = [_mock_worker(i) for i in range(3)]
    router = InferenceRouter(workers)
    router.mark_dead(1)
    router.mark_dead(1)
    assert router.n_alive == 2


def test_reset_alive_restores_all():
    workers = [_mock_worker(i) for i in range(5)]
    router = InferenceRouter(workers)
    router.mark_dead(0)
    router.mark_dead(3)
    router.reset_alive()
    assert router.n_alive == 5
    assert router.alive_indices == [0, 1, 2, 3, 4]


# ── stats ─────────────────────────────────────────────────────────────────────

def test_accuracy_zero_requests():
    router = InferenceRouter([_mock_worker()])
    assert router.accuracy == 0.0


def test_accuracy_all_correct():
    router = InferenceRouter([_mock_worker()])
    router.total_requests = 10
    router.correct = 10
    assert router.accuracy == 1.0


def test_accuracy_partial():
    router = InferenceRouter([_mock_worker()])
    router.total_requests = 20
    router.correct = 15
    assert router.accuracy == 0.75


def test_avg_latency_empty():
    router = InferenceRouter([_mock_worker()])
    assert router.avg_latency_ms == 0.0


def test_avg_latency_computed():
    router = InferenceRouter([_mock_worker()])
    router._latencies = [10.0, 20.0, 30.0]
    assert abs(router.avg_latency_ms - 20.0) < 0.01


# ── infer() ───────────────────────────────────────────────────────────────────

def test_infer_no_alive_workers_raises():
    workers = [_mock_worker(i) for i in range(3)]
    router = InferenceRouter(workers)
    for i in range(3):
        router.mark_dead(i)
    with pytest.raises(RuntimeError, match="No workers alive"):
        router.infer(np.array([1.0, 0.0]))


def test_infer_returns_prediction_key():
    workers = [_mock_worker(i, result_cls=1) for i in range(3)]
    # Each recv returns result with req_id="r0" for request 0
    router = InferenceRouter(workers)
    out = router.infer(np.array([1.0, 0.0]))
    assert "prediction" in out


def test_infer_majority_vote_applied():
    # 4 workers say class 2, 1 says class 0 → should pick 2
    workers = [_mock_worker(i, result_cls=2) for i in range(4)] + [_mock_worker(4, result_cls=0)]
    for i, w in enumerate(workers):
        w.recv.return_value = _fake_result(2 if i < 4 else 0, "r0", i)
    router = InferenceRouter(workers)
    out = router.infer(np.array([1.0, 0.0]))
    assert out["prediction"] == 2


def test_infer_tracks_correct_count():
    workers = [_mock_worker(i, result_cls=1) for i in range(3)]
    for i, w in enumerate(workers):
        w.recv.return_value = _fake_result(1, "r0", i)
    router = InferenceRouter(workers)
    out = router.infer(np.array([0.0, 1.0]), true_label=1)
    assert out["correct"] is True
    assert router.correct == 1

    for i, w in enumerate(workers):
        w.recv.return_value = _fake_result(1, "r1", i)
    out2 = router.infer(np.array([0.0, 1.0]), true_label=0)
    assert out2["correct"] is False
    assert router.correct == 1  # didn't increase
    assert router.total_requests == 2


def test_infer_records_latency():
    workers = [_mock_worker(i) for i in range(2)]
    for i, w in enumerate(workers):
        w.recv.return_value = _fake_result(0, "r0", i)
    router = InferenceRouter(workers)
    router.infer(np.array([0.0, 1.0]))
    assert len(router._latencies) == 1
    assert router._latencies[0] >= 0


def test_infer_dead_worker_timeout_marks_dead():
    """Worker returning None (timeout) and alive=False → removed from alive set."""
    workers = [_mock_worker(i) for i in range(3)]
    # Worker 1 is dead: recv returns None, alive=False
    workers[1].recv.return_value = None
    workers[1].alive = False
    for i in [0, 2]:
        workers[i].recv.return_value = _fake_result(0, "r0", i)

    router = InferenceRouter(workers)
    out = router.infer(np.array([1.0, 0.0]))

    # 1 should be marked dead
    assert 1 not in router.alive_indices
    # Still got a result from workers 0 and 2
    assert out["n_voters"] >= 1


def test_infer_n_voters_in_result():
    workers = [_mock_worker(i) for i in range(4)]
    for i, w in enumerate(workers):
        w.recv.return_value = _fake_result(0, "r0", i)
    router = InferenceRouter(workers)
    out = router.infer(np.array([1.0, 1.0]))
    assert out["n_voters"] == 4


def test_infer_all_workers_dead_raises():
    workers = [_mock_worker(i) for i in range(3)]
    for w in workers:
        w.recv.return_value = None
        w.alive = False
    router = InferenceRouter(workers)
    with pytest.raises(RuntimeError):
        router.infer(np.array([1.0, 0.0]))
