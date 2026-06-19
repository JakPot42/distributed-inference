"""
Routes inference requests across alive workers, collects responses in parallel,
majority-votes the result. Dead workers time out and are silently skipped.
"""
from __future__ import annotations

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import majority_vote


class InferenceRouter:
    def __init__(self, workers: list) -> None:
        self._workers = list(workers)
        self._alive: set[int] = set(range(len(workers)))
        self.total_requests: int = 0
        self.correct: int = 0
        self._latencies: list[float] = []

    # ── state ──────────────────────────────────────────────────────────────

    @property
    def n_alive(self) -> int:
        return len(self._alive)

    @property
    def alive_indices(self) -> list[int]:
        return sorted(self._alive)

    def mark_dead(self, worker_idx: int) -> None:
        self._alive.discard(worker_idx)

    def reset_alive(self) -> None:
        self._alive = set(range(len(self._workers)))

    # ── inference ──────────────────────────────────────────────────────────

    def infer(self, x: np.ndarray, true_label: int | None = None) -> dict:
        """
        Fan out to all alive workers simultaneously (parallel threads),
        collect responses within timeout, majority-vote, return result.

        Workers that stop responding are detected via timeout and marked dead.
        Only RuntimeError is raised if zero workers respond.
        """
        if not self._alive:
            raise RuntimeError("No workers alive")

        t0 = time.perf_counter()
        alive_now = list(self._alive)
        req_id = f"r{self.total_requests}"

        # Send requests first (non-blocking)
        for idx in alive_now:
            self._workers[idx].send(req_id, x)

        # Collect responses in parallel — each recv() blocks up to timeout
        results: list[dict] = []

        def _collect(idx: int) -> dict | None:
            r = self._workers[idx].recv(timeout=2.0)
            if r is None and not self._workers[idx].alive:
                self._alive.discard(idx)
            return r

        with ThreadPoolExecutor(max_workers=len(alive_now)) as ex:
            futs = {ex.submit(_collect, idx): idx for idx in alive_now}
            for fut in as_completed(futs, timeout=4.0):
                r = fut.result()
                if r is not None and r.get("req_id") == req_id:
                    results.append(r)

        if not results:
            raise RuntimeError("All workers failed to respond")

        vote = majority_vote(results)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(elapsed_ms)
        self.total_requests += 1

        correct_flag: bool | None = None
        if true_label is not None:
            correct_flag = vote["class"] == true_label
            if correct_flag:
                self.correct += 1

        return {
            "prediction": vote["class"],
            "vote_counts": vote["vote_counts"],
            "confidence": vote["confidence"],
            "n_voters": vote["n_voters"],
            "latency_ms": elapsed_ms,
            "correct": correct_flag,
        }

    # ── stats ──────────────────────────────────────────────────────────────

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_requests if self.total_requests else 0.0

    @property
    def avg_latency_ms(self) -> float:
        recent = self._latencies[-50:]
        return float(np.mean(recent)) if recent else 0.0
