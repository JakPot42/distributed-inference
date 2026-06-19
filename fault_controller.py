"""
Kills worker threads to simulate node failure.
Keeps authoritative alive/dead bookkeeping and notifies the router.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router import InferenceRouter


class FaultController:
    def __init__(self, workers: list, router: "InferenceRouter | None" = None) -> None:
        self._workers = list(workers)
        self._alive: set[int] = set(range(len(workers)))
        self._dead: set[int] = set()
        self._router = router
        self.kill_log: list[dict] = []

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def n_workers(self) -> int:
        return len(self._workers)

    @property
    def n_alive(self) -> int:
        return len(self._alive)

    @property
    def n_dead(self) -> int:
        return len(self._dead)

    @property
    def alive_indices(self) -> list[int]:
        return sorted(self._alive)

    @property
    def dead_indices(self) -> list[int]:
        return sorted(self._dead)

    # ── kill logic ─────────────────────────────────────────────────────────

    def kill_count_for_rate(self, rate: float) -> int:
        """Workers to kill at this rate (0.0–1.0), capped by currently alive count."""
        return max(0, min(self.n_alive, round(self.n_workers * rate)))

    def kill(self, n: int = 1) -> list[int]:
        """Kill n random alive workers. Returns sorted list of killed indices."""
        n = min(n, len(self._alive))
        if n == 0:
            return []

        to_kill = random.sample(sorted(self._alive), n)
        killed: list[int] = []
        for idx in to_kill:
            self._workers[idx].kill()
            self._alive.discard(idx)
            self._dead.add(idx)
            if self._router is not None:
                self._router.mark_dead(idx)
            self.kill_log.append({"killed_idx": idx, "alive_after": len(self._alive)})
            killed.append(idx)

        return sorted(killed)

    def kill_by_rate(self, rate: float) -> list[int]:
        """Kill floor(n_workers × rate) workers."""
        return self.kill(self.kill_count_for_rate(rate))

    def reset(self, new_workers: list | None = None) -> None:
        """
        Restore alive/dead state (for benchmark resets).
        Caller must pass freshly-started workers if re-using after a benchmark run.
        """
        if new_workers is not None:
            self._workers = list(new_workers)
        self._alive = set(range(len(self._workers)))
        self._dead = set()
        self.kill_log.clear()
        if self._router is not None:
            self._router.reset_alive()
