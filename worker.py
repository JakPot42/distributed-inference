"""
Worker thread: holds one full replica of the MLP, processes inference requests.
Killing a worker sets its dead flag — it stops consuming from its queue,
causing the router's recv() to time out exactly like a crashed remote node.
"""
from __future__ import annotations

import queue
import threading
import time
import numpy as np

from model import MLPWeights, forward


class InferenceWorker(threading.Thread):
    def __init__(self, worker_id: int, weights: MLPWeights) -> None:
        super().__init__(daemon=True, name=f"worker-{worker_id:02d}")
        self.worker_id = worker_id
        self.weights = weights
        self.req_queue: queue.Queue = queue.Queue()
        self.resp_queue: queue.Queue = queue.Queue()
        self._dead = threading.Event()
        self.call_count: int = 0
        self._latency_sum: float = 0.0

    # ── thread body ────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._dead.is_set():
            try:
                item = self.req_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            req_id, x = item
            t0 = time.perf_counter()
            probs = forward(x.reshape(1, -1), self.weights)[0]
            cls = int(probs.argmax())
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.call_count += 1
            self._latency_sum += elapsed_ms
            if not self._dead.is_set():
                self.resp_queue.put({
                    "req_id": req_id,
                    "class": cls,
                    "probs": probs.copy(),
                    "worker_id": self.worker_id,
                    "latency_ms": elapsed_ms,
                })

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return not self._dead.is_set()

    def kill(self) -> None:
        """Simulate node failure: thread stops consuming requests."""
        self._dead.set()

    def send(self, req_id: str, x: np.ndarray) -> None:
        if self.alive:
            self.req_queue.put((req_id, x))

    def recv(self, timeout: float = 2.0) -> dict | None:
        try:
            return self.resp_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def avg_latency_ms(self) -> float:
        return self._latency_sum / max(self.call_count, 1)
