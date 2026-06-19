#!/usr/bin/env python
"""
Benchmark: 100 inference passes at 0%, 10%, 20%, 30% node destruction.
Plots accuracy and latency vs. node-loss percentage → benchmark.png.

Usage:
    python benchmark.py [--workers N] [--trials T] [--output PATH]
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import generate_data, train, accuracy as eval_accuracy
from worker import InferenceWorker
from router import InferenceRouter
from fault_controller import FaultController


def _spawn(n: int, weights) -> list[InferenceWorker]:
    workers = [InferenceWorker(i, weights) for i in range(n)]
    for w in workers:
        w.start()
    return workers


def _shutdown(workers: list[InferenceWorker]) -> None:
    for w in workers:
        w.kill()


def run_benchmark(
    n_workers: int = 7,
    n_trials: int = 100,
    kill_rates: list[float] | None = None,
    seed: int = 42,
) -> dict[float, dict]:
    if kill_rates is None:
        kill_rates = [0.0, 0.10, 0.20, 0.30]

    print("Generating data and training MLP...")
    X, y = generate_data(n_per_class=400, seed=seed)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    weights = train(X_train, y_train, seed=seed, epochs=450, lr=0.05)
    print(f"Base (single-model) accuracy: {eval_accuracy(X_test, y_test, weights) * 100:.1f}%\n")

    rng = np.random.default_rng(seed=99)
    results: dict[float, dict] = {}

    for rate in kill_rates:
        print(f"--- Kill rate {rate * 100:.0f}% ---")
        workers = _spawn(n_workers, weights)
        router = InferenceRouter(workers)
        fault_ctrl = FaultController(workers, router=router)

        killed = fault_ctrl.kill_by_rate(rate)
        if killed:
            print(f"  Killed {len(killed)} workers: {[k + 1 for k in killed]}")
        print(f"  Workers alive: {fault_ctrl.n_alive}/{n_workers}")

        latencies: list[float] = []
        idx_arr = rng.choice(len(X_test), size=n_trials, replace=True)

        for idx in idx_arr:
            x = X_test[idx]
            true_label = int(y_test[idx])
            t0 = time.perf_counter()
            try:
                router.infer(x, true_label=true_label)
            except RuntimeError:
                break
            latencies.append((time.perf_counter() - t0) * 1000)

        acc = router.accuracy
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        print(f"  Accuracy: {acc * 100:.1f}%   Avg latency: {avg_lat:.1f} ms\n")

        results[rate] = {
            "accuracy": acc,
            "avg_latency_ms": avg_lat,
            "n_alive": fault_ctrl.n_alive,
            "n_trials": router.total_requests,
        }
        _shutdown(workers)

    return results


def plot_results(results: dict[float, dict], output: str = "benchmark.png") -> None:
    rates = sorted(results)
    pcts = [r * 100 for r in rates]
    accs = [results[r]["accuracy"] * 100 for r in rates]
    lats = [results[r]["avg_latency_ms"] for r in rates]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Distributed Resilient Inference — Benchmark\n"
        f"7-worker majority-vote ensemble, {results[rates[0]]['n_trials']}-trial runs",
        fontsize=12,
    )

    # Accuracy
    ax1.plot(pcts, accs, "o-", color="#1565C0", linewidth=2.5, markersize=9, zorder=3)
    ax1.axhline(90, color="#2E7D32", linestyle="--", alpha=0.7, linewidth=1.5, label="90% threshold")
    below_90 = [a < 90 for a in accs]
    if any(below_90):
        ax1.fill_between(pcts, accs, 90, where=below_90, alpha=0.15, color="#C62828", label="Below 90%")
    for pct, acc in zip(pcts, accs):
        ax1.annotate(f"{acc:.1f}%", (pct, acc), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10, color="#1565C0")
    ax1.set_xlabel("Node Loss (%)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy vs. Node Loss", fontsize=12)
    ax1.set_ylim(0, 108)
    ax1.set_xticks(pcts)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Latency
    ax2.plot(pcts, lats, "o-", color="#E65100", linewidth=2.5, markersize=9)
    for pct, lat in zip(pcts, lats):
        ax2.annotate(f"{lat:.0f} ms", (pct, lat), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10, color="#E65100")
    ax2.set_xlabel("Node Loss (%)", fontsize=11)
    ax2.set_ylabel("Avg Latency (ms)", fontsize=11)
    ax2.set_title("Latency vs. Node Loss", fontsize=12)
    ax2.set_xticks(pcts)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--output", type=str, default="benchmark.png")
    args = p.parse_args()

    results = run_benchmark(n_workers=args.workers, n_trials=args.trials)
    plot_results(results, output=args.output)

    print("\nFull results:")
    print(f"{'Kill %':>7}  {'Accuracy':>9}  {'Latency ms':>11}  {'Alive':>5}")
    for rate in sorted(results):
        r = results[rate]
        print(
            f"{rate * 100:>7.0f}%  "
            f"{r['accuracy'] * 100:>8.1f}%  "
            f"{r['avg_latency_ms']:>11.1f}  "
            f"{r['n_alive']:>5}"
        )


if __name__ == "__main__":
    main()
