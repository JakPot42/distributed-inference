#!/usr/bin/env python
"""
Interactive demo: N worker threads, configured kill rate, Rich live dashboard.

Usage:
    python main.py [--workers N] [--kill-rate RATE] [--duration SECS]

Example:
    python main.py --workers 7 --kill-rate 0.3 --duration 40
"""
from __future__ import annotations

import argparse
import time
from rich.live import Live
from rich.console import Console

from model import generate_data, train, accuracy as eval_accuracy
from worker import InferenceWorker
from router import InferenceRouter
from fault_controller import FaultController
from dashboard import build_layout

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed Resilient Inference Demo")
    p.add_argument("--workers", type=int, default=7,
                   help="Number of worker threads (default 7)")
    p.add_argument("--kill-rate", type=float, default=0.3,
                   help="Fraction of workers destroyed over the demo (default 0.3 = 30%%)")
    p.add_argument("--duration", type=int, default=45,
                   help="Demo duration in seconds (default 45)")
    return p.parse_args()


def _split_into_waves(total: int, n_waves: int = 3) -> list[int]:
    if total == 0:
        return [0] * n_waves
    base, rem = divmod(total, n_waves)
    return [base + (1 if i < rem else 0) for i in range(n_waves)]


def main() -> None:
    args = parse_args()

    console.print()
    console.print("[bold blue]*** Distributed Resilient Inference ***[/bold blue]")
    console.print("[dim]Replicated model shards | majority voting | live fault injection[/dim]")
    console.print()

    # ── Train ─────────────────────────────────────────────────────────────
    console.print("[dim]Generating synthetic 4-class dataset and training MLP...[/dim]")
    X, y = generate_data(n_per_class=350, seed=42)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    weights = train(X_train, y_train, seed=42, epochs=450, lr=0.05)
    base_acc = eval_accuracy(X_test, y_test, weights)
    console.print(f"[green]OK Single-model accuracy: {base_acc * 100:.1f}%[/green]")

    # ── Spawn workers ─────────────────────────────────────────────────────
    n = args.workers
    console.print(f"[dim]Spawning {n} InferenceWorker threads...[/dim]")
    workers = [InferenceWorker(i, weights) for i in range(n)]
    for w in workers:
        w.start()

    router = InferenceRouter(workers)
    fault_ctrl = FaultController(workers, router=router)

    # Kill schedule: destroy workers in 3 waves spread across the demo
    kill_rate = args.kill_rate
    n_total_kills = fault_ctrl.kill_count_for_rate(kill_rate)
    wave_sizes = _split_into_waves(n_total_kills, n_waves=3)
    wave_times = [args.duration * frac for frac in (0.25, 0.50, 0.70)]
    wave_fired = [False] * 3

    events: list[str] = []
    last_vote: dict | None = None
    sample_idx = 0
    start = time.time()

    console.print("[bold]Starting live demo -- press Ctrl+C to exit early[/bold]\n")

    try:
        with Live(console=console, refresh_per_second=8, screen=False) as live:
            while time.time() - start < args.duration:
                elapsed = time.time() - start

                # Scheduled fault injection
                for i, (wt, ws) in enumerate(zip(wave_times, wave_sizes)):
                    if not wave_fired[i] and elapsed >= wt and ws > 0 and fault_ctrl.n_alive > 1:
                        killed = fault_ctrl.kill(ws)
                        wave_fired[i] = True
                        if killed:
                            ids = [k + 1 for k in killed]
                            events.append(
                                f"[red][{elapsed:.0f}s] KILLED worker(s) {ids} -- "
                                f"{fault_ctrl.n_alive}/{n} alive[/red]"
                            )

                # Inference pass
                if fault_ctrl.n_alive > 0 and len(X_test) > 0:
                    x = X_test[sample_idx % len(X_test)]
                    true_label = int(y_test[sample_idx % len(X_test)])
                    try:
                        result = router.infer(x, true_label=true_label)
                        last_vote = result
                        if result.get("correct") is False:
                            events.append(
                                f"[yellow][{elapsed:.0f}s] Misclassified "
                                f"pred={result['prediction']} true={true_label}[/yellow]"
                            )
                    except RuntimeError as exc:
                        events.append(f"[red][{elapsed:.0f}s] {exc}[/red]")
                    sample_idx += 1

                live.update(build_layout(
                    alive_indices=fault_ctrl.alive_indices,
                    dead_indices=fault_ctrl.dead_indices,
                    n_total=n,
                    accuracy=router.accuracy,
                    correct=router.correct,
                    total=router.total_requests,
                    avg_latency_ms=router.avg_latency_ms,
                    last_vote=last_vote,
                    events=events,
                    kill_rate=kill_rate,
                ))
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    # ── Final summary ─────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Demo complete.[/bold green]")
    console.print(f"  Final accuracy : [bold]{router.accuracy * 100:.1f}%[/bold]  "
                  f"({router.correct}/{router.total_requests} correct)")
    console.print(f"  Workers alive  : {fault_ctrl.n_alive}/{n}")
    console.print(f"  Avg latency    : {router.avg_latency_ms:.1f} ms")
    console.print()

    for w in workers:
        w.kill()


if __name__ == "__main__":
    main()
