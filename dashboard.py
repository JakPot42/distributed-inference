"""Rich terminal dashboard -- rendered via Live in main.py."""
from __future__ import annotations

import numpy as np
from rich import box
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

_ALIVE_LABEL = "[bold green]>> ALIVE[/bold green]"
_DEAD_LABEL  = "[bold red]   DEAD [/bold red]"


def _node_panel(alive_indices: list[int], dead_indices: list[int], n_total: int) -> Panel:
    tbl = Table(show_header=False, box=None, padding=(0, 1))
    for i in range(n_total):
        label = _ALIVE_LABEL if i in alive_indices else _DEAD_LABEL
        tbl.add_row(f"[dim]Worker {i + 1:02d}[/dim]", label)
    alive = len(alive_indices)
    return Panel(
        tbl,
        title=f"[bold cyan]NODES  {alive}/{n_total} alive[/bold cyan]",
        border_style="cyan",
    )


def _stats_panel(
    accuracy: float,
    correct: int,
    total: int,
    avg_latency_ms: float,
    n_alive: int,
    last_vote: dict | None,
) -> Panel:
    acc_pct = accuracy * 100
    color = "green" if acc_pct >= 90 else "yellow" if acc_pct >= 70 else "red"

    tbl = Table(show_header=False, box=None, padding=(0, 1))
    tbl.add_row(
        "Accuracy",
        f"[bold {color}]{acc_pct:.1f}%[/bold {color}]  [dim]({correct}/{total})[/dim]",
    )
    tbl.add_row("Avg latency", f"[cyan]{avg_latency_ms:.1f} ms[/cyan]")
    tbl.add_row("Live voters", f"[white]{n_alive}[/white]")

    if last_vote and last_vote.get("vote_counts"):
        votes_str = "  ".join(
            f"C{cls}:[bold]{cnt}[/bold]"
            for cls, cnt in sorted(last_vote["vote_counts"].items())
        )
        tbl.add_row("Last vote", f"[dim]{votes_str}[/dim]")

    return Panel(tbl, title="[bold magenta]INFERENCE STATS[/bold magenta]", border_style="magenta")


def _event_panel(events: list[str]) -> Panel:
    body = "\n".join(events[-8:]) if events else "[dim]Waiting for events...[/dim]"
    return Panel(body, title="[bold yellow]EVENT LOG[/bold yellow]", border_style="yellow")


def build_layout(
    alive_indices: list[int],
    dead_indices: list[int],
    n_total: int,
    accuracy: float,
    correct: int,
    total: int,
    avg_latency_ms: float,
    last_vote: dict | None,
    events: list[str],
    kill_rate: float,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=6),
    )
    layout["body"].split_row(
        Layout(name="nodes", ratio=1),
        Layout(name="stats", ratio=2),
    )

    layout["header"].update(Panel(
        f"[bold white]*** DISTRIBUTED RESILIENT INFERENCE ***[/bold white]  "
        f"[dim]Replicated shards | Majority voting | Kill rate {kill_rate * 100:.0f}%[/dim]",
        border_style="blue",
    ))
    layout["nodes"].update(_node_panel(alive_indices, dead_indices, n_total))
    layout["stats"].update(_stats_panel(
        accuracy, correct, total, avg_latency_ms, len(alive_indices), last_vote,
    ))
    layout["footer"].update(_event_panel(events))

    return layout
