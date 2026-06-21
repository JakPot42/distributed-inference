# Distributed Resilient Inference

Fault-tolerant distributed inference system — 7 worker threads each hold a full neural network replica, majority voting maintains accuracy under node destruction, fault injection kills workers in waves during a live Rich terminal dashboard, and a benchmark suite plots accuracy vs. kill rate across 100 trials.

Built to demonstrate that distributed systems resilience principles apply directly to AI inference workloads.

---

## What It Does

Single-point-of-failure inference is the default. If the model server dies, inference stops. This system demonstrates the alternative: N replicas, majority vote, graceful degradation.

1. **Workers** — 7 threads each hold an independent copy of a pure-numpy MLP (2-layer, 4-class Gaussian classification). Each worker is isolated: it can be killed without affecting others.
2. **Router** — fan-out sends each inference request to all alive workers; collects responses with a timeout; performs majority vote with soft tie-break via probability sums
3. **Fault injection** — `fault_controller.py` kills workers in 3 waves during the live demo (simulates cascading node failure)
4. **Dashboard** — Rich terminal UI at 8 fps shows: node status grid (alive/dead), current accuracy, vote distribution per class, recent event log
5. **Benchmark** — 100 trials × 4 kill rates (0%, 10%, 20%, 30%) → `benchmark.png` accuracy/latency plot

---

## Results

| Kill rate | Workers alive | Accuracy |
|-----------|--------------|---------|
| 0% | 7/7 | ~94% |
| 10% | ~6/7 | ~92% |
| 20% | ~5–6/7 | ~90% |
| 30% | ~4–5/7 | ~86% |

Accuracy degrades gracefully — the system continues operating under 30% node destruction that would crash a single-replica setup.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Workers | Pure-numpy 2-layer MLP (4-class Gaussian data, no PyTorch) |
| Concurrency | `threading.Thread` + `concurrent.futures.ThreadPoolExecutor` |
| Voting | Hard majority vote + soft tie-break (probability sum) |
| Terminal UI | Rich (live 8 fps dashboard) |
| Benchmark | matplotlib (`benchmark.png`) |
| Tests | pytest (75 tests) |
| Python | 3.14 compatible |

---

## Quick Start

```bash
git clone https://github.com/JakPot42/distributed-inference.git
cd distributed-inference
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

Run the live demo (fault injection waves + Rich dashboard):

```bash
# Windows: set PYTHONUTF8=1 first for Rich unicode box characters
$env:PYTHONUTF8 = "1"
python main.py
```

Run the benchmark (100 trials × 4 kill rates, saves benchmark.png):

```bash
python benchmark.py
```

No API key needed — this tool does not call any external service.

---

## Architecture

```
model.py              Pure-numpy MLP: 2-layer (input→32→16→4), ReLU, softmax, Gaussian training data
worker.py             WorkerNode: holds model replica, processes inference requests from queue, killed via Event
router.py             InferenceRouter: fan-out to all alive workers, collect with timeout, majority vote
fault_controller.py   FaultController: kills workers in 3 scheduled waves during demo
dashboard.py          Rich Live display: node status grid, accuracy meter, vote bars, event log (8 fps)
main.py               Demo entrypoint: spawn 7 workers, router, fault controller, dashboard
benchmark.py          100-trial benchmark across 0/10/20/30% kill rates, matplotlib plot
```

---

## Key Architecture Decisions

**Why threading.Thread instead of Ray:**
Ray is the standard library for distributed Python inference. Ray does not support Python 3.14 as of this build date. `threading.Thread` + `ThreadPoolExecutor` is architecturally identical for this use case: each worker is an isolated unit that can be killed without affecting others. The failure mode (`kill()` sets a dead Event → thread stops consuming queue → router `recv()` times out → marks dead) exactly replicates a crashed remote node.

**Why pure-numpy MLP:**
The demonstration is about the distribution system, not the model. A 2-layer numpy MLP on 4-class Gaussian data is fast, zero-dependency, and still demonstrates the accuracy-under-failure behavior cleanly. Adding PyTorch would add a 500 MB dependency for no additional insight about the resilience system.

**Why 7 workers:**
Seven gives a strong majority (4/7 minimum) while fitting comfortably in a terminal dashboard. The threshold is flexible via config — 5 or 9 workers show the same behavior with adjusted accuracy curves.

**Hard majority + soft tie-break:**
With even numbers of alive workers after kills, pure hard majority can tie. The soft tie-break sums raw probability vectors from each worker's softmax output and selects the class with the highest aggregate probability. This is more principled than random tie-breaking and is resistant to a single miscalibrated worker skewing the result.

---

## Honest Limitations

- Workers run as threads, not separate processes or machines. This demonstrates the voting and fault-tolerance logic, not physical network failure.
- The MLP is trained on synthetic Gaussian data — the accuracy numbers reflect this distribution, not a real benchmark dataset.
- The benchmark saves `benchmark.png` to the working directory; large numbers of trials take a few minutes.
- Rich unicode box characters require `PYTHONUTF8=1` on Windows.

---

## Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
# 75 passed
```

Covers: MLP forward pass and training, worker lifecycle (alive/dead state), router fan-out and timeout handling, majority vote correctness (including tie-break), fault controller wave scheduling, accuracy computation, benchmark data structure.

---

*CLI only — no web interface, no external API calls. Pure Python demonstration of distributed inference resilience.*
