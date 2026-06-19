"""
Pure-numpy MLP: train on synthetic 4-class Gaussian data, forward pass, majority vote.
No PyTorch, no sklearn — just numpy. Fast enough to train in <2 seconds.
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple

NUM_CLASSES = 4
INPUT_DIM = 2
H1 = 32
H2 = 16


class MLPWeights(NamedTuple):
    W1: np.ndarray  # (INPUT_DIM, H1)
    b1: np.ndarray  # (H1,)
    W2: np.ndarray  # (H1, H2)
    b2: np.ndarray  # (H2,)
    W3: np.ndarray  # (H2, NUM_CLASSES)
    b3: np.ndarray  # (NUM_CLASSES,)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def forward(x: np.ndarray, w: MLPWeights) -> np.ndarray:
    """x: (..., INPUT_DIM) → probs: (..., NUM_CLASSES). All values ≥0, sum to 1."""
    h1 = relu(x @ w.W1 + w.b1)
    h2 = relu(h1 @ w.W2 + w.b2)
    return softmax(h2 @ w.W3 + w.b3)


def predict_single(x: np.ndarray, w: MLPWeights) -> int:
    probs = forward(x.reshape(1, -1), w)[0]
    return int(probs.argmax())


def generate_data(n_per_class: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    4 Gaussian clusters in 2D (std=1.3 — enough overlap to give ~91% single-model accuracy).
    Centers: (2,2), (-2,2), (-2,-2), (2,-2).
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
    Xs, ys = [], []
    for cls, c in enumerate(centers):
        Xs.append(rng.normal(loc=c, scale=1.3, size=(n_per_class, 2)))
        ys.append(np.full(n_per_class, cls))
    X = np.vstack(Xs)
    y = np.concatenate(ys).astype(int)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    epochs: int = 400,
    lr: float = 0.05,
) -> MLPWeights:
    """Numpy SGD. Converges in <2s on any modern machine."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.3, (INPUT_DIM, H1))
    b1 = np.zeros(H1)
    W2 = rng.normal(0, 0.3, (H1, H2))
    b2 = np.zeros(H2)
    W3 = rng.normal(0, 0.3, (H2, NUM_CLASSES))
    b3 = np.zeros(NUM_CLASSES)

    n = len(X)
    for _ in range(epochs):
        h1 = relu(X @ W1 + b1)
        h2 = relu(h1 @ W2 + b2)
        probs = softmax(h2 @ W3 + b3)

        # Cross-entropy gradient at output
        dL = probs.copy()
        dL[np.arange(n), y] -= 1
        dL /= n

        dW3 = h2.T @ dL;      db3 = dL.sum(0)
        dh2 = (dL @ W3.T) * (h2 > 0)
        dW2 = h1.T @ dh2;     db2 = dh2.sum(0)
        dh1 = (dh2 @ W2.T) * (h1 > 0)
        dW1 = X.T @ dh1;      db1 = dh1.sum(0)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
        W3 -= lr * dW3; b3 -= lr * db3

    return MLPWeights(W1, b1, W2, b2, W3, b3)


def accuracy(X: np.ndarray, y: np.ndarray, w: MLPWeights) -> float:
    preds = forward(X, w).argmax(axis=1)
    return float((preds == y).mean())


def majority_vote(results: list[dict]) -> dict:
    """
    Hard majority vote with soft tie-breaking via probability sums.

    Each result dict must have: {"class": int, "probs": ndarray}
    Returns: {"class": int, "vote_counts": dict[int,int], "confidence": float, "n_voters": int}
    """
    if not results:
        raise ValueError("majority_vote requires at least one result")

    votes: dict[int, int] = {}
    prob_sums = np.zeros(NUM_CLASSES)
    for r in results:
        cls = int(r["class"])
        votes[cls] = votes.get(cls, 0) + 1
        prob_sums += np.asarray(r["probs"])

    max_votes = max(votes.values())
    tied = [cls for cls, cnt in votes.items() if cnt == max_votes]
    winner = int(tied[int(np.argmax(prob_sums[tied]))])

    return {
        "class": winner,
        "vote_counts": votes,
        "confidence": max_votes / len(results),
        "n_voters": len(results),
    }
