"""Tests for model.py — pure numpy, no Ray required."""
import numpy as np
import pytest
from model import (
    relu, softmax, forward, predict_single, generate_data, train, accuracy,
    majority_vote, MLPWeights, NUM_CLASSES, INPUT_DIM, H1, H2,
)


# ── Activation functions ────────────────────────────────────────────────────

def test_relu_zeros_negatives():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    out = relu(x)
    assert np.all(out >= 0)
    assert out[0] == 0.0
    assert out[4] == 3.0


def test_relu_identity_for_positive():
    x = np.array([1.0, 2.0, 100.0])
    assert np.allclose(relu(x), x)


def test_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    probs = softmax(x)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_softmax_all_nonnegative():
    x = np.array([[-100.0, 0.0, 100.0, 50.0]])
    probs = softmax(x)
    assert np.all(probs >= 0)


def test_softmax_batch():
    x = np.random.default_rng(0).normal(size=(10, 4))
    probs = softmax(x)
    assert probs.shape == (10, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ── forward() ────────────────────────────────────────────────────────────────

def _dummy_weights(seed: int = 0) -> MLPWeights:
    rng = np.random.default_rng(seed)
    return MLPWeights(
        W1=rng.normal(size=(INPUT_DIM, H1)),
        b1=np.zeros(H1),
        W2=rng.normal(size=(H1, H2)),
        b2=np.zeros(H2),
        W3=rng.normal(size=(H2, NUM_CLASSES)),
        b3=np.zeros(NUM_CLASSES),
    )


def test_forward_output_shape_single():
    w = _dummy_weights()
    x = np.array([[1.0, 2.0]])
    out = forward(x, w)
    assert out.shape == (1, NUM_CLASSES)


def test_forward_output_shape_batch():
    w = _dummy_weights()
    X = np.random.default_rng(0).normal(size=(20, INPUT_DIM))
    out = forward(X, w)
    assert out.shape == (20, NUM_CLASSES)


def test_forward_probabilities_sum_to_one():
    w = _dummy_weights()
    X = np.random.default_rng(1).normal(size=(15, INPUT_DIM))
    probs = forward(X, w)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_forward_all_probs_nonnegative():
    w = _dummy_weights()
    X = np.random.default_rng(2).normal(size=(10, INPUT_DIM))
    assert np.all(forward(X, w) >= 0)


def test_predict_single_valid_class():
    w = _dummy_weights()
    x = np.array([1.5, -0.5])
    cls = predict_single(x, w)
    assert 0 <= cls < NUM_CLASSES


# ── Data generation ───────────────────────────────────────────────────────────

def test_generate_data_shapes():
    X, y = generate_data(n_per_class=50, seed=0)
    assert X.shape == (200, INPUT_DIM)
    assert y.shape == (200,)


def test_generate_data_class_distribution():
    X, y = generate_data(n_per_class=100, seed=0)
    for cls in range(NUM_CLASSES):
        assert (y == cls).sum() == 100


def test_generate_data_reproducible():
    X1, y1 = generate_data(seed=7)
    X2, y2 = generate_data(seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_generate_data_different_seeds_differ():
    X1, _ = generate_data(seed=1)
    X2, _ = generate_data(seed=2)
    assert not np.array_equal(X1, X2)


def test_generate_data_two_dimensional():
    X, _ = generate_data(n_per_class=20)
    assert X.shape[1] == 2


# ── Training ──────────────────────────────────────────────────────────────────

def test_train_returns_mlp_weights():
    X, y = generate_data(n_per_class=50, seed=0)
    w = train(X, y, seed=0, epochs=10, lr=0.05)
    assert isinstance(w, MLPWeights)


def test_train_weight_shapes():
    X, y = generate_data(n_per_class=30, seed=0)
    w = train(X, y, seed=0, epochs=5, lr=0.05)
    assert w.W1.shape == (INPUT_DIM, H1)
    assert w.b1.shape == (H1,)
    assert w.W2.shape == (H1, H2)
    assert w.b2.shape == (H2,)
    assert w.W3.shape == (H2, NUM_CLASSES)
    assert w.b3.shape == (NUM_CLASSES,)


def test_train_improves_accuracy():
    X, y = generate_data(n_per_class=200, seed=42)
    w_untrained = _dummy_weights(seed=0)
    acc_before = accuracy(X, y, w_untrained)
    w_trained = train(X, y, seed=42, epochs=300, lr=0.05)
    acc_after = accuracy(X, y, w_trained)
    assert acc_after > acc_before


def test_train_achieves_decent_accuracy():
    X, y = generate_data(n_per_class=300, seed=42)
    split = int(0.8 * len(X))
    w = train(X[:split], y[:split], seed=42, epochs=400, lr=0.05)
    acc = accuracy(X[split:], y[split:], w)
    # 4 Gaussian clusters with std=1.3 — should be well above chance (25%)
    assert acc > 0.70, f"Accuracy {acc:.2f} too low — training failed"


def test_train_deterministic_with_same_seed():
    X, y = generate_data(n_per_class=50, seed=0)
    w1 = train(X, y, seed=5, epochs=20, lr=0.05)
    w2 = train(X, y, seed=5, epochs=20, lr=0.05)
    assert np.allclose(w1.W1, w2.W1)


# ── majority_vote() ────────────────────────────────────────────────────────────

def _fake_result(cls: int, seed: int = 0) -> dict:
    """Create a fake worker result with argmax at cls."""
    probs = np.zeros(NUM_CLASSES)
    probs[cls] = 0.9
    probs[(cls + 1) % NUM_CLASSES] = 0.1
    return {"class": cls, "probs": probs, "worker_id": seed}


def test_majority_vote_empty_raises():
    with pytest.raises(ValueError):
        majority_vote([])


def test_majority_vote_single_result():
    r = _fake_result(2)
    out = majority_vote([r])
    assert out["class"] == 2
    assert out["n_voters"] == 1


def test_majority_vote_unanimous():
    results = [_fake_result(1) for _ in range(5)]
    out = majority_vote(results)
    assert out["class"] == 1
    assert out["confidence"] == 1.0


def test_majority_vote_clear_majority():
    results = [_fake_result(0)] * 4 + [_fake_result(2)] * 2
    out = majority_vote(results)
    assert out["class"] == 0


def test_majority_vote_tie_breaks_via_probs():
    # 2 votes for class 0, 2 votes for class 1; class 1 has higher total prob
    r0 = {"class": 0, "probs": np.array([0.6, 0.1, 0.15, 0.15])}
    r1 = {"class": 1, "probs": np.array([0.1, 0.8, 0.05, 0.05])}
    r2 = {"class": 0, "probs": np.array([0.55, 0.2, 0.15, 0.1])}
    r3 = {"class": 1, "probs": np.array([0.05, 0.85, 0.05, 0.05])}
    # Class 1 total prob = 0.8+0.85 = 1.65, class 0 = 0.6+0.55 = 1.15
    out = majority_vote([r0, r1, r2, r3])
    assert out["class"] == 1


def test_majority_vote_vote_counts_correct():
    results = [_fake_result(0)] * 3 + [_fake_result(2)] * 2
    out = majority_vote(results)
    assert out["vote_counts"][0] == 3
    assert out["vote_counts"][2] == 2


def test_majority_vote_n_voters():
    results = [_fake_result(0)] * 5
    out = majority_vote(results)
    assert out["n_voters"] == 5


def test_majority_vote_confidence_full():
    results = [_fake_result(3)] * 4
    out = majority_vote(results)
    assert out["confidence"] == 1.0


def test_majority_vote_confidence_partial():
    results = [_fake_result(0)] * 4 + [_fake_result(1)] * 2
    out = majority_vote(results)
    assert abs(out["confidence"] - 4 / 6) < 1e-9
