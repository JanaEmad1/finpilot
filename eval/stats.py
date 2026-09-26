"""Statistics used to decide whether a model is good enough to release.

Everything here is small and dependency-light (numpy + scipy) so it is easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(metric: Callable[[np.ndarray], float], n_items: int, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Estimate:
    """Percentile bootstrap confidence interval.

    `metric(idx)` computes the metric on the items selected by the index array `idx`,
    so the same helper works for accuracy, macro-F1, recall@k, differences, ...
    """
    rng = np.random.default_rng(seed)
    point = metric(np.arange(n_items))
    samples = np.array([metric(rng.integers(0, n_items, n_items)) for _ in range(n_boot)])
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(point), float(low), float(high))


def accuracy_ci(correct: np.ndarray, **kw) -> Estimate:
    correct = np.asarray(correct, dtype=float)
    return bootstrap_ci(lambda idx: correct[idx].mean(), len(correct), **kw)


def paired_difference_ci(correct_a: np.ndarray, correct_b: np.ndarray, **kw) -> Estimate:
    """CI for accuracy(B) - accuracy(A) on the SAME items (paired bootstrap)."""
    diff = np.asarray(correct_b, dtype=float) - np.asarray(correct_a, dtype=float)
    return bootstrap_ci(lambda idx: diff[idx].mean(), len(diff), **kw)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test for two systems evaluated on the same items.

    Only the items where the systems disagree carry information:
      b = A right, B wrong;  c = A wrong, B right.
    Under "no difference", each disagreement is a fair coin flip.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    p = 1.0 if b + c == 0 else stats.binomtest(c, b + c, 0.5).pvalue
    return {"a_only": b, "b_only": c, "p_value": float(p)}


def mcnemar_sample_size(p_discordant: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """How many paired items we need to detect an accuracy difference `effect`,
    given that a fraction `p_discordant` of items get different answers from A and B
    (Connor, 1987 normal approximation)."""
    if not 0 < abs(effect) <= p_discordant <= 1:
        raise ValueError("need 0 < |effect| <= p_discordant <= 1")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = (z_a * np.sqrt(p_discordant) + z_b * np.sqrt(p_discordant - effect**2)) ** 2 / effect**2
    return int(np.ceil(n))


def format_p(p: float) -> str:
    """p-values can underflow to exactly 0.0 in floating point; never print "p = 0"."""
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


# ---------- calibration ----------

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: how far the model's confidence is from its real accuracy, averaged over
    confidence bins (weighted by how many predictions fall in each bin)."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one number T that divides the logits,
    chosen to minimise negative log-likelihood on a VALIDATION set. It changes the
    confidence but never the predicted class."""
    labels = np.asarray(labels)

    def nll(log_t: float) -> float:
        p = softmax(logits, np.exp(log_t))
        return -np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12))

    result = optimize.minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def coverage_curve(confidence: np.ndarray, correct: np.ndarray,
                   thresholds: np.ndarray | None = None) -> list[dict]:
    """For each threshold t: answer automatically only when confidence >= t, hand the rest
    to a human. Coverage = share answered automatically; accuracy = accuracy on those."""
    thresholds = np.linspace(0, 0.95, 20) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        answered = confidence >= t
        rows.append({
            "threshold": float(t),
            "coverage": float(answered.mean()),
            "accuracy": float(correct[answered].mean()) if answered.any() else float("nan"),
        })
    return rows


def pick_threshold(confidence: np.ndarray, correct: np.ndarray, target_accuracy: float) -> float:
    """Lowest threshold whose auto-answered accuracy reaches the target (max coverage)."""
    for row in coverage_curve(confidence, correct, np.linspace(0, 0.99, 100)):
        if row["accuracy"] >= target_accuracy:
            return row["threshold"]
    return 0.99
