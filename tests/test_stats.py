import numpy as np
import pytest

from eval import stats


def test_accuracy_ci_contains_true_value():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.8
    est = stats.accuracy_ci(correct, n_boot=500)
    assert est.low < 0.8 < est.high
    assert est.high - est.low < 0.05  # n=2000 -> roughly +/- 1.8 points


def test_mcnemar_identical_systems():
    correct = np.array([1, 0, 1, 1, 0], dtype=bool)
    assert stats.mcnemar(correct, correct)["p_value"] == 1.0


def test_mcnemar_detects_clear_difference():
    a = np.zeros(100, dtype=bool)
    b = np.ones(100, dtype=bool)
    assert stats.mcnemar(a, b)["p_value"] < 1e-10


def test_sample_size_grows_for_smaller_effects():
    small = stats.mcnemar_sample_size(0.1, 0.01)
    big = stats.mcnemar_sample_size(0.1, 0.05)
    assert small > big > 0
    with pytest.raises(ValueError):
        stats.mcnemar_sample_size(0.01, 0.05)


def test_ece_is_small_for_calibrated_model():
    rng = np.random.default_rng(1)
    n, k = 20000, 5
    probs = rng.dirichlet(np.ones(k) * 0.5, size=n)
    # sample labels FROM the predicted distribution -> perfectly calibrated by construction
    labels = np.array([rng.choice(k, p=p) for p in probs])
    assert stats.expected_calibration_error(probs, labels) < 0.02


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(2)
    n, k, true_t = 5000, 10, 2.5
    logits = rng.normal(0, 3, size=(n, k))
    true_probs = stats.softmax(logits, true_t)
    labels = np.array([rng.choice(k, p=p) for p in true_probs])
    assert abs(stats.fit_temperature(logits, labels) - true_t) < 0.2


def test_pick_threshold_trades_coverage_for_accuracy():
    confidence = np.array([0.95, 0.9, 0.8, 0.6, 0.4, 0.3])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    t = stats.pick_threshold(confidence, correct, target_accuracy=1.0)
    assert 0.6 < t <= 0.8
