import numpy as np

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import (
    _ks_statistic,
    run_overlap_diagnostics,
)
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def test_ks_statistic_ties_identical_samples_zero_distance():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    assert _ks_statistic(a, b) == 0.0


def test_ks_from_overlap_report_identical_group_distributions_is_small():
    rng = np.random.default_rng(1)
    n = 400
    m = np.clip(rng.uniform(0.2, 0.8, size=n), 1e-6, 1.0 - 1e-6)
    d = np.array([0, 1] * (n // 2))
    rng.shuffle(d)
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)

    report = run_overlap_diagnostics(data, estimate)

    assert np.isfinite(report["ks"])
    assert 0.0 <= report["ks"] <= 0.2


def test_auc_flip_hint_present_for_anti_predictive_scores():
    n = 300
    d = np.array([0, 1] * (n // 2))
    m = 1.0 - d
    data, estimate = make_overlap_data_and_estimate(m_hat=m, d=d)

    report = run_overlap_diagnostics(data, estimate)

    assert report["flags"]["auc"] == "RED"
    assert report["flags"].get("auc_flip_suspected") == "YELLOW"
