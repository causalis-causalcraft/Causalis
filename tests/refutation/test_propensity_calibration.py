import numpy as np

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def _simulate_probs_and_labels(n: int = 4000, seed: int = 123):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.8 * x)))
    y = rng.binomial(1, p)
    return p, y


def test_calibration_is_well_formed_and_finite():
    p, d = _simulate_probs_and_labels(n=3000, seed=11)
    data, estimate = make_overlap_data_and_estimate(m_hat=p, d=d)

    report = run_overlap_diagnostics(data, estimate, n_bins=10)
    cal = report["calibration"]

    assert isinstance(cal, dict)
    for key in ["auc", "brier", "ece", "reliability_table", "recalibration", "flags"]:
        assert key in cal
    assert np.isfinite(cal["ece"])
    assert cal["ece"] >= 0.0
    assert cal["flags"]["ece"] in {"GREEN", "YELLOW", "RED", "NA"}


def test_misscaled_propensity_triggers_slope_flag():
    p, d = _simulate_probs_and_labels(n=4000, seed=21)
    z = np.log(np.clip(p, 1e-12, 1.0 - 1e-12) / (1.0 - np.clip(p, 1e-12, 1.0 - 1e-12)))
    p_bad = 1.0 / (1.0 + np.exp(-0.5 * z))
    data, estimate = make_overlap_data_and_estimate(m_hat=p_bad, d=d)

    report = run_overlap_diagnostics(data, estimate, n_bins=10)
    cal = report["calibration"]

    assert cal["flags"]["slope"] == "RED"
    assert np.isfinite(cal["ece"])
