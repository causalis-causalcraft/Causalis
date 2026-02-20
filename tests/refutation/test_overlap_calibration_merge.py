import numpy as np

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def test_overlap_report_includes_calibration_section_and_flags():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-x))
    d = rng.binomial(1, p)
    data, estimate = make_overlap_data_and_estimate(m_hat=p, d=d)

    report = run_overlap_diagnostics(data, estimate)

    assert "calibration" in report
    cal = report["calibration"]
    assert set(["auc", "brier", "ece", "reliability_table", "recalibration", "flags"]).issubset(cal.keys())

    flags = report["flags"]
    for key in ("calibration_ece", "calibration_slope", "calibration_intercept"):
        assert key in flags
        assert flags[key] in {"GREEN", "YELLOW", "RED", "NA"}
