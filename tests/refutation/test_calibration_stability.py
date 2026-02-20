import numpy as np
import warnings

from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def test_overlap_calibration_no_runtime_warnings_on_extreme_inputs():
    n = 1000
    p = np.zeros(n, dtype=float)
    p[n // 2 :] = 1.0
    d = np.zeros(n, dtype=int)
    d[n // 2 :] = 1
    data, estimate = make_overlap_data_and_estimate(m_hat=p, d=d)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=RuntimeWarning)
        report = run_overlap_diagnostics(data, estimate, n_bins=10)

    runtime_warns = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert len(runtime_warns) == 0

    recal = report["calibration"]["recalibration"]
    assert np.isfinite(recal["intercept"])
    assert np.isfinite(recal["slope"])
