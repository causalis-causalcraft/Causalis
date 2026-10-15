"""Compare benchmark transfer with DoubleML's independent implementation."""

from types import SimpleNamespace

import numpy as np
import pytest

from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import (
    _calibrate_benchmark_gain_statistics,
)


def _reference_model(sigma2, nu2, theta):
    return SimpleNamespace(
        framework=SimpleNamespace(
            sensitivity_elements={
                "sigma2": np.array([[[sigma2]]], dtype=float),
                "nu2": np.array([[[nu2]]], dtype=float),
            }
        ),
        all_coef=np.array([[theta]], dtype=float),
        _dml_data=SimpleNamespace(y=np.array([-2.0, 0.0, 2.0])),
    )


@pytest.mark.parametrize(
    ("sigma2_long", "sigma2_short", "nu2_long", "nu2_short", "theta_short"),
    [
        (4.0, 5.0, 9.0, 8.0, 2.3),
        (4.0, 5.0, 9.0, 8.0, 1.7),
        (4.0, 5.0, 9.0, 8.0, 10.0),
        (1.0, 1.0, 2.0, 1.5, 2.1),
        (1.0, 1.5, 2.0, 2.0, 2.1),
        (1.0, 1.5, 2.0, 3.0, 1.9),
        (1.0, 0.5, 2.0, 1.5, 2.1),
        (1.0, 3.0, 3.0, 1.0, 2.1),
        (1.0, 1.0, 2.0, 2.0, 2.0),
        (2.0, 4.0, 2.0, 1.0, 2.1),
    ],
)
def test_relative_benchmark_matches_doubleml(
    sigma2_long, sigma2_short, nu2_long, nu2_short, theta_short
):
    # DoubleML is an optional test dependency, not a runtime dependency.
    gain_statistics = pytest.importorskip("doubleml.utils").gain_statistics
    reference = gain_statistics(
        _reference_model(sigma2_long, nu2_long, 2.0),
        _reference_model(sigma2_short, nu2_short, theta_short),
    )
    actual = _calibrate_benchmark_gain_statistics(
        theta_long=2.0,
        theta_short=theta_short,
        sigma2_long=sigma2_long,
        sigma2_short=sigma2_short,
        nu2_long=nu2_long,
        nu2_short=nu2_short,
    )

    for name in ("cf_y", "cf_d", "rho"):
        assert actual[name] == pytest.approx(float(reference[name][0]), abs=1e-12)
    if actual["cf_y"] == 1.0 or actual["cf_d"] == 1.0:
        assert actual["calibration_valid"] is False
        assert actual["strengths_valid"] is False
    else:
        assert actual["calibration_valid"] is True
