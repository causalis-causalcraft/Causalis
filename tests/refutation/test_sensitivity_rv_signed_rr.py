import math

import numpy as np
import pytest

from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import compute_bias_aware_ci


class _DummyIRM:
    def __init__(self, theta: float, se: float, elements: dict) -> None:
        self.coef_ = np.array([theta], dtype=float)
        self.se_ = np.array([se], dtype=float)
        self._elements = elements

    def _sensitivity_element_est(self) -> dict:
        return self._elements


def _make_elements(n: int = 5, *, sigma2: float = 1.0, nu2: float = 0.25,
                   rr_val: float = -0.4, m_alpha_val: float = 0.25) -> dict:
    psi = np.zeros(n, dtype=float)
    psi_sigma2 = np.zeros(n, dtype=float)
    psi_nu2 = np.zeros(n, dtype=float)
    rr = np.full(n, rr_val, dtype=float)
    m_alpha = np.full(n, m_alpha_val, dtype=float)
    return {
        "sigma2": sigma2,
        "nu2": nu2,
        "psi_sigma2": psi_sigma2,
        "psi_nu2": psi_nu2,
        "riesz_rep": rr,
        "m_alpha": m_alpha,
        "psi": psi,
    }


def test_rv_formula_equal_strength():
    elements = _make_elements()
    model = _DummyIRM(theta=1.0, se=0.1, elements=elements)

    out = compute_bias_aware_ci(model, cf_y=0.2, cf_d=0.2, rho=1.0, H0=0.0, alpha=0.05)

    denom = 1.0 * math.sqrt(1.0 * 0.25)
    def _rv(delta: float) -> float:
        d2 = (delta / denom) ** 2
        return (math.sqrt(d2 * d2 + 4.0 * d2) - d2) / 2.0

    expected_rv = _rv(1.0)
    z = out["z"]
    expected_rva = _rv(max(1.0 - z * 0.1, 0.0))

    assert out["rv"] == pytest.approx(expected_rv, rel=1e-12, abs=1e-12)
    assert out["rva"] == pytest.approx(expected_rva, rel=1e-12, abs=1e-12)
    # RV is the common strength v at which |rho| * sqrt(sigma2*nu2) * v / sqrt(1 - v) hits delta
    assert denom * out["rv"] / math.sqrt(1.0 - out["rv"]) == pytest.approx(1.0, rel=1e-10)


def test_unsigned_bound_uses_cf_y_directly_and_cf_d_odds():
    model = _DummyIRM(theta=1.0, se=0.1, elements=_make_elements())

    out = compute_bias_aware_ci(model, cf_y=0.2, cf_d=0.2, rho=1.0)

    expected = math.sqrt(1.0 * 0.25) * math.sqrt(0.2 * 0.2 / (1.0 - 0.2))
    assert out["bound_width"] == pytest.approx(expected, rel=1e-12)
    assert out["params"]["cf_y"] == 0.2
    assert out["params"]["cf_d"] == 0.2


def test_binary_deprecated_keywords_use_new_strengths_directly():
    model = _DummyIRM(theta=1.0, se=0.1, elements=_make_elements())
    with pytest.warns(DeprecationWarning) as warnings:
        legacy = compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.3, rho=0.7)
    current = compute_bias_aware_ci(model, cf_y=0.2, cf_d=0.3, rho=0.7)
    assert len(warnings) == 2
    assert legacy["params"] == current["params"]
    assert legacy["bound_width"] == pytest.approx(current["bound_width"])
    assert legacy["bias_aware_ci"] == pytest.approx(current["bias_aware_ci"])


def test_signed_rr_nu2_uses_cf_y_directly_and_cf_d_odds():
    model = _DummyIRM(theta=1.0, se=0.1, elements=_make_elements())
    cf_y, cf_d, rho = 0.2, 0.2, 1.0

    out = compute_bias_aware_ci(model, cf_y=cf_y, cf_d=cf_d, rho=rho, use_signed_rr=True)

    a = math.sqrt(2.0 * 0.25)
    b = -0.4  # signed riesz representer
    k_d = cf_d / (1.0 - cf_d)
    base = a * a * cf_y + b * b * k_d + 2.0 * rho * math.sqrt(cf_y * k_d) * a * b
    assert out["nu2"] == pytest.approx(max(base, 0.0), rel=1e-12)
    assert out["bound_width"] == pytest.approx(math.sqrt(1.0 * max(base, 0.0)), rel=1e-12)
    assert out["params"]["use_signed_rr"] is True
    # RV/RVa are not defined under the signed-rr quadratic form
    assert math.isnan(out["rv"])
    assert math.isnan(out["rva"])


