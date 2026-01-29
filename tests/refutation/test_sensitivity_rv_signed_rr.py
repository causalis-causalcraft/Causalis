import math

import numpy as np
import pytest

from causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity import compute_bias_aware_ci


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

    out = compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.2, rho=1.0, H0=0.0, alpha=0.05)

    denom = 1.0 * math.sqrt(1.0 * 0.25)
    D = 1.0 / denom
    expected_rv = D / (1.0 + D)

    z = out["z"]
    delta_a = max(1.0 - z * 0.1, 0.0)
    Da = delta_a / denom
    expected_rva = Da / (1.0 + Da)

    assert out["rv"] == pytest.approx(expected_rv, rel=1e-12, abs=1e-12)
    assert out["rva"] == pytest.approx(expected_rva, rel=1e-12, abs=1e-12)


