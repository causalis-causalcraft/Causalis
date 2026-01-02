import numpy as np


from causalis.data.dgps import CausalDatasetGenerator


def test_target_d_rate_calibration_is_conditional_on_u_draw_when_u_strength_d_nonzero():
    """Regression test for documented behavior.

    When u_strength_d != 0, the intercept calibration uses the realized U draw (targets mean m_obs).
    We verify this by comparing the calibrated alpha_d under U=0 vs a non-degenerate U.
    """

    n = 4000
    target = 0.2

    gen = CausalDatasetGenerator(
        k=0,
        u_strength_d=1.5,
        outcome_type="continuous",
        seed=0,
    )

    # No observed confounders
    X = np.zeros((n, 0), dtype=float)
    U0 = np.zeros(n, dtype=float)
    U1 = np.random.default_rng(123).normal(size=n)

    a0 = float(gen._calibrate_alpha_d(X, U0, target))
    a1 = float(gen._calibrate_alpha_d(X, U1, target))

    # With U==0, the mixture collapses and the solution is the plain logit.
    logit_target = float(np.log(target / (1.0 - target)))
    assert abs(a0 - logit_target) < 1e-5

    # With non-degenerate U and u_strength_d != 0, the required intercept should change.
    assert abs(a1 - a0) > 1e-3
