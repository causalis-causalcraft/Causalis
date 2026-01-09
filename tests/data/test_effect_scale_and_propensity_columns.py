import numpy as np
import pytest


from causalis.data.dgps import CausalDatasetGenerator


@pytest.mark.parametrize(
    "outcome_type,alpha_y,theta",
    [
        ("binary", -0.4, 0.8),
        ("poisson", 0.2, 0.6),
    ],
)
def test_tau_link_constant_can_imply_heterogeneous_cate_for_binary_and_poisson(outcome_type, alpha_y, theta):
    """For binary/poisson, cate=g1-g0 is on the natural scale, so it can vary with baseline.

    Even with constant structural effect (tau_link), baseline risk/mean varies with X, making
    risk/mean differences heterogeneous.
    """

    gen = CausalDatasetGenerator(
        theta=theta,
        tau=None,
        k=3,
        beta_y=np.array([0.9, -0.6, 0.4]),
        beta_d=np.array([0.0, 0.0, 0.0]),
        g_y=None,
        g_d=None,
        alpha_y=alpha_y,
        alpha_d=0.0,
        outcome_type=outcome_type,
        u_strength_d=0.0,
        u_strength_y=0.0,
        seed=123,
    )

    df = gen.generate(5000)

    tau_link = df["tau_link"].to_numpy(dtype=float)
    cate = df["cate"].to_numpy(dtype=float)

    # tau_link should be constant when tau is None (theta constant)
    assert np.max(np.abs(tau_link - float(theta))) < 1e-12
    assert float(np.std(tau_link)) < 1e-12

    # cate should vary with baseline (X) for binary/poisson
    assert float(np.std(cate)) > 1e-4


def test_m_column_means_p_d_eq_1_given_x_not_including_u():
    """m should follow the common convention e(X)=P(D=1|X).

    When u_strength_d != 0, m_obs depends on U while m is marginalized over U. They should differ.
    """

    gen = CausalDatasetGenerator(
        theta=0.0,
        tau=None,
        k=2,
        beta_d=np.array([1.2, -0.8]),
        beta_y=np.array([0.5, 0.1]),
        alpha_d=0.1,
        alpha_y=0.0,
        outcome_type="continuous",
        u_strength_d=1.0,
        u_strength_y=0.0,
        seed=999,
    )

    df = gen.generate(3000)
    m = df["m"].to_numpy(dtype=float)
    m_obs = df["m_obs"].to_numpy(dtype=float)

    assert float(np.mean(np.abs(m - m_obs))) > 1e-3
