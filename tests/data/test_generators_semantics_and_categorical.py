import numpy as np

from causalis.dgp import CausalDatasetGenerator


def test_continuous_g_excludes_U_when_u_strength_y_nonzero():
    # Continuous outcomes: g0/g1 should exclude U (mean-zero unobserved)
    gen = CausalDatasetGenerator(
        theta=0.7,
        alpha_y=0.1,
        alpha_d=0.0,
        outcome_type="continuous",
        sigma_y=1.0,
        k=3,
        seed=123,
        u_strength_y=1.5,
        beta_y=np.array([0.2, -0.1, 0.3]),
    )
    n = 1000
    df = gen.generate(n)

    # Recompute oracle using generator's oracle_nuisance
    m_fn, g0_fn, g1_fn = gen.oracle_nuisance()
    X = df[[c for c in df.columns if c.startswith("x")]].to_numpy()
    g0_row = np.array([g0_fn(x) for x in X])
    g1_row = np.array([g1_fn(x) for x in X])

    assert np.allclose(df["g0"].to_numpy(), g0_row, atol=1e-3)
    assert np.allclose(df["g1"].to_numpy(), g1_row, atol=1e-3)


def test_m_and_mobs_semantics():
    # When u_strength_d != 0, m (marginal) and m_obs (realized) should generally differ per-row
    gen = CausalDatasetGenerator(
        theta=0.0,
        outcome_type="binary",
        sigma_y=1.0,
        u_strength_d=1.2,
        k=2,
        seed=42,
    )
    df = gen.generate(8000)

    assert "m" in df.columns and "m_obs" in df.columns

    mobs_mean = float(df["m_obs"].mean())
    d_rate = float(df["d"].mean())
    # Realized propensity mean should match empirical treatment rate
    assert abs(mobs_mean - d_rate) < 2e-2

    # The arrays should not be (near-)identical when u affects treatment
    assert not np.allclose(df["m"].to_numpy(), df["m_obs"].to_numpy(), atol=1e-3)

