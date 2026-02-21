import numpy as np

from causalis.dgp.multicausaldata import MultiCausalDatasetGenerator


def test_multicausal_tau_is_additive_with_theta():
    n = 2000

    def tau_t1(x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], 0.10, dtype=float)

    def tau_t2(x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], -0.05, dtype=float)

    gen = MultiCausalDatasetGenerator(
        n_treatments=3,
        k=2,
        beta_y=np.array([0.0, 0.0], dtype=float),
        beta_d=np.zeros((3, 2), dtype=float),
        theta=[0.0, 0.25, -0.10],
        tau=[None, tau_t1, tau_t2],
        outcome_type="continuous",
        sigma_y=1.0,
        include_oracle=True,
        seed=123,
    )

    df = gen.generate(n)

    assert np.max(np.abs(df["tau_link_d_1"].to_numpy(dtype=float) - 0.35)) < 1e-12
    assert np.max(np.abs(df["tau_link_d_2"].to_numpy(dtype=float) - (-0.15))) < 1e-12


def test_multicausal_scalar_u_strength_d_changes_m_obs_vs_m():
    n = 4000

    gen_conf = MultiCausalDatasetGenerator(
        n_treatments=3,
        k=2,
        beta_y=np.array([0.0, 0.0], dtype=float),
        beta_d=np.zeros((3, 2), dtype=float),
        alpha_d=[0.0, 0.0, 0.0],
        u_strength_d=1.0,  # scalar now maps to [0, c, c]
        outcome_type="continuous",
        include_oracle=True,
        seed=77,
    )
    df_conf = gen_conf.generate(n)

    diff_t0 = np.mean(np.abs(df_conf["m_d_0"].to_numpy(dtype=float) - df_conf["m_obs_d_0"].to_numpy(dtype=float)))
    assert float(diff_t0) > 1e-3
    assert float(np.std(df_conf["m_obs_d_0"].to_numpy(dtype=float))) > 1e-3

    gen_no_conf = MultiCausalDatasetGenerator(
        n_treatments=3,
        k=2,
        beta_y=np.array([0.0, 0.0], dtype=float),
        beta_d=np.zeros((3, 2), dtype=float),
        alpha_d=[0.0, 0.0, 0.0],
        u_strength_d=0.0,
        outcome_type="continuous",
        include_oracle=True,
        seed=77,
    )
    df_no_conf = gen_no_conf.generate(n)

    diff_t0_no_conf = np.mean(
        np.abs(df_no_conf["m_d_0"].to_numpy(dtype=float) - df_no_conf["m_obs_d_0"].to_numpy(dtype=float))
    )
    assert float(diff_t0_no_conf) < 1e-12
