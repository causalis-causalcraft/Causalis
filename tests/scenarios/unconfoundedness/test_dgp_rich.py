import numpy as np

from causalis.scenarios.unconfoundedness.dgp import generate_obs_hte_26_rich


def test_generate_obs_hte_26_rich_oracle_cate_consistency() -> None:
    df = generate_obs_hte_26_rich(
        n=5000,
        seed=19,
        include_oracle=True,
        return_causal_data=False,
    )

    np.testing.assert_allclose(
        df["cate"].to_numpy(),
        (df["g1"] - df["g0"]).to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    # Tweedie tau_link is on the link scale and should not be reused as natural-scale CATE.
    equal_fraction = np.isclose(
        df["cate"].to_numpy(),
        df["tau_link"].to_numpy(),
        atol=1e-6,
        rtol=1e-6,
    ).mean()
    assert equal_fraction < 0.01


def test_generate_obs_hte_26_rich_observed_gap_negative_but_atte_positive() -> None:
    df = generate_obs_hte_26_rich(
        n=20000,
        seed=42,
        include_oracle=True,
        return_causal_data=False,
    )

    treated = df[df["d"] == 1]
    assert len(treated) > 0

    # Keep ATTE positive while allowing adverse selection in observed outcomes.
    atte = float(treated["cate"].mean())
    assert atte > 0.0

    y_treated = float(treated["y"].mean())
    y_control = float(df.loc[df["d"] == 0, "y"].mean())
    assert y_treated < y_control

    np.testing.assert_allclose(
        df["cate"].to_numpy(),
        (df["g1"] - df["g0"]).to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )
