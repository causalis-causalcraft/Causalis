import numpy as np
import pandas as pd
import pytest

from causalis.scenarios.multi_unconfoundedness.dgp import generate_multitreatment_gamma_26


@pytest.mark.parametrize("include_oracle", [False, True])
def test_gamma_26_preserves_stable_user_ids_in_dataframe_and_contract(include_oracle):
    kwargs = dict(n=256, seed=7, include_oracle=include_oracle)
    df = generate_multitreatment_gamma_26(**kwargs, return_causal_data=False)
    repeated = generate_multitreatment_gamma_26(**kwargs, return_causal_data=False)
    data = generate_multitreatment_gamma_26(**kwargs, return_causal_data=True)

    assert df["user_id"].is_unique
    assert df["user_id"].notna().all()
    assert df["user_id"].map(lambda value: isinstance(value, str) and bool(value)).all()
    pd.testing.assert_frame_equal(df, repeated)
    assert data.user_id == "user_id"
    assert "user_id" not in data.confounders
    contract_df = data.get_df(include_user_id=True)
    pd.testing.assert_frame_equal(contract_df, df[contract_df.columns], check_dtype=False)


def test_gamma_26_oracle_g_and_cate_columns_are_consistent():
    df = generate_multitreatment_gamma_26(
        n=4000,
        seed=123,
        include_oracle=True,
        return_causal_data=False,
    )

    required = {
        "g_d_0",
        "g_d_1",
        "g_d_2",
        "cate_d_1",
        "cate_d_2",
        "tau_link_d_0",
        "tau_link_d_1",
        "tau_link_d_2",
    }
    assert required.issubset(df.columns)

    cate_t1_expected = df["g_d_1"].to_numpy(dtype=float) - df["g_d_0"].to_numpy(dtype=float)
    cate_t2_expected = df["g_d_2"].to_numpy(dtype=float) - df["g_d_0"].to_numpy(dtype=float)

    assert np.allclose(df["cate_d_1"].to_numpy(dtype=float), cate_t1_expected, atol=1e-12, rtol=0.0)
    assert np.allclose(df["cate_d_2"].to_numpy(dtype=float), cate_t2_expected, atol=1e-12, rtol=0.0)


def test_gamma_26_respects_treatment_ordering_vs_control():
    df = generate_multitreatment_gamma_26(
        n=4000,
        seed=321,
        include_oracle=True,
        return_causal_data=False,
    )

    g0 = df["g_d_0"].to_numpy(dtype=float)
    g1 = df["g_d_1"].to_numpy(dtype=float)
    g2 = df["g_d_2"].to_numpy(dtype=float)
    cate1 = df["cate_d_1"].to_numpy(dtype=float)
    cate2 = df["cate_d_2"].to_numpy(dtype=float)

    assert np.all(g1 < g0)
    assert np.all(g2 > g0)
    assert np.all(cate1 < 0.0)
    assert np.all(cate2 > 0.0)


def test_gamma_26_oracle_atte_is_separated_from_ate_and_shares_stay_calibrated():
    df = generate_multitreatment_gamma_26(
        n=4000,
        seed=42,
        include_oracle=True,
        return_causal_data=False,
    )

    ate_d_1 = float(df["cate_d_1"].mean())
    atte_d_1 = float(df.loc[df["d_1"] == 1, "cate_d_1"].mean())
    ate_d_2 = float(df["cate_d_2"].mean())
    atte_d_2 = float(df.loc[df["d_2"] == 1, "cate_d_2"].mean())

    share_d_0 = float(df["d_0"].mean())
    share_d_1 = float(df["d_1"].mean())
    share_d_2 = float(df["d_2"].mean())

    assert atte_d_1 < ate_d_1 - 0.05
    assert atte_d_2 > ate_d_2 + 0.05

    assert abs(share_d_0 - 0.50) < 0.03
    assert abs(share_d_1 - 0.25) < 0.03
    assert abs(share_d_2 - 0.25) < 0.03
