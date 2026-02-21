import pandas as pd
import pytest

from causalis.dgp.causaldata import CausalData
from causalis.dgp.multicausaldata import MultiCausalData
from causalis.shared import confounders_balance


def test_confounders_balance_causaldata_legacy_api():
    df = pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1],
            "outcome": [1.0, 2.0, 3.0, 4.0],
            "x": [1.0, 3.0, 5.0, 7.0],
            "z": [10.0, 20.0, 30.0, 40.0],
        }
    )
    data = CausalData.from_df(
        df,
        treatment="treatment",
        outcome="outcome",
        confounders=["x", "z"],
    )

    balance = confounders_balance(data)

    assert isinstance(balance, pd.DataFrame)
    assert set(balance.columns) == {"confounders", "mean_d_0", "mean_d_1", "abs_diff", "smd", "ks_pvalue"}
    assert set(balance["confounders"]) == {"x", "z"}


def test_confounders_balance_multicausal_pairwise_api():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "d_0": [1, 1, 0, 0, 0, 0],
            "d_1": [0, 0, 1, 1, 0, 0],
            "d_2": [0, 0, 0, 0, 1, 1],
            "x": [1.0, 3.0, 100.0, 100.0, 5.0, 7.0],
            "z": [2.0, 4.0, 200.0, 200.0, 6.0, 8.0],
        }
    )
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d_0", "d_1", "d_2"],
        confounders=["x", "z"],
        control_treatment="d_0",
    )

    balance = confounders_balance(data, "d_0", "d_2")

    row_x = balance.loc[balance["confounders"] == "x"].iloc[0]
    assert row_x["mean_d_0"] == pytest.approx(2.0)
    assert row_x["mean_d_1"] == pytest.approx(6.0)
    assert row_x["abs_diff"] == pytest.approx(4.0)

    row_z = balance.loc[balance["confounders"] == "z"].iloc[0]
    assert row_z["mean_d_0"] == pytest.approx(3.0)
    assert row_z["mean_d_1"] == pytest.approx(7.0)


def test_confounders_balance_multicausal_requires_pairwise_treatments():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "d_0": [1, 0, 1, 0],
            "d_1": [0, 1, 0, 1],
            "x": [10.0, 20.0, 30.0, 40.0],
        }
    )
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d_0", "d_1"],
        confounders=["x"],
        control_treatment="d_0",
    )

    with pytest.raises(ValueError, match="provide two treatment columns"):
        confounders_balance(data)
