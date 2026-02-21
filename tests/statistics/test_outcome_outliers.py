import pandas as pd
import pytest

from causalis.dgp.causaldata import CausalData
from causalis.dgp.multicausaldata import MultiCausalData
from causalis.shared import outcome_outliers


def _make_data():
    df = pd.DataFrame({
        "treatment": [0] * 6 + [1] * 6,
        "outcome": [1, 1, 1, 1, 1, 100, 2, 2, 2, 2, 2, 2],
        "confounder": list(range(12)),
    })
    return CausalData.from_df(
        df,
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder"],
    )


def test_outcome_outliers_iqr():
    data = _make_data()

    summary = outcome_outliers(data)
    row0 = summary.loc[summary["treatment"] == 0].iloc[0]
    row1 = summary.loc[summary["treatment"] == 1].iloc[0]

    assert row0["outlier_count"] == 1
    assert bool(row0["has_outliers"]) is True
    assert row1["outlier_count"] == 0
    assert bool(row1["has_outliers"]) is False
    assert row0["outlier_rate"] == pytest.approx(1 / 6)


def test_outcome_outliers_return_rows():
    data = _make_data()

    summary, outliers = outcome_outliers(data, return_rows=True)

    assert summary.shape[0] == 2
    assert outliers.shape[0] == 1
    assert outliers["outcome"].iloc[0] == 100
    assert list(outliers.columns) == ["outcome", "treatment", "confounder"]


def _make_multicausal_data():
    n = 6
    df = pd.DataFrame(
        {
            "y": [1, 1, 1, 1, 1, 100] + [2] * n + [3] * n,
            "t0": [1] * n + [0] * n + [0] * n,
            "t1": [0] * n + [1] * n + [0] * n,
            "t2": [0] * n + [0] * n + [1] * n,
            "x": list(range(3 * n)),
        }
    )
    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["t0", "t1", "t2"],
        confounders=["x"],
        control_treatment="t0",
    )


def test_outcome_outliers_multicausal_default_treatment():
    data = _make_multicausal_data()

    summary, outliers = outcome_outliers(data, return_rows=True)
    row_t0 = summary.loc[summary["treatment"] == "t0"].iloc[0]
    row_t1 = summary.loc[summary["treatment"] == "t1"].iloc[0]
    row_t2 = summary.loc[summary["treatment"] == "t2"].iloc[0]

    assert row_t0["outlier_count"] == 1
    assert bool(row_t0["has_outliers"]) is True
    assert row_t1["outlier_count"] == 0
    assert row_t2["outlier_count"] == 0
    assert outliers.shape[0] == 1
    assert outliers["y"].iloc[0] == 100
    assert outliers["t0"].iloc[0] == 1
