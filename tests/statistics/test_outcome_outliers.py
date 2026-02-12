import pandas as pd
import pytest

from causalis.dgp.causaldata import CausalData
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
