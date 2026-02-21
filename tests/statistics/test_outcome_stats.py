import pandas as pd
import pytest
from causalis.dgp.causaldata import CausalData
from causalis.dgp.multicausaldata import MultiCausalData
from causalis.shared import outcome_stats

def test_outcome_stats():
    df = pd.DataFrame({
        "treatment": [0, 0, 0, 1, 1],
        "outcome": [1.0, 2.0, 3.0, 4.0, 5.0],
        "confounder": [1, 2, 3, 4, 6]
    })
    data = CausalData.from_df(df, treatment="treatment", outcome="outcome", confounders=["confounder"])
    
    stats = outcome_stats(data)
    
    assert isinstance(stats, pd.DataFrame)
    assert "treatment" in stats.columns
    assert stats.index.name is None
    assert len(stats) == 2
    assert list(stats.columns) == ['treatment', 'count', 'mean', 'std', 'min', 'p10', 'p25', 'median', 'p75', 'p90', 'max']
    
    # Check values for treatment 0
    assert stats.loc[0, 'count'] == 3
    assert stats.loc[0, 'mean'] == 2.0
    assert stats.loc[0, 'min'] == 1.0
    assert stats.loc[0, 'max'] == 3.0
    assert stats.loc[0, 'median'] == 2.0
    
    # Check values for treatment 1
    assert stats.loc[1, 'count'] == 2
    assert stats.loc[1, 'mean'] == 4.5
    assert stats.loc[1, 'min'] == 4.0
    assert stats.loc[1, 'max'] == 5.0

def test_outcome_stats_non_numeric_treatment():
    df = pd.DataFrame({
        "treatment": ["A", "A", "B"],
        "outcome": [1, 2, 3]
    })
    # CausalData validates this on creation
    with pytest.raises(Exception):
        CausalData.from_df(df, treatment="treatment", outcome="outcome")


def test_outcome_stats_multicausaldata():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "t0": [1, 0, 0, 1, 0, 0],
            "t1": [0, 1, 0, 0, 1, 0],
            "t2": [0, 0, 1, 0, 0, 1],
            "x": [10, 11, 12, 13, 14, 15],
        }
    )
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["t0", "t1", "t2"],
        confounders=["x"],
        control_treatment="t0",
    )

    stats = outcome_stats(data)

    assert isinstance(stats, pd.DataFrame)
    assert list(stats["treatment"]) == ["t0", "t1", "t2"]
    assert list(stats["count"]) == [2, 2, 2]

    row_t0 = stats.loc[stats["treatment"] == "t0"].iloc[0]
    row_t1 = stats.loc[stats["treatment"] == "t1"].iloc[0]
    row_t2 = stats.loc[stats["treatment"] == "t2"].iloc[0]

    assert row_t0["mean"] == 2.5
    assert row_t1["mean"] == 3.5
    assert row_t2["mean"] == 4.5
