import pandas as pd
import numpy as np
import pytest
from causalis.data.causaldata import CausalData
from causalis.statistics.functions import outcome_stats

def test_outcome_stats():
    df = pd.DataFrame({
        "treatment": [0, 0, 0, 1, 1],
        "outcome": [1.0, 2.0, 3.0, 4.0, 5.0],
        "confounder": [1, 2, 3, 4, 5]
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
