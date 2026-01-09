import pandas as pd
import numpy as np
from causalis.eda.eda import CausalEDA, CausalDataLite
from causalis.data.causaldata import CausalData

def test_causal_eda_confounders_means():
    df = pd.DataFrame({
        "t": [0] * 50 + [1] * 50,
        "y": np.random.randn(100),
        "x1": np.random.randn(100),
        "x2": np.random.randn(100)
    })
    
    # Test with CausalDataLite
    data_lite = CausalDataLite(df=df, treatment="t", outcome="y", confounders=["x1", "x2"])
    eda_lite = CausalEDA(data_lite)
    balance_lite = eda_lite.confounders_means()
    assert isinstance(balance_lite, pd.DataFrame)
    assert "ks_pvalue" in balance_lite.columns
    assert isinstance(balance_lite["ks_pvalue"].iloc[0], str)

    # Test with CausalData
    cd = CausalData.from_df(df, treatment="t", outcome="y", confounders=["x1", "x2"])
    eda_cd = CausalEDA(cd)
    balance_cd = eda_cd.confounders_means()
    assert isinstance(balance_cd, pd.DataFrame)
    assert "ks_pvalue" in balance_cd.columns
    assert isinstance(balance_cd["ks_pvalue"].iloc[0], str)
