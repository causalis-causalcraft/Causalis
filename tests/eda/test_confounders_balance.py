import pandas as pd
import numpy as np
from causalis.data.causaldata import CausalData
from causalis.statistics.functions import confounders_balance

def test_confounders_balance_basic():
    df = pd.DataFrame({
        "t": [0, 0, 1, 1] * 25,
        "y": [1, 2, 3, 4] * 25,
        "x1": [1, 2, 1.1, 2.1] * 25,
        "x2": [1, 1, 0, 0] * 25,
        # "cat": ["A", "A", "B", "B"] * 25
    })
    cd = CausalData.from_df(df, treatment="t", outcome="y", confounders=["x1", "x2"])
    
    # New call style (what we want)
    result = confounders_balance(cd)
    
    assert isinstance(result, pd.DataFrame)
    assert "mean_d_0" in result.columns
    assert "mean_d_1" in result.columns
    assert "abs_diff" in result.columns
    assert "smd" in result.columns
    assert "ks" not in result.columns
    assert "ks_pvalue" in result.columns
    
    # Check if x1 is in confounders column
    assert "x1" in result["confounders"].values

def test_ks_pvalue_rounding():
    # Construct a case with very small p-value to check scientific notation/rounding
    df = pd.DataFrame({
        "t": [0] * 100 + [1] * 100,
        "y": np.random.randn(200),
        "x1": np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10, 1, 100)])
    })
    cd = CausalData.from_df(df, treatment="t", outcome="y", confounders=["x1"])
    
    result = confounders_balance(cd)
    
    p_val_str = result[result["confounders"] == "x1"]["ks_pvalue"].iloc[0]
    assert isinstance(p_val_str, str)
    
    # Check if it has 5 digits after the decimal point
    assert len(p_val_str.split(".")[1]) == 5
    # Check if it's not in scientific notation
    assert "e" not in p_val_str.lower()
    
    # In this extreme case, it should be "0.00000"
    assert p_val_str == "0.00000"
