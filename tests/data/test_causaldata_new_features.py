import pytest
import pandas as pd
import numpy as np
from causalis.data_contracts.causaldata import CausalData
from causalis.data_contracts.multicausaldata import MultiCausalData
from pydantic import ValidationError

def test_causal_data_extra_forbid():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "t": [0, 1, 0],
        "x": [5, 6, 7]
    })
    # Should work
    CausalData(df=df, outcome="y", treatment="t", confounders=["x"])
    
    # Should fail due to extra="forbid"
    with pytest.raises(ValidationError):
        CausalData(df=df, outcome="y", treatment="t", confounders=["x"], extra_arg="oops")

def test_causal_data_dtype_agnostic_duplicates():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "t": [1, 0, 1],
        "x": [1.0, 0.0, 1.0]  # Same values as t, but different dtype (float vs int)
    })
    # This should now raise ValueError because t and x have identical values
    with pytest.raises(ValueError, match="have identical values"):
        CausalData(df=df, outcome="y", treatment="t", confounders=["x"])

def test_causal_data_get_df_empty():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "t": [0, 1, 0],
        "x": [5, 6, 7]
    })
    cd = CausalData(df=df, outcome="y", treatment="t", confounders=["x"])
    
    empty_df = cd.get_df(include_outcome=False, include_treatment=False, include_confounders=False)
    assert empty_df.shape == (3, 0)
    assert isinstance(empty_df, pd.DataFrame)

def test_multi_causal_data_basic():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "t1": [0, 1, 0],
        "t2": [1, 0, 1],
        "x": [5, 6, 7]
    })
    mcd = MultiCausalData(df=df, outcome="y", treatments=["t1", "t2"], confounders=["x"])
    assert mcd.treatment_names == ["t1", "t2"]
    assert mcd.df.shape == (3, 4)
    # Check canonicalization to int8
    assert mcd.df["t1"].dtype == "int8"
    assert mcd.df["t2"].dtype == "int8"

def test_multi_causal_data_max_treatments():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "t1": [0, 1, 0],
        "t2": [0, 1, 0],
        "t3": [0, 1, 0],
        "t4": [0, 1, 0],
        "t5": [0, 1, 0],
        "t6": [0, 1, 0],
    })
    with pytest.raises(ValueError, match="Too many treatment columns"):
        MultiCausalData(df=df, outcome="y", treatments=["t1", "t2", "t3", "t4", "t5", "t6"])

def test_multi_causal_data_binary_validation():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "t1": [0, 1, 2], # Not binary
    })
    with pytest.raises(ValueError, match="must be binary encoded"):
        MultiCausalData(df=df, outcome="y", treatments=["t1"])

    df_float = pd.DataFrame({
        "y": [1, 2, 3],
        "t1": [0.0, 1.0, 1e-13], # Close enough to 0
    })
    mcd = MultiCausalData(df=df_float, outcome="y", treatments=["t1"])
    assert mcd.df["t1"].dtype == "int8"
    assert (mcd.df["t1"] == [0, 1, 0]).all()

def test_multi_causal_data_get_df():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "t1": [0, 1, 0],
        "t2": [1, 0, 1],
        "x": [5, 6, 7]
    })
    mcd = MultiCausalData(df=df, outcome="y", treatments=["t1", "t2"], confounders=["x"])
    
    df_only_t = mcd.get_df(include_outcome=False, include_confounders=False)
    assert list(df_only_t.columns) == ["t1", "t2"]
    
    df_empty = mcd.get_df(include_outcome=False, include_confounders=False, include_treatments=False)
    assert df_empty.shape == (3, 0)
