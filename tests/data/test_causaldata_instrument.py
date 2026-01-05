import pytest
import pandas as pd
import numpy as np
from causalis.data import CausalData, CausalDataInstrumental

def test_causaldata_with_instrument():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "d": [0, 1, 0],
        "x": [0.1, 0.2, 0.3],
        "z": [1, 0, 1]
    })
    
    cd = CausalDataInstrumental.from_df(df, treatment="d", outcome="y", confounders=["x"], instrument="z")
    
    assert cd.instrument_name == "z"
    assert cd.instrument.tolist() == [1, 0, 1]
    assert "z" in cd.df.columns
    
    # Test get_df
    df_with_z = cd.get_df(include_instrument=True)
    assert "z" in df_with_z.columns
    assert df_with_z["z"].tolist() == [1, 0, 1]

def test_instrument_validation():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "d": [0, 1, 0],
        "x": [0.1, 0.2, 0.3],
        "z_const": [1, 1, 1],
        "z_non_numeric": ["a", "b", "c"],
        "z_overlap": [0, 1, 0] # same as d
    })
    
    # Constant instrument
    with pytest.raises(ValueError, match="instrument is constant"):
        CausalDataInstrumental.from_df(df, treatment="d", outcome="y", confounders=["x"], instrument="z_const")
    
    # Non-numeric instrument
    with pytest.raises(ValueError, match="instrument must contain only int, float, or bool values"):
        CausalDataInstrumental.from_df(df, treatment="d", outcome="y", confounders=["x"], instrument="z_non_numeric")
        
    # Overlap with treatment
    with pytest.raises(ValueError, match="Column 'd' cannot be both treatment and instrument"):
        CausalDataInstrumental.from_df(df, treatment="d", outcome="y", confounders=["x"], instrument="d")
# Overlap with confounders
    with pytest.raises(ValueError, match="confounder columns must be disjoint from treatment/outcome/user_id/instrument"):
        CausalDataInstrumental.from_df(df, treatment="d", outcome="y", confounders=["x"], instrument="x")

def test_instrument_bool_casting():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "d": [0, 1, 0],
        "x": [0.1, 0.2, 0.3],
        "z": [True, False, True]
    })
    
    cd = CausalDataInstrumental.from_df(df, treatment="d", outcome="y", confounders=["x"], instrument="z")
    
    assert cd.df["z"].dtype == np.int8
    assert cd.instrument.tolist() == [1, 0, 1]

def test_instrument_repr():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0],
        "d": [0, 1, 0],
        "x": [0.1, 0.2, 0.3],
        "z": [1, 0, 1]
    })
    
    cd = CausalDataInstrumental.from_df(df, treatment="d", outcome="y", confounders=["x"], instrument="z")
    repr_str = repr(cd)
    assert "instrument='z'" in repr_str
