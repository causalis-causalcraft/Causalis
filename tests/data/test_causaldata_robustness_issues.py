import pytest
import pandas as pd
import numpy as np
from causalis.dgp.causaldata import CausalData

def test_duplicate_column_names():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d": [0, 1, 0],
        "x": [0.1, 0.2, 0.3],
    })
    # Add a duplicate column name manually
    df.columns = ["y", "d", "y"]
    
    with pytest.raises(ValueError, match="DataFrame has duplicate column names"):
        CausalData.from_df(df, treatment="d", outcome="y", confounders=[])

def test_invalid_confounders_type():
    df = pd.DataFrame({
        "y": [1, 2, 3],
        "d": [0, 1, 0],
        "x": [0.1, 0.2, 0.3],
    })
    
    with pytest.raises(TypeError, match="confounders must be None, a string, or a list of strings"):
        CausalData.from_df(df, treatment="d", outcome="y", confounders=123)
        
    with pytest.raises(TypeError, match="confounders must be None, a string, or a list of strings"):
        CausalData.from_df(df, treatment="d", outcome="y", confounders=("x1", "x2"))

    with pytest.raises(TypeError, match="All confounder names must be strings"):
        CausalData.from_df(df, treatment="d", outcome="y", confounders=[1, 2])

def test_user_id_boolean_not_casted():
    df = pd.DataFrame({
        "y": [1.0, 2.0],
        "d": [0, 1],
        "x": [0.1, 0.2],
        "uid": [True, False]
    })
    
    cd = CausalData.from_df(df, treatment="d", outcome="y", confounders=["x"], user_id="uid")
    
    # Check that uid column in cd.df is still boolean, not int8
    assert cd.df["uid"].dtype == bool or cd.df["uid"].dtype == object
    # If it was casted to int8, it would be np.int8
    assert cd.df["uid"].dtype != np.int8
