import pandas as pd
import pytest
from causalis.dgp.causaldata import CausalData

def test_causaldata_raises_error_on_constant_confounders():
    df = pd.DataFrame({
        "y": [1.1, 2.2, 3.3],
        "d": [0, 1, 0],
        "x1": [1.0, 1.0, 1.0],  # Constant
        "x2": [1.0, 2.0, 3.0]
    })
    
    with pytest.raises(ValueError, match="Column 'x1' specified as confounder is constant"):
        CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])
