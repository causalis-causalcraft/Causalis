import pandas as pd
import pytest
from causalis.scenarios.rct import check_srm
from causalis.data.causaldata import CausalData

def test_check_srm_basic():
    assignments = ["control"] * 50 + ["treatment"] * 50
    target_allocation = {"control": 0.5, "treatment": 0.5}
    result = check_srm(assignments, target_allocation)
    assert not result.is_srm
    assert result.p_value > 0.05

def test_check_srm_with_srm():
    assignments = ["control"] * 70 + ["treatment"] * 30
    target_allocation = {"control": 0.5, "treatment": 0.5}
    result = check_srm(assignments, target_allocation)
    assert result.is_srm
    assert result.p_value < 1e-3

def test_check_srm_causal_data():
    df = pd.DataFrame({
        "t": [0] * 50 + [1] * 50,
        "y": [0, 1] * 50
    })
    cd = CausalData.from_df(df, treatment="t", outcome="y")
    target_allocation = {0: 0.5, 1: 0.5}
    
    result = check_srm(cd, target_allocation)
    assert not result.is_srm

def test_p_value_rounding():
    # Construct a case where p-value would have many digits
    assignments = ["control"] * 52 + ["treatment"] * 48
    target_allocation = {"control": 0.5, "treatment": 0.5}
    result = check_srm(assignments, target_allocation)
    
    # Check if p_value has at most 5 decimal places
    # We can do this by comparing it to itself rounded to 5 decimal places
    assert result.p_value == round(result.p_value, 5)
    
    # Also ensure it's not accidentally 0 if it should be small but non-zero
    # For 52/48, chi2 = (52-50)^2/50 + (48-50)^2/50 = 4/50 + 4/50 = 8/50 = 0.16
    # df = 1. p-value for chi2=0.16, df=1 is ~0.6891565...
    # Rounded to 5 digits it should be 0.68916
    assert result.p_value == 0.68916

def test_srm_result_repr():
    assignments = ["control"] * 50 + ["treatment"] * 50
    target_allocation = {"control": 0.5, "treatment": 0.5}
    result = check_srm(assignments, target_allocation)
    
    # We don't want scientific notation in repr
    # For 50/50, chi2=0, df=1, p_value=1.0
    repr_str = repr(result)
    assert "p_value=1.00000" in repr_str
    assert "e-" not in repr_str
