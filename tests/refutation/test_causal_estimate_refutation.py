import pytest
import numpy as np
import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness import IRM
from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import run_overlap_diagnostics
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import (
    run_score_diagnostics, 
    refute_placebo_outcome, 
    refute_placebo_treatment, 
    refute_subset
)
from causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation import run_uncofoundedness_diagnostics, validate_uncofoundedness_balance

@pytest.fixture
def sample_causal_data():
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, (n, 2))
    d = np.random.binomial(1, 0.5, n)
    y = 0.5 * d + x[:, 0] + np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x[:, 0],
        'x2': x[:, 1]
    })
    return CausalData(df=df, treatment='d', outcome='y', confounders=['x1', 'x2'])

def test_overlap_diagnostics_with_causal_estimate(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = run_overlap_diagnostics(res=result)
    assert "summary" in report
    assert "ks" in report
    assert "auc" in report

def test_score_diagnostics_with_causal_estimate(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = run_score_diagnostics(res=result)
    assert "summary" in report
    assert "orthogonality_derivatives" in report

def test_unconfoundedness_diagnostics_with_causal_estimate(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = run_uncofoundedness_diagnostics(res=result)
    assert "summary" in report
    assert "balance" in report
    assert "overall_flag" in report

def test_validate_uncofoundedness_balance_with_causal_estimate(sample_causal_data):
    model = IRM().fit(sample_causal_data)
    result = model.estimate(score='ATE')
    
    # This should work with CausalEstimate
    report = validate_uncofoundedness_balance(result)
    assert "smd" in report
    assert "smd_unweighted" in report
    assert "pass" in report

def test_placebo_refutations_with_causal_estimate(sample_causal_data):
    # This tests if refute_* functions work when the model.estimate returns CausalEstimate
    # We pass the estimate method (or a lambda) as the inference_fn
    model = IRM()
    
    def inference_fn(data, **kwargs):
        return model.fit(data).estimate(**kwargs)
    
    res_y = refute_placebo_outcome(inference_fn, sample_causal_data, random_state=42)
    assert "theta" in res_y
    assert "p_value" in res_y
    
    res_t = refute_placebo_treatment(inference_fn, sample_causal_data, random_state=42)
    assert "theta" in res_t
    assert "p_value" in res_t
    
    res_s = refute_subset(inference_fn, sample_causal_data, fraction=0.7, random_state=42)
    assert "theta" in res_s
    assert "p_value" in res_s
