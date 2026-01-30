
import pytest
import numpy as np
import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.scenarios.cuped.model import CUPEDModel

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    X = np.random.normal(10, 1, size=(n, 1))
    D = np.random.binomial(1, 0.5, size=(n, 1))
    # Y = 1 + 5*D + 2*X + noise
    Y = 1 + 5*D.flatten() + 2*X.flatten() + np.random.normal(0, 0.1, size=n)
    
    df = pd.DataFrame({
        'y': Y,
        'd': D.flatten(),
        'x1': X.flatten()
    })
    return CausalData(df=df, treatment='d', outcome='y', confounders=['x1'])

def test_cuped_init():
    # Valid initializations
    CUPEDModel(cov_type='HC0', alpha=0.01)
    model = CUPEDModel()
    assert model.center_covariates is True
    assert model.adjustment == 'lin'

def test_cuped_fit_estimate(sample_data):
    # Standard Lin adjustment
    model = CUPEDModel()
    model.fit(sample_data)
    results = model.estimate()
    
    assert isinstance(results, CausalEstimate)
    assert np.isclose(results.value, 5.0, atol=0.1)
    assert results.diagnostic_data.adj_type == 'lin'
    assert len(results.diagnostic_data.beta_covariates) == 1
    # In Lin model with 1 covariate, we should have 1 main effect and 1 interaction
    assert len(results.diagnostic_data.gamma_interactions) == 1

def test_cuped_lin_with_interactions():
    # Data with true interaction
    np.random.seed(42)
    n = 1000
    X = np.random.normal(10, 1, size=(n, 1))
    D = np.random.binomial(1, 0.5, size=(n, 1))
    # Y = 1 + 2*D + 3*X + 5*D*X + noise
    # ATE = 2 + 5*E[X] = 2 + 5*10 = 52
    Y = 1 + 2*D.flatten() + 3*X.flatten() + 5*D.flatten()*X.flatten() + np.random.normal(0, 0.1, size=n)
    
    df = pd.DataFrame({'y': Y, 'd': D.flatten(), 'x': X.flatten()})
    data = CausalData(df=df, treatment='d', outcome='y', confounders=['x'])
    
    model = CUPEDModel()
    model.fit(data)
    results = model.estimate()
    
    # ATE should be correctly estimated around 52
    assert np.isclose(results.value, 52.0, atol=0.5)
    # gamma interaction coefficient should be around 5
    assert np.isclose(results.diagnostic_data.gamma_interactions[0], 5.0, atol=0.5)

def test_cuped_no_covariates():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'y': np.random.normal(0, 1, n),
        'd': np.random.binomial(1, 0.5, n)
    })
    data = CausalData(df=df, treatment='d', outcome='y', confounders=[])
    
    model = CUPEDModel()
    model.fit(data)
    results = model.estimate()
    assert results.diagnostic_data.ate_naive == results.value
    assert len(results.diagnostic_data.beta_covariates) == 0

def test_cuped_summary_methods(sample_data):
    model = CUPEDModel().fit(sample_data)
    results = model.estimate()
    
    summary_df = results.summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.loc[0, 'coefficient'] == results.value
    
    sd = model.summary_dict()
    assert sd['ate'] == results.value
