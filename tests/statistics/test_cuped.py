
import pytest
import numpy as np
import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.preperiod import corr_on_scale
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.scenarios.cuped.dgp import generate_cuped_tweedie_26, _resolve_second_pre_target
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
    model.fit(sample_data, covariates=['x1'])
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
    model.fit(data, covariates=['x'])
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
    model.fit(data, covariates=[])
    results = model.estimate()
    assert results.diagnostic_data.ate_naive == results.value
    assert len(results.diagnostic_data.beta_covariates) == 0


def test_cuped_requires_explicit_covariates(sample_data):
    model = CUPEDModel()
    with pytest.raises(ValueError, match="covariates must be provided explicitly"):
        model.fit(sample_data)


def test_make_cuped_tweedie_26_has_two_pre_covariates():
    data = generate_cuped_tweedie_26(
        n=2000,
        seed=123,
        add_pre=True,
        include_oracle=False,
        return_causal_data=True
    )

    assert "y_pre" in data.df.columns
    assert "y_pre_2" in data.df.columns
    assert "y_pre" in data.confounders_names
    assert "y_pre_2" in data.confounders_names

    corr = np.corrcoef(data.df["y_pre"], data.df["y_pre_2"])[0, 1]
    assert np.isfinite(corr)
    assert abs(corr) < 0.999


def test_make_cuped_tweedie_26_hits_requested_control_correlations():
    target_1 = 0.70
    target_2 = 0.55
    df = generate_cuped_tweedie_26(
        n=6000,
        seed=42,
        add_pre=True,
        pre_target_corr=target_1,
        pre_target_corr_2=target_2,
        include_oracle=False,
        return_causal_data=False,
    )
    ctrl = (df["d"].to_numpy() == 0)
    corr_1 = corr_on_scale(
        df.loc[ctrl, "y_pre"].to_numpy(dtype=float),
        df.loc[ctrl, "y"].to_numpy(dtype=float),
    )
    corr_2 = corr_on_scale(
        df.loc[ctrl, "y_pre_2"].to_numpy(dtype=float),
        df.loc[ctrl, "y"].to_numpy(dtype=float),
    )
    assert np.isclose(corr_1, target_1, atol=0.015)
    assert np.isclose(corr_2, target_2, atol=0.015)


def test_resolve_second_pre_target_default_formula():
    assert _resolve_second_pre_target(0.82, None) == pytest.approx(0.72)
    assert _resolve_second_pre_target(0.60, None) == pytest.approx(0.50)
    assert _resolve_second_pre_target(0.40, None) == pytest.approx(0.30)
    assert _resolve_second_pre_target(0.05, None) == pytest.approx(0.0)
    assert _resolve_second_pre_target(0.82, 0.64) == pytest.approx(0.64)


def test_make_cuped_tweedie_26_rejects_reserved_pre_column_names():
    with pytest.raises(ValueError, match="collides with an existing generated column"):
        generate_cuped_tweedie_26(
            n=500,
            seed=7,
            pre_name="y",
            return_causal_data=False
        )
