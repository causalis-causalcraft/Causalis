import pytest
import pandas as pd
import numpy as np
from causalis.data import CausalData
from causalis.statistics.models.diff_in_means import DiffInMeans


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'treatment': np.random.choice([0, 1], size=n),
        'outcome': np.random.normal(0, 1, size=n)
    })
    # Add some effect
    df.loc[df['treatment'] == 1, 'outcome'] += 0.5
    return CausalData(df=df, treatment='treatment', outcome='outcome')


@pytest.fixture
def binary_outcome_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'treatment': np.random.choice([0, 1], size=n),
        'outcome': np.random.choice([0, 1], size=n)
    })
    return CausalData(df=df, treatment='treatment', outcome='outcome')


def test_diff_in_means_fit_effect_ttest(sample_data):
    model = DiffInMeans()
    model.fit(sample_data)
    result = model.effect(method='ttest')
    assert 'p_value' in result
    assert 'absolute_difference' in result
    assert isinstance(result['p_value'], float)


def test_diff_in_means_fit_effect_bootstrap(sample_data):
    model = DiffInMeans()
    model.fit(sample_data)
    result = model.effect(method='bootstrap', n_simul=100)
    assert 'p_value' in result
    assert 'absolute_difference' in result


def test_diff_in_means_fit_effect_conversion(binary_outcome_data):
    model = DiffInMeans()
    model.fit(binary_outcome_data)
    result = model.effect(method='conversion_ztest')
    assert 'p_value' in result
    assert 'absolute_difference' in result


def test_diff_in_means_not_fitted():
    model = DiffInMeans()
    with pytest.raises(RuntimeError):
        model.effect()


def test_diff_in_means_invalid_method(sample_data):
    model = DiffInMeans()
    model.fit(sample_data)
    with pytest.raises(ValueError):
        model.effect(method='invalid_method')


def test_diff_in_means_aliases(sample_data, binary_outcome_data):
    model = DiffInMeans()
    model.fit(sample_data)
    # Test 'bootsrap' alias
    result_bootstrap = model.effect(method='bootsrap', n_simul=10)
    assert 'p_value' in result_bootstrap
    
    # Test 'coversion_ztest' alias
    model.fit(binary_outcome_data)
    result_conversion = model.effect(method='coversion_ztest')
    assert 'p_value' in result_conversion

def test_diff_in_means_repr(sample_data):
    model = DiffInMeans()
    assert "status='unfitted'" in repr(model)
    model.fit(sample_data)
    assert "status='fitted'" in repr(model)


def test_diff_in_means_confidence_level(sample_data):
    model = DiffInMeans(confidence_level=0.9)
    model.fit(sample_data)
    result = model.effect(method='ttest')
    # We could check if CI matches 0.9, but here we just check if it works
    assert 'absolute_ci' in result
