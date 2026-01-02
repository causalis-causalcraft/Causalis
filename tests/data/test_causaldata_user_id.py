import pandas as pd
import numpy as np
import pytest
import uuid
from causalis.data.causaldata import CausalData

def test_causaldata_with_user_id_string():
    n = 20
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'user_id': [str(uuid.uuid4()) for _ in range(n)],
        'y': rng.standard_normal(n),
        'd': rng.integers(0, 2, n),
        'x': rng.standard_normal(n)
    })
    
    cd = CausalData(df=df, treatment='d', outcome='y', confounders=['x'], user_id='user_id')
    
    assert cd.user_id_name == 'user_id'
    assert cd.df.shape[1] == 4
    assert 'user_id' in cd.df.columns
    assert cd.df['user_id'].dtype == object or cd.df['user_id'].dtype == 'string'
    
    # Test property
    pd.testing.assert_series_equal(cd.user_id, df['user_id'], check_names=True)

def test_causaldata_from_df_positional():
    n = 20
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'user_id': [str(uuid.uuid4()) for _ in range(n)],
        'y': rng.standard_normal(n),
        'd': rng.integers(0, 2, n),
        'x': rng.standard_normal(n)
    })

    # CausalData.from_df(df, treatment, outcome, confounders, user_id)
    cd = CausalData.from_df(df, 'd', 'y', ['x'], 'user_id')

    assert cd.treatment_name == 'd'
    assert cd.outcome_name == 'y'
    assert cd.confounders_names == ['x']
    assert cd.user_id_name == 'user_id'
    assert cd.df.shape[1] == 4

def test_causaldata_user_id_overlap_fails():
    df = pd.DataFrame({
        'y': [1, 2],
        'd': [0, 1],
        'x': [0.1, 0.2]
    })
    
    # user_id same as treatment should fail
    with pytest.raises(Exception, match="cannot be both"):
        CausalData(df=df, treatment='d', outcome='y', user_id='d')

def test_causaldata_user_id_missing_fails():
    df = pd.DataFrame({
        'y': [1, 2],
        'd': [0, 1],
        'x': [0.1, 0.2]
    })
    
    with pytest.raises(Exception, match="does not exist"):
        CausalData(df=df, treatment='d', outcome='y', user_id='missing_id')

def test_causaldata_user_id_duplicates_fail():
    df = pd.DataFrame({
        'user_id': [1, 1, 2],
        'y': [1.0, 2.0, 3.0],
        'd': [0, 1, 0],
        'x': [0.1, 0.2, 0.3]
    })
    
    with pytest.raises(ValueError, match="contains duplicate values"):
        CausalData(df=df, treatment='d', outcome='y', confounders=['x'], user_id='user_id')
