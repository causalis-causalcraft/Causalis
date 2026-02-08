"""
Tests for the CausalDatasetGenerator class.
"""

import pytest
import pandas as pd
import numpy as np
from causalis.data_contracts import CausalData
from causalis.dgp import CausalDatasetGenerator


@pytest.fixture
def random_seed():
    """Fixture to provide a consistent random seed for tests."""
    return 42


@pytest.fixture
def basic_generator(random_seed):
    """Fixture to provide a basic CausalDatasetGenerator instance."""
    return CausalDatasetGenerator(
        theta=2.0,
        beta_y=np.array([1.0, -0.5, 0.2]),
        beta_d=np.array([0.8, 1.2, -0.3]),
        target_d_rate=0.35,
        outcome_type="continuous",
        sigma_y=1.0,
        seed=random_seed,
        confounder_specs=[
            {"name": "age", "dist": "normal", "mu": 50, "sd": 10},
            {"name": "smoker", "dist": "bernoulli", "p": 0.3},
            {"name": "bmi", "dist": "normal", "mu": 27, "sd": 4},
        ],
    )


def test_generator_initialization(random_seed):
    """Test that the generator can be initialized with various parameters."""
    # Basic initialization
    gen = CausalDatasetGenerator(seed=random_seed)
    assert gen.theta == 1.0
    assert gen.outcome_type == "continuous"
    assert gen.k == 5

    # Custom initialization
    gen = CausalDatasetGenerator(
        theta=3.0,
        outcome_type="binary",
        seed=random_seed,
        k=10
    )
    assert gen.theta == 3.0
    assert gen.outcome_type == "binary"
    assert gen.k == 10

    # With confounder specs
    gen = CausalDatasetGenerator(
        confounder_specs=[
            {"name": "x1", "dist": "normal", "mu": 0, "sd": 1},
            {"name": "x2", "dist": "bernoulli", "p": 0.5},
        ],
        seed=random_seed
    )
    assert gen.k == 2


def test_generate_continuous_outcome(basic_generator):
    """Test generating data_contracts with continuous outcome."""
    df = basic_generator.generate(100)
    
    # Check DataFrame shape and columns
    assert df.shape[0] == 100
    assert set(df.columns) >= {"y", "d", "age", "smoker", "bmi", "m", "g0", "g1", "cate"}
    
    # Check data_contracts types
    assert df["y"].dtype == np.float64
    assert df["d"].isin([0.0, 1.0]).all()
    
    # Check treatment rate is close to outcome
    assert abs(df["d"].mean() - 0.35) < 0.1
    
    # Check CATE calculation
    assert np.allclose(df["cate"], df["g1"] - df["g0"])


def test_generate_binary_outcome(random_seed):
    """Test generating data_contracts with binary outcome."""
    gen = CausalDatasetGenerator(
        theta=1.0,
        outcome_type="binary",
        seed=random_seed,
        confounder_specs=[
            {"name": "x1", "dist": "normal", "mu": 0, "sd": 1},
        ],
    )
    
    df = gen.generate(100)
    
    # Check DataFrame shape and columns
    assert df.shape[0] == 100
    assert set(df.columns) >= {"y", "d", "x1", "m", "g0", "g1", "cate"}
    
    # Check data_contracts types
    assert df["y"].isin([0.0, 1.0]).all()
    assert df["d"].isin([0.0, 1.0]).all()
    
    # Check g0 and g1 are probabilities (between 0 and 1)
    assert (df["g0"] >= 0).all() and (df["g0"] <= 1).all()
    assert (df["g1"] >= 0).all() and (df["g1"] <= 1).all()


def test_generate_poisson_outcome(random_seed):
    """Test generating data_contracts with Poisson outcome."""
    gen = CausalDatasetGenerator(
        theta=0.5,
        outcome_type="poisson",
        seed=random_seed,
        confounder_specs=[
            {"name": "x1", "dist": "normal", "mu": 0, "sd": 1},
        ],
    )
    
    df = gen.generate(100)
    
    # Check DataFrame shape and columns
    assert df.shape[0] == 100
    assert set(df.columns) >= {"y", "d", "x1", "m", "g0", "g1", "cate"}
    
    # Check data_contracts types
    assert df["y"].dtype == np.float64
    assert (df["y"] >= 0).all()  # Poisson values are non-negative
    assert df["d"].isin([0.0, 1.0]).all()
    
    # Check g0 and g1 are non-negative (Poisson means)
    assert (df["g0"] >= 0).all()
    assert (df["g1"] >= 0).all()


def test_heterogeneous_treatment_effect(random_seed):
    """Test generating data_contracts with heterogeneous treatment effect."""
    # Define a heterogeneous treatment effect function
    def tau_func(X):
        return 1.0 + 0.5 * X[:, 0]  # Effect depends on first covariate
    
    gen = CausalDatasetGenerator(
        tau=tau_func,  # Use heterogeneous effect instead of constant theta
        seed=random_seed,
        confounder_specs=[
            {"name": "x1", "dist": "normal", "mu": 0, "sd": 1},
            {"name": "x2", "dist": "normal", "mu": 0, "sd": 1},
        ],
    )
    
    df = gen.generate(100)
    
    # Check that CATE varies with x1
    x1_low = df[df["x1"] < df["x1"].median()]
    x1_high = df[df["x1"] >= df["x1"].median()]
    
    # Higher x1 should have higher treatment effect on average
    assert x1_high["cate"].mean() > x1_low["cate"].mean()


def test_confounder_distributions(random_seed):
    """Test different confounder distributions."""
    gen = CausalDatasetGenerator(
        seed=random_seed,
        confounder_specs=[
            {"name": "normal_var", "dist": "normal", "mu": 10, "sd": 2},
            {"name": "uniform_var", "dist": "uniform", "a": 0, "b": 5},
            {"name": "bernoulli_var", "dist": "bernoulli", "p": 0.7},
            {"name": "categorical_var", "dist": "categorical", 
             "categories": ["A", "B", "C"], "probs": [0.5, 0.3, 0.2]},
        ],
    )
    
    df = gen.generate(1000)
    
    # Check normal distribution
    assert 9 < df["normal_var"].mean() < 11
    assert 1.8 < df["normal_var"].std() < 2.2
    
    # Check uniform distribution
    assert 0 <= df["uniform_var"].min()
    assert df["uniform_var"].max() <= 5
    assert 2.3 < df["uniform_var"].mean() < 2.7  # Expected mean is 2.5
    
    # Check bernoulli distribution
    assert df["bernoulli_var"].isin([0.0, 1.0]).all()
    assert 0.65 < df["bernoulli_var"].mean() < 0.75  # Close to p=0.7
    
    # Check categorical distribution (one-hot encoded)
    assert "categorical_var_B" in df.columns
    assert "categorical_var_C" in df.columns
    assert df["categorical_var_B"].isin([0.0, 1.0]).all()
    assert df["categorical_var_C"].isin([0.0, 1.0]).all()


def test_to_causal_data(basic_generator):
    """Test conversion to CausalData object."""
    # Generate data_contracts and convert to CausalData
    causal_data = basic_generator.to_causal_data(100)
    
    # Check that it's a CausalData object
    assert isinstance(causal_data, CausalData)
    
    # Check that it has the correct columns
    assert causal_data.outcome.name == "y"
    assert causal_data.treatment.name == "d"
    assert set(causal_data.confounders) == {"age", "smoker", "bmi"}
    
    # Check that the data_contracts is accessible
    assert causal_data.df.shape[0] == 100
    assert set(causal_data.df.columns) == {"y", "d", "age", "smoker", "bmi"}
    
    # Test with specific confounders
    causal_data_specific = basic_generator.to_causal_data(100, confounders=["age"])
    assert causal_data_specific.confounders == ["age"]
    assert set(causal_data_specific.df.columns) == {"y", "d", "age"}



def test_invalid_confounder_spec(random_seed):
    """Test that an error is raised for invalid confounder specification."""
    gen = CausalDatasetGenerator(
        seed=random_seed,
        confounder_specs=[
            {"name": "invalid_dist", "dist": "unknown_distribution"},
        ],
    )
    
    with pytest.raises(ValueError) as excinfo:
        gen.generate(100)
    
    assert "Unknown dist: unknown_distribution" in str(excinfo.value)


def test_score_bounding(random_seed):
    """Test that score_bounding limits the range of propensity scores."""
    # Without bounding, with very large coefficients, propensity scores will be near 0 or 1
    gen_no_bound = CausalDatasetGenerator(
        beta_d=np.array([100.0]),
        confounder_specs=[{"name": "x1", "dist": "normal"}],
        seed=random_seed,
        include_oracle=True
    )
    df_no_bound = gen_no_bound.generate(1000)
    # Check that some propensities are near 0 or 1
    assert (df_no_bound["m"] < 0.01).any() or (df_no_bound["m"] > 0.99).any()

    # With bounding, propensities should be more balanced
    gen_bound = CausalDatasetGenerator(
        beta_d=np.array([100.0]),
        confounder_specs=[{"name": "x1", "dist": "normal"}],
        seed=random_seed,
        score_bounding=1.0,  # Bound scores to tanh
        include_oracle=True
    )
    df_bound = gen_bound.generate(1000)
    # With bounding c=1.0, max logit is ~1.0, so max propensity is sigmoid(1.0) ~ 0.73
    # min propensity is sigmoid(-1.0) ~ 0.27
    # Note: alpha_d calibration might shift this, but it should still be bounded.
    assert (df_bound["m"] > 0.05).all()
    assert (df_bound["m"] < 0.95).all()


def test_confounder_clipping(random_seed):
    """Test that confounder clipping works correctly."""
    gen = CausalDatasetGenerator(
        seed=random_seed,
        confounder_specs=[
            {"name": "x1", "dist": "normal", "mu": 0, "sd": 10, "clip_min": -5, "clip_max": 5},
        ],
    )
    df = gen.generate(1000)
    assert (df["x1"] >= -5).all()
    assert (df["x1"] <= 5).all()
    # Check that it's not just all zeros (i.e. clipping is happening on a real distribution)
    assert df["x1"].std() > 0