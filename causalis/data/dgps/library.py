from __future__ import annotations
from typing import Optional, List, Dict, Any
from .base import CausalDatasetGenerator
from .functional import obs_linear_effect

def make_gold_linear(n: int = 10000, seed: int = 42):
    """
    A standard linear benchmark with moderate confounding.
    Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "normal",   "mu": 24, "sd": 12},
        {"name": "avg_sessions_week", "dist": "normal",   "mu": 5,  "sd": 2},
        {"name": "spend_last_month",  "dist": "uniform",  "a": 0,   "b": 200},
        {"name": "premium_user",      "dist": "bernoulli","p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli","p": 0.60},
    ]
    
    # Coefficients aligned to the specs above
    beta_y = [0.05, 0.60, 0.005, 0.80, 0.20]
    beta_d = [0.08, 0.12, 0.004, 0.25, 0.10]
    
    gen = CausalDatasetGenerator(
        theta=0.80,
        beta_y=beta_y,
        beta_d=beta_d,
        alpha_y=0.0,
        alpha_d=0.0,
        sigma_y=1.0,
        outcome_type="continuous",
        confounder_specs=confounder_specs,
        target_d_rate=0.20,
        seed=seed
    )
    return gen.to_causal_data(n)

def obs_linear_26_dataset(n: int = 10000, seed: int = 42, return_causal_data: bool = True):
    """
    A pre-configured observational linear dataset with 5 standard confounders.
    Based on the scenario in docs/examples/dml_ate.ipynb.

    Parameters
    ----------
    n : int, default=10000
        Number of samples.
    seed : int, default=42
        Random seed.
    return_causal_data : bool, default=True
        If True, returns a CausalData object. If False, returns a pandas DataFrame.
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "normal",   "mu": 24, "sd": 12},
        {"name": "avg_sessions_week", "dist": "normal",   "mu": 5,  "sd": 2},
        {"name": "spend_last_month",  "dist": "uniform",  "a": 0,   "b": 200},
        {"name": "premium_user",      "dist": "bernoulli","p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli","p": 0.60},
    ]
    # Coefficients from dml_ate.ipynb
    beta_y = [0.05, 0.40, 0.02, 2.00, 1.00]
    beta_d = [0.015, 0.10, 0.002, 0.75, 0.30]

    df = obs_linear_effect(
        n=n,
        theta=1.8,
        sigma_y=3.5,
        target_d_rate=0.2,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        random_state=seed,
        add_ancillary=False
    )
    
    if not return_causal_data:
        return df

    from ..causaldata import CausalData
    confounder_names = [s["name"] for s in confounder_specs]
    return CausalData(df=df, treatment='d', outcome='y', confounders=confounder_names)

class SmokingDGP(CausalDatasetGenerator):
    """
    A specialized generating class for smoking-related causal scenarios.
    Example of how users can extend CausalDatasetGenerator for specific domains.
    """
    def __init__(self, effect_size: float = 2.0, seed: Optional[int] = 42, **kwargs):
        confounder_specs = [
            {"name": "age", "dist": "normal", "mu": 45, "sd": 15},
            {"name": "income", "dist": "uniform", "a": 20, "b": 150},
            {"name": "education_years", "dist": "normal", "mu": 12, "sd": 3},
        ]
        super().__init__(
            theta=effect_size,
            confounder_specs=confounder_specs,
            seed=seed,
            **kwargs
        )
