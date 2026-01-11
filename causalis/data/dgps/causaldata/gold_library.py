from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from .base import CausalDatasetGenerator
from .functional import obs_linear_effect, generate_classic_rct
from causalis.data.causaldata import CausalData

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

def obs_linear_26_dataset(n: int = 10000, seed: int = 42, include_oracle: bool = True, return_causal_data: bool = True):
    """
    A pre-configured observational linear dataset with 5 standard confounders.
    Based on the scenario in docs/cases/dml_ate.ipynb.

    Parameters
    ----------
    n : int, default=10000
        Number of samples.
    seed : int, default=42
        Random seed.
    include_oracle : bool, default=True
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
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
        include_oracle=include_oracle,
        add_ancillary=False
    )
    
    if not return_causal_data:
        return df

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


def generate_classic_rct_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True
):
    """
    A pre-configured classic RCT dataset with 3 binary confounders.
    n=10000, split=0.5, outcome is conversion (binary), real effect = 0.01.

    Parameters
    ----------
    seed : int, default=42
        Random seed.
    add_pre : bool, default=False
        Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
    beta_y : array-like, optional
        Linear coefficients for confounders in the outcome model.
    outcome_depends_on_x : bool, default=True
        Whether to add default effects for confounders if beta_y is None.
    include_oracle : bool, default=False
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        Whether to return a CausalData object.

    Returns
    -------
    CausalData or pd.DataFrame
    """
    outcome_params = {"p": {"A": 0.10, "B": 0.11}}
    df = generate_classic_rct(
        n=10000,
        split=0.5,
        random_state=seed,
        outcome_params=outcome_params,
        add_pre=add_pre,
        beta_y=beta_y,
        outcome_depends_on_x=outcome_depends_on_x,
        include_oracle=include_oracle,
        return_causal_data=False
    )
    # The requirement asks for outcome - conversion(binary)
    df = df.rename(columns={"y": "conversion"})

    if not return_causal_data:
        return df

    confounders = ["platform_ios", "country_usa", "source_paid"]
    if add_pre:
        confounders.append("y_pre")

    return CausalData(
        df=df,
        treatment="d",
        outcome="conversion",
        confounders=confounders
    )
