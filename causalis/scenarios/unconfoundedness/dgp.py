from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.functional import obs_linear_effect
from causalis.dgp.causaldata.base import CausalDatasetGenerator
from causalis.dgp.base import _gaussian_copula

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
        {"name": "tenure_months",     "dist": "normal",   "mu": 24, "sd": 12, "clip_min": 0, "clip_max": 120},
        {"name": "avg_sessions_week", "dist": "normal",   "mu": 5,  "sd": 2,  "clip_min": 0, "clip_max": 40},
        {"name": "spend_last_month",  "dist": "lognormal","mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "premium_user",      "dist": "bernoulli","p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli","p": 0.60},
    ]
    # Coefficients from dml_ate.ipynb
    beta_y = [0.05, 0.40, 0.02, 2.00, 1.00]
    beta_d = [0.015, 0.10, 0.002, 0.75, 0.30]

    gen = CausalDatasetGenerator(
        theta=1.8,
        sigma_y=3.5,
        target_d_rate=0.2,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        seed=seed,
        include_oracle=include_oracle,
        use_copula=True,
        copula_corr=np.array([
            [1.00, 0.30, 0.20, 0.15, 0.00],
            [0.30, 1.00, 0.40, 0.35, 0.00],
            [0.20, 0.40, 1.00, 0.30, 0.00],
            [0.15, 0.35, 0.30, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ])
    )
    df = gen.generate(n)
    
    if not return_causal_data:
        return df

    confounder_names = [s["name"] for s in confounder_specs]
    return CausalData(df=df, treatment='d', outcome='y', confounders=confounder_names)

def generate_obs_hte_26(
    n: int = 10000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True
) -> Union[pd.DataFrame, CausalData]:
    """
    Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
    and a heterogeneous (nonlinear) treatment effect tau(X).
    Based on the scenario in notebooks/cases/dml_atte.ipynb.

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
    # 1) confounders and their distributions
    confounder_specs = [
        {"name": "tenure_months",     "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
        {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
        {"name": "spend_last_month",  "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "premium_user",      "dist": "bernoulli", "p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli", "p": 0.60},
    ]

    # Indices (for convenience inside g_y, g_t, tau)
    TENURE, SESS, SPEND, PREMIUM, URBAN = range(5)

    # 2) Nonlinear baseline for outcome f_y(X) = X @ beta_y + g_y(X)
    beta_y = np.array([
        0.01,   # tenure_months
        0.00,   # avg_sessions_week (moved to g_y)
        0.00,   # spend_last_month (moved to g_y)
        1.20,   # premium_user
        0.60,   # urban_resident
    ], dtype=float)

    def g_y(X: np.ndarray) -> np.ndarray:
        # Nonlinearities and interactions in outcome baseline
        tenure_months = X[:, TENURE]
        sessions = X[:, SESS]
        spend = X[:, SPEND]
        premium = X[:, PREMIUM]
        urban = X[:, URBAN]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)

        return (
            1.5 * np.tanh(tenure_months / 24.0)               # saturating tenure curve
            + 0.5 * (log_sessions - np.log1p(5.0)) ** 2        # convex effect of sessions
            + 0.2 * (log_spend - np.log1p(60.0)) * (log_sessions - np.log1p(5.0)) # log-log interaction
            + 0.5 * premium * (log_sessions - np.log1p(5.0))  # premium × sessions interaction
            + 0.8 * urban * np.tanh((log_spend - np.log1p(60.0)) / 2.0) # nonlinear spend effect differs by urban
        )

    # 3) Nonlinear treatment score f_t(X) = X @ beta_t + g_t(X)
    beta_d = np.array([
        0.005,  # tenure_months
        0.00,   # avg_sessions_week (moved to g_d)
        0.00,   # spend_last_month (moved to g_d)
        0.80,   # premium_user
        0.25,   # urban_resident
    ], dtype=float)

    def g_d(X: np.ndarray) -> np.ndarray:
        tenure_months = X[:, TENURE]
        sessions = X[:, SESS]
        spend = X[:, SPEND]
        premium = X[:, PREMIUM]
        urban = X[:, URBAN]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)

        # Smoothly increasing selection with log_spend; interactions make selection non-separable
        soft_spend = 1.0 * np.tanh(log_spend - np.log1p(60.0))
        return (
            0.8 * soft_spend
            + 0.2 * (log_sessions - np.log1p(5.0)) * np.tanh(tenure_months / 24.0 - 1.0)
            + 0.4 * premium * (urban - 0.5)
        )

    # 4) Heterogeneous, nonlinear treatment effect tau(X) on the natural scale (continuous outcome)
    def tau_fn(X: np.ndarray) -> np.ndarray:
        tenure_months = X[:, TENURE]
        sessions = X[:, SESS]
        spend = X[:, SPEND]
        premium = X[:, PREMIUM]
        urban = X[:, URBAN]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)

        # Base effect + stronger effect for higher sessions and premium users,
        # diminishes with tenure, mild modulation by spend and urban
        tau = (
            1.2
            + 0.6 * np.tanh(log_sessions - np.log1p(5.0))       # saturating in sessions
            + 0.4 * premium
            - 0.5 * np.tanh(tenure_months / 48.0)               # taper with long tenure
            + 0.2 * urban * np.tanh(log_spend - np.log1p(60.0))
        )
        return np.clip(tau, 0.1, 3.0)

    def hte_26_x_sampler(n: int, k: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Base features for derivation
        base_specs = [
            {"name": "tenure_months",     "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
            {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
            {"name": "spend_last_month",  "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
            {"name": "urban_resident",    "dist": "bernoulli", "p": 0.60},
        ]
        # Correlation among base features
        corr = np.array([
            [1.0, 0.3, 0.2, 0.0],
            [0.3, 1.0, 0.4, 0.0],
            [0.2, 0.4, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        X_base, _ = _gaussian_copula(rng, n, base_specs, corr)
        
        tenure = X_base[:, 0]
        sessions = X_base[:, 1]
        spend = X_base[:, 2]
        urban = X_base[:, 3]
        
        # Derive premium_user: more likely for high sessions, spend, and tenure
        logits_p = -5.0 + 0.7 * np.log1p(sessions) + 0.5 * np.log1p(spend) + 0.01 * tenure
        p_premium = 1.0 / (1.0 + np.exp(-logits_p))
        premium = rng.binomial(1, p_premium).astype(float)
        
        # Order: tenure, sessions, spend, premium, urban
        return np.column_stack([tenure, sessions, spend, premium, urban])

    # 5) Build generator
    gen = CausalDatasetGenerator(
        outcome_type="continuous",
        sigma_y=3.5,
        target_d_rate=0.35,  # enforce ~35% treated via intercept calibration
        seed=seed,
        # confounders
        confounder_specs=confounder_specs,
        x_sampler=hte_26_x_sampler,
        score_bounding=2.0,
        # Outcome/treatment structure
        beta_y=beta_y,
        beta_d=beta_d,
        g_y=g_y,
        g_d=g_d,
        # Heterogeneous effect
        tau=tau_fn,
        include_oracle=include_oracle,
    )

    # 6) Generate data_contracts
    df = gen.generate(n)

    if not return_causal_data:
        return df

    confounder_names = [s["name"] for s in confounder_specs]
    return CausalData(
        df=df,
        treatment="d",
        outcome="y",
        confounders=confounder_names,
    )
