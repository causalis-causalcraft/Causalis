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
        target_d_rate=0.05,
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

def generate_obs_hte_26_rich(
    n: int = 100000,
    seed: int = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True
) -> Union[pd.DataFrame, CausalData]:
    """
    Observational dataset with richer confounding, nonlinear outcome model,
    nonlinear treatment assignment, and heterogeneous treatment effects.
    Adds additional realistic covariates and dependencies to mimic real data.

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
        {"name": "tenure_months",        "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
        {"name": "avg_sessions_week",    "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
        {"name": "spend_last_month",     "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "age_years",            "dist": "normal",    "mu": 36, "sd": 12, "clip_min": 18, "clip_max": 80},
        {"name": "income_monthly",       "dist": "lognormal", "mu": np.log(4000), "sigma": 0.5, "clip_max": 20000},
        {"name": "prior_purchases_12m",  "dist": "poisson",   "lam": 3, "clip_max": 50},
        {"name": "support_tickets_90d",  "dist": "poisson",   "lam": 1.5, "clip_max": 20},
        {"name": "premium_user",         "dist": "bernoulli", "p": 0.25},
        {"name": "mobile_user",          "dist": "bernoulli", "p": 0.65},
        {"name": "urban_resident",       "dist": "bernoulli", "p": 0.60},
        {"name": "referred_user",        "dist": "bernoulli", "p": 0.20},
    ]

    TENURE, SESS, SPEND, AGE, INCOME, PURCHASES, TICKETS, PREMIUM, MOBILE, URBAN, REFERRED = range(11)

    beta_y = np.array([
        0.005,   # tenure_months
        0.00,    # avg_sessions_week (moved to g_y)
        0.00,    # spend_last_month (moved to g_y)
        0.02,    # age_years
        0.00005, # income_monthly
        0.08,    # prior_purchases_12m
        -0.55,   # support_tickets_90d
        1.10,    # premium_user
        0.40,    # mobile_user
        0.60,    # urban_resident
        0.30,    # referred_user
    ], dtype=float)

    def g_y(X: np.ndarray) -> np.ndarray:
        tenure = X[:, TENURE]
        sessions = X[:, SESS]
        spend = X[:, SPEND]
        age = X[:, AGE]
        income = X[:, INCOME]
        purchases = X[:, PURCHASES]
        tickets = X[:, TICKETS]
        premium = X[:, PREMIUM]
        mobile = X[:, MOBILE]
        urban = X[:, URBAN]
        referred = X[:, REFERRED]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)
        log_income = np.log1p(income)
        log_purchases = np.log1p(purchases)

        return (
            1.4 * np.tanh(tenure / 24.0)
            + 0.6 * (log_sessions - np.log1p(5.0)) ** 2
            + 0.25 * (log_spend - np.log1p(60.0)) * (log_sessions - np.log1p(5.0))
            + 0.35 * np.tanh((log_income - np.log1p(4000.0)) / 1.5)
            - 0.45 * np.log1p(tickets)
            + 0.20 * log_purchases * np.tanh(tenure / 18.0)
            + 0.30 * mobile * (log_sessions - np.log1p(5.0))
            + 0.25 * premium * (urban - 0.5)
            + 0.15 * referred * np.tanh(tenure / 12.0 - 1.0)
            - 0.20 * np.tanh((age - 40.0) / 15.0)
        )

    beta_d = np.array([
        0.002,   # tenure_months
        0.00,    # avg_sessions_week (moved to g_d)
        0.00,    # spend_last_month (moved to g_d)
        -0.01,   # age_years
        0.00003, # income_monthly
        0.05,    # prior_purchases_12m
        0.10,    # support_tickets_90d
        0.70,    # premium_user
        0.30,    # mobile_user
        0.20,    # urban_resident
        0.40,    # referred_user
    ], dtype=float)

    def g_d(X: np.ndarray) -> np.ndarray:
        tenure = X[:, TENURE]
        sessions = X[:, SESS]
        spend = X[:, SPEND]
        age = X[:, AGE]
        tickets = X[:, TICKETS]
        premium = X[:, PREMIUM]
        mobile = X[:, MOBILE]
        urban = X[:, URBAN]
        referred = X[:, REFERRED]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)

        soft_spend = 0.9 * np.tanh(log_spend - np.log1p(60.0))
        return (
            soft_spend
            + 0.25 * (log_sessions - np.log1p(5.0)) * np.tanh(tenure / 24.0 - 1.0)
            + 0.35 * premium * (urban - 0.5)
            + 0.30 * mobile * np.tanh(log_sessions - np.log1p(3.0))
            + 0.25 * referred * (1.0 - np.tanh(tenure / 36.0))
            + 0.15 * np.tanh(np.log1p(tickets) - np.log1p(1.5))
            - 0.10 * np.tanh((age - 45.0) / 12.0)
        )

    def tau_fn(X: np.ndarray) -> np.ndarray:
        tenure = X[:, TENURE]
        sessions = X[:, SESS]
        spend = X[:, SPEND]
        age = X[:, AGE]
        tickets = X[:, TICKETS]
        premium = X[:, PREMIUM]
        mobile = X[:, MOBILE]
        urban = X[:, URBAN]
        referred = X[:, REFERRED]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)

        tau = (
            1.1
            + 0.55 * np.tanh(log_sessions - np.log1p(5.0))
            + 0.90 * premium
            + 0.15 * mobile
            + 0.20 * referred
            - 0.45 * np.tanh(tenure / 48.0)
            - 0.25 * np.tanh((age - 40.0) / 15.0)
            + 0.15 * urban * np.tanh(log_spend - np.log1p(60.0))
            - 0.15 * np.tanh(np.log1p(tickets) - np.log1p(1.5))
        )
        return np.clip(tau, 0.05, 3.0)

    def hte_26_rich_x_sampler(n: int, k: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        base_specs = [
            {"name": "tenure_months",     "dist": "lognormal", "mu": np.log(24), "sigma": 0.6, "clip_max": 120},
            {"name": "avg_sessions_week", "dist": "negbin",    "mu": 5, "alpha": 0.5, "clip_max": 40},
            {"name": "spend_last_month",  "dist": "lognormal", "mu": np.log(60), "sigma": 0.9, "clip_max": 500},
            {"name": "age_years",         "dist": "normal",    "mu": 36, "sd": 12, "clip_min": 18, "clip_max": 80},
            {"name": "income_monthly",    "dist": "lognormal", "mu": np.log(4000), "sigma": 0.5, "clip_max": 20000},
            {"name": "urban_resident",    "dist": "bernoulli", "p": 0.60},
        ]
        corr = np.array([
            [1.00, 0.20, 0.20, 0.30, 0.20, 0.00],
            [0.20, 1.00, 0.45, -0.20, 0.30, 0.10],
            [0.20, 0.45, 1.00, -0.10, 0.40, 0.10],
            [0.30, -0.20, -0.10, 1.00, 0.20, -0.10],
            [0.20, 0.30, 0.40, 0.20, 1.00, 0.10],
            [0.00, 0.10, 0.10, -0.10, 0.10, 1.00],
        ])
        X_base, _ = _gaussian_copula(rng, n, base_specs, corr)

        tenure = X_base[:, 0]
        sessions = X_base[:, 1]
        spend = X_base[:, 2]
        age = X_base[:, 3]
        income = X_base[:, 4]
        urban = X_base[:, 5]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)
        log_income = np.log1p(income)

        logits_p = -5.0 + 0.7 * log_sessions + 0.45 * log_spend + 0.35 * log_income + 0.015 * tenure
        p_premium = 1.0 / (1.0 + np.exp(-logits_p))
        premium = rng.binomial(1, p_premium).astype(float)

        logits_m = 1.5 - 0.03 * (age - 35.0) + 0.30 * urban + 0.25 * log_sessions
        p_mobile = 1.0 / (1.0 + np.exp(-logits_m))
        mobile = rng.binomial(1, p_mobile).astype(float)

        logits_r = -1.2 - 0.02 * (tenure - 12.0) + 0.45 * mobile + 0.20 * urban
        p_referred = 1.0 / (1.0 + np.exp(-logits_r))
        referred = rng.binomial(1, p_referred).astype(float)

        mean_purchases = 1.0 + 0.55 * log_sessions + 0.45 * log_spend + 0.25 * premium
        purchases = rng.poisson(lam=np.clip(mean_purchases, 0.1, 30.0)).astype(float)

        mean_tickets = 0.6 + 0.25 * log_sessions + 0.30 * (1.0 - premium) + 0.15 * np.tanh((age - 45.0) / 12.0)
        tickets = rng.poisson(lam=np.clip(mean_tickets, 0.05, 10.0)).astype(float)

        purchases = np.clip(purchases, 0, 50)
        tickets = np.clip(tickets, 0, 20)

        return np.column_stack([
            tenure,
            sessions,
            spend,
            age,
            income,
            purchases,
            tickets,
            premium,
            mobile,
            urban,
            referred,
        ])

    def g_zi(X: np.ndarray) -> np.ndarray:
        sessions = X[:, SESS]
        spend = X[:, SPEND]
        purchases = X[:, PURCHASES]
        tickets = X[:, TICKETS]
        premium = X[:, PREMIUM]
        mobile = X[:, MOBILE]

        log_sessions = np.log1p(sessions)
        log_spend = np.log1p(spend)
        log_purchases = np.log1p(purchases)

        return (
            0.45 * np.tanh(log_sessions - np.log1p(4.0))
            + 0.25 * np.tanh(log_spend - np.log1p(50.0))
            + 0.20 * log_purchases
            - 0.30 * np.tanh(np.log1p(tickets) - np.log1p(1.0))
            + 0.15 * premium
            + 0.10 * mobile
        )

    def tau_zi_fn(X: np.ndarray) -> np.ndarray:
        tenure = X[:, TENURE]
        premium = X[:, PREMIUM]
        referred = X[:, REFERRED]

        return (
            0.15
            + 0.10 * premium
            + 0.05 * referred
            - 0.10 * np.tanh(tenure / 48.0)
        )

    gen = CausalDatasetGenerator(
        outcome_type="tweedie",
        sigma_y=3.8,
        target_d_rate=0.05,
        seed=seed,
        confounder_specs=confounder_specs,
        x_sampler=hte_26_rich_x_sampler,
        score_bounding=2.0,
        beta_y=beta_y,
        beta_d=beta_d,
        g_y=g_y,
        g_d=g_d,
        tau=tau_fn,
        alpha_zi=-0.8,
        beta_zi=np.array([
            0.00,   # tenure_months
            0.02,   # avg_sessions_week
            0.00,   # spend_last_month
            -0.01,  # age_years
            0.00002,# income_monthly
            0.10,   # prior_purchases_12m
            -0.15,  # support_tickets_90d
            0.30,   # premium_user
            0.20,   # mobile_user
            0.05,   # urban_resident
            0.15,   # referred_user
        ], dtype=float),
        g_zi=g_zi,
        tau_zi=tau_zi_fn,
        pos_dist="gamma",
        gamma_shape=2.2,
        include_oracle=include_oracle,
    )

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
