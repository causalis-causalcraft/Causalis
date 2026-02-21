from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Union

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.dgp.multicausaldata.functional import generate_multitreatment

# Shared defaults across the 26 multi-treatment scenarios.
_D_NAMES = ["d_0", "d_1", "d_2"]
_TARGET_D_RATE = [0.5, 0.25, 0.25]
_COPULA_RHO = 0.30


def _toeplitz_copula_corr(n_features: int, rho: float = _COPULA_RHO) -> np.ndarray:
    # Positive-definite Toeplitz structure: nearby features are more correlated.
    idx = np.arange(n_features)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _run_multitreatment_26(
    *,
    n: int,
    seed: int,
    include_oracle: bool,
    return_causal_data: bool,
    outcome_type: str,
    confounder_specs: List[dict],
    beta_y: np.ndarray,
    beta_d: np.ndarray,
    theta: List[float],
    tau: List[Optional[Callable[[np.ndarray], np.ndarray]]],
    alpha_y: float = 0.0,
    gamma_shape: float = 2.0,
) -> Union[pd.DataFrame, MultiCausalData]:
    # Centralized wrapper so binary/gamma variants share calibration + naming conventions.
    return generate_multitreatment(
        n=n,
        n_treatments=3,
        outcome_type=outcome_type,
        alpha_y=alpha_y,
        gamma_shape=gamma_shape,
        tau=tau,
        target_d_rate=_TARGET_D_RATE,
        confounder_specs=confounder_specs,
        use_copula=True,
        copula_corr=_toeplitz_copula_corr(len(confounder_specs)),
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        random_state=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        d_names=_D_NAMES,
    )


def generate_multitreatment_gamma_26(
    n: int = 100_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    """
    Pre-configured multi-treatment dataset with Gamma-distributed outcome.

    - 3 treatment classes: control + 2 treatments
    - 8 confounders with realistic marginals
    - Gamma outcome with log-link linear confounding
    - Heterogeneous treatment effects and correlated confounders via Gaussian copula
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "normal",   "mu": 24, "sd": 12, "clip_min": 0, "clip_max": 120},
        {"name": "avg_sessions_week", "dist": "normal",   "mu": 5,  "sd": 2,  "clip_min": 0, "clip_max": 40},
        {"name": "spend_last_month",  "dist": "lognormal","mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "premium_user",      "dist": "bernoulli","p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli","p": 0.60},
        {"name": "support_tickets_q", "dist": "poisson",  "lam": 1.5, "clip_max": 15},
        {"name": "discount_eligible", "dist": "bernoulli","p": 0.35},
        {"name": "credit_utilization","dist": "beta",     "mean": 0.45, "kappa": 20.0},
    ]

    beta_y = np.array([0.01, 0.08, 0.0015, 0.35, 0.12, 0.06, 0.20, 0.50], dtype=float)

    beta_d = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.10, 0.0015, 0.50, 0.20, 0.05, 0.35, 0.40],
        [-0.005, 0.07, 0.0010, 0.35, 0.10, 0.08, 0.20, 0.25],
    ], dtype=float)

    theta = [0.0, -0.05, 0.10]

    def tau_d1(x: np.ndarray) -> np.ndarray:
        # Columns: 0=tenure, 1=sessions, 3=premium, 6=discount, 7=credit_utilization.
        tenure = np.clip(x[:, 0], 0.0, 120.0)
        sessions = np.clip(x[:, 1], 0.0, 40.0)
        premium = x[:, 3]
        discount = x[:, 6]
        credit = np.clip(x[:, 7], 0.0, 1.0)
        raw = -0.22 - 0.0010 * tenure - 0.006 * sessions - 0.05 * premium - 0.04 * discount - 0.10 * (credit - 0.45)
        # Enforce d_1 < d_0 on the link scale for all rows.
        return np.minimum(raw, -0.02)

    def tau_d2(x: np.ndarray) -> np.ndarray:
        # Columns: 1=sessions, 2=spend_last_month, 4=urban, 5=tickets, 7=credit_utilization.
        sessions = np.clip(x[:, 1], 0.0, 40.0)
        spend = np.log1p(np.clip(x[:, 2], 0.0, 500.0))
        urban = x[:, 4]
        tickets = np.clip(x[:, 5], 0.0, 15.0)
        credit = np.clip(x[:, 7], 0.0, 1.0)
        raw = 0.16 + 0.014 * sessions + 0.030 * spend + 0.06 * urban - 0.006 * tickets + 0.12 * (credit - 0.45)
        # Enforce d_2 > d_0 on the link scale for all rows.
        return np.maximum(raw, 0.02)

    tau = [None, tau_d1, tau_d2]

    return _run_multitreatment_26(
        n=n,
        outcome_type="gamma",
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        tau=tau,
        alpha_y=0.0,
        gamma_shape=2.0,
        seed=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
    )


def generate_multitreatment_binary_26(
    n: int = 100_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    """
    Pre-configured multi-treatment dataset with Binary outcome.

    - 3 treatment classes: control + 2 treatments
    - 8 confounders with realistic marginals
    - Binary outcome with logistic-link linear confounding
    - Heterogeneous treatment effects and correlated confounders via Gaussian copula
    """
    confounder_specs = [
        {"name": "tenure_months",      "dist": "normal",   "mu": 24, "sd": 12, "clip_min": 0, "clip_max": 120},
        {"name": "weekly_active_days", "dist": "normal",   "mu": 4.0, "sd": 1.5, "clip_min": 0, "clip_max": 7},
        {"name": "annual_income_k",    "dist": "gamma",    "shape": 4.0, "scale": 18.0, "clip_max": 300},
        {"name": "premium_user",       "dist": "bernoulli","p": 0.22},
        {"name": "family_plan",        "dist": "bernoulli","p": 0.38},
        {"name": "recent_complaints",  "dist": "poisson",  "lam": 0.8, "clip_max": 10},
        {"name": "discount_eligible",  "dist": "bernoulli","p": 0.30},
        {"name": "engagement_score",   "dist": "beta",     "mean": 0.60, "kappa": 16.0},
    ]

    beta_y = np.array([0.003, 0.11, 0.004, 0.40, -0.25, -0.12, 0.20, 0.90], dtype=float)

    beta_d = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.09, 0.0018, 0.45, 0.20, 0.08, 0.30, 0.28],
        [-0.004, 0.07, 0.0012, 0.30, 0.12, 0.10, 0.18, 0.22],
    ], dtype=float)

    theta = [0.0, -0.18, 0.26]

    def tau_d1(x: np.ndarray) -> np.ndarray:
        # Columns: 0=tenure, 1=weekly_active_days, 3=premium, 5=complaints, 7=engagement.
        tenure = np.clip(x[:, 0], 0.0, 120.0)
        active_days = np.clip(x[:, 1], 0.0, 7.0)
        premium = x[:, 3]
        complaints = np.clip(x[:, 5], 0.0, 10.0)
        engagement = np.clip(x[:, 7], 0.0, 1.0)
        raw = -0.16 - 0.0008 * tenure - 0.020 * active_days - 0.08 * premium - 0.03 * complaints - 0.10 * (engagement - 0.60)
        # Enforce d_1 < d_0 on the link scale for all rows.
        return np.minimum(raw, -0.02)

    def tau_d2(x: np.ndarray) -> np.ndarray:
        # Columns: 1=weekly_active_days, 2=annual_income_k, 4=family_plan, 5=complaints, 7=engagement.
        active_days = np.clip(x[:, 1], 0.0, 7.0)
        income = np.log1p(np.clip(x[:, 2], 0.0, 300.0))
        family_plan = x[:, 4]
        complaints = np.clip(x[:, 5], 0.0, 10.0)
        engagement = np.clip(x[:, 7], 0.0, 1.0)
        raw = 0.14 + 0.020 * active_days + 0.028 * income + 0.05 * family_plan - 0.010 * complaints + 0.12 * (engagement - 0.60)
        # Enforce d_2 > d_0 on the link scale for all rows.
        return np.maximum(raw, 0.02)

    tau = [None, tau_d1, tau_d2]

    return _run_multitreatment_26(
        n=n,
        outcome_type="binary",
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        tau=tau,
        # Keep baseline probability away from 0/1 saturation before treatment shifts.
        alpha_y=-1.1,
        seed=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
    )


def generate_multitreatment_irm_26(
    n: int = 100_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    # Backward-compatible alias.
    return generate_multitreatment_gamma_26(
        n=n,
        seed=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
    )
