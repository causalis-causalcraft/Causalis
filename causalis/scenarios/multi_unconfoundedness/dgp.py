from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.dgp.multicausaldata.functional import generate_multitreatment


def generate_multitreatment_irm_26(
    n: int = 10_000,
    seed: int = 42,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, MultiCausalData]:
    """
    Pre-configured multi-treatment dataset suitable for MultiTreatmentIRM.

    - 3 treatment classes: control + 2 treatments
    - 5 confounders with realistic marginals
    - Continuous outcome with linear confounding
    """
    confounder_specs = [
        {"name": "tenure_months",     "dist": "normal",   "mu": 24, "sd": 12, "clip_min": 0, "clip_max": 120},
        {"name": "avg_sessions_week", "dist": "normal",   "mu": 5,  "sd": 2,  "clip_min": 0, "clip_max": 40},
        {"name": "spend_last_month",  "dist": "lognormal","mu": np.log(60), "sigma": 0.9, "clip_max": 500},
        {"name": "premium_user",      "dist": "bernoulli","p": 0.25},
        {"name": "urban_resident",    "dist": "bernoulli","p": 0.60},
    ]

    beta_y = np.array([0.05, 0.40, 0.02, 2.00, 1.00], dtype=float)

    beta_d = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.02, 0.12, 0.003, 0.80, 0.40],
        [-0.01, 0.08, 0.001, 0.50, 0.20],
    ], dtype=float)

    theta = [0.0, 1.5, 2.2]

    return generate_multitreatment(
        n=n,
        n_treatments=3,
        outcome_type="continuous",
        sigma_y=3.5,
        target_d_rate=[0.5, 0.25, 0.25],
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        random_state=seed,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        treatment_names=["t_0", "t_1", "t_2"],
    )
