from __future__ import annotations
import uuid
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.functional import generate_classic_rct, classic_rct_gamma
from causalis.dgp.base import _deterministic_ids

def generate_classic_rct_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = False,
    deterministic_ids: bool = True,
    **kwargs
):
    """
    A pre-configured classic RCT dataset with 3 binary confounders.
    n=10000, split=0.5, outcome is conversion (binary). Baseline control p=0.10
    and treatment p=0.11 are set on the log-odds scale (X=0), so marginal rates
    and ATE can differ once covariate effects are included. Includes a
    deterministic `user_id` column.

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
    n : int, default=10000
        Number of samples.
    split : float, default=0.5
        Proportion of samples assigned to the treatment group.
    outcome_params : dict, optional
        Binary outcome parameters, e.g. {"p": {"A": 0.10, "B": 0.11}}.
    add_ancillary : bool, default=False
        Whether to add standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default=True
        Whether to generate deterministic user IDs.
    **kwargs :
        Additional arguments passed to `generate_classic_rct`.

    Returns
    -------
    CausalData or pd.DataFrame
    """
    if outcome_params is None:
        outcome_params = {"p": {"A": 0.10, "B": 0.11}}
    df = generate_classic_rct(
        n=n,
        split=split,
        random_state=seed,
        outcome_params=outcome_params,
        add_pre=add_pre,
        beta_y=beta_y,
        outcome_depends_on_x=outcome_depends_on_x,
        include_oracle=include_oracle,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
        return_causal_data=False,
        **kwargs
    )
    if "user_id" not in df.columns:
        rng = np.random.default_rng(seed)
        if deterministic_ids:
            user_ids = _deterministic_ids(rng, len(df))
        else:
            user_ids = [uuid.uuid4().hex[:5] for _ in range(len(df))]
        df.insert(0, "user_id", user_ids)
    # The requirement asks for outcome - conversion(binary)
    df = df.rename(columns={"y": "conversion"})

    if not return_causal_data:
        return df

    exclude = {"conversion", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id"}
    confounders = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return CausalData(
        df=df,
        treatment="d",
        outcome="conversion",
        confounders=confounders,
        user_id="user_id"
    )


def classic_rct_gamma_26(
    seed: int = 42,
    add_pre: bool = False,
    beta_y: Optional[Union[List[float], np.ndarray]] = None,
    outcome_depends_on_x: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    *,
    n: int = 10000,
    split: float = 0.5,
    outcome_params: Optional[Dict] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = True,
    **kwargs
):
    """
    A pre-configured classic RCT dataset with a gamma outcome.
    n=10000, split=0.5, mean uplift ~10%.
    Includes deterministic `user_id` and ancillary columns.

    Parameters
    ----------
    seed : int, default=42
        Random seed.
    add_pre : bool, default=False
        Whether to generate a pre-period covariate ('y_pre').
    beta_y : array-like, optional
        Linear coefficients for confounders in the outcome model.
    outcome_depends_on_x : bool, default=True
        Whether to add default effects for confounders if beta_y is None.
    include_oracle : bool, default=False
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        Whether to return a CausalData object.
    n : int, default=10000
        Number of samples.
    split : float, default=0.5
        Proportion of samples assigned to the treatment group.
    outcome_params : dict, optional
        Gamma outcome parameters, e.g. {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}.
    add_ancillary : bool, default=True
        Whether to add standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default=True
        Whether to generate deterministic user IDs.
    **kwargs :
        Additional arguments passed to `classic_rct_gamma`.

    Returns
    -------
    CausalData or pd.DataFrame
    """
    if outcome_params is None:
        outcome_params = {"shape": 2.0, "scale": {"A": 15.0, "B": 16.5}}
    df = classic_rct_gamma(
        n=n,
        split=split,
        random_state=seed,
        outcome_params=outcome_params,
        add_pre=add_pre,
        beta_y=beta_y,
        outcome_depends_on_x=outcome_depends_on_x,
        include_oracle=include_oracle,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
        return_causal_data=False,
        **kwargs
    )
    if "user_id" not in df.columns:
        rng = np.random.default_rng(seed)
        if deterministic_ids:
            user_ids = _deterministic_ids(rng, len(df))
        else:
            user_ids = [uuid.uuid4().hex[:5] for _ in range(len(df))]
        df.insert(0, "user_id", user_ids)
    df = df.rename(columns={"y": "revenue"})

    if not return_causal_data:
        return df

    exclude = {"revenue", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "user_id"}
    confounders = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return CausalData(
        df=df,
        treatment="d",
        outcome="revenue",
        confounders=confounders,
        user_id="user_id"
    )
