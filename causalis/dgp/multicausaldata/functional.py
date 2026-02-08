from __future__ import annotations

import pandas as pd
from typing import Optional, Union, List, Dict, Any

from causalis.data_contracts.multicausaldata import MultiCausalData
from .base import MultiCausalDatasetGenerator


def generate_multitreatment(
    n: int = 10_000,
    n_treatments: int = 3,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    target_d_rate: Optional[Union[List[float], Any]] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[Any] = None,
    beta_d: Optional[Any] = None,
    theta: Optional[Any] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Any] = None,
    include_oracle: bool = True,
    return_causal_data: bool = False,
    treatment_names: Optional[List[str]] = None,
) -> Union[pd.DataFrame, MultiCausalData]:
    """
    Generate a multi-treatment dataset using MultiCausalDatasetGenerator.

    Parameters
    ----------
    n : int, default=10_000
        Number of samples.
    n_treatments : int, default=3
        Number of treatment classes (including control).
    outcome_type : {"continuous", "binary", "poisson"}, default="continuous"
        Outcome family.
    sigma_y : float, default=1.0
        Noise level for continuous outcomes.
    target_d_rate : array-like, optional
        Target marginal class probabilities (length K).
    confounder_specs : list of dict, optional
        Schema for confounder distributions.
    beta_y : array-like, optional
        Linear coefficients for outcome model.
    beta_d : array-like, optional
        Linear coefficients for treatment model.
    theta : float or array-like, optional
        Constant treatment effects per class.
    random_state : int, optional
        Random seed.
    k : int, default=0
        Number of confounders if confounder_specs is None.
    x_sampler : callable, optional
        Custom sampler for confounders.
    include_oracle : bool, default=True
        Whether to include oracle columns.
    return_causal_data : bool, default=False
        Whether to return a MultiCausalData object.
    treatment_names : list of str, optional
        Names of treatment columns.

    Returns
    -------
    pd.DataFrame or MultiCausalData
    """
    gen = MultiCausalDatasetGenerator(
        n_treatments=n_treatments,
        treatment_names=treatment_names,
        outcome_type=outcome_type,
        sigma_y=sigma_y,
        target_d_rate=target_d_rate,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        theta=theta,
        seed=random_state,
        k=int(k),
        x_sampler=x_sampler,
        include_oracle=include_oracle,
    )
    df = gen.generate(n)

    if not return_causal_data:
        return df

    return MultiCausalData(
        df=df,
        outcome="y",
        treatments=gen.treatment_names,
        confounders=gen.confounder_names_,
    )
