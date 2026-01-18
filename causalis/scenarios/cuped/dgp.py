from __future__ import annotations
import pandas as pd
from typing import Optional, Union
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.functional import make_cuped_tweedie
from causalis.dgp.causaldata.preperiod import PreCorrSpec

def make_cuped_tweedie_26(
    n: int = 100000,
    seed: int = 42,
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_target_corr: float = 0.6,
    pre_spec: Optional[PreCorrSpec] = None,
    include_oracle: bool = False,
    return_causal_data: bool = True,
    theta_log: float = 0.2
) -> Union[pd.DataFrame, CausalData]:
    """
    Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
    Features many zeros and a heavy right tail. 
    Includes a pre-period covariate 'y_pre' by default, making it suitable for CUPED benchmarks.
    Wrapper for make_tweedie().

    Parameters
    ----------
    n : int, default=10000
        Number of samples to generate.
    seed : int, default=42
        Random seed.
    add_pre : bool, default=True
        Whether to add a pre-period covariate 'y_pre'.
    pre_name : str, default="y_pre"
        Name of the pre-period covariate column.
    pre_target_corr : float, default=0.6
        Target correlation between y_pre and post-outcome y in control group.
    pre_spec : PreCorrSpec, optional
        Detailed specification for pre-period calibration (transform, method, etc.).
    include_oracle : bool, default=False
        Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
    return_causal_data : bool, default=True
        Whether to return a CausalData object.
    theta_log : float, default=0.12
        The log-uplift theta parameter for the treatment effect.

    Returns
    -------
    pd.DataFrame or CausalData
    """
    return make_cuped_tweedie(
        n=n,
        seed=seed,
        add_pre=add_pre,
        pre_name=pre_name,
        pre_target_corr=pre_target_corr,
        pre_spec=pre_spec,
        include_oracle=include_oracle,
        return_causal_data=return_causal_data,
        theta_log=theta_log
    )
