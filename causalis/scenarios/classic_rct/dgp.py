from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, List, Union
from causalis.dgp.causaldata import CausalData
from causalis.dgp.causaldata.functional import generate_classic_rct

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
