from __future__ import annotations

from typing import Union

import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from causalis.dgp.panel_data_scm import generate_scm_gamma_data, generate_scm_poisson_data


_ORACLE_COLS = (
    "is_treated_unit",
    "y_cf",
    "tau_realized_true",
    "mu_cf",
    "mu_treated",
    "tau_mean_true",
)
_COVARIATE_COLS = ("exposure", "macro_index", "seasonality_index")


def _apply_include_oracles(
    out: Union[pd.DataFrame, PanelDataSCM],
    *,
    include_oracles: bool,
) -> Union[pd.DataFrame, PanelDataSCM]:
    drop_cols = list(_COVARIATE_COLS)
    if not include_oracles:
        drop_cols.extend(_ORACLE_COLS)
    if isinstance(out, pd.DataFrame):
        return out.drop(columns=drop_cols, errors="ignore")
    out.df = out.df.drop(columns=drop_cols, errors="ignore")
    return out


def generate_scm_gamma_26(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.12,
    treatment_effect_slope: float = 0.01,
    missing_outcome_frac: float = 0.0,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Generate realistic Gamma synthetic-control panel data.

    Parameters
    ----------
    n : int, default=432
        Target total number of panel rows used to infer a pre/post horizon when
        `n_pre_periods` and `n_post_periods` are not provided.
    seed : int, default=42
        Random seed.
    return_panel_data : bool, default=True
        If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
        object. If False, return a pandas DataFrame.
    include_oracles : bool, default=False
        Whether to include oracle truth columns in the returned data:
        `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
        `mu_treated`, `tau_mean_true`.
        Scenario-level outputs always exclude synthetic covariates
        `exposure`, `macro_index`, `seasonality_index`.
    n_donors : int, default=8
        Number of donor units.
    treatment_effect_rate : float, default=0.12
        Baseline post-treatment relative effect level used by the DGP.
    treatment_effect_slope : float, default=0.01
        Linear slope of the post-treatment relative effect path.
    missing_outcome_frac : float, default=0.0
        Fraction of outcomes to mask as missing in the base generator.
    **advanced_params
        Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_gamma_data`.
        Common advanced knobs include `n_pre_periods`, `n_post_periods`,
        and `time_start`.

    Returns
    -------
    pandas.DataFrame or PanelDataSCM
        Long panel data for SCM experiments.

    Notes
    -----
    Time-axis semantics:

    - `n_pre_periods`: number of periods strictly before treatment.
    - `time_start`: first value of `time_id` (default from low-level generator is 1).
    - `intervention_time`: first post-treatment period boundary, computed as
      ``intervention_time = time_start + n_pre_periods``.
    - With this function's default arguments (`n=432`, `n_donors=8`) and inferred
      horizon, the explicit values are:
      ``n_pre_periods=36``, ``n_post_periods=12``, ``time_start=1``,
      ``intervention_time=37``.
    """
    out = generate_scm_gamma_data(
        n=n,
        seed=seed,
        return_panel_data=return_panel_data,
        n_donors=n_donors,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        missing_outcome_frac=missing_outcome_frac,
        **advanced_params,
    )
    return _apply_include_oracles(out, include_oracles=include_oracles)


def generate_scm_poisson_26(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    include_oracles: bool = False,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.10,
    treatment_effect_slope: float = 0.005,
    donor_missing_block_frac: float = 0.08,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Generate realistic Poisson synthetic-control panel data.

    Parameters
    ----------
    n : int, default=432
        Target total number of panel rows used to infer a pre/post horizon when
        `n_pre_periods` and `n_post_periods` are not provided.
    seed : int, default=42
        Random seed.
    return_panel_data : bool, default=True
        If True, return a :class:`~causalis.data_contracts.panel_data_scm.PanelDataSCM`
        object. If False, return a pandas DataFrame.
    include_oracles : bool, default=False
        Whether to include oracle truth columns in the returned data:
        `is_treated_unit`, `y_cf`, `tau_realized_true`, `mu_cf`,
        `mu_treated`, `tau_mean_true`.
        Scenario-level outputs always exclude synthetic covariates
        `exposure`, `macro_index`, `seasonality_index`.
    n_donors : int, default=8
        Number of donor units.
    treatment_effect_rate : float, default=0.10
        Baseline post-treatment relative effect level used by the DGP.
    treatment_effect_slope : float, default=0.005
        Linear slope of the post-treatment relative effect path.
    donor_missing_block_frac : float, default=0.08
        Fraction of donor-only rows to mask via contiguous missing-time blocks.
    **advanced_params
        Forwarded to :func:`causalis.dgp.panel_data_scm.generate_scm_poisson_data`.
        Common advanced knobs include `n_pre_periods`, `n_post_periods`,
        and `time_start`.

    Returns
    -------
    pandas.DataFrame or PanelDataSCM
        Long panel data for SCM experiments.

    Notes
    -----
    Time-axis semantics:

    - `n_pre_periods`: number of periods strictly before treatment.
    - `time_start`: first value of `time_id` (default from low-level generator is 1).
    - `intervention_time`: first post-treatment period boundary, computed as
      ``intervention_time = time_start + n_pre_periods``.
    - With this function's default arguments (`n=432`, `n_donors=8`) and inferred
      horizon, the explicit values are:
      ``n_pre_periods=36``, ``n_post_periods=12``, ``time_start=1``,
      ``intervention_time=37``.
    """
    out = generate_scm_poisson_data(
        n=n,
        seed=seed,
        return_panel_data=return_panel_data,
        n_donors=n_donors,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        donor_missing_block_frac=donor_missing_block_frac,
        **advanced_params,
    )
    return _apply_include_oracles(out, include_oracles=include_oracles)
