"""Single-trial organic-activation data for the IPCW ATT scenario."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import pandas as pd

from causalis.data_contracts.ipcw_data import IPCWData
from causalis.dgp.organic_activation import (
    generate_organic_activation_data,
    make_ipcw_view,
)


def generate_ipcw_data(
    n_clients: int = 5_000,
    n_periods: int = 24,
    g: int = 10,
    horizon: int = 6,
    scenario: str = "baseline",
    seed: int | None = 42,
    return_ipcw_data: bool = True,
    **dgp_params: Any,
) -> tuple[IPCWData | pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Generate one artificial-censoring trial and a separate oracle.

    Parameters
    ----------
    n_clients, n_periods : int
        Number of clients and calendar periods in the underlying panel.
    g, horizon : int
        Trial baseline period and complete follow-up length, including baseline.
    scenario : str
        An organic-activation scenario, such as ``"baseline"``,
        ``"observed_timevarying_confounding"`` or ``"null_effect"``.
    seed : int or None
        Seed passed to the organic-activation generator.
    return_ipcw_data : bool
        Return :class:`IPCWData` by default; use ``False`` for the trial frame.
    **dgp_params
        Additional ``OrganicActivationConfig`` fields. ``main_cohorts`` is fixed
        to ``(g,)`` so that the aggregate oracle matches the returned trial.

    Returns
    -------
    data : IPCWData or pandas.DataFrame
        The trial returned by ``make_ipcw_view``. Its model-feature allowlists
        contain observed variables only.
    oracle : dict
        Original organic-activation oracle, kept separate from model data.
        ``att_aggregate['att_no_activation']`` contains monthly true effects;
        their sum is the total ATT targeted by this scenario.
    metadata : dict
        Organic-activation design metadata with ``main_cohorts == (g,)``.

    Notes
    -----
    The number of generated panel clients can exceed the trial's baseline
    population: clients must have entered before ``g`` and remain untreated
    before its activation decision. Empty random treatment cohorts are not
    resampled. Changing the selection scenario may invalidate IPCW's identifying
    assumptions even though the oracle remains available for comparison.
    """
    if isinstance(g, bool) or not isinstance(g, Integral) or g < 1:
        raise ValueError("g must be an integer >= 1.")
    if not isinstance(return_ipcw_data, bool):
        raise ValueError("return_ipcw_data must be a boolean.")
    g = int(g)
    requested_cohorts = dgp_params.pop("main_cohorts", (g,))
    try:
        matching_cohort = tuple(requested_cohorts) == (g,)
    except TypeError:
        matching_cohort = False
    if not matching_cohort:
        raise ValueError("main_cohorts is fixed to (g,) for a single IPCW trial.")

    panel, oracle, metadata = generate_organic_activation_data(
        n_clients=n_clients,
        n_periods=n_periods,
        horizon=horizon,
        scenario=scenario,
        seed=seed,
        main_cohorts=(g,),
        **dgp_params,
    )
    view = make_ipcw_view(panel, g=g, horizon=horizon)
    data = IPCWData.from_df(view) if return_ipcw_data else view
    return data, oracle, metadata


__all__ = ["generate_ipcw_data"]
