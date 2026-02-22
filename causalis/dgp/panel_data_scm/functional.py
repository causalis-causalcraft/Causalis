from __future__ import annotations

from typing import Hashable, Literal, Optional, Union

import numpy as np
import pandas as pd

from causalis.data_contracts.panel_data_scm import PanelDataSCM
from .base import PanelSCMGenerator, PanelSCMGeneratorConfig


def _infer_pre_post_periods(
    *,
    n: int,
    n_donors: int,
    pre_share: float = 0.75,
    min_periods: int = 12,
    min_post_periods: int = 4,
) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be > 0.")
    if n_donors < 1:
        raise ValueError("n_donors must be >= 1.")
    if not (0.0 < pre_share < 1.0):
        raise ValueError("pre_share must be in (0, 1).")

    n_units = n_donors + 1
    periods = int(max(min_periods, round(n / n_units)))
    n_post = int(max(min_post_periods, round((1.0 - pre_share) * periods)))
    n_post = min(n_post, periods - 1)
    n_pre = periods - n_post
    return n_pre, n_post


def _inject_donor_missing_periods(
    *,
    df: pd.DataFrame,
    treated_unit: Hashable,
    random_state: int,
    donor_missing_block_frac: float,
    donor_missing_block_min_len: int,
    donor_missing_block_max_len: Optional[int],
) -> pd.DataFrame:
    """Inject contiguous missing-outcome periods for donor units only."""
    if donor_missing_block_frac <= 0.0:
        return df
    if not (0.0 <= donor_missing_block_frac < 1.0):
        raise ValueError("donor_missing_block_frac must be in [0, 1).")
    if donor_missing_block_min_len < 1:
        raise ValueError("donor_missing_block_min_len must be >= 1.")
    if donor_missing_block_max_len is not None and donor_missing_block_max_len < 1:
        raise ValueError("donor_missing_block_max_len must be >= 1 when provided.")
    if (
        donor_missing_block_max_len is not None
        and donor_missing_block_max_len < donor_missing_block_min_len
    ):
        raise ValueError("donor_missing_block_max_len must be >= donor_missing_block_min_len.")

    out = df.copy()
    donors = [u for u in out["unit_id"].unique().tolist() if u != treated_unit]
    if not donors:
        return out

    rng = np.random.default_rng(random_state)
    donor_mask = out["unit_id"] != treated_unit
    n_target = int(round(donor_missing_block_frac * int(donor_mask.sum())))
    if n_target <= 0:
        return out

    donor_idx = {
        unit: out[out["unit_id"] == unit].sort_values("time_id").index.to_numpy(dtype=int)
        for unit in donors
    }
    protected_set = {int(idx_arr[0]) for idx_arr in donor_idx.values() if idx_arr.size > 0}
    miss_set: set[int] = set()
    n_tries = max(100, 25 * n_target)
    for _ in range(n_tries):
        if len(miss_set) >= n_target:
            break
        unit = donors[int(rng.integers(0, len(donors)))]
        idx = donor_idx[unit]
        n_unit = int(idx.size)
        if n_unit <= 1:
            continue
        min_len = int(min(max(1, donor_missing_block_min_len), n_unit))
        max_len_candidate = n_unit if donor_missing_block_max_len is None else int(donor_missing_block_max_len)
        max_len = int(min(max_len_candidate, n_unit))
        if max_len < min_len:
            continue

        block_len = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, n_unit - block_len + 1))
        for idx_i in idx[start : start + block_len]:
            idx_int = int(idx_i)
            if idx_int in protected_set:
                continue
            miss_set.add(idx_int)
            if len(miss_set) >= n_target:
                break

    if miss_set:
        out.loc[list(miss_set), "y"] = np.nan
    return out


def _hide_internal_oracle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal oracle diagnostics that should not be exposed in DGP outputs."""
    return df.drop(columns=["tau_rate_true"], errors="ignore")


def _reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a canonical output schema order for public DGP returns."""
    lead = [
        "unit_id",
        "is_treated_unit",
        "time_id",
        "observed",
        "y",
        "y_cf",
        "tau_realized_true",
    ]
    ordered = [c for c in lead if c in df.columns]
    ordered.extend(c for c in df.columns if c not in ordered)
    return df.loc[:, ordered]


def _finalize_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Hide internal diagnostics and enforce public column ordering."""
    return _reorder_output_columns(_hide_internal_oracle_columns(df))


def generate_scm_data(
    n_donors: int = 5,
    n_pre_periods: int = 20,
    n_post_periods: int = 10,
    treatment_effect: float = 2.0,
    treatment_effect_slope: float = 0.0,
    donor_noise_std: float = 0.20,
    treated_noise_std: float = 0.10,
    common_factor_std: float = 0.15,
    time_start: int = 1,
    treated_unit: Hashable = "treated",
    donor_prefix: str = "donor_",
    random_state: Optional[int] = 42,
    missing_outcome_frac: float = 0.0,
    missing_cell_frac: float = 0.0,
    return_panel_data: bool = True,
    dirichlet_alpha: float = 1.0,
    rho_common: float = 0.0,
    rho_donor: float = 0.0,
    n_latent_factors: int = 0,
    latent_factor_std: float = 0.20,
    latent_loading_std: float = 0.35,
    rho_latent: float = 0.0,
    prefit_mismatch_std: float = 0.0,
    rho_prefit_mismatch: float = 0.0,
    missing_block_frac: float = 0.0,
    missing_block_min_len: int = 2,
    missing_block_max_len: Optional[int] = None,
    protect_treated_pre: bool = False,
    protect_treated_post: bool = False,
    treatment_effect_mode: Literal["additive", "multiplicative"] = "additive",
) -> Union[pd.DataFrame, PanelDataSCM]:
    """Medium-level wrapper for Gaussian SCM panel generation."""
    config = PanelSCMGeneratorConfig(
        outcome_distribution="gaussian",
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect=treatment_effect,
        treatment_effect_slope=treatment_effect_slope,
        donor_noise_std=donor_noise_std,
        treated_noise_std=treated_noise_std,
        common_factor_std=common_factor_std,
        time_start=time_start,
        treated_unit=treated_unit,
        donor_prefix=donor_prefix,
        random_state=random_state,
        missing_outcome_frac=missing_outcome_frac,
        missing_cell_frac=missing_cell_frac,
        return_panel_data=return_panel_data,
        dirichlet_alpha=dirichlet_alpha,
        rho_common=rho_common,
        rho_donor=rho_donor,
        n_latent_factors=n_latent_factors,
        latent_factor_std=latent_factor_std,
        latent_loading_std=latent_loading_std,
        rho_latent=rho_latent,
        prefit_mismatch_std=prefit_mismatch_std,
        rho_prefit_mismatch=rho_prefit_mismatch,
        missing_block_frac=missing_block_frac,
        missing_block_min_len=missing_block_min_len,
        missing_block_max_len=missing_block_max_len,
        protect_treated_pre=protect_treated_pre,
        protect_treated_post=protect_treated_post,
        treatment_effect_mode=treatment_effect_mode,
    )
    out = PanelSCMGenerator(config).generate()
    if isinstance(out, pd.DataFrame):
        return _finalize_output_df(out)
    out.df = _finalize_output_df(out.df)
    return out


def generate_scm_gamma_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.12,
    treatment_effect_slope: float = 0.01,
    missing_outcome_frac: float = 0.0,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Medium-level wrapper for realistic Gamma SCM panel generation.

    If `n_pre_periods`/`n_post_periods` are omitted, they are inferred from `n`.
    """
    if n_pre_periods is None or n_post_periods is None:
        n_pre_periods, n_post_periods = _infer_pre_post_periods(n=n, n_donors=n_donors)

    config_params = dict(
        outcome_distribution="gamma",
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        random_state=seed,
        missing_outcome_frac=missing_outcome_frac,
        return_panel_data=return_panel_data,
    )
    config_params.update(advanced_params)

    config = PanelSCMGeneratorConfig(**config_params)
    out = PanelSCMGenerator(config).generate()
    if isinstance(out, pd.DataFrame):
        return _finalize_output_df(out)
    out.df = _finalize_output_df(out.df)
    return out


def generate_scm_poisson_data(
    n: int = 432,
    seed: int = 42,
    return_panel_data: bool = True,
    n_donors: int = 8,
    treatment_effect_rate: float = 0.10,
    treatment_effect_slope: float = 0.005,
    donor_missing_block_frac: float = 0.08,
    donor_missing_block_min_len: int = 2,
    donor_missing_block_max_len: Optional[int] = 4,
    n_pre_periods: Optional[int] = None,
    n_post_periods: Optional[int] = None,
    **advanced_params,
) -> Union[pd.DataFrame, PanelDataSCM]:
    """
    Medium-level wrapper for realistic Poisson SCM panel generation.

    Default behavior injects donor-only missing periods, keeping treated post
    periods observed so RobustSyntheticControl can be exercised reliably.
    """
    if n_pre_periods is None or n_post_periods is None:
        n_pre_periods, n_post_periods = _infer_pre_post_periods(n=n, n_donors=n_donors)

    config_params = dict(
        outcome_distribution="poisson",
        n_donors=n_donors,
        n_pre_periods=n_pre_periods,
        n_post_periods=n_post_periods,
        treatment_effect_rate=treatment_effect_rate,
        treatment_effect_slope=treatment_effect_slope,
        random_state=seed,
        # We inject donor-only missingness below.
        missing_outcome_frac=0.0,
        missing_cell_frac=0.0,
        missing_block_frac=0.0,
        return_panel_data=False,
    )
    config_params.update(advanced_params)
    # Keep base generator missingness disabled; we inject donor-only periods below.
    config_params["missing_outcome_frac"] = 0.0
    config_params["missing_cell_frac"] = 0.0
    config_params["missing_block_frac"] = 0.0
    config_params["protect_treated_post"] = True

    config = PanelSCMGeneratorConfig(**config_params)
    df = PanelSCMGenerator(config).generate(return_panel_data=False)

    df = _inject_donor_missing_periods(
        df=df,
        treated_unit=config.treated_unit,
        random_state=seed + 11_813,
        donor_missing_block_frac=donor_missing_block_frac,
        donor_missing_block_min_len=donor_missing_block_min_len,
        donor_missing_block_max_len=donor_missing_block_max_len,
    )
    df = df.sort_values(["unit_id", "time_id"]).reset_index(drop=True)
    df["observed"] = (~df["y"].isna()).astype(int)
    df = _finalize_output_df(df)

    if not return_panel_data:
        return df

    donor_names = [f"{config.donor_prefix}{j + 1}" for j in range(config.n_donors)]
    return PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
        df=df,
        treated_unit=config.treated_unit,
        intervention_time=int(config.time_start + config.n_pre_periods),
        donor_units=donor_names,
        covariate_cols=("exposure", "macro_index", "seasonality_index"),
        observed_col="observed",
        allow_missing_outcome=True,
    )
