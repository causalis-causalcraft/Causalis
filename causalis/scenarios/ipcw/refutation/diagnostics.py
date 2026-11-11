"""Support and weight diagnostics for fixed-horizon IPCW estimates.

These checks describe observed support and fitted weights. They cannot verify
baseline or sequential exchangeability, or establish population positivity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causalis.data_contracts.ipcw_data import IPCWData
from causalis.data_contracts.ipcw_estimate import IPCWEstimate

_SUPPORT_COLUMNS = [
    "trial_id",
    "time",
    "followup_month",
    "n_treated",
    "n_control",
    "n_control_at_risk",
    "n_censored",
    "n_control_usable",
    "retention_share",
    "is_supported",
]
_WEIGHT_COLUMNS = [
    "trial_id",
    "time",
    "followup_month",
    "n_control_usable",
    "control_weight_sum",
    "control_ess",
    "weight_mean",
    "weight_std",
    "weight_min",
    "weight_max",
    "propensity_clipping_share",
    "censoring_clipping_share",
    "min_survival",
]


def _diagnostic_data(estimate: IPCWEstimate):
    if not isinstance(estimate, IPCWEstimate):
        raise TypeError("estimate must be an IPCWEstimate instance.")
    if estimate.diagnostic_data is None:
        raise ValueError(
            "Fitted diagnostics are unavailable. Refit with store_diagnostics=True."
        )
    return estimate.diagnostic_data


def _data_rows(data: IPCWData) -> pd.DataFrame:
    roles = {
        data.unit_col: "unit_id",
        data.trial_col: "trial_id",
        data.time_col: "time",
        data.followup_col: "followup_month",
        data.treatment: "treatment",
        data.at_risk_col: "at_risk",
        data.censor_col: "censor_now",
        data.outcome_usable_col: "outcome_usable",
    }
    return data.get_df()[list(roles)].rename(columns=roles)


def _support_table_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Build observed support from the standardized fit-time row table."""
    records: list[dict[str, Any]] = []
    for key, part in rows.groupby(
        ["trial_id", "time", "followup_month"], sort=True, observed=True
    ):
        controls = part["treatment"].eq(0)
        n_treated = int(part.loc[~controls, "unit_id"].nunique())
        n_control = int(part.loc[controls, "unit_id"].nunique())
        usable = controls & part["outcome_usable"].astype(bool)
        n_usable = int(part.loc[usable, "unit_id"].nunique())
        records.append(
            {
                "trial_id": key[0],
                "time": key[1],
                "followup_month": key[2],
                "n_treated": n_treated,
                "n_control": n_control,
                "n_control_at_risk": int(
                    (controls & part["at_risk"].astype(bool)).sum()
                ),
                "n_censored": int((controls & part["censor_now"].astype(bool)).sum()),
                "n_control_usable": n_usable,
                "retention_share": n_usable / n_control if n_control else float("nan"),
                "is_supported": n_treated > 0 and n_usable > 0,
            }
        )
    return pd.DataFrame(records, columns=_SUPPORT_COLUMNS)


def _weight_summary_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize weights on usable controls, without counting censored zeros."""
    records: list[dict[str, Any]] = []
    for key, part in rows.groupby(
        ["trial_id", "time", "followup_month"], sort=True, observed=True
    ):
        controls = part["treatment"].eq(0)
        usable = controls & part["outcome_usable"].astype(bool)
        risk = controls & part["at_risk"].astype(bool)
        weights = part.loc[usable, "control_weight"].to_numpy(dtype=float)
        valid = weights.size > 0 and np.isfinite(weights).all() and (weights > 0).all()
        if valid:
            # Rescaling avoids overflow in the ESS sum of squares.
            scaled = weights / weights.max()
            ess = float(scaled.sum() ** 2 / np.dot(scaled, scaled))
            moments = {
                "control_weight_sum": float(weights.sum()),
                "control_ess": ess,
                "weight_mean": float(weights.mean()),
                "weight_std": float(weights.std(ddof=0)),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
            }
        else:
            moments = {
                name: float("nan")
                for name in (
                    "control_weight_sum",
                    "control_ess",
                    "weight_mean",
                    "weight_std",
                    "weight_min",
                    "weight_max",
                )
            }
        survival = part.loc[usable, "survival"].to_numpy(dtype=float)
        records.append(
            {
                "trial_id": key[0],
                "time": key[1],
                "followup_month": key[2],
                "n_control_usable": int(usable.sum()),
                **moments,
                "propensity_clipping_share": float(part["propensity_clipped"].mean()),
                "censoring_clipping_share": (
                    float(part.loc[risk, "censoring_clipped"].mean())
                    if risk.any()
                    else 0.0
                ),
                "min_survival": (
                    float(survival.min()) if survival.size else float("nan")
                ),
            }
        )
    return pd.DataFrame(records, columns=_WEIGHT_COLUMNS)


def ipcw_support_table(data: IPCWData | IPCWEstimate) -> pd.DataFrame:
    """Return monthly treated counts, censoring events and usable-control support.

    Accepts a validated trial before fitting or an estimate retaining diagnostics.
    ``retention_share`` is the observed fraction of baseline controls whose
    outcomes remain usable, rather than a model-predicted survival probability.
    Returned tables are independent copies of stored diagnostic tables.
    """
    if isinstance(data, IPCWData):
        return _support_table_from_rows(_data_rows(data))
    if isinstance(data, IPCWEstimate):
        diagnostics = _diagnostic_data(data)
        if not diagnostics.support.empty:
            return diagnostics.support.copy(deep=True)
        return _support_table_from_rows(diagnostics.row_data)
    raise TypeError("data must be an IPCWData or IPCWEstimate instance.")


def ipcw_weight_summary_table(estimate: IPCWEstimate) -> pd.DataFrame:
    """Return monthly weight moments, control ESS, survival and clipping shares.

    Weight moments and ESS use only controls with usable outcomes. Propensity
    clipping shares include both baseline groups; censoring clipping shares use
    only controls at risk in that month. An empty risk set has clipping share 0.
    """
    diagnostics = _diagnostic_data(estimate)
    if not diagnostics.weight_summary.empty:
        return diagnostics.weight_summary.copy(deep=True)
    return _weight_summary_from_rows(diagnostics.row_data)


def _positive_threshold(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive.")
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _share_threshold(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and in [0, 1].")
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return number


def run_ipcw_diagnostics(
    data: IPCWData | IPCWEstimate,
    *,
    min_treated: float = 20,
    min_control: float = 20,
    min_control_retention: float = 0.25,
    min_control_ess: float = 20,
    max_clipping_share: float = 0.05,
) -> pd.DataFrame:
    """Flag observed support and, after fitting, weight instability.

    Returns ``test``, ``flag`` (GREEN/YELLOW/RED), ``value``, ``threshold`` and
    ``message`` columns. RED indicates missing support or invalid weights;
    YELLOW indicates a requested diagnostic threshold was exceeded. Data-only
    calls run support checks, without implying that fitted weights were checked.
    These diagnostics do not test exchangeability or prove causal identification.
    """
    min_treated = _positive_threshold(min_treated, "min_treated")
    min_control = _positive_threshold(min_control, "min_control")
    min_control_ess = _positive_threshold(min_control_ess, "min_control_ess")
    min_control_retention = _share_threshold(
        min_control_retention, "min_control_retention"
    )
    max_clipping_share = _share_threshold(max_clipping_share, "max_clipping_share")
    support = ipcw_support_table(data)
    checks: list[dict[str, Any]] = []

    def append(test: str, value: Any, threshold: str, flag: str, message: str) -> None:
        checks.append(
            {
                "test": test,
                "flag": flag,
                "value": value,
                "threshold": threshold,
                "message": message,
            }
        )

    supported = not support.empty and bool(support["is_supported"].all())
    append(
        "monthly_outcome_support",
        supported,
        "all months supported",
        "GREEN" if supported else "RED",
        "Every month requires treated clients and controls with usable outcomes.",
    )
    for test, column, threshold in (
        ("min_treated_count", "n_treated", min_treated),
        ("min_usable_control_count", "n_control_usable", min_control),
        ("min_control_retention", "retention_share", min_control_retention),
    ):
        value = float(support[column].min()) if not support.empty else float("nan")
        flag = (
            "RED"
            if not np.isfinite(value)
            else "YELLOW" if value < threshold else "GREEN"
        )
        append(test, value, f">= {threshold:g}", flag, f"Minimum monthly {column}.")

    if isinstance(data, IPCWEstimate):
        diagnostics = _diagnostic_data(data)
        rows = diagnostics.row_data
        usable = rows["treatment"].eq(0) & rows["outcome_usable"].astype(bool)
        weights = rows.loc[usable, "control_weight"].to_numpy(dtype=float)
        valid = weights.size > 0 and np.isfinite(weights).all() and (weights > 0).all()
        append(
            "finite_control_weights",
            bool(valid),
            "finite and > 0",
            "GREEN" if valid else "RED",
            "Usable control outcomes require finite positive weights.",
        )
        summary = ipcw_weight_summary_table(data)
        ess = float(summary["control_ess"].min()) if not summary.empty else float("nan")
        append(
            "min_control_ess",
            ess,
            f">= {min_control_ess:g}",
            (
                "RED"
                if not np.isfinite(ess)
                else "YELLOW" if ess < min_control_ess else "GREEN"
            ),
            "Minimum monthly effective control sample size after weighting.",
        )
        baseline = rows.drop_duplicates(["trial_id", "unit_id"])
        risk = rows.loc[rows["treatment"].eq(0) & rows["at_risk"].astype(bool)]
        for test, part, column in (
            ("baseline_propensity_clipping", baseline, "propensity_clipped"),
            ("censoring_clipping", risk, "censoring_clipped"),
        ):
            share = float(part[column].mean()) if not part.empty else 0.0
            append(
                test,
                share,
                f"<= {max_clipping_share:g}",
                "YELLOW" if share > max_clipping_share else "GREEN",
                "Clipping controls numerical instability but may introduce bias.",
            )
    return pd.DataFrame(
        checks, columns=["test", "flag", "value", "threshold", "message"]
    )
