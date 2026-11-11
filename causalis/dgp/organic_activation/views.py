"""Analysis views for the organic activation panel.

All decisions happen at the beginning of a month. A client entering at ``g-1``
therefore has one complete baseline month and is eligible for activation at
``g``. Helpers never change the source panel or use future treatment labels as
model features.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

STATIC_FEATURES = (
    "x_income",
    "x_salary",
    "x_segment",
    "x_digital",
    "x_noise_1",
    "x_noise_2",
)
DID_FEATURES = STATIC_FEATURES + ("entry_period",)
DML_FEATURES = DID_FEATURES + ("l_activity_pre",)
IPCW_BASELINE_FEATURES = tuple(f"b_{name}" for name in DML_FEATURES)
IPCW_CENSOR_FEATURES = STATIC_FEATURES + (
    "entry_period",
    "period",
    "tenure_months",
    "l_activity_pre",
)


def _integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _validate_panel(df: pd.DataFrame, features: tuple[str, ...]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("df must contain at least one client-month.")
    if not df.columns.is_unique:
        raise ValueError("df must have unique column names.")
    required = {
        "client_id",
        "period",
        "entry_period",
        "first_activation_period",
        "eligible_to_activate",
        "ltv_monthly",
        *features,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[["client_id", "period", "entry_period"]].isna().any().any():
        raise ValueError("client_id, period, and entry_period must not be missing.")
    for name in ("period", "entry_period"):
        if not pd.api.types.is_integer_dtype(df[name]) or (df[name] < 0).any():
            raise ValueError(f"{name} must contain non-negative integers.")
    if df.duplicated(["client_id", "period"]).any():
        raise ValueError("Duplicate client_id × period keys are not allowed.")
    invariant = df.groupby("client_id", sort=False)[
        ["entry_period", "first_activation_period"]
    ].nunique(dropna=False)
    if invariant.gt(1).any().any():
        raise ValueError(
            "entry_period and first_activation_period must be constant within client."
        )
    if (
        df["eligible_to_activate"].isna().any()
        or not df["eligible_to_activate"].isin([0, 1, False, True]).all()
    ):
        raise ValueError(
            "eligible_to_activate must contain non-missing boolean values."
        )
    values = pd.to_numeric(df["ltv_monthly"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("ltv_monthly must contain finite numeric outcomes.")


def _require_rows(
    df: pd.DataFrame,
    clients: pd.Index,
    periods: tuple[int, ...],
    *,
    purpose: str,
) -> pd.DataFrame:
    rows = df.loc[df["client_id"].isin(clients) & df["period"].isin(periods)]
    if len(rows) != len(clients) * len(periods):
        raise ValueError(f"Missing required {purpose} rows for selected clients.")
    return rows


def _baseline(df: pd.DataFrame, g: int) -> pd.DataFrame:
    if g > int(df["period"].max()):
        raise ValueError("g must fall inside the observed panel.")
    clients = df.drop_duplicates("client_id")[
        ["client_id", "entry_period", "first_activation_period"]
    ]
    first = clients["first_activation_period"]
    risk = (clients["entry_period"] <= g - 1) & (first.isna() | first.ge(g))
    ids = pd.Index(clients.loc[risk, "client_id"])
    _require_rows(df, ids, (g - 1, g), purpose="baseline")
    baseline = df.loc[(df["period"] == g) & df["client_id"].isin(ids)]
    return baseline.loc[baseline["eligible_to_activate"].astype(bool)].copy()


def make_did_pair(df: pd.DataFrame, g: int, t: int) -> pd.DataFrame:
    """Return monthly-outcome pairs for an ``ATT(g,t)`` comparison.

    The baseline outcome is measured at ``g-1``. Both groups must be eligible
    immediately before the decision at ``g``; controls remain untreated through
    ``t``. The static conditioning set deliberately excludes noisy outcome lags
    and dynamic state, preserving the stated parallel-trends guarantee in the
    latent-level scenario. The oracle benchmark is ``ATT_no_activation``.
    """
    g = _integer(g, "g", minimum=1)
    t = _integer(t, "t", minimum=g)
    _validate_panel(df, DID_FEATURES)
    if t > int(df["period"].max()):
        raise ValueError("t must fall inside the observed panel.")
    baseline = _baseline(df, g)
    first = baseline["first_activation_period"]
    selected = baseline.loc[
        first.eq(g).fillna(False) | first.gt(t).fillna(False) | first.isna()
    ]
    ids = pd.Index(selected["client_id"])
    paired = _require_rows(df, ids, (g - 1, t), purpose="baseline/follow-up")
    out = selected[["client_id", *DID_FEATURES]].copy()
    out["baseline_period"] = g - 1
    out["g"] = g
    out["t"] = t
    out["d"] = selected["first_activation_period"].eq(g).fillna(False).astype(np.int8)
    before = paired.loc[paired["period"] == g - 1].set_index("client_id")["ltv_monthly"]
    after = paired.loc[paired["period"] == t].set_index("client_id")["ltv_monthly"]
    out["y_pre"] = out["client_id"].map(before)
    out["y_post"] = out["client_id"].map(after)
    out["delta_y"] = out["y_post"] - out["y_pre"]
    out = out[
        [
            "client_id",
            "baseline_period",
            "g",
            "t",
            "d",
            *DID_FEATURES,
            "y_pre",
            "y_post",
            "delta_y",
        ]
    ].reset_index(drop=True)
    out.attrs = {"model_features": list(DID_FEATURES), "estimand": "ATT_no_activation"}
    return out


def make_dml_view(df: pd.DataFrame, g: int = 10, horizon: int = 6) -> pd.DataFrame:
    """Return a baseline risk-set sample for ATT IRM/unconfoundedness.

    Features are measured before the decision at ``g``. Controls who activate
    later retain ``d=0`` and their actual horizon outcomes, so the benchmark is
    ``ATT_defer_natural``. A complete administrative horizon is required.
    """
    g = _integer(g, "g", minimum=1)
    horizon = _integer(horizon, "horizon", minimum=1)
    _validate_panel(df, DML_FEATURES)
    if g + horizon - 1 > int(df["period"].max()):
        raise ValueError(
            "The full follow-up horizon must fall inside the observed panel."
        )
    baseline = _baseline(df, g)
    rows = _require_rows(
        df,
        pd.Index(baseline["client_id"]),
        tuple(range(g, g + horizon)),
        purpose="follow-up",
    )
    out = baseline[["client_id", *DML_FEATURES]].copy()
    out["baseline_period"] = g
    out["d"] = baseline["first_activation_period"].eq(g).fillna(False).astype(np.int8)
    totals = rows.groupby("client_id", sort=False)["ltv_monthly"].sum()
    out["y_ltv_h"] = out["client_id"].map(totals)
    out = out[
        ["client_id", "baseline_period", "d", *DML_FEATURES, "y_ltv_h"]
    ].reset_index(drop=True)
    out.attrs = {"model_features": list(DML_FEATURES), "estimand": "ATT_defer_natural"}
    return out


def make_ipcw_view(df: pd.DataFrame, g: int = 10, horizon: int = 6) -> pd.DataFrame:
    """Return fixed-horizon trial rows with artificial-censoring indicators.

    The event row remains available for fitting the censoring hazard, but its
    outcome and all later outcomes are unusable for a baseline control. Always
    filter outcomes by ``outcome_usable`` and train the censoring model only on
    ``at_risk_of_censoring`` rows. At baseline, assignment propensity is handled
    separately: the censoring-survival product begins at ``g+1``. When combining
    trials, group cross-fitting and bootstrap by ``client_id``.
    """
    g = _integer(g, "g", minimum=1)
    horizon = _integer(horizon, "horizon", minimum=1)
    _validate_panel(df, IPCW_CENSOR_FEATURES)
    if "date" not in df:
        raise ValueError("Missing required columns: ['date']")
    if g + horizon - 1 > int(df["period"].max()):
        raise ValueError(
            "The full follow-up horizon must fall inside the observed panel."
        )
    baseline = _baseline(df, g)
    rows = _require_rows(
        df,
        pd.Index(baseline["client_id"]),
        tuple(range(g, g + horizon)),
        purpose="follow-up",
    )
    out = rows[
        [
            "client_id",
            "period",
            "date",
            *[name for name in IPCW_CENSOR_FEATURES if name != "period"],
            "ltv_monthly",
        ]
    ].copy()
    frozen = baseline[["client_id", *DML_FEATURES]].rename(
        columns={name: f"b_{name}" for name in DML_FEATURES}
    )
    out = out.merge(
        frozen, on="client_id", how="left", sort=False, validate="many_to_one"
    )
    first = out["client_id"].map(
        baseline.set_index("client_id")["first_activation_period"]
    )
    out["trial_id"] = g
    out["baseline_period"] = g
    out["baseline_d"] = first.eq(g).fillna(False).astype(np.int8)
    out["followup_month"] = out["period"] - g
    controls = out["baseline_d"].eq(0)
    out["at_risk_of_censoring"] = (
        controls
        & out["period"].gt(g)
        & (first.isna() | first.ge(out["period"]).fillna(False))
    ).astype(bool)
    out["censor_now"] = (
        controls & out["period"].gt(g) & first.eq(out["period"]).fillna(False)
    ).astype(bool)
    out["uncensored_through_t"] = (
        ~controls | first.isna() | first.gt(out["period"]).fillna(False)
    ).astype(bool)
    out["outcome_usable"] = out["uncensored_through_t"]
    out = (
        out[
            [
                "client_id",
                "trial_id",
                "baseline_period",
                "baseline_d",
                "period",
                "date",
                "followup_month",
                *IPCW_BASELINE_FEATURES,
                *[name for name in IPCW_CENSOR_FEATURES if name != "period"],
                "at_risk_of_censoring",
                "censor_now",
                "uncensored_through_t",
                "outcome_usable",
                "ltv_monthly",
            ]
        ]
        .sort_values(["client_id", "period"], kind="stable")
        .reset_index(drop=True)
    )
    out.attrs = {
        "model_features": {
            "baseline": list(IPCW_BASELINE_FEATURES),
            "censoring": list(IPCW_CENSOR_FEATURES),
        },
        "estimand": "ATT_no_activation",
    }
    return out


__all__ = [
    "STATIC_FEATURES",
    "DID_FEATURES",
    "DML_FEATURES",
    "IPCW_BASELINE_FEATURES",
    "IPCW_CENSOR_FEATURES",
    "make_did_pair",
    "make_dml_view",
    "make_ipcw_view",
]
