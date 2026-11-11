"""Structural validation and compact diagnostics for organic activation data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import (
    OrganicActivationConfig,
    SCENARIOS,
    _activation_from_hazard,
    _effect_path,
    _structural_paths,
)
from .views import (
    DID_FEATURES,
    DML_FEATURES,
    IPCW_BASELINE_FEATURES,
    IPCW_CENSOR_FEATURES,
    STATIC_FEATURES,
    make_ipcw_view,
)


def validate_data(
    df: pd.DataFrame, oracle: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, bool]:
    """Validate panel invariants, exact structural replay and feature isolation.

    Returns named checks, raising ``ValueError`` when structural checks fail.
    The final three ``support_*`` checks are informational: small random samples
    may legitimately lack a treated, later-treated or never-treated group.
    No samples are regenerated and no panel or oracle objects are modified.
    """
    required = {
        "client_id",
        "period",
        "date",
        "entry_period",
        "tenure_months",
        "n_pre_periods_available",
        "first_activation_period",
        "activation_now",
        "treated",
        "eligible_to_activate",
        "event_time",
        "l_activity_pre",
        "ltv_monthly",
        "ltv_cumulative",
        *STATIC_FEATURES,
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a nonempty pandas.DataFrame.")
    if not df.columns.is_unique or not required.issubset(df.columns):
        raise ValueError(
            "df must have unique columns and the complete observed schema."
        )
    checks: dict[str, bool] = {}

    def check(name: str, value: Any) -> None:
        checks[name] = bool(value)
        if not checks[name]:
            raise ValueError(f"Organic activation validation failed: {name}")

    check("unique_client_period", not df.duplicated(["client_id", "period"]).any())
    c = oracle["config"]
    if not isinstance(c, OrganicActivationConfig):
        raise ValueError("oracle['config'] must be OrganicActivationConfig.")
    ordered = df.sort_values(["client_id", "period"], kind="stable")
    clients = ordered.drop_duplicates("client_id").set_index("client_id")
    by_client = ordered.groupby("client_id", sort=False)
    check(
        "client_ids", np.array_equal(clients.index.to_numpy(), np.arange(c.n_clients))
    )
    check(
        "constant_client_design",
        not by_client[["entry_period", "first_activation_period", *STATIC_FEATURES]]
        .nunique(dropna=False)
        .gt(1)
        .any()
        .any(),
    )
    check(
        "administrative_followup",
        (
            by_client["period"].min().eq(clients["entry_period"]).all()
            and by_client["period"].max().eq(c.n_periods - 1).all()
            and by_client.size().eq(c.n_periods - clients["entry_period"]).all()
            and by_client["period"].diff().dropna().eq(1).all()
        ),
    )
    tenure = ordered["period"] - ordered["entry_period"]
    check(
        "history_lengths",
        ordered["tenure_months"].eq(tenure).all()
        and ordered["n_pre_periods_available"].eq(tenure).all(),
    )
    first = ordered["first_activation_period"]
    expected_activation = first.eq(ordered["period"]).fillna(False).to_numpy(dtype=bool)
    expected_treated = first.le(ordered["period"]).fillna(False).to_numpy(dtype=bool)
    expected_eligible = (
        tenure.ge(1) & (first.isna() | first.ge(ordered["period"]).fillna(False))
    ).to_numpy(dtype=bool)
    check(
        "one_first_activation",
        by_client["activation_now"].sum().le(1).all()
        and np.array_equal(ordered["activation_now"], expected_activation),
    )
    check(
        "absorbing_treated",
        np.array_equal(ordered["treated"], expected_treated)
        and by_client["treated"].diff().dropna().astype(int).ge(0).all(),
    )
    check(
        "activation_eligibility",
        np.array_equal(ordered["eligible_to_activate"], expected_eligible)
        and not (
            ordered["activation_now"].astype(bool)
            & ~ordered["eligible_to_activate"].astype(bool)
        ).any(),
    )
    expected_event = pd.array(ordered["period"] - first, dtype="Int64")
    check(
        "event_time",
        pd.Series(expected_event, index=ordered.index).equals(ordered["event_time"]),
    )
    activated_core = ordered["activation_now"] & first.isin(c.main_cohorts)
    check(
        "core_pre_history",
        ordered.loc[activated_core, "n_pre_periods_available"]
        .between(1, max(c.main_cohorts))
        .all(),
    )
    check(
        "cumulative_ltv",
        np.allclose(
            ordered["ltv_cumulative"],
            by_client["ltv_monthly"].cumsum(),
            rtol=1e-12,
            atol=1e-9,
        ),
    )
    check(
        "finite_observed_data",
        np.isfinite(
            ordered[
                [*STATIC_FEATURES, "l_activity_pre", "ltv_monthly", "ltv_cumulative"]
            ].to_numpy(dtype=float)
        ).all(),
    )
    calendar = pd.period_range(
        c.calendar_start, periods=c.n_periods, freq="M"
    ).to_timestamp()
    check(
        "monthly_dates",
        np.array_equal(
            ordered["date"].to_numpy(),
            calendar.take(ordered["period"].to_numpy()).to_numpy(),
        ),
    )

    potential = oracle["potential_outcomes"].set_index(["client_id", "period"])
    keys = pd.MultiIndex.from_frame(ordered[["client_id", "period"]])
    check(
        "oracle_keys",
        potential.index.is_unique
        and len(potential) == len(ordered)
        and keys.isin(potential.index).all(),
    )
    potential = potential.loc[keys]
    check(
        "no_anticipation",
        potential.loc[~expected_treated, "effect_factual"].eq(0).all(),
    )
    check(
        "observed_consistency",
        np.array_equal(
            potential["y_factual"].to_numpy(), ordered["ltv_monthly"].to_numpy()
        )
        and np.allclose(
            potential["y_no_activation"] + potential["effect_factual"],
            ordered["ltv_monthly"].to_numpy(),
        ),
    )

    # Replay both the stochastic paths and decisions, rather than subtracting
    # stored effects from outcomes and calling that a consistency check.
    exo = oracle["exogenous"]
    client_idx = ordered["client_id"].to_numpy(dtype=int)
    period_idx = ordered["period"].to_numpy(dtype=int)
    check(
        "observed_covariate_replay",
        all(
            np.array_equal(ordered[name].to_numpy(), exo[name][client_idx])
            for name in (*STATIC_FEATURES, "entry_period")
        ),
    )
    state, hidden, hazard, y0 = _structural_paths(exo, c)
    factual_g = _activation_from_hazard(exo, hazard)
    actual_g = clients["first_activation_period"].fillna(-1).to_numpy(dtype=np.int64)
    effects = _effect_path(exo, c, factual_g)
    check(
        "factual_replay",
        np.array_equal(actual_g, factual_g)
        and np.array_equal(
            (y0 + effects)[client_idx, period_idx], ordered["ltv_monthly"].to_numpy()
        ),
    )
    check(
        "potential_outcome_replay",
        np.array_equal(
            potential["y_no_activation"].to_numpy(), y0[client_idx, period_idx]
        )
        and np.array_equal(
            potential["effect_factual"].to_numpy(), effects[client_idx, period_idx]
        ),
    )
    check(
        "state_and_hazard_replay",
        (
            np.array_equal(
                state[client_idx, period_idx], ordered["l_activity_pre"].to_numpy()
            )
            and np.allclose(
                hazard[client_idx, period_idx], potential["hazard"], equal_nan=True
            )
            and np.array_equal(
                hidden[client_idx, period_idx], potential["V"].to_numpy()
            )
            and np.array_equal(
                exo["latent_level"][client_idx], potential["U"].to_numpy()
            )
        ),
    )
    active_probabilities = potential.loc[expected_eligible, "hazard"]
    check(
        "risk_set_positivity",
        active_probabilities.between(0, 1, inclusive="neither").all(),
    )

    individual = oracle["individual_effects"]
    cohort = oracle["att_by_cohort"]
    aggregate = oracle["att_aggregate"]
    effect_cols = [
        "att_no_activation",
        "att_defer_natural",
        "cumulative_no_activation",
        "cumulative_defer_natural",
    ]
    check(
        "individual_oracle_keys",
        not individual.duplicated(["client_id", "event_time"]).any()
        and len(individual) == np.isin(factual_g, c.main_cohorts).sum() * c.horizon,
    )
    deferred_ok = True
    censoring_ok = True
    for g in c.main_cohorts:
        ids = np.flatnonzero(factual_g == g)
        sub = {name: value[ids] for name, value in exo.items()}
        skip_g = _activation_from_hazard(sub, hazard[ids], skip_period=g)
        stored_skip = (
            oracle["skip_activation_period"]
            .reindex(ids)
            .fillna(-1)
            .to_numpy(dtype=np.int64)
        )
        expected_no = effects[ids][:, g : g + c.horizon]
        alternative_effect = _effect_path(sub, c, skip_g)[:, g : g + c.horizon]
        expected_defer = expected_no - alternative_effect
        part = individual.loc[individual["g"] == g].sort_values(
            ["client_id", "event_time"]
        )
        deferred_ok &= (
            np.array_equal(stored_skip, skip_g)
            and np.array_equal(part["client_id"], np.repeat(ids, c.horizon))
            and np.array_equal(
                part["event_time"], np.tile(np.arange(c.horizon), len(ids))
            )
            and np.array_equal(part["att_no_activation"], expected_no.ravel())
            and np.array_equal(part["att_defer_natural"], expected_defer.ravel())
            and np.array_equal(
                part["cumulative_no_activation"], expected_no.cumsum(axis=1).ravel()
            )
            and np.array_equal(
                part["cumulative_defer_natural"], expected_defer.cumsum(axis=1).ravel()
            )
            and np.array_equal(
                part["y_no_activation"], y0[ids][:, g : g + c.horizon].ravel()
            )
            and np.array_equal(
                part["y_start"], (y0[ids][:, g : g + c.horizon] + expected_no).ravel()
            )
            and np.array_equal(
                part["y_defer_natural"],
                (y0[ids][:, g : g + c.horizon] + alternative_effect).ravel(),
            )
        )
        trial = make_ipcw_view(df, g=g, horizon=c.horizon)
        trial_g = factual_g[trial["client_id"].to_numpy(dtype=int)]
        s = trial["period"].to_numpy()
        control = trial["baseline_d"].eq(0).to_numpy()
        expected_censor = control & (s > g) & (trial_g == s)
        expected_risk = control & (s > g) & ((trial_g < 0) | (trial_g >= s))
        expected_usable = ~control | (trial_g < 0) | (trial_g > s)
        censoring_ok &= (
            np.array_equal(trial["censor_now"], expected_censor)
            and np.array_equal(trial["at_risk_of_censoring"], expected_risk)
            and np.array_equal(trial["uncensored_through_t"], expected_usable)
            and np.array_equal(trial["outcome_usable"], expected_usable)
        )
    check("deferred_regime_replay", deferred_ok)
    check("artificial_censoring", censoring_ok)
    total_core = int(np.isin(factual_g, c.main_cohorts).sum())
    cohort_ok = (
        len(cohort) == len(c.main_cohorts) * c.horizon
        and not cohort.duplicated(["g", "event_time"]).any()
    )
    for g in c.main_cohorts:
        part = cohort.loc[cohort["g"] == g].sort_values("event_time")
        n_g = int(np.count_nonzero(factual_g == g))
        weight = n_g / total_core if total_core else 0
        cohort_ok &= (
            len(part) == c.horizon
            and part["n_treated"].eq(n_g).all()
            and part["weight"].eq(weight).all()
        )
        expected_means = (
            individual.loc[individual["g"] == g]
            .groupby("event_time")[effect_cols]
            .mean()
            .reindex(np.arange(c.horizon))
        )
        cohort_ok &= np.allclose(part[effect_cols], expected_means, equal_nan=True)
    check("fixed_cohort_weights", cohort_ok)
    check(
        "aggregate_oracles",
        len(aggregate) == c.horizon
        and np.array_equal(aggregate["event_time"], np.arange(c.horizon)),
    )
    for col in effect_cols:
        expected = (
            (cohort[col].fillna(0) * cohort["weight"])
            .groupby(cohort["event_time"])
            .sum()
            .to_numpy()
            if total_core
            else np.full(c.horizon, np.nan)
        )
        check(f"aggregate_{col}", np.allclose(aggregate[col], expected, equal_nan=True))
    if c.scenario == "null_effect":
        check(
            "null_effect_exact_zero",
            (
                potential["effect_factual"].eq(0).all()
                and individual[effect_cols].eq(0).all().all()
                and cohort.loc[cohort["n_treated"] > 0, effect_cols].eq(0).all().all()
                and aggregate[effect_cols].eq(0).all().all()
                if total_core
                else potential["effect_factual"].eq(0).all()
            ),
        )

    expected_features = {
        "did": list(DID_FEATURES),
        "dml": list(DML_FEATURES),
        "ipcw_baseline": list(IPCW_BASELINE_FEATURES),
        "ipcw_censoring": list(IPCW_CENSOR_FEATURES),
    }
    check(
        "feature_isolation",
        set(metadata["model_features"]) == set(SCENARIOS)
        and all(
            methods == expected_features
            for methods in metadata["model_features"].values()
        )
        and not set(df.columns).intersection(
            {"U", "V", "hazard", "y_no_activation", "y_factual", "effect_factual"}
        ),
    )
    expected_weights = (
        cohort[["g", "n_treated", "weight"]].drop_duplicates().reset_index(drop=True)
    )
    check(
        "metadata_consistency",
        metadata["config"] == c.__dict__
        and metadata["scenario"] == c.scenario
        and tuple(metadata["main_cohorts"]) == c.main_cohorts
        and metadata["cohort_weights"].equals(expected_weights),
    )
    checks["support_treated"] = total_core > 0
    checks["support_never_treated"] = bool((factual_g < 0).any())
    checks["support_later_treated"] = bool(
        any(((factual_g > g) & (exo["entry_period"] < g)).any() for g in c.main_cohorts)
    )
    return checks


def summarize_data(
    df: pd.DataFrame, oracle: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Return compact diagnostics as scalars/tables, without printing or fitting.

    ``control_support`` describes every main-cohort/event-month comparison.
    ``dml_later_activation`` distinguishes activation inside the horizon from
    activation anywhere later in the administrative panel. True hazard quantiles
    use only factual eligible risk rows, never post-activation probabilities.
    """
    c = oracle["config"]
    clients = df.drop_duplicates("client_id")
    first = clients["first_activation_period"]
    support = []
    later = []
    for g in c.main_cohorts:
        eligible = (clients["entry_period"] < g) & (
            first.isna() | first.ge(g).fillna(False)
        )
        controls = eligible & ~first.eq(g).fillna(False)
        n_controls = int(controls.sum())
        activated_later = controls & first.gt(g).fillna(False)
        inside_horizon = activated_later & first.lt(g + c.horizon).fillna(False)
        later.append(
            {
                "g": g,
                "n_controls": n_controls,
                "n_later_activators": int(activated_later.sum()),
                "fraction_later_activation": (
                    float(activated_later.sum() / n_controls) if n_controls else np.nan
                ),
                "n_activations_within_horizon": int(inside_horizon.sum()),
                "fraction_activation_within_horizon": (
                    float(inside_horizon.sum() / n_controls) if n_controls else np.nan
                ),
            }
        )
        for h in range(c.horizon):
            t = g + h
            support.append(
                {
                    "g": g,
                    "t": t,
                    "event_time": h,
                    "n_treated": int((eligible & first.eq(g).fillna(False)).sum()),
                    "n_controls": int(
                        (eligible & (first.isna() | first.gt(t).fillna(False))).sum()
                    ),
                    "n_censored_controls": int(
                        (controls & first.le(t).fillna(False)).sum()
                    ),
                }
            )
    activation_rows = df.loc[df["activation_now"].astype(bool)]
    core_rows = activation_rows.loc[
        activation_rows["first_activation_period"].isin(c.main_cohorts)
    ]
    potential = oracle["potential_outcomes"].set_index(["client_id", "period"])
    risk_keys = pd.MultiIndex.from_frame(
        df.loc[df["eligible_to_activate"].astype(bool), ["client_id", "period"]]
    )
    hazard_quantiles = potential.loc[risk_keys, "hazard"].quantile(
        [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    )
    aggregate = oracle["att_aggregate"].copy()
    aggregate["monthly_estimand_gap"] = (
        aggregate["att_no_activation"] - aggregate["att_defer_natural"]
    )
    aggregate["cumulative_estimand_gap"] = (
        aggregate["cumulative_no_activation"] - aggregate["cumulative_defer_natural"]
    )
    return {
        "overview": {
            "scenario": c.scenario,
            "n_clients": int(len(clients)),
            "n_rows": int(len(df)),
            "never_activated_fraction": float(first.isna().mean()),
            "n_main_cohort_treated": int(len(core_rows)),
        },
        "activation_counts": first.value_counts(dropna=False).sort_index(),
        "pre_history_counts": core_rows["n_pre_periods_available"]
        .value_counts()
        .sort_index(),
        "control_support": pd.DataFrame(support),
        "dml_later_activation": pd.DataFrame(later),
        "hazard_quantiles": hazard_quantiles,
        "att_by_cohort": oracle["att_by_cohort"].copy(),
        "att_aggregate": aggregate,
        "scenario_table": metadata["scenario_table"].copy(),
    }
