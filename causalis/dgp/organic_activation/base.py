"""Structural organic-activation panel DGP with shared-innovation oracle regimes.

Treatment changes only outcomes. Independent, time-specific activation uniforms
are shared across interventions, allowing exact replay of a skipped decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from .views import (
    DID_FEATURES,
    DML_FEATURES,
    IPCW_BASELINE_FEATURES,
    IPCW_CENSOR_FEATURES,
    STATIC_FEATURES,
)

SCENARIOS = (
    "baseline",
    "latent_level_confounding",
    "observed_timevarying_confounding",
    "hidden_timevarying_confounding",
    "null_effect",
    "weak_overlap",
)


@dataclass(frozen=True)
class OrganicActivationConfig:
    """Panel dimensions, distributions and structural coefficients.

    ``main_cohorts`` must have complete administrative follow-up. Coefficients
    for latent/observed confounding are enabled only in their named scenario;
    ``null_effect`` always disables the complete treatment-effect function.
    Currency coefficients are in rubles. State innovations are Gaussian and
    outcome innovations mix standardized Student-t and centered lognormal noise.
    """

    n_clients: int = 50_000
    n_periods: int = 24
    horizon: int = 6
    scenario: str = "baseline"
    seed: int | None = 42
    entry_max: int = 9
    main_cohorts: tuple[int, ...] = tuple(range(1, 11))
    calendar_start: str = "2024-01"

    income_log_mean: float = float(np.log(90_000.0))
    income_log_std: float = 0.6
    salary_probability: float = 0.55
    segment_probabilities: tuple[float, ...] = (0.55, 0.30, 0.15)
    digital_alpha: float = 2.0
    digital_beta: float = 2.0
    state_rho: float = 0.65
    state_innovation_std: float = 0.7
    state_income: float = 0.35
    state_salary: float = 0.25
    state_digital: float = 0.6
    latent_level_std: float = 250.0
    hidden_rho: float = 0.55
    hidden_innovation_std: float = 1.0

    hazard_intercept: float = -3.0
    hazard_tenure: float = 0.15
    hazard_seasonality: float = 0.2
    hazard_income: float = 0.55
    hazard_salary: float = 0.45
    hazard_segment: float = 0.25
    hazard_digital: float = 1.1
    hazard_interaction: float = 0.35
    hazard_score_bound: float = 2.0
    hazard_latent: float = 0.9
    hazard_observed_state: float = 0.8
    hazard_hidden_state: float = 0.9
    weak_overlap_strength: float = 10.0

    outcome_intercept: float = 300.0
    outcome_income: float = 200.0
    outcome_salary: float = 130.0
    outcome_segment: float = 90.0
    outcome_digital: float = 80.0
    outcome_interaction: float = 100.0
    outcome_tenure: float = 60.0
    outcome_seasonality: float = 100.0
    outcome_seasonal_harmonic: float = 0.5
    outcome_covariate_trend: float = 6.0
    outcome_trend_income: float = 0.6
    outcome_trend_salary: float = 1.0
    outcome_trend_segment: float = 0.2
    outcome_observed_state: float = 160.0
    outcome_hidden_state: float = 180.0
    outcome_noise_std: float = 260.0
    outcome_noise_df: float = 5.0
    outcome_asymmetry_scale: float = 60.0
    outcome_asymmetry_sigma: float = 0.6

    effect_base: float = 220.0
    effect_income: float = 0.25
    effect_salary: float = 0.2
    effect_digital: float = 0.25
    effect_ramp: float = 2.5
    effect_slope: float = 0.02

    def __post_init__(self) -> None:
        for name in ("n_clients", "n_periods", "horizon", "entry_max"):
            value = getattr(self, name)
            minimum = 0 if name == "entry_max" else 1
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.entry_max >= self.n_periods:
            raise ValueError("entry_max must be less than n_periods.")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}.")
        if self.seed is not None and (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be None or a nonnegative integer.")
        cohorts = tuple(self.main_cohorts)
        if (
            not cohorts
            or any(
                isinstance(g, (bool, np.bool_))
                or not isinstance(g, (int, np.integer))
                or g < 1
                for g in cohorts
            )
            or tuple(sorted(set(cohorts))) != cohorts
        ):
            raise ValueError(
                "main_cohorts must be increasing, unique positive integers."
            )
        if cohorts[-1] + self.horizon > self.n_periods:
            raise ValueError("Every main_cohorts horizon must fit inside n_periods.")
        object.__setattr__(self, "main_cohorts", cohorts)
        excluded = {
            "n_clients",
            "n_periods",
            "horizon",
            "seed",
            "scenario",
            "entry_max",
            "main_cohorts",
            "calendar_start",
            "segment_probabilities",
        }
        for field in fields(self):
            if field.name not in excluded:
                try:
                    valid = np.isfinite(getattr(self, field.name))
                except TypeError as exc:
                    raise ValueError(f"{field.name} must be a finite number.") from exc
                if not valid:
                    raise ValueError(f"{field.name} must be a finite number.")
        for name in (
            "income_log_std",
            "digital_alpha",
            "digital_beta",
            "effect_ramp",
            "hazard_score_bound",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "state_innovation_std",
            "hidden_innovation_std",
            "latent_level_std",
            "outcome_noise_std",
            "outcome_asymmetry_scale",
            "outcome_asymmetry_sigma",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if abs(self.state_rho) >= 1 or abs(self.hidden_rho) >= 1:
            raise ValueError(
                "state_rho and hidden_rho must lie strictly between -1 and 1."
            )
        if not 0 < self.salary_probability < 1:
            raise ValueError("salary_probability must lie strictly between 0 and 1.")
        probabilities = np.asarray(self.segment_probabilities, dtype=float)
        if (
            probabilities.shape != (3,)
            or not np.isfinite(probabilities).all()
            or (probabilities <= 0).any()
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                "segment_probabilities must contain three positive probabilities summing to one."
            )
        object.__setattr__(self, "segment_probabilities", tuple(probabilities))
        if self.outcome_noise_df <= 2:
            raise ValueError("outcome_noise_df must exceed 2 for finite variance.")
        if self.weak_overlap_strength <= 1:
            raise ValueError("weak_overlap_strength must exceed 1.")
        try:
            pd.Period(self.calendar_start, freq="M")
        except (TypeError, ValueError) as exc:
            raise ValueError("calendar_start must define a valid month.") from exc


def _sample_exogenous(config: OrganicActivationConfig) -> dict[str, np.ndarray]:
    """Independent streams keep innovations fixed when scenario/effect changes."""
    c = config
    streams = [
        np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(7)
    ]
    x_rng, state_rng, hidden_rng, level_rng, assignment_rng, noise_rng, asym_rng = (
        streams
    )
    shape = (c.n_clients, c.n_periods)
    z_income = x_rng.normal(size=c.n_clients)
    exo = {
        "entry_period": x_rng.integers(0, c.entry_max + 1, size=c.n_clients),
        "x_income": np.exp(c.income_log_mean + c.income_log_std * z_income),
        "x_salary": x_rng.binomial(1, c.salary_probability, size=c.n_clients),
        "x_segment": x_rng.choice(3, size=c.n_clients, p=c.segment_probabilities),
        "x_digital": x_rng.beta(c.digital_alpha, c.digital_beta, size=c.n_clients),
        "x_noise_1": x_rng.normal(size=c.n_clients),
        "x_noise_2": x_rng.normal(size=c.n_clients),
        "state_initial": state_rng.normal(size=c.n_clients),
        "state_innovation": state_rng.normal(0, c.state_innovation_std, size=shape),
        "hidden_initial": hidden_rng.normal(size=c.n_clients),
        "hidden_innovation": hidden_rng.normal(0, c.hidden_innovation_std, size=shape),
        "latent_standard": level_rng.normal(size=c.n_clients),
        "activation_uniform": assignment_rng.uniform(size=shape),
    }
    exo["latent_level"] = c.latent_level_std * exo["latent_standard"]
    heavy = noise_rng.standard_t(c.outcome_noise_df, size=shape)
    heavy *= c.outcome_noise_std * np.sqrt(
        (c.outcome_noise_df - 2) / c.outcome_noise_df
    )
    skew = (
        asym_rng.lognormal(
            -0.5 * c.outcome_asymmetry_sigma**2, c.outcome_asymmetry_sigma, size=shape
        )
        - 1
    )
    exo["outcome_innovation"] = heavy + c.outcome_asymmetry_scale * skew
    return exo


def _structural_paths(
    exo: dict[str, np.ndarray], c: OrganicActivationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return treatment-free L, hidden V, prospective hazard and Y(0)."""
    n, periods = c.n_clients, c.n_periods
    l_state = np.zeros((n, periods))
    v_state = np.zeros_like(l_state)
    hazard = np.full_like(l_state, np.nan)
    y0 = np.full_like(l_state, np.nan)
    z = np.tanh((np.log(exo["x_income"]) - c.income_log_mean) / c.income_log_std)
    salary, segment, digital = (
        exo[f"x_{name}"] for name in ("salary", "segment", "digital")
    )
    state_mean = (
        c.state_income * z + c.state_salary * salary + c.state_digital * digital
    )
    score = (
        c.hazard_income * z
        + c.hazard_salary * (salary - c.salary_probability)
        + c.hazard_segment * (segment - np.dot(np.arange(3), c.segment_probabilities))
        + c.hazard_digital * (digital - 0.5)
        + c.hazard_interaction * (salary - c.salary_probability) * (digital - 0.5)
    )
    strength = c.weak_overlap_strength if c.scenario == "weak_overlap" else 1.0
    # Bounded covariate score preserves numerical positivity in the stress case.
    bounded_score = c.hazard_score_bound * np.tanh(score / c.hazard_score_bound)
    for t in range(periods):
        observed = exo["entry_period"] <= t
        entering = exo["entry_period"] == t
        tenure = np.maximum(t - exo["entry_period"], 0)
        previous_l = l_state[:, t - 1] if t else np.zeros(n)
        previous_v = v_state[:, t - 1] if t else np.zeros(n)
        l = (
            state_mean
            + c.state_rho * (previous_l - state_mean)
            + exo["state_innovation"][:, t]
        )
        l = np.where(
            entering,
            state_mean
            + c.state_innovation_std
            / np.sqrt(1 - c.state_rho**2)
            * exo["state_initial"],
            l,
        )
        v = c.hidden_rho * previous_v + exo["hidden_innovation"][:, t]
        v = np.where(
            entering,
            c.hidden_innovation_std
            / np.sqrt(1 - c.hidden_rho**2)
            * exo["hidden_initial"],
            v,
        )
        l_state[:, t] = np.where(observed, l, 0)
        v_state[:, t] = np.where(observed, v, 0)
        seasonal = np.sin(2 * np.pi * t / 12)
        logit = (
            c.hazard_intercept
            + strength * bounded_score
            + c.hazard_tenure * np.log1p(tenure)
            + c.hazard_seasonality * seasonal
        )
        if c.scenario == "latent_level_confounding":
            logit = logit + c.hazard_latent * exo["latent_standard"]
        elif c.scenario == "observed_timevarying_confounding":
            logit = logit + c.hazard_observed_state * l
        elif c.scenario == "hidden_timevarying_confounding":
            logit = logit + c.hazard_hidden_state * v
        # Stable logistic; no clipping of probabilities or observations.
        exp_negative_abs = np.exp(-np.abs(logit))
        p = np.where(
            logit >= 0,
            1 / (1 + exp_negative_abs),
            exp_negative_abs / (1 + exp_negative_abs),
        )
        eligible = observed & (tenure >= 1)
        if np.any((p[eligible] <= 0) | (p[eligible] >= 1)):
            raise ValueError(
                "Configured hazard reaches a numerical positivity boundary; reduce its coefficients."
            )
        hazard[eligible, t] = p[eligible]
        mean = (
            c.outcome_intercept
            + c.outcome_income * z
            + c.outcome_salary * salary
            + c.outcome_segment * segment
            + c.outcome_digital * np.sin(np.pi * digital)
            + c.outcome_interaction * salary * digital
            + c.outcome_tenure * np.log1p(tenure)
            + c.outcome_seasonality
            * (seasonal + c.outcome_seasonal_harmonic * np.cos(2 * np.pi * t / 6))
            + c.outcome_covariate_trend
            * t
            * (
                c.outcome_trend_income * z
                + c.outcome_trend_salary * (salary - c.salary_probability)
                + c.outcome_trend_segment * segment
            )
            + exo["latent_level"]
            + c.outcome_observed_state * l
        )
        if c.scenario == "hidden_timevarying_confounding":
            mean = mean + c.outcome_hidden_state * v
        y0[observed, t] = (mean + exo["outcome_innovation"][:, t])[observed]
    observed_mask = np.arange(periods)[None, :] >= exo["entry_period"][:, None]
    if not np.isfinite(y0[observed_mask]).all():
        raise ValueError("Configured outcome parameters produce nonfinite outcomes.")
    return l_state, v_state, hazard, y0


def _activation_from_hazard(
    exo: dict[str, np.ndarray],
    hazard: np.ndarray,
    *,
    skip_period: int | None = None,
    force_start_period: int | None = None,
    never_activate: bool = False,
) -> np.ndarray:
    first = np.full(hazard.shape[0], -1, dtype=np.int64)
    if never_activate:
        return first
    for t in range(hazard.shape[1]):
        risk = (first < 0) & np.isfinite(hazard[:, t])
        now = risk & (exo["activation_uniform"][:, t] < hazard[:, t])
        if t == skip_period:
            now[:] = False
        if t == force_start_period:
            now = risk
        first[now] = t
    return first


def replay_activation(
    exogenous: dict[str, np.ndarray],
    config: OrganicActivationConfig,
    *,
    skip_period: int | None = None,
    force_start_period: int | None = None,
    never_activate: bool = False,
) -> np.ndarray:
    """Replay first activations; ``-1`` denotes no activation by panel end.

    This low-level oracle utility operates on the whole simulated population.
    A forced start overrides the decision in that month for still-eligible
    clients; earlier natural activations are retained.
    """
    _, _, hazard, _ = _structural_paths(exogenous, config)
    return _activation_from_hazard(
        exogenous,
        hazard,
        skip_period=skip_period,
        force_start_period=force_start_period,
        never_activate=never_activate,
    )


def _effect_path(
    exo: dict[str, np.ndarray], c: OrganicActivationConfig, first: np.ndarray
) -> np.ndarray:
    event = np.arange(c.n_periods)[None, :] - first[:, None]
    active = (first[:, None] >= 0) & (event >= 0)
    if c.scenario == "null_effect":
        return np.zeros_like(event, dtype=float)
    z = np.tanh((np.log(exo["x_income"]) - c.income_log_mean) / c.income_log_std)
    multiplier = (
        1
        + c.effect_income * z
        + c.effect_salary * exo["x_salary"]
        + c.effect_digital * exo["x_digital"]
    )
    h = np.maximum(event, 0)
    effect = (
        c.effect_base
        * multiplier[:, None]
        * (-np.expm1(-(h + 1) / c.effect_ramp))
        * (1 + c.effect_slope * h)
    )
    return np.where(active, effect, 0.0)


def _scenario_table() -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for method in ("did", "dml", "ipcw"):
            identified = (
                scenario in {"baseline", "null_effect", "weak_overlap"}
                or (scenario == "latent_level_confounding" and method == "did")
                or (scenario == "observed_timevarying_confounding" and method != "did")
            )
            if method == "did":
                features = "static X, entry_period; monthly outcomes; no outcome lags"
                assumption = "Conditional parallel trends, no anticipation, overlap"
            elif method == "dml":
                features = "static X, entry_period, predecision L_g"
                assumption = "Baseline exchangeability and overlap; natural follow-up"
            else:
                features = "baseline X, entry_period, L_g; censoring X, entry_period, period, tenure, L_s"
                assumption = (
                    "Baseline and sequential exchangeability, consistency, positivity"
                )
            reason = ""
            if not identified:
                if scenario == "latent_level_confounding":
                    reason = (
                        "Unobserved U confounds baseline assignment and outcome levels."
                    )
                elif scenario == "observed_timevarying_confounding":
                    reason = "Selection on current and future L innovations can violate parallel trends."
                else:
                    reason = (
                        "Unobserved changing V affects activation and outcome dynamics."
                    )
            elif scenario == "weak_overlap":
                reason = "Formal identification; near-boundary probabilities can destabilize estimation."
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "estimand": (
                        "ATT_defer_natural" if method == "dml" else "ATT_no_activation"
                    ),
                    "features": features,
                    "assumptions": assumption,
                    "identified": identified,
                    "limitation": reason,
                }
            )
    return pd.DataFrame(rows)


def _oracle_effects(
    exo: dict[str, np.ndarray],
    c: OrganicActivationConfig,
    first: np.ndarray,
    hazard: np.ndarray,
    factual_effect: np.ndarray,
    untreated_outcome: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    individual_parts = []
    cohort_rows = []
    skip_parts = []
    counts = {g: int(np.count_nonzero(first == g)) for g in c.main_cohorts}
    total = sum(counts.values())
    for g in c.main_cohorts:
        clients = np.flatnonzero(first == g)
        weight = counts[g] / total if total else 0.0
        # Resume the structural hazard with the same future innovations.
        # Do not substitute factual G for the skipped-decision regime.
        sub_exo = {"activation_uniform": exo["activation_uniform"][clients]}
        skip_g = _activation_from_hazard(sub_exo, hazard[clients], skip_period=g)
        # In the factual g-cohort earlier decisions all failed naturally.
        skip_parts.append(
            pd.Series(
                pd.array(np.where(skip_g >= 0, skip_g, None), dtype="Int64"),
                index=clients,
            )
        )
        sub = {name: value[clients] for name, value in exo.items()}
        deferred_effect = _effect_path(sub, c, skip_g)
        periods = g + np.arange(c.horizon)
        no_effect = factual_effect[clients][:, periods]
        defer_effect = no_effect - deferred_effect[:, periods]
        no_outcome = untreated_outcome[clients][:, periods]
        cumulative_no = np.cumsum(no_effect, axis=1)
        cumulative_defer = np.cumsum(defer_effect, axis=1)
        part = pd.DataFrame(
            {
                "client_id": np.repeat(clients, c.horizon),
                "g": g,
                "event_time": np.tile(np.arange(c.horizon), len(clients)),
                "y_start": (no_outcome + no_effect).ravel(),
                "y_no_activation": no_outcome.ravel(),
                "y_defer_natural": (no_outcome + deferred_effect[:, periods]).ravel(),
                "att_no_activation": no_effect.ravel(),
                "att_defer_natural": defer_effect.ravel(),
                "cumulative_no_activation": cumulative_no.ravel(),
                "cumulative_defer_natural": cumulative_defer.ravel(),
            }
        )
        individual_parts.append(part)
        for h in range(c.horizon):
            cohort_rows.append(
                {
                    "g": g,
                    "event_time": h,
                    "n_treated": counts[g],
                    "weight": weight,
                    "att_no_activation": (
                        float(no_effect[:, h].mean()) if counts[g] else np.nan
                    ),
                    "att_defer_natural": (
                        float(defer_effect[:, h].mean()) if counts[g] else np.nan
                    ),
                    "cumulative_no_activation": (
                        float(cumulative_no[:, h].mean()) if counts[g] else np.nan
                    ),
                    "cumulative_defer_natural": (
                        float(cumulative_defer[:, h].mean()) if counts[g] else np.nan
                    ),
                }
            )
    individual = pd.concat(individual_parts, ignore_index=True)
    cohorts = pd.DataFrame(cohort_rows)
    effect_cols = [
        "att_no_activation",
        "att_defer_natural",
        "cumulative_no_activation",
        "cumulative_defer_natural",
    ]
    aggregate = pd.DataFrame({"event_time": np.arange(c.horizon)})
    for col in effect_cols:
        weighted = cohorts[col].fillna(0) * cohorts["weight"]
        aggregate[col] = (
            weighted.groupby(cohorts["event_time"]).sum().to_numpy()
            if total
            else np.nan
        )
    skipped = pd.concat(skip_parts).sort_index()
    skipped.index.name = "client_id"
    skipped.name = "skip_activation_period"
    return individual, cohorts, aggregate, skipped


class OrganicActivationGenerator:
    """Generate one organic-activation scenario and two distinct oracle ATTs."""

    def __init__(self, config: OrganicActivationConfig):
        self.config = config

    def generate(self) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        c = self.config
        exo = _sample_exogenous(c)
        state, hidden, hazard, y0 = _structural_paths(exo, c)
        first = _activation_from_hazard(exo, hazard)
        effect = _effect_path(exo, c, first)
        outcome = y0 + effect
        observed = np.arange(c.n_periods)[None, :] >= exo["entry_period"][:, None]
        if not np.isfinite(outcome[observed]).all():
            raise ValueError("Configured treatment effects produce nonfinite outcomes.")
        client, period = np.nonzero(observed)
        g = first[client]
        tenure = period - exo["entry_period"][client]
        calendar = pd.period_range(
            c.calendar_start, periods=c.n_periods, freq="M"
        ).to_timestamp()
        df = pd.DataFrame(
            {
                "client_id": client,
                "period": period,
                "date": calendar.take(period),
                "entry_period": exo["entry_period"][client],
                "tenure_months": tenure,
                "n_pre_periods_available": tenure,
                "first_activation_period": pd.array(
                    np.where(g >= 0, g, None), dtype="Int64"
                ),
                "activation_now": g == period,
                "treated": (g >= 0) & (period >= g),
                "eligible_to_activate": (tenure >= 1) & ((g < 0) | (g >= period)),
                "event_time": pd.array(
                    np.where(g >= 0, period - g, None), dtype="Int64"
                ),
                **{name: exo[name][client] for name in STATIC_FEATURES},
                "l_activity_pre": state[client, period],
                "ltv_monthly": outcome[client, period],
                "ltv_cumulative": np.cumsum(np.where(observed, outcome, 0), axis=1)[
                    client, period
                ],
            }
        )
        individual, cohorts, aggregate, skipped = _oracle_effects(
            exo, c, first, hazard, effect, y0
        )
        potential = pd.DataFrame(
            {
                "client_id": client,
                "period": period,
                "y_no_activation": y0[client, period],
                "y_factual": outcome[client, period],
                "effect_factual": effect[client, period],
                "hazard": hazard[client, period],
                "U": exo["latent_level"][client],
                "V": hidden[client, period],
            }
        )
        oracle = {
            "potential_outcomes": potential,
            "exogenous": exo,
            "config": c,
            "individual_effects": individual,
            "att_by_cohort": cohorts,
            "att_aggregate": aggregate,
            "skip_activation_period": skipped,
        }
        feature_sets = {
            "did": list(DID_FEATURES),
            "dml": list(DML_FEATURES),
            "ipcw_baseline": list(IPCW_BASELINE_FEATURES),
            "ipcw_censoring": list(IPCW_CENSOR_FEATURES),
        }
        metadata = {
            "config": asdict(c),
            "scenario": c.scenario,
            "main_cohorts": c.main_cohorts,
            "cohort_weights": cohorts[["g", "n_treated", "weight"]]
            .drop_duplicates()
            .reset_index(drop=True),
            "model_features": {
                scenario: {k: list(v) for k, v in feature_sets.items()}
                for scenario in SCENARIOS
            },
            "scenario_table": _scenario_table(),
            "estimands": {
                "ATT_no_activation": "Start at g versus no activation throughout the horizon, among factual G=g.",
                "ATT_defer_natural": "Start at g versus skip g then natural activation, among factual G=g.",
            },
            "oracle_target": "Realized sample ATT among factual treated clients, with fixed external innovations.",
            "timing": "Entry month has tenure zero; L is predecision; activation precedes that month's outcome.",
            "effect_columns": {
                "monthly": ["att_no_activation", "att_defer_natural"],
                "cumulative": ["cumulative_no_activation", "cumulative_defer_natural"],
            },
        }
        df.attrs["organic_activation"] = {
            "scenario": c.scenario,
            "n_periods": c.n_periods,
            "main_cohorts": c.main_cohorts,
        }
        return df, oracle, metadata


def generate_organic_activation_data(
    n_clients: int = 50_000,
    n_periods: int = 24,
    horizon: int = 6,
    scenario: str = "baseline",
    seed: int | None = 42,
    **dgp_params: Any,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Generate a long bank-client panel, isolated oracle and design metadata.

    Parameters
    ----------
    n_clients, n_periods, horizon : int
        Client count, calendar months and complete ATT follow-up length.
    scenario : str
        One of baseline, latent_level_confounding,
        observed_timevarying_confounding, hidden_timevarying_confounding,
        null_effect or weak_overlap. Only one panel is generated per call.
    seed : int or None
        Seed for independent exogenous random streams.
    **dgp_params
        Additional :class:`OrganicActivationConfig` fields, including
        ``main_cohorts``, ``entry_max`` and all structural coefficients.

    Returns
    -------
    df : pandas.DataFrame
        Observed client-month panel with monthly and cumulative ruble LTV.
    oracle : dict
        Exogenous arrays indexed [client_id, period], prospective hazards,
        latent variables, potential outcomes, skipped activation dates and
        individual/cohort/aggregate effects for both benchmark estimands.
        ``potential_outcomes`` is aligned with the returned panel's row order.
    metadata : dict
        Configuration, model-feature allowlists, scenario assumptions and
        fixed cohort-size weights. Oracle data must never enter model features.

    Notes
    -----
    DML with naturally activating controls targets ATT_defer_natural. Monthly
    staggered DiD and baseline-adjusted IPCW target ATT_no_activation under
    their respective identification conditions. No estimators or weights are
    fitted, and no files are saved. Empty random cohorts are not resampled.

    Examples
    --------
    >>> df, oracle, metadata = generate_organic_activation_data(n_clients=100)
    >>> df[['client_id', 'period']].duplicated().any()
    False
    """
    known = {field.name for field in fields(OrganicActivationConfig)}
    unknown = set(dgp_params) - known
    if unknown:
        raise ValueError(f"Unknown DGP parameters: {sorted(unknown)}")
    config = OrganicActivationConfig(
        n_clients=n_clients,
        n_periods=n_periods,
        horizon=horizon,
        scenario=scenario,
        seed=seed,
        **dgp_params,
    )
    return OrganicActivationGenerator(config).generate()


generate_data = generate_organic_activation_data
