"""Benchmark-calibrated sensitivity decisions, with explicit scenario specifications."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal, Mapping, Optional
import numpy as np
import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness import _sensitivity_core as core
from causalis.scenarios.unconfoundedness._sensitivity_inputs import (
    ResolvedEstimate,
    resolve_estimate,
)
from . import _sensitivity_benchmark as benchmark
from . import _sensitivity_reporting as reporting

_PROTOCOL_SCENARIO_COLUMNS = [
    "benchmark",
    "benchmark_group",
    "scenario",
    "multiplier",
    "cf_y",
    "cf_d",
    "rho",
    "theta_long",
    "theta_short",
    "delta",
    "relative_delta",
    "theta_lower",
    "theta_upper",
    "ci_lower",
    "ci_upper",
    "rv",
    "rva",
    "scenario_valid",
    "scenario_issue",
    "passed",
]


def _scale_strengths(cf_y: float, cf_d: float, multiplier: float) -> tuple[float, float]:
    """Scale benchmark strengths as a confounder ``multiplier`` times as strong.

    The variance gains are multiplied: ``gap_y -> k * gap_y`` (so ``cf_y`` scales
    linearly) and ``gap_d -> k * gap_d`` (so ``cf_d / (1 - cf_d)`` scales linearly).
    The bias bound then scales by ``k``. ``cf_y`` is capped just below one.
    """
    cf_y = float(cf_y)
    cf_d = float(cf_d)
    k = float(multiplier)
    for name, value in (("cf_y", cf_y), ("cf_d", cf_d)):
        if not np.isfinite(value) or not (0.0 <= value < 1.0):
            raise ValueError(f"Benchmark {name} must be in [0, 1). Got {value!r}.")
    scaled_cf_y = float(min(k * cf_y, 1.0 - 1e-12))
    odds_d = k * cf_d / (1.0 - cf_d)
    return scaled_cf_y, float(odds_d / (1.0 + odds_d))


def _protocol_ci_passes(
    ci: tuple[float, float],
    *,
    direction: Literal["positive", "negative"],
    decision_threshold: float,
) -> bool:
    """Apply the resolved directional decision rule to an interval."""
    lo, hi = map(float, ci)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    if direction == "positive":
        return bool(lo > decision_threshold)
    return bool(hi < decision_threshold)


@dataclass(frozen=True)
class ProtocolPolicy:
    direction: str
    decision_threshold: float
    alpha: float
    stress_multiplier: float
    preconditions_passed: bool

    def metadata(self) -> dict[str, Any]:
        return dict(
            direction=self.direction,
            decision_threshold=self.decision_threshold,
            alpha=self.alpha,
            stress_multiplier=self.stress_multiplier,
            preconditions_passed=self.preconditions_passed,
        )


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    multiplier: float
    cf_y: float
    cf_d: float
    rho: float
    valid: bool
    issue: str | None


@dataclass(frozen=True)
class ProtocolDecision:
    passed: bool
    limited_margin: bool
    sensitivity_passed: bool
    invalid_primary: list[str]
    failed_primary: list[str]
    boundary_benchmarks: list[str]
    inference_issue: str | None

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _validate_policy(
    direction: str,
    decision_threshold: float,
    alpha: float,
    stress_multiplier: float,
    preconditions_passed: bool,
) -> ProtocolPolicy:
    if direction not in {"auto", "positive", "negative"}:
        raise ValueError("direction must be 'auto', 'positive', or 'negative'.")
    if not np.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite and pre-specified.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if not np.isfinite(stress_multiplier) or float(stress_multiplier) <= 1.0:
        raise ValueError("stress_multiplier must be finite and greater than 1.")
    if not isinstance(preconditions_passed, (bool, np.bool_)):
        raise TypeError("preconditions_passed must be a boolean.")
    return ProtocolPolicy(
        direction,
        float(decision_threshold),
        float(alpha),
        float(stress_multiplier),
        bool(preconditions_passed),
    )


def _resolve_direction(
    policy: ProtocolPolicy, estimate: ResolvedEstimate, warnings: list[str]
) -> ProtocolPolicy:
    if policy.direction != "auto":
        return policy
    if not np.isfinite(estimate.theta):
        raise ValueError("direction='auto' requires a finite effect estimate (theta).")
    direction = (
        "positive" if estimate.theta >= policy.decision_threshold else "negative"
    )
    warnings.append(
        f"direction='auto' inferred '{direction}' from theta={estimate.theta:g} "
        f"relative to decision_threshold={policy.decision_threshold:g}; "
        "use an explicit direction for a pre-specified directional claim."
    )
    return replace(policy, direction=direction)


def _normalize_groups(groups: Mapping[str, list[str]]) -> list[tuple[str, list[str]]]:
    normalized = []
    seen = set()
    for raw_name, group in groups.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Every benchmark label must be a non-empty string.")
        if not isinstance(group, list):
            raise TypeError(
                f"Benchmark {raw_name!r} must be a list of confounder names."
            )
        name = raw_name.strip()
        if name in seen:
            raise ValueError(
                "Benchmark labels must be unique after trimming whitespace."
            )
        seen.add(name)
        normalized.append((name, group))
    return normalized


def _empty_report(policy: ProtocolPolicy, warnings: list[str]) -> dict[str, Any]:
    warnings.append("At least one justified primary benchmark group is required.")
    empty = pd.DataFrame(columns=_PROTOCOL_SCENARIO_COLUMNS)
    return dict(
        status="FAIL",
        passed=False,
        **policy.metadata(),
        benchmarks=pd.DataFrame(),
        primary=empty.copy(),
        stress=empty.copy(),
        adversarial=empty.copy(),
        scenarios=empty.copy(),
        details={},
        limited_margin=False,
        warnings=warnings,
        summary="FAIL: no primary benchmark group was supplied; hidden-confounding robustness is not established.",
    )


def _run_benchmarks(
    context: benchmark._BenchmarkContext, groups: list[tuple[str, tuple[str, ...]]]
) -> pd.DataFrame:
    cache: dict[frozenset[str], pd.DataFrame] = {}
    frames = []
    for name, group in groups:
        key = frozenset(group)
        if key not in cache:
            try:
                cache[key] = benchmark._benchmark_group_refit(context, group)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                cache[key] = benchmark._failed_benchmark(
                    context, group, f"short-model refit failed: {exc}"
                )
        frame = cache[key].copy()
        frame.at[0, "benchmark_group"] = group
        frame.insert(0, "benchmark", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _benchmark_warnings(row: pd.Series) -> list[str]:
    name = row["benchmark"]
    warnings = []
    if bool(row.get("rho_clipped", False)):
        warnings.append(
            f"Benchmark {name!r} produced |rho_raw| > 1; the calibrated primary/stress rho was clipped to +/-1."
        )
    if bool(row.get("boundary_calibration", False)):
        reason = str(
            row.get("calibration_warning")
            or "DoubleML boundary clipping/fallback was used"
        )
        warnings.append(
            f"Benchmark {name!r} used DoubleML boundary calibration: {reason}."
        )
    if not bool(row.get("calibration_valid", True)):
        reason = _calibration_issue(row)
        warnings.append(f"Benchmark {name!r} is not calibratable: {reason}.")
    return warnings


def _calibration_issue(row: pd.Series) -> str:
    return str(
        row.get("calibration_issue")
        or "benchmark gain statistics could not be calibrated"
    )


def _scenario_specs(
    row: pd.Series, multiplier: float
) -> tuple[tuple[str, ScenarioSpec], ...]:
    y, d, rho = map(float, (row["cf_y"], row["cf_d"], row["rho"]))
    strengths_valid = bool(row.get("strengths_valid", True))
    calibrated = bool(row.get("calibration_valid", True))
    stress_y, stress_d = (
        _scale_strengths(y, d, multiplier) if strengths_valid else (np.nan, np.nan)
    )
    issue = _calibration_issue(row)
    return (
        (
            "stress",
            ScenarioSpec(
                f"{multiplier:g}x",
                multiplier,
                stress_y,
                stress_d,
                rho,
                calibrated,
                issue,
            ),
        ),
        (
            "adversarial",
            ScenarioSpec("adversarial", 1.0, y, d, 1.0, strengths_valid, issue),
        ),
        ("primary", ScenarioSpec("primary", 1.0, y, d, rho, calibrated, issue)),
    )


def _scenario_row(
    row: pd.Series,
    spec: ScenarioSpec,
    raw: dict[str, Any] | None,
    issue: str | None,
    policy: ProtocolPolicy,
) -> dict[str, Any]:
    """Build the same report schema for valid and invalid scenarios."""
    theta_long, delta = float(row["theta_long"]), float(row["delta"])
    bounds = (
        (np.nan, np.nan)
        if raw is None
        else tuple(map(float, raw["theta_bounds_cofounding"]))
    )
    ci = (np.nan, np.nan) if raw is None else tuple(map(float, raw["bias_aware_ci"]))
    return dict(
        benchmark=row["benchmark"],
        benchmark_group=tuple(row["benchmark_group"]),
        scenario=spec.name,
        multiplier=spec.multiplier,
        cf_y=spec.cf_y,
        cf_d=spec.cf_d,
        rho=spec.rho,
        theta_long=theta_long,
        theta_short=float(row["theta_short"]),
        delta=delta,
        relative_delta=(
            np.nan if abs(theta_long) <= 1e-16 else abs(delta) / abs(theta_long)
        ),
        theta_lower=bounds[0],
        theta_upper=bounds[1],
        ci_lower=ci[0],
        ci_upper=ci[1],
        rv=np.nan if raw is None else float(raw.get("rv", np.nan)),
        rva=np.nan if raw is None else float(raw.get("rva", np.nan)),
        scenario_valid=raw is not None,
        scenario_issue=issue,
        passed=raw is not None
        and _protocol_ci_passes(
            ci, direction=policy.direction, decision_threshold=policy.decision_threshold
        ),
    )


def _evaluate_scenario(
    context: core.SensitivityContext | None,
    row: pd.Series,
    spec: ScenarioSpec,
    policy: ProtocolPolicy,
    inference_issue: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = None
    issue = spec.issue if not spec.valid else inference_issue
    if spec.valid:
        try:
            if context is None:
                raise ValueError(
                    inference_issue or "Sensitivity inference is unavailable."
                )
            raw = core.evaluate_sensitivity(
                context,
                cf_y=spec.cf_y,
                cf_d=spec.cf_d,
                rho=spec.rho,
                H0=policy.decision_threshold,
            )
            issue = None
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            issue = str(exc)
    result = _scenario_row(row, spec, raw, issue, policy)
    return result, (
        dict(raw)
        if raw is not None
        else dict(scenario_valid=False, scenario_issue=issue)
    )


def _evaluate_benchmarks(
    benchmarks: pd.DataFrame,
    context: core.SensitivityContext | None,
    policy: ProtocolPolicy,
    inference_issue: str | None,
    warnings: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any] | None]:
    rows: dict[str, list[dict[str, Any]]] = {
        role: [] for role in ("primary", "stress", "adversarial")
    }
    details = {}
    last_primary = None
    for _, row in benchmarks.iterrows():
        warnings.extend(_benchmark_warnings(row))
        raw_results = {}
        for role, spec in _scenario_specs(row, policy.stress_multiplier):
            result, raw = _evaluate_scenario(
                context, row, spec, policy, inference_issue
            )
            rows[role].append(result)
            raw_results[role] = raw
            if role == "primary" and result["scenario_valid"]:
                last_primary = raw
        details[row["benchmark"]] = {role: raw_results[role] for role in rows}
    frames = {
        role: pd.DataFrame(values, columns=_PROTOCOL_SCENARIO_COLUMNS)
        for role, values in rows.items()
    }
    return frames, details, last_primary


def _make_decision(
    frames: dict[str, pd.DataFrame],
    benchmarks: pd.DataFrame,
    policy: ProtocolPolicy,
    inference_issue: str | None,
) -> ProtocolDecision:
    primary = frames["primary"]
    sensitivity_passed = bool(primary["passed"].all())
    passed = policy.preconditions_passed and sensitivity_passed
    margin = bool(frames["stress"]["passed"].all()) and bool(
        frames["adversarial"]["passed"].all()
    )
    boundary = (
        benchmarks.loc[
            benchmarks["boundary_calibration"].astype(bool), "benchmark"
        ].tolist()
        if "boundary_calibration" in benchmarks
        else []
    )
    return ProtocolDecision(
        passed=passed,
        limited_margin=passed and not margin,
        sensitivity_passed=sensitivity_passed,
        invalid_primary=primary.loc[~primary["scenario_valid"], "benchmark"].tolist(),
        failed_primary=primary.loc[~primary["passed"], "benchmark"].tolist(),
        boundary_benchmarks=boundary,
        inference_issue=inference_issue,
    )


def run_sensitivity_protocol(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    *,
    benchmark_groups: Mapping[str, List[str]],
    decision_threshold: float,
    direction: Literal["auto", "positive", "negative"] = "auto",
    alpha: float = 0.05,
    stress_multiplier: float = 2.0,
    preconditions_passed: bool = True,
    fit_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""Run a binary, benchmark-calibrated sensitivity decision protocol.

    The primary decision uses one-times benchmark strength and the benchmark's
    estimated ``rho``. Every primary benchmark must pass. A benchmark passes
    when its bias-aware confidence interval lies strictly beyond the
    pre-specified practical threshold in ``direction``. The two-times and
    ``rho=1`` adversarial scenarios are reported as stress tests and do not
    change the primary PASS/FAIL decision.

    Parameters
    ----------
    effect_estimation : dict or Any
        Fitted IRM-like estimate accepted by :func:`sensitivity_analysis`.
    data : CausalData
        Exact data and row order used to fit the long model.
    benchmark_groups : mapping[str, list[str]]
        Non-empty mapping from meaningful benchmark labels to groups of
        observed pre-treatment confounders. Each group is omitted in its own
        genuine short-model refit.
    decision_threshold : float
        Pre-specified practically meaningful effect boundary. For a positive
        claim, the lower bias-aware CI endpoint must exceed this value. For a
        negative claim, the upper endpoint must be below it.
    direction : {"auto", "positive", "negative"}, default "auto"
        Direction relative to ``decision_threshold``. ``"auto"`` selects
        ``"positive"`` when the original estimate is at or above the threshold
        and ``"negative"`` otherwise, using the same direction for every
        scenario. Automatic selection requires a finite estimate. An estimate
        equal to the threshold does not pass the strict CI rule. Explicit
        directions are never overridden; use one for a pre-specified
        directional causal claim.
    alpha : float, default 0.05
        Two-sided significance level used for bias-aware confidence intervals.
    stress_multiplier : float, default 2.0
        Strength multiplier for the secondary stress scenario. Outcome strength
        cf_y and Riesz odds cf_d / (1 - cf_d) scale linearly, so the bias
        bound scales by this multiplier until the outcome cap is reached.
    preconditions_passed : bool, default True
        External design gate covering the causal adjustment set, overlap,
        nuisance-model quality, and estimator stability. ``False`` forces the
        overall result to FAIL while still returning sensitivity diagnostics.
    fit_args : dict, optional
        Additional keyword arguments for each short-model estimate.

    Returns
    -------
    dict
        A report with ``status`` (``"PASS"`` or ``"FAIL"``), ``passed``,
        benchmark and scenario DataFrames, raw scenario details, warnings, and
        a standard human-readable conclusion. ``primary`` determines status;
        ``stress`` and ``adversarial`` describe the robustness margin.
        ``direction`` contains the resolved positive/negative direction;
        ``warnings`` records when it was inferred from the estimate.

    Notes
    -----
    The robustness values ``rv`` and ``rva`` are reported but are not used as
    universal pass/fail cutoffs. Benchmark selection and ``decision_threshold``
    must be justified before inspecting the effect estimate. ``rho`` for both
    singleton and multi-feature benchmarks is calibrated from long/short
    sensitivity elements. A value of ``+/-1`` in primary or stress can mean
    either that ``rho_raw`` saturated at the correlation boundary or that
    DoubleML's non-positive-gain fallback was used. Only the adversarial
    scenario forces ``rho=1`` independently of the benchmark effect shift.

    DoubleML-compatible boundary calibration clips non-positive raw gains to
    zero and keeps the scenario numerically evaluable. Raw values and boundary
    flags remain available in ``benchmarks`` for audit. Missing/non-finite
    sensitivity elements and upper-bound ``cf_y=1`` or ``cf_d=1`` still fail closed,
    because calibrated sensitivity strengths must remain below one.
    Long/short overlap masks must coincide; otherwise the benchmark fails closed.

    The current IRM estimator supports ``n_rep=1`` only. Repeated cross-fitting
    stability must therefore be assessed outside this function until IRM adds
    repeated-split support.

    Examples
    --------
    >>> report = run_sensitivity_protocol(  # doctest: +SKIP
    ...     estimate,
    ...     data,
    ...     benchmark_groups={"engagement": ["sessions", "activity_days"]},
    ...     decision_threshold=0.0,
    ...     direction="auto",
    ... )
    >>> report["status"]  # doctest: +SKIP
    'PASS'
    >>> report["primary"][["benchmark", "ci_lower", "ci_upper", "passed"]]  # doctest: +SKIP
    """

    policy = _validate_policy(
        direction, decision_threshold, alpha, stress_multiplier, preconditions_passed
    )
    if not isinstance(benchmark_groups, Mapping):
        raise TypeError(
            "benchmark_groups must be a mapping from labels to feature lists."
        )
    warnings: list[str] = []
    estimate = None
    if policy.direction == "auto" or benchmark_groups:
        estimate = resolve_estimate(effect_estimation, policy.alpha)
        policy = _resolve_direction(policy, estimate, warnings)
    if not benchmark_groups:
        return _empty_report(policy, warnings)

    requested = _normalize_groups(benchmark_groups)
    benchmark_context = benchmark._prepare_benchmark_context(estimate, data, fit_args)
    groups = [
        (name, benchmark._validate_benchmark_group(benchmark_context, group))
        for name, group in requested
    ]
    inference_issue = None
    try:
        context = core.prepare_sensitivity_context(
            theta=estimate.theta,
            se=estimate.se,
            alpha=policy.alpha,
            elements=benchmark_context.elements,
        )
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        context = None
        inference_issue = str(exc)
        warnings.append(f"Sensitivity inference is invalid: {inference_issue}.")

    benchmarks = _run_benchmarks(benchmark_context, groups)
    frames, details, last_primary = _evaluate_benchmarks(
        benchmarks, context, policy, inference_issue, warnings
    )
    decision = _make_decision(frames, benchmarks, policy, inference_issue)
    if last_primary is not None:
        estimate.cache_result(
            reporting._wrap_sensitivity_result(last_primary, label=estimate.label)
        )
    if getattr(benchmark_context.model, "n_rep", None) == 1:
        warnings.append(
            "IRM currently uses n_rep=1; repeated cross-fitting stability must be assessed separately."
        )
    return dict(
        status=decision.status,
        passed=decision.passed,
        **policy.metadata(),
        benchmarks=benchmarks,
        **frames,
        scenarios=pd.concat(list(frames.values()), ignore_index=True),
        details=details,
        limited_margin=decision.limited_margin,
        boundary_benchmarks=decision.boundary_benchmarks,
        warnings=warnings,
        summary=reporting.format_protocol_summary(decision, policy),
    )
