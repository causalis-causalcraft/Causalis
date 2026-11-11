"""Short-model refits and observed-confounder calibration for binary IRM."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness._cluster_utils import compact_cluster_codes
from causalis.scenarios.unconfoundedness._sensitivity_inputs import (
    ResolvedEstimate,
    resolve_estimate,
    _normalize_ate_atte_score,
)

_BENCHMARK_RESULT_COLUMNS = [
    "cf_y",
    "cf_d",
    "rho",
    "theta_long",
    "theta_short",
    "delta",
    "r2_y",
    "r2_d",
    "gain_y",
    "gain_d",
    "cf_y_raw",
    "cf_d_raw",
    "sigma2_long",
    "sigma2_short",
    "nu2_long",
    "nu2_short",
    "rho_raw",
    "rho_clipped",
    "cf_y_clipped",
    "cf_d_clipped",
    "rho_fallback",
    "boundary_calibration",
    "strengths_valid",
    "calibration_valid",
    "calibration_issue",
    "calibration_warning",
]


def _read_benchmark_sensitivity_elements(model: Any) -> tuple[float, float, str | None]:
    """Read the two scalar sensitivity elements needed for gain statistics."""
    try:
        elements = model._sensitivity_element_est()
        if not isinstance(elements, dict):
            raise TypeError("_sensitivity_element_est() did not return a dictionary")
        sigma2 = float(elements["sigma2"])
        nu2 = float(elements["nu2"])
    except (KeyError, TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        return np.nan, np.nan, f"sensitivity elements are unavailable: {exc}"
    return sigma2, nu2, None


def _resolve_benchmark_fold_assignments(
    model: Any,
    *,
    full_sample_size: int,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve full-sample folds for aligned long/short benchmark refits.

    New IRM fits retain fold ids before applying the overlap policy. Older
    fitted objects may only expose ``folds_``; those ids are reusable when they
    still cover the full input sample. If a dropped legacy fit has a fixed
    random seed, leaving the assignments unset recreates the same stratified
    split on the unchanged full sample.
    """
    full_folds = getattr(model, "_full_sample_folds_", None)
    if full_folds is not None:
        full_folds_arr = np.asarray(full_folds, dtype=int).ravel()
        if full_folds_arr.size == full_sample_size:
            return full_folds_arr.copy(), None
        return None, (
            "stored full-sample cross-fitting assignments have inconsistent "
            f"length: expected {full_sample_size}, got {full_folds_arr.size}"
        )

    diagnostic_folds = getattr(model, "folds_", None)
    if diagnostic_folds is not None:
        diagnostic_folds_arr = np.asarray(diagnostic_folds, dtype=int).ravel()
        if diagnostic_folds_arr.size == full_sample_size:
            return diagnostic_folds_arr.copy(), None

    if getattr(model, "cluster_codes_", None) is not None:
        return None, (
            "clustered long/short cross-fitting requires stored full-sample fold assignments"
        )

    if getattr(model, "random_state", None) is not None:
        # The short model receives the same full sample, treatment vector,
        # n_folds, and seed, so StratifiedKFold recreates the long-model split.
        return None, None

    return None, (
        "long/short cross-fitting splits cannot be aligned because the long "
        "model has no full-sample fold assignments and random_state=None"
    )


def _calibrate_benchmark_gain_statistics(
    *,
    theta_long: float,
    theta_short: float,
    sigma2_long: float,
    sigma2_short: float,
    nu2_long: float,
    nu2_short: float,
    element_issue: str | None = None,
) -> dict[str, Any]:
    """Calibrate relative benchmark gains for sensitivity of the full model.

    Following Chernozhukov et al. (2022), with ``gap_y = sigma2_short - sigma2_long``
    and ``gap_d = nu2_long - nu2_short``:

    - ``gain_y = gap_y / sigma2_long``,
    - ``gain_d = gap_d / nu2_short``,
    - ``rho = (theta_short - theta_long) / sqrt(gap_y * gap_d)`` clipped to [-1, 1].

    These are the relative-strength gains in Appendix F, equations (22)-(23),
    of Chernozhukov et al. They calibrate a hypothetical omitted confounder
    relative to the model with all observed covariates, as in DoubleML's
    ``gain_statistics``. Normalizing by ``sigma2_short`` and ``nu2_long`` would
    instead describe the effect of adding the observed benchmark to the short
    model, which is a different sensitivity scenario.

    ``cf_y`` and ``cf_d`` are the raw relative gains clipped to [0, 1], matching
    DoubleML. A value of 1 is retained for auditing but marked uncalibratable
    because this sensitivity API requires both strengths to be below 1.
    ``gain_y``/``gain_d`` retain nonnegative gains before upper clipping.
    If the denominator for ``rho`` is not positive, ``rho`` falls back to the
    sign of the long/short effect shift.
    ``cf_y_raw``/``cf_d_raw`` hold the unclipped values; ``r2_y``/``r2_d`` are
    deprecated copies of ``cf_y``/``cf_d``.
    """
    result: dict[str, Any] = {
        "cf_y": np.nan,
        "cf_d": np.nan,
        "rho": np.nan,
        "r2_y": np.nan,
        "r2_d": np.nan,
        "gain_y": np.nan,
        "gain_d": np.nan,
        "cf_y_raw": np.nan,
        "cf_d_raw": np.nan,
        "sigma2_long": float(sigma2_long),
        "sigma2_short": float(sigma2_short),
        "nu2_long": float(nu2_long),
        "nu2_short": float(nu2_short),
        "rho_raw": np.nan,
        "rho_clipped": False,
        "cf_y_clipped": False,
        "cf_d_clipped": False,
        "rho_fallback": False,
        "boundary_calibration": False,
        "strengths_valid": False,
        "calibration_valid": False,
        "calibration_issue": None,
        "calibration_warning": None,
    }
    issues: list[str] = []
    calibration_warnings: list[str] = []
    if element_issue:
        issues.append(element_issue)

    scalar_values = np.asarray(
        [theta_long, theta_short, sigma2_long, sigma2_short, nu2_long, nu2_short],
        dtype=float,
    )
    if not np.isfinite(scalar_values).all():
        issues.append("long/short estimates and sensitivity elements must be finite")

    for name, value in (
        ("sigma2_long", sigma2_long),
        ("sigma2_short", sigma2_short),
        ("nu2_long", nu2_long),
        ("nu2_short", nu2_short),
    ):
        if np.isfinite(value) and value <= 0.0:
            issues.append(f"{name} must be positive")

    gap_y = float(sigma2_short - sigma2_long)
    gap_d = float(nu2_long - nu2_short)

    if not issues:
        result["cf_y_raw"] = float(gap_y / sigma2_long)
        result["cf_d_raw"] = float(gap_d / nu2_short)
        result["gain_y"] = float(max(gap_y, 0.0) / sigma2_long)
        result["gain_d"] = float(max(gap_d, 0.0) / nu2_short)
        if not np.isfinite(result["cf_y_raw"]):
            issues.append("cf_y_raw must be finite")
        if not np.isfinite(result["cf_d_raw"]):
            issues.append("cf_d_raw must be finite")

    if not issues:
        cf_y = float(np.clip(result["cf_y_raw"], 0.0, 1.0))
        cf_d = float(np.clip(result["cf_d_raw"], 0.0, 1.0))
        cf_y_clipped = bool(cf_y != result["cf_y_raw"])
        cf_d_clipped = bool(cf_d != result["cf_d_raw"])
        result.update(
            {
                "cf_y": cf_y,
                "cf_d": cf_d,
                "r2_y": cf_y,
                "r2_d": cf_d,
                "cf_y_clipped": cf_y_clipped,
                "cf_d_clipped": cf_d_clipped,
            }
        )
        if cf_y_clipped:
            calibration_warnings.append(
                f"cf_y_raw={result['cf_y_raw']:.6g} was clipped to {cf_y:.6g}"
            )
        if cf_d_clipped:
            calibration_warnings.append(
                f"cf_d_raw={result['cf_d_raw']:.6g} was clipped to {cf_d:.6g}"
            )

        if cf_y >= 1.0 or cf_d >= 1.0:
            issues.append("calibrated cf_y/cf_d must be < 1; benchmark gains reach the upper boundary")
        else:
            result["strengths_valid"] = True

        if gap_y > 0.0 and gap_d > 0.0:
            rho_denom = float(np.sqrt(gap_y) * np.sqrt(gap_d))
            rho_raw = float((theta_short - theta_long) / rho_denom)
            if not np.isfinite(rho_raw):
                issues.append("rho is not finite")
            else:
                rho = float(np.clip(rho_raw, -1.0, 1.0))
                result.update(
                    {
                        "rho_raw": rho_raw,
                        "rho": rho,
                        "rho_clipped": bool(
                            not np.isclose(rho, rho_raw, rtol=0.0, atol=0.0)
                        ),
                    }
                )
        else:
            rho = float(np.sign(theta_short - theta_long))
            result.update(
                {
                    "rho": rho,
                    "rho_fallback": True,
                }
            )
            calibration_warnings.append(
                "rho used the DoubleML boundary fallback because both gain "
                "components were not strictly positive"
            )

    result["boundary_calibration"] = bool(
        result["cf_y_clipped"]
        or result["cf_d_clipped"]
        or result["rho_fallback"]
        or result["cf_y"] == 1.0
        or result["cf_d"] == 1.0
    )
    result["calibration_valid"] = bool(
        not issues
        and result["strengths_valid"]
        and np.isfinite(result["rho"])
    )
    result["calibration_issue"] = "; ".join(dict.fromkeys(issues)) or None
    result["calibration_warning"] = (
        "; ".join(dict.fromkeys(calibration_warnings)) or None
    )
    return result


@dataclass
class _BenchmarkContext:
    model: Any
    data: CausalData
    df_input: pd.DataFrame
    treatment_name: str
    outcome_name: str
    confounders: list[str]
    estimate_args: dict[str, Any]
    store_diagnostics: bool
    theta: float
    elements: dict[str, Any]
    sigma2: float
    nu2: float
    folds: np.ndarray | None
    issue: str | None


@dataclass(frozen=True)
class _BenchmarkSample:
    data: CausalData
    df_input: pd.DataFrame
    treatment_name: str
    outcome_name: str
    confounders: list[str]


def _validate_benchmark_sample(
    estimate: ResolvedEstimate, data: CausalData
) -> _BenchmarkSample:
    """Check data roles, row order, and the immutable fit-sample fingerprint."""
    model = estimate.model
    if model is None:
        raise TypeError(
            "Sensitivity benchmarking requires a fitted IRM model reference."
        )
    if not isinstance(data, CausalData):
        raise TypeError(f"data must be a CausalData instance. Got {type(data)}.")

    required_attrs = ["data", "coef_", "se_", "_sensitivity_element_est"]
    for attr in required_attrs:
        if not hasattr(model, attr):
            if attr == "data" and hasattr(model, "data_contracts"):
                continue
            raise NotImplementedError(
                "Sensitivity benchmarking requires a fitted IRM model with sensitivity elements. "
                f"Missing: {attr}"
            )

    model_data = getattr(model, "data", getattr(model, "data_contracts", None))
    if not isinstance(model_data, CausalData):
        raise TypeError(
            "effect_estimation must resolve to a fitted IRM model with CausalData in model.data."
        )

    fitted_cluster_name = getattr(model, "_fit_cluster_name_", model_data.cluster_name)
    if data.cluster_name != fitted_cluster_name or model_data.cluster_name != fitted_cluster_name:
        raise ValueError("data cluster declaration must match the fitted long model cluster declaration.")
    if data.user_id_name != model_data.user_id_name:
        raise ValueError("data user_id declaration must match the fitted long model.")

    treatment_name = str(data.treatment_name)
    outcome_name = str(data.outcome_name)
    model_treatment_name = str(model_data.treatment_name)
    model_outcome_name = str(model_data.outcome_name)
    if treatment_name != model_treatment_name:
        raise ValueError(
            "data treatment column must match the fitted long model treatment column. "
            f"Got {treatment_name!r} and {model_treatment_name!r}."
        )
    if outcome_name != model_outcome_name:
        raise ValueError(
            "data outcome column must match the fitted long model outcome column. "
            f"Got {outcome_name!r} and {model_outcome_name!r}."
        )

    x_list_long = list(getattr(model_data, "confounders", []))
    data_confounders = list(data.confounders)
    if len(data_confounders) == 0:
        raise ValueError("data.confounders must not be empty.")
    if set(data_confounders) != set(x_list_long):
        raise ValueError(
            "data.confounders must match the fitted long model confounders as a set. "
            f"Got {data_confounders} and {x_list_long}."
        )

    if len(x_list_long) <= 1:
        raise ValueError(
            "Benchmarking requires at least two confounders in the long model."
        )

    compare_cols = [outcome_name, treatment_name] + x_list_long
    input_cols = list(compare_cols)
    if data.user_id_name is not None:
        input_cols.append(str(data.user_id_name))
    if data.cluster_name is not None:
        input_cols.append(str(data.cluster_name))
    df_input = data.get_df(columns=input_cols)
    if df_input.shape[0] != model_data.df.shape[0]:
        raise ValueError(
            "data must have the same number of rows as the fitted long model. "
            f"Got {df_input.shape[0]} and {model_data.df.shape[0]}."
        )
    # Compare one column at a time: do not materialize two full numeric matrices
    # just to validate a million-row benchmark sample.
    if any(not np.array_equal(df_input[column].to_numpy(copy=False),
                              model_data.df[column].to_numpy(copy=False))
           for column in compare_cols):
        raise ValueError(
            "data must match the fitted long model sample and row order exactly."
        )
    if not data.df.index.equals(model_data.df.index):
        raise ValueError("data must match the fitted long model sample and row order exactly.")
    if data.user_id_name is not None and not data.user_id.equals(model_data.user_id):
        raise ValueError("data user_id values must match the fitted long model sample and row order.")

    codes = None
    if data.cluster_name is not None:
        raw_codes = pd.factorize(data.cluster, sort=False)[0]
        codes = compact_cluster_codes(raw_codes, len(df_input))
        model_codes = pd.factorize(model_data.cluster, sort=False)[0]
        if not np.array_equal(codes, model_codes):
            raise ValueError("data cluster partition must match the fitted long model sample and row order.")

    fingerprint = getattr(model, "_fit_sample_fingerprint_", None)
    if fingerprint is not None:
        fingerprint_args = dict(
            X=df_input[x_list_long].to_numpy(dtype=float),
            y=df_input[outcome_name].to_numpy(dtype=float),
            d=df_input[treatment_name].to_numpy(dtype=int),
        )
        if codes is not None:
            fingerprint_args["cluster_codes"] = codes
        current = model._compute_sample_fingerprint(**fingerprint_args)
        if current != fingerprint:
            raise ValueError(
                "The long-model data changed after fit; refit before benchmarking."
            )
    return _BenchmarkSample(
        data, df_input, treatment_name, outcome_name, data_confounders
    )


def _prepare_refit_options(
    estimate: ResolvedEstimate, fit_args: dict[str, Any] | None
) -> tuple[dict[str, Any], bool]:
    """Resolve short-model options while preserving the long-model estimand."""
    if fit_args is not None and not isinstance(fit_args, dict):
        raise TypeError(
            f"fit_args must be a dict. {fit_args} of type {type(fit_args)} was passed."
        )
    args = dict(fit_args or {})
    store_diagnostics = bool(
        args.pop("store_diagnostics", args.pop("diagnostic_data", False))
    )
    explicit_score = args.get("score")
    score = (
        estimate.score
        if explicit_score is None
        else _normalize_ate_atte_score(explicit_score)
    )
    model_score = _normalize_ate_atte_score(getattr(estimate.model, "score", "ATE"))
    if score != model_score or estimate.score != model_score:
        raise ValueError(
            "Benchmark estimand/score must match the fitted long-model state; refit or use the matching estimate."
        )
    args["score"] = score
    return args, store_diagnostics


def _validate_estimate_state(
    estimate: ResolvedEstimate, elements: dict[str, Any]
) -> None:
    """Reject a saved estimate or diagnostics from a different model state."""
    model = estimate.model
    if not np.isclose(
        estimate.theta, float(model.coef_[0]), rtol=1e-12, atol=1e-14, equal_nan=True
    ):
        raise ValueError(
            "The supplied estimate does not match the fitted long-model state."
        )
    if estimate.diagnostic_data is None:
        return
    diag = estimate.diagnostic_data
    has_membership = hasattr(diag, "cluster_codes") or (
        isinstance(diag, dict) and "cluster_codes" in diag
    )
    if has_membership:
        saved_codes = diag.get("cluster_codes") if isinstance(diag, dict) else diag.cluster_codes
        current_codes = getattr(model, "cluster_codes_", None)
        if (saved_codes is None) != (current_codes is None):
            raise ValueError("The supplied diagnostics do not match the fitted long-model cluster state.")
    for key, current in (
        ("overlap_mask", getattr(model, "overlap_mask_", None)),
        ("psi", elements.get("psi")),
        ("m_hat", getattr(model, "m_hat_", None)),
        ("cluster_codes", getattr(model, "cluster_codes_", None)),
    ):
        saved = getattr(estimate.diagnostic_data, key, None)
        if (
            saved is not None
            and current is not None
            and not np.array_equal(saved, current, equal_nan=True)
        ):
            raise ValueError(
                "The supplied diagnostics do not match the fitted long-model state."
            )


def _prepare_benchmark_context(
    effect_estimation: Any, data: CausalData, fit_args: dict[str, Any] | None
) -> _BenchmarkContext:
    """Prepare one shared long-model context before running any short refits."""
    estimate = resolve_estimate(effect_estimation)
    sample = _validate_benchmark_sample(estimate, data)
    args, store_diagnostics = _prepare_refit_options(estimate, fit_args)
    model = estimate.model
    element_issue = None
    try:
        elements = model._sensitivity_element_est()
        sigma2, nu2 = float(elements["sigma2"]), float(elements["nu2"])
    except (KeyError, TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        elements, sigma2, nu2 = {}, np.nan, np.nan
        element_issue = f"sensitivity elements are unavailable: {exc}"
    _validate_estimate_state(estimate, elements)
    folds, split_issue = _resolve_benchmark_fold_assignments(
        model, full_sample_size=len(sample.df_input)
    )
    issue = "; ".join(x for x in (element_issue, split_issue) if x) or None
    return _BenchmarkContext(
        model=model,
        data=sample.data,
        df_input=sample.df_input,
        treatment_name=sample.treatment_name,
        outcome_name=sample.outcome_name,
        confounders=sample.confounders,
        estimate_args=args,
        store_diagnostics=store_diagnostics,
        theta=float(model.coef_[0]),
        elements=elements,
        sigma2=sigma2,
        nu2=nu2,
        folds=folds,
        issue=issue,
    )


def _validate_benchmark_group(
    ctx: _BenchmarkContext, group: list[str]
) -> tuple[str, ...]:
    if not isinstance(group, list):
        raise TypeError("benchmarking_group must be a list of confounder names.")
    if not group:
        raise ValueError("benchmarking_group must not be empty.")
    if not all(isinstance(item, str) for item in group):
        raise TypeError("benchmarking_group must contain only strings.")
    result = tuple(dict.fromkeys(group))
    if not set(result) <= set(ctx.confounders):
        raise ValueError("benchmarking_group must be a subset of data.confounders.")
    if len(set(result)) == len(ctx.confounders):
        raise ValueError(
            "Benchmarking this group would leave no confounders for the short model."
        )
    return result


def _benchmark_group_refit(
    ctx: _BenchmarkContext, group: tuple[str, ...]
) -> pd.DataFrame:
    from causalis.scenarios.unconfoundedness.model import IRM

    model = ctx.model
    short_data = CausalData(
        df=ctx.df_input,
        treatment=ctx.treatment_name,
        outcome=ctx.outcome_name,
        confounders=[name for name in ctx.confounders if name not in set(group)],
        user_id=ctx.data.user_id_name,
        cluster=ctx.data.cluster_name,
    )
    short = IRM(
        data=short_data,
        ml_g=model.ml_g,
        ml_m=model.ml_m,
        n_folds=getattr(model, "n_folds", 4),
        n_rep=getattr(model, "n_rep", 1),
        normalize_ipw=getattr(
            model, "normalize_ipw_effective_", getattr(model, "normalize_ipw", False)
        ),
        overlap_policy=getattr(model, "overlap_policy", "clip"),
        overlap_threshold=getattr(model, "overlap_threshold", 1e-2),
        weights=getattr(model, "weights", None),
        random_state=getattr(model, "random_state", None),
        n_jobs=getattr(model, "n_jobs", 1),
    )
    if ctx.folds is not None:
        short._fixed_fold_assignments_ = ctx.folds.copy()
    short.fit(store_diagnostics=ctx.store_diagnostics)
    short.estimate(**ctx.estimate_args)
    sigma2, nu2, short_issue = _read_benchmark_sensitivity_elements(short)
    mask_issue = None
    if not np.array_equal(
        getattr(model, "overlap_mask_", None), getattr(short, "overlap_mask_", None)
    ):
        mask_issue = "long/short overlap masks differ; gain statistics require the same analysis population"
    issue = "; ".join(x for x in (ctx.issue, short_issue, mask_issue) if x) or None
    calibration = _calibrate_benchmark_gain_statistics(
        theta_long=ctx.theta,
        theta_short=float(short.coef_[0]),
        sigma2_long=ctx.sigma2,
        sigma2_short=sigma2,
        nu2_long=ctx.nu2,
        nu2_short=nu2,
        element_issue=issue,
    )
    return pd.DataFrame(
        [
            dict(
                benchmark_group=group,
                **calibration,
                theta_long=ctx.theta,
                theta_short=float(short.coef_[0]),
                delta=ctx.theta - float(short.coef_[0]),
            )
        ],
        columns=["benchmark_group", *_BENCHMARK_RESULT_COLUMNS],
    )


def _failed_benchmark(
    ctx: _BenchmarkContext, group: tuple[str, ...], issue: str
) -> pd.DataFrame:
    calibration = _calibrate_benchmark_gain_statistics(
        theta_long=ctx.theta,
        theta_short=np.nan,
        sigma2_long=ctx.sigma2,
        sigma2_short=np.nan,
        nu2_long=ctx.nu2,
        nu2_short=np.nan,
        element_issue=issue,
    )
    return pd.DataFrame(
        [
            dict(
                benchmark_group=group,
                **calibration,
                theta_long=ctx.theta,
                theta_short=np.nan,
                delta=np.nan,
            )
        ],
        columns=["benchmark_group", *_BENCHMARK_RESULT_COLUMNS],
    )


def _sensitivity_benchmark_refits(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    benchmarking_set: List[str] | Literal["all"],
    fit_args: Optional[Dict[str, Any]] = None,
    *,
    grouped: bool = False,
) -> pd.DataFrame:
    """Refit aligned short IRMs, sharing long-model validation and moments."""
    ctx = _prepare_benchmark_context(effect_estimation, data, fit_args)
    requested = ctx.confounders if benchmarking_set == "all" else benchmarking_set
    # Validate every requested group before performing any expensive refits.
    if not isinstance(requested, list):
        raise TypeError("benchmarking_set must be a list of names or 'all'.")
    if grouped:
        groups = [_validate_benchmark_group(ctx, requested)]
    else:
        if not requested:
            raise ValueError("benchmarking_set must not be empty.")
        if not all(isinstance(name, str) for name in requested):
            raise TypeError("benchmarking_set must contain only strings.")
        groups = [
            _validate_benchmark_group(ctx, [name]) for name in dict.fromkeys(requested)
        ]
    result = pd.concat(
        [_benchmark_group_refit(ctx, group) for group in groups], ignore_index=True
    )
    if not grouped:
        result["benchmark_group"] = result["benchmark_group"].map(
            lambda group: group[0]
        )
        result = result.rename(columns={"benchmark_group": "benchmark_confounder"})
    return result


def sensitivity_benchmark(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    benchmarking_set: List[str] | Literal["all"],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    r"""
    Benchmark confounders one by one using genuine short-model IRM refits.

    Each requested confounder is excluded separately. The returned DataFrame
    therefore contains one row and one short-model estimate per confounder.
    Pass ``"all"`` to benchmark all confounders in ``data.confounders`` order.

    See Also
    --------
    sensitivity_benchmark_group
        Exclude several confounders together in one short-model refit.
    """
    return _sensitivity_benchmark_refits(
        effect_estimation,
        data,
        benchmarking_set,
        fit_args,
        grouped=False,
    )


def sensitivity_benchmark_group(
    effect_estimation: Dict[str, Any] | Any,
    data: CausalData,
    benchmarking_group: List[str],
    fit_args: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    r"""
    Benchmark a group of confounders using one genuine short-model IRM refit.

    All features in ``benchmarking_group`` are excluded together. Group-level
    confounding strengths and ``rho`` are calibrated jointly from long/short
    IRM sensitivity elements and the actual effect-estimate shift.

    Parameters
    ----------
    effect_estimation : dict or Any
        Estimate/model container exposing a fitted IRM-like model.
    data : CausalData
        The exact causal dataset and row order used to fit the long model.
    benchmarking_group : list[str]
        Non-empty group of confounders to exclude together. Duplicate names are
        ignored while preserving their first occurrence.
    fit_args : dict, optional
        Additional keyword arguments passed to ``IRM.estimate(...)`` for the
        short model. Short-model diagnostics are disabled by default.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame retaining ``benchmark_group``, ``cf_y``, ``cf_d``,
        ``rho``, ``theta_long``, ``theta_short``, and ``delta`` and appending
        element-based gain statistics and calibration audit fields. The group
        is stored as a tuple in ``benchmark_group``.

    Notes
    -----
    At least one long-model confounder must remain after removing the group.
    Runtime is normally dominated by the single short-model refit.
    """
    if not isinstance(benchmarking_group, list):
        raise TypeError(
            "benchmarking_group must be a list of confounder names. "
            f"Got {benchmarking_group!r} of type {type(benchmarking_group)}."
        )

    return _sensitivity_benchmark_refits(
        effect_estimation,
        data,
        benchmarking_group,
        fit_args,
        grouped=True,
    )
