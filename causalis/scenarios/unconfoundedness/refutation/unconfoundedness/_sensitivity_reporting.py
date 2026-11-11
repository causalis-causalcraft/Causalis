"""Tabular and human-readable sensitivity reports; no fitting or numerical inference."""

from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from causalis.data_contracts.sensitivity_analysis_result import (
    SensitivityAnalysisResult,
)
from causalis.scenarios.unconfoundedness._sensitivity_inputs import resolve_estimate, _sensitivity_reference

if TYPE_CHECKING:
    from ._sensitivity_protocol import ProtocolDecision, ProtocolPolicy


def _summary_dataframe_from_result(
    res: Dict[str, Any],
    *,
    label: str | None = None,
) -> pd.DataFrame:
    """Build a simple two-column sensitivity summary."""

    def _round_value(value: Any) -> Any:
        if isinstance(value, list):
            return [round(float(item), 4) if pd.notna(item) else None for item in value]
        if pd.isna(value):
            return np.nan
        return round(float(value), 4)

    sampling_ci = tuple(map(float, res.get("sampling_ci", (np.nan, np.nan))))
    theta_bounds = tuple(
        map(float, res.get("theta_bounds_cofounding", (np.nan, np.nan)))
    )
    bias_aware_ci = tuple(map(float, res.get("bias_aware_ci", (np.nan, np.nan))))
    return pd.DataFrame(
        {
            "statistics": [
                "bias_aware_ci",
                "theta",
                "sampling_ci",
                "rv",
                "rva",
                "se",
                "max_bias",
                "max_bias_base",
                "bound_width",
                "sigma2",
                "nu2",
            ],
            "value": [
                _round_value([bias_aware_ci[0], bias_aware_ci[1]]),
                _round_value(
                    [theta_bounds[0], float(res.get("theta", np.nan)), theta_bounds[1]]
                ),
                _round_value([sampling_ci[0], sampling_ci[1]]),
                _round_value(float(res.get("rv", np.nan))),
                _round_value(float(res.get("rva", np.nan))),
                _round_value(float(res.get("se", np.nan))),
                _round_value(float(res.get("max_bias", np.nan))),
                _round_value(float(res.get("max_bias_base", np.nan))),
                _round_value(float(res.get("bound_width", np.nan))),
                _round_value(float(res.get("sigma2", np.nan))),
                _round_value(float(res.get("nu2", np.nan))),
            ],
        }
    )


def _format_summary_table(df: pd.DataFrame) -> str:
    """Render summary tables while formatting floats inside scalar/list values."""
    display_df = df.copy().astype(object)

    def _fmt(value: Any) -> Any:
        if isinstance(value, list):
            return [round(float(item), 6) if pd.notna(item) else None for item in value]
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.6f}"
        return value

    for col in display_df.columns:
        display_df[col] = display_df[col].map(_fmt)
    return display_df.to_string()


def format_bias_aware_summary(res: Dict[str, Any], label: str | None = None) -> str:
    """Render a single, unified bias-aware summary string.

    Parameters
    ----------
    res : Dict[str, Any]
        The result dictionary from compute_bias_aware_ci.
    label : str, optional, default None
        The label for the estimand.

    Returns
    -------
    str
        Formatted summary string.
    """
    alpha = res["alpha"]
    cf = dict(res["params"])
    cf.setdefault("cf_y", cf.get("r2_y", np.nan))
    cf.setdefault("cf_d", cf.get("r2_d", np.nan))
    summary_df = _summary_dataframe_from_result(res, label=label)

    lines = []
    lines.append("================== Bias-aware Interval ==================")
    lines.append("")
    lines.append("------------------ Scenario          ------------------")
    lines.append(f"Significance Level: alpha={alpha}")
    lines.append(f"Null Hypothesis: H0={res.get('H0', 0.0)}")
    if res.get("cov_type") == "cluster":
        lines.append(f"Clustered inference: {res['n_clusters']} clusters; t({res['inference_df']})")
    lines.append(
        f"Sensitivity parameters: cf_y={cf['cf_y']}; cf_d={cf['cf_d']}, rho={cf['rho']}, use_signed_rr={cf['use_signed_rr']}"
    )
    lines.append("")
    lines.append(_format_summary_table(summary_df))

    return "\n".join(lines)


def _wrap_sensitivity_result(
    res: Dict[str, Any],
    *,
    label: str | None = None,
) -> SensitivityAnalysisResult:
    """Attach rich summary behavior to a dict-compatible sensitivity result."""
    return SensitivityAnalysisResult(
        res,
        summary_builder=_summary_dataframe_from_result,
        text_summary_builder=format_bias_aware_summary,
        summary_kwargs={"label": label},
    )


def get_sensitivity_summary(
    effect_estimation: Dict[str, Any] | Any,
    *,
    label: Optional[str] = None,
) -> Optional[str]:
    """Render a single, unified bias-aware summary string.

    If bias-aware components are missing, shows a sampling-only variant with max_bias=0
    and then formats via `format_bias_aware_summary` for consistency.

    Parameters
    ----------
    effect_estimation : Dict[str, Any] or Any
        The effect estimation object.
    label : str, optional, default None
        The label for the estimand.

    Returns
    -------
    Optional[str]
        Formatted summary string or None if extraction fails.

    Notes
    -----
    The summary reports the point estimate, standard error, and confidence
    intervals under both the null (no unobserved confounding) and the assumed
    level of confounding ($R^2_Y, R^2_D$).

    It also includes the Robustness Value (RV), which is the minimum strength
    of confounding ($R^2_Y = R^2_D$) required to change the conclusion
    (e.g., make the estimate non-significant or zero).

    Examples
    --------
    >>> from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import sensitivity_analysis, get_sensitivity_summary
    >>> # Assuming 'estimate' is a fitted CausalEstimate from IRM
    >>> res = sensitivity_analysis(estimate, cf_y=0.05, cf_d=0.05) # doctest: +SKIP
    >>> summary = get_sensitivity_summary(estimate) # doctest: +SKIP
    >>> print(summary) # doctest: +SKIP
    """

    if isinstance(effect_estimation, SensitivityAnalysisResult):
        return effect_estimation.text_summary()
    if isinstance(effect_estimation, dict) and "model" not in effect_estimation:
        return None
    alpha = getattr(effect_estimation, "alpha", None)
    if alpha is None and isinstance(effect_estimation, dict):
        alpha = effect_estimation.get("alpha")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        alpha = 0.05
    if not 0.0 < alpha < 1.0:
        alpha = 0.05
    try:
        estimate = resolve_estimate(effect_estimation, alpha)
    except TypeError:
        return None
    result = estimate.cached_result
    if not isinstance(result, dict) or not result:
        result = _sampling_only_result(estimate)
    return format_bias_aware_summary(
        result, label=estimate.label if label is None else label
    )


def _sampling_only_result(estimate: Any) -> dict[str, Any]:
    theta, se, alpha = estimate.theta, estimate.se, estimate.alpha
    reference, metadata = _sensitivity_reference(estimate.source)
    z = float(reference.ppf(1 - alpha / 2.0))
    ci = (float(theta - z * se), float(theta + z * se))
    return dict(
        theta=float(theta),
        se=float(se),
        alpha=float(alpha),
        z=z,
        H0=0.0,
        sampling_ci=ci,
        theta_bounds_cofounding=(float(theta), float(theta)),
        bias_aware_ci=ci,
        max_bias_base=0.0,
        max_bias=0.0,
        bound_width=0.0,
        sigma2=np.nan,
        nu2=np.nan,
        rv=np.nan,
        rva=np.nan,
        **metadata,
        params=dict(cf_y=0.0, cf_d=0.0, rho=0.0, use_signed_rr=False),
    )


def interpret_result(res: dict[str, Any]) -> dict[str, Any]:
    """Describe a completed sensitivity analysis without recomputing or caching it."""
    H0 = float(res["H0"])

    def _ci_excludes_h0(ci: tuple[float, float], h0: float) -> bool:
        """Return whether a confidence interval excludes the null value."""
        lo, hi = float(ci[0]), float(ci[1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        return (lo > h0) or (hi < h0)

    theta = float(res.get("theta", np.nan))
    se = float(res.get("se", np.nan))
    sampling_ci = tuple(map(float, res.get("sampling_ci", (np.nan, np.nan))))
    bias_aware_ci = tuple(map(float, res.get("bias_aware_ci", (np.nan, np.nan))))
    confounding_bounds = tuple(
        map(float, res.get("theta_bounds_cofounding", (np.nan, np.nan)))
    )
    rv = float(res.get("rv", np.nan))
    rva = float(res.get("rva", np.nan))

    if theta > H0:
        direction = "positive"
    elif theta < H0:
        direction = "negative"
    else:
        direction = "null"

    significant_no_confounding = _ci_excludes_h0(sampling_ci, H0)
    significant_with_assumed_confounding = _ci_excludes_h0(bias_aware_ci, H0)

    if not np.isfinite(rv):
        robustness_level = "not_available"
    elif rv < 0.02:
        robustness_level = "low"
    elif rv < 0.05:
        robustness_level = "moderate"
    else:
        robustness_level = "high"

    rv_str = "nan" if not np.isfinite(rv) else f"{rv:.4f}"
    rva_str = "nan" if not np.isfinite(rva) else f"{rva:.4f}"
    summary = (
        f"Effect estimate theta={theta:.4f} ({direction}), se={se:.4f}. "
        f"Sampling CI [{sampling_ci[0]:.4f}, {sampling_ci[1]:.4f}] "
        f"{'excludes' if significant_no_confounding else 'includes'} H0={H0:.4f}. "
        f"Under specified confounding, theta bounds are "
        f"[{confounding_bounds[0]:.4f}, {confounding_bounds[1]:.4f}] and bias-aware CI is "
        f"[{bias_aware_ci[0]:.4f}, {bias_aware_ci[1]:.4f}], which "
        f"{'excludes' if significant_with_assumed_confounding else 'includes'} H0. "
        f"RV={rv_str}, RVa={rva_str}, robustness={robustness_level}."
    )

    return {
        "raw": res,
        "interpretation": {
            "direction": direction,
            "significant_no_confounding": significant_no_confounding,
            "significant_with_assumed_confounding": significant_with_assumed_confounding,
            "robustness_level": robustness_level,
            "rv": rv,
            "rva": rva,
        },
        "summary": summary,
    }


def format_protocol_summary(decision: ProtocolDecision, policy: ProtocolPolicy) -> str:
    """Format a decision already made by the protocol."""
    rule = (
        f"ci_lower > {policy.decision_threshold:g}"
        if policy.direction == "positive"
        else f"ci_upper < {policy.decision_threshold:g}"
    )
    if not policy.preconditions_passed:
        return (
            "FAIL: the external design/overlap/nuisance preconditions did not pass; "
            "the causal claim is not established."
        )
    if decision.inference_issue:
        return f"FAIL: sensitivity inference is invalid: {decision.inference_issue}; the sensitivity gate fails closed."
    if decision.invalid_primary:
        return (
            "FAIL: primary benchmark gain statistics could not be calibrated for "
            f"{decision.invalid_primary}; the sensitivity gate fails closed."
        )
    if not decision.sensitivity_passed:
        return (
            "FAIL: under hidden confounding comparable to primary benchmark(s) "
            f"{decision.failed_primary}, the bias-aware confidence interval does not satisfy the "
            f"{policy.direction} decision rule ({rule}) at the pre-specified "
            "decision threshold; the causal claim is not robust."
        )
    margin = "limited" if decision.limited_margin else "strong"
    boundary = (
        f" DoubleML boundary calibration was used for {decision.boundary_benchmarks}."
        if decision.boundary_benchmarks
        else ""
    )
    return (
        "PASS: the practically meaningful effect is robust to hidden confounding "
        "comparable to every primary benchmark. "
        f"Every primary bias-aware confidence interval satisfies the {policy.direction} "
        f"decision rule ({rule}). "
        f"The reported stress-test margin is {margin}." + boundary
    )
