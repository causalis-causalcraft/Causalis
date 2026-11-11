r"""Public sensitivity API for binary unconfoundedness models.

Input adaptation, numerical inference, benchmark refits, decision policy, and
report rendering live in separate private modules. This facade preserves the
established imports and owns the caching performed by standalone analyses.

Confounder strengths follow Chernozhukov et al. (2022), "Long Story Short":
``cf_y`` is the share of short-model residual outcome variance explained by
the omitted confounder. ``cf_d = 1 - R^2_{alpha ~ alpha_s}`` is the share of
long Riesz-representer variance not explained by the short representer.
The bias bound is
``abs(rho) * sqrt(sigma2 * nu2) * sqrt(cf_y * cf_d / (1 - cf_d))``.
"""

from __future__ import annotations
from typing import Any, Dict
from causalis.shared.sensitivity_params import bias_factor, resolve_strength_aliases
import numpy as np
from causalis.scenarios.unconfoundedness import _sensitivity_core as core
from causalis.scenarios.unconfoundedness._sensitivity_inputs import (
    ResolvedEstimate,
    compute_irm_sensitivity_elements,
    resolve_estimate,
)
from ._sensitivity_benchmark import (
    _calibrate_benchmark_gain_statistics,
    sensitivity_benchmark,
    sensitivity_benchmark_group,
)
from ._sensitivity_protocol import run_sensitivity_protocol
from . import _sensitivity_reporting as reporting
from ._sensitivity_reporting import format_bias_aware_summary, get_sensitivity_summary

__all__ = [
    "sensitivity_analysis",
    "sensitivity_benchmark",
    "sensitivity_benchmark_group",
    "run_sensitivity_protocol",
    "get_sensitivity_summary",
    "interpret_sensitivity_analysis",
]


def _analyze(
    effect: Any,
    *,
    cf_y: float | None = None,
    cf_d: float | None = None,
    rho: float,
    H0: float,
    alpha: float,
    use_signed_rr: bool,
    r2_y: float | None = None,
    r2_d: float | None = None,
) -> tuple[ResolvedEstimate, dict[str, Any]]:
    if use_signed_rr:
        raise ValueError(
            "use_signed_rr=True is unsupported: the quadratic construction is not a valid sensitivity bound."
        )
    cf_y, cf_d = resolve_strength_aliases(
        cf_y=cf_y, cf_d=cf_d, r2_y=r2_y, r2_d=r2_d
    )
    bias_factor(cf_y, cf_d)  # Validate the shared public domain [0, 1).
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    estimate = resolve_estimate(effect, alpha)
    core.validate_inference(estimate.theta, estimate.se, alpha)
    context = core.prepare_sensitivity_context(
        theta=estimate.theta, se=estimate.se, alpha=alpha, elements=estimate.elements()
    )
    result = core.evaluate_sensitivity(
        context, cf_y=float(cf_y), cf_d=float(cf_d), rho=float(rho), H0=float(H0)
    )
    return estimate, result


def compute_bias_aware_ci(
    effect_estimation: Dict[str, Any] | Any,
    *,
    cf_y: float | None = None,
    cf_d: float | None = None,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
    r2_y: float | None = None,
    r2_d: float | None = None,
) -> Dict[str, Any]:
    r"""Compute orthogonal bias-aware confidence intervals for binary IRM.

    The bound width is
    ``abs(rho) * sqrt(sigma2 * nu2 * cf_y * cf_d / (1-cf_d))``.
    ``cf_y`` is the explained residual outcome share in [0, 1); ``cf_d`` is
    the Riesz-representer strength in [0, 1). There is no additional odds
    transformation of the outcome channel.

    Complete finite influence-function vectors are required for inference.
    ``RV`` and ``RVa`` use equal strength ``cf_y=cf_d``; ``RVa`` accounts for
    the changing standard error of the relevant bound. Unattainable robustness
    values (e.g. rho=0 for an initially significant effect) are NaN.
    ``alpha`` is a two-sided significance level. ``use_signed_rr=True`` is
    rejected because the former quadratic construction is not a valid bias
    bound in this framework. The result dictionary retains its existing keys.

    Stored cluster membership automatically selects joint one-way CR1
    covariance and Student's t reference with G-1 degrees of freedom. The
    sensitivity moments and confounding widths remain observation weighted.
    ``r2_y`` and ``r2_d`` are deprecated aliases, used directly without
    an odds conversion; do not pass a canonical name and its alias together.
    """

    _, result = _analyze(
        effect_estimation,
        cf_y=cf_y,
        cf_d=cf_d,
        r2_y=r2_y,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr,
    )
    return result


def sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    cf_y: float | None = None,
    cf_d: float | None = None,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
    r2_y: float | None = None,
    r2_d: float | None = None,
) -> Dict[str, Any]:
    r"""Compute bias-aware components and cache them.

    This function turns a fitted estimate into a simple hidden-confounding
    stress test. In the default mode, the bound width is

    .. math::

        |\rho| \cdot \sqrt{\sigma^2 \nu^2}
        \cdot
        \sqrt{cf_y\frac{cf_d}{1-cf_d}},

    so the reported confounding interval is

    .. math::

        [\theta - \text{bound\_width}, \theta + \text{bound\_width}].

    Here :math:`cf_y` controls how much residual outcome variation an omitted
    confounder could explain. :math:`cf_d = 1 - R^2_{\alpha \sim \alpha_s}`
    measures the share of long Riesz-representer variance unexplained by the
    short representer, and :math:`|\rho|` sets their adversity. The sign of
    rho does not change these symmetric bounds.

    Parameters
    ----------
    effect_estimation : Dict[str, Any] or Any
        The effect estimation object.
    cf_y : float
        Explained residual outcome share in [0, 1), without an odds transformation.
    cf_d : float
        Riesz-representer strength in [0, 1).
    rho : float, default 1.0
        Correlation parameter.
    H0 : float, default 0.0
        Null hypothesis for robustness values.
    alpha : float, default 0.05
        Significance level.
    use_signed_rr : bool, default False
        Retained for signature compatibility. True is rejected because the former
        quadratic construction is not a valid sensitivity bound.
    r2_y, r2_d : float, optional
        Deprecated aliases for cf_y and cf_d, used directly without conversion.

    Returns
    -------
    dict
        Dictionary with bias-aware results:
          - theta, se, alpha, z
          - sampling_ci
          - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
          - bias_aware_ci = faithful CI for the bounds
          - max_bias and components (sigma2, nu2)
          - params (cf_y, cf_d, rho, use_signed_rr)

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    >>> from causalis.dgp import obs_linear_26_dataset
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> data = obs_linear_26_dataset(
    ...     n=1000,
    ...     seed=3141,
    ...     include_oracle=False,
    ...     return_causal_data=True,
    ... )
    >>> irm = IRM(
    ...     data=data,
    ...     ml_g=RandomForestRegressor(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     ml_m=RandomForestClassifier(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     n_folds=3,
    ...     random_state=3141,
    ... )
    >>> estimate = irm.fit().estimate(score="ATE")
    >>> out = sensitivity_analysis(estimate, cf_y=0.02, cf_d=0.02, rho=1.0)
    >>> out["theta_bounds_cofounding"]  # doctest: +SKIP
    >>> out["bias_aware_ci"]  # doctest: +SKIP
    """

    estimate, raw = _analyze(
        effect_estimation,
        cf_y=cf_y,
        cf_d=cf_d,
        r2_y=r2_y,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr,
    )
    result = reporting._wrap_sensitivity_result(raw, label=estimate.label)
    estimate.cache_result(result)
    return result


def interpret_sensitivity_analysis(
    effect_estimation: Dict[str, Any] | Any,
    *,
    cf_y: float | None = None,
    cf_d: float | None = None,
    rho: float = 1.0,
    H0: float = 0.0,
    alpha: float = 0.05,
    use_signed_rr: bool = False,
    r2_y: float | None = None,
    r2_d: float | None = None,
) -> Dict[str, Any]:
    """Run sensitivity analysis and return a structured interpretation.

    Parameters
    ----------
    effect_estimation : Dict[str, Any] or Any
        The effect estimation object.
    cf_y : float
        Explained short-model residual outcome share in [0, 1).
    cf_d : float
        Unexplained long Riesz-representer variance share in [0, 1).
    rho : float, default 1.0
        Correlation parameter for unobserved confounding.
    H0 : float, default 0.0
        Null hypothesis used for significance checks.
    alpha : float, default 0.05
        Significance level.
    use_signed_rr : bool, default False
        Retained for signature compatibility; True is unsupported and rejected.
    r2_y, r2_d : float, optional
        Deprecated aliases for cf_y and cf_d, used directly without conversion.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
          - raw: the output of ``sensitivity_analysis(...)``
          - interpretation: machine-readable interpretation fields
          - summary: compact human-readable interpretation

    Notes
    -----
    This function wraps `sensitivity_analysis` and provides a textual
    interpretation of the results, including whether the estimate remains
    significant under the assumed confounding.

    Examples
    --------
    >>> from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import interpret_sensitivity_analysis
    >>> # Assuming 'estimate' is a fitted CausalEstimate
    >>> interpretation = interpret_sensitivity_analysis(estimate, cf_y=0.01, cf_d=0.01) # doctest: +SKIP
    >>> print(interpretation["summary"]) # doctest: +SKIP
    """

    result = sensitivity_analysis(
        effect_estimation,
        cf_y=cf_y,
        cf_d=cf_d,
        r2_y=r2_y,
        r2_d=r2_d,
        rho=rho,
        H0=H0,
        alpha=alpha,
        use_signed_rr=use_signed_rr,
    )
    return reporting.interpret_result(result)
