"""Numerical sensitivity calculations for binary IRM, independent of model objects."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np
from ._cluster_utils import compact_cluster_codes, inference_metadata, mean_covariance


def _finite_vector(values: Any, name: str, n: int | None = None) -> np.ndarray:
    """Validate score vectors before arithmetic can accidentally broadcast them."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2 or not np.isfinite(arr).all():
        raise ValueError(
            f"{name} must be a finite one-dimensional vector with at least two observations."
        )
    if n is not None and arr.size != n:
        raise ValueError(f"{name} must have sample size {n}, got {arr.size}.")
    return arr


def _bias_and_influence(
    sigma2: np.ndarray | float,
    nu2: np.ndarray | float,
    psi_sigma2: np.ndarray,
    psi_nu2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute max bias and its influence function.

    max_bias = sqrt(sigma2 * nu2). Influence function via delta method.
    Negative/non-finite moments are rejected; a zero residual variance has zero IF.

    Parameters
    ----------
    sigma2 : np.ndarray or float
        Variance of the outcome residuals.
    nu2 : np.ndarray or float
        Variance related to the Riesz representer.
    psi_sigma2 : np.ndarray
        Influence function for sigma2.
    psi_nu2 : np.ndarray
        Influence function for nu2.

    Returns
    -------
    max_bias : float
        The maximum bias.
    psi_max_bias : np.ndarray
        The influence function for the maximum bias.
    """
    sigma2_f = float(np.asarray(sigma2).reshape(()))
    nu2_f = float(np.asarray(nu2).reshape(()))
    if not np.isfinite([sigma2_f, nu2_f]).all() or sigma2_f < 0.0 or nu2_f <= 0.0:
        raise ValueError(
            "sigma2 must be non-negative and nu2 positive; both must be finite."
        )
    if sigma2_f == 0.0:
        return 0.0, np.zeros_like(psi_sigma2, dtype=float)
    max_bias = float(np.sqrt(sigma2_f) * np.sqrt(nu2_f))
    psi_sigma2 = np.asarray(psi_sigma2, float)
    psi_sigma2 = psi_sigma2 - float(np.mean(psi_sigma2))
    psi_nu2 = np.asarray(psi_nu2, float)
    psi_nu2 = psi_nu2 - float(np.mean(psi_nu2))
    sigma, nu = np.sqrt(sigma2_f), np.sqrt(nu2_f)
    psi_max_bias = 0.5 * ((sigma / nu) * psi_nu2 + (nu / sigma) * psi_sigma2)
    return max_bias, psi_max_bias


@dataclass
class SensitivityContext:
    """Scenario-independent moments; all scenario arithmetic is scalar."""

    theta: float
    se: float
    alpha: float
    z: float
    sigma2: float
    nu2: float
    max_bias_base: float
    var_theta: float
    var_bias: float
    cov_theta_bias: float
    inference: dict[str, Any] = field(default_factory=dict)
    robustness: dict[tuple[float, float], tuple[float, float]] = field(
        default_factory=dict
    )

    def bound_se(self, strength: float, side: float) -> float:
        variance = (
            self.var_theta
            + 2.0 * side * strength * self.cov_theta_bias
            + strength**2 * self.var_bias
        )
        return float(np.sqrt(max(variance, 0.0)))


def validate_inference(theta: float, se: float, alpha: float) -> None:
    """Validate scalar inference inputs before resolving expensive nuisance elements."""
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite.")
    if not np.isfinite(se) or se < 0.0:
        raise ValueError("The standard error must be finite and non-negative.")


def prepare_sensitivity_context(
    *,
    theta: float,
    se: float,
    alpha: float,
    elements: dict[str, Any],
) -> SensitivityContext:
    """Validate influence functions and precompute scalar inference moments."""
    from scipy.stats import norm, t

    validate_inference(theta, se, alpha)
    elems = elements
    if not isinstance(elems, dict):
        raise ValueError("Sensitivity elements must be a dictionary.")
    try:
        sigma2, nu2 = float(elems["sigma2"]), float(elems["nu2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Finite sigma2 and nu2 sensitivity elements are required."
        ) from exc
    if not np.isfinite(sigma2) or sigma2 < 0.0:
        raise ValueError("sigma2 must be finite and non-negative.")
    if not np.isfinite(nu2) or nu2 <= 0.0:
        raise ValueError("nu2 must be finite and positive.")
    psi = _finite_vector(elems.get("psi"), "psi")
    n = psi.size
    psi_sigma2 = _finite_vector(elems.get("psi_sigma2"), "psi_sigma2", n)
    psi_nu2 = _finite_vector(elems.get("psi_nu2"), "psi_nu2", n)
    if sigma2 == 0.0 and np.any(psi_sigma2 != 0.0):
        raise ValueError("A zero sigma2 requires a zero psi_sigma2.")
    raw_codes = elems.get("cluster_codes")
    codes = None if raw_codes is None else compact_cluster_codes(raw_codes, n)
    if codes is not None and elems.get("cov_type") == "iid":
        raise ValueError("Iid covariance metadata conflicts with cluster membership.")
    if codes is None and (
        elems.get("cov_type") == "cluster" or elems.get("cluster_name") is not None
    ):
        raise ValueError("Clustered sensitivity inference requires stored cluster membership.")
    metadata = inference_metadata(codes)
    if not metadata.get("inferential_available", True):
        raise ValueError("Clustered sensitivity inference requires at least two retained clusters.")

    # Work with the joint moments before applying the delta method. Aggregating
    # the three vectors separately avoids an n-by-3 temporary at large n.
    covariance = mean_covariance(psi, psi_sigma2, psi_nu2, cluster_codes=codes)
    base = float(np.sqrt(sigma2) * np.sqrt(nu2))
    if sigma2 == 0.0:
        var_bias = cov_theta_bias = 0.0
    else:
        sigma, nu = np.sqrt(sigma2), np.sqrt(nu2)
        gradient = np.array([nu / sigma, sigma / nu]) / 2.0
        var_bias = float(gradient @ covariance[1:, 1:] @ gradient)
        cov_theta_bias = float(covariance[0, 1:] @ gradient)
    if not np.isfinite(covariance).all() or not np.isfinite(base):
        raise ValueError("Sensitivity influence-function moments must be finite.")
    reference = t(df=metadata["inference_df"]) if codes is not None else norm
    return SensitivityContext(
        theta,
        se,
        float(alpha),
        float(reference.ppf(1.0 - alpha / 2.0)),
        sigma2,
        nu2,
        base,
        float(covariance[0, 0]),
        var_bias,
        cov_theta_bias,
        metadata,
    )


def _equal_strength_r2(ratio: float) -> float:
    """Solve r / sqrt(1-r) = ratio without subtractive cancellation."""
    if ratio <= 1.0:
        return float(2.0 * ratio / (np.hypot(ratio, 2.0) + ratio))
    inverse_ratio = 1.0 / ratio
    return float(2.0 / (1.0 + np.hypot(1.0, 2.0 * inverse_ratio)))


def _robustness_values(
    ctx: SensitivityContext, rho: float, H0: float
) -> tuple[float, float]:
    from scipy.optimize import brentq

    key = (abs(rho), H0)
    if key in ctx.robustness:
        return ctx.robustness[key]
    delta = abs(ctx.theta - H0)
    side = -1.0 if ctx.theta >= H0 else 1.0
    baseline_margin = delta - ctx.z * ctx.bound_se(0.0, side)
    rv = 0.0 if delta == 0.0 else np.nan
    rva = 0.0 if baseline_margin <= 0.0 else np.nan
    if abs(rho) > 0.0 and ctx.max_bias_base > 0.0 and delta > 0.0:
        max_strength = delta / ctx.max_bias_base
        rv = _equal_strength_r2(max_strength / abs(rho))
        if baseline_margin > 0.0:
            # Scale the root variable to [0, 1] so very small biases do not
            # inherit an absolute tolerance on the original strength scale.
            def margin(fraction: float) -> float:
                strength = fraction * max_strength
                return delta * (1.0 - fraction) - ctx.z * ctx.bound_se(strength, side)

            fraction = brentq(margin, 0.0, 1.0, xtol=1e-14, rtol=1e-12)
            rva = _equal_strength_r2(fraction * max_strength / abs(rho))
    ctx.robustness[key] = (float(rv), float(rva))
    return ctx.robustness[key]


def evaluate_sensitivity(
    ctx: SensitivityContext,
    *,
    cf_y: float,
    cf_d: float,
    rho: float,
    H0: float,
) -> Dict[str, Any]:
    if not np.isfinite([cf_y, cf_d, rho, H0]).all():
        raise ValueError("cf_y, cf_d, rho, and H0 must be finite.")
    if not 0.0 <= cf_y <= 1.0:
        raise ValueError("cf_y must be in [0, 1].")
    if not 0.0 <= cf_d < 1.0:
        raise ValueError("cf_d must be in [0, 1).")
    if abs(rho) > 1.0:
        raise ValueError("rho must be in [-1, 1].")
    factor = float(np.sqrt(cf_y * cf_d / (1.0 - cf_d)))
    strength = abs(rho) * factor
    max_bias = ctx.max_bias_base * factor
    width = abs(rho) * max_bias
    lower, upper = ctx.theta - width, ctx.theta + width
    ci = (
        lower - ctx.z * ctx.bound_se(strength, -1.0),
        upper + ctx.z * ctx.bound_se(strength, 1.0),
    )
    if not np.isfinite([max_bias, width, *ci]).all():
        raise ValueError("Sensitivity bounds and confidence intervals must be finite.")
    rv, rva = _robustness_values(ctx, rho, H0)
    return dict(
        theta=ctx.theta,
        se=ctx.se,
        alpha=ctx.alpha,
        z=ctx.z,
        H0=float(H0),
        sampling_ci=(ctx.theta - ctx.z * ctx.se, ctx.theta + ctx.z * ctx.se),
        theta_bounds_cofounding=(float(lower), float(upper)),
        bias_aware_ci=tuple(map(float, ci)),
        max_bias_base=ctx.max_bias_base,
        max_bias=float(max_bias),
        bound_width=float(width),
        sigma2=ctx.sigma2,
        nu2=ctx.nu2,
        rv=rv,
        rva=rva,
        **ctx.inference,
        params=dict(
            cf_y=float(cf_y), cf_d=float(cf_d), rho=float(rho), use_signed_rr=False
        ),
    )


def compute_sensitivity_elements(
    *,
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    w: np.ndarray,
    w_bar: np.ndarray,
    inv_m: np.ndarray,
    inv_1m: np.ndarray,
    psi: np.ndarray | None,
) -> dict[str, Any]:
    """Compute moments from arrays validated by the IRM input adapter."""
    sigma2_score_element = np.square(y - d * g1 - (1.0 - d) * g0)
    sigma2 = float(np.mean(sigma2_score_element))
    psi_sigma2 = sigma2_score_element - sigma2

    m_alpha = w * w_bar * (inv_m + inv_1m)
    rr = w_bar * (d * inv_m - (1.0 - d) * inv_1m)
    nu2_score_element = 2.0 * m_alpha - np.square(rr)
    nu2 = float(np.mean(nu2_score_element))
    psi_nu2 = nu2_score_element - nu2

    return {
        "sigma2": sigma2,
        "nu2": nu2,
        "psi_sigma2": psi_sigma2,
        "psi_nu2": psi_nu2,
        "riesz_rep": rr,
        "m_alpha": m_alpha,
        "psi": psi,
    }
