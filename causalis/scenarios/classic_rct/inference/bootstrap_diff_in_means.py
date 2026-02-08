"""
Bootstrap difference-in-means inference.

This module computes the ATE-style difference in means (treated - control) and provides:
- Two-sided p-value using a normal approximation with bootstrap standard error.
- Percentile confidence interval for the absolute difference.
- Relative difference (%) and corresponding CI relative to control mean.
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from causalis.dgp.causaldata import CausalData


def bootstrap_diff_means(
    data: CausalData,
    alpha: float = 0.05,
    n_simul: int = 10_000,
    *,
    batch_size: int = 512,
    seed: Optional[int] = None,
    index_dtype=np.int32,   # int32 halves RAM vs int64
) -> Dict[str, Any]:
    """
    Bootstrap inference for difference in means between treated and control groups.

    This function computes the ATE-style difference in means (treated - control)
    and provides a two-sided p-value using a normal approximation with bootstrap
    standard error, a percentile confidence interval for the absolute difference,
    and relative difference with its corresponding confidence interval.

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and outcome variables.
    alpha : float, default 0.05
        The significance level for calculating confidence intervals (between 0 and 1).
    n_simul : int, default 10000
        Number of bootstrap resamples.
    batch_size : int, default 512
        Number of bootstrap samples to process per batch.
    seed : int, optional
        Random seed for reproducibility.
    index_dtype : numpy dtype, default np.int32
        Integer dtype for bootstrap indices to reduce memory usage.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - p_value: Two-sided p-value using normal approximation.
        - absolute_difference: The absolute difference (treated - control).
        - absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
        - relative_difference: The relative difference (%) relative to control mean.
        - relative_ci: Tuple of (lower, upper) bounds for the relative difference CI (delta method).

    Raises
    ------
    ValueError
        If inputs are invalid, treatment is not binary, or groups are empty.
    """
    # Validate inputs
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    if not isinstance(n_simul, int) or n_simul <= 0:
        raise ValueError("n_simul must be a positive integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    treatment = data.treatment
    outcome = data.outcome

    if not isinstance(treatment, pd.Series) or treatment.empty:
        raise ValueError("causaldata object must have a treatment variable defined")
    if not isinstance(outcome, pd.Series) or outcome.empty:
        raise ValueError("causaldata object must have a outcome variable defined")

    uniq = pd.unique(treatment)
    if len(uniq) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")

    control = outcome[treatment == 0].to_numpy()
    treated = outcome[treatment == 1].to_numpy()

    # Optional: drop NaNs (bootstrap breaks if NaNs present)
    control = control[np.isfinite(control)]
    treated = treated[np.isfinite(treated)]

    n0 = int(control.shape[0])
    n1 = int(treated.shape[0])
    if n0 < 1 or n1 < 1:
        raise ValueError("Need at least 1 observation per group for bootstrap")

    control_mean = float(control.mean())
    treated_mean = float(treated.mean())
    abs_diff = float(treated_mean - control_mean)

    rng = np.random.default_rng(seed)

    boot_diffs = np.empty(n_simul, dtype=np.float64)
    ctrl_sum = 0.0
    ctrl_sumsq = 0.0
    trt_sum = 0.0
    trt_sumsq = 0.0

    # chunked vectorized bootstrap
    pos = 0
    while pos < n_simul:
        b = min(batch_size, n_simul - pos)

        ctrl_idx = rng.integers(0, n0, size=(b, n0), dtype=index_dtype)
        trt_idx = rng.integers(0, n1, size=(b, n1), dtype=index_dtype)

        # mean = sum / n  (tiny bit faster than mean() in some cases)
        ctrl_means = control[ctrl_idx].sum(axis=1) / n0
        trt_means = treated[trt_idx].sum(axis=1) / n1

        boot_diffs[pos:pos + b] = trt_means - ctrl_means
        ctrl_sum += float(ctrl_means.sum())
        ctrl_sumsq += float((ctrl_means * ctrl_means).sum())
        trt_sum += float(trt_means.sum())
        trt_sumsq += float((trt_means * trt_means).sum())
        pos += b

    # Percentile CI for absolute difference
    lower = float(np.quantile(boot_diffs, alpha / 2))
    upper = float(np.quantile(boot_diffs, 1 - alpha / 2))
    absolute_ci = (lower, upper)

    # p-value using bootstrap SE and normal approximation
    se_boot = float(np.std(boot_diffs, ddof=1))
    if se_boot == 0:
        p_value = 1.0 if abs_diff == 0 else 0.0
    else:
        z = abs_diff / se_boot
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Relative effects and CI via delta method
    outcome_scale = float(np.mean(np.abs(np.concatenate([control, treated])))) if (n0 + n1) > 0 else 1.0
    eps = 1e-12 * max(1.0, outcome_scale)

    if (not np.isfinite(control_mean)) or abs(control_mean) < eps:
        relative_diff = np.inf if abs_diff > 0 else 0.0 if abs_diff == 0 else -np.inf
        relative_ci = (np.nan, np.nan)
    else:
        denom = control_mean
        relative_diff = (abs_diff / denom) * 100.0

        if n_simul > 1:
            ctrl_var = (ctrl_sumsq - (ctrl_sum ** 2) / n_simul) / (n_simul - 1)
            trt_var = (trt_sumsq - (trt_sum ** 2) / n_simul) / (n_simul - 1)
            ctrl_var = float(max(ctrl_var, 0.0))
            trt_var = float(max(trt_var, 0.0))
        else:
            ctrl_var = 0.0
            trt_var = 0.0

        w1 = (1.0 / control_mean) ** 2
        w0 = (treated_mean / (control_mean ** 2)) ** 2
        var_rel_scaled = float(max(w1 * trt_var + w0 * ctrl_var, 0.0))
        relative_se = 100.0 * float(np.sqrt(var_rel_scaled))
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        moe_rel = z_crit * relative_se
        relative_ci = (relative_diff - moe_rel, relative_diff + moe_rel)

    return {
        "p_value": float(p_value),
        "absolute_difference": float(abs_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
