"""
Bootstrap difference-in-means inference.

This module computes the ATE-style difference in means (treated - control) and provides:
- Two-sided p-value using a normal approximation with bootstrap standard error.
- Percentile confidence interval for the absolute difference.
- Relative difference (%) and corresponding CI relative to control mean.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from causalis.dgp.causaldata import CausalData


def bootstrap_diff_means(
    data: CausalData,
    alpha: float = 0.05,
    n_simul: int = 10000,
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

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - p_value: Two-sided p-value using normal approximation.
        - absolute_difference: The absolute difference (treated - control).
        - absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
        - relative_difference: The relative difference (%) relative to control mean.
        - relative_ci: Tuple of (lower, upper) bounds for the relative difference CI.

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

    treatment = data.treatment
    outcome = data.outcome

    if not isinstance(treatment, pd.Series) or treatment.empty:
        raise ValueError("causaldata object must have a treatment variable defined")
    if not isinstance(outcome, pd.Series) or outcome.empty:
        raise ValueError("causaldata object must have a outcome variable defined")

    uniq = treatment.unique()
    if len(uniq) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")

    control = outcome[treatment == 0]
    treated = outcome[treatment == 1]

    n0 = int(control.shape[0])
    n1 = int(treated.shape[0])
    if n0 < 1 or n1 < 1:
        raise ValueError("Not enough observations in one of the groups for bootstrap (need at least 1 per group)")

    control_mean = float(control.mean())
    treated_mean = float(treated.mean())
    abs_diff = float(treated_mean - control_mean)

    # Prepare for bootstrap: indices for resampling within each group
    ctrl_vals = control.to_numpy()
    trt_vals = treated.to_numpy()
    rng = np.random.default_rng()

    # Vectorized bootstrap using random integers for indices
    ctrl_idx = rng.integers(0, n0, size=(n_simul, n0))
    trt_idx = rng.integers(0, n1, size=(n_simul, n1))

    ctrl_boot_means = ctrl_vals[ctrl_idx].mean(axis=1)
    trt_boot_means = trt_vals[trt_idx].mean(axis=1)
    boot_diffs = trt_boot_means - ctrl_boot_means

    # Percentile CI for absolute difference
    lower = float(np.quantile(boot_diffs, alpha / 2))
    upper = float(np.quantile(boot_diffs, 1 - alpha / 2))
    absolute_ci = (lower, upper)

    # p-value using bootstrap SE and normal approximation
    se_boot = float(np.std(boot_diffs, ddof=1))
    if se_boot == 0:
        p_value = 1.0
    else:
        z = abs_diff / se_boot
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Relative effects and CI by scaling
    if control_mean == 0:
        relative_diff = np.inf if abs_diff > 0 else 0.0 if abs_diff == 0 else -np.inf
        relative_ci = (np.nan, np.nan)
    else:
        relative_diff = (abs_diff / abs(control_mean)) * 100.0
        rel_lower = (lower / abs(control_mean)) * 100.0
        rel_upper = (upper / abs(control_mean)) * 100.0
        relative_ci = (float(rel_lower), float(rel_upper))

    return {
        "p_value": float(p_value),
        "absolute_difference": float(abs_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
