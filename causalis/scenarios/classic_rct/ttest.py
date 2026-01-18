"""
T-test inference for Diff_in_Means model
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

from causalis.dgp.causaldata import CausalData


def ttest(data: CausalData, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform a t-test to compare the outcome between treated and control groups.

    This function performs an independent two-sample t-test (Welch's t-test)
    on a CausalData object to compare the outcome variable between treated (D=1)
    and control (D=0) groups. It returns the p-value, absolute and relative
    differences, and their corresponding confidence intervals.

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and outcome variables.
    alpha : float, default 0.05
        The significance level for calculating confidence intervals (between 0 and 1).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - p_value: The p-value from the t-test.
        - absolute_difference: The absolute difference between treatment and control means.
        - absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
        - relative_difference: The relative difference (percentage change) between means.
        - relative_ci: Tuple of (lower, upper) bounds for the relative difference CI.

    Raises
    ------
    ValueError
        If the CausalData object doesn't have both treatment and outcome variables
        defined, or if the treatment variable is not binary.
    """
    # Basic validation: ensure treatment and outcome are proper Series and non-empty
    treatment_var = data.treatment
    outcome_var = data.outcome

    if not isinstance(treatment_var, pd.Series) or treatment_var.empty:
        raise ValueError("causaldata object must have a treatment variable defined")
    if not isinstance(outcome_var, pd.Series) or outcome_var.empty:
        raise ValueError("causaldata object must have a outcome variable defined")

    # Ensure binary treatment
    unique_treatments = treatment_var.unique()
    if len(unique_treatments) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")

    # Build groups by conventional 0/1 coding
    control_data = outcome_var[treatment_var == 0]
    treatment_data = outcome_var[treatment_var == 1]

    # Sample sizes
    n1 = int(len(treatment_data))
    n2 = int(len(control_data))

    # Guard against degenerate cases with very small groups
    if n1 < 2 or n2 < 2:
        raise ValueError("Not enough observations in one of the groups for t-test (need at least 2 per group)")

    # Independent two-sample t-test (Welch's t-test by default with equal_var=False)
    res = stats.ttest_ind(treatment_data, control_data, equal_var=False)
    p_value = float(res.pvalue)
    df = float(res.df)

    # Means
    control_mean = float(control_data.mean())
    treatment_mean = float(treatment_data.mean())

    # Absolute difference
    absolute_diff = treatment_mean - control_mean

    # Variances
    s1_squared = float(treatment_data.var(ddof=1))
    s2_squared = float(control_data.var(ddof=1))

    # Standard error using Welch's formula (unpooled variance)
    se_diff = float(np.sqrt(s1_squared / n1 + s2_squared / n2))

    # Confidence interval
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")

    t_critical = float(stats.t.ppf(1 - alpha / 2, df))

    margin_of_error = t_critical * se_diff
    absolute_ci = (absolute_diff - margin_of_error, absolute_diff + margin_of_error)

    # Relative difference (%), relative CI
    if control_mean == 0:
        relative_diff = np.inf if absolute_diff > 0 else -np.inf if absolute_diff < 0 else 0.0
        relative_ci = (np.nan, np.nan)
    else:
        relative_diff = (absolute_diff / abs(control_mean)) * 100.0
        relative_margin = (margin_of_error / abs(control_mean)) * 100.0
        relative_ci = (relative_diff - relative_margin, relative_diff + relative_margin)

    return {
        "p_value": float(p_value),
        "absolute_difference": float(absolute_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
