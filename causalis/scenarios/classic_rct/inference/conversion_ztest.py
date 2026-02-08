"""
Two-proportion z-test

Compares conversion rates between treated (D=1) and control (D=0) groups.
Returns p-value, absolute/relative differences, and their confidence intervals
"""

from typing import Dict, Any, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from causalis.dgp.causaldata import CausalData


def conversion_ztest(
    data: CausalData,
    alpha: float = 0.05,
    ci_method: Literal["newcombe", "wald_unpooled", "wald_pooled"] = "newcombe",
    se_for_test: Literal["pooled", "unpooled"] = "pooled",
) -> Dict[str, Any]:
    """
    Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

    Parameters
    ----------
    data : CausalData
        The CausalData object containing treatment and outcome variables.
    alpha : float, default 0.05
        The significance level for calculating confidence intervals (between 0 and 1).
    ci_method : {"newcombe", "wald_unpooled", "wald_pooled"}, default "newcombe"
        Method for calculating the confidence interval for the absolute difference.
        "newcombe" is the most robust default for conversion rates.
    se_for_test : {"pooled", "unpooled"}, default "pooled"
        Method for calculating the standard error for the z-test p-value.
        "pooled" (score test) is generally preferred for testing equality of proportions.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - p_value: Two-sided p-value from the z-test
        - absolute_difference: Difference in conversion rates (treated - control)
        - absolute_ci: Tuple (lower, upper) for the absolute difference CI
        - relative_difference: Percentage change relative to control rate
        - relative_ci: Tuple (lower, upper) for the relative difference CI (delta method)

    Raises
    ------
    ValueError
        If treatment/outcome are missing, treatment is not binary, outcome is not binary,
        groups are empty, or alpha is outside (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")

    treatment_var = data.treatment
    outcome_var = data.outcome

    if not isinstance(treatment_var, pd.Series) or treatment_var.empty:
        raise ValueError("CausalData must have a non-empty treatment Series")
    if not isinstance(outcome_var, pd.Series) or outcome_var.empty:
        raise ValueError("CausalData must have a non-empty outcome Series")

    # Pairwise drop missing (prevents denominator/numerator mismatch)
    df = pd.concat(
        [treatment_var.rename("D"), outcome_var.rename("Y")],
        axis=1
    ).dropna()

    if df.empty:
        raise ValueError("No non-missing (treatment, outcome) pairs available")

    # Strict 0/1 validation
    d_set = set(pd.unique(df["D"]))
    y_set = set(pd.unique(df["Y"]))

    # allow bools (True/False) since they are 1/0 in Python
    if not d_set.issubset({0, 1, False, True}):
        raise ValueError("Treatment must be binary coded as 0/1 (or False/True)")
    if not y_set.issubset({0, 1, False, True}):
        raise ValueError("Outcome must be binary coded as 0/1 (or False/True)")

    # Convert to int 0/1 to avoid surprises
    df["D"] = df["D"].astype(int)
    df["Y"] = df["Y"].astype(int)

    if set(pd.unique(df["D"])) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1")

    control = df.loc[df["D"] == 0, "Y"]
    treat = df.loc[df["D"] == 1, "Y"]

    n0 = int(control.shape[0])
    n1 = int(treat.shape[0])
    if n0 < 1 or n1 < 1:
        raise ValueError("Need at least 1 observation per group")

    x0 = float(control.sum())
    x1 = float(treat.sum())

    p0 = x0 / n0
    p1 = x1 / n1
    absolute_diff = float(p1 - p0)

    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    # 1) p-value (two-sided)
    if se_for_test == "pooled":
        p_pool = (x0 + x1) / (n0 + n1)
        se_test = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1)))
    else:
        se_test = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))

    if se_test == 0.0:
        p_value = 1.0 if absolute_diff == 0.0 else 0.0
    else:
        z_stat = float(absolute_diff / se_test)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # 2) Absolute CI
    def wilson_ci(p: float, n: int, z: float) -> Tuple[float, float]:
        den = 1.0 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / den
        half = (z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n**2))) / den
        return float(center - half), float(center + half)

    if ci_method == "newcombe":
        l0, u0 = wilson_ci(p0, n0, z_crit)
        l1, u1 = wilson_ci(p1, n1, z_crit)
        absolute_ci = (float(l1 - u0), float(u1 - l0))
    elif ci_method == "wald_pooled":
        p_pool = (x0 + x1) / (n0 + n1)
        se_ci = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1)))
        margin = z_crit * se_ci
        absolute_ci = (absolute_diff - margin, absolute_diff + margin)
    else:  # wald_unpooled
        se_ci = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))
        margin = z_crit * se_ci
        absolute_ci = (absolute_diff - margin, absolute_diff + margin)

    # 3) Relative effect (% lift) and CI via delta method
    # lift = (p1/p0 - 1) * 100
    eps = 1e-12
    if (not np.isfinite(p0)) or abs(p0) < eps:
        relative_diff = np.inf if p1 > 0 else 0.0
        relative_ci = (np.nan, np.nan)
    else:
        rr = p1 / p0
        relative_diff = float((rr - 1.0) * 100.0)

        v1 = p1 * (1 - p1) / n1
        v0 = p0 * (1 - p0) / n0
        w1 = (1.0 / p0) ** 2
        w0 = (p1 / (p0 ** 2)) ** 2
        var_rel_scaled = float(max(w1 * v1 + w0 * v0, 0.0))
        relative_se = 100.0 * float(np.sqrt(var_rel_scaled))
        moe = z_crit * relative_se
        relative_ci = (relative_diff - moe, relative_diff + moe)

    return {
        "p_value": float(p_value),
        "absolute_difference": float(absolute_diff),
        "absolute_ci": (float(absolute_ci[0]), float(absolute_ci[1])),
        "relative_difference": float(relative_diff),
        "relative_ci": (float(relative_ci[0]), float(relative_ci[1])),
    }
