from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def confounders_balance(
    df: pd.DataFrame,
    treatment: str,
    confounders: Iterable[str],
) -> pd.DataFrame:
    """Compute balance diagnostics for confounders between treatment groups.

    Produces a DataFrame indexed by expanded confounder columns (after one-hot
    encoding categorical variables) with:
      - mean_t_0: mean value for control group (t=0)
      - mean_t_1: mean value for treated group (t=1)
      - abs_diff: abs(mean_t_1 - mean_t_0)
      - smd: standardized mean difference (Cohen's d using pooled std)
      - ks: Kolmogorov–Smirnov statistic between treated and control
      - ks_pvalue: p-value for the KS test

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe containing treatment and confounders.
    treatment : str
        Name of the treatment column (binary, 0/1 or False/True).
    confounders : Iterable[str]
        Column names of confounders to assess.

    Returns
    -------
    pd.DataFrame
        Balance table sorted by |smd| (descending), index named 'confounders'.
    """
    X = df[list(confounders)]
    t = df[treatment].astype(int).values

    # Convert categorical variables to dummy variables for analysis
    X_num = pd.get_dummies(X, drop_first=False)

    rows = []
    for col in X_num.columns:
        x = X_num[col].values.astype(float)

        mask_c = t == 0
        mask_t = t == 1

        # Calculate means for each treatment group
        mean_t_0 = float(np.mean(x[mask_c])) if mask_c.any() else np.nan
        mean_t_1 = float(np.mean(x[mask_t])) if mask_t.any() else np.nan

        # Absolute difference
        abs_diff = float(abs(mean_t_1 - mean_t_0)) if np.isfinite(mean_t_0) and np.isfinite(mean_t_1) else np.nan

        # Standardized mean difference (SMD)
        v_control = float(np.var(x[mask_c], ddof=1)) if np.sum(mask_c) > 1 else 0.0
        v_treated = float(np.var(x[mask_t], ddof=1)) if np.sum(mask_t) > 1 else 0.0
        pooled_std = float(np.sqrt((v_control + v_treated) / 2))
        smd = float((mean_t_1 - mean_t_0) / pooled_std) if pooled_std > 0 else 0.0

        # Kolmogorov–Smirnov test (works for continuous or dummy-discrete features)
        try:
            ks_res = ks_2samp(x[mask_t], x[mask_c])
            ks_stat = float(ks_res.statistic)
            ks_pvalue = float(ks_res.pvalue)
        except Exception:
            # Fallback to NaN if something unforeseen happens
            ks_stat = np.nan
            ks_pvalue = np.nan

        rows.append(
            {
                "confounders": col,
                "mean_t_0": mean_t_0,
                "mean_t_1": mean_t_1,
                "abs_diff": abs_diff,
                "smd": smd,
                "ks": ks_stat,
                "ks_pvalue": ks_pvalue,
            }
        )

    balance_df = pd.DataFrame(rows).set_index("confounders")

    # Sort by absolute SMD value (most imbalanced first)
    if "smd" in balance_df.columns:
        balance_df = balance_df.reindex(balance_df["smd"].abs().sort_values(ascending=False).index)

    return balance_df
