from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

if TYPE_CHECKING:
    from causalis.data.causaldata import CausalData


def confounders_balance(data: CausalData) -> pd.DataFrame:
    """
    Compute balance diagnostics for confounders between treatment groups.

    Produces a DataFrame containing expanded confounder columns (after one-hot
    encoding categorical variables if present) with:
      - confounders: name of the confounder
      - mean_d_0: mean value for control group (t=0)
      - mean_d_1: mean value for treated group (t=1)
      - abs_diff: abs(mean_d_1 - mean_d_0)
      - smd: standardized mean difference (Cohen's d using pooled std)
      - ks_pvalue: p-value for the KS test (rounded to 5 decimal places, non-scientific)

    Parameters
    ----------
    data : CausalData
        The causal dataset containing the dataframe, treatment, and confounders.
        Accepts CausalData or any object with `df`, `treatment`, and `confounders`
        attributes/properties.

    Returns
    -------
    pd.DataFrame
        Balance table sorted by |smd| (descending).
    """
    # Extract components from data object
    df = getattr(data, "df")

    # Handle both string column names (Lite) and Series properties (CausalData)
    t_attr = getattr(data, "treatment")
    treatment = t_attr.name if isinstance(t_attr, pd.Series) else t_attr

    # Both Lite and CausalData have 'confounders' returning List[str]
    confounders = list(getattr(data, "confounders"))

    X = df[confounders]
    t = df[treatment].astype(int).values

    # Convert categorical variables to dummy variables for analysis
    # Even if CausalData is strict, other inputs might not be.
    X_num = pd.get_dummies(X, drop_first=False)

    rows = []
    for col in X_num.columns:
        x = X_num[col].values.astype(float)

        mask_c = t == 0
        mask_t = t == 1

        # Calculate means for each treatment group
        mean_d_0 = float(np.mean(x[mask_c])) if mask_c.any() else np.nan
        mean_d_1 = float(np.mean(x[mask_t])) if mask_t.any() else np.nan

        # Absolute difference
        abs_diff = float(abs(mean_d_1 - mean_d_0)) if np.isfinite(mean_d_0) and np.isfinite(mean_d_1) else np.nan

        # Standardized mean difference (SMD)
        v_control = float(np.var(x[mask_c], ddof=1)) if np.sum(mask_c) > 1 else 0.0
        v_treated = float(np.var(x[mask_t], ddof=1)) if np.sum(mask_t) > 1 else 0.0
        pooled_std = float(np.sqrt((v_control + v_treated) / 2))
        smd = float((mean_d_1 - mean_d_0) / pooled_std) if pooled_std > 0 else 0.0

        # Kolmogorov–Smirnov test
        try:
            ks_res = ks_2samp(x[mask_t], x[mask_c])
            ks_pvalue = float(ks_res.pvalue)
        except Exception:
            ks_pvalue = np.nan

        rows.append(
            {
                "confounders": col,
                "mean_d_0": mean_d_0,
                "mean_d_1": mean_d_1,
                "abs_diff": abs_diff,
                "smd": smd,
                "ks_pvalue": ks_pvalue,
            }
        )

    balance_df = pd.DataFrame(rows)

    # Sort by absolute SMD value (most imbalanced first)
    if "smd" in balance_df.columns:
        balance_df["abs_smd"] = balance_df["smd"].abs()
        balance_df = balance_df.sort_values("abs_smd", ascending=False).drop(columns=["abs_smd"])

    # Round ks_pvalue and format to avoid scientific notation
    if "ks_pvalue" in balance_df.columns:
        balance_df["ks_pvalue"] = (
            balance_df["ks_pvalue"]
            .apply(lambda x: f"{round(float(x), 5):.5f}" if pd.notnull(x) else x)
        )

    return balance_df.reset_index(drop=True)
