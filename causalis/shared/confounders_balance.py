from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

if TYPE_CHECKING:
    from causalis.dgp.causaldata import CausalData
    from causalis.dgp.multicausaldata import MultiCausalData


def _compute_balance_table(
    df: pd.DataFrame,
    confounders: list[str],
    mask_d_0: np.ndarray,
    mask_d_1: np.ndarray,
) -> pd.DataFrame:
    X = df[confounders]

    # Convert categorical variables to dummy variables for analysis
    # Even if CausalData is strict, other inputs might not be.
    X_num = pd.get_dummies(X, drop_first=False)

    rows = []
    for col in X_num.columns:
        x = X_num[col].values.astype(float)

        # Calculate means for each treatment group
        mean_d_0 = float(np.mean(x[mask_d_0])) if mask_d_0.any() else np.nan
        mean_d_1 = float(np.mean(x[mask_d_1])) if mask_d_1.any() else np.nan

        # Absolute difference
        abs_diff = float(abs(mean_d_1 - mean_d_0)) if np.isfinite(mean_d_0) and np.isfinite(mean_d_1) else np.nan

        # Standardized mean difference (SMD)
        v_control = float(np.var(x[mask_d_0], ddof=1)) if np.sum(mask_d_0) > 1 else 0.0
        v_treated = float(np.var(x[mask_d_1], ddof=1)) if np.sum(mask_d_1) > 1 else 0.0
        pooled_std = float(np.sqrt((v_control + v_treated) / 2))
        smd = float((mean_d_1 - mean_d_0) / pooled_std) if pooled_std > 0 else 0.0

        # Kolmogorov-Smirnov test
        try:
            ks_res = ks_2samp(x[mask_d_1], x[mask_d_0])
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


def confounders_balance(
    data: CausalData | MultiCausalData,
    treatment_d_0: Optional[str] = None,
    treatment_d_1: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute balance diagnostics for confounders between two treatment groups.

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
    data : CausalData or MultiCausalData
        The causal dataset containing the dataframe and confounders.
    treatment_d_0 : str, optional
        For MultiCausalData, name of the first treatment column to compare.
        Mapped to output column `mean_d_0`.
    treatment_d_1 : str, optional
        For MultiCausalData, name of the second treatment column to compare.
        Mapped to output column `mean_d_1`.

    Returns
    -------
    pd.DataFrame
        Balance table sorted by |smd| (descending).
    """
    # Extract shared components from data contract object
    df = getattr(data, "df")
    confounders = list(getattr(data, "confounders"))

    is_multicausal = hasattr(data, "treatment_names") and hasattr(data, "control_treatment")
    if is_multicausal:
        if treatment_d_0 is None or treatment_d_1 is None:
            raise ValueError(
                "For MultiCausalData, provide two treatment columns to compare, "
                "for example confounders_balance(data, 'd_0', 'd_2')."
            )
        if treatment_d_0 == treatment_d_1:
            raise ValueError("Compared treatment columns must be different.")

        t_cols = list(getattr(data, "treatment_names"))
        missing = [t for t in (treatment_d_0, treatment_d_1) if t not in t_cols]
        if missing:
            raise ValueError(
                f"Compared treatment column(s) {missing} are not in MultiCausalData.treatment_names={t_cols}."
            )

        mask_d_0_full = df[treatment_d_0].to_numpy(dtype=int, copy=False) == 1
        mask_d_1_full = df[treatment_d_1].to_numpy(dtype=int, copy=False) == 1

        if not mask_d_0_full.any() or not mask_d_1_full.any():
            raise ValueError(
                "Both compared treatments must have at least one assigned row in the dataset."
            )

        selected = mask_d_0_full | mask_d_1_full
        df_cmp = df.loc[selected]
        mask_d_0 = mask_d_0_full[selected]
        mask_d_1 = mask_d_1_full[selected]

        return _compute_balance_table(
            df=df_cmp,
            confounders=confounders,
            mask_d_0=mask_d_0,
            mask_d_1=mask_d_1,
        )

    # Keep legacy CausalData behavior (single binary treatment column).
    t_attr = getattr(data, "treatment")
    treatment = t_attr.name if isinstance(t_attr, pd.Series) else t_attr

    t = df[treatment].astype(int).to_numpy(copy=False)
    mask_d_0 = t == 0
    mask_d_1 = t == 1

    return _compute_balance_table(
        df=df,
        confounders=confounders,
        mask_d_0=mask_d_0,
        mask_d_1=mask_d_1,
    )
