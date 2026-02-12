from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from causalis.dgp.causaldata import CausalData


def outcome_outliers(
    data: CausalData,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    *,
    method: Literal["iqr", "zscore"] = "iqr",
    iqr_k: float = 1.5,
    z_thresh: float = 3.0,
    tail: Literal["both", "lower", "upper"] = "both",
    return_rows: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outcome outliers per treatment group using IQR or z-score rules.

    Parameters
    ----------
    data : CausalData
        Causal dataset containing the dataframe and metadata.
    treatment : str, optional
        Treatment column name. Defaults to `data.treatment`.
    outcome : str, optional
        Outcome column name. Defaults to `data.outcome`.
    method : {"iqr", "zscore"}, default "iqr"
        Outlier detection rule.
    iqr_k : float, default 1.5
        Multiplier for the IQR rule.
    z_thresh : float, default 3.0
        Z-score threshold for the z-score rule.
    tail : {"both", "lower", "upper"}, default "both"
        Which tail(s) to flag as outliers.
    return_rows : bool, default False
        If True, also return the rows flagged as outliers (subset of `data.df`).

    Returns
    -------
    summary : pandas.DataFrame
        Per-treatment summary with counts, rates, bounds, and flags.
    outliers : pandas.DataFrame
        Only returned when `return_rows=True`. Subset of `data.df` containing
        flagged outlier rows.

    Notes
    -----
    Bounds are computed within each treatment group.
    """
    if not isinstance(data, CausalData):
        raise ValueError("data must be a CausalData object.")
    df = data.df

    t_attr = getattr(data, "treatment", None)
    t_col_default = t_attr.name if isinstance(t_attr, pd.Series) else t_attr
    y_attr = getattr(data, "outcome", None)
    y_col_default = y_attr.name if isinstance(y_attr, pd.Series) else y_attr

    t_col = treatment or t_col_default
    y_col = outcome or y_col_default

    if not t_col or not y_col:
        raise ValueError("treatment and outcome column names must be provided.")
    if t_col not in df.columns or y_col not in df.columns:
        raise ValueError("Specified treatment/outcome columns not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError("Outcome must be numeric for outlier detection.")

    if method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'.")
    if iqr_k <= 0:
        raise ValueError("iqr_k must be > 0.")
    if z_thresh <= 0:
        raise ValueError("z_thresh must be > 0.")
    if tail not in {"both", "lower", "upper"}:
        raise ValueError("tail must be 'both', 'lower', or 'upper'.")

    df_valid = df[[t_col, y_col]].dropna()
    if df_valid.empty:
        raise ValueError("No valid rows with both treatment and outcome present.")
    if not np.isfinite(df_valid[y_col].to_numpy(dtype=float)).all():
        raise ValueError("Outcome contains non-finite values.")

    rows = []
    flagged_rows = []

    for tr, g in df_valid.groupby(t_col, sort=False):
        y = g[y_col].to_numpy(dtype=float)
        n = y.size

        if method == "iqr":
            q1 = float(np.percentile(y, 25))
            q3 = float(np.percentile(y, 75))
            iqr = q3 - q1
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
        else:
            mean = float(np.mean(y))
            std = float(np.std(y, ddof=1)) if n > 1 else 0.0
            if not np.isfinite(std) or std <= 0:
                lower = mean
                upper = mean
            else:
                lower = mean - z_thresh * std
                upper = mean + z_thresh * std

        if tail == "both":
            mask = (y < lower) | (y > upper)
        elif tail == "lower":
            mask = y < lower
        else:
            mask = y > upper

        outlier_count = int(mask.sum())
        outlier_rate = float(outlier_count / n) if n else 0.0

        rows.append(
            {
                "treatment": tr,
                "n": int(n),
                "outlier_count": outlier_count,
                "outlier_rate": outlier_rate,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "has_outliers": bool(outlier_count > 0),
                "method": method,
                "tail": tail,
            }
        )

        if return_rows and outlier_count:
            flagged_index = g.index[mask]
            flagged_rows.append(df.loc[flagged_index])

    summary = pd.DataFrame(rows)

    if return_rows:
        if flagged_rows:
            outliers_df = pd.concat(flagged_rows, axis=0)
        else:
            outliers_df = df.head(0).copy()
        return summary, outliers_df

    return summary
