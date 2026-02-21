"""
Outcome shared grouped by treatment for CausalData.
"""

from __future__ import annotations

import pandas as pd
from causalis.dgp.causaldata import CausalData
from causalis.dgp.multicausaldata import MultiCausalData


def outcome_stats(data: CausalData | MultiCausalData) -> pd.DataFrame:
    """
    Comprehensive outcome shared grouped by treatment.

    Returns a DataFrame with detailed outcome shared for each treatment group,
    including count, mean, std, min, various percentiles, and max.
    This function provides comprehensive outcome analysis and returns
    data_contracts in a clean DataFrame format suitable for reporting.

    Parameters
    ----------
    data : CausalData or MultiCausalData
        The causal dataset containing treatment and outcome variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with treatment groups as index and the following columns:
        - count: number of observations in each group
        - mean: average outcome value
        - std: standard deviation of outcome
        - min: minimum outcome value
        - p10: 10th percentile
        - p25: 25th percentile (Q1)
        - median: 50th percentile (median)
        - p75: 75th percentile (Q3)
        - p90: 90th percentile
        - max: maximum outcome value

    Examples
    --------
    >>> stats = outcome_stats(causal_data)
    >>> print(stats)
       treatment  count      mean       std       min       p10       p25    median       p75       p90       max
    0          0   3000  5.123456  2.345678  0.123456  2.345678  3.456789  5.123456  6.789012  7.890123  9.876543
    1          1   2000  6.789012  2.456789  0.234567  3.456789  4.567890  6.789012  8.901234  9.012345  10.987654
    """
    if isinstance(data, MultiCausalData):
        df, y = data.df, data.outcome
        t_cols = list(data.treatment_names)
        assigned_idx = df[t_cols].to_numpy(dtype=int, copy=False).argmax(axis=1)
        assigned_treatment = pd.Categorical.from_codes(
            assigned_idx,
            categories=t_cols,
            ordered=True,
        )
        grouped = (
            pd.DataFrame({"treatment": assigned_treatment, y: df[y]})
            .groupby("treatment", sort=False, observed=True)[y]
        )
    else:
        df, y = data.df, data.outcome_name
        t = data.treatment_name
        # Keep legacy behavior for single-treatment CausalData.
        if not pd.api.types.is_numeric_dtype(df[t]):
            raise ValueError("Treatment must be numeric 0/1 for outcome_stats().")
        grouped = df.groupby(t)[y]

    # Calculate basic shared using built-in methods
    basic_stats = grouped.agg(['count', 'mean', 'std', 'min', 'median', 'max'])

    # Calculate percentiles separately to avoid pandas aggregation mixing issues
    p10 = grouped.quantile(0.10)
    p25 = grouped.quantile(0.25)
    p75 = grouped.quantile(0.75)
    p90 = grouped.quantile(0.90)

    # Combine all shared into a single DataFrame
    stats_df = pd.DataFrame({
        'count': basic_stats['count'],
        'mean': basic_stats['mean'],
        'std': basic_stats['std'],
        'min': basic_stats['min'],
        'p10': p10,
        'p25': p25,
        'median': basic_stats['median'],
        'p75': p75,
        'p90': p90,
        'max': basic_stats['max']
    })

    # Ensure the index is named appropriately and reset it to have 'treatment' as a column
    stats_df.index.name = 'treatment'
    stats_df = stats_df.reset_index()

    return stats_df
