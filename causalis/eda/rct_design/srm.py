"""
Sample Ratio Mismatch (SRM) utilities for randomized experiments.

This module implements a chi-square goodness-of-fit SRM check mirroring the
reference implementation demonstrated in docs/examples/rct_design.ipynb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Union

import numpy as np
import pandas as pd

try:  # Optional SciPy dependency: only required at runtime for p-value
    from scipy.stats import chi2  # type: ignore
except ImportError as e:  # pragma: no cover
    chi2 = None  # type: ignore[assignment]
    _scipy_import_error = e
else:  # pragma: no cover - presence validated implicitly when used
    _scipy_import_error = None


Number = Union[int, float]


@dataclass
class SRMResult:
    """
    Result of a Sample Ratio Mismatch (SRM) check.
    """
    chi2: float
    df: int
    p_value: float
    expected: Dict[Hashable, float]
    observed: Dict[Hashable, int]
    alpha: float
    is_srm: bool
    warning: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - repr formatting
        status = "SRM DETECTED" if self.is_srm else "no SRM"
        return (
            f"SRMResult(status={status}, p_value={self.p_value:.3e}, "
            f"chi2={self.chi2:.4f}, df={self.df})"
        )


def check_srm(
    assignments: Union[Iterable[Hashable], pd.Series],
    target_allocation: Dict[Hashable, Number],
    alpha: float = 1e-3,
    min_expected: float = 5.0,
    strict_variants: bool = True,
) -> SRMResult:
    """
    Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

    Parameters
    ----------
    assignments:
        Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
        E.g. Series of ["control", "treatment", ...].

    target_allocation:
        Mapping {variant: p} describing intended allocation as PROBABILITIES.
        - Each p must be > 0.
        - Sum of all p must be 1.0 (within numerical tolerance).

        Examples:
            {"control": 0.5, "treatment": 0.5}
            {"A": 0.2, "B": 0.3, "C": 0.5}

    alpha:
        Significance level. Use strict values like 1e-3 or 1e-4 in production.

    min_expected:
        If any expected count < min_expected, a warning is attached.

    strict_variants:
        - True: fail if observed variants differ from target keys.
        - False: drop unknown variants and test only on declared ones.

    Returns
    -------
    SRMResult
    """
    # --- Prepare data
    s = pd.Series(list(assignments)).dropna()
    if s.empty:
        raise ValueError("No assignments provided for SRM check.")

    if not target_allocation:
        raise ValueError("target_allocation cannot be empty.")

    # Validate probabilities
    probs = np.array(list(target_allocation.values()), dtype=float)

    if (probs <= 0).any():
        raise ValueError("All target allocation probabilities must be > 0.")

    total = float(probs.sum())
    if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(
            f"target_allocation probabilities must sum to 1.0, got {total:.6f}."
        )

    variants = list(target_allocation.keys())
    target_map = dict(zip(variants, probs))

    # Observed counts
    if strict_variants:
        unexpected = set(s.unique()) - set(variants)
        if unexpected:
            raise ValueError(
                f"Found assignments to variants not in target_allocation: {unexpected}"
            )

    if not strict_variants:
        s = s[s.isin(variants)]
        if s.empty:
            raise ValueError(
                "After filtering to target variants, no assignments remain."
            )

    observed_counts = s.value_counts().reindex(variants).fillna(0).astype(int)
    n = int(observed_counts.sum())
    if n == 0:
        raise ValueError("Total sample size is zero after preprocessing.")

    # Expected counts from probabilities * n
    expected_counts = np.array(
        [target_map[v] * n for v in variants],
        dtype=float
    )

    # Chi-square statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_components = (observed_counts.values - expected_counts) ** 2 / expected_counts
    chi2_components = np.nan_to_num(chi2_components, nan=0.0, posinf=0.0)
    chi2_stat = float(chi2_components.sum())

    # Degrees of freedom
    k = (expected_counts > 0).sum()
    df = max(int(k) - 1, 1)

    if chi2 is None:
        raise ImportError(
            "scipy is required for p-value computation in check_srm(). "
            f"Original error: {_scipy_import_error}"
        )

    p_value = float(chi2.sf(chi2_stat, df))

    warning = None
    if (expected_counts < min_expected).any():
        warning = (
            f"Some expected cell counts are < {min_expected:.1f}. "
            "Chi-square approximation may be unreliable; "
            "consider exact or simulation-based tests."
        )

    is_srm = p_value < alpha

    return SRMResult(
        chi2=chi2_stat,
        df=df,
        p_value=p_value,
        expected={v: float(e) for v, e in zip(variants, expected_counts)},
        observed={v: int(o) for v, o in zip(variants, observed_counts)},
        alpha=alpha,
        is_srm=is_srm,
        warning=warning,
    )


__all__ = ["SRMResult", "check_srm"]
