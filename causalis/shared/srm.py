"""
Sample Ratio Mismatch (SRM) utilities for randomized experiments.

This module provides a chi-square goodness-of-fit SRM check for randomized
experiments. It accepts observed assignments as labels or aggregated counts
and returns a compact result object with diagnostics.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers
from typing import Dict, Hashable, Iterable, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from causalis.dgp.causaldata import CausalData

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

    Attributes
    ----------
    chi2 : float
        The calculated chi-square statistic.
    p_value : float
        The p-value of the test, rounded to 5 decimals.
    expected : dict[Hashable, float]
        Expected counts for each variant.
    observed : dict[Hashable, int]
        Observed counts for each variant.
    alpha : float
        Significance level used for the check.
    is_srm : bool
        True if an SRM was detected (chi-square p-value < alpha), False otherwise.
    warning : str or None
        Warning message if the test assumptions might be violated (e.g., small expected counts).
    """
    chi2: float
    p_value: float
    expected: Dict[Hashable, float]
    observed: Dict[Hashable, int]
    alpha: float
    is_srm: bool
    warning: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - repr formatting
        status = "SRM DETECTED" if self.is_srm else "no SRM"
        return (
            f"SRMResult(status={status}, p_value={self.p_value:.5f}, "
            f"chi2={self.chi2:.4f})"
        )


def check_srm(
    assignments: Union[Iterable[Hashable], pd.Series, CausalData, Mapping[Hashable, Number]],
    target_allocation: Dict[Hashable, Number],
    alpha: float = 1e-3,
    min_expected: float = 5.0,
    strict_variants: bool = True,
) -> SRMResult:
    """
    Check Sample Ratio Mismatch (SRM) for an RCT via a chi-square goodness-of-fit test.

    Parameters
    ----------
    assignments : Iterable[Hashable] or pandas.Series or CausalData or Mapping[Hashable, Number]
        Observed variant assignments. If iterable or Series, elements are labels per
        unit (user_id, session_id, etc.). If CausalData is provided, the treatment
        column is used. If a mapping is provided, it is treated as
        ``{variant: observed_count}`` with non-negative integer counts.
    target_allocation : dict[Hashable, Number]
        Mapping ``{variant: p}`` describing intended allocation as probabilities.
    alpha : float, default 1e-3
        Significance level. Use strict values like 1e-3 or 1e-4 in production.
    min_expected : float, default 5.0
        If any expected count < min_expected, a warning is attached.
    strict_variants : bool, default True
        - True: fail if observed variants differ from target keys.
        - False: drop unknown variants and test only on declared ones.

    Returns
    -------
    SRMResult
        The result of the SRM check.

    Raises
    ------
    ValueError
        If inputs are invalid or empty.
    ImportError
        If scipy is required but not installed.

    Notes
    -----
    - Target allocation probabilities must sum to 1 within numerical tolerance.
    - ``is_srm`` is computed using the unrounded p-value; the returned
      ``p_value`` is rounded to 5 decimals.
    - Missing assignments are dropped and reported via ``warning``.
    - Requires SciPy for p-value computation.

    Examples
    --------
    >>> assignments = ["control"] * 50 + ["treatment"] * 50
    >>> check_srm(assignments, {"control": 0.5, "treatment": 0.5}, alpha=1e-3)
    SRMResult(status=no SRM, p_value=1.00000, chi2=0.0000)

    >>> counts = {"control": 70, "treatment": 30}
    >>> check_srm(counts, {"control": 0.5, "treatment": 0.5})
    SRMResult(status=SRM DETECTED, p_value=0.00006, chi2=16.0000)
    """
    # --- Prepare data
    warning_messages: list[str] = []

    def is_finite_number(value: object) -> bool:
        if hasattr(value, "is_finite"):
            return bool(value.is_finite())
        try:
            return math.isfinite(value)  # type: ignore[arg-type]
        except TypeError:
            return False

    if isinstance(assignments, Mapping) and not isinstance(assignments, pd.Series):
        validated_counts: dict[Hashable, int] = {}
        for variant, value in assignments.items():
            if isinstance(value, bool):
                raise ValueError("Assignment counts must be integers.")
            if isinstance(value, numbers.Integral):
                count = int(value)
            elif isinstance(value, numbers.Real):
                if not is_finite_number(value):
                    raise ValueError("All assignment counts must be finite numbers.")
                if value != int(value):
                    raise ValueError("Assignment counts must be integers.")
                count = int(value)
            else:
                raise ValueError("Assignment counts must be numeric.")
            if count < 0:
                raise ValueError("Assignment counts must be >= 0.")
            validated_counts[variant] = count

        observed_counts = pd.Series(validated_counts)
        if observed_counts.empty:
            raise ValueError("No assignments provided for SRM check.")
        s = None
    else:
        if hasattr(assignments, "treatment"):
            s_raw = assignments.treatment
        elif isinstance(assignments, pd.Series):
            s_raw = assignments
        else:
            s_raw = pd.Series(list(assignments))

        missing_count = int(s_raw.isna().sum())
        s = s_raw.dropna()

        if s.empty:
            raise ValueError("No assignments provided for SRM check.")
        if missing_count:
            warning_messages.append(
                f"{missing_count} assignments were missing and were dropped."
            )

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    if min_expected <= 0:
        raise ValueError("min_expected must be > 0.")

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
        if s is None:
            unexpected = set(observed_counts.index) - set(variants)
        else:
            unexpected = set(s.unique()) - set(variants)
        if unexpected:
            raise ValueError(
                f"Found assignments to variants not in target_allocation: {unexpected}"
            )

    if not strict_variants:
        if s is None:
            observed_counts = observed_counts[observed_counts.index.isin(variants)]
            if observed_counts.sum() == 0:
                raise ValueError(
                    "After filtering to target variants, no assignments remain."
                )
        else:
            s = s[s.isin(variants)]
            if s.empty:
                raise ValueError(
                    "After filtering to target variants, no assignments remain."
                )

    if s is None:
        observed_counts = observed_counts.reindex(variants).fillna(0).astype(int)
    else:
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
    k = int((expected_counts > 0).sum())
    if k < 2:
        raise ValueError("SRM check requires at least two variants.")
    df = k - 1

    if chi2 is None:
        raise ImportError(
            "scipy is required for p-value computation in check_srm(). "
            f"Original error: {_scipy_import_error}"
        )

    p_value_raw = float(chi2.sf(chi2_stat, df))
    p_value = round(p_value_raw, 5)

    warning = None
    if (expected_counts < min_expected).any():
        warning_messages.append(
            f"Some expected cell counts are < {min_expected:.1f}. "
            "Chi-square approximation may be unreliable; "
            "consider exact or simulation-based tests."
        )

    if warning_messages:
        warning = " ".join(warning_messages)

    is_srm = p_value_raw < alpha

    return SRMResult(
        chi2=chi2_stat,
        p_value=p_value,
        expected={v: float(e) for v, e in zip(variants, expected_counts)},
        observed={v: int(o) for v, o in zip(variants, observed_counts)},
        alpha=alpha,
        is_srm=is_srm,
        warning=warning,
    )


__all__ = ["SRMResult", "check_srm"]
