"""Normal-approximation RCT planning, with optional historical-control CUPED fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from causalis.data_contracts.rct_causal_data import RctCausalData


@dataclass
class _Design:
    outcome: str
    arms: list[str]
    allocation: np.ndarray
    baseline: float
    variance_raw: float
    variance_used: float
    n_reference: int
    alpha: float
    power: float
    critical: float
    shift: float

    def table(self, sizes: Sequence[int]) -> pd.DataFrame:
        total = sum(int(size) for size in sizes)
        return pd.DataFrame([
            {
                "outcome": self.outcome,
                "treatment": arm,
                "control": self.arms[0],
                "baseline_mean": self.baseline,
                "variance_raw": self.variance_raw,
                "variance_used": self.variance_used,
                "variance_reduction_pct": 100 * (
                    1 - self.variance_used / self.variance_raw
                ),
                "n_reference": self.n_reference,
                "n_control": int(sizes[0]),
                "n_treatment": int(sizes[j]),
                "n_total": total,
                "alpha": self.alpha,
                "power": self.power,
            }
            for j, arm in enumerate(self.arms[1:], start=1)
        ])


def _names(value: Sequence[str], allowed: list[str], label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of column names.")
    names = list(value)
    if any(not isinstance(name, str) for name in names):
        raise ValueError(f"{label} must contain only strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicates.")
    if any(name not in allowed for name in names):
        raise ValueError(f"{label} must be a subset of the declared {label}: {allowed}.")
    return names


def _positive(value: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number.")
    return value


def _normal_power(shift: float | np.ndarray, critical: float) -> float | np.ndarray:
    # Both rejection regions of a two-sided z test, including the smaller tail.
    return norm.sf(critical - shift) + norm.sf(critical + shift)


def _prepare(
    data: RctCausalData,
    covariates: Sequence[str] | None,
    outcome: str,
    allocation: Mapping[str, float] | None,
    alpha: float,
    power: float,
) -> _Design:
    if not isinstance(data, RctCausalData):
        raise TypeError("data must be RctCausalData.")
    if not isinstance(outcome, str) or outcome not in data.outcome_names:
        raise ValueError(f"outcome must name one declared outcome: {data.outcome_names}.")
    x_names = _names([] if covariates is None else covariates, data.confounders_names, "covariates")
    alpha, power = _positive(alpha, "alpha"), _positive(power, "power")
    if not 0 < alpha < power < 1:
        raise ValueError("Require 0 < alpha < power < 1.")
    critical = float(norm.isf(alpha / 2))
    if not np.isfinite(critical):
        raise ValueError("alpha is too small for a finite normal critical value.")
    upper = max(1.0, critical + float(norm.ppf(power)) + 1.0)
    shift = brentq(
        lambda value: (alpha if value == 0 else _normal_power(value, critical)) - power,
        0.0, upper, xtol=np.finfo(float).tiny,
    )

    if data.control_treatment is None:
        treatment = data.treatment_names[0]
        arms = [f"{treatment}=0", f"{treatment}=1"]
        control_mask = data.df[treatment].to_numpy() == 0
    else:
        arms = [data.control_treatment] + [
            name for name in data.treatment_names if name != data.control_treatment
        ]
        control_mask = data.df[data.control_treatment].to_numpy() == 1
    if allocation is None:
        shares = np.full(len(arms), 1.0 / len(arms))
    else:
        if not isinstance(allocation, Mapping) or set(allocation) != set(arms):
            raise ValueError(f"allocation must map exactly these groups to shares: {arms}.")
        shares = np.array([_positive(allocation[arm], f"allocation[{arm!r}]") for arm in arms])
        if not np.isclose(shares.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("allocation shares must sum to 1.")
        shares /= shares.sum()

    n = int(control_mask.sum())
    if n < 2:
        raise ValueError("Historical control requires at least two observations.")
    # Select only control rows. Neither active arms nor other outcomes enter planning.
    frame = data.df.loc[control_mask, x_names + [outcome]]
    y = frame[outcome].to_numpy(dtype=float)
    baseline = float(y.mean())
    centered_y = y - baseline
    if not np.isfinite(centered_y).all():
        raise ValueError("Historical control values are too large for finite variance estimation.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        total_ss = float(centered_y @ centered_y)
        variance_raw = total_ss / (n - 1)
    if not np.isfinite(variance_raw) or variance_raw <= 0:
        raise ValueError(f"Historical control has zero or non-finite variance for: {outcome}.")

    variance_used = variance_raw
    if x_names:
        x = frame[x_names].to_numpy(dtype=float)
        constant = np.all(x == x[0], axis=0)
        if constant.any():
            dropped = [name for name, drop in zip(x_names, constant) if drop]
            warnings.warn(
                f"Dropped covariates constant in historical control: {dropped}.",
                UserWarning, stacklevel=3,
            )
            x = x[:, ~constant]
        # If every covariate was constant, fall back to the unadjusted variance.
        if x.shape[1]:
            x = x - x.mean(axis=0)
            if not np.isfinite(x).all():
                raise ValueError("Historical control values are too large for a finite regression.")
            x /= np.max(np.abs(x), axis=0)
            matrix = np.column_stack([np.ones(n), x])
            coefficients, _, rank, _ = np.linalg.lstsq(matrix, centered_y, rcond=None)
            if rank != matrix.shape[1]:
                raise ValueError("Historical control design matrix is rank deficient.")
            if n <= rank:
                raise ValueError("Historical control requires positive residual degrees of freedom.")
            residuals = centered_y - matrix @ coefficients
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                residual_ss = float(residuals @ residuals)
                variance_used = residual_ss / (n - rank)
                # Exact linear fits leave floating-point residue; do not advertise zero MDEs.
                zero_tolerance = (np.finfo(float).eps * max(matrix.shape)) ** 2 * total_ss
            if not np.isfinite(variance_used) or variance_used <= 0 or residual_ss <= zero_tolerance:
                raise ValueError(f"Historical control has zero or non-finite residual variance for: {outcome}.")
    return _Design(
        outcome, arms, shares, baseline, variance_raw, variance_used,
        n, alpha, power, critical, shift,
    )


def _standard_error(table: pd.DataFrame) -> np.ndarray:
    return np.sqrt(table.variance_used.to_numpy()) * np.sqrt(
        1.0 / table.n_control.to_numpy() + 1.0 / table.n_treatment.to_numpy()
    )


def _relative(absolute: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    result = np.full_like(absolute, np.nan, dtype=float)
    np.divide(absolute, baseline, out=result, where=baseline > 0)
    return result * 100


def calculate_sample_size(
    data: RctCausalData,
    *,
    outcome: str,
    covariates: Sequence[str] | None = None,
    scenario: Literal["default", "user"] = "default",
    mde_type: Literal["absolute", "relative"] | None = None,
    mde: Sequence[float] | None = None,
    allocation: Mapping[str, float] | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    include_details: bool = False,
) -> pd.DataFrame:
    """Plan separate experiments for target MDEs of one outcome.

    Parameters
    ----------
    data : RctCausalData
        Past RCT. Only control rows estimate the baseline and residual variance;
        all declared groups define the future experiment.
    outcome : str
        Required name of one declared outcome.
    covariates : Sequence[str], optional
        Explicit pre-treatment confounders for CUPED. Omitted, None, or [] uses
        unadjusted sample variance, even if data declares confounders.
    scenario : {"default", "user"}
        Default plans relative targets [0.5, 1, 5, 10, 20] percent and rejects
        custom mde or mde_type. User requires both arguments explicitly.
    mde_type : {"relative", "absolute"}, optional
        In user mode, percentages (1 means 1%) or differences in outcome units.
        Relative targets require a positive historical control mean.
    mde : Sequence[float], optional
        In user mode, non-empty finite positive targets. Order and duplicates
        are preserved. Each target defines a separate candidate experiment.
    allocation : Mapping[str, float], optional
        Positive shares summing to one for every group; defaults to equal shares.
        Keys are one-hot column names or '<treatment>=0' / '<treatment>=1' for
        binary treatment. Historical allocation is never reused implicitly.
    alpha, power : float
        Per-comparison significance and target power; require 0 < alpha < power < 1.
    include_details : bool
        Append historical fit diagnostics and achieved power. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        One row per target and active arm, in input target and contract arm order.
        Compact columns: outcome, control, treatment, mde_relative, mde_absolute,
        n_control, n_treatment, n_total. Both MDE columns describe the requested
        target; relative values are percentages, or NaN for nonpositive baselines
        in absolute mode. Sizes are integers and other values are not rounded.
        Details append baseline_mean, variance_raw, variance_used,
        variance_reduction_pct, n_reference, alpha, power, achieved_power.
        Within each target, all rows share one experiment sized to meet every
        arm's target power. n_total counts the shared control once; do not sum rows.

    Notes
    -----
    Estimates historical-control variance once for all targets. Without covariates,
    uses sample variance (ddof=1) without regression. With covariates, uses CUPED
    residual variance SSE / (n_reference - design rank). The two-sided normal
    approximation assumes independent randomization units and the same variance in
    all future groups. Binary outcomes also use this fixed-variance approximation.
    This is a planning estimate, not an exact
    finite-sample power guarantee for CUPEDModel. There is no multiple-testing
    correction or joint-power calculation. Each group's continuous size is rounded
    up; this is conservative, not a search for the smallest integer allocation.
    Control-constant covariates are dropped with a warning; singular designs,
    non-positive residual degrees of freedom, and zero residual variance raise
    ValueError. Negative estimated variance reduction is retained.
    """
    if not isinstance(data, RctCausalData):
        raise TypeError("data must be RctCausalData.")
    if not isinstance(outcome, str) or outcome not in data.outcome_names:
        raise ValueError(f"outcome must name one declared outcome: {data.outcome_names}.")
    if not isinstance(include_details, (bool, np.bool_)):
        raise ValueError("include_details must be a boolean.")
    if scenario == "default":
        if mde is not None or mde_type is not None:
            raise ValueError("scenario='default' does not accept mde or mde_type.")
        targets = np.array([0.5, 1, 5, 10, 20])
        mde_type = "relative"
    elif scenario == "user":
        if mde_type not in ("relative", "absolute"):
            raise ValueError("scenario='user' requires mde_type='relative' or 'absolute'.")
        if isinstance(mde, (str, bytes)) or not isinstance(mde, Sequence) or not len(mde):
            raise ValueError("mde must be a non-empty sequence of finite positive targets.")
        targets = np.array([_positive(value, f"mde[{i}]") for i, value in enumerate(mde)])
    else:
        raise ValueError("scenario must be 'default' or 'user'.")
    design = _prepare(data, covariates, outcome, allocation, alpha, power)
    if mde_type == "relative":
        if design.baseline <= 0:
            raise ValueError("Relative targets require a positive historical control mean.")
        with np.errstate(over="ignore", under="ignore"):
            targets = (targets / 100) * design.baseline
        if not np.isfinite(targets).all() or (targets <= 0).any():
            raise ValueError("Relative targets must convert to finite positive absolute effects.")
    table = pd.concat(
        [_sample_size_table(design, float(target)) for target in targets], ignore_index=True,
    )
    return _result(table, include_details)


def calculate_mde(
    data: RctCausalData,
    *,
    outcome: str,
    sample_size: int,
    covariates: Sequence[str] | None = None,
    allocation: Mapping[str, float] | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    include_details: bool = False,
) -> pd.DataFrame:
    """Calculate absolute and relative MDE for a fixed total future sample size.

    Parameters
    ----------
    data : RctCausalData
        Historical RCT. Only control rows estimate the baseline and variance.
    outcome : str
        Required name of one declared outcome.
    sample_size : int
        Positive integer total across all future groups, at most 2**53.
        Allocate by largest remainders; ties follow contract order, control first.
        Every group must receive at least one participant.
    covariates : Sequence[str], optional
        Explicit pre-treatment confounders for CUPED. Omitted, None, or [] uses
        unadjusted sample variance, even if data declares confounders.
    allocation : Mapping[str, float], optional
        Positive shares summing to one for every declared group; defaults to equal
        shares. Keys are one-hot names or '<treatment>=0' / '<treatment>=1' for
        binary treatment. Historical allocation is never reused implicitly.
    alpha, power : float
        Per-comparison significance and target power; require 0 < alpha < power < 1.
    include_details : bool
        Append historical fit and design diagnostics. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        One row per active arm in contract order. Compact columns: outcome,
        control, treatment, mde_relative, mde_absolute, n_control, n_treatment,
        n_total. Absolute MDE uses outcome units; relative MDE is a percentage
        (1 means 1%) of the raw historical control mean, or NaN if it is <= 0.
        Sizes are integers and other values are not rounded. Details append
        baseline_mean, variance_raw, variance_used, variance_reduction_pct,
        n_reference, alpha, power. All rows describe the same experiment;
        n_total counts shared control once and must not be summed across rows.

    Notes
    -----
    Uses the same deterministic, two-sided normal power calculation and historical
    variance assumptions as calculate_sample_size. With no covariates, classic
    planning uses sample variance (ddof=1) without regression. With covariates,
    CUPED uses residual variance SSE / (n_reference - design rank). Future arms
    share that variance, including for binary outcomes. Power is per comparison,
    without multiple-testing correction or a joint detection guarantee. Constant
    control covariates are dropped with a warning; invalid fits raise ValueError.
    """
    if not isinstance(include_details, (bool, np.bool_)):
        raise ValueError("include_details must be a boolean.")
    if isinstance(sample_size, (bool, np.bool_)) or not isinstance(sample_size, Integral) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    if sample_size > 2**53:
        raise ValueError("sample_size is too large for reliable floating-point allocation.")
    design = _prepare(data, covariates, outcome, allocation, alpha, power)
    quotas = int(sample_size) * design.allocation
    sizes = np.floor(quotas).astype(np.int64)
    remaining = int(sample_size) - sum(int(size) for size in sizes)
    if not 0 <= remaining < len(sizes):
        raise ValueError("sample_size is too large for reliable floating-point allocation.")
    priority = np.argsort(-(quotas - sizes), kind="stable")
    sizes[priority[:remaining]] += 1
    if (sizes <= 0).any():
        raise ValueError("sample_size and allocation must give every group at least one participant.")
    table = design.table(sizes)
    table["mde_absolute"] = design.shift * _standard_error(table)
    table["mde_relative"] = _relative(table.mde_absolute.to_numpy(), table.baseline_mean.to_numpy())
    return _result(table, include_details)


def _sample_size_table(design: _Design, target: float) -> pd.DataFrame:
    """Size one shared experiment for an absolute target on a prepared design."""
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        factor = (design.shift * np.sqrt(design.variance_used) / target) ** 2
        requirements = factor * (1 / design.allocation[0] + 1 / design.allocation[1:])
    total = float(np.max(requirements))
    if not np.isfinite(total) or total > 2**53:
        raise ValueError("Required sample size is too large for reliable floating-point allocation.")
    sizes = np.maximum(1, np.ceil(total * design.allocation)).astype(np.int64)
    table = design.table(sizes)
    table["mde_absolute"] = target
    table["mde_relative"] = _relative(table.mde_absolute.to_numpy(), table.baseline_mean.to_numpy())
    with np.errstate(over="ignore"):
        table["achieved_power"] = _normal_power(target / _standard_error(table), design.critical)
    return table


def _result(table: pd.DataFrame, include_details: bool) -> pd.DataFrame:
    columns = [
        "outcome", "control", "treatment", "mde_relative", "mde_absolute",
        "n_control", "n_treatment", "n_total",
    ]
    if include_details:
        columns += [
            "baseline_mean", "variance_raw", "variance_used", "variance_reduction_pct",
            "n_reference", "alpha", "power",
        ]
        if "achieved_power" in table:
            columns.append("achieved_power")
    return table[columns]
