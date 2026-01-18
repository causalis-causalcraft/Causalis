from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

CorrMethod = Literal["pearson", "spearman"]
Transform = Literal["none", "log1p", "rank"]

def _apply_transform(x: np.ndarray, transform: Transform) -> np.ndarray:
    x = np.asarray(x, float)
    if transform == "none":
        return x
    if transform == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    if transform == "rank":
        return pd.Series(x).rank(method="average").to_numpy(dtype=float)
    raise ValueError("transform must be one of: 'none','log1p','rank'.")

def _winsorize(x: np.ndarray, q: Optional[float]) -> np.ndarray:
    if q is None:
        return x
    q = float(q)
    if not (0.5 < q <= 1.0):
        raise ValueError("winsor_q must be in (0.5, 1.0].")
    hi = np.quantile(x, q)
    lo = np.quantile(x, 1.0 - q)
    return np.clip(x, lo, hi)

def _corr(a: np.ndarray, b: np.ndarray, method: CorrMethod) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
        rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
        sa = np.std(ra)
        sb = np.std(rb)
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])
    raise ValueError("method must be 'pearson' or 'spearman'.")

def corr_on_scale(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    *,
    transform: Transform = "log1p",
    winsor_q: Optional[float] = 0.999,
    method: CorrMethod = "pearson",
) -> float:
    a = _apply_transform(y_pre, transform)
    b = _apply_transform(y_post, transform)
    a = _winsorize(a, winsor_q)
    b = _winsorize(b, winsor_q)
    return _corr(a, b, method)

@dataclass(frozen=True)
class PreCorrSpec:
    target_corr: float = 0.7
    transform: Transform = "log1p"
    winsor_q: Optional[float] = 0.999
    method: CorrMethod = "pearson"
    # noise calibration
    sigma_lo: float = 0.0
    sigma_hi: float = 50.0
    sigma_tol: float = 1e-3
    max_iter: int = 40

def calibrate_sigma_for_target_corr(
    y_pre_base: np.ndarray,
    y_post: np.ndarray,
    rng: np.random.Generator,
    spec: PreCorrSpec,
    *,
    noise: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Find sigma such that Corr(T(y_pre_base + sigma*eps), T(y_post)) ~ target_corr.
    Returns (sigma, achieved_corr).
    """
    target = float(spec.target_corr)
    if not (0.0 <= target < 1.0):
        raise ValueError("target_corr must be in [0,1).")

    y_pre_base = np.asarray(y_pre_base, float)
    y_post = np.asarray(y_post, float)

    if noise is None:
        eps = rng.normal(size=y_pre_base.shape[0])
    else:
        eps = np.asarray(noise, float)
        if eps.shape != y_pre_base.shape:
            raise ValueError("noise must have same shape as y_pre_base.")

    # Corr at sigma=0 is the maximum achievable under this base signal
    c0 = corr_on_scale(y_pre_base, y_post, transform=spec.transform, winsor_q=spec.winsor_q, method=spec.method)
    if target >= c0:
        # cannot increase correlation by adding independent noise; return sigma=0
        return 0.0, c0

    def c(sigma: float) -> float:
        return corr_on_scale(
            y_pre_base + float(sigma) * eps,
            y_post,
            transform=spec.transform,
            winsor_q=spec.winsor_q,
            method=spec.method
        )

    lo, hi = float(spec.sigma_lo), float(spec.sigma_hi)
    clo, chi = c(lo), c(hi)

    # Ensure bracket: need clo > target > chi
    if clo <= target:
        return lo, clo
    if chi >= target:
        # extend hi if needed (rare if hi is large)
        for mult in (2.0, 5.0, 10.0):
            hi2 = hi * mult
            chi2 = c(hi2)
            if chi2 < target:
                hi, chi = hi2, chi2
                break
        else:
            return hi, chi  # best effort

    for _ in range(int(spec.max_iter)):
        mid = 0.5 * (lo + hi)
        cmid = c(mid)
        if abs(cmid - target) <= spec.sigma_tol:
            return mid, cmid
        if cmid > target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, c(mid)

def add_preperiod_covariate(
    df: pd.DataFrame,
    y_col: str,
    d_col: str,
    pre_name: str,
    base_builder: Callable[[pd.DataFrame], np.ndarray],
    spec: PreCorrSpec,
    rng: np.random.Generator,
    mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Standardized utility to add a calibrated pre-period covariate to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    y_col : str
        Name of the outcome column.
    d_col : str
        Name of the treatment column.
    pre_name : str
        Name of the new pre-period covariate column.
    base_builder : callable
        Function df -> y_pre_base (np.ndarray) providing the shared signal.
    spec : PreCorrSpec
        Specification for target correlation and scale.
    rng : np.random.Generator
        Random number generator.
    mask : np.ndarray, optional
        Boolean mask of rows to use for calibration (e.g. control group).
        If None, use control group (d == 0).
    """
    y_post = df[y_col].to_numpy()
    if mask is None:
        mask = (df[d_col].to_numpy() == 0)
    
    y_pre_base = base_builder(df)
    
    # To maintain RNG state invariance (reproducibility) regardless of mask size,
    # we generate noise for the full dataframe first.
    eps = rng.normal(size=len(df))

    sigma, achieved = calibrate_sigma_for_target_corr(
        y_pre_base=y_pre_base[mask],
        y_post=y_post[mask],
        rng=rng,
        spec=spec,
        noise=eps[mask]
    )
    
    df[pre_name] = y_pre_base + sigma * eps
    return df
