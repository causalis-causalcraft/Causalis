"""
Uncofoundedness validation module
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings








def validate_uncofoundedness_balance(
    effect_estimation: Dict[str, Any] | Any,
    *,
    threshold: float = 0.1,
    normalize: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Assess covariate balance under the uncofoundedness assumption by computing
    standardized mean differences (SMD) both before weighting (raw groups) and
    after weighting using the IPW / ATT weights implied by the DML/IRM estimation.

    This function expects the result dictionary or CausalEstimate returned by dml_ate() or dml_att(),
    which includes a fitted IRM model and a 'diagnostic_data' entry with the
    necessary arrays.

    We compute, for each confounder X_j:
      - For ATE (weighted): w1 = D/m_hat, w0 = (1-D)/(1-m_hat).
      - For ATTE (weighted): Treated weight = 1 for D=1; Control weight w0 = m_hat/(1-m_hat) for D=0.
      - If estimation used normalized IPW (normalize_ipw=True), we scale the corresponding
        weights by their sample mean (as done in IRM) before computing balance.

    The SMD is defined as |mu1 - mu0| / s_pooled, where mu_g are (weighted) means in the
    (pseudo-)populations and s_pooled is the square root of the average of the (weighted)
    variances in the two groups.

    Parameters
    ----------
    effect_estimation : Dict[str, Any] | Any
        Output from dml_ate() or dml_att(). Must contain 'diagnostic_data'.
    threshold : float, default 0.1
        Threshold for SMD; values below indicate acceptable balance for most use cases.
    normalize : Optional[bool]
        Whether to use normalized weights. If None, inferred from diagnostic_data.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys:
        - 'smd': pd.Series of weighted SMD values indexed by confounder names
        - 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
        - 'score': 'ATE' or 'ATTE'
        - 'normalized': bool used for weighting
        - 'threshold': float
        - 'pass': bool indicating whether all weighted SMDs are below threshold
    """
    # 1) Handle CausalEstimate / Pydantic
    if hasattr(effect_estimation, "diagnostic_data") and effect_estimation.diagnostic_data is not None:
        diag = effect_estimation.diagnostic_data
    # 2) Handle dict
    elif isinstance(effect_estimation, dict):
        if 'diagnostic_data' not in effect_estimation:
            raise ValueError("Input must contain 'diagnostic_data' (from IRM.estimate())")
        diag = effect_estimation['diagnostic_data']
    else:
        raise TypeError("effect_estimation must be a dictionary or CausalEstimate produced by IRM.estimate()")

    # Helper to get values from dict or object
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Required arrays
    try:
        m_hat = np.asarray(_get(diag, 'm_hat'), dtype=float)
        d = np.asarray(_get(diag, 'd'), dtype=float)
        X = np.asarray(_get(diag, 'x'), dtype=float)
        
        # Infer score
        score = _get(diag, 'score', None)
        if score is None:
             score = _get(effect_estimation, 'estimand', 'ATE')
        score = str(score).upper()

        # Infer normalization
        used_norm = _get(diag, 'normalize_ipw', None)
        if used_norm is None:
            opts = _get(effect_estimation, 'model_options', {})
            used_norm = _get(opts, 'normalize_ipw', False)
        used_norm = bool(used_norm) if normalize is None else bool(normalize)
        
    except Exception as e:
        raise ValueError(f"diagnostic_data missing or incompatible: {e}")

    if score not in {"ATE", "ATTE", "ATT"}:
        raise ValueError(f"Score {score} not supported for balance.")
    if "ATT" in score:
        score = "ATTE"
        
    if X.ndim != 2:
        raise ValueError("diagnostic_data['x'] must be a 2D array of shape (n, p)")
    n, p = X.shape
    if m_hat.shape[0] != n or d.shape[0] != n:
        raise ValueError("Length of m_hat and d must match number of rows in x")

    # Obtain confounder names if available
    names = None
    try:
        # Try to find model and its data_contracts
        model = _get(effect_estimation, 'model', None)
        if model is not None:
            names = list(getattr(model.data, 'confounders', []))
        if not names:
            # Maybe it's in diagnostic_data (not common but possible)
            names = _get(diag, 'x_names', None)
    except Exception:
        pass
        
    if not names or len(names) != p:
        names = [f"x{j+1}" for j in range(p)]

    # Build weights per group according to estimand
    eps = 1e-12
    m = np.clip(m_hat, eps, 1.0 - eps)

    if score == "ATE":
        w1 = d / m
        w0 = (1.0 - d) / (1.0 - m)
        if used_norm:
            w1_mean = float(np.mean(w1))
            w0_mean = float(np.mean(w0))
            w1 = w1 / (w1_mean if w1_mean != 0 else 1.0)
            w0 = w0 / (w0_mean if w0_mean != 0 else 1.0)
    else:  # ATTE
        w1 = d  # treated weight = 1 for treated, 0 otherwise
        w0 = (1.0 - d) * (m / (1.0 - m))
        if used_norm:
            w1_mean = float(np.mean(w1))
            w0_mean = float(np.mean(w0))
            w1 = w1 / (w1_mean if w1_mean != 0 else 1.0)
            w0 = w0 / (w0_mean if w0_mean != 0 else 1.0)

    # Guard against no-mass groups
    s1 = float(np.sum(w1))
    s0 = float(np.sum(w0))
    if s1 <= 0 or s0 <= 0:
        raise RuntimeError("Degenerate weights: zero total mass in a pseudo-population")

    # Weighted means and variances (population-style)
    WX1 = (w1[:, None] * X)
    WX0 = (w0[:, None] * X)
    mu1 = WX1.sum(axis=0) / s1
    mu0 = WX0.sum(axis=0) / s0

    # Weighted variance: E[(X-mu)^2] under weight distribution
    var1 = (w1[:, None] * (X - mu1) ** 2).sum(axis=0) / s1
    var0 = (w0[:, None] * (X - mu0) ** 2).sum(axis=0) / s0
    # Pooled standard deviation directly from variances for clarity
    s_pooled = np.sqrt(0.5 * (np.maximum(var1, 0.0) + np.maximum(var0, 0.0)))

    # SMD with explicit zero-variance handling (weighted)
    smd = np.full(p, np.nan, dtype=float)
    zero_both = (var1 <= 1e-16) & (var0 <= 1e-16)
    diff = np.abs(mu1 - mu0)
    mask = (~zero_both) & (s_pooled > 1e-16)
    smd[mask] = diff[mask] / s_pooled[mask]
    smd[zero_both & (diff <= 1e-16)] = 0.0
    smd[zero_both & (diff > 1e-16)] = np.inf
    smd_s = pd.Series(smd, index=names, dtype=float)

    # --- Unweighted (pre-weighting) SMD using raw treated/control groups ---
    mask1 = d.astype(bool)
    mask0 = ~mask1
    # If any group is empty (shouldn't be in typical settings), guard with nan SMDs
    if not np.any(mask1) or not np.any(mask0):
        smd_unw = np.full(p, np.nan, dtype=float)
    else:
        X1 = X[mask1]
        X0 = X[mask0]
        mu1_u = X1.mean(axis=0)
        mu0_u = X0.mean(axis=0)
        # population-style std to mirror weighted computation
        sd1_u = X1.std(axis=0, ddof=0)
        sd0_u = X0.std(axis=0, ddof=0)
        var1_u = sd1_u ** 2
        var0_u = sd0_u ** 2
        s_pool_u = np.sqrt(0.5 * (np.maximum(var1_u, 0.0) + np.maximum(var0_u, 0.0)))
        smd_unw = np.full(p, np.nan, dtype=float)
        zero_both_u = (var1_u <= 1e-16) & (var0_u <= 1e-16)
        diff_u = np.abs(mu1_u - mu0_u)
        mask_u = (~zero_both_u) & (s_pool_u > 1e-16)
        smd_unw[mask_u] = diff_u[mask_u] / s_pool_u[mask_u]
        smd_unw[zero_both_u & (diff_u <= 1e-16)] = 0.0
        smd_unw[zero_both_u & (diff_u > 1e-16)] = np.inf
    smd_unweighted_s = pd.Series(smd_unw, index=names, dtype=float)

    # Extra quick‑report fields
    smd_max = float(np.nanmax(smd)) if smd.size else float('nan')
    worst_features = smd_s.sort_values(ascending=False).head(10)

    # Decide pass/fail: ignore non-finite entries; also require low fraction of violations
    finite_mask = np.isfinite(smd_s.values)
    if np.any(finite_mask):
        frac_viol = float(np.mean(smd_s.values[finite_mask] >= float(threshold)))
        pass_bal = bool(np.all(smd_s.values[finite_mask] < float(threshold)) and (frac_viol < 0.10))
    else:
        # If all SMDs are non-finite (e.g., no variation across all features), treat as pass
        frac_viol = 0.0
        pass_bal = True

    out = {
        'smd': smd_s,
        'smd_unweighted': smd_unweighted_s,
        'score': score,
        'normalized': used_norm,
        'threshold': float(threshold),
        'pass': pass_bal,
        'smd_max': smd_max,
        'worst_features': worst_features,
    }
    return out







# ================= Uncofoundedness diagnostics (balance + overlap + weights) =================
from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Tuple as _Tuple, List as _List
from copy import deepcopy as _deepcopy  # not strictly needed, kept for future extensions


def _grade(val: float, warn: float, strong: float, *, larger_is_worse: bool = True) -> str:
    """Map a scalar to GREEN/YELLOW/RED; NA for nan/inf."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "NA"
    v = float(val)
    if larger_is_worse:
        # smaller values are better
        return "GREEN" if v < warn else ("YELLOW" if v < strong else "RED")
    else:
        # larger values are better
        return "GREEN" if v >= warn else ("YELLOW" if v >= strong else "RED")


def _safe_quantiles(a: np.ndarray, qs=(0.5, 0.9, 0.99)) -> _List[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(a, q)) for q in qs]


def _ks_unweighted(a: np.ndarray, b: np.ndarray) -> float:
    """Simple unweighted KS distance between two 1D arrays."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Evaluate ECDFs on merged grid
    grid = np.r_[a, b]
    grid.sort(kind="mergesort")
    Fa = np.searchsorted(a, grid, side="right") / a.size
    Fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(Fa - Fb)))


def _extract_balance_inputs_from_result(
    res: _Dict[str, _Any] | _Any,
) -> _Tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, _List[str]]:
    """
    Returns X (n,p), m_hat (n,), d (n,), score ('ATE'/'ATTE'), used_norm (bool), names (len p).
    Accepts:
      - CausalEstimate object with .diagnostic_data
      - dict with keys 'model' and 'diagnostic_data' (preferred)
      - model-like object with .data_contracts and cross-fitted m_hat_/predictions
    """
    # 0) CausalEstimate
    if hasattr(res, "diagnostic_data") and res.diagnostic_data is not None:
        dd = res.diagnostic_data
        X = getattr(dd, "x", None)
        m = getattr(dd, "m_hat", None)
        d = getattr(dd, "d", None)
        if X is not None and m is not None and d is not None:
            X = np.asarray(X, dtype=float)
            m = np.asarray(m, dtype=float).ravel()
            d = np.asarray(d, dtype=float).ravel()
            score = str(getattr(res, "estimand", "ATE")).upper()
            used_norm = False
            if hasattr(res, "model_options") and isinstance(res.model_options, dict):
                used_norm = bool(res.model_options.get("normalize_ipw", False))

            names = None
            # Try to get names from model if possible
            model = getattr(res, "model", None)
            if model is not None and hasattr(model, "data_contracts"):
                try:
                    names = list(getattr(model.data_contracts, "confounders", []))
                except Exception:
                    pass
            if not names:
                names = [f"x{j+1}" for j in range(X.shape[1])]
            return X, m, d, ("ATTE" if "ATT" in score else "ATE"), used_norm, names

    # 1) Result dict path
    if isinstance(res, dict):
        diag = res.get("diagnostic_data", {})
        if isinstance(diag, dict) and all(k in diag for k in ("x", "m_hat", "d")):
            X = np.asarray(diag["x"], dtype=float)
            m = np.asarray(diag["m_hat"], dtype=float).ravel()
            d = np.asarray(diag["d"], dtype=float).ravel()
            score = str(diag.get("score", "ATE")).upper()
            used_norm = bool(diag.get("normalize_ipw", False))
            names = diag.get("x_names")
            if not names or len(names) != X.shape[1]:
                # try model data_contracts for names
                model = res.get("model", None)
                if model is not None and getattr(model, "data_contracts", None) is not None:
                    try:
                        names = list(getattr(model.data, "confounders", []))
                    except Exception:
                        names = None
            if not names or len(names) != X.shape[1]:
                names = [f"x{j+1}" for j in range(X.shape[1])]
            return X, m, d, ("ATTE" if "ATT" in score else "ATE"), used_norm, names

        # Fall through to model extraction if diag missing
        res = res.get("model", res)

    # Model-like path
    model = res
    data_obj = getattr(model, "data_contracts", None)
    if data_obj is None or not hasattr(data_obj, "get_df"):
        raise ValueError("Could not extract data_contracts arrays. Provide `res` with diagnostic_data or a model with .data_contracts.get_df().")

    df = data_obj.get_df()
    confs = list(getattr(data_obj, "confounders", [])) or []
    if not confs:
        raise ValueError("CausalData must include confounders to compute balance (X).")
    X = df[confs].to_numpy(dtype=float)
    names = confs

    # m_hat
    if hasattr(model, "m_hat_") and getattr(model, "m_hat_", None) is not None:
        m = np.asarray(model.m_hat_, dtype=float).ravel()
    else:
        preds = getattr(model, "predictions", None)
        if isinstance(preds, dict) and "ml_m" in preds:
            m = np.asarray(preds["ml_m"], dtype=float).ravel()
        else:
            raise AttributeError("Could not locate propensity predictions (m_hat_ or predictions['ml_m']).")

    # d
    tname = getattr(getattr(data_obj, "treatment", None), "name", None) or getattr(data_obj, "_treatment", "D")
    d = df[tname].to_numpy(dtype=float).ravel()

    # score & normalization
    sc = getattr(model, "score", None) or getattr(model, "_score", "ATE")
    score = "ATTE" if "ATT" in str(sc).upper() else "ATE"
    used_norm = bool(getattr(model, "normalize_ipw", False))
    return X, m, d, score, used_norm, names


# ---------------- core SMD routine (adapted) ----------------

def _balance_smd(
    X: np.ndarray,
    d: np.ndarray,
    m_hat: np.ndarray,
    *,
    score: str,
    normalize: bool,
    threshold: float,
) -> _Dict[str, _Any]:
    """
    Compute weighted/unweighted SMDs + quick summaries using ATE/ATTE implied weights.
    """
    n, p = X.shape
    eps = 1e-12
    m = np.clip(np.asarray(m_hat, dtype=float).ravel(), eps, 1.0 - eps)
    d = np.asarray(d, dtype=float).ravel()
    score_u = str(score).upper()

    if score_u == "ATE":
        w1 = d / m
        w0 = (1.0 - d) / (1.0 - m)
    else:  # ATTE
        w1 = d
        w0 = (1.0 - d) * (m / (1.0 - m))

    if normalize:
        w1 /= float(np.mean(w1)) if np.mean(w1) != 0 else 1.0
        w0 /= float(np.mean(w0)) if np.mean(w0) != 0 else 1.0

    s1 = float(np.sum(w1))
    s0 = float(np.sum(w0))
    if s1 <= 0 or s0 <= 0:
        raise RuntimeError("Degenerate weights: zero total mass in a pseudo-population.")

    # weighted means/vars
    mu1 = (w1[:, None] * X).sum(axis=0) / s1
    mu0 = (w0[:, None] * X).sum(axis=0) / s0
    var1 = (w1[:, None] * (X - mu1) ** 2).sum(axis=0) / s1
    var0 = (w0[:, None] * (X - mu0) ** 2).sum(axis=0) / s0
    s_pool = np.sqrt(0.5 * (np.maximum(var1, 0.0) + np.maximum(var0, 0.0)))
    smd_w = np.full(p, np.nan, dtype=float)
    zero_both = (var1 <= 1e-16) & (var0 <= 1e-16)
    diff = np.abs(mu1 - mu0)
    mask = (~zero_both) & (s_pool > 1e-16)
    smd_w[mask] = diff[mask] / s_pool[mask]
    smd_w[zero_both & (diff <= 1e-16)] = 0.0
    smd_w[zero_both & (diff > 1e-16)] = np.inf

    # raw SMD
    mask1 = d.astype(bool)
    mask0 = ~mask1
    if not np.any(mask1) or not np.any(mask0):
        smd_u = np.full(p, np.nan)
    else:
        mu1_u = X[mask1].mean(axis=0)
        mu0_u = X[mask0].mean(axis=0)
        sd1_u = X[mask1].std(axis=0, ddof=0)
        sd0_u = X[mask0].std(axis=0, ddof=0)
        var1_u = sd1_u ** 2
        var0_u = sd0_u ** 2
        s_pool_u = np.sqrt(0.5 * (np.maximum(var1_u, 0.0) + np.maximum(var0_u, 0.0)))
        smd_u = np.full(p, np.nan)
        zero_both_u = (var1_u <= 1e-16) & (var0_u <= 1e-16)
        diff_u = np.abs(mu1_u - mu0_u)
        mask_u = (~zero_both_u) & (s_pool_u > 1e-16)
        smd_u[mask_u] = diff_u[mask_u] / s_pool_u[mask_u]
        smd_u[zero_both_u & (diff_u <= 1e-16)] = 0.0
        smd_u[zero_both_u & (diff_u > 1e-16)] = np.inf

    # summaries
    finite = np.isfinite(smd_w)
    smd_max = float(np.nanmax(smd_w)) if np.any(finite) else float("nan")
    frac_viol = float(np.mean(smd_w[finite] >= float(threshold))) if np.any(finite) else 0.0

    return {
        "smd_weighted": smd_w,
        "smd_unweighted": smd_u,
        "smd_max": smd_max,
        "frac_violations": frac_viol,
        "weights": (w1, w0),
        "mass": (s1, s0),
    }


# ---------------- main entry point ----------------

def run_uncofoundedness_diagnostics(
    *,
    res: _Dict[str, _Any] | _Any = None,
    X: _Optional[np.ndarray] = None,
    d: _Optional[np.ndarray] = None,
    m_hat: _Optional[np.ndarray] = None,
    names: _Optional[_List[str]] = None,
    score: _Optional[str] = None,
    normalize: _Optional[bool] = None,
    threshold: float = 0.10,
    eps_overlap: float = 0.01,
    return_summary: bool = True,
) -> _Dict[str, _Any]:
    """
    Uncofoundedness diagnostics focused on balance (SMD).

    Inputs:
      - Either a result/model via `res`, or raw arrays X, d, m_hat (+ optional names, score, normalize).

    Returns a dictionary:
      {
        "params": {"score", "normalize", "smd_threshold"},
        "balance": {"smd", "smd_unweighted", "smd_max", "frac_violations", "pass", "worst_features"},
        "flags": {"balance_max_smd", "balance_violations"},
        "overall_flag": max severity across balance flags,
        "summary": pd.DataFrame with balance rows only
      }
    """
    # ---- Resolve inputs ----
    if (X is None or d is None or m_hat is None) and res is None:
        raise ValueError("Pass either (X, d, m_hat) or `res` with diagnostic_data/model.")

    if X is None or d is None or m_hat is None:
        X, m_hat, d, score_auto, used_norm_auto, names_auto = _extract_balance_inputs_from_result(res)
        if score is None:
            score = score_auto
        if normalize is None:
            normalize = used_norm_auto
        if names is None:
            names = names_auto
    else:
        if score is None:
            score = "ATE"
        if normalize is None:
            normalize = False
        if names is None:
            names = [f"x{j+1}" for j in range(X.shape[1])]

    score_u = str(score).upper()
    used_norm = bool(normalize)
    X = np.asarray(X, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    m_hat = np.asarray(m_hat, dtype=float).ravel()
    n, p = X.shape
    if m_hat.size != n or d.size != n:
        raise ValueError("X, d, m_hat must have matching length n.")

    # ---- Balance (SMD only) ----
    bal = _balance_smd(X, d, m_hat, score=score_u, normalize=used_norm, threshold=threshold)

    smd_w = pd.Series(bal["smd_weighted"], index=names, dtype=float, name="SMD_weighted")
    smd_u = pd.Series(bal["smd_unweighted"], index=names, dtype=float, name="SMD_unweighted")
    worst = smd_w.sort_values(ascending=False).head(10)

    frac_viol = float(bal["frac_violations"]) if np.isfinite(bal["frac_violations"]) else 0.0
    pass_bal = bool(np.all(smd_w[np.isfinite(smd_w)] < float(threshold)) and (frac_viol < 0.10)) if np.any(np.isfinite(smd_w)) else True

    balance_block = {
        "smd": smd_w,
        "smd_unweighted": smd_u,
        "smd_max": float(bal["smd_max"]),
        "frac_violations": frac_viol,
        "pass": pass_bal,
        "worst_features": worst,
    }


    # ---- Balance-only flags & thresholds ----
    def _grade_balance(val: float, warn: float, strong: float) -> str:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "NA"
        v = float(val)
        return "GREEN" if v < warn else ("YELLOW" if v < strong else "RED")

    smd_warn = float(threshold)
    smd_strong = float(threshold) * 2.0  # conventional 0.10/0.20 if threshold=0.10
    viol_frac_warn = 0.10
    viol_frac_strong = 0.25

    balance_flags = {
        "balance_max_smd": _grade_balance(balance_block["smd_max"], smd_warn, smd_strong),
        "balance_violations": _grade_balance(balance_block["frac_violations"], viol_frac_warn, viol_frac_strong),
    }

    # ---- Overall flag severity (balance only) ----
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2, "NA": -1}
    worst_flag = max((order.get(f, -1) for f in balance_flags.values()), default=-1)
    inv = {v: k for k, v in order.items()}
    overall_flag = inv.get(worst_flag, "NA")

    # ---- Params ----
    params = {
        "score": score_u,
        "normalize": used_norm,
        "smd_threshold": float(threshold),
    }

    report: _Dict[str, _Any] = {
        "params": params,
        "balance": balance_block,
        "flags": balance_flags,
        "overall_flag": overall_flag,
        "meta": {"n": int(n), "p": int(p)},
    }

    # ---- Optional summary table (balance-only) ----
    if return_summary:
        bal_rows = [
            {"metric": "balance_max_smd", "value": balance_block["smd_max"], "flag": balance_flags["balance_max_smd"]},
            {"metric": "balance_frac_violations", "value": balance_block["frac_violations"], "flag": balance_flags["balance_violations"]},
        ]
        report["summary"] = pd.DataFrame(bal_rows)

    return report

