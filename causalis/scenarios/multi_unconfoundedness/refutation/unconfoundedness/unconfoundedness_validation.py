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
    Multitreatment version (one-hot d, matrix m_hat) for ATE only.

    Assumes:
      - d: shape (n, K) one-hot treatment indicators (baseline is column 0)
      - m_hat: shape (n, K) propensity scores for each treatment
      - X: shape (n, p) confounders

    Computes pairwise balance for (0 vs k) for k=1..K-1:
      - ATE weights: w_g = D_g / m_g
      - Optional normalization: divide each group's weights by its mean (over all n),
        mirroring the binary IRM normalization pattern.

    Returns SMDs (weighted and unweighted) as DataFrames with:
      - rows = confounders
      - columns = comparisons "0_vs_k"
    """

    # ---- 1) Extract diagnostic_data ----
    if hasattr(effect_estimation, "diagnostic_data") and effect_estimation.diagnostic_data is not None:
        diag = effect_estimation.diagnostic_data
    elif isinstance(effect_estimation, dict):
        if "diagnostic_data" not in effect_estimation:
            raise ValueError("Input must contain 'diagnostic_data' (from IRM.estimate())")
        diag = effect_estimation["diagnostic_data"]
    else:
        raise TypeError("effect_estimation must be a dict or an object with .diagnostic_data")

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ---- 2) Load arrays ----
    try:
        m_hat = np.asarray(_get(diag, "m_hat"), dtype=float)
        d = np.asarray(_get(diag, "d"), dtype=float)
        X = np.asarray(_get(diag, "x"), dtype=float)

        score = _get(diag, "score", None)
        if score is None:
            score = _get(effect_estimation, "estimand", "ATE")
        score = str(score).upper()

        used_norm = _get(diag, "normalize_ipw", None)
        if used_norm is None:
            opts = _get(effect_estimation, "model_options", {})
            used_norm = _get(opts, "normalize_ipw", False)
        used_norm = bool(used_norm) if normalize is None else bool(normalize)

    except Exception as e:
        raise ValueError(f"diagnostic_data missing or incompatible: {e}")

    # ---- 3) Only ATE supported ----
    if score != "ATE":
        raise ValueError(f"Only ATE is supported for multi-treatment balance. Got score={score}.")

    if X.ndim != 2:
        raise ValueError("diagnostic_data['x'] must be a 2D array of shape (n, p)")
    n, p = X.shape

    if d.ndim != 2:
        raise ValueError("For multi-treatment, diagnostic_data['d'] must be 2D one-hot array (n, K)")
    if m_hat.ndim != 2:
        raise ValueError("For multi-treatment, diagnostic_data['m_hat'] must be 2D array (n, K)")

    if d.shape[0] != n or m_hat.shape[0] != n:
        raise ValueError("First dimension of d and m_hat must match number of rows in x")
    if d.shape[1] != m_hat.shape[1]:
        raise ValueError("d and m_hat must have the same number of treatment columns")

    K = d.shape[1]
    if K < 2:
        raise ValueError("Need at least 2 treatments (K>=2) to assess balance")

    # ---- 4) Confounder names ----
    names = None
    try:
        model = _get(effect_estimation, "model", None)
        if model is not None:
            names = list(getattr(model.data, "confounders", []))
        if not names:
            names = _get(diag, "x_names", None)
    except Exception:
        pass
    if not names or len(names) != p:
        names = [f"x{j+1}" for j in range(p)]

    # ---- 5) Treatment names (optional) ----
    t_names = _get(diag, "treatment_names", None) or _get(diag, "d_names", None)
    if not t_names or len(t_names) != K:
        t_names = [str(k) for k in range(K)]

    # ---- helpers ----
    eps = 1e-12
    m = np.clip(m_hat, eps, 1.0 - eps)

    def _wmean_var(X2: np.ndarray, w: np.ndarray):
        """Population-style weighted mean/variance under weight distribution."""
        sw = float(np.sum(w))
        if sw <= 0:
            return None, None, 0.0
        mu = (w[:, None] * X2).sum(axis=0) / sw
        var = (w[:, None] * (X2 - mu) ** 2).sum(axis=0) / sw
        return mu, var, sw

    def _smd_from_moments(mu_a, var_a, mu_b, var_b):
        s_pooled = np.sqrt(0.5 * (np.maximum(var_a, 0.0) + np.maximum(var_b, 0.0)))
        diff = np.abs(mu_a - mu_b)

        out = np.full(p, np.nan, dtype=float)
        zero_both = (var_a <= 1e-16) & (var_b <= 1e-16)
        ok = (~zero_both) & (s_pooled > 1e-16)

        out[ok] = diff[ok] / s_pooled[ok]
        out[zero_both & (diff <= 1e-16)] = 0.0
        out[zero_both & (diff > 1e-16)] = np.inf
        return out

    # ---- 6) Compute pairwise (0 vs k) ----
    smd_cols = []
    smd_mat = []
    smd_unw_mat = []

    for k in range(1, K):
        col = f"{t_names[0]}_vs_{t_names[k]}"
        smd_cols.append(col)

        # ATE weights for group 0 and group k
        w0 = d[:, 0] / m[:, 0]
        wk = d[:, k] / m[:, k]

        if used_norm:
            m0 = float(np.mean(w0))
            mk = float(np.mean(wk))
            w0 = w0 / (m0 if m0 != 0 else 1.0)
            wk = wk / (mk if mk != 0 else 1.0)

        mu0, var0, s0 = _wmean_var(X, w0)
        muk, vark, sk = _wmean_var(X, wk)

        if s0 <= 0 or sk <= 0:
            raise RuntimeError(f"Degenerate weights in comparison {col}: zero total mass")

        smd_k = _smd_from_moments(muk, vark, mu0, var0)
        smd_mat.append(smd_k)

        # Unweighted SMD for raw groups 0 and k
        mask0 = d[:, 0].astype(bool)
        maskk = d[:, k].astype(bool)
        if (not np.any(mask0)) or (not np.any(maskk)):
            smd_unw = np.full(p, np.nan, dtype=float)
        else:
            X0 = X[mask0]
            Xk = X[maskk]
            mu0_u = X0.mean(axis=0)
            muk_u = Xk.mean(axis=0)
            var0_u = X0.var(axis=0, ddof=0)
            vark_u = Xk.var(axis=0, ddof=0)
            smd_unw = _smd_from_moments(muk_u, vark_u, mu0_u, var0_u)

        smd_unw_mat.append(smd_unw)

    smd_df = pd.DataFrame(np.vstack(smd_mat).T, index=names, columns=smd_cols, dtype=float)
    smd_unweighted_df = pd.DataFrame(np.vstack(smd_unw_mat).T, index=names, columns=smd_cols, dtype=float)

    # ---- 7) Pass/fail summary (overall across all pairs/features) ----
    vals = smd_df.to_numpy().ravel()
    finite = np.isfinite(vals)
    if np.any(finite):
        frac_viol = float(np.mean(vals[finite] >= float(threshold)))
        pass_bal = bool(np.all(vals[finite] < float(threshold)) and (frac_viol < 0.10))
        smd_max = float(np.nanmax(vals[finite]))
    else:
        frac_viol = 0.0
        pass_bal = True
        smd_max = float("nan")

    # Worst features: take max SMD across comparisons per feature, then top 10
    worst = smd_df.max(axis=1).sort_values(ascending=False).head(10)

    return {
        "smd": smd_df,
        "smd_unweighted": smd_unweighted_df,
        "score": "ATE",
        "normalized": used_norm,
        "threshold": float(threshold),
        "pass": pass_bal,
        "frac_violations": frac_viol,
        "smd_max": smd_max,
        "worst_features": worst,
        "comparisons": smd_cols,
        "treatment_names": t_names,
        "baseline_treatment": t_names[0],
    }


# ================= Uncofoundedness diagnostics (balance + overlap + weights) =================

from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Tuple as _Tuple, List as _List

import numpy as np
import pandas as pd


def _grade(val: float, warn: float, strong: float, *, larger_is_worse: bool = True) -> str:
    """Map a scalar to GREEN/YELLOW/RED; NA for nan/inf."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "NA"
    v = float(val)
    if larger_is_worse:
        return "GREEN" if v < warn else ("YELLOW" if v < strong else "RED")
    else:
        return "GREEN" if v >= warn else ("YELLOW" if v >= strong else "RED")


def _safe_quantiles(a: np.ndarray, qs=(0.5, 0.9, 0.99)) -> _List[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(a, q)) for q in qs]


def _ks_unweighted(a: np.ndarray, b: np.ndarray) -> float:
    """Simple unweighted KS distance between two 1D arrays."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")

    grid = np.r_[a, b]
    grid.sort(kind="mergesort")
    Fa = np.searchsorted(a, grid, side="right") / a.size
    Fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(Fa - Fb)))


def _extract_balance_inputs_from_result(
    res: _Dict[str, _Any] | _Any,
) -> _Tuple[np.ndarray, np.ndarray, np.ndarray, str, bool, _List[str], _List[str]]:
    """
    Multi-treatment ATE-only extraction.

    Returns:
      X: (n, p)
      m_hat: (n, K) propensity for all treatments
      d: (n, K) one-hot treatment indicators
      score: 'ATE' (only)
      used_norm: bool
      x_names: list[str] length p
      treatment_names: list[str] length K (defaults to ['0', '1', ...])

    Accepts:
      - CausalEstimate-like object with .diagnostic_data
      - dict with key 'diagnostic_data' (preferred)
      - (fallback) model-like object with .data_contracts.get_df() and predictions
        NOTE: fallback may not work in your stack unless you standardize where d/m_hat live.
    """

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # --- Path A: CausalEstimate-like with .diagnostic_data ---
    if hasattr(res, "diagnostic_data") and res.diagnostic_data is not None:
        dd = res.diagnostic_data
        X = _get(dd, "x", None)
        m_hat = _get(dd, "m_hat", None)
        d = _get(dd, "d", None)

        if X is not None and m_hat is not None and d is not None:
            X = np.asarray(X, dtype=float)
            m_hat = np.asarray(m_hat, dtype=float)
            d = np.asarray(d, dtype=float)

            score = str(_get(dd, "score", _get(res, "estimand", "ATE"))).upper()
            if score != "ATE":
                raise ValueError(f"Only ATE is supported for multi-treatment diagnostics. Got score={score}")

            used_norm = _get(dd, "normalize_ipw", None)
            if used_norm is None:
                mo = _get(res, "model_options", {}) or {}
                used_norm = bool(mo.get("normalize_ipw", False))
            used_norm = bool(used_norm)

            # X names
            x_names = None
            try:
                model = _get(res, "model", None)
                if model is not None:
                    # common patterns
                    x_names = list(getattr(getattr(model, "data", None), "confounders", []) or [])
                    if not x_names and hasattr(model, "data_contracts"):
                        x_names = list(getattr(model.data_contracts, "confounders", []) or [])
            except Exception:
                x_names = None
            if not x_names:
                x_names = _get(dd, "x_names", None)
            if not x_names or len(x_names) != X.shape[1]:
                x_names = [f"x{j+1}" for j in range(X.shape[1])]

            # treatment names
            t_names = _get(dd, "treatment_names", None) or _get(dd, "d_names", None)
            K = d.shape[1] if d.ndim == 2 else m_hat.shape[1]
            if not t_names or len(t_names) != K:
                t_names = [str(k) for k in range(K)]

            return X, m_hat, d, "ATE", used_norm, list(x_names), list(t_names)

    # --- Path B: dict with diagnostic_data ---
    if isinstance(res, dict):
        diag = res.get("diagnostic_data", None)
        if isinstance(diag, dict) and all(k in diag for k in ("x", "m_hat", "d")):
            X = np.asarray(diag["x"], dtype=float)
            m_hat = np.asarray(diag["m_hat"], dtype=float)
            d = np.asarray(diag["d"], dtype=float)

            score = str(diag.get("score", "ATE")).upper()
            if score != "ATE":
                raise ValueError(f"Only ATE is supported for multi-treatment diagnostics. Got score={score}")

            used_norm = bool(diag.get("normalize_ipw", False))

            x_names = diag.get("x_names", None)
            if not x_names or len(x_names) != X.shape[1]:
                x_names = [f"x{j+1}" for j in range(X.shape[1])]

            t_names = diag.get("treatment_names", None) or diag.get("d_names", None)
            K = d.shape[1] if d.ndim == 2 else m_hat.shape[1]
            if not t_names or len(t_names) != K:
                t_names = [str(k) for k in range(K)]

            return X, m_hat, d, "ATE", used_norm, list(x_names), list(t_names)

        # fallback to model if dict didn't contain diag
        res = res.get("model", res)

    # --- Path C: model-like fallback (best-effort) ---
    model = res
    data_obj = getattr(model, "data_contracts", None)
    if data_obj is None or not hasattr(data_obj, "get_df"):
        raise ValueError(
            "Could not extract arrays. Provide `res` with diagnostic_data containing x/m_hat/d "
            "(multi-treatment: m_hat and d must be (n,K))."
        )

    df = data_obj.get_df()
    confs = list(getattr(data_obj, "confounders", []) or [])
    if not confs:
        raise ValueError("CausalData must include confounders to compute balance (X).")
    X = df[confs].to_numpy(dtype=float)
    x_names = confs

    # m_hat: must be (n,K)
    if hasattr(model, "m_hat_") and getattr(model, "m_hat_", None) is not None:
        m_hat = np.asarray(model.m_hat_, dtype=float)
    else:
        preds = getattr(model, "predictions", None)
        if isinstance(preds, dict) and "ml_m" in preds:
            m_hat = np.asarray(preds["ml_m"], dtype=float)
        else:
            raise AttributeError("Could not locate propensity predictions (m_hat_ or predictions['ml_m']).")

    # d: must be one-hot (n,K)
    if hasattr(model, "d_") and getattr(model, "d_", None) is not None:
        d = np.asarray(model.d_, dtype=float)
    else:
        raise AttributeError("Could not locate one-hot treatment indicators d (expected model.d_ or diagnostic_data).")

    score = "ATE"
    used_norm = bool(getattr(model, "normalize_ipw", False))

    K = d.shape[1]
    t_names = [str(k) for k in range(K)]
    return X, m_hat, d, score, used_norm, list(x_names), list(t_names)


# ---------------- core SMD routine (multi-treatment, pairwise 0 vs k) ----------------
def _balance_smd(
    X: np.ndarray,
    d: np.ndarray,
    m_hat: np.ndarray,
    *,
    score: str,
    normalize: bool,
    threshold: float,
    treatment_names: _Optional[_List[str]] = None,
) -> _Dict[str, _Any]:
    """
    Multi-treatment ATE-only balance.

    Pairwise comparisons: baseline 0 vs k (k=1..K-1)

    Weights (ATE):
      w_g = D_g / m_g
    Optional normalization:
      w_g <- w_g / mean(w_g)

    Returns SMD matrices as DataFrames (rows=features, cols=comparisons).
    """
    X = np.asarray(X, dtype=float)
    d = np.asarray(d, dtype=float)
    m_hat = np.asarray(m_hat, dtype=float)

    if str(score).upper() != "ATE":
        raise ValueError(f"Only ATE is supported for multi-treatment balance. Got score={score}")

    if X.ndim != 2:
        raise ValueError("X must be (n,p)")
    n, p = X.shape

    if d.ndim != 2 or m_hat.ndim != 2:
        raise ValueError("For multi-treatment, d and m_hat must be 2D arrays of shape (n,K)")

    if d.shape[0] != n or m_hat.shape[0] != n:
        raise ValueError("First dimension of d and m_hat must match X")
    if d.shape[1] != m_hat.shape[1]:
        raise ValueError("d and m_hat must have the same number of treatment columns")

    K = d.shape[1]
    if K < 2:
        raise ValueError("Need at least 2 treatments (K>=2)")

    if treatment_names is None or len(treatment_names) != K:
        treatment_names = [str(k) for k in range(K)]

    eps = 1e-12
    m = np.clip(m_hat, eps, 1.0 - eps)

    def _w_moments(w: np.ndarray) -> _Tuple[np.ndarray, np.ndarray, float]:
        sw = float(np.sum(w))
        if sw <= 0:
            return np.full(p, np.nan), np.full(p, np.nan), 0.0
        mu = (w[:, None] * X).sum(axis=0) / sw
        var = (w[:, None] * (X - mu) ** 2).sum(axis=0) / sw
        return mu, var, sw

    def _smd(mu_a, var_a, mu_b, var_b) -> np.ndarray:
        s_pool = np.sqrt(0.5 * (np.maximum(var_a, 0.0) + np.maximum(var_b, 0.0)))
        diff = np.abs(mu_a - mu_b)

        out = np.full(p, np.nan, dtype=float)
        zero_both = (var_a <= 1e-16) & (var_b <= 1e-16)
        ok = (~zero_both) & (s_pool > 1e-16)

        out[ok] = diff[ok] / s_pool[ok]
        out[zero_both & (diff <= 1e-16)] = 0.0
        out[zero_both & (diff > 1e-16)] = np.inf
        return out

    comp_cols: _List[str] = []
    smd_w_cols: _List[np.ndarray] = []
    smd_u_cols: _List[np.ndarray] = []

    weights_by_comp: _Dict[str, _Tuple[np.ndarray, np.ndarray]] = {}
    mass_by_comp: _Dict[str, _Tuple[float, float]] = {}

    for k in range(1, K):
        comp = f"{treatment_names[0]}_vs_{treatment_names[k]}"
        comp_cols.append(comp)

        w0 = d[:, 0] / m[:, 0]
        wk = d[:, k] / m[:, k]

        if normalize:
            m0 = float(np.mean(w0))
            mk = float(np.mean(wk))
            w0 = w0 / (m0 if m0 != 0 else 1.0)
            wk = wk / (mk if mk != 0 else 1.0)

        mu0, var0, s0 = _w_moments(w0)
        muk, vark, sk = _w_moments(wk)
        if s0 <= 0 or sk <= 0:
            raise RuntimeError(f"Degenerate weights in comparison {comp}: zero total mass.")

        smd_w = _smd(muk, vark, mu0, var0)
        smd_w_cols.append(smd_w)

        # unweighted within raw groups (0 vs k)
        mask0 = d[:, 0].astype(bool)
        maskk = d[:, k].astype(bool)
        if (not np.any(mask0)) or (not np.any(maskk)):
            smd_u = np.full(p, np.nan, dtype=float)
        else:
            X0 = X[mask0]
            Xk = X[maskk]
            mu0_u = X0.mean(axis=0)
            muk_u = Xk.mean(axis=0)
            var0_u = X0.var(axis=0, ddof=0)
            vark_u = Xk.var(axis=0, ddof=0)
            smd_u = _smd(muk_u, vark_u, mu0_u, var0_u)
        smd_u_cols.append(smd_u)

        weights_by_comp[comp] = (wk, w0)  # (treated=k, control=0) within this comparison
        mass_by_comp[comp] = (float(np.sum(wk)), float(np.sum(w0)))

    smd_weighted = pd.DataFrame(np.vstack(smd_w_cols).T, columns=comp_cols)
    smd_unweighted = pd.DataFrame(np.vstack(smd_u_cols).T, columns=comp_cols)

    # quick summaries (overall across all comparisons/features)
    flat = smd_weighted.to_numpy().ravel()
    finite = np.isfinite(flat)
    smd_max = float(np.nanmax(flat[finite])) if np.any(finite) else float("nan")
    frac_viol = float(np.mean(flat[finite] >= float(threshold))) if np.any(finite) else 0.0

    return {
        "smd_weighted": smd_weighted,       # DataFrame (p, K-1)
        "smd_unweighted": smd_unweighted,   # DataFrame (p, K-1)
        "smd_max": smd_max,
        "frac_violations": frac_viol,
        "comparisons": comp_cols,
        "weights_by_comp": weights_by_comp,  # dict[comp] -> (w_k, w_0)
        "mass_by_comp": mass_by_comp,        # dict[comp] -> (sum_wk, sum_w0)
    }


def run_uncofoundedness_diagnostics(
    *,
    res: _Dict[str, _Any] | _Any = None,
    X: _Optional[np.ndarray] = None,
    d: _Optional[np.ndarray] = None,          # (n, K) one-hot
    m_hat: _Optional[np.ndarray] = None,      # (n, K)
    names: _Optional[_List[str]] = None,      # len p (confounders)
    treatment_names: _Optional[_List[str]] = None,  # len K (optional)
    score: _Optional[str] = None,             # only ATE supported
    normalize: _Optional[bool] = None,
    threshold: float = 0.10,
    eps_overlap: float = 0.01,                # kept for API compatibility (not used here)
    return_summary: bool = True,
) -> _Dict[str, _Any]:
    """
    Multi-treatment unconfoundedness diagnostics focused on balance (SMD), ATE only.

    Pairwise comparisons: baseline treatment 0 vs k (k=1..K-1)

    Inputs:
      - Either `res` containing diagnostic_data with x, d(one-hot), m_hat(matrix),
        or raw arrays X, d, m_hat (+ optional names, treatment_names, normalize).
    Returns:
      {
        "params": {"score", "normalize", "smd_threshold"},
        "balance": {
            "smd": pd.DataFrame (p, K-1),
            "smd_unweighted": pd.DataFrame (p, K-1),
            "smd_max": float,
            "frac_violations": float,
            "pass": bool,
            "worst_features": pd.Series (top 10 by max SMD across comparisons),
            "comparisons": list[str],
        },
        "flags": {"balance_max_smd", "balance_violations"},
        "overall_flag": str,
        "meta": {"n", "p", "K", "treatment_names"},
        "summary": pd.DataFrame (optional)
      }
    """

    # ---- Resolve inputs ----
    if (X is None or d is None or m_hat is None) and res is None:
        raise ValueError("Pass either (X, d, m_hat) or `res` with diagnostic_data/model.")

    if X is None or d is None or m_hat is None:
        X, m_hat, d, score_auto, used_norm_auto, names_auto, tnames_auto = _extract_balance_inputs_from_result(res)
        if score is None:
            score = score_auto
        if normalize is None:
            normalize = used_norm_auto
        if names is None:
            names = names_auto
        if treatment_names is None:
            treatment_names = tnames_auto
    else:
        X = np.asarray(X, dtype=float)
        d = np.asarray(d, dtype=float)
        m_hat = np.asarray(m_hat, dtype=float)

        if score is None:
            score = "ATE"
        if normalize is None:
            normalize = False
        if names is None:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array (n,p)")
            names = [f"x{j+1}" for j in range(X.shape[1])]
        if treatment_names is None:
            if d.ndim != 2:
                raise ValueError("For multi-treatment, d must be 2D one-hot (n,K)")
            treatment_names = [str(k) for k in range(d.shape[1])]

    score_u = str(score).upper()
    if score_u != "ATE":
        raise ValueError(f"Only ATE is supported. Got score={score_u}")
    used_norm = bool(normalize)

    # ---- Validate shapes ----
    X = np.asarray(X, dtype=float)
    d = np.asarray(d, dtype=float)
    m_hat = np.asarray(m_hat, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be (n,p)")
    if d.ndim != 2 or m_hat.ndim != 2:
        raise ValueError("d and m_hat must be 2D arrays of shape (n,K)")

    n, p = X.shape
    if d.shape[0] != n or m_hat.shape[0] != n:
        raise ValueError("X, d, m_hat must have matching first dimension n.")
    if d.shape[1] != m_hat.shape[1]:
        raise ValueError("d and m_hat must have matching number of treatments K.")
    K = d.shape[1]

    if names is None or len(names) != p:
        names = [f"x{j+1}" for j in range(p)]
    if treatment_names is None or len(treatment_names) != K:
        treatment_names = [str(k) for k in range(K)]

    # ---- Balance (SMD only) ----
    bal = _balance_smd(
        X, d, m_hat,
        score=score_u,
        normalize=used_norm,
        threshold=threshold,
        treatment_names=treatment_names,
    )

    # attach row index = feature names
    smd_w_df = bal["smd_weighted"].copy()
    smd_u_df = bal["smd_unweighted"].copy()
    smd_w_df.index = names
    smd_u_df.index = names

    # worst features by max across comparisons
    worst = smd_w_df.max(axis=1).sort_values(ascending=False).head(10)

    frac_viol = float(bal["frac_violations"]) if np.isfinite(bal["frac_violations"]) else 0.0

    flat = smd_w_df.to_numpy().ravel()
    finite = np.isfinite(flat)
    pass_bal = (
        bool(np.all(flat[finite] < float(threshold)) and (frac_viol < 0.10))
        if np.any(finite)
        else True
    )

    balance_block = {
        "smd": smd_w_df,                 # (p, K-1)
        "smd_unweighted": smd_u_df,      # (p, K-1)
        "smd_max": float(bal["smd_max"]),
        "frac_violations": frac_viol,
        "pass": pass_bal,
        "worst_features": worst,
        "comparisons": list(bal.get("comparisons", [])),
    }

    # ---- Balance-only flags & thresholds ----
    def _grade_balance(val: float, warn: float, strong: float) -> str:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "NA"
        v = float(val)
        return "GREEN" if v < warn else ("YELLOW" if v < strong else "RED")

    smd_warn = float(threshold)
    smd_strong = float(threshold) * 2.0
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
        "score": "ATE",
        "normalize": used_norm,
        "smd_threshold": float(threshold),
    }

    report: _Dict[str, _Any] = {
        "params": params,
        "balance": balance_block,
        "flags": balance_flags,
        "overall_flag": overall_flag,
        "meta": {
            "n": int(n),
            "p": int(p),
            "K": int(K),
            "treatment_names": list(treatment_names),
            "baseline_treatment": treatment_names[0] if treatment_names else "0",
        },
    }

    # ---- Optional summary table (balance-only) ----
    if return_summary:
        report["summary"] = pd.DataFrame(
            [
                {"metric": "balance_max_smd", "value": balance_block["smd_max"], "flag": balance_flags["balance_max_smd"]},
                {"metric": "balance_frac_violations", "value": balance_block["frac_violations"], "flag": balance_flags["balance_violations"]},
            ]
        )

    return report