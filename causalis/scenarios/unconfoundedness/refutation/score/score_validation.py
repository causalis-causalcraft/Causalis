"""Score diagnostics focused on orthogonality and EIF stability."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness._cluster_utils import mean_covariance
from causalis.scenarios.unconfoundedness._score_utils import (
    _normalize_ipw_terms,
    _resolve_ate_weights,
)
from causalis.scenarios.unconfoundedness.refutation._shared import (
    _aligned_data_array,
    _normalize_score,
    _resolve_cluster_context,
)


def _resolve_overlap_threshold(
    overlap_threshold: Optional[float],
    diagnostic_data: Any,
    estimate: CausalEstimate,
) -> float:
    """Resolve the clipping threshold from explicit, diagnostic, or model state."""
    if overlap_threshold is not None:
        return float(overlap_threshold)

    overlap_thr = getattr(diagnostic_data, "overlap_threshold", None)
    if overlap_thr is None:
        overlap_thr = estimate.model_options.get("overlap_threshold", None)
    if overlap_thr is None:
        overlap_thr = 0.01
    return float(overlap_thr)


def _resolve_normalize_ipw(score: str, diagnostic_data: Any, estimate: CausalEstimate) -> bool:
    """Resolve whether normalized IPW should be used for diagnostics."""
    normalize_ipw = getattr(diagnostic_data, "normalize_ipw", None)
    if normalize_ipw is None:
        normalize_ipw = estimate.model_options.get("normalize_ipw", False)
    if score == "ATTE":
        return False
    return bool(normalize_ipw)


def _resolve_saved_ate_weights(estimate: CausalEstimate, n: int):
    """Resolve frozen weights without calling back into a mutable live model."""
    diagnostic = estimate.diagnostic_data
    w_raw, w_bar_raw = getattr(diagnostic, "w", None), getattr(diagnostic, "w_bar", None)
    model = getattr(diagnostic, "_model", None)
    custom = (
        estimate.model_options.get("se_approx_weight_norm", False)
        or estimate.model_options.get("uses_custom_weights", False)
        or getattr(model, "_fit_weights_used_", False)
        or getattr(model, "weights", None) is not None
    )
    if custom and (w_raw is None or w_bar_raw is None):
        raise ValueError("Custom-weight diagnostics require saved w and w_bar; refit with diagnostics before recomputing scores.")
    return _resolve_ate_weights(n=n, w_raw=w_raw, w_bar_raw=w_bar_raw)


def _aipw_score_ate(
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    theta: float,
    overlap_threshold: float,
    *,
    normalize_ipw: bool,
    w: np.ndarray,
    w_bar: np.ndarray,
) -> np.ndarray:
    """Compute per-observation ATE score residuals."""
    m_clipped = np.clip(m, overlap_threshold, 1.0 - overlap_threshold)
    h1, h0 = _normalize_ipw_terms(d, m_clipped, normalize_ipw=normalize_ipw)
    u0 = y - g0
    u1 = y - g1
    psi_b = w * (g1 - g0) + w_bar * (u1 * h1 - u0 * h0)
    return psi_b - theta


def _aipw_score_atte(
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    theta: float,
    overlap_threshold: float,
) -> np.ndarray:
    """Compute per-observation ATTE score residuals."""
    m_clipped = np.clip(m, overlap_threshold, 1.0 - overlap_threshold)
    p_treated = float(np.mean(d))
    gamma = m_clipped / (1.0 - m_clipped)
    num = d * (y - g0 - theta) - (1.0 - d) * gamma * (y - g0)
    return num / (p_treated + 1e-12)


def _basis_columns(x_basis: np.ndarray, *, standardize: bool = False):
    """Yield one basis column at a time without an additional n-by-p matrix."""
    if standardize:
        yield np.ones(x_basis.shape[0], dtype=float)
    for column in range(x_basis.shape[1]):
        values = x_basis[:, column]
        if standardize:
            values = (values - np.mean(values)) / (np.std(values) + 1e-8)
        yield values


def _diagnostic_se(influence: np.ndarray, cluster_codes: Optional[np.ndarray]) -> float:
    if not np.isfinite(influence).all():
        return float("nan")
    variance = float(mean_covariance(influence, cluster_codes=cluster_codes)[0, 0])
    return float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else float("nan")


def _ratio_mean(numerator: np.ndarray, denominator: np.ndarray):
    """A ratio of empirical means and its delta-method row influence."""
    scale = float(np.mean(denominator))
    if not np.isfinite(scale) or abs(scale) <= 1e-12:
        return float("nan"), np.full(numerator.size, np.nan)
    value = float(np.mean(numerator) / scale)
    return value, (numerator - value * denominator) / scale


def _ratio_directional_derivative(r: np.ndarray, t: np.ndarray, q: np.ndarray):
    """Derivative of mean(t*r)/mean(r) when dr=q, and its row influence."""
    c, a, q_mean = float(np.mean(r)), float(np.mean(t * r)), float(np.mean(q))
    if not np.isfinite(c) or c <= 1e-12:
        return float("nan"), np.full(r.size, np.nan)
    u = float(np.mean(t * q))
    value = u / c - a * q_mean / c**2
    influence = (
        (t * q - u) / c
        - (q_mean / c**2) * (t * r - a)
        - (a / c**2) * (q - q_mean)
        + (-u / c**2 + 2 * a * q_mean / c**3) * (r - c)
    )
    return value, influence


def _orthogonality_table(rows):
    table = pd.DataFrame(rows)
    for nuisance in ("g1", "g0", "m"):
        table[f"t_{nuisance}"] = table[f"d_{nuisance}"] / np.maximum(table[f"se_{nuisance}"], 1e-12)
    return table[["basis", "d_g1", "se_g1", "t_g1", "d_g0", "se_g0", "t_g0", "d_m", "se_m", "t_m"]]


def _orthogonality_derivatives_ate(
    x_basis: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    overlap_threshold: float,
    *,
    normalize_ipw: bool,
    w: np.ndarray,
    w_bar: np.ndarray,
    cluster_codes: Optional[np.ndarray] = None,
    standardize_basis: bool = False,
) -> pd.DataFrame:
    """Finite-basis derivatives, including empirical IPW normalizer derivatives.

    Propensity directions perturb the effective propensity on the interior of
    the clipping interval. Clipped rows have zero derivative (the boundary uses
    that same zero convention). Custom target weights are held fixed.
    """
    p = np.clip(m, overlap_threshold, 1.0 - overlap_threshold)
    active = (m > overlap_threshold) & (m < 1.0 - overlap_threshold)
    r1, r0 = d / p, (1.0 - d) / (1.0 - p)
    t1, t0 = w_bar * (y - g1), w_bar * (y - g0)
    rows = []
    for j, basis in enumerate(_basis_columns(x_basis, standardize=standardize_basis)):
        wb = w * basis
        if normalize_ipw:
            ratio1, if_ratio1 = _ratio_mean(w_bar * basis * r1, r1)
            ratio0, if_ratio0 = _ratio_mean(w_bar * basis * r0, r0)
            dg1 = float(np.mean(wb)) - ratio1
            dg0 = -float(np.mean(wb)) + ratio0
            if_g1 = wb - np.mean(wb) - if_ratio1
            if_g0 = -wb + np.mean(wb) + if_ratio0
            dm1, if_m1 = _ratio_directional_derivative(r1, t1, -r1 * basis * active / p)
            dm0, if_m0 = _ratio_directional_derivative(r0, t0, r0 * basis * active / (1.0 - p))
            dm, if_m = dm1 - dm0, if_m1 - if_m0
        else:
            if_g1 = basis * (w - w_bar * r1)
            if_g0 = basis * (-w + w_bar * r0)
            if_m = -basis * active * (t1 * r1 / p + t0 * r0 / (1.0 - p))
            dg1, dg0, dm = float(np.mean(if_g1)), float(np.mean(if_g0)), float(np.mean(if_m))
        rows.append({
            "basis": j, "d_g1": dg1, "se_g1": _diagnostic_se(if_g1, cluster_codes),
            "d_g0": dg0, "se_g0": _diagnostic_se(if_g0, cluster_codes),
            "d_m": dm, "se_m": _diagnostic_se(if_m, cluster_codes),
        })
    return _orthogonality_table(rows)


def _orthogonality_derivatives_atte(
    x_basis: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    overlap_threshold: float,
    *,
    cluster_codes: Optional[np.ndarray] = None,
    standardize_basis: bool = False,
) -> pd.DataFrame:
    """ATTE derivatives with the treated-share denominator in their influence."""
    p = np.clip(m, overlap_threshold, 1.0 - overlap_threshold)
    active = (m > overlap_threshold) & (m < 1.0 - overlap_threshold)
    odds = p / (1.0 - p)
    rows = []
    for j, basis in enumerate(_basis_columns(x_basis, standardize=standardize_basis)):
        dg0, if_g0 = _ratio_mean(basis * ((1.0 - d) * odds - d), d)
        dm, if_m = _ratio_mean(-basis * active * (1.0 - d) * (y - g0) / (1.0 - p)**2, d)
        rows.append({
            "basis": j, "d_g1": 0.0, "se_g1": 0.0,
            "d_g0": dg0, "se_g0": _diagnostic_se(if_g0, cluster_codes),
            "d_m": dm, "se_m": _diagnostic_se(if_m, cluster_codes),
        })
    return _orthogonality_table(rows)


def _resolve_influence_level(influence_level: str, cluster_codes: Optional[np.ndarray]) -> str:
    if influence_level not in {"auto", "row", "cluster"}:
        raise ValueError("influence_level must be 'auto', 'row', or 'cluster'.")
    if influence_level == "auto":
        return "cluster" if cluster_codes is not None else "row"
    if influence_level == "cluster" and cluster_codes is None:
        raise ValueError("Cluster influence requires cluster membership in the estimate or aligned data.")
    return influence_level


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    n = values.size
    k = min(max(int(k), 1), n)
    if k == 0:
        return np.zeros(0, dtype=int)
    indices = np.arange(n) if k == n else np.argpartition(-values, k - 1)[:k]
    return indices[np.lexsort((indices, -values[indices]))]


def _aggregate_cluster_influence(psi, d, m, cluster_codes):
    """Observation-weighted cluster contributions; no within-cluster pairs."""
    count = np.bincount(cluster_codes)
    sums = np.bincount(cluster_codes, weights=psi, minlength=count.size)
    # Row filtering or a recomputed score can leave a nonzero empirical mean.
    # Variance shares must use the same centering as the covariance estimator.
    centered_sums = sums - count * float(np.mean(psi))
    sum_squares = float(np.dot(centered_sums, centered_sums))
    return {
        "psi": sums,
        "n": count,
        "m": np.bincount(cluster_codes, weights=m, minlength=count.size) / count,
        "treated_share": np.bincount(cluster_codes, weights=d, minlength=count.size) / count,
        "contribution": sums / psi.size,
        "variance_share": centered_sums**2 / sum_squares if sum_squares > 0 else np.full(count.size, np.nan),
    }


def _influence_summary(
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m: np.ndarray,
    theta: float,
    score: str,
    overlap_threshold: float,
    *,
    normalize_ipw: bool,
    w: np.ndarray,
    w_bar: np.ndarray,
    psi_override: Optional[np.ndarray] = None,
    cluster_codes: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    row_positions: Optional[np.ndarray] = None,
    influence_level: str = "auto",
    k: int = 10,
) -> Dict[str, Any]:
    """Summarize row or independent-cluster influence and robust plug-in SE."""
    if psi_override is not None:
        psi = np.asarray(psi_override, dtype=float).ravel()
    elif score == "ATE":
        psi = _aipw_score_ate(y, d, g0, g1, m, theta, overlap_threshold,
                            normalize_ipw=normalize_ipw, w=w, w_bar=w_bar)
    else:
        psi = _aipw_score_atte(y, d, g0, m, theta, overlap_threshold)
    level = _resolve_influence_level(influence_level, cluster_codes)
    se = _diagnostic_se(psi, cluster_codes)
    if level == "cluster":
        units = _aggregate_cluster_influence(psi, d, m, cluster_codes)
        values = units["psi"]
        idx = _top_k_indices(np.abs(values), k)
        labels = np.arange(values.size) if cluster_labels is None else cluster_labels
        top = pd.DataFrame({
            "cluster": [labels[i] for i in idx], "cluster_code": idx,
            **{name: array[idx] for name, array in units.items()},
        })
    else:
        values = psi
        idx = _top_k_indices(np.abs(values), k)
        top = pd.DataFrame({
            "i": idx if row_positions is None else row_positions[idx],
            "psi": psi[idx], "m": m[idx],
            "res_t": d[idx] * (y[idx] - g1[idx]),
            "res_c": (1.0 - d[idx]) * (y[idx] - g0[idx]),
        })
    abs_values = np.abs(values)
    med, p99 = float(np.median(abs_values)), float(np.quantile(abs_values, 0.99))
    variance = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
    kurtosis = float(np.mean((values - np.mean(values))**4) / variance**2) if variance > 0 else float("nan")
    return {
        "se_plugin": se, "influence_level": level, "n_units": int(values.size),
        "kurtosis": kurtosis, "p99_over_med": p99 / (med + 1e-12),
        "top_influential": top,
    }


def _grade(value: float, warn: float, strong: float) -> str:
    """Map a scalar diagnostic value to a traffic-light severity flag."""
    if value is None or not np.isfinite(value):
        return "NA"
    value_f = float(value)
    if value_f < warn:
        return "GREEN"
    if value_f < strong:
        return "YELLOW"
    return "RED"


def _oos_moment_test_from_psi(
    psi_a: np.ndarray,
    psi_b: np.ndarray,
    folds: np.ndarray,
) -> Dict[str, Any]:
    """Return descriptive fold stability; pooled OOS inference is degenerate.

    Leave-one-fold estimates share their training observations. In particular,
    equal-size ATE folds cancel exactly when their held-out moments are pooled.
    An iid or clustered denominator cannot turn that pooled mean into a test.
    """
    fold_rows = []
    fold_ids = np.asarray(folds).ravel()
    for fold_value in np.unique(fold_ids):
        test = fold_ids == fold_value
        train = ~test
        if not np.any(train) or not np.any(test):
            continue
        mean_a, mean_b = float(np.mean(psi_a[train])), float(np.mean(psi_b[train]))
        theta = -mean_b / mean_a if np.isfinite(mean_a) and abs(mean_a) > 1e-12 else float("nan")
        heldout = psi_b[test] + psi_a[test] * theta
        fold_rows.append({
            "fold": fold_value, "n": int(np.sum(test)), "theta_minus_k": theta,
            "psi_mean": float(np.mean(heldout)),
            "psi_var": float(np.var(heldout, ddof=1)) if heldout.size > 1 else float("nan"),
        })
    return {
        "available": False,
        "descriptive_available": bool(fold_rows),
        "reason": "Pooled leave-one-fold moments can cancel algebraically and share training data; fold results are descriptive only.",
        "oos_tstat_fold": float("nan"), "oos_tstat_strict": float("nan"),
        "p_value_fold": float("nan"), "p_value_strict": float("nan"),
        "fold_table": pd.DataFrame(fold_rows, columns=["fold", "n", "theta_minus_k", "psi_mean", "psi_var"]),
    }


def run_score_diagnostics(
    data: Optional[CausalData],
    estimate: CausalEstimate,
    *,
    overlap_threshold: Optional[float] = None,
    n_basis_funcs: Optional[int] = None,
    return_summary: bool = True,
    influence_level: str = "auto",
) -> Dict[str, Any]:
    r"""Run orthogonality and influence diagnostics for ATE or ATTE scores.

    The main object is the per-observation score contribution. For ATE, this
    diagnostic uses

    .. math::

        \hat\psi_i =
        w_i(\hat g_1(X_i) - \hat g_0(X_i))
        + \bar w_i
        \left[
        (Y_i - \hat g_1(X_i)) \frac{D_i}{\hat m_i}
        -
        (Y_i - \hat g_0(X_i)) \frac{1-D_i}{1-\hat m_i}
        \right]
        - \hat\theta.

    Good score behavior means:

    - the empirical score average is close to zero,
    - finite-basis derivatives with respect to nuisance parts are small,
    - the influence distribution is not driven by a tiny number of very large
      :math:`|\hat\psi_i|`.

    Parameters
    ----------
    data : CausalData
        Dataset used to fit the estimator.
    estimate : CausalEstimate
        Effect estimate with ``diagnostic_data`` containing nuisance predictions
        and optionally cached score arrays.
    overlap_threshold : float, optional
        Propensity clipping threshold. If omitted, the value is inferred from
        diagnostic or model metadata.
    n_basis_funcs : int, optional
        Number of simple basis functions used in orthogonality checks. Defaults
        to one intercept plus all available confounders.
    return_summary : bool, default True
        Include a compact summary table in the returned payload.
    influence_level : {"auto", "row", "cluster"}, default "auto"
        Rank independent cluster sums for clustered estimates, otherwise rows.
        Explicit ``row`` retains observation-level tail summaries and rankings.

    Returns
    -------
    Dict[str, Any]
        Diagnostic report with orthogonality checks, influence summaries,
        descriptive fold stability, and a summary table. Pooled OOS tests are
        unavailable because their fold moments can cancel algebraically.

    Raises
    ------
    ValueError
        If required diagnostic arrays are missing or have incompatible shapes.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    >>> from causalis.dgp import obs_linear_26_dataset
    >>> from causalis.scenarios.unconfoundedness.model import IRM
    >>> data = obs_linear_26_dataset(
    ...     n=1000,
    ...     seed=3141,
    ...     include_oracle=False,
    ...     return_causal_data=True,
    ... )
    >>> irm = IRM(
    ...     data=data,
    ...     ml_g=RandomForestRegressor(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     ml_m=RandomForestClassifier(
    ...         n_estimators=200,
    ...         max_depth=6,
    ...         min_samples_leaf=5,
    ...         random_state=3141,
    ...     ),
    ...     n_folds=3,
    ...     random_state=3141,
    ... )
    >>> estimate = irm.fit().estimate(score="ATE")
    >>> report = run_score_diagnostics(data, estimate)
    >>> report["summary"]  # doctest: +SKIP
    >>> report["influence"]["top_influential"].head()  # doctest: +SKIP
    """
    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    m_raw = getattr(diagnostic_data, "m_hat", None)
    g0_raw = getattr(diagnostic_data, "g0_hat", None)
    if m_raw is None or g0_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `g0_hat`.")

    score = _normalize_score(getattr(diagnostic_data, "score", estimate.estimand))
    overlap_thr = _resolve_overlap_threshold(overlap_threshold, diagnostic_data, estimate)
    normalize_ipw = _resolve_normalize_ipw(score, diagnostic_data, estimate)

    y_raw = getattr(diagnostic_data, "y", None)
    if y_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `y`, or pass `data` for fallback.")
        y_raw = _aligned_data_array(data, diagnostic_data, [str(estimate.outcome)]).ravel()

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `d`, or pass `data` for fallback.")
        d_raw = _aligned_data_array(data, diagnostic_data, [str(estimate.treatment)]).ravel()

    g1_raw = getattr(diagnostic_data, "g1_hat", None)
    if g1_raw is None:
        g1_raw = np.asarray(g0_raw, dtype=float)

    x_raw = getattr(diagnostic_data, "x", None)
    if x_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `x`, or pass `data` for fallback.")
        x_raw = _aligned_data_array(data, diagnostic_data, list(estimate.confounders))

    psi_raw = getattr(diagnostic_data, "psi", None)
    psi_b_raw = getattr(diagnostic_data, "psi_b", None)
    folds_raw = getattr(diagnostic_data, "folds", None)
    diag_w_raw = getattr(diagnostic_data, "w", None)
    diag_w_bar_raw = getattr(diagnostic_data, "w_bar", None)

    y = np.asarray(y_raw, dtype=float).ravel()
    d = np.asarray(d_raw, dtype=float).ravel()
    g0 = np.asarray(g0_raw, dtype=float).ravel()
    g1 = np.asarray(g1_raw, dtype=float).ravel()
    m = np.asarray(m_raw, dtype=float).ravel()
    x = np.asarray(x_raw, dtype=float)

    if x.ndim != 2:
        raise ValueError("Confounder matrix must be 2D with shape (n, p).")

    n = int(y.size)
    if any(arr.size != n for arr in (d, g0, g1, m)) or x.shape[0] != n:
        raise ValueError("All diagnostic arrays must have matching sample size n.")
    context = _resolve_cluster_context(data, estimate, n)
    _resolve_influence_level(influence_level, context.codes)

    if score == "ATE":
        w, w_bar = _resolve_saved_ate_weights(estimate, n)
    else:
        # Resolve the empirical treated share after the common finite mask.
        w = np.ones(n, dtype=float)
        w_bar = np.ones(n, dtype=float)

    psi = None
    if psi_raw is not None:
        psi = np.asarray(psi_raw, dtype=float).ravel()
        if psi.size != n:
            psi = None
    psi_b = None
    if psi_b_raw is not None:
        psi_b = np.asarray(psi_b_raw, dtype=float).ravel()
        if psi_b.size != n:
            psi_b = None
    folds = None
    if folds_raw is not None:
        folds = np.asarray(folds_raw).ravel()
        if folds.size != n:
            folds = None

    # Standardize one retained basis column at a time inside the derivative
    # helpers, avoiding additional n-by-basis arrays and NaN-contaminated means.
    if n_basis_funcs is None:
        n_basis_funcs = int(x.shape[1] + 1)
    n_covs = min(max(int(n_basis_funcs) - 1, 0), int(x.shape[1]))

    x_basis = x[:, :n_covs]

    finite_rows = (
        np.isfinite(y)
        & np.isfinite(d)
        & np.isfinite(g0)
        & np.isfinite(g1)
        & np.isfinite(m)
        & np.isfinite(w)
        & np.isfinite(w_bar)
        & np.all(np.isfinite(x_basis), axis=1)
    )
    if psi is not None:
        finite_rows = finite_rows & np.isfinite(psi)
    if psi_b is not None:
        finite_rows = finite_rows & np.isfinite(psi_b)
    if folds is not None:
        finite_rows = finite_rows & np.isfinite(np.asarray(folds, dtype=float))

    if not np.any(finite_rows):
        raise ValueError("No finite observations available for score diagnostics.")
    if not np.all(finite_rows):
        y, d, g0, g1, m = (arr[finite_rows] for arr in (y, d, g0, g1, m))
        w, w_bar, x_basis = w[finite_rows], w_bar[finite_rows], x_basis[finite_rows]
        psi = psi[finite_rows] if psi is not None else None
        psi_b = psi_b[finite_rows] if psi_b is not None else None
        folds = folds[finite_rows] if folds is not None else None
        context = context.subset(finite_rows)
    d = (d > 0.5).astype(float)
    if score == "ATTE":
        w = d / max(float(np.mean(d)), 1e-12)
        w_bar = np.clip(m, overlap_thr, 1.0 - overlap_thr) / max(float(np.mean(d)), 1e-12)

    if score == "ATE":
        psi_a = -np.ones_like(d, dtype=float)
    else:
        psi_a = -w

    theta = float(estimate.value)
    influence = _influence_summary(
        y=y,
        d=d,
        g0=g0,
        g1=g1,
        m=m,
        theta=theta,
        score=score,
        overlap_threshold=overlap_thr,
        normalize_ipw=normalize_ipw,
        w=w,
        w_bar=w_bar,
        psi_override=psi,
        cluster_codes=context.codes,
        cluster_labels=context.labels,
        row_positions=context.row_positions,
        influence_level=influence_level,
    )

    if score == "ATE":
        ortho = _orthogonality_derivatives_ate(
            x_basis=x_basis,
            y=y,
            d=d,
            g0=g0,
            g1=g1,
            m=m,
            overlap_threshold=overlap_thr,
            normalize_ipw=normalize_ipw,
            w=w,
            w_bar=w_bar,
            cluster_codes=context.codes,
            standardize_basis=True,
        )
    else:
        ortho = _orthogonality_derivatives_atte(
            x_basis=x_basis,
            y=y,
            d=d,
            g0=g0,
            m=m,
            overlap_threshold=overlap_thr,
            cluster_codes=context.codes,
            standardize_basis=True,
        )

    if not context.metadata["inferential_available"]:
        # Even the structurally unused ATTE g1 derivative must not produce a
        # green inferential grade when fewer than two clusters remain.
        for nuisance in ("g1", "g0", "m"):
            ortho[f"se_{nuisance}"] = float("nan")
            ortho[f"t_{nuisance}"] = float("nan")

    def max_abs_t(column):
        values = np.abs(ortho[column].to_numpy(dtype=float))
        return float(np.max(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float("nan")

    max_t_g1, max_t_g0, max_t_m = (max_abs_t(name) for name in ("t_g1", "t_g0", "t_m"))

    thresholds = {
        "tail_ratio_warn": 10.0,
        "tail_ratio_strong": 20.0,
        "kurt_warn": 10.0,
        "kurt_strong": 30.0,
        "t_warn": 2.0,
        "t_strong": 4.0,
    }

    oos_moment_test = {
        "available": False,
        "descriptive_available": False,
        "reason": "Fold scores are unavailable; no OOS inference is reported.",
        "oos_tstat_fold": float("nan"),
        "oos_tstat_strict": float("nan"),
        "p_value_fold": float("nan"),
        "p_value_strict": float("nan"),
        "fold_table": pd.DataFrame(columns=["fold", "n", "theta_minus_k", "psi_mean", "psi_var"]),
    }
    if psi_b is not None and folds is not None and psi_b.size == d.size and folds.size == d.size:
        oos_moment_test = _oos_moment_test_from_psi(psi_a=psi_a, psi_b=psi_b, folds=folds)

    oos_abs_t_candidates = [
        abs(float(oos_moment_test["oos_tstat_fold"])) if np.isfinite(oos_moment_test["oos_tstat_fold"]) else np.nan,
        abs(float(oos_moment_test["oos_tstat_strict"])) if np.isfinite(oos_moment_test["oos_tstat_strict"]) else np.nan,
    ]
    oos_abs_t = float(np.nanmax(oos_abs_t_candidates)) if np.any(np.isfinite(oos_abs_t_candidates)) else float("nan")

    flags = {
        "psi_tail_ratio": _grade(
            float(influence["p99_over_med"]),
            thresholds["tail_ratio_warn"],
            thresholds["tail_ratio_strong"],
        ),
        "psi_kurtosis": _grade(
            float(influence["kurtosis"]),
            thresholds["kurt_warn"],
            thresholds["kurt_strong"],
        ),
        "ortho_max_|t|_g1": _grade(max_t_g1, thresholds["t_warn"], thresholds["t_strong"]),
        "ortho_max_|t|_g0": _grade(max_t_g0, thresholds["t_warn"], thresholds["t_strong"]),
        "ortho_max_|t|_m": _grade(max_t_m, thresholds["t_warn"], thresholds["t_strong"]),
        "oos_moment": _grade(oos_abs_t, thresholds["t_warn"], thresholds["t_strong"]),
    }

    level = {"NA": -1, "GREEN": 0, "YELLOW": 1, "RED": 2}
    inv_level = {value: key for key, value in level.items()}
    overall_flag = inv_level[max(level.get(flag, -1) for flag in flags.values())]

    report: Dict[str, Any] = {
        "params": {
            "score": score,
            "overlap_threshold": float(overlap_thr),
            "normalize_ipw": bool(normalize_ipw),
            "influence_level": influence["influence_level"],
        },
        "orthogonality_derivatives": ortho,
        "influence_diagnostics": influence,
        "oos_moment_test": oos_moment_test,
        "flags": flags,
        "thresholds": thresholds,
        "overall_flag": overall_flag,
        "meta": {
            "n": int(y.size),
            "score": score,
            "used_estimator_psi": bool(psi is not None),
            "uses_custom_weights": bool(score == "ATE" and (diag_w_raw is not None or diag_w_bar_raw is not None)),
            **context.metadata,
        },
    }

    if return_summary:
        summary_rows = [
            {
                "metric": "psi_p99_over_med",
                "value": float(influence["p99_over_med"]),
                "flag": flags["psi_tail_ratio"],
            },
            {
                "metric": "psi_kurtosis",
                "value": float(influence["kurtosis"]),
                "flag": flags["psi_kurtosis"],
            },
        ]
        if score == "ATE":
            summary_rows.append({"metric": "max_|t|_g1", "value": max_t_g1, "flag": flags["ortho_max_|t|_g1"]})
        summary_rows.extend(
            [
                {"metric": "max_|t|_g0", "value": max_t_g0, "flag": flags["ortho_max_|t|_g0"]},
                {"metric": "max_|t|_m", "value": max_t_m, "flag": flags["ortho_max_|t|_m"]},
                {
                    "metric": "oos_max_abs_t",
                    "value": oos_abs_t,
                    "flag": flags["oos_moment"],
                },
            ]
        )
        summary = pd.DataFrame(summary_rows)
        report["summary"] = summary

    return report


__all__ = ["run_score_diagnostics"]
