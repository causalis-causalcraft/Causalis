r"""Residual diagnostic plots for nuisance models g0/g1 and m."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData

from .._shared import _aligned_data_array, _resolve_cluster_context


def _binned_mean_line(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quantile-binned x midpoints and mean y values."""
    mids, means, _ = _binned_summary(x, y, n_bins=n_bins)
    return mids, means


def _binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute quantile-binned x midpoints, mean y values, and counts."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    n = x_arr.size
    n_bins_eff = max(2, min(int(n_bins), n))
    quantiles = np.linspace(0.0, 1.0, n_bins_eff + 1)
    edges = np.quantile(x_arr, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    # Right edges enter the next bin, except the final edge, matching the
    # historical interval convention without rescanning all rows for each bin.
    bins = np.searchsorted(edges[1:-1], x_arr, side="right")
    counts = np.bincount(bins, minlength=edges.size - 1).astype(float)
    sum_x = np.bincount(bins, weights=x_arr, minlength=counts.size)
    sum_y = np.bincount(bins, weights=y_arr, minlength=counts.size)
    present = counts > 0
    return sum_x[present] / counts[present], sum_y[present] / counts[present], counts[present]


def _maybe_subsample_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: Optional[int],
    rng: Optional[np.random.Generator],
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally downsample scatter inputs without affecting summary overlays."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size != y_arr.size:
        raise ValueError("Scatter arrays must have matching sample size.")

    if max_points is None or x_arr.size <= int(max_points):
        return x_arr, y_arr

    if int(max_points) <= 0:
        raise ValueError("max_scatter_points must be a positive integer when provided.")

    if rng is None:
        rng = np.random.default_rng()
    idx = np.asarray(rng.choice(x_arr.size, size=int(max_points), replace=False), dtype=int)
    return x_arr[idx], y_arr[idx]


def _resolve_residual_inputs(
    estimate: CausalEstimate,
    data: Optional[CausalData],
    *,
    clip_propensity: float,
) -> dict[str, Any]:
    """Resolve residual-plot arrays from diagnostic data or a fallback dataset."""
    diagnostic_data = estimate.diagnostic_data
    if diagnostic_data is None:
        raise ValueError(
            "Missing estimate.diagnostic_data. "
            "Fit IRM with store_diagnostics=True and call estimate() first."
        )

    cache = getattr(diagnostic_data, "residual_plot_cache", None)
    m_source = getattr(diagnostic_data, "m_hat", None)
    source_n = None
    if m_source is None and isinstance(cache, dict):
        m_source = cache.get("m")
        if cache.get("finite_mask") is not None:
            source_n = np.asarray(cache["finite_mask"]).size
    if m_source is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `g0_hat`.")
    if source_n is None:
        source_n = np.asarray(m_source).size
    context = _resolve_cluster_context(data, estimate, source_n)

    # Fast path: reuse cached residual-plot inputs stored in diagnostic_data.
    if isinstance(cache, dict):
        required = {"y", "d", "g0", "g1", "m"}
        if required.issubset(set(cache.keys())):
            y_cached = np.asarray(cache.get("y"), dtype=float).ravel()
            d_cached = np.asarray(cache.get("d"), dtype=float).ravel()
            g0_cached = np.asarray(cache.get("g0"), dtype=float).ravel()
            g1_cached = np.asarray(cache.get("g1"), dtype=float).ravel()
            m_cached = np.asarray(cache.get("m"), dtype=float).ravel()
            n_cached = y_cached.size
            finite_cached = cache.get("finite_mask")
            cache_context = context
            if finite_cached is not None:
                finite_cached = np.asarray(finite_cached, dtype=bool)
                if finite_cached.ndim == 1 and finite_cached.size == source_n:
                    cache_context = context.subset(finite_cached)
                else:
                    n_cached = -1
            if (
                n_cached > 0
                and d_cached.size == n_cached
                and g0_cached.size == n_cached
                and g1_cached.size == n_cached
                and m_cached.size == n_cached
                and np.all(np.isfinite(y_cached))
                and np.all(np.isfinite(d_cached))
                and np.all(np.isfinite(g0_cached))
                and np.all(np.isfinite(g1_cached))
                and np.all(np.isfinite(m_cached))
                and cache_context.row_positions.size == n_cached
            ):
                eps = float(max(clip_propensity, 1e-12))
                return {
                    "y": y_cached,
                    "d": (d_cached > 0.5).astype(float),
                    "g0": g0_cached,
                    "g1": g1_cached,
                    "m": np.clip(m_cached, eps, 1.0 - eps),
                    "metadata": {**cache_context.metadata, "weighting": "observation"},
                }

    m_raw = getattr(diagnostic_data, "m_hat", None)
    g0_raw = getattr(diagnostic_data, "g0_hat", None)
    if m_raw is None or g0_raw is None:
        raise ValueError("estimate.diagnostic_data must include `m_hat` and `g0_hat`.")

    g1_raw = getattr(diagnostic_data, "g1_hat", None)
    if g1_raw is None:
        raise ValueError(
            "estimate.diagnostic_data must include `g1_hat` for treated residual diagnostics."
        )

    y_raw = getattr(diagnostic_data, "y", None)
    if y_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `y`, or pass `data` for fallback.")
        y_raw = _aligned_data_array(data, diagnostic_data, str(data.outcome_name))

    d_raw = getattr(diagnostic_data, "d", None)
    if d_raw is None:
        if data is None:
            raise ValueError("diagnostic_data must include `d`, or pass `data` for fallback.")
        d_raw = _aligned_data_array(data, diagnostic_data, str(data.treatment_name))

    y = np.asarray(y_raw, dtype=float).ravel()
    d = np.asarray(d_raw, dtype=float).ravel()
    g0 = np.asarray(g0_raw, dtype=float).ravel()
    g1 = np.asarray(g1_raw, dtype=float).ravel()
    m = np.asarray(m_raw, dtype=float).ravel()

    n = int(y.size)
    if any(arr.size != n for arr in (d, g0, g1, m)):
        raise ValueError("All diagnostic arrays must have matching sample size n.")

    finite = np.isfinite(y) & np.isfinite(d) & np.isfinite(g0) & np.isfinite(g1) & np.isfinite(m)
    if not np.all(finite):
        y, d, g0, g1, m = y[finite], d[finite], g0[finite], g1[finite], m[finite]
        context = context.subset(finite)
    d = (d > 0.5).astype(float)
    if y.size == 0:
        raise ValueError("No finite observations available for residual plotting.")

    m_unc = np.asarray(m, dtype=float).ravel()
    eps = float(max(clip_propensity, 1e-12))
    m = np.clip(m_unc, eps, 1.0 - eps)

    resolved = {
        "y": y,
        "d": d,
        "g0": g0,
        "g1": g1,
        "m": m,
        "metadata": {**context.metadata, "weighting": "observation"},
    }
    try:
        diagnostic_data.residual_plot_cache = {
            "y": np.asarray(y, dtype=float).ravel(),
            "d": np.asarray(d, dtype=float).ravel(),
            "g0": np.asarray(g0, dtype=float).ravel(),
            "g1": np.asarray(g1, dtype=float).ravel(),
            "m": m_unc,
            "finite_mask": finite,
        }
    except Exception:
        pass
    return resolved


def plot_residual_diagnostics(
    estimate: CausalEstimate,
    data: Optional[CausalData] = None,
    *,
    clip_propensity: float = 1e-6,
    n_bins: int = 20,
    marker_size: float = 12.0,
    alpha: float = 0.35,
    max_scatter_points: Optional[int] = 20_000,
    random_state: Optional[int] = 42,
    figsize: Tuple[float, float] = (14.0, 4.8),
    dpi: int = 220,
    font_scale: float = 1.10,
    save: Optional[str] = None,
    save_dpi: Optional[int] = None,
    transparent: bool = False,
) -> plt.Figure:
    r"""
    Plot residual diagnostics for nuisance models.

    Panels
    ------
    1. Treated-only: ``u1 = y - g1`` vs ``g1``.
    2. Control-only: ``u0 = y - g0`` vs ``g0``.
    3. Binned calibration error: ``E[d - m | m in bin]`` vs binned ``m``.

    Notes
    -----
    These plots check the nuisance pieces directly:

    - outcome residuals :math:`u_1 = Y - \hat g_1(X)` and
      :math:`u_0 = Y - \hat g_0(X)` should look roughly centered around zero
      without strong patterns against fitted values,
    - treatment residuals :math:`D - \hat m(X)` should average near zero within
      propensity bins.

    Clear trends usually mean the nuisance models still leave structure in the
    data, which can show up later as unstable score diagnostics.
    Summaries weight observations, including with unequal cluster sizes. The
    binned curves are descriptive and do not draw independence-based intervals.

    Parameters
    ----------
    estimate : CausalEstimate
        Estimate with diagnostic data (`m_hat`, `g0_hat`; optionally `g1_hat`, `y`, `d`).
    data : CausalData, optional
        Optional fallback source for `y` and `d` when missing in diagnostic data.
    clip_propensity : float, default 1e-6
        Clipping epsilon for propensity values in the treatment-residual panel.
    n_bins : int, default 20
        Number of quantile bins for the binned-mean trend overlays.
    marker_size : float, default 12.0
        Scatter marker size.
    alpha : float, default 0.35
        Scatter opacity.
    max_scatter_points : int, optional, default 20000
        Maximum number of points drawn in each residual scatter panel. When set,
        scatter points are sampled uniformly without replacement, while the
        binned-mean overlays and calibration panel still use all observations.
        Pass None to draw all observations.
    random_state : int, optional, default 42
        Random seed used when ``max_scatter_points`` triggers scatter sampling.
    figsize : tuple, default (14.0, 4.8)
        Figure size.
    dpi : int, default 220
        Dots per inch.
    font_scale : float, default 1.10
        Font scaling factor.
    save : str, optional
        Path to save the figure.
    save_dpi : int, optional
        DPI for saving.
    transparent : bool, default False
        Whether to save with transparency.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

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
    >>> fig = plot_residual_diagnostics(estimate, data=data)  # doctest: +SKIP
    """

    resolved = _resolve_residual_inputs(
        estimate=estimate,
        data=data,
        clip_propensity=float(clip_propensity),
    )
    y = resolved["y"]
    d = resolved["d"]
    g0 = resolved["g0"]
    g1 = resolved["g1"]
    m = resolved["m"]

    u1 = y - g1
    u0 = y - g0
    rd = d - m

    mask_t = d > 0.5
    mask_c = ~mask_t
    if not np.any(mask_t) or not np.any(mask_c):
        raise ValueError("Both treated and control observations are required for residual plots.")

    if max_scatter_points is not None and int(max_scatter_points) <= 0:
        raise ValueError("max_scatter_points must be a positive integer when provided.")

    rng = np.random.default_rng(random_state) if max_scatter_points is not None else None

    rc = {
        "font.size": 11 * font_scale,
        "axes.titlesize": 12.5 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 9.5 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
    }

    with mpl.rc_context(rc):
        fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        ax_t, ax_c, ax_m = axs

        # 1) Treated: u1 vs g1
        x_t = g1[mask_t]
        y_t = u1[mask_t]
        x_t_scatter, y_t_scatter = _maybe_subsample_scatter(
            x_t,
            y_t,
            max_points=max_scatter_points,
            rng=rng,
        )
        ax_t.scatter(
            x_t_scatter,
            y_t_scatter,
            s=marker_size,
            alpha=alpha,
            color="C0",
            linewidths=0.0,
        )
        xb_t, yb_t = _binned_mean_line(x_t, y_t, n_bins=int(n_bins))
        if xb_t.size > 0:
            ax_t.plot(xb_t, yb_t, color="black", linewidth=1.6, label="Binned mean")
            ax_t.legend(frameon=False)
        ax_t.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_t.set_xlabel(r"$\hat g_1(X)$")
        ax_t.set_ylabel(r"$u_1 = Y - \hat g_1(X)$")
        ax_t.set_title("Treated: Residual vs Fitted")
        ax_t.grid(True, linewidth=0.5, alpha=0.45)

        # 2) Control: u0 vs g0
        x_c = g0[mask_c]
        y_c = u0[mask_c]
        x_c_scatter, y_c_scatter = _maybe_subsample_scatter(
            x_c,
            y_c,
            max_points=max_scatter_points,
            rng=rng,
        )
        ax_c.scatter(
            x_c_scatter,
            y_c_scatter,
            s=marker_size,
            alpha=alpha,
            color="C1",
            linewidths=0.0,
        )
        xb_c, yb_c = _binned_mean_line(x_c, y_c, n_bins=int(n_bins))
        if xb_c.size > 0:
            ax_c.plot(xb_c, yb_c, color="black", linewidth=1.6, label="Binned mean")
            ax_c.legend(frameon=False)
        ax_c.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_c.set_xlabel(r"$\hat g_0(X)$")
        ax_c.set_ylabel(r"$u_0 = Y - \hat g_0(X)$")
        ax_c.set_title("Control: Residual vs Fitted")
        ax_c.grid(True, linewidth=0.5, alpha=0.45)

        # 3) Calibration error by propensity bin: E[D - m | m in bin] vs E[m | m in bin]
        xb_m, yb_m, cb_m = _binned_summary(m, rd, n_bins=int(n_bins))
        if xb_m.size > 0:
            cmax = float(np.max(cb_m)) if np.max(cb_m) > 0.0 else 1.0
            sizes = 35.0 + 220.0 * (cb_m / cmax)
            ax_m.scatter(
                xb_m,
                yb_m,
                s=sizes,
                alpha=0.80,
                color="C2",
                edgecolors="white",
                linewidths=0.6,
                label="Bin calibration error (size = count)",
                zorder=3,
            )
            ax_m.plot(xb_m, yb_m, color="black", linewidth=1.5, zorder=2)
        ax_m.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_m.set_xlim(0.0, 1.0)
        ax_m.set_xlabel(r"$\hat m(X)$")
        ax_m.set_ylabel(r"$\mathbb{E}[D-\hat m(X)\mid \hat m(X)\in \mathrm{bin}]$")
        ax_m.set_title("Calibration Error by Propensity Bin")
        ax_m.grid(True, linewidth=0.5, alpha=0.45)
        ax_m.legend(frameon=False, loc="best")

        for axis in axs:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        fig.tight_layout()

        if save is not None:
            ext = str(save).lower().split(".")[-1]
            _dpi = save_dpi or (300 if ext in {"png", "jpg", "jpeg", "tif", "tiff"} else dpi)
            fig.savefig(
                save,
                dpi=_dpi,
                bbox_inches="tight",
                pad_inches=0.1,
                transparent=transparent,
                facecolor="none" if transparent else "white",
            )

    # Close notebook-managed figure to avoid duplicate Jupyter rendering
    # (inline auto-display + returned Figure rich repr).
    plt.close(fig)
    return fig


__all__ = ["plot_residual_diagnostics"]
