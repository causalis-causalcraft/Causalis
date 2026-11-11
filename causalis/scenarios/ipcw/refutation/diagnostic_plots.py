"""Plots for fixed-horizon IPCW estimates, with lazy matplotlib imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from causalis.data_contracts.ipcw_estimate import IPCWEstimate

from .diagnostics import _diagnostic_data

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_ipcw_weights(
    estimate: IPCWEstimate,
    *,
    log_scale: bool = False,
    figsize: tuple[float, float] = (9.0, 5.0),
    dpi: int = 160,
) -> Figure:
    """Return a closed figure of monthly weights among usable controls."""
    diagnostics = _diagnostic_data(estimate)
    rows = diagnostics.row_data
    usable = rows.loc[rows["treatment"].eq(0) & rows["outcome_usable"].astype(bool)]
    if usable.empty:
        raise ValueError("No usable control outcomes are available to plot.")
    weights = usable["control_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Usable control weights must be finite and positive.")

    import matplotlib.pyplot as plt

    grouped = list(usable.groupby("followup_month", sort=True, observed=True))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.boxplot(
        [part["control_weight"].to_numpy(dtype=float) for _, part in grouped],
        positions=np.arange(len(grouped)),
        widths=0.55,
        showfliers=True,
    )
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels([str(month) for month, _ in grouped])
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("IPCW Control Weights")
    ax.set_xlabel("Follow-up month")
    ax.set_ylabel("Baseline odds × inverse censoring survival")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_ipcw_effects(
    estimate: IPCWEstimate,
    *,
    cumulative: bool = True,
    figsize: tuple[float, float] = (9.0, 5.0),
    dpi: int = 160,
) -> Figure:
    """Return a closed monthly or cumulative ATT figure with available CIs.

    Effect plots remain available when fit-time diagnostics were not stored.
    Missing bootstrap intervals are omitted while point estimates are plotted.
    """
    if not isinstance(estimate, IPCWEstimate):
        raise TypeError("estimate must be an IPCWEstimate instance.")
    table = estimate.cumulative_effects if cumulative else estimate.period_effects
    if table.empty:
        raise ValueError("No IPCW effects are available to plot.")
    table = table.sort_values("followup_month")
    x = table["followup_month"].to_numpy(dtype=float)
    values = table["value"].to_numpy(dtype=float)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, values, color="#1f77b4", marker="o", label="ATT")
    if {"ci_lower_absolute", "ci_upper_absolute"}.issubset(table.columns):
        lower = table["ci_lower_absolute"].to_numpy(dtype=float)
        upper = table["ci_upper_absolute"].to_numpy(dtype=float)
        available = np.isfinite(lower) & np.isfinite(upper)
        if available.any():
            ax.vlines(
                x[available],
                lower[available],
                upper[available],
                color="#1f77b4",
                alpha=0.7,
            )
    ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xlabel("Follow-up month")
    ax.set_ylabel("Cumulative ATT" if cumulative else "Monthly ATT")
    ax.set_title("IPCW Cumulative Effects" if cumulative else "IPCW Monthly Effects")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig
