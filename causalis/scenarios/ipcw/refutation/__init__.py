"""Observed support and fitted-weight diagnostics for IPCW."""

from .diagnostics import (
    ipcw_support_table,
    ipcw_weight_summary_table,
    run_ipcw_diagnostics,
)


def plot_ipcw_weights(*args, **kwargs):
    from .diagnostic_plots import plot_ipcw_weights as _plot_ipcw_weights

    return _plot_ipcw_weights(*args, **kwargs)


def plot_ipcw_effects(*args, **kwargs):
    from .diagnostic_plots import plot_ipcw_effects as _plot_ipcw_effects

    return _plot_ipcw_effects(*args, **kwargs)


__all__ = [
    "ipcw_support_table",
    "ipcw_weight_summary_table",
    "run_ipcw_diagnostics",
    "plot_ipcw_weights",
    "plot_ipcw_effects",
]
