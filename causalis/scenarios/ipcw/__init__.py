"""Fixed-horizon IPCW estimators, data generation and diagnostics."""

from causalis.data_contracts.ipcw_data import IPCWData
from causalis.data_contracts.ipcw_estimate import IPCWDiagnosticData, IPCWEstimate

from . import dgp, refutation
from .dgp import generate_ipcw_data
from .model import IPCW
from .refutation import (
    ipcw_support_table,
    ipcw_weight_summary_table,
    plot_ipcw_effects,
    plot_ipcw_weights,
    run_ipcw_diagnostics,
)

__all__ = [
    "IPCW",
    "IPCWData",
    "IPCWEstimate",
    "IPCWDiagnosticData",
    "dgp",
    "refutation",
    "generate_ipcw_data",
    "ipcw_support_table",
    "ipcw_weight_summary_table",
    "run_ipcw_diagnostics",
    "plot_ipcw_weights",
    "plot_ipcw_effects",
]
