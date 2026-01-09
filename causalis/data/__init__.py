"""
Data generation utilities for causal inference tasks.
"""

from . import dgps, causaldata, causaldata_instrumental, causal_estimate
from causalis.data.dgps import (
    generate_rct,
    generate_classic_rct,
    obs_linear_effect,
    make_gold_linear, SmokingDGP, obs_linear_26_dataset,
    generate_classic_rct_26,
    CausalDatasetGenerator
)
from causalis.data.causaldata import CausalData
from causalis.data.causaldata_instrumental import CausalDataInstrumental
from causalis.data.causal_estimate import CausalEstimate
from causalis.data.causal_diagnostic_data import DiagnosticData, UnconfoundednessDiagnosticData

__all__ = [
    "dgps", "causaldata", "causaldata_instrumental", "causal_estimate",
    "generate_rct",
    "generate_classic_rct",
    "obs_linear_effect",
    "make_gold_linear", "SmokingDGP", "obs_linear_26_dataset",
    "generate_classic_rct_26",
    "CausalData", "CausalDataInstrumental", "CausalDatasetGenerator",
    "CausalEstimate",
    "DiagnosticData",
    "UnconfoundednessDiagnosticData",
]

