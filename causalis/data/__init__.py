"""
Data generation utilities for causal inference tasks.
"""

from . import dgps, causaldata, causaldata_instrumental
from causalis.data.dgps import (
    generate_rct,
    obs_linear_effect, obs_linear_effect_data,
    make_gold_linear, SmokingDGP, obs_linear_26_dataset,
    CausalDatasetGenerator
)
from causalis.data.causaldata import CausalData
from causalis.data.causaldata_instrumental import CausalDataInstrumental

__all__ = [
    "dgps", "causaldata", "causaldata_instrumental",
    "generate_rct",
    "obs_linear_effect", "obs_linear_effect_data",
    "make_gold_linear", "SmokingDGP", "obs_linear_26_dataset",
    "CausalData", "CausalDataInstrumental", "CausalDatasetGenerator"
]

