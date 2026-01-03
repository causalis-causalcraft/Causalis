"""
Data generation utilities for causal inference tasks.
"""

from . import dgps, causaldata
from causalis.data.dgps import (
    generate_rct, generate_rct_data,
    obs_linear_effect, obs_linear_effect_data,
    make_gold_linear, SmokingDGP, obs_linear_26_dataset,
    CausalDatasetGenerator
)
from causalis.data.causaldata import CausalData

__all__ = [
    "dgps", "causaldata",
    "generate_rct", "generate_rct_data",
    "obs_linear_effect", "obs_linear_effect_data",
    "make_gold_linear", "SmokingDGP", "obs_linear_26_dataset",
    "CausalData", "CausalDatasetGenerator"
]

