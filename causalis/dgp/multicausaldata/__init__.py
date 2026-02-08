from causalis.data_contracts.multicausaldata import MultiCausalData
from .base import MultiCausalDatasetGenerator
from .functional import generate_multitreatment
def generate_multitreatment_irm_26(*args, **kwargs):
    # Lazy import to avoid circular dependency at module import time.
    from causalis.scenarios.multi_uncofoundedness.dgp import (
        generate_multitreatment_irm_26 as _generate_multitreatment_irm_26,
    )

    return _generate_multitreatment_irm_26(*args, **kwargs)

__all__ = [
    "MultiCausalData",
    "MultiCausalDatasetGenerator",
    "generate_multitreatment",
    "generate_multitreatment_irm_26",
]
