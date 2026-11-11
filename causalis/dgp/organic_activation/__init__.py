"""Organic staggered activation, LTV panel views and isolated oracle effects."""

from .base import (
    OrganicActivationConfig,
    OrganicActivationGenerator,
    generate_data,
    generate_organic_activation_data,
)
from .diagnostics import summarize_data, validate_data
from .views import make_did_pair, make_dml_view, make_ipcw_view

__all__ = [
    "OrganicActivationConfig",
    "OrganicActivationGenerator",
    "generate_data",
    "generate_organic_activation_data",
    "make_did_pair",
    "make_dml_view",
    "make_ipcw_view",
    "validate_data",
    "summarize_data",
]
