"""
Refutation and robustness utilities for Causalis.

Importing this package exposes the public functions from all refutation
submodules (overlap, score, uncofoundedness) so you can access
commonly used helpers directly via `causalis.refutation`.
"""

from . import overlap, score, uncofoundedness

# Re-export public API from subpackages
# Overlap diagnostics (public API defined in overlap/__init__.py)
from .overlap import *  # noqa: F401,F403

# Score-based refutations and diagnostics
from .score.score_validation import *  # noqa: F401,F403

# Uncofoundedness sensitivity and balance checks
from .uncofoundedness.uncofoundedness_validation import *  # noqa: F401,F403
from .uncofoundedness.sensitivity import *  # noqa: F401,F403

# Best-effort __all__: include common high-level entry points for discoverability.
# Note: wildcard imports above already populate the module namespace.
try:
    from .overlap import __all__ as __all_overlap  # type: ignore
except Exception:
    __all_overlap = []

# score_validation and uncofoundedness modules don't define __all__; curate a minimal list
__all_score = [
    "refute_placebo_outcome",
    "refute_placebo_treatment",
    "refute_subset",
    "refute_irm_orthogonality",
    "influence_summary",
]

__all_unconf = [
    "sensitivity_analysis",
    "get_sensitivity_summary",
    "validate_uncofoundedness_balance",
    "sensitivity_benchmark",
]

__all__ = ["overlap", "score", "uncofoundedness"] + list(dict.fromkeys([*__all_overlap, *__all_score, *__all_unconf]))
