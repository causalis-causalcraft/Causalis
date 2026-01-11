"""
Causalis: A Python package for causal inference.
"""

import warnings

# Suppress noisy tqdm warning in environments without ipywidgets
try:
    from tqdm import TqdmWarning  # type: ignore
    # Apply more comprehensive filter
    warnings.filterwarnings(
        "ignore",
        message=".*IProgress not found.*",
        category=TqdmWarning,
    )
    # Also filter the exact message from the test
    warnings.filterwarnings(
        "ignore", 
        message="IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html",
        category=TqdmWarning,
    )
except Exception:
    # If tqdm is not installed or any issue arises, do not fail import
    pass

from causalis import data
# 'design' is optional; keep import non-fatal if missing in editable installs
try:
    from causalis import design  # type: ignore
except Exception:
    design = None  # type: ignore

__version__ = "0.1.2"
__all__ = ["data", "scenarios", "statistics", "eda", "refutation"]

# Lazily import heavy optional subpackages
from typing import TYPE_CHECKING
import importlib

def __getattr__(name):  # pragma: no cover - behavior tested via subprocess
    if name in ["scenarios", "statistics", "eda", "refutation"]:
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    
    # Compatibility mapping if needed, or just let it fail
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if TYPE_CHECKING:  # Hint for static type checkers without importing at runtime
    from . import scenarios as scenarios  # noqa: F401
    from . import statistics as statistics  # noqa: F401
    from . import eda as eda  # noqa: F401
    from .scenarios.unconfoundedness import refutation as refutation
