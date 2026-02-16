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

from causalis import data_contracts, dgp
# 'design' is optional; keep import non-fatal if missing in editable installs
try:
    from causalis import design  # type: ignore
except Exception:
    design = None  # type: ignore

__version__ = "0.1.2"
__all__ = ["data_contracts", "dgp", "scenarios", "shared"]

# Lazily import heavy optional subpackages
from typing import TYPE_CHECKING
import importlib

def __getattr__(name):  # pragma: no cover - behavior tested via subprocess
    if name in ["scenarios", "shared"]:
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    
    # Compatibility mapping
    if name == "data":
        warnings.warn("causalis.data is deprecated, use causalis.data_contracts instead", DeprecationWarning, stacklevel=2)
        return data_contracts

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if TYPE_CHECKING:  # Hint for static type checkers without importing at runtime
    from . import scenarios as scenarios  # noqa: F401
    from . import shared as shared  # noqa: F401
    from . import dgp as dgp  # noqa: F401
    from . import data_contracts as data_contracts  # noqa: F401
