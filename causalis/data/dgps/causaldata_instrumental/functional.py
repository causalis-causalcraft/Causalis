from __future__ import annotations
import pandas as pd
from .base import InstrumentalGenerator

def generate_iv_data(n: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic dataset with instrumental variables.

    Placeholder implementation.

    Parameters
    ----------
    n : int, default=1000
        Number of samples to generate.

    Returns
    -------
    pandas.DataFrame
        Synthetic IV dataset.
    """
    gen = InstrumentalGenerator()
    return gen.generate(n)
