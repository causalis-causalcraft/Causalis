from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class DiagnosticData(BaseModel):
    """Base class for all diagnostic data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class UnconfoundednessDiagnosticData(DiagnosticData):
    """Fields common to all models assuming unconfoundedness."""

    m_hat: np.ndarray  # Propensity scores
    d: np.ndarray  # Treatment indicators
    y: Optional[np.ndarray] = None  # Outcomes
    x: Optional[np.ndarray] = None  # Confounders (for balance checks)
    trimming_threshold: float = 0.0
