from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from causalis.data_contracts.causal_diagnostic_data import DiagnosticData


class MultiCausalEstimate(BaseModel):
    """
    Result container for causal effect estimates.

    Parameters
    ----------
    estimand : str
        The estimand being estimated (e.g., 'ATE', 'ATTE', 'CATE').
    model : str
        The name of the model used for estimation.
    model_options : dict
        Options passed to the model.
    value : float
        The estimated absolute effect.
    ci_upper_absolute : float
        Upper bound of the absolute confidence interval.
    ci_lower_absolute : float
        Lower bound of the absolute confidence interval.
    value_relative : float, optional
        The estimated relative effect.
    ci_upper_relative : float, optional
        Upper bound of the relative confidence interval.
    ci_lower_relative : float, optional
        Lower bound of the relative confidence interval.
    alpha : float
        The significance level (e.g., 0.05).
    p_value : float, optional
        The p-value from the test.
    is_significant : bool
        Whether the result is statistically significant at alpha.
    n_treated : int
        Number of units in the treatment group.
    n_control : int
        Number of units in the control group.
    outcome : str
        The name of the outcome variable.
    treatment : str
        The name of the treatment variable.
    confounders : list of str, optional
        The names of the confounders used in the model.
    time : datetime
        The date and time when the estimate was created.
    diagnostic_data : DiagnosticData, optional
        Additional diagnostic data_contracts.
    sensitivity_analysis : dict, optional
        Results from sensitivity analysis.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimand: str
    model: str
    model_options: Dict[str, Any] = Field(default_factory=dict)
    value: np.ndarray
    ci_upper_absolute: np.ndarray
    ci_lower_absolute: np.ndarray
    value_relative: Optional[np.ndarray] = None
    ci_upper_relative: Optional[np.ndarray] = None
    ci_lower_relative: Optional[np.ndarray] = None
    alpha: float
    p_value: Optional[np.ndarray] = None
    is_significant: List[bool]
    n_treated: int
    n_control: int
    outcome: str
    treatment: List[str]
    confounders: List[str] = Field(default_factory=list)
    time: datetime = Field(default_factory=datetime.now)
    diagnostic_data: Optional[DiagnosticData] = None
    sensitivity_analysis: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of the results.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame.
        """
        return pd.DataFrame(
            {
                "estimand": [self.estimand],
                "coefficient": [self.value],
                "p_val": [self.p_value],
                "lower_ci": [self.ci_lower_absolute],
                "upper_ci": [self.ci_upper_absolute],
                "relative_diff_%": [self.value_relative],
                "is_significant": [self.is_significant],
            }
        )
