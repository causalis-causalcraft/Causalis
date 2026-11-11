"""Result and diagnostic contracts for fixed-horizon IPCW estimates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Hashable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .causal_diagnostic_data import DiagnosticData


class IPCWDiagnosticData(DiagnosticData):
    """Keyed nuisance predictions, support and weights retained by IPCW.

    ``row_data`` is keyed by unit and follow-up period; ``baseline_data`` has
    one row per unit. The estimator supplies normalized key names so reports
    can consume results independently of the source role column names.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    row_data: pd.DataFrame = Field(default_factory=pd.DataFrame)
    baseline_data: pd.DataFrame = Field(default_factory=pd.DataFrame)
    support: pd.DataFrame = Field(default_factory=pd.DataFrame)
    weight_summary: pd.DataFrame = Field(default_factory=pd.DataFrame)

    @model_validator(mode="after")
    def _copy_frames(self) -> IPCWDiagnosticData:
        for name in ("row_data", "baseline_data", "support", "weight_summary"):
            setattr(self, name, getattr(self, name).copy(deep=True))
        return self


class IPCWEstimate(BaseModel):
    """ATT for the sum of outcomes over a single trial's fixed horizon.

    ``treatment_mean`` is the observed treated horizon-total mean;
    ``control_mean`` is the weighted no-activation counterfactual total for
    that target population. Their difference is ``value``. Inference fields
    are ``None`` when bootstrap inference was not requested. Monthly and
    cumulative effect tables retain the same follow-up indexing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    estimand: Literal["ATT_no_activation"] = "ATT_no_activation"
    model: str = "IPCW"
    value: float
    std_error: float | None = None
    ci_lower_absolute: float | None = None
    ci_upper_absolute: float | None = None
    p_value: float | None = None
    is_significant: bool | None = None
    alpha: float = 0.05
    n_treated: int
    n_control: int
    horizon: int
    trial_id: Hashable
    treatment_mean: float
    control_mean: float
    outcome: str
    treatment: str
    baseline_covariates: list[str] = Field(default_factory=list)
    censoring_covariates: list[str] = Field(default_factory=list)
    model_options: dict[str, Any] = Field(default_factory=dict)
    period_effects: pd.DataFrame = Field(default_factory=pd.DataFrame)
    cumulative_effects: pd.DataFrame = Field(default_factory=pd.DataFrame)
    diagnostic_data: IPCWDiagnosticData | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_fields(self) -> IPCWEstimate:
        for name in ("value", "treatment_mean", "control_mean", "alpha"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if self.n_treated <= 0 or self.n_control <= 0 or self.horizon <= 0:
            raise ValueError("Group counts and horizon must be positive.")
        if (
            not self.model.strip()
            or not self.outcome.strip()
            or not self.treatment.strip()
        ):
            raise ValueError("Model and outcome/treatment names must be non-empty.")
        for name in ("baseline_covariates", "censoring_covariates"):
            if any(not value.strip() for value in getattr(self, name)):
                raise ValueError(f"{name} must contain non-empty column names.")
        for name in ("std_error", "ci_lower_absolute", "ci_upper_absolute", "p_value"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when supplied.")
        if self.std_error is not None and self.std_error < 0.0:
            raise ValueError("std_error must be non-negative.")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ValueError("p_value must be in [0, 1].")
        if (self.ci_lower_absolute is None) != (self.ci_upper_absolute is None):
            raise ValueError(
                "Both confidence-interval bounds must be supplied together."
            )
        if (
            self.ci_lower_absolute is not None
            and self.ci_lower_absolute > self.ci_upper_absolute
        ):
            raise ValueError("Confidence-interval bounds must be ordered.")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware.")
        self.period_effects = self.period_effects.copy(deep=True)
        self.cumulative_effects = self.cumulative_effects.copy(deep=True)
        return self

    @property
    def att(self) -> float:
        """Alias for the horizon-total ATT."""
        return self.value

    @property
    def se(self) -> float | None:
        """Alias for the bootstrap standard error."""
        return self.std_error

    @property
    def ci_lower(self) -> float | None:
        return self.ci_lower_absolute

    @property
    def ci_upper(self) -> float | None:
        return self.ci_upper_absolute

    def summary(self) -> pd.DataFrame:
        """Return scalar ATT, uncertainty and trial metadata as a table."""
        value = f"{self.value:.4f}"
        if self.ci_lower_absolute is not None:
            value += (
                f" (ci_abs: {self.ci_lower_absolute:.4f}, {self.ci_upper_absolute:.4f})"
            )
        fields = {
            "outcome": self.outcome,
            "estimand": self.estimand,
            "model": self.model,
            "value": value,
            "std_error": self.std_error,
            "alpha": self.alpha,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "trial_id": self.trial_id,
            "horizon": self.horizon,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "treatment_mean": self.treatment_mean,
            "control_mean": self.control_mean,
            "time": self.created_at.strftime("%Y-%m-%d"),
        }
        return pd.DataFrame(
            {"field": list(fields), "value": list(fields.values())}
        ).set_index("field")


__all__ = ["IPCWDiagnosticData", "IPCWEstimate"]
