"""Validated fixed-horizon trial data for artificial-censoring IPCW."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Hashable

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_validator,
)


class IPCWData(BaseModel):
    """A single baseline trial with complete rows through a common horizon.

    Features are selected explicitly or from the ``model_features`` manifest
    attached by :func:`causalis.dgp.organic_activation.make_ipcw_view`. Baseline
    features are frozen within client. Censoring features need only be finite
    on at-risk rows, and outcomes need only be finite on usable rows. Rows at
    and after artificial censoring remain in the contract.

    Construction copies only the declared columns. :meth:`get_df` always
    returns an independent copy of the validated snapshot, even if callers
    subsequently mutate the source frame or the public ``df`` attribute.
    Units retain their first-seen order and rows are ordered by follow-up.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    df: pd.DataFrame
    outcome: str = "ltv_monthly"
    treatment: str = "baseline_d"
    unit_col: str = "client_id"
    trial_col: str = "trial_id"
    time_col: str = "period"
    followup_col: str = "followup_month"
    at_risk_col: str = "at_risk_of_censoring"
    censor_col: str = "censor_now"
    outcome_usable_col: str = "outcome_usable"
    baseline_covariates: list[str] | None = None
    censoring_covariates: list[str] | None = None

    _df_validated: pd.DataFrame | None = PrivateAttr(default=None)
    _horizon: int = PrivateAttr(default=0)
    _trial_id: Any = PrivateAttr(default=None)
    _n_units: int = PrivateAttr(default=0)

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs: Any) -> IPCWData:
        """Construct a trial, optionally overriding any role or feature list."""
        return cls(df=df, **kwargs)

    @field_validator(
        "outcome",
        "treatment",
        "unit_col",
        "trial_col",
        "time_col",
        "followup_col",
        "at_risk_col",
        "censor_col",
        "outcome_usable_col",
    )
    @classmethod
    def _nonempty_role(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Role column names must be non-empty strings.")
        return value

    @field_validator("baseline_covariates", "censoring_covariates", mode="before")
    @classmethod
    def _feature_names(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(name, str) or not name.strip() for name in value
        ):
            raise ValueError("Covariates must be a list of non-empty column names.")
        return list(dict.fromkeys(value))

    @staticmethod
    def _finite_values(series: pd.Series, name: str) -> None:
        if pd.api.types.is_complex_dtype(series.dtype):
            raise ValueError(
                f"Column '{name}' must contain finite real numeric values."
            )
        try:
            values = series.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column '{name}' must contain finite real numeric values."
            ) from exc
        if not np.isfinite(values).all():
            raise ValueError(
                f"Column '{name}' must contain finite real numeric values."
            )

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> IPCWData:
        df = self.df
        if df.empty:
            raise ValueError("IPCWData requires at least one trial row.")
        if not df.columns.is_unique:
            raise ValueError("DataFrame must have unique column names.")

        manifest = df.attrs.get("model_features", {})
        for field, key in (
            ("baseline_covariates", "baseline"),
            ("censoring_covariates", "censoring"),
        ):
            if getattr(self, field) is None:
                if not isinstance(manifest, dict) or key not in manifest:
                    raise ValueError(
                        f"Specify {field} or provide df.attrs['model_features']['{key}']."
                    )
                names = self._feature_names(manifest[key])
                if names is None:
                    raise ValueError(f"The {key} feature manifest must contain a list.")
                setattr(self, field, names)

        roles = [
            self.unit_col,
            self.trial_col,
            self.time_col,
            self.followup_col,
            self.treatment,
            self.at_risk_col,
            self.censor_col,
            self.outcome_usable_col,
            self.outcome,
        ]
        if len(set(roles)) != len(roles):
            raise ValueError("IPCW role columns must be distinct.")
        baseline = list(self.baseline_covariates or [])
        censoring = list(self.censoring_covariates or [])
        protected = {
            self.outcome,
            self.treatment,
            self.unit_col,
            self.trial_col,
            self.at_risk_col,
            self.censor_col,
            self.outcome_usable_col,
        }
        if protected.intersection(baseline + censoring):
            raise ValueError(
                "Outcome, treatment, ID and censoring-indicator roles cannot be predictors."
            )
        columns = list(dict.fromkeys([*roles, *baseline, *censoring]))
        missing = [name for name in columns if name not in df]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")

        for name in (self.unit_col, self.trial_col, self.time_col):
            if df[name].isna().any():
                raise ValueError(
                    f"Column '{name}' must not contain missing identifiers."
                )
            try:
                for value in pd.unique(df[name]):
                    hash(value)
            except TypeError as exc:
                raise ValueError(
                    f"Column '{name}' must contain hashable identifiers."
                ) from exc
        trials = pd.unique(df[self.trial_col])
        if len(trials) != 1:
            raise ValueError("IPCWData supports exactly one trial per fit.")
        if df.duplicated([self.unit_col, self.trial_col, self.time_col]).any():
            raise ValueError("Duplicate unit × trial × time keys are not allowed.")

        followup = df[self.followup_col]
        if (
            not pd.api.types.is_integer_dtype(followup.dtype)
            or pd.api.types.is_bool_dtype(followup.dtype)
            or followup.isna().any()
            or followup.lt(0).any()
        ):
            raise ValueError("Follow-up values must be non-negative integers.")
        for name in (
            self.treatment,
            self.at_risk_col,
            self.censor_col,
            self.outcome_usable_col,
        ):
            if df[name].isna().any() or not df[name].isin([0, 1, False, True]).all():
                raise ValueError(
                    f"Column '{name}' must contain non-missing binary values."
                )

        # Factorization gives a stable order even for heterogeneous client IDs.
        codes, units = pd.factorize(df[self.unit_col], sort=False)
        order = np.lexsort((followup.to_numpy(dtype=np.int64), codes))
        out = df.loc[:, columns].iloc[order].copy(deep=True).reset_index(drop=True)
        horizon = int(followup.max()) + 1
        n_units = len(units)
        expected = np.tile(np.arange(horizon), n_units)
        if len(out) != n_units * horizon or not np.array_equal(
            out[self.followup_col].to_numpy(dtype=np.int64), expected
        ):
            raise ValueError(
                "Every unit must have a complete common follow-up grid starting at zero."
            )
        calendar = out.groupby(self.followup_col, sort=True)[self.time_col]
        if not calendar.nunique(dropna=False).eq(1).all():
            raise ValueError(
                "Calendar time must be common to all units at each follow-up."
            )
        times = pd.Index(calendar.first().to_numpy())
        if not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("Calendar time must increase with follow-up.")

        d = out[self.treatment].to_numpy(dtype=np.int8).reshape(n_units, horizon)
        if not np.equal(d, d[:, :1]).all():
            raise ValueError("Baseline treatment must be constant within unit.")
        if set(np.unique(d[:, 0])) != {0, 1}:
            raise ValueError(
                "IPCWData requires both treated and control baseline units."
            )
        for name in baseline:
            self._finite_values(out[name], name)
            if (
                not out.groupby(self.unit_col, sort=False)[name]
                .nunique(dropna=False)
                .eq(1)
                .all()
            ):
                raise ValueError(
                    f"Baseline covariate '{name}' must be constant within unit."
                )

        censor = out[self.censor_col].to_numpy(dtype=bool).reshape(n_units, horizon)
        if (
            censor[:, 0].any()
            or censor[d[:, 0] == 1].any()
            or (censor.sum(axis=1) > 1).any()
        ):
            raise ValueError(
                "Only controls may be censored, at most once and strictly after baseline."
            )
        occurred = np.cumsum(censor, axis=1)
        expected_risk = (
            (d == 0) & (np.arange(horizon)[None, :] > 0) & ((occurred - censor) == 0)
        )
        expected_usable = (d == 1) | (occurred == 0)
        risk = out[self.at_risk_col].to_numpy(dtype=bool).reshape(n_units, horizon)
        usable = (
            out[self.outcome_usable_col].to_numpy(dtype=bool).reshape(n_units, horizon)
        )
        if not np.array_equal(risk, expected_risk):
            raise ValueError(
                "At-risk indicators must include control follow-up through the censoring event only."
            )
        if not np.array_equal(usable, expected_usable):
            raise ValueError(
                "Outcomes must be usable before censoring and unusable from its event onward."
            )
        self._finite_values(out.loc[usable.ravel(), self.outcome], self.outcome)
        for name in censoring:
            self._finite_values(out.loc[risk.ravel(), name], name)

        out[self.treatment] = d.ravel().astype(np.int8)
        out[self.followup_col] = expected.astype(np.int64)
        for name in (self.at_risk_col, self.censor_col, self.outcome_usable_col):
            out[name] = out[name].astype(bool)
        out.attrs = deepcopy(df.attrs)
        self.df = out
        self._df_validated = out.copy(deep=True)
        self._horizon = horizon
        self._trial_id = trials[0]
        self._n_units = n_units
        return self

    def get_df(self) -> pd.DataFrame:
        """Return an independent copy of the validated analysis rows."""
        if self._df_validated is None:
            raise RuntimeError("IPCWData has no validated data snapshot.")
        return self._df_validated.copy(deep=True)

    @property
    def horizon(self) -> int:
        """Number of follow-up periods, including baseline month zero."""
        return self._horizon

    @property
    def trial_id(self) -> Hashable:
        """The single trial identifier."""
        return self._trial_id

    @property
    def n_units(self) -> int:
        """Number of distinct baseline units."""
        return self._n_units


__all__ = ["IPCWData"]
