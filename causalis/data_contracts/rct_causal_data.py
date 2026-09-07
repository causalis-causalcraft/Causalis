"""RCT data with multiple outcomes and binary or one-hot treatment arms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdtypes
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .causaldata import CausalData
from .multicausaldata import MultiCausalData


class RctCausalData(BaseModel):
    """Validated data for multiple outcomes sharing treatment and confounders.

    ``outcomes`` accepts a column name or a non-empty list of names. Repeated
    names are removed in input order. All roles must be disjoint. Outcomes and
    confounders must be finite, real numeric or boolean, and non-constant;
    each treatment column must contain both 0 and 1. A single treatment column
    uses 0 as control. Multiple columns must encode mutually exclusive arms,
    including an explicit ``control_treatment`` column, with exactly one 1 per
    row. The control column is ordered first. This validates data structure,
    not whether treatment assignment was actually randomized.
    Used columns cannot contain missing
    values or identical column values. Optional user IDs must be unique.

    Construction stores an independent copy of only the selected columns,
    preserving the index and numeric dtypes (booleans and treatment become
    int8). Shared columns are validated once, regardless of outcome count.
    Missing/finite checks operate column by column, avoiding a table-sized
    boolean mask. Duplicate screening uses a small sample, then full-column
    fingerprints and exact comparisons for candidates only. Sampling never
    determines equality. There is no fixed limit on the number of outcomes.

    ``outcomes``/``Y``, ``treatments``/``D``, and ``X`` return independent
    DataFrames. ``treatment`` exposes a stored Series for a single treatment,
    or a copied DataFrame for multiple arms. ``user_id`` exposes a stored Series.
    Use :meth:`for_outcome` to create a validated single-outcome contract for
    existing estimators; that conversion copies and validates its subset.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid"
    )

    df: pd.DataFrame
    treatment_names: list[str] = Field(
        validation_alias=AliasChoices("treatment_names", "treatments", "treatment", "treatment_name")
    )
    control_treatment: str | None = None
    outcome_names: list[str] = Field(alias="outcomes")
    confounders_names: list[str] = Field(alias="confounders", default_factory=list)
    user_id_name: str | None = Field(alias="user_id", default=None)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        treatment: str | list[str] | None = None,
        outcomes: str | list[str] | None = None,
        confounders: str | list[str] | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> RctCausalData:
        """Construct from a DataFrame and role column names."""
        if treatment is not None:
            kwargs["treatment"] = treatment
        return cls(
            df=df, outcomes=outcomes,
            confounders=confounders, user_id=user_id, **kwargs,
        )

    @field_validator("outcome_names", "confounders_names", "treatment_names", mode="before")
    @classmethod
    def _normalize_names(cls, value: Any, info: Any) -> list[str]:
        if value is None and info.field_name == "confounders_names":
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ValueError(f"{info.field_name} must be a string or a list of strings.")
        if any(not v.strip() for v in value):
            raise ValueError(f"{info.field_name} must contain non-empty column names.")
        if not value and info.field_name != "confounders_names":
            raise ValueError(f"{info.field_name} must contain at least one column name.")
        if info.field_name == "treatment_names" and len(set(value)) != len(value):
            raise ValueError("treatment_names contains duplicate names.")
        return list(dict.fromkeys(value))

    @field_validator("control_treatment", "user_id_name")
    @classmethod
    def _nonempty_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Column names must be non-empty strings.")
        return value

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> RctCausalData:
        df = self.df
        if df.columns.has_duplicates:
            raise ValueError("DataFrame has duplicate column names.")

        if len(self.treatment_names) > 1:
            if self.control_treatment not in self.treatment_names:
                raise ValueError("control_treatment must name one of the treatment columns for multiple arms.")
            self.treatment_names = [self.control_treatment] + [
                name for name in self.treatment_names if name != self.control_treatment
            ]
        elif self.control_treatment is not None:
            raise ValueError("A single binary treatment uses 0 as control; omit control_treatment.")

        roles: dict[str, str] = {}
        pairs = [(name, "outcome") for name in self.outcome_names]
        pairs += [(name, "treatment") for name in self.treatment_names]
        pairs += [(name, "confounder") for name in self.confounders_names]
        if self.user_id_name is not None:
            pairs.insert(0, (self.user_id_name, "user_id"))
        for name, role in pairs:
            if name in roles:
                raise ValueError(f"Column '{name}' cannot be both {roles[name]} and {role}.")
            if name not in df.columns:
                raise ValueError(f"Column '{name}' specified as {role} does not exist in the DataFrame.")
            roles[name] = role

        # One row-sized counter avoids allocating a dense rows-by-arms matrix.
        arm_counts = np.zeros(len(df), dtype=np.int64) if len(self.treatment_names) > 1 else None
        for name, role in roles.items():
            series = df[name]
            if series.isna().any():
                raise ValueError(f"Column '{name}' contains NaN values, which are not allowed.")
            if role == "user_id":
                if series.duplicated().any():
                    raise ValueError(f"Column '{name}' specified as user_id contains duplicate values.")
                continue
            if (not pdtypes.is_numeric_dtype(series) or pdtypes.is_complex_dtype(series)):
                raise ValueError(f"Column '{name}' specified as {role} must contain only real int, float, or bool values.")
            if not np.isfinite(series).all():
                raise ValueError(f"Column '{name}' specified as {role} must contain only finite values.")
            if CausalData._is_constant_series(series):
                raise ValueError(f"Column '{name}' specified as {role} is constant (has zero variance).")
            if role == "treatment" and not ((series == 0) | (series == 1)).all():
                raise ValueError(f"Column '{name}' specified as treatment must be binary encoded in {{0,1}}.")
            if role == "treatment" and arm_counts is not None:
                arm_counts += series.to_numpy(dtype=np.int64, copy=False)

        if arm_counts is not None and not (arm_counts == 1).all():
            raise ValueError("Treatment columns must be one-hot encoded per row (exactly one active arm).")

        self._check_duplicate_values(df, roles)

        # Copy each selected column once; a selection followed by .copy() can
        # allocate two whole tables on pandas versions without copy-on-write.
        columns = {}
        for name, role in roles.items():
            series = df[name]
            if role == "treatment" or (role != "user_id" and pdtypes.is_bool_dtype(series)):
                columns[name] = series.astype("int8")
            else:
                columns[name] = series.copy(deep=True)
        self.df = pd.DataFrame(columns, index=df.index, copy=False)
        return self

    @staticmethod
    def _check_duplicate_values(df: pd.DataFrame, roles: dict[str, str]) -> None:
        # Different samples prove inequality. Matching samples require a full
        # fingerprint, and matching fingerprints still require exact equality.
        positions = np.linspace(0, len(df) - 1, min(len(df), 64), dtype=np.intp)
        samples: dict[tuple, list[str]] = {}
        for name in roles:
            signature = CausalData._column_value_signature(df[name].iloc[positions])
            samples.setdefault(signature, []).append(name)
        for candidates in samples.values():
            if len(candidates) < 2:
                continue
            groups = CausalData._column_value_signatures(df, candidates)
            for group in groups.values():
                for i, first in enumerate(group):
                    for second in group[i + 1:]:
                        # Object comparisons preserve exact mixed int/float
                        # equality, including integers above float64 precision.
                        equal = all(
                            np.array_equal(
                                df[first].iloc[start:start + 65536].to_numpy(dtype=object),
                                df[second].iloc[start:start + 65536].to_numpy(dtype=object),
                            )
                            for start in range(0, len(df), 65536)
                        )
                        if equal:
                            raise ValueError(
                                f"Columns '{first}' ({roles[first]}) and '{second}' ({roles[second]}) "
                                "have identical values, which is not allowed for causal inference."
                            )

    @property
    def outcomes(self) -> pd.DataFrame:
        """Outcome matrix in the requested order, as an independent copy."""
        return self._copy_columns(self.outcome_names)

    @property
    def Y(self) -> pd.DataFrame:
        """Alias for the two-dimensional outcome matrix, even for one outcome."""
        return self.outcomes

    @property
    def treatment(self) -> pd.Series | pd.DataFrame:
        """Binary Series for one column, or independent treatment-arm matrix."""
        if len(self.treatment_names) == 1:
            return self.df[self.treatment_names[0]]
        return self.treatments

    @property
    def treatment_name(self) -> str:
        """Single treatment name; use treatment_names for multiple arms."""
        if len(self.treatment_names) != 1:
            raise ValueError("Multiple treatment arms: use treatment_names.")
        return self.treatment_names[0]

    @property
    def treatments(self) -> pd.DataFrame:
        """Independent treatment matrix, with control first for multiple arms."""
        return self._copy_columns(self.treatment_names)

    @property
    def D(self) -> pd.DataFrame:
        """Alias for the two-dimensional treatment matrix."""
        return self.treatments

    @property
    def confounders(self) -> list[str]:
        """Confounder names in input order."""
        return list(self.confounders_names)

    @property
    def X(self) -> pd.DataFrame:
        """Independent confounder matrix, retaining the observation index."""
        return self._copy_columns(self.confounders_names)

    @property
    def user_id(self) -> pd.Series:
        """Stored user IDs, or an empty Series when no ID was specified."""
        if self.user_id_name is None:
            return pd.Series(dtype=object)
        return self.df[self.user_id_name]

    def _copy_columns(self, columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {name: self.df[name].copy(deep=True) for name in columns},
            index=self.df.index, copy=False,
        )

    def get_df(
        self,
        columns: list[str] | None = None,
        include_treatment: bool = True,
        include_outcomes: bool = True,
        include_confounders: bool = True,
        include_user_id: bool = False,
    ) -> pd.DataFrame:
        """Copy selected columns; explicit columns are additive to role flags."""
        selected = list(columns) if columns is not None else []
        if include_outcomes:
            selected.extend(self.outcome_names)
        if include_confounders:
            selected.extend(self.confounders_names)
        if include_treatment:
            selected.extend(self.treatment_names)
        if include_user_id and self.user_id_name is not None:
            selected.append(self.user_id_name)
        selected = list(dict.fromkeys(selected))
        missing = [name for name in selected if name not in self.df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} do not exist in the DataFrame.")
        return self._copy_columns(selected)

    def for_outcome(self, outcome: str) -> CausalData | MultiCausalData:
        """Create CausalData (binary) or MultiCausalData (multiple arms).

        The destination contract validates the copied subset, including its
        own constraints (such as MultiCausalData's treatment-count limit).
        """
        if outcome not in self.outcome_names:
            raise ValueError(f"Column '{outcome}' is not a declared outcome.")
        if len(self.treatment_names) > 1:
            return MultiCausalData.from_df(
                self.df, treatment_names=self.treatment_names, outcome=outcome,
                confounders=self.confounders_names, user_id=self.user_id_name,
                control_treatment=self.control_treatment,
            )
        return CausalData.from_df(
            self.df, treatment=self.treatment_name, outcome=outcome,
            confounders=self.confounders_names, user_id=self.user_id_name,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"treatments={self.treatment_names!r}, control_treatment={self.control_treatment!r}, "
            f"outcomes={self.outcome_names!r}, "
            f"confounders={self.confounders_names!r}, user_id={self.user_id_name!r})"
        )
