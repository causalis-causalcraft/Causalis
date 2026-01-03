"""
Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.
"""

from __future__ import annotations
import pandas as pd
import pandas.api.types as pdtypes
from typing import Union, List, Optional, Any
import warnings
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class CausalData(BaseModel):
    """
    Container for causal inference datasets.

    Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
    The stored DataFrame is restricted to only those columns.
    Uses Pydantic for validation and as a data contract.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
        Only columns specified in outcome, treatment, and confounders will be stored.
        NaN values are not allowed in the stored/used columns.
    treatment : str
        Column name representing the treatment variable.
    outcome : str
        Column name representing the outcome variable.
    confounders : Union[str, List[str]], optional
        Column name(s) representing the confounders/covariates.
    user_id : str, optional
        Column name representing the unique identifier for each observation/user.

    Attributes
    ----------
    df : pd.DataFrame
        A copy of the original data restricted to [outcome, treatment] + confounders [+ user_id].
    treatment : pd.Series
        The treatment column as a pandas Series.
    outcome : pd.Series
        The outcome column as a pandas Series.
    confounders : list[str]
        Names of the confounder columns (may be empty).
    user_id : pd.Series, optional
        The user_id column as a pandas Series (if specified).
    X : pd.DataFrame
        Design matrix (baseline covariates) used for modeling: df[confounders].
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    df: pd.DataFrame
    treatment_name: str = Field(alias="treatment")
    outcome_name: str = Field(alias="outcome")
    confounders_names: List[str] = Field(alias="confounders", default_factory=list)
    user_id_name: Optional[str] = Field(alias="user_id", default=None)
    instrument_name: Optional[str] = Field(alias="instrument", default=None)

    @classmethod
    def from_df(
            cls,
            df: pd.DataFrame,
            treatment: str,
            outcome: str,
            confounders: Optional[Union[str, List[str]]] = None,
            user_id: Optional[str] = None,
            instrument: Optional[str] = None,
            **kwargs: Any
    ) -> 'CausalData':
        """
        Friendly constructor for CausalData.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data.
        treatment : str
            Column name representing the treatment variable.
        outcome : str
            Column name representing the outcome  variable.
        confounders : Union[str, List[str]], optional
            Column name(s) representing the confounders/covariates.
        user_id : str, optional
            Column name representing the unique identifier for each observation/user.
        instrument : str, optional
            Column name representing the instrumental variable.
        **kwargs : Any
            Additional arguments passed to the Pydantic model constructor.

        Returns
        -------
        CausalData
            A validated CausalData instance.
        """
        return cls(
            df=df,
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            user_id=user_id,
            instrument=instrument,
            **kwargs
        )

    @field_validator("confounders_names", mode="before")
    @classmethod
    def _normalize_confounders(cls, v: Any) -> List[str]:
        """
        Normalize confounders to a list of unique strings.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, str):
                    raise TypeError(f"All confounder names must be strings. Found {type(item).__name__}: {item}")
            seen = set()
            unique_confs = []
            for c in v:
                if c not in seen:
                    unique_confs.append(c)
                    seen.add(c)
            return unique_confs
        raise TypeError("confounders must be None, a string, or a list of strings.")

    @model_validator(mode='after')
    def _validate_and_normalize(self) -> 'CausalData':
        """
        Perform complex validation and normalize the stored DataFrame.
        """
        df = self.df
        treatment = self.treatment_name
        outcome = self.outcome_name
        confounders = self.confounders_names
        user_id = self.user_id_name
        instrument = self.instrument_name

        # 0. Guard against duplicate column names
        if df.columns.has_duplicates:
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"DataFrame has duplicate column names: {dupes}. This is not supported.")

        # 1. Disjoint role validation
        roles = {
            "outcome": outcome,
            "treatment": treatment,
        }
        if user_id:
            roles["user_id"] = user_id
        if instrument:
            roles["instrument"] = instrument

        # Check for overlaps between primary roles
        role_names = list(roles.keys())
        for i, r1 in enumerate(role_names):
            for r2 in role_names[i+1:]:
                if roles[r1] == roles[r2]:
                    raise ValueError(f"Column '{roles[r1]}' cannot be both {r1} and {r2}.")

        overlap = [c for c in confounders if c in set(roles.values())]
        if overlap:
            raise ValueError(
                "confounder columns must be disjoint from treatment/outcome/user_id/instrument; overlapping columns: "
                + ", ".join(overlap)
            )

        # 2. Check if all specified columns exist in df
        all_columns = set(df.columns)
        for role_name, col in roles.items():
            if col not in all_columns:
                raise ValueError(f"Column '{col}' specified as {role_name} does not exist in the DataFrame.")
        
        for col in confounders:
            if col not in all_columns:
                raise ValueError(f"Column '{col}' specified as confounders does not exist in the DataFrame.")

        # 3. Validate types and constant variance
        # Outcome
        if not (pdtypes.is_numeric_dtype(df[outcome]) or pdtypes.is_bool_dtype(df[outcome])):
            raise ValueError(f"Column '{outcome}' specified as outcome must contain only int, float, or bool values.")
        if df[outcome].nunique(dropna=False) <= 1:
            raise ValueError(
                f"Column '{outcome}' specified as outcome is constant (has zero variance / single unique value), "
                f"which is not allowed for causal inference."
            )

        # Treatment
        if not (pdtypes.is_numeric_dtype(df[treatment]) or pdtypes.is_bool_dtype(df[treatment])):
            raise ValueError(f"Column '{treatment}' specified as treatment must contain only int, float, or bool values.")
        if df[treatment].nunique(dropna=False) <= 1:
            raise ValueError(
                f"Column '{treatment}' specified as treatment is constant (has zero variance / single unique value), "
                f"which is not allowed for causal inference."
            )

        # Instrument
        if instrument:
            if not (pdtypes.is_numeric_dtype(df[instrument]) or pdtypes.is_bool_dtype(df[instrument])):
                raise ValueError(f"Column '{instrument}' specified as instrument must contain only int, float, or bool values.")
            if df[instrument].nunique(dropna=False) <= 1:
                raise ValueError(
                    f"Column '{instrument}' specified as instrument is constant (has zero variance / single unique value), "
                    f"which is not allowed for causal inference."
                )

        # confounders
        kept_confounders: List[str] = []
        dropped_constants: List[str] = []
        for col in confounders:
            if not (pdtypes.is_numeric_dtype(df[col]) or pdtypes.is_bool_dtype(df[col])):
                raise ValueError(f"Column '{col}' specified as confounders must contain only int, float, or bool values.")
            
            if df[col].nunique(dropna=False) <= 1:
                dropped_constants.append(col)
            else:
                kept_confounders.append(col)

        if dropped_constants:
            warnings.warn(
                "Dropping constant confounder columns (zero variance): " + ", ".join(dropped_constants),
                UserWarning,
                stacklevel=2,
            )
        # Update confounders names
        self.confounders_names = kept_confounders
        confounders = kept_confounders # Update local variable for next steps

        # 4. Check for NaN values in used columns
        cols_to_check = [outcome, treatment] + confounders
        if user_id:
            cols_to_check.append(user_id)
        if instrument:
            cols_to_check.append(instrument)
        
        # Unique columns preserving order
        cols_to_check = list(dict.fromkeys(cols_to_check))
        
        if df[cols_to_check].isna().any().any():
            raise ValueError("DataFrame contains NaN values in used columns, which are not allowed.")

        # 5. Store only the relevant columns and coerce booleans to int8 (except user_id)
        self.df = df[cols_to_check].copy()
        for col in self.df.columns:
            if col != user_id and pdtypes.is_bool_dtype(self.df[col]):
                self.df[col] = self.df[col].astype("int8")
            
            # Final safeguard for numeric types (excluding user_id)
            if col != user_id:
                if not pdtypes.is_numeric_dtype(self.df[col]):
                    raise ValueError(
                        f"All columns in stored DataFrame must be numeric; column '{col}' has dtype {self.df[col].dtype}."
                    )

        # 6. Check for duplicate column values and duplicate rows on the normalized subset
        self._check_duplicate_column_values(self.df)
        self._check_duplicate_rows(self.df)
        self._check_user_id_uniqueness(self.df)

        return self

    def _check_user_id_uniqueness(self, df: pd.DataFrame):
        """Check if user_id column has unique values."""
        if self.user_id_name and df[self.user_id_name].duplicated().any():
            raise ValueError(f"Column '{self.user_id_name}' specified as user_id contains duplicate values.")

    def _check_duplicate_column_values(self, df: pd.DataFrame):
        """Check for identical values in different columns."""
        cols = [self.outcome_name, self.treatment_name] + self.confounders_names
        if self.user_id_name:
            cols.append(self.user_id_name)
        if self.instrument_name:
            cols.append(self.instrument_name)
        
        # Unique columns preserving order
        cols = list(dict.fromkeys(cols))

        for i, col1 in enumerate(cols):
            for j in range(i + 1, len(cols)):
                col2 = cols[j]
                if df[col1].equals(df[col2]):
                    col1_role = self._get_column_type(col1)
                    col2_role = self._get_column_type(col2)
                    raise ValueError(
                        f"Columns '{col1}' ({col1_role}) and '{col2}' ({col2_role}) have identical values, "
                        f"which is not allowed for causal inference. Only column names differ."
                    )

    def _check_duplicate_rows(self, df: pd.DataFrame):
        """Check for duplicate rows and issue a warning."""
        num_duplicates = int(df.duplicated().sum())
        if num_duplicates > 0:
            total_rows = len(df)
            unique_rows = total_rows - num_duplicates
            warnings.warn(
                f"Found {num_duplicates} duplicate rows out of {total_rows} total rows in the DataFrame. "
                f"This leaves {unique_rows} unique rows for analysis. "
                f"Duplicate rows may affect the quality of causal inference results. "
                f"Consider removing duplicates if they are not intentional.",
                UserWarning,
                stacklevel=2
            )

    def _get_column_type(self, column_name: str) -> str:
        """Determine the type/role of a column."""
        if column_name == self.outcome_name:
            return "outcome"
        elif column_name == self.treatment_name:
            return "treatment"
        elif self.user_id_name and column_name == self.user_id_name:
            return "user_id"
        elif self.instrument_name and column_name == self.instrument_name:
            return "instrument"
        elif column_name in self.confounders_names:
            return "confounder"
        return "unknown"

    @property
    def outcome(self) -> pd.Series:
        """Outcome column as a Series."""
        if not self.outcome_name or self.outcome_name not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df[self.outcome_name]

    @property
    def treatment(self) -> pd.Series:
        """Treatment column as a Series."""
        if not self.treatment_name or self.treatment_name not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df[self.treatment_name]

    @property
    def confounders(self) -> List[str]:
        """List of confounder column names."""
        return list(self.confounders_names)

    @property
    def user_id(self) -> pd.Series:
        """user_id column as a Series."""
        if not self.user_id_name or self.user_id_name not in self.df.columns:
            return pd.Series(dtype=object)
        return self.df[self.user_id_name]

    @property
    def instrument(self) -> pd.Series:
        """instrument column as a Series."""
        if not self.instrument_name or self.instrument_name not in self.df.columns:
            return pd.Series(dtype=float)
        return self.df[self.instrument_name]

    @property
    def X(self) -> pd.DataFrame:
        """Design matrix of confounders."""
        if not self.confounders_names:
            return self.df.iloc[:, 0:0].copy()
        return self.df[self.confounders_names].copy()

    def get_df(
            self,
            columns: Optional[List[str]] = None,
            include_treatment: bool = True,
            include_outcome: bool = True,
            include_confounders: bool = True,
            include_user_id: bool = False,
            include_instrument: bool = False
    ) -> pd.DataFrame:
        """Get a DataFrame with specified columns."""
        cols_to_include = []
        if columns is not None:
            cols_to_include.extend(columns)

        if columns is None and not any([include_outcome, include_confounders, include_treatment, include_user_id, include_instrument]):
            return self.df.copy()

        if include_outcome:
            cols_to_include.append(self.outcome_name)
        if include_confounders:
            cols_to_include.extend(self.confounders_names)
        if include_treatment:
            cols_to_include.append(self.treatment_name)
        if include_user_id and self.user_id_name:
            cols_to_include.append(self.user_id_name)
        if include_instrument and self.instrument_name:
            cols_to_include.append(self.instrument_name)

        # Remove duplicates while preserving order
        seen = set()
        cols_to_include = [x for x in cols_to_include if not (x in seen or seen.add(x))]

        # Validate existence
        missing = [c for c in cols_to_include if c not in self.df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} do not exist in the DataFrame.")

        return self.df[cols_to_include].copy()

    def __repr__(self) -> str:
        res = (
            f"CausalData(df={self.df.shape}, "
            f"treatment='{self.treatment_name}', "
            f"outcome='{self.outcome_name}', "
            f"confounders={self.confounders_names}"
        )
        if self.user_id_name:
            res += f", user_id='{self.user_id_name}'"
        if self.instrument_name:
            res += f", instrument='{self.instrument_name}'"
        res += ")"
        return res