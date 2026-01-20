"""
Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference with multiple treatments.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import pandas.api.types as pdtypes
from typing import Union, List, Optional, Any, ClassVar
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class MultiCausalData(BaseModel):
    """
    Data contract for cross-sectional causal data with multiple binary treatment columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the causal data.
    outcome_name : str
        The name of the outcome column. (Alias: "outcome")
    treatment_names : List[str]
        The names of the treatment columns. (Alias: "treatments")
    confounders_names : List[str], optional
        The names of the confounder columns, by default []. (Alias: "confounders")
    user_id_name : Optional[str], optional
        The name of the user ID column, by default None. (Alias: "user_id")

    Notes
    -----
    This class enforces several constraints on the data, including:
    - Maximum number of treatments (default 5).
    - No duplicate column names in the input DataFrame.
    - Disjoint roles for columns (outcome, treatments, confounders, user_id).
    - Existence of all specified columns in the DataFrame.
    - Numeric or boolean types for outcome and confounders.
    - Non-constant values for outcome, treatments, and confounders.
    - No NaN values in used columns.
    - Binary (0/1) encoding for treatment columns.
    - No identical values between different columns.
    - Unique values for user_id (if specified).
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
    )

    # Hard constraints
    MAX_TREATMENTS: ClassVar[int] = 5
    FLOAT_TOL: ClassVar[float] = 1e-12  # for float 0/1 acceptance

    df: pd.DataFrame
    outcome_name: str = Field(alias="outcome")
    treatment_names: List[str] = Field(alias="treatments")
    confounders_names: List[str] = Field(alias="confounders", default_factory=list)
    user_id_name: Optional[str] = Field(alias="user_id", default=None)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        *,
        outcome: str,
        treatments: Union[str, List[str]],
        confounders: Optional[Union[str, List[str]]] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "MultiCausalData":
        """
        Create a MultiCausalData instance from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        outcome : str
            The name of the outcome column.
        treatments : Union[str, List[str]]
            The name(s) of the treatment column(s).
        confounders : Union[str, List[str]], optional
            The name(s) of the confounder column(s), by default None.
        user_id : str, optional
            The name of the user ID column, by default None.
        **kwargs : Any
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        MultiCausalData
            An instance of MultiCausalData.
        """
        return cls(
            df=df,
            outcome=outcome,
            treatments=treatments,
            confounders=confounders,
            user_id=user_id,
            **kwargs,
        )

    @field_validator("treatment_names", mode="before")
    @classmethod
    def _normalize_treatments(cls, v: Any) -> List[str]:
        """
        Normalize and validate treatment names.

        Parameters
        ----------
        v : Any
            The input treatment name(s).

        Returns
        -------
        List[str]
            A list of unique treatment names.

        Raises
        -------
        TypeError
            If input is not a string or list of strings.
        ValueError
            If treatments list is empty.
        """
        if v is None:
            raise TypeError("treatments must be a string or a list of strings (cannot be None).")
        if isinstance(v, str):
            out = [v]
        elif isinstance(v, list):
            for item in v:
                if not isinstance(item, str):
                    raise TypeError(f"All treatment names must be strings. Found {type(item).__name__}: {item}")
            seen = set()
            out = []
            for t in v:
                if t not in seen:
                    out.append(t)
                    seen.add(t)
        else:
            raise TypeError("treatments must be a string or a list of strings.")

        if not out:
            raise ValueError("treatments cannot be empty.")
        return out

    @field_validator("confounders_names", mode="before")
    @classmethod
    def _normalize_confounders(cls, v: Any) -> List[str]:
        """
        Normalize and validate confounder names.

        Parameters
        ----------
        v : Any
            The input confounder name(s).

        Returns
        -------
        List[str]
            A list of unique confounder names.

        Raises
        -------
        TypeError
            If input is not None, a string, or a list of strings.
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
            out = []
            for c in v:
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            return out
        raise TypeError("confounders must be None, a string, or a list of strings.")

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> "MultiCausalData":
        """
        Perform cross-field validation and data normalization.

        This method checks for:
        - Maximum number of treatments.
        - Disjoint roles for columns.
        - Existence of all columns in the DataFrame.
        - Correct data types.
        - Non-constant values.
        - Lack of NaN values.
        - Binary encoding for treatments.
        - Unique user IDs.

        Returns
        -------
        MultiCausalData
            The validated and normalized instance.

        Raises
        -------
        ValueError
            If any validation rule is violated.
        """
        df = self.df
        outcome = self.outcome_name
        treatments = self.treatment_names
        confounders = self.confounders_names
        user_id = self.user_id_name

        # 0) duplicate column names
        if df.columns.has_duplicates:
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"DataFrame has duplicate column names: {dupes}. This is not supported.")

        # 1) cap treatments
        if len(treatments) > self.MAX_TREATMENTS:
            raise ValueError(
                f"Too many treatment columns: {len(treatments)}. "
                f"Maximum allowed is {self.MAX_TREATMENTS}. Treatments={treatments}"
            )

        # 2) disjoint roles
        if outcome in set(treatments):
            raise ValueError(f"Column '{outcome}' cannot be both outcome and treatment.")
        if user_id and user_id == outcome:
            raise ValueError(f"Column '{user_id}' cannot be both user_id and outcome.")
        if user_id and user_id in set(treatments):
            raise ValueError(f"Column '{user_id}' cannot be both user_id and treatment.")

        overlap_confs = [c for c in confounders if (c == outcome or c in set(treatments) or (user_id and c == user_id))]
        if overlap_confs:
            raise ValueError(
                "confounder columns must be disjoint from outcome/treatments/user_id; overlapping columns: "
                + ", ".join(overlap_confs)
            )

        # 3) existence checks
        all_cols = set(df.columns)
        if outcome not in all_cols:
            raise ValueError(f"Column '{outcome}' specified as outcome does not exist in the DataFrame.")
        missing_t = [t for t in treatments if t not in all_cols]
        if missing_t:
            raise ValueError(f"Treatment column(s) {missing_t} do not exist in the DataFrame.")
        missing_x = [c for c in confounders if c not in all_cols]
        if missing_x:
            raise ValueError(f"Confounder column(s) {missing_x} do not exist in the DataFrame.")
        if user_id and user_id not in all_cols:
            raise ValueError(f"Column '{user_id}' specified as user_id does not exist in the DataFrame.")

        # 4) outcome type + non-constant
        if not (pdtypes.is_numeric_dtype(df[outcome]) or pdtypes.is_bool_dtype(df[outcome])):
            raise ValueError(f"Column '{outcome}' specified as outcome must contain only int, float, or bool values.")
        if df[outcome].nunique(dropna=False) <= 1:
            raise ValueError(f"Column '{outcome}' specified as outcome is constant (zero variance).")

        # 5) confounders type + non-constant
        for c in confounders:
            if not (pdtypes.is_numeric_dtype(df[c]) or pdtypes.is_bool_dtype(df[c])):
                raise ValueError(f"Column '{c}' specified as confounder must contain only int, float, or bool values.")
            if df[c].nunique(dropna=False) <= 1:
                raise ValueError(f"Column '{c}' specified as confounder is constant (zero variance).")

        # 6) NaNs disallowed in used cols
        cols_to_keep: List[str] = []
        if user_id:
            cols_to_keep.append(user_id)
        cols_to_keep.append(outcome)
        cols_to_keep.extend(confounders)
        cols_to_keep.extend(treatments)
        cols_to_keep = list(dict.fromkeys(cols_to_keep))

        if df[cols_to_keep].isna().any().any():
            raise ValueError("DataFrame contains NaN values in used columns, which are not allowed.")

        # 7) store subset
        self.df = df[cols_to_keep].copy()

        # 8) canonicalize bool -> int8 for all non-user columns
        for col in self.df.columns:
            if user_id and col == user_id:
                continue
            if pdtypes.is_bool_dtype(self.df[col]):
                self.df[col] = self.df[col].astype("int8")
            if not pdtypes.is_numeric_dtype(self.df[col]):
                raise ValueError(
                    f"All non-user_id columns in stored DataFrame must be numeric; column '{col}' has dtype {self.df[col].dtype}."
                )

        # 9) validate + canonicalize treatments as int8 binary
        for t in treatments:
            self.df[t] = self._validate_and_cast_binary_treatment(self.df[t], t)

        # 10) duplicate-column values check (dtype-agnostic)
        self._check_duplicate_column_values_dtype_agnostic(self.df)

        # 11) user_id uniqueness
        if user_id and self.df[user_id].duplicated().any():
            raise ValueError(f"Column '{user_id}' specified as user_id contains duplicate values.")

        return self

    def _validate_and_cast_binary_treatment(self, s: pd.Series, name: str) -> pd.Series:
        """
        Validate that a treatment column is binary and cast it to int8.

        Parameters
        ----------
        s : pd.Series
            The treatment column.
        name : str
            The name of the treatment column.

        Returns
        -------
        pd.Series
            The validated and casted treatment column.

        Raises
        -------
        ValueError
            If the column is not binary or is constant.
        """
        # Non-constant
        if s.nunique(dropna=False) <= 1:
            raise ValueError(f"Treatment column '{name}' is constant (zero variance).")

        # bool already converted to int8 earlier
        if pdtypes.is_integer_dtype(s):
            vals = set(pd.unique(s))
            if not vals.issubset({0, 1}):
                raise ValueError(
                    f"Treatment column '{name}' must be binary encoded in {{0,1}}. Found values: {sorted(vals)}"
                )
            return s.astype("int8")

        # Validates and converts float treatment columns to int8
        if pdtypes.is_float_dtype(s):
            arr = s.to_numpy(dtype=float, copy=False)
            is0 = np.isclose(arr, 0.0, atol=self.FLOAT_TOL, rtol=0.0)
            is1 = np.isclose(arr, 1.0, atol=self.FLOAT_TOL, rtol=0.0)
            if not np.all(is0 | is1):
                bad = np.unique(arr[~(is0 | is1)])
                bad_show = bad[:10]
                suffix = "" if bad.size <= 10 else f" ... (+{bad.size - 10} more)"
                raise ValueError(
                    f"Treatment column '{name}' must be binary (0/1). "
                    f"Found non-binary float values (up to 10 shown): {bad_show}{suffix}"
                )
            # canonicalize to int8
            return pd.Series(np.rint(arr).astype("int8"), index=s.index, name=s.name)

        raise ValueError(f"Treatment column '{name}' must be bool/int/float binary; dtype={s.dtype}.")

    def _check_duplicate_column_values_dtype_agnostic(self, df: pd.DataFrame) -> None:
        """
        Check for columns with identical values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check.

        Raises
        -------
        ValueError
            If any two columns have identical values.
        """
        cols: List[str] = [self.outcome_name] + self.confounders_names + self.treatment_names
        if self.user_id_name:
            cols.append(self.user_id_name)
        cols = list(dict.fromkeys(cols))

        def _eq(a: pd.Series, b: pd.Series) -> bool:
            return np.array_equal(
                a.to_numpy(dtype=object, copy=False),
                b.to_numpy(dtype=object, copy=False),
            )

        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                if _eq(df[c1], df[c2]):
                    raise ValueError(
                        f"Columns '{c1}' and '{c2}' have identical values, which is not allowed for causal inference."
                    )

    @property
    def outcome(self) -> pd.Series:
        """
        Return the outcome column as a pandas Series.

        Returns
        -------
        pd.Series
            The outcome column.
        """
        return self.df[self.outcome_name]

    @property
    def treatments(self) -> pd.DataFrame:
        """
        Return the treatment columns as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The treatment columns.
        """
        return self.df[self.treatment_names].copy()

    @property
    def treatment(self) -> pd.Series:
        """
        Return the single treatment column as a pandas Series.

        Returns
        -------
        pd.Series
            The treatment column.

        Raises
        -------
        AttributeError
            If there is more than one treatment column.
        """
        if len(self.treatment_names) != 1:
            raise AttributeError(
                f"MultiCausalData has {len(self.treatment_names)} treatments. "
                "Use `.treatments` or select one column explicitly."
            )
        return self.df[self.treatment_names[0]]

    @property
    def X(self) -> pd.DataFrame:
        """
        Return the confounder columns as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The confounder columns.
        """
        if not self.confounders_names:
            return self.df.iloc[:, 0:0].copy()
        return self.df[self.confounders_names].copy()

    def get_df(
        self,
        columns: Optional[List[str]] = None,
        include_outcome: bool = True,
        include_confounders: bool = True,
        include_treatments: bool = True,
        include_user_id: bool = False,
    ) -> pd.DataFrame:
        """
        Get a subset of the underlying DataFrame.

        Parameters
        ----------
        columns : List[str], optional
            Specific columns to include, by default None.
        include_outcome : bool, optional
            Whether to include the outcome column, by default True.
        include_confounders : bool, optional
            Whether to include confounder columns, by default True.
        include_treatments : bool, optional
            Whether to include treatment columns, by default True.
        include_user_id : bool, optional
            Whether to include the user ID column, by default False.

        Returns
        -------
        pd.DataFrame
            A copy of the requested DataFrame subset.

        Raises
        -------
        ValueError
            If any of the requested columns do not exist.
        """
        cols: List[str] = []
        if columns is not None:
            cols.extend(columns)

        if columns is None and not any([include_outcome, include_confounders, include_treatments, include_user_id]):
            return self.df.iloc[:, 0:0].copy()

        if include_outcome:
            cols.append(self.outcome_name)
        if include_confounders:
            cols.extend(self.confounders_names)
        if include_treatments:
            cols.extend(self.treatment_names)
        if include_user_id and self.user_id_name:
            cols.append(self.user_id_name)

        cols = list(dict.fromkeys(cols))
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} do not exist in the DataFrame.")
        return self.df[cols].copy()
