from __future__ import annotations

from datetime import date, datetime
from typing import Hashable, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


TimeLike = Union[int, float, str, date, datetime]


class PanelDataSCM(BaseModel):
    """Long-format panel contract for Synthetic Control estimators.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with one row per ``(unit_id, time_id)``.
    unit_id : str
        Name of the unit identifier column in ``df``.
    time_id : str
        Name of the time column in ``df``.
    y : str
        Name of the outcome column in ``df``.
    treated_unit : Hashable
        Identifier of the treated unit.
    intervention_time : TimeLike
        First post-treatment boundary. Pre periods satisfy
        ``t < intervention_time`` and post periods satisfy
        ``t >= intervention_time``.

    Notes
    -----
    This contract supports both augmented/standard synthetic control workflows.
    It validates schema-level assumptions only; estimator-level requirements
    (for example panel balance) are typically enforced downstream.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---------- core data ----------
    df: pd.DataFrame = Field(..., description="Long-format panel data.")

    unit_col: str = Field(..., alias="unit_id", description="Unit id column name in df.")
    time_col: str = Field(..., alias="time_id", description="Time id column name in df.")
    outcome_col: str = Field(..., alias="y", description="Outcome column name in df.")

    # ---------- adoption spec (single treated unit) ----------
    treated_unit: Hashable = Field(..., description="ID of the treated unit.")
    intervention_time: TimeLike = Field(
        ...,
        description=(
            "First post-treatment time boundary. "
            "Pre: t < intervention_time. Post: t >= intervention_time."
        ),
    )

    # ---------- optional selectors ----------
    donor_units: Optional[Sequence[Hashable]] = Field(
        default=None,
        description=(
            "Optional explicit donor pool. If None, donors = all units except treated_unit."
        ),
    )

    # Choose ONE way to define the analysis window:
    time_window: Optional[Tuple[Optional[TimeLike], Optional[TimeLike]]] = Field(
        default=None,
        description="(t_min, t_max) inclusive bounds for time filtering. Use None for open end.",
    )
    pre_periods: Optional[Sequence[TimeLike]] = Field(
        default=None,
        description="Optional explicit list of pre-treatment times (overrides rule t < intervention_time).",
    )
    post_periods: Optional[Sequence[TimeLike]] = Field(
        default=None,
        description=(
            "Optional explicit list of post-treatment times "
            "(overrides rule t >= intervention_time)."
        ),
    )

    # ---------- optional extras ----------
    covariate_cols: Sequence[str] = Field(
        default_factory=tuple,
        description="Optional covariate columns (time-varying or invariant).",
    )

    # Missingness: either outcome is NaN, or you provide an explicit observed mask.
    observed_col: Optional[str] = Field(
        default=None,
        description="Optional boolean column: True if outcome observed for this row.",
    )

    # Optional row weights (rare but sometimes useful)
    weights_col: Optional[str] = Field(
        default=None,
        description="Optional non-negative row weights.",
    )

    # ---------- behavior flags ----------
    allow_missing_outcome: bool = Field(
        default=True,
        description=(
            "If False, contract enforces no missing outcome in df "
            "(useful for ASCM-only pipelines)."
        ),
    )
    allow_duplicate_unit_time: bool = Field(
        default=False,
        description="If False, requires uniqueness of (unit_col, time_col).",
    )
    time_kind: Literal["auto", "datetime", "numeric"] = Field(
        default="auto",
        description=(
            "How to coerce and compare time values. "
            "'auto' infers numeric when possible, otherwise datetime."
        ),
    )
    strict_observed_mask: bool = Field(
        default=True,
        description=(
            "If True and observed_col is provided, enforces observed=False <=> outcome missing."
        ),
    )

    def _coerce_time_series(
        self, s: pd.Series, kind: Literal["datetime", "numeric"]
    ) -> pd.Series:
        """Coerce a time column to a single comparable dtype.

        Parameters
        ----------
        s : pd.Series
            Input time values.
        kind : {"datetime", "numeric"}
            Target coercion kind.

        Returns
        -------
        pd.Series
            Coerced series in the requested representation.
        """
        if kind == "numeric":
            return pd.to_numeric(s, errors="raise")
        return pd.to_datetime(s, errors="raise")

    def _coerce_time_value(
        self, t: Optional[TimeLike], kind: Literal["datetime", "numeric"]
    ) -> Optional[TimeLike]:
        """Coerce one time value using the same rules as the time column.

        Parameters
        ----------
        t : TimeLike or None
            Scalar time value to convert.
        kind : {"datetime", "numeric"}
            Target coercion kind.

        Returns
        -------
        TimeLike or None
            Coerced scalar value, or ``None`` when input is ``None``.
        """
        if t is None:
            return None
        if kind == "numeric":
            out = pd.to_numeric(pd.Series([t]), errors="raise")
            return out.iloc[0]
        out = pd.to_datetime(pd.Series([t]), errors="raise")
        return out.iloc[0]

    @model_validator(mode="after")
    def _validate_schema(self) -> "PanelDataSCM":
        """Validate panel schema and normalize time-related fields.

        Returns
        -------
        PanelDataSCM
            Validated instance with normalized time fields and dataframe.
        """
        df = self.df

        # Required columns
        required = {self.unit_col, self.time_col, self.outcome_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Key columns must be non-null for deterministic uniqueness/filtering behavior.
        if df[self.unit_col].isna().any():
            raise ValueError(f"{self.unit_col!r} contains nulls.")
        if df[self.time_col].isna().any():
            raise ValueError(f"{self.time_col!r} contains nulls.")

        # Optional columns existence checks
        for col in self.covariate_cols:
            if col not in df.columns:
                raise ValueError(f"covariate_cols contains missing column: {col}")
        if self.observed_col is not None and self.observed_col not in df.columns:
            raise ValueError(f"observed_col not found: {self.observed_col}")
        if self.weights_col is not None and self.weights_col not in df.columns:
            raise ValueError(f"weights_col not found: {self.weights_col}")

        # Keep only columns that estimators can consume directly.
        keep_columns = [self.unit_col, self.time_col, self.outcome_col]
        for col in self.covariate_cols:
            if col not in keep_columns:
                keep_columns.append(col)
        if self.observed_col is not None and self.observed_col not in keep_columns:
            keep_columns.append(self.observed_col)
        if self.weights_col is not None and self.weights_col not in keep_columns:
            keep_columns.append(self.weights_col)
        df = df.loc[:, keep_columns].copy()

        # (unit, time) uniqueness
        if not self.allow_duplicate_unit_time:
            has_duplicates = df.duplicated([self.unit_col, self.time_col]).any()
            if has_duplicates:
                raise ValueError(
                    f"Duplicate (unit,time) rows found in [{self.unit_col}, {self.time_col}]. "
                    "Aggregate first or set allow_duplicate_unit_time=True."
                )

        # Normalize all time objects to a single comparable type so downstream
        # comparisons never mix datetime-like and numeric-like representations.
        normalized_kind: Literal["datetime", "numeric"]
        if self.time_kind == "auto":
            if pd.api.types.is_datetime64_any_dtype(df[self.time_col]) or isinstance(
                self.intervention_time, (date, datetime)
            ):
                normalized_kind = "datetime"
            else:
                try:
                    pd.to_numeric(df[self.time_col], errors="raise")
                    normalized_kind = "numeric"
                except Exception:
                    normalized_kind = "datetime"
        else:
            normalized_kind = self.time_kind

        try:
            df[self.time_col] = self._coerce_time_series(df[self.time_col], normalized_kind)
            intervention_time = self._coerce_time_value(self.intervention_time, normalized_kind)
        except Exception as exc:
            raise ValueError(
                "Failed to coerce time axis to a comparable type. "
                f"Set time_kind explicitly or clean {self.time_col!r}."
            ) from exc

        time_window = self.time_window
        if time_window is not None:
            try:
                t_min, t_max = time_window
                t_min = self._coerce_time_value(t_min, normalized_kind)
                t_max = self._coerce_time_value(t_max, normalized_kind)
            except Exception as exc:
                raise ValueError("time_window contains values incompatible with time axis.") from exc
            if t_min is not None and t_max is not None and t_min > t_max:
                raise ValueError("time_window must satisfy t_min <= t_max.")
            time_window = (t_min, t_max)

        pre_periods = self.pre_periods
        if pre_periods is not None:
            try:
                # Keep caller-provided order; coercion should not reorder periods.
                pre_periods = [self._coerce_time_value(t, normalized_kind) for t in pre_periods]
            except Exception as exc:
                raise ValueError("pre_periods contains values incompatible with time axis.") from exc

        post_periods = self.post_periods
        if post_periods is not None:
            try:
                # Keep caller-provided order; coercion should not reorder periods.
                post_periods = [self._coerce_time_value(t, normalized_kind) for t in post_periods]
            except Exception as exc:
                raise ValueError("post_periods contains values incompatible with time axis.") from exc

        # Persist normalized fields on the immutable model so helper methods
        # can rely on one canonical representation.
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "time_kind", normalized_kind)
        object.__setattr__(self, "intervention_time", intervention_time)
        object.__setattr__(self, "time_window", time_window)
        object.__setattr__(self, "pre_periods", pre_periods)
        object.__setattr__(self, "post_periods", post_periods)

        # Treated unit exists
        units = pd.Index(self.df[self.unit_col].unique())
        if self.treated_unit not in set(units):
            raise ValueError(f"treated_unit={self.treated_unit!r} not found in {self.unit_col}.")

        # Donor units validity
        if self.donor_units is not None:
            donor_set = set(self.donor_units)
            if self.treated_unit in donor_set:
                raise ValueError("donor_units must not include treated_unit.")
            missing_donors = donor_set - set(units)
            if missing_donors:
                raise ValueError(
                    f"donor_units contain unknown unit ids: {sorted(missing_donors)}"
                )
            if len(donor_set) < 2:
                raise ValueError("donor_units must contain at least 2 unique units.")
        elif len(units) < 3:
            raise ValueError(
                "Need at least 2 donor units (dataset provides fewer than two non-treated units)."
            )

        # Outcome must be numeric; allow missingness depending on mode.
        y_num = pd.to_numeric(self.df[self.outcome_col], errors="coerce")
        if not self.allow_missing_outcome and y_num.isna().any():
            raise ValueError(
                f"Outcome {self.outcome_col!r} must be numeric and non-missing when "
                "allow_missing_outcome=False."
            )
        if self.allow_missing_outcome:
            created_nan = y_num.isna() & ~self.df[self.outcome_col].isna()
            if created_nan.any():
                raise ValueError(f"Outcome {self.outcome_col!r} contains non-numeric values.")

        # observed_col sanity
        if self.observed_col is not None:
            obs = self.df[self.observed_col]
            # allow 0/1 too
            if not set(obs.dropna().unique()).issubset({0, 1, True, False}):
                raise ValueError(
                    f"observed_col={self.observed_col!r} must be boolean or 0/1."
                )
            if self.strict_observed_mask:
                obs_bool = obs.astype("boolean")
                y_is_na = self.df[self.outcome_col].isna()
                mismatch = ((obs_bool == False) & (~y_is_na)) | ((obs_bool == True) & y_is_na)
                mismatch = mismatch.fillna(False)
                if mismatch.any():
                    raise ValueError(
                        "observed_col/outcome mismatch: observed=False requires outcome NaN, "
                        "observed=True requires outcome present."
                    )

        # weights_col sanity
        if self.weights_col is not None:
            weights = pd.to_numeric(self.df[self.weights_col], errors="coerce")
            if weights.isna().any():
                raise ValueError(
                    f"weights_col={self.weights_col!r} contains non-numeric values."
                )
            if (weights < 0).any():
                raise ValueError(f"weights_col={self.weights_col!r} must be non-negative.")

        # Period consistency and explicit-period coverage checks.
        if self.pre_periods is not None and self.post_periods is not None:
            pre_set = set(self.pre_periods)
            post_set = set(self.post_periods)
            if pre_set.intersection(post_set):
                raise ValueError("pre_periods and post_periods must be disjoint.")
            if self.pre_periods and self.post_periods and max(self.pre_periods) >= min(self.post_periods):
                raise ValueError("Expected all pre_periods < all post_periods.")

        analysis_df = self.df_analysis()
        if self.pre_periods is not None or self.post_periods is not None:
            # Coverage checks run against already-filtered analysis data so
            # explicit periods cannot silently refer to out-of-window times.
            available_times = set(pd.Index(analysis_df[self.time_col].unique()).tolist())
            if self.pre_periods is not None:
                missing_pre = set(self.pre_periods) - available_times
                if missing_pre:
                    raise ValueError(
                        f"pre_periods contain times not present in analysis data: "
                        f"{sorted(missing_pre)}"
                    )
            if self.post_periods is not None:
                missing_post = set(self.post_periods) - available_times
                if missing_post:
                    raise ValueError(
                        f"post_periods contain times not present in analysis data: "
                        f"{sorted(missing_post)}"
                    )

        # Treated post outcomes must be observed for ATT computation.
        post_times = list(self.post_times())
        if post_times:
            treated_post = analysis_df[
                (analysis_df[self.unit_col] == self.treated_unit)
                & (analysis_df[self.time_col].isin(post_times))
            ]
            observed_post_times = set(pd.Index(treated_post[self.time_col].unique()).tolist())
            missing_post_times = sorted(set(post_times) - observed_post_times)

            unobserved_times = set()
            if self.observed_col is not None and not treated_post.empty:
                obs_mask = treated_post[self.observed_col].astype("boolean") == True
                unobserved_times.update(
                    pd.Index(treated_post.loc[~obs_mask.fillna(False), self.time_col].unique()).tolist()
                )

            if not treated_post.empty:
                missing_outcome_mask = treated_post[self.outcome_col].isna()
                unobserved_times.update(
                    pd.Index(treated_post.loc[missing_outcome_mask, self.time_col].unique()).tolist()
                )

            if missing_post_times or unobserved_times:
                bad_times = sorted(set(missing_post_times) | set(unobserved_times))
                raise ValueError(
                    "treated_unit must have observed outcomes in all post-treatment periods. "
                    f"Missing/unobserved treated post periods: {bad_times}."
                )

        return self

    def donor_pool(self) -> Sequence[Hashable]:
        """Return donor unit identifiers used for analysis.

        Returns
        -------
        Sequence[Hashable]
            Explicit donor pool when provided, otherwise all non-treated units.
        """
        if self.donor_units is not None:
            return list(self.donor_units)

        units = pd.Index(self.df[self.unit_col].unique())
        return [unit for unit in units.tolist() if unit != self.treated_unit]

    def _apply_time_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the inclusive analysis time window to a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the configured time column.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe. Returns input unchanged when no window is set.
        """
        if self.time_window is None:
            return df

        t_min, t_max = self.time_window
        out = df
        if t_min is not None:
            out = out[out[self.time_col] >= t_min]
        if t_max is not None:
            out = out[out[self.time_col] <= t_max]
        return out

    def df_analysis(self) -> pd.DataFrame:
        """Build the estimator-facing analysis dataframe.

        Returns
        -------
        pd.DataFrame
            Data restricted to treated unit plus donor pool and optional
            inclusive ``time_window``.
        """
        keep_units = set(self.donor_pool()) | {self.treated_unit}
        out = self.df[self.df[self.unit_col].isin(keep_units)].copy()
        return self._apply_time_window(out)

    def pre_times(self) -> Sequence[TimeLike]:
        """Return pre-treatment periods used for estimation.

        Returns
        -------
        Sequence[TimeLike]
            Explicit ``pre_periods`` when provided, else sorted analysis times
            satisfying ``t < intervention_time``.
        """
        if self.pre_periods is not None:
            return list(self.pre_periods)

        df = self.df_analysis()
        times = pd.Index(df[self.time_col].unique()).tolist()
        return sorted([t for t in times if t < self.intervention_time])

    def post_times(self) -> Sequence[TimeLike]:
        """Return post-treatment periods used for estimation.

        Returns
        -------
        Sequence[TimeLike]
            Explicit ``post_periods`` when provided, else sorted analysis times
            satisfying ``t >= intervention_time``.
        """
        if self.post_periods is not None:
            return list(self.post_periods)

        df = self.df_analysis()
        times = pd.Index(df[self.time_col].unique()).tolist()
        return sorted([t for t in times if t >= self.intervention_time])

    def __repr__(self) -> str:
        def _display_scalar(value: TimeLike) -> TimeLike:
            if hasattr(value, "item") and callable(value.item):
                try:
                    return value.item()
                except Exception:
                    return value
            return value

        donor_units = (
            list(self.donor_units) if self.donor_units is not None else list(self.donor_pool())
        )
        intervention_time = _display_scalar(self.intervention_time)
        res = (
            f"{self.__class__.__name__}(df={self.df.shape}, "
            f"unit_id={self.unit_col!r}, "
            f"time_id={self.time_col!r}, "
            f"y={self.outcome_col!r}, "
            f"treated_unit={self.treated_unit!r}, "
            f"intervention_time={intervention_time!r}, "
            f"donor_units={donor_units!r}"
        )
        if self.time_window is not None:
            t_min, t_max = self.time_window
            res += f", time_window={(_display_scalar(t_min), _display_scalar(t_max))!r}"
        if self.pre_periods is not None:
            res += f", pre_periods={[_display_scalar(t) for t in self.pre_periods]!r}"
        if self.post_periods is not None:
            res += f", post_periods={[_display_scalar(t) for t in self.post_periods]!r}"
        if self.covariate_cols:
            res += f", covariate_cols={list(self.covariate_cols)!r}"
        if self.observed_col is not None:
            res += f", observed_col={self.observed_col!r}"
        if self.weights_col is not None:
            res += f", weights_col={self.weights_col!r}"
        res += ")"
        return res
