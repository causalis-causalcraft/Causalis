import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelDataSCM


def _base_panel_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["T", "T", "T", "C1", "C1", "C1", "C2", "C2", "C2"],
            "time_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [10.0, 11.0, 13.0, 9.0, 9.5, 10.0, 8.5, 9.0, 9.2],
            "x1": [1.0, 1.1, 1.2, 0.8, 0.9, 1.0, 0.7, 0.8, 0.9],
            "observed": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "w": [1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
        }
    )


def test_minimum_contract_and_helpers_work():
    df = _base_panel_df()
    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2)

    assert sorted(panel.donor_pool()) == ["C1", "C2"]
    assert panel.pre_times() == [1]
    assert panel.post_times() == [2, 3]

    analysis = panel.df_analysis()
    assert set(analysis["unit_id"].unique()) == {"T", "C1", "C2"}
    assert set(analysis["time_id"].unique()) == {1, 2, 3}


def test_explicit_donors_and_window_filter_analysis_df():
    df = _base_panel_df()
    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
        df=df,
        treated_unit="T",
        intervention_time=2,
        donor_units=["C1", "C2"],
        time_window=(2, 3),
    )

    analysis = panel.df_analysis()
    assert set(analysis["unit_id"].unique()) == {"T", "C1", "C2"}
    assert set(analysis["time_id"].unique()) == {2, 3}


def test_explicit_periods_override_time_split_rule():
    df = _base_panel_df()
    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
        df=df,
        treated_unit="T",
        intervention_time=2,
        pre_periods=[1, 2],
        post_periods=[3],
    )

    assert panel.pre_times() == [1, 2]
    assert panel.post_times() == [3]


def test_missing_required_columns_raise():
    df = _base_panel_df().drop(columns=["y"])

    with pytest.raises(ValueError, match="Missing required columns"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2)


def test_duplicate_unit_time_rejected_by_default_and_optional_override():
    df = _base_panel_df()
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=r"Duplicate \(unit,time\) rows"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df_dup, treated_unit="T", intervention_time=2)

    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
        df=df_dup,
        treated_unit="T",
        intervention_time=2,
        allow_duplicate_unit_time=True,
    )
    assert panel.allow_duplicate_unit_time is True


def test_treated_and_donor_validation():
    df = _base_panel_df()

    with pytest.raises(ValueError, match="treated_unit='Z' not found"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="Z", intervention_time=2)

    with pytest.raises(ValueError, match="must not include treated_unit"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2, donor_units=["T", "C1"])

    with pytest.raises(ValueError, match="unknown unit ids"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2, donor_units=["C1", "C9"])

    with pytest.raises(ValueError, match="at least 2 unique units"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2, donor_units=[])


def test_missing_outcome_is_gateable():
    df = _base_panel_df()
    df.loc[0, "y"] = np.nan

    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2)
    assert panel.allow_missing_outcome is True

    with pytest.raises(ValueError, match="allow_missing_outcome=False"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            allow_missing_outcome=False,
        )


def test_observed_col_must_be_bool_or_binary():
    df = _base_panel_df()
    df.loc[0, "observed"] = 2

    with pytest.raises(ValueError, match="must be boolean or 0/1"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            observed_col="observed",
        )

    df_ok = _base_panel_df()
    df_ok.loc[0, "observed"] = np.nan
    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
        df=df_ok,
        treated_unit="T",
        intervention_time=2,
        observed_col="observed",
    )
    assert panel.observed_col == "observed"


def test_weights_col_must_be_numeric_and_non_negative():
    df_non_numeric = _base_panel_df()
    df_non_numeric["w"] = df_non_numeric["w"].astype(object)
    df_non_numeric.loc[0, "w"] = "bad"

    with pytest.raises(ValueError, match="contains non-numeric values"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df_non_numeric,
            treated_unit="T",
            intervention_time=2,
            weights_col="w",
        )

    df_negative = _base_panel_df()
    df_negative.loc[0, "w"] = -0.1

    with pytest.raises(ValueError, match="must be non-negative"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df_negative,
            treated_unit="T",
            intervention_time=2,
            weights_col="w",
        )


def test_optional_column_references_must_exist():
    df = _base_panel_df()

    with pytest.raises(ValueError, match="covariate_cols contains missing column"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            covariate_cols=["x_missing"],
        )

    with pytest.raises(ValueError, match="observed_col not found"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            observed_col="missing_obs",
        )

    with pytest.raises(ValueError, match="weights_col not found"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            weights_col="missing_w",
        )


def test_df_keeps_only_estimation_columns():
    df = _base_panel_df().copy()
    df["oracle"] = np.arange(len(df))

    panel = PanelDataSCM(
        unit_id="unit_id",
        time_id="time_id",
        y="y",
        df=df,
        treated_unit="T",
        intervention_time=2,
        covariate_cols=["x1"],
        observed_col="observed",
    )

    assert panel.df.columns.tolist() == ["unit_id", "time_id", "y", "x1", "observed"]
    assert "w" not in panel.df.columns
    assert "oracle" not in panel.df.columns

    panel_with_weights = PanelDataSCM(
        unit_id="unit_id",
        time_id="time_id",
        y="y",
        df=df,
        treated_unit="T",
        intervention_time=2,
        weights_col="w",
    )

    assert panel_with_weights.df.columns.tolist() == ["unit_id", "time_id", "y", "w"]


def test_unit_and_time_keys_must_be_non_null():
    df_unit_null = _base_panel_df()
    df_unit_null.loc[0, "unit_id"] = np.nan
    with pytest.raises(ValueError, match="unit_id' contains nulls"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df_unit_null, treated_unit="T", intervention_time=2)

    df_time_null = _base_panel_df()
    df_time_null.loc[0, "time_id"] = np.nan
    with pytest.raises(ValueError, match="time_id' contains nulls"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df_time_null, treated_unit="T", intervention_time=2)


def test_implicit_donor_pool_must_have_at_least_two_units():
    df = pd.DataFrame(
        {
            "unit_id": ["T", "T", "T"],
            "time_id": [1, 2, 3],
            "y": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2)


def test_implicit_donor_pool_rejects_single_donor():
    df = pd.DataFrame(
        {
            "unit_id": ["T", "T", "T", "C1", "C1", "C1"],
            "time_id": [1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
        }
    )

    with pytest.raises(ValueError, match="Need at least 2 donor units"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2)


def test_time_is_normalized_to_numeric_in_auto_mode():
    df = _base_panel_df().copy()
    df["time_id"] = df["time_id"].astype(str)
    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time="2")

    assert panel.time_kind == "numeric"
    assert panel.pre_times() == [1]
    assert panel.post_times() == [2, 3]


def test_time_coercion_rejects_incompatible_values():
    df = _base_panel_df().copy()
    df["time_id"] = df["time_id"].astype(object)
    df.loc[0, "time_id"] = "not-a-date"

    with pytest.raises(ValueError, match="Failed to coerce time axis"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2)


def test_outcome_must_be_numeric():
    df = _base_panel_df().copy()
    df["y"] = df["y"].astype(object)
    df.loc[0, "y"] = "bad"

    with pytest.raises(ValueError, match="contains non-numeric values"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=2, allow_missing_outcome=True)


def test_treated_post_outcomes_must_be_observed():
    df = _base_panel_df().copy()
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == 3), "y"] = np.nan

    with pytest.raises(ValueError, match="treated_unit must have observed outcomes"):
        PanelDataSCM(
            unit_id="unit_id",
            time_id="time_id",
            y="y",
            df=df,
            treated_unit="T",
            intervention_time=2,
            allow_missing_outcome=True,
        )


def test_observed_mask_can_be_strict_or_relaxed():
    df = _base_panel_df().copy()
    df.loc[0, "observed"] = 0

    with pytest.raises(ValueError, match="observed_col/outcome mismatch"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            observed_col="observed",
        )

    panel = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
        df=df,
        treated_unit="T",
        intervention_time=2,
        observed_col="observed",
        strict_observed_mask=False,
    )
    assert panel.strict_observed_mask is False


def test_explicit_periods_must_be_disjoint_and_present_in_analysis_data():
    df = _base_panel_df()

    with pytest.raises(ValueError, match="must be disjoint"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            pre_periods=[1, 2],
            post_periods=[2, 3],
        )

    with pytest.raises(ValueError, match="Expected all pre_periods < all post_periods"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            pre_periods=[3],
            post_periods=[1],
        )

    with pytest.raises(ValueError, match="not present in analysis data"):
        PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", 
            df=df,
            treated_unit="T",
            intervention_time=2,
            time_window=(2, 3),
            pre_periods=[1],
            post_periods=[2, 3],
        )


def test_panel_repr_is_compact_and_informative():
    df = _base_panel_df()
    panel = PanelDataSCM(
        unit_id="unit_id",
        time_id="time_id",
        y="y",
        df=df,
        treated_unit="T",
        intervention_time=2,
        covariate_cols=["x1"],
        observed_col="observed",
    )

    repr_str = repr(panel)

    assert repr_str.startswith("PanelDataSCM(df=(9, 5),")
    assert "unit_id='unit_id'" in repr_str
    assert "time_id='time_id'" in repr_str
    assert "y='y'" in repr_str
    assert "treated_unit='T'" in repr_str
    assert "intervention_time=2" in repr_str
    assert "donor_units=['C1', 'C2']" in repr_str
    assert "covariate_cols=['x1']" in repr_str
    assert "observed_col='observed'" in repr_str
