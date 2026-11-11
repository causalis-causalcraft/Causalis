import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts.ipcw_data import IPCWData
from causalis.data_contracts.ipcw_estimate import IPCWDiagnosticData, IPCWEstimate
from causalis.dgp.organic_activation import (
    generate_organic_activation_data,
    make_ipcw_view,
)


@pytest.fixture
def trial():
    rows = []
    # One treated client, one never-censored control and two censoring times.
    for unit, treatment, event in [
        ("treated", 1, None),
        ("late", 0, 2),
        ("control", 0, None),
        ("early", 0, 1),
    ]:
        for month in range(3):
            rows.append(
                {
                    "client_id": unit,
                    "trial_id": 10,
                    "period": 10 + month,
                    "followup_month": month,
                    "baseline_d": treatment,
                    "at_risk_of_censoring": treatment == 0
                    and month > 0
                    and (event is None or month <= event),
                    "censor_now": event == month,
                    "outcome_usable": event is None or month < event,
                    "ltv_monthly": 100.0 + month + treatment * 10,
                    "b_x": 0.5 + len(unit),
                    "x_dynamic": 0.1 + month,
                    "oracle_truth": 777.0,
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["model_features"] = {
        "baseline": ["b_x"],
        "censoring": ["period", "x_dynamic"],
    }
    return df


def test_contract_accepts_existing_organic_view():
    panel, _, _ = generate_organic_activation_data(
        n_clients=250,
        n_periods=6,
        horizon=3,
        seed=4,
        main_cohorts=(2,),
        entry_max=1,
    )
    view = make_ipcw_view(panel, g=2, horizon=3)
    data = IPCWData.from_df(view)

    assert data.horizon == 3
    assert data.trial_id == 2
    assert data.n_units == view.client_id.nunique()
    assert data.baseline_covariates == view.attrs["model_features"]["baseline"]
    assert data.censoring_covariates == view.attrs["model_features"]["censoring"]
    assert "date" not in data.get_df()
    assert "uncensored_through_t" not in data.get_df()


def test_snapshot_is_independent_and_rows_are_sorted_stably(trial):
    shuffled = pd.concat(
        [trial.iloc[6:9], trial.iloc[0:3], trial.iloc[9:12], trial.iloc[3:6]]
    )
    shuffled = shuffled.iloc[[2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10]].copy()
    data = IPCWData(df=shuffled)
    expected = data.get_df()

    assert expected.client_id.drop_duplicates().tolist() == [
        "control",
        "treated",
        "early",
        "late",
    ]
    assert expected.followup_month.tolist() == [0, 1, 2] * 4
    assert "oracle_truth" not in expected
    shuffled.loc[:, "ltv_monthly"] = -100.0
    data.df.loc[:, "ltv_monthly"] = -200.0
    returned = data.get_df()
    returned.loc[:, "ltv_monthly"] = -300.0
    returned.attrs["model_features"]["baseline"].append("oracle_truth")
    pd.testing.assert_frame_equal(data.get_df(), expected)
    assert data.get_df().attrs["model_features"]["baseline"] == ["b_x"]


def test_explicit_features_override_manifest_and_empty_lists_work(trial):
    trial.attrs = {}
    data = IPCWData.from_df(trial, baseline_covariates=[], censoring_covariates=[])
    assert data.baseline_covariates == []
    assert data.censoring_covariates == []
    assert "b_x" not in data.get_df()
    with pytest.raises(ValueError, match="baseline_covariates"):
        IPCWData(df=trial)


def test_explicit_role_overrides(trial):
    names = {
        "client_id": "person",
        "trial_id": "trial",
        "period": "month",
        "followup_month": "h",
        "baseline_d": "a",
        "at_risk_of_censoring": "risk",
        "censor_now": "event",
        "outcome_usable": "usable",
        "ltv_monthly": "y",
    }
    renamed = trial.rename(columns=names)
    data = IPCWData.from_df(
        renamed,
        unit_col="person",
        trial_col="trial",
        time_col="month",
        followup_col="h",
        treatment="a",
        at_risk_col="risk",
        censor_col="event",
        outcome_usable_col="usable",
        outcome="y",
        baseline_covariates=["b_x"],
        censoring_covariates=["month", "x_dynamic"],
    )
    assert data.get_df()["a"].sum() == 3
    assert data.time_col in data.censoring_covariates


def test_finite_values_only_required_where_used(trial):
    trial.loc[~trial.outcome_usable, "ltv_monthly"] = np.nan
    trial.loc[~trial.at_risk_of_censoring, "x_dynamic"] = np.nan
    data = IPCWData(df=trial)
    assert data.get_df().ltv_monthly.isna().sum() == 3

    invalid = trial.copy(deep=True)
    invalid.loc[invalid.outcome_usable.idxmax(), "ltv_monthly"] = np.nan
    with pytest.raises(ValueError, match="ltv_monthly"):
        IPCWData(df=invalid)
    invalid = trial.copy(deep=True)
    invalid.loc[invalid.at_risk_of_censoring.idxmax(), "x_dynamic"] = np.inf
    with pytest.raises(ValueError, match="x_dynamic"):
        IPCWData(df=invalid)


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda df: df.assign(trial_id=np.where(df.client_id == "early", 11, 10)),
            "exactly one trial",
        ),
        (lambda df: df.iloc[:-1].copy(), "complete common follow-up grid"),
        (lambda df: pd.concat([df, df.iloc[:1]]), "Duplicate unit"),
        (
            lambda df: df.assign(followup_month=df.followup_month.astype(float)),
            "non-negative integers",
        ),
        (lambda df: df.assign(baseline_d=0), "both treated and control"),
        (lambda df: df.assign(outcome_usable=True), "unusable from its event"),
        (lambda df: df.assign(at_risk_of_censoring=False), "At-risk indicators"),
        (lambda df: df.assign(censor_now=False), "At-risk indicators"),
    ],
)
def test_invalid_trial_structure_is_rejected(trial, change, message):
    with pytest.raises(ValueError, match=message):
        IPCWData(df=change(trial))


def test_baseline_treatment_features_and_calendar_are_invariant(trial):
    for name, value, message in [
        ("baseline_d", 0, "Baseline treatment"),
        ("b_x", 888.0, "constant within unit"),
        ("period", 999, "Calendar time"),
    ]:
        invalid = trial.copy(deep=True)
        invalid.loc[1, name] = value
        with pytest.raises(ValueError, match=message):
            IPCWData(df=invalid)


@pytest.mark.parametrize("unit, month", [("treated", 1), ("control", 0), ("early", 2)])
def test_invalid_censoring_events_are_rejected(trial, unit, month):
    invalid = trial.copy(deep=True)
    invalid.loc[
        (invalid.client_id == unit) & (invalid.followup_month == month), "censor_now"
    ] = True
    with pytest.raises(ValueError, match="Only controls may be censored"):
        IPCWData(df=invalid)


@pytest.mark.parametrize(
    "feature",
    [
        "ltv_monthly",
        "baseline_d",
        "client_id",
        "trial_id",
        "censor_now",
        "outcome_usable",
    ],
)
def test_role_columns_cannot_leak_into_feature_lists(trial, feature):
    with pytest.raises(ValueError, match="cannot be predictors"):
        IPCWData(df=trial, baseline_covariates=[feature])


def test_one_period_trial_does_not_require_censoring_features(trial):
    baseline = trial.loc[trial.followup_month == 0].copy()
    baseline["x_dynamic"] = np.nan
    data = IPCWData(df=baseline)
    assert data.horizon == 1
    assert not data.get_df().at_risk_of_censoring.any()
    assert data.get_df().outcome_usable.all()


def _estimate_kwargs():
    return dict(
        value=30.0,
        n_treated=20,
        n_control=80,
        horizon=3,
        trial_id=10,
        treatment_mean=360.0,
        control_mean=330.0,
        outcome="y",
        treatment="d",
        period_effects=pd.DataFrame(
            {"followup_month": [0, 1, 2], "estimate": [5.0, 10.0, 15.0]}
        ),
        cumulative_effects=pd.DataFrame(
            {"followup_month": [0, 1, 2], "estimate": [5.0, 15.0, 30.0]}
        ),
    )


def test_result_optional_inference_aliases_and_summary():
    result = IPCWEstimate(**_estimate_kwargs())
    assert result.att == 30.0
    assert result.se is None
    assert result.ci_lower is None
    assert result.is_significant is None
    assert result.summary().loc["horizon", "value"] == 3
    assert result.summary().loc["estimand", "value"] == "ATT_no_activation"

    result = IPCWEstimate(
        **_estimate_kwargs(),
        std_error=2.0,
        ci_lower_absolute=26.0,
        ci_upper_absolute=34.0,
        p_value=0.01,
        is_significant=True,
    )
    assert result.se == 2.0
    assert result.ci_lower == 26.0
    assert "ci_abs" in result.summary().loc["value", "value"]


def test_result_and_diagnostic_frames_are_copied():
    frames = _estimate_kwargs()
    row_data = pd.DataFrame({"unit_id": [1], "survival": [0.9]})
    diagnostics = IPCWDiagnosticData(row_data=row_data)
    result = IPCWEstimate(**frames, diagnostic_data=diagnostics)
    row_data.loc[0, "survival"] = 0.1
    frames["period_effects"].loc[0, "estimate"] = -100.0
    assert result.diagnostic_data.row_data.loc[0, "survival"] == 0.9
    assert result.period_effects.loc[0, "estimate"] == 5.0


@pytest.mark.parametrize(
    "override, message",
    [
        ({"alpha": 0.0}, "alpha"),
        ({"std_error": -1.0}, "non-negative"),
        ({"ci_lower_absolute": 5.0}, "bounds"),
        ({"ci_lower_absolute": 5.0, "ci_upper_absolute": 4.0}, "ordered"),
        ({"p_value": 1.1}, "p_value"),
        ({"n_control": 0}, "positive"),
    ],
)
def test_invalid_result_values_are_rejected(override, message):
    kwargs = _estimate_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        IPCWEstimate(**kwargs)
