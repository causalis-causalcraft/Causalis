import numpy as np
import pandas as pd
import pytest

from causalis.dgp.organic_activation.views import (
    DID_FEATURES,
    DML_FEATURES,
    IPCW_BASELINE_FEATURES,
    IPCW_CENSOR_FEATURES,
    STATIC_FEATURES,
    make_did_pair,
    make_dml_view,
    make_ipcw_view,
)


@pytest.fixture
def panel():
    # At g=2, client 1 has one baseline month, 2 activates next month,
    # 3 activates at the horizon end, 4 just after it, and 5 never activates.
    entries = [0, 1, 0, 1, 1, 1, 2]
    activations = [1, 2, 3, 4, 5, None, 3]
    rows = []
    for client, (entry, first) in enumerate(zip(entries, activations)):
        for period in range(entry, 7):
            row = {
                "client_id": client,
                "period": period,
                "date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=period),
                "entry_period": entry,
                "tenure_months": period - entry,
                "n_pre_periods_available": period - entry,
                "first_activation_period": first,
                "activation_now": first is not None and period == first,
                "treated": first is not None and period >= first,
                "eligible_to_activate": period > entry
                and (first is None or first >= period),
                "event_time": period - first if first is not None else None,
                "l_activity_pre": float(100 * client + period),
                "ltv_monthly": float(10 * client + period),
            }
            row.update(
                {
                    name: float(client + index)
                    for index, name in enumerate(STATIC_FEATURES)
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["first_activation_period"] = frame["first_activation_period"].astype("Int64")
    frame["event_time"] = frame["event_time"].astype("Int64")
    frame["ltv_cumulative"] = frame.groupby("client_id")["ltv_monthly"].cumsum()
    frame.attrs = {"sentinel": {"source": True}}
    return frame


def test_did_pair_uses_eligibility_at_g_and_strict_not_yet_treated_controls(panel):
    out = make_did_pair(panel, g=2, t=4).set_index("client_id")
    assert out.index.tolist() == [1, 4, 5]
    assert out["d"].tolist() == [1, 0, 0]
    assert out["baseline_period"].eq(1).all()
    assert out["g"].eq(2).all()
    assert out["t"].eq(4).all()
    assert out.loc[1, "y_pre"] == 11.0
    assert out.loc[1, "y_post"] == 14.0
    assert out["delta_y"].eq(3.0).all()
    assert out.attrs["model_features"] == list(DID_FEATURES)
    assert out.attrs["estimand"] == "ATT_no_activation"
    assert "l_activity_pre" not in out.attrs["model_features"]
    assert not panel.loc[
        (panel.client_id == 1) & (panel.period == 1), "eligible_to_activate"
    ].item()


def test_dml_keeps_future_activators_and_sums_exact_horizon(panel):
    out = make_dml_view(panel, g=2, horizon=3).set_index("client_id")
    assert out.index.tolist() == [1, 2, 3, 4, 5]
    assert out["d"].tolist() == [1, 0, 0, 0, 0]
    assert out["baseline_period"].eq(2).all()
    assert out["y_ltv_h"].tolist() == [39.0, 69.0, 99.0, 129.0, 159.0]
    assert out["l_activity_pre"].tolist() == [102.0, 202.0, 302.0, 402.0, 502.0]
    assert out.attrs["model_features"] == list(DML_FEATURES)
    assert out.attrs["estimand"] == "ATT_defer_natural"


def test_ipcw_keeps_censoring_event_but_excludes_its_outcome(panel):
    out = make_ipcw_view(panel, g=2, horizon=3)
    assert len(out) == 15
    assert out.groupby("client_id").size().eq(3).all()
    assert out["trial_id"].eq(2).all()
    assert out["baseline_period"].eq(2).all()
    assert out["followup_month"].tolist() == [0, 1, 2] * 5
    assert out.loc[out.client_id == 1, "baseline_d"].eq(1).all()
    assert out.loc[out.client_id != 1, "baseline_d"].eq(0).all()

    treated = out.loc[out.client_id == 1]
    assert not treated["at_risk_of_censoring"].any()
    assert not treated["censor_now"].any()
    assert treated["outcome_usable"].all()

    later = out.loc[out.client_id == 2]
    assert later["at_risk_of_censoring"].tolist() == [False, True, False]
    assert later["censor_now"].tolist() == [False, True, False]
    assert later["uncensored_through_t"].tolist() == [True, False, False]
    assert later["outcome_usable"].tolist() == [True, False, False]
    assert later["ltv_monthly"].tolist() == [22.0, 23.0, 24.0]

    end = out.loc[out.client_id == 3]
    assert end["at_risk_of_censoring"].tolist() == [False, True, True]
    assert end["censor_now"].tolist() == [False, False, True]
    assert end["outcome_usable"].tolist() == [True, True, False]
    unaffected = out.loc[out.client_id.isin([4, 5])]
    assert unaffected["outcome_usable"].all()
    assert not unaffected["censor_now"].any()
    assert not out.loc[out.period == 2, "at_risk_of_censoring"].any()


def test_ipcw_freezes_baseline_features_and_declares_safe_manifests(panel):
    out = make_ipcw_view(panel, g=2, horizon=3)
    assert out.groupby("client_id")["b_l_activity_pre"].nunique().eq(1).all()
    assert out.loc[out.client_id == 2, "b_l_activity_pre"].tolist() == [202.0] * 3
    assert out.loc[out.client_id == 2, "l_activity_pre"].tolist() == [
        202.0,
        203.0,
        204.0,
    ]
    manifests = out.attrs["model_features"]
    assert manifests == {
        "baseline": list(IPCW_BASELINE_FEATURES),
        "censoring": list(IPCW_CENSOR_FEATURES),
    }
    forbidden = {
        "client_id",
        "first_activation_period",
        "activation_now",
        "treated",
        "event_time",
        "ltv_monthly",
        "ltv_cumulative",
        "baseline_d",
        "censor_now",
        "outcome_usable",
    }
    for features in manifests.values():
        assert set(features).issubset(out.columns)
        assert not forbidden.intersection(features)


@pytest.mark.parametrize(
    "helper",
    [
        lambda df: make_did_pair(df, g=2, t=4),
        lambda df: make_dml_view(df, g=2, horizon=3),
        lambda df: make_ipcw_view(df, g=2, horizon=3),
    ],
)
def test_views_leave_source_and_nullable_dtypes_unchanged(panel, helper):
    before = panel.copy(deep=True)
    helper(panel)
    pd.testing.assert_frame_equal(panel, before)
    assert panel.attrs == before.attrs
    assert panel["first_activation_period"].dtype == pd.Int64Dtype()
    assert panel["event_time"].dtype == pd.Int64Dtype()


@pytest.mark.parametrize(
    "helper",
    [
        lambda df: make_did_pair(df, g=2, t=4),
        lambda df: make_dml_view(df, g=2, horizon=3),
        lambda df: make_ipcw_view(df, g=2, horizon=3),
    ],
)
def test_views_reject_duplicate_keys(panel, helper):
    broken = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        helper(broken)


@pytest.mark.parametrize(
    "helper",
    [
        lambda df: make_did_pair(df, g=2, t=4),
        lambda df: make_dml_view(df, g=2, horizon=3),
        lambda df: make_ipcw_view(df, g=2, horizon=3),
    ],
)
@pytest.mark.parametrize("missing_period", [1, 2])
def test_views_reject_missing_baseline_rows(panel, helper, missing_period):
    broken = panel.loc[~((panel.client_id == 1) & (panel.period == missing_period))]
    with pytest.raises(ValueError, match="Missing required baseline"):
        helper(broken)


@pytest.mark.parametrize("helper", [make_dml_view, make_ipcw_view])
def test_followup_helpers_reject_incomplete_individual_horizons(panel, helper):
    broken = panel.loc[~((panel.client_id == 5) & (panel.period == 3))]
    with pytest.raises(ValueError, match="Missing required follow-up"):
        helper(broken, g=2, horizon=3)
    with pytest.raises(ValueError, match="full follow-up horizon"):
        helper(panel, g=5, horizon=3)


def test_did_pair_rejects_missing_comparison_outcome(panel):
    broken = panel.loc[~((panel.client_id == 4) & (panel.period == 4))]
    with pytest.raises(ValueError, match="Missing required baseline/follow-up"):
        make_did_pair(broken, g=2, t=4)


@pytest.mark.parametrize(
    "helper,kwargs",
    [
        (make_did_pair, {"g": 0, "t": 4}),
        (make_did_pair, {"g": 2, "t": 1}),
        (make_did_pair, {"g": 2, "t": 8}),
        (make_did_pair, {"g": 2.0, "t": 4}),
        (make_dml_view, {"g": True, "horizon": 3}),
        (make_dml_view, {"g": 2, "horizon": 0}),
        (make_ipcw_view, {"g": 2, "horizon": 1.5}),
        (make_ipcw_view, {"g": 7, "horizon": 1}),
    ],
)
def test_views_reject_invalid_calendar_arguments(panel, helper, kwargs):
    with pytest.raises(ValueError):
        helper(panel, **kwargs)


def test_empty_risk_sets_return_typed_empty_views(panel):
    late_entry_only = panel.loc[panel.client_id == 6]
    outputs = [
        make_did_pair(late_entry_only, g=1, t=3),
        make_dml_view(late_entry_only, g=1, horizon=3),
        make_ipcw_view(late_entry_only, g=1, horizon=3),
    ]
    assert all(out.empty for out in outputs)
    assert outputs[0]["d"].dtype == np.dtype("int8")
    assert outputs[1]["d"].dtype == np.dtype("int8")
    assert outputs[2]["baseline_d"].dtype == np.dtype("int8")


def test_horizon_one_has_no_artificial_censoring(panel):
    out = make_ipcw_view(panel, g=2, horizon=1)
    assert len(out) == 5
    assert out["outcome_usable"].all()
    assert not out["at_risk_of_censoring"].any()
    assert not out["censor_now"].any()
