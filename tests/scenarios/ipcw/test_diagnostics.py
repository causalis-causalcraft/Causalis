from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts.ipcw_data import IPCWData
from causalis.data_contracts.ipcw_estimate import IPCWDiagnosticData, IPCWEstimate
from causalis.scenarios.ipcw.refutation import (
    ipcw_support_table,
    ipcw_weight_summary_table,
    plot_ipcw_effects,
    plot_ipcw_weights,
    run_ipcw_diagnostics,
)


@pytest.fixture
def trial() -> IPCWData:
    records = []
    for unit, censor_month in (("treated", None), ("c1", 1), ("c2", 2), ("c3", None)):
        treated = unit == "treated"
        for month in range(3):
            records.append(
                {
                    "client": unit,
                    "g": 10,
                    "period": 10 + month,
                    "month": month,
                    "d": int(treated),
                    "risk": not treated
                    and month > 0
                    and (censor_month is None or month <= censor_month),
                    "censored": not treated and month == censor_month,
                    "usable": treated or censor_month is None or month < censor_month,
                    "y": float(5 + month),
                    "b_x": 1.0,
                    "x": float(month),
                }
            )
    return IPCWData(
        df=pd.DataFrame(records),
        unit_col="client",
        trial_col="g",
        time_col="period",
        followup_col="month",
        treatment="d",
        outcome="y",
        at_risk_col="risk",
        censor_col="censored",
        outcome_usable_col="usable",
        baseline_covariates=["b_x"],
        censoring_covariates=["x", "period"],
    )


@pytest.fixture
def estimate(trial: IPCWData) -> IPCWEstimate:
    rows = trial.get_df().rename(
        columns={
            "client": "unit_id",
            "g": "trial_id",
            "period": "time",
            "month": "followup_month",
            "d": "treatment",
            "risk": "at_risk",
            "censored": "censor_now",
            "usable": "outcome_usable",
        }
    )
    weight_by_unit = {
        "treated": [0, 0, 0],
        "c1": [1, 0, 0],
        "c2": [2, 4, 0],
        "c3": [3, 6, 12],
    }
    rows["control_weight"] = [
        weight_by_unit[unit][month]
        for unit, month in zip(rows.unit_id, rows.followup_month)
    ]
    rows["propensity_clipped"] = rows.unit_id.eq("c1")
    # The post-censoring true flag must not enter censoring-clipping shares.
    rows["censoring_clipped"] = rows.unit_id.eq("c1") & rows.followup_month.gt(0)
    rows["survival"] = np.where(
        rows.followup_month.eq(0), 1.0, 0.5**rows.followup_month
    )
    periods = pd.DataFrame(
        {
            "time": [10, 11, 12],
            "followup_month": [0, 1, 2],
            "value": [1.0, 2.0, 3.0],
            "treatment_mean": [6.0, 8.0, 10.0],
            "control_mean": [5.0, 6.0, 7.0],
            "std_error": [None] * 3,
            "ci_lower_absolute": [None] * 3,
            "ci_upper_absolute": [None] * 3,
            "p_value": [None] * 3,
            "is_significant": [None] * 3,
        }
    )
    cumulative = periods.copy(deep=True)
    for column in ("value", "treatment_mean", "control_mean"):
        cumulative[column] = periods[column].cumsum()
    return IPCWEstimate(
        value=6.0,
        n_treated=1,
        n_control=3,
        horizon=3,
        trial_id=10,
        treatment_mean=24.0,
        control_mean=18.0,
        outcome="y",
        treatment="d",
        period_effects=periods,
        cumulative_effects=cumulative,
        diagnostic_data=IPCWDiagnosticData(row_data=rows),
    )


def test_support_counts_include_events_but_exclude_their_outcomes(trial, estimate):
    support = ipcw_support_table(trial)
    assert support.n_treated.tolist() == [1, 1, 1]
    assert support.n_control.tolist() == [3, 3, 3]
    assert support.n_control_at_risk.tolist() == [0, 3, 2]
    assert support.n_censored.tolist() == [0, 1, 1]
    assert support.n_control_usable.tolist() == [3, 2, 1]
    np.testing.assert_allclose(support.retention_share, [1, 2 / 3, 1 / 3])
    assert support.is_supported.all()
    pd.testing.assert_frame_equal(support, ipcw_support_table(estimate))


def test_weight_summary_uses_usable_controls_and_at_risk_clipping(estimate):
    summary = ipcw_weight_summary_table(estimate)
    np.testing.assert_allclose(summary.control_weight_sum, [6, 10, 12])
    np.testing.assert_allclose(summary.control_ess, [36 / 14, 100 / 52, 1])
    np.testing.assert_allclose(summary.weight_min, [1, 4, 12])
    np.testing.assert_allclose(summary.weight_max, [3, 6, 12])
    np.testing.assert_allclose(summary.propensity_clipping_share, [0.25] * 3)
    np.testing.assert_allclose(summary.censoring_clipping_share, [0, 1 / 3, 0])
    np.testing.assert_allclose(summary.min_survival, [1, 0.5, 0.25])


def test_reports_distinguish_prefit_support_from_fitted_weight_checks(trial, estimate):
    kwargs = dict(
        min_treated=1, min_control=1, min_control_ess=2, max_clipping_share=0.1
    )
    before_fit = run_ipcw_diagnostics(trial, **kwargs)
    after_fit = run_ipcw_diagnostics(estimate, **kwargs).set_index("test")
    assert before_fit.columns.tolist() == [
        "test",
        "flag",
        "value",
        "threshold",
        "message",
    ]
    assert before_fit.flag.eq("GREEN").all()
    assert not before_fit.test.str.contains("clipping|ess|weights").any()
    assert after_fit.loc["monthly_outcome_support", "flag"] == "GREEN"
    assert after_fit.loc["min_control_ess", "flag"] == "YELLOW"
    assert after_fit.loc["baseline_propensity_clipping", "value"] == 0.25
    assert after_fit.loc["censoring_clipping", "value"] == pytest.approx(1 / 5)
    assert after_fit.loc["censoring_clipping", "flag"] == "YELLOW"


def test_invalid_usable_weights_are_flagged_instead_of_discarded(estimate):
    rows = estimate.diagnostic_data.row_data
    rows.loc[rows.unit_id.eq("c3") & rows.followup_month.eq(2), "control_weight"] = (
        np.nan
    )
    report = run_ipcw_diagnostics(estimate).set_index("test")
    assert report.loc["finite_control_weights", "flag"] == "RED"
    # The monthly summary must not silently remove the invalid observation.
    assert np.isnan(ipcw_weight_summary_table(estimate).iloc[-1].control_ess)
    with pytest.raises(ValueError, match="finite and positive"):
        plot_ipcw_weights(estimate)


def test_missing_monthly_control_support_is_red(estimate):
    rows = estimate.diagnostic_data.row_data
    rows.loc[rows.followup_month.eq(2) & rows.treatment.eq(0), "outcome_usable"] = False
    report = run_ipcw_diagnostics(estimate).set_index("test")
    assert report.loc["monthly_outcome_support", "flag"] == "RED"


def test_returned_tables_do_not_mutate_stored_diagnostics(estimate):
    diagnostics = estimate.diagnostic_data
    diagnostics.support = ipcw_support_table(estimate)
    diagnostics.weight_summary = ipcw_weight_summary_table(estimate)
    support = ipcw_support_table(estimate)
    weights = ipcw_weight_summary_table(estimate)
    support.loc[0, "n_control"] = -1
    weights.loc[0, "control_ess"] = -1
    assert diagnostics.support.loc[0, "n_control"] == 3
    assert diagnostics.weight_summary.loc[0, "control_ess"] > 0


def test_effect_plots_work_without_diagnostic_payload_and_close_figures(estimate):
    weight_figure = plot_ipcw_weights(estimate, log_scale=True)
    estimate.diagnostic_data = None
    monthly_figure = plot_ipcw_effects(estimate, cumulative=False)
    cumulative_figure = plot_ipcw_effects(estimate)
    for figure in (weight_figure, monthly_figure, cumulative_figure):
        assert isinstance(figure, matplotlib.figure.Figure)
        assert not plt.fignum_exists(figure.number)
    assert weight_figure.axes[0].get_yscale() == "log"
    assert monthly_figure.axes[0].get_title() == "IPCW Monthly Effects"
    np.testing.assert_allclose(monthly_figure.axes[0].lines[0].get_ydata(), [1, 2, 3])
    np.testing.assert_allclose(
        cumulative_figure.axes[0].lines[0].get_ydata(), [1, 3, 6]
    )


def test_fitted_reports_require_retained_diagnostics(estimate):
    estimate.diagnostic_data = None
    for function in (
        ipcw_support_table,
        ipcw_weight_summary_table,
        run_ipcw_diagnostics,
        plot_ipcw_weights,
    ):
        with pytest.raises(ValueError, match="store_diagnostics=True"):
            function(estimate)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_treated": 0},
        {"min_control": np.nan},
        {"min_control_ess": True},
        {"max_clipping_share": -0.1},
        {"max_clipping_share": 1.1},
        {"min_control_retention": np.inf},
    ],
)
def test_diagnostic_thresholds_are_validated(trial, kwargs):
    with pytest.raises(ValueError):
        run_ipcw_diagnostics(trial, **kwargs)


def test_refutation_rejects_unvalidated_inputs():
    with pytest.raises(TypeError, match="IPCWData or IPCWEstimate"):
        ipcw_support_table(pd.DataFrame())
    with pytest.raises(TypeError, match="IPCWEstimate"):
        ipcw_weight_summary_table(pd.DataFrame())
