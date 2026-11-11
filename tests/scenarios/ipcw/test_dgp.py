import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from causalis.data_contracts import IPCWData
from causalis.dgp.organic_activation import (
    generate_organic_activation_data,
    make_ipcw_view,
)
from causalis.scenarios.ipcw import IPCW, generate_ipcw_data
from causalis.scenarios.ipcw.refutation import run_ipcw_diagnostics

GENERATOR_PARAMS = {
    "n_clients": 600,
    "n_periods": 16,
    "g": 6,
    "horizon": 4,
    "seed": 29,
}


def test_generator_returns_contract_and_separate_single_cohort_oracle():
    data, oracle, metadata = generate_ipcw_data(**GENERATOR_PARAMS)

    assert isinstance(data, IPCWData)
    view = data.get_df()
    assert view["trial_id"].unique().tolist() == [6]
    assert view["followup_month"].unique().tolist() == [0, 1, 2, 3]
    assert metadata["main_cohorts"] == (6,)
    assert oracle["config"].main_cohorts == (6,)
    assert oracle["att_by_cohort"]["g"].unique().tolist() == [6]
    assert oracle["individual_effects"]["g"].unique().tolist() == [6]
    np.testing.assert_allclose(
        oracle["att_aggregate"]["att_no_activation"],
        oracle["att_by_cohort"]["att_no_activation"],
    )
    assert "oracle" not in view.attrs
    assert not {"y_no_activation", "y_start", "U", "V"}.intersection(view.columns)


def test_raw_generator_reproduces_existing_view_and_oracle():
    view, oracle, metadata = generate_ipcw_data(
        **GENERATOR_PARAMS, return_ipcw_data=False
    )
    panel, expected_oracle, expected_metadata = generate_organic_activation_data(
        n_clients=600,
        n_periods=16,
        horizon=4,
        main_cohorts=(6,),
        seed=29,
    )
    expected = make_ipcw_view(panel, g=6, horizon=4)

    assert isinstance(view, pd.DataFrame)
    pd.testing.assert_frame_equal(view, expected)
    assert view.attrs == expected.attrs
    assert set(view.attrs["model_features"]) == {"baseline", "censoring"}
    assert view.attrs["estimand"] == "ATT_no_activation"
    pd.testing.assert_frame_equal(
        oracle["att_aggregate"], expected_oracle["att_aggregate"]
    )
    pd.testing.assert_frame_equal(
        oracle["potential_outcomes"], expected_oracle["potential_outcomes"]
    )
    assert metadata["config"] == expected_metadata["config"]


def test_generator_reproducibility_and_matching_explicit_cohort():
    first, first_oracle, first_metadata = generate_ipcw_data(
        **GENERATOR_PARAMS, return_ipcw_data=False
    )
    second, second_oracle, second_metadata = generate_ipcw_data(
        **GENERATOR_PARAMS, return_ipcw_data=False, main_cohorts=(6,)
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.attrs == second.attrs
    pd.testing.assert_frame_equal(
        first_oracle["individual_effects"], second_oracle["individual_effects"]
    )
    assert first_metadata["config"] == second_metadata["config"]


def test_oracle_effect_shift_reaches_estimator_without_outcome_leakage():
    # Shared innovations and treatment-independent features let us remove the
    # entire treatment effect while preserving all estimated nuisance weights.
    # Finite-sample confounding error need not be zero in either individual fit.
    data, oracle, _ = generate_ipcw_data(**GENERATOR_PARAMS)
    null_data, null_oracle, _ = generate_ipcw_data(
        **GENERATOR_PARAMS, scenario="null_effect"
    )

    def estimate(trial):
        return (
            IPCW(
                data=trial,
                ml_m=DummyClassifier(strategy="prior"),
                ml_c=DummyClassifier(strategy="prior"),
                n_folds=2,
                random_state=42,
            )
            .fit()
            .estimate(n_bootstrap=0)
        )

    result = estimate(data)
    null_result = estimate(null_data)
    monthly_truth = oracle["att_aggregate"]["att_no_activation"].to_numpy()
    np.testing.assert_allclose(
        result.period_effects["value"] - null_result.period_effects["value"],
        monthly_truth,
    )
    assert result.value - null_result.value == pytest.approx(monthly_truth.sum())
    assert result.control_mean == pytest.approx(null_result.control_mean)
    assert result.cumulative_effects["value"].iloc[-1] == pytest.approx(result.value)
    np.testing.assert_array_equal(
        null_oracle["att_aggregate"]["att_no_activation"], 0.0
    )
    pd.testing.assert_frame_equal(
        result.diagnostic_data.weight_summary,
        null_result.diagnostic_data.weight_summary,
    )
    checks = run_ipcw_diagnostics(result)
    assert list(checks.columns) == ["test", "flag", "value", "threshold", "message"]
    assert checks["flag"].isin(["GREEN", "YELLOW", "RED"]).all()


@pytest.mark.parametrize("g", [True, 0, -1, 1.5, "6"])
def test_generator_rejects_invalid_baseline(g):
    with pytest.raises(ValueError, match="g must be an integer"):
        generate_ipcw_data(g=g)


@pytest.mark.parametrize("main_cohorts", [(1, 6), (5,), (), None])
def test_generator_rejects_conflicting_oracle_cohorts(main_cohorts):
    with pytest.raises(ValueError, match="main_cohorts is fixed"):
        generate_ipcw_data(g=6, main_cohorts=main_cohorts)


@pytest.mark.parametrize("return_ipcw_data", [0, 1, "false", None])
def test_generator_rejects_nonboolean_return_switch(return_ipcw_data):
    with pytest.raises(ValueError, match="return_ipcw_data must be a boolean"):
        generate_ipcw_data(return_ipcw_data=return_ipcw_data)


def test_generator_passes_validation_to_organic_activation_config():
    with pytest.raises(ValueError, match="horizon must fit"):
        generate_ipcw_data(g=14, horizon=4, n_periods=16)
    with pytest.raises(ValueError, match="Unknown DGP parameters"):
        generate_ipcw_data(not_a_parameter=1)
