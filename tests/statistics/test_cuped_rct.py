"""RCT batch CUPED must reproduce independent binary analyses exactly."""
import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import CausalData, RctCausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.scenarios.cuped import CUPEDModel


@pytest.fixture
def data():
    rng = np.random.default_rng(72)
    n = 360
    arm = np.arange(n) % 3
    x = rng.normal(size=n) + arm
    df = pd.DataFrame({
        "control": arm == 0, "a": arm == 1, "b": arm == 2, "x": x,
        "y1": 20 + 3 * arm + x * (1 + arm) + rng.normal(size=n),
        "y2": 30 - 2 * arm - x * (2 + arm) + rng.normal(size=n),
    }, index=np.arange(n) // 2)
    return RctCausalData.from_df(
        df, ["a", "control", "b"], ["y1", "y2"], "x", control_treatment="control",
    )


def binary(data, outcome, arm):
    frame = data.df.loc[(data.df.control == 1) | (data.df[arm] == 1)]
    return CausalData.from_df(frame, arm, outcome, data.confounders_names)


def assert_estimates_equal(actual, expected):
    fields = ["value", "p_value", "ci_lower_absolute", "ci_upper_absolute",
              "value_relative", "ci_lower_relative", "ci_upper_relative",
              "n_treated", "n_control", "treatment_mean", "control_mean",
              "adjusted_treatment_mean", "adjusted_control_mean"]
    for field in fields:
        assert getattr(actual, field) == pytest.approx(getattr(expected, field), nan_ok=True, abs=1e-10)
    assert actual.outcome == expected.outcome
    assert actual.treatment == expected.treatment
    assert actual.alpha == expected.alpha
    if actual.diagnostic_data is not None:
        for field in ["se_naive", "r2_adj", "r2_naive", "variance_reduction_pct_same_cov"]:
            assert getattr(actual.diagnostic_data, field) == pytest.approx(getattr(expected.diagnostic_data, field))
        np.testing.assert_allclose(actual.diagnostic_data.beta_covariates, expected.diagnostic_data.beta_covariates)
        np.testing.assert_allclose(actual.diagnostic_data.gamma_interactions, expected.diagnostic_data.gamma_interactions)


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3", "nonrobust"])
@pytest.mark.parametrize("covariates", [[], ["x"]])
def test_batch_matches_independent_pairwise_models(data, cov_type, covariates):
    model = CUPEDModel(cov_type=cov_type).fit(data, covariates=covariates, run_checks=False)
    estimates = model.estimate(alpha=0.1)
    assert list(estimates) == ["y1", "y2"]
    for outcome, arms in estimates.items():
        assert list(arms) == ["a", "b"]
        for arm, actual in arms.items():
            reference = CUPEDModel(cov_type=cov_type).fit(
                binary(data, outcome, arm), covariates=covariates, run_checks=False,
            ).estimate(alpha=0.1)
            assert_estimates_equal(actual, reference)
            assert actual.model_options["control_treatment"] == "control"
            assert actual.n_control == actual.n_treated == 120


def test_shared_design_and_input_unchanged(data, monkeypatch):
    original = data.df.copy(deep=True)
    def unexpected(*args, **kwargs):
        raise AssertionError("Must not revalidate CausalData for each outcome")
    monkeypatch.setattr(CausalData, "from_df", unexpected)
    model = CUPEDModel().fit(data, covariates=["x"], run_checks=False)
    for arm in ["a", "b"]:
        first = model._comparison_models["y1"][arm]
        second = model._comparison_models["y2"][arm]
        assert first._data.df is second._data.df
        for attr in ["_result", "_result_naive"]:
            first_ols = getattr(first, attr).model
            second_ols = getattr(second, attr).model
            assert np.shares_memory(first_ols.exog, second_ols.exog)
            assert first_ols.pinv_wexog is second_ols.pinv_wexog
    pd.testing.assert_frame_equal(data.df, original)


def test_selectors_summaries_and_checks(data):
    model = CUPEDModel().fit(data, covariates=["x"])
    effect = model.estimate(outcome="y2", treatment="b", diagnostic_data=False)
    assert isinstance(effect, CausalEstimate)
    assert effect.diagnostic_data is None
    selected = CUPEDModel().fit(data, covariates=["x"], outcome="y2", treatment="b")
    assert_estimates_equal(effect, selected.estimate(diagnostic_data=False))
    assert model.summary_dict()["y2"]["b"]["ate"] == pytest.approx(effect.value)
    assert model.summary_dict(outcome="y2", treatment="b")["ate"] == pytest.approx(effect.value)
    table = model.assumptions_table()
    assert set(table.outcome) == {"y1", "y2"}
    assert set(table.treatment) == {"a", "b"}
    assert set(model.assumptions_table(treatment="a").treatment) == {"a"}
    table.iloc[0, table.columns.get_loc("outcome")] = "modified"
    assert "modified" not in set(model.assumptions_table().outcome)
    assert list(model.estimate(outcome="y1")) == ["y1"]
    assert list(model.estimate(treatment="a")) == ["y1", "y2"]


@pytest.mark.parametrize("denominator", ["adjusted_control", "raw_control"])
@pytest.mark.parametrize("method", ["delta", "bootstrap"])
def test_relative_configuration_propagates(data, denominator, method):
    options = dict(relative_ci_method=method, relative_denominator=denominator,
                   relative_ci_bootstrap_draws=30, relative_ci_bootstrap_seed=14, use_t=False)
    model = CUPEDModel(**options).fit(data, covariates=["x"], run_checks=False)
    actual = model.estimate(outcome="y2", treatment="b")
    expected = CUPEDModel(**options).fit(binary(data, "y2", "b"), covariates=["x"], run_checks=False).estimate()
    assert_estimates_equal(actual, expected)
    assert model.assumptions_table() is None


def test_single_binary_and_multiple_binary_outcomes(data):
    frame = binary(data, "y1", "a").df.copy()
    frame["y2"] = frame.y1 * 2 + np.arange(len(frame)) % 4
    rct = RctCausalData.from_df(frame, "a", ["y1", "y2"], "x")
    model = CUPEDModel().fit(rct, covariates=["x"], run_checks=False)
    assert list(model.estimate()) == ["y1", "y2"]
    actual = model.estimate(outcome="y1")
    reference = CUPEDModel().fit(binary(data, "y1", "a"), covariates=["x"], run_checks=False).estimate()
    assert_estimates_equal(actual, reference)
    one = RctCausalData.from_df(frame, "a", "y1", "x")
    model.fit(one, covariates=["x"], run_checks=False)
    assert isinstance(model.estimate(), CausalEstimate)
    assert_estimates_equal(model.estimate(), reference)


def test_invalid_selectors_and_refit_state(data):
    model = CUPEDModel().fit(data, covariates=["x"], run_checks=False)
    for kwargs in [dict(outcome="missing"), dict(treatment="control")]:
        with pytest.raises(ValueError):
            model.estimate(**kwargs)
        with pytest.raises(ValueError):
            model.summary_dict(**kwargs)
        with pytest.raises(ValueError):
            model.assumptions_table(**kwargs)
    with pytest.raises(ValueError):
        model.fit(data, covariates=["x"], treatment="control")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.estimate()
    model.fit(binary(data, "y1", "a"), covariates=["x"], run_checks=False)
    assert isinstance(model.estimate(), CausalEstimate)
    with pytest.raises(ValueError):
        model.estimate(outcome="y2")
    with pytest.raises(ValueError):
        model.fit(data, covariates=["y1"])
    with pytest.raises(RuntimeError):
        model.estimate()


def test_pair_specific_covariate_drop_and_failed_batch(data):
    df = data.df.copy()
    df["z"] = df.b * (np.arange(len(df)) + 1)
    rct = RctCausalData.from_df(df, data.treatment_names, data.outcome_names, ["z"], control_treatment="control")
    # z is constant in control/a, and perfectly confounded in control/b.
    model = CUPEDModel()
    with pytest.raises(ValueError, match="rank deficient"):
        model.fit(rct, covariates=["z"], run_checks=False)
    with pytest.raises(RuntimeError):
        model.estimate()
    model.fit(rct, covariates=["z"], treatment="a", run_checks=False)
    assert model.estimate()["y1"]["a"].model_options["dropped_covariates"] == ["z"]


def test_batch_summary_matches_causal_estimate_format(data):
    from causalis.data_contracts import RctEstimates
    estimates = CUPEDModel().fit(data, covariates=["x"], run_checks=False).estimate()
    assert isinstance(estimates, RctEstimates)
    assert isinstance(estimates, dict)
    summary = estimates.summary(outcome="y1")
    assert list(summary.columns) == ["a vs control", "b vs control"]
    assert summary.columns.name == "comparison"
    for arm in ["a", "b"]:
        effect = estimates["y1"][arm]
        assert effect.adjusted_treatment_mean - effect.adjusted_control_mean == pytest.approx(effect.value)
        assert summary.loc["adjusted_control_mean", f"{arm} vs control"] == f"{effect.adjusted_control_mean:.4f}"
        diag = effect.diagnostic_data
        assert summary.loc["variance_reduction_pct_same_cov", f"{arm} vs control"] == (
            f"{diag.variance_reduction_pct_same_cov:.4f}"
        )
    for arm in ["a", "b"]:
        pd.testing.assert_series_equal(
            summary[f"{arm} vs control"], estimates["y1"][arm].summary()["value"],
            check_names=False,
        )
    all_outcomes = estimates.summary()
    assert all_outcomes.columns.names == ["outcome", "comparison"]
    pd.testing.assert_frame_equal(all_outcomes["y1"], summary)
    selected = estimates.summary(outcome="y2", treatment="b")
    assert list(selected.columns) == ["b vs control"]
    assert selected.loc["outcome", "b vs control"] == "y2"
    summary.loc["value", "a vs control"] = "edited"
    assert estimates.summary(outcome="y1").loc["value", "a vs control"] != "edited"
    # Native pandas HTML rendering used by Jupyter remains available.
    assert "a vs control" in estimates.summary(outcome="y1").to_html()


def test_batch_summary_filters_and_empty_mapping(data):
    from causalis.data_contracts import RctEstimates
    estimates = CUPEDModel().fit(data, covariates=[], run_checks=False).estimate(diagnostic_data=False)
    assert "variance_reduction_pct_same_cov" not in estimates.summary(outcome="y1").index
    with pytest.raises(ValueError, match="Unknown outcome"):
        estimates.summary(outcome="missing")
    with pytest.raises(ValueError, match="Unknown treatment"):
        estimates.summary(outcome="y1", treatment="control")
    assert list(estimates.summary(treatment="a").columns) == [
        ("y1", "a vs control"), ("y2", "a vs control"),
    ]
    assert RctEstimates().summary().empty


def test_binary_batch_summary_labels(data):
    frame = binary(data, "y1", "a").df.copy()
    frame["y2"] = frame.y1 * 2 + np.arange(len(frame)) % 4
    rct = RctCausalData.from_df(frame, "a", ["y1", "y2"], "x")
    estimates = CUPEDModel().fit(rct, covariates=["x"], run_checks=False).estimate()
    assert list(estimates.summary(outcome="y1").columns) == ["a=1 vs a=0"]
