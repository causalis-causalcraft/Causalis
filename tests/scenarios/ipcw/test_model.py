"""Weight identities, client isolation and full-refit inference for IPCW."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from causalis.data_contracts import IPCWData, IPCWEstimate
from causalis.scenarios.ipcw import IPCW, generate_ipcw_data, run_ipcw_diagnostics


class FeatureProbability(BaseEstimator):
    fits = []

    def __init__(self, kind="baseline"):
        self.kind = kind

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        self.fits.append((self.kind, np.array(x, copy=True), np.array(y, copy=True)))
        return self

    def predict_proba(self, x):
        x = np.asarray(x)
        probability = (
            0.2 + 0.6 * x[:, 0] if self.kind == "baseline" else 0.1 + 0.1 * x[:, 1]
        )
        return np.column_stack([1 - probability, probability])


class BrokenProbability(FeatureProbability):
    def predict_proba(self, x):
        return np.full((len(x), 2), np.nan)


class HeterogeneousHazard(FeatureProbability):
    def predict_proba(self, x):
        probability = 0.05 + 0.4 * np.asarray(x)[:, 0]
        return np.column_stack([1 - probability, probability])


class ReversedProbability(FeatureProbability):
    def fit(self, x, y):
        super().fit(x, y)
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, x):
        return super().predict_proba(x)[:, ::-1]


class ThreeClassProbability(FeatureProbability):
    def fit(self, x, y):
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, x):
        return np.tile([0.2, 0.3, 0.5], (len(x), 1))


def _frame(n=64, horizon=3, censor=True):
    rows = []
    for client in range(n):
        treated = int(client < n // 2)
        event = None
        if not treated and censor and horizon > 1:
            if client % 4 == 0:
                event = 1
            elif client % 4 == 1 and horizon > 2:
                event = 2
        for month in range(horizon):
            rows.append(
                {
                    "client_id": client,
                    "trial_id": 10,
                    "period": 10 + month,
                    "followup_month": month,
                    "baseline_d": treated,
                    "b_z": client / (n - 1),
                    "z": client / (n - 1),
                    "at_risk_of_censoring": not treated
                    and month > 0
                    and (event is None or month <= event),
                    "censor_now": month == event,
                    "outcome_usable": event is None or month < event,
                    "ltv_monthly": (
                        50 + client / 10 + month if treated else client + month**2
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["model_features"] = {
        "baseline": ["b_z"],
        "censoring": ["z", "followup_month"],
    }
    return frame


def _model(data=None, **kwargs):
    return IPCW(
        data=data,
        ml_m=FeatureProbability(),
        ml_c=FeatureProbability("censoring"),
        n_folds=4,
        random_state=42,
        **kwargs,
    )


def test_manual_weights_monthly_normalization_and_total():
    frame = _frame()
    result = _model(IPCWData.from_df(frame)).fit().estimate(n_bootstrap=0)
    expected = []
    for month in range(3):
        part = frame.loc[frame.followup_month == month]
        treated = part.loc[part.baseline_d == 1, "ltv_monthly"].mean()
        controls = part.loc[(part.baseline_d == 0) & part.outcome_usable]
        e = 0.2 + 0.6 * controls.b_z.to_numpy()
        survival = np.prod([1 - (0.1 + 0.1 * s) for s in range(1, month + 1)])
        weights = e / (1 - e) / survival
        counterfactual = np.average(controls.ltv_monthly, weights=weights)
        expected.append(treated - counterfactual)
    np.testing.assert_allclose(result.period_effects.value, expected)
    np.testing.assert_allclose(result.cumulative_effects.value, np.cumsum(expected))
    assert result.value == result.cumulative_effects.value.iloc[-1]
    assert result.value == pytest.approx(result.treatment_mean - result.control_mean)
    assert result.n_treated == result.n_control == 32
    assert result.std_error is result.p_value is result.is_significant is None
    assert result.period_effects.std_error.isna().all()
    assert "horizon" in result.summary().index
    rows = result.diagnostic_data.row_data
    assert rows.loc[rows.followup_month == 0, "survival"].eq(1).all()
    assert rows.loc[~rows.outcome_usable, "control_weight"].eq(0).all()
    retained = rows.loc[
        (rows.treatment == 0) & (rows.followup_month == 2) & rows.outcome_usable
    ]
    np.testing.assert_allclose(retained.survival, 0.8 * 0.7)
    # A pooled weighted mean cannot replace monthly normalization.
    actual = frame.merge(
        rows[["unit_id", "followup_month", "control_weight"]],
        left_on=["client_id", "followup_month"],
        right_on=["unit_id", "followup_month"],
    )
    pooled = np.average(actual.ltv_monthly, weights=actual.control_weight) * 3
    assert not np.isclose(pooled, result.control_mean)


def test_client_specific_survival_changes_the_weighted_counterfactual():
    frame = _frame()
    result = (
        IPCW(
            IPCWData.from_df(frame),
            ml_m=FeatureProbability(),
            ml_c=HeterogeneousHazard(),
            n_folds=4,
            random_state=42,
        )
        .fit()
        .estimate(n_bootstrap=0)
    )
    without_censoring_adjustment = []
    expected = []
    for month in range(3):
        part = frame.loc[frame.followup_month == month]
        controls = part.loc[(part.baseline_d == 0) & part.outcome_usable]
        e = 0.2 + 0.6 * controls.b_z.to_numpy()
        odds = e / (1 - e)
        survival = (1 - (0.05 + 0.4 * controls.z.to_numpy())) ** month
        expected.append(np.average(controls.ltv_monthly, weights=odds / survival))
        without_censoring_adjustment.append(
            np.average(controls.ltv_monthly, weights=odds)
        )
    np.testing.assert_allclose(result.period_effects.control_mean, expected)
    assert not np.allclose(expected[1:], without_censoring_adjustment[1:])


def test_binary_class_order_is_resolved_and_multiclass_predictions_are_rejected():
    data = IPCWData.from_df(_frame())
    standard = _model(data).fit().estimate(n_bootstrap=0)
    reversed_result = (
        IPCW(
            data,
            ml_m=ReversedProbability(),
            ml_c=ReversedProbability("censoring"),
            random_state=42,
        )
        .fit()
        .estimate(n_bootstrap=0)
    )
    assert reversed_result.value == standard.value
    with pytest.raises(ValueError, match="binary classes"):
        IPCW(data, ml_m=ThreeClassProbability()).fit()


@pytest.mark.parametrize("horizon,censor", [(1, True), (3, False)])
def test_no_censoring_retention_is_exactly_one_without_censor_learner_fit(
    horizon, censor
):
    FeatureProbability.fits.clear()
    result = (
        _model(IPCWData.from_df(_frame(horizon=horizon, censor=censor)))
        .fit()
        .estimate(n_bootstrap=0)
    )
    assert result.diagnostic_data.row_data.survival.eq(1).all()
    assert {kind for kind, _, _ in FeatureProbability.fits} == {"baseline"}


def test_training_uses_baseline_once_and_only_current_at_risk_histories():
    FeatureProbability.fits.clear()
    data = IPCWData.from_df(_frame())
    model = _model(data).fit()
    result = model.estimate(n_bootstrap=0)
    rows = result.diagnostic_data.row_data
    baseline_fits = [x for kind, x, _ in FeatureProbability.fits if kind == "baseline"]
    hazard_fits = [
        (x, y) for kind, x, y in FeatureProbability.fits if kind == "censoring"
    ]
    assert len(baseline_fits) == len(hazard_fits) == 4
    data_frame = data.get_df()
    for fold, (x, (censor_x, censor_y)) in enumerate(zip(baseline_fits, hazard_fits)):
        heldout = rows.loc[rows.fold == fold, "unit_id"].unique()
        expected_baseline = data_frame.loc[
            (data_frame.followup_month == 0) & ~data_frame.client_id.isin(heldout)
        ]
        np.testing.assert_allclose(x[:, 0], expected_baseline.b_z)
        expected_risk = data_frame.loc[
            data_frame.at_risk_of_censoring & ~data_frame.client_id.isin(heldout)
        ]
        np.testing.assert_allclose(censor_x, expected_risk[["z", "followup_month"]])
        np.testing.assert_array_equal(censor_y, expected_risk.censor_now.astype(int))
        assert censor_y.any()  # Event rows are retained in training.
        assert not np.isin(censor_x[:, 0], heldout / 63).any()
    assert rows.groupby("unit_id").fold.nunique().eq(1).all()
    assert not hasattr(model.ml_m, "classes_")


def test_unused_values_and_input_mutation_do_not_change_fitted_estimate():
    original = _frame()
    changed = original.copy(deep=True)
    changed.loc[~changed.outcome_usable, "ltv_monthly"] = np.nan
    changed.loc[~changed.at_risk_of_censoring, "z"] = np.nan
    a = _model(IPCWData.from_df(original)).fit()
    data = IPCWData.from_df(changed)
    b = _model(data).fit()
    changed.loc[:, "ltv_monthly"] = -100000
    data.df.loc[:, "ltv_monthly"] = 100000
    assert a.estimate(n_bootstrap=0).value == b.estimate(n_bootstrap=0).value
    # Changing the fitted estimator's public data reference cannot alter its snapshot.
    a.data = None
    assert a.estimate(n_bootstrap=0).value == b.estimate(n_bootstrap=0).value


def test_censoring_clips_retention_only_from_below():
    result = (
        _model(IPCWData.from_df(_frame()), censoring_clip=0.9, propensity_clip=0.3)
        .fit()
        .estimate(n_bootstrap=0)
    )
    rows = result.diagnostic_data.row_data
    assert rows.loc[rows.at_risk, "retention"].eq(0.9).all()
    assert rows.loc[~rows.at_risk, "retention"].eq(1).all()
    assert rows.censoring_clipped.any()
    assert rows.propensity.between(0.3, 0.7).all()


def test_bootstrap_refits_both_models_and_uses_joint_cumulative_draws():
    FeatureProbability.fits.clear()
    model = _model(IPCWData.from_df(_frame())).fit()
    FeatureProbability.fits.clear()
    result = model.estimate(n_bootstrap=6)
    baseline_fits = [x for kind, x, _ in FeatureProbability.fits if kind == "baseline"]
    assert len(baseline_fits) == 6 * 4
    assert sum(kind == "censoring" for kind, _, _ in FeatureProbability.fits) == 6 * 4
    assert any(len(np.unique(x[:, 0])) < len(x) for x in baseline_fits)
    # Duplicate copies remain in the same original client fold.
    fold_by_z = dict(zip(model._prepared_.x_baseline[:, 0], model._prepared_.folds))
    for index, x in enumerate(baseline_fits):
        excluded_fold = index % 4
        assert all(fold_by_z[z] != excluded_fold for z in x[:, 0])
    assert result.std_error > 0
    assert result.std_error == result.cumulative_effects.std_error.iloc[-1]
    assert result.ci_lower_absolute < result.value < result.ci_upper_absolute
    assert result.p_value is not None
    assert result.model_options["n_bootstrap"] == 6
    from causalis.scenarios.ipcw.model import _bootstrap_replication

    prepared = model._prepared_
    draws = np.array(
        [
            _bootstrap_replication(
                i,
                prepared,
                model._baseline_learner_,
                model._censoring_learner_,
                0.01,
                0.01,
            )
            for i in range(6)
        ]
    )
    np.testing.assert_allclose(
        result.period_effects.std_error, draws.std(axis=0, ddof=1)
    )
    assert result.std_error == pytest.approx(draws.sum(axis=1).std(ddof=1))


def test_parallel_seed_repeatability_and_diagnostics_toggle():
    data = IPCWData.from_df(_frame())
    sequential = _model(data, n_jobs=1).fit().estimate(n_bootstrap=6)
    parallel = _model(data, n_jobs=2).fit().estimate(n_bootstrap=6)
    light = _model(data, store_diagnostics=False).fit().estimate(n_bootstrap=6)
    for other in (parallel, light):
        assert other.value == sequential.value
        assert other.std_error == sequential.std_error
        pd.testing.assert_frame_equal(other.period_effects, sequential.period_effects)
    assert light.diagnostic_data is None


def test_bootstrap_support_failure_is_numbered_without_losing_point_estimate():
    data = IPCWData.from_df(_frame(n=8, horizon=1))
    model = _model(data).fit()
    with pytest.raises(RuntimeError, match=r"replication \d+ failed.*fold"):
        model.estimate(n_bootstrap=50)
    assert np.isfinite(model.estimate(n_bootstrap=0).value)


def test_default_catboost_end_to_end_with_oracle_and_diagnostics():
    data, oracle, _ = generate_ipcw_data(
        n_clients=600, n_periods=16, g=6, horizon=3, entry_max=5, seed=29
    )
    model = IPCW(random_state=7, n_folds=2).fit(data)
    result = model.estimate(n_bootstrap=2)
    assert isinstance(result, IPCWEstimate)
    assert np.isfinite(result.value)
    assert result.std_error is not None
    assert len(result.period_effects) == len(oracle["att_aggregate"]) == 3
    assert set(run_ipcw_diagnostics(result).flag) <= {"GREEN", "YELLOW", "RED"}
    assert model._baseline_learner_.get_params()["iterations"] == 200


@pytest.mark.parametrize(
    "option,value",
    [
        ("n_folds", 1),
        ("n_folds", True),
        ("n_folds", 2.5),
        ("n_jobs", 0),
        ("n_jobs", -2),
        ("n_jobs", -1.0),
        ("n_jobs", True),
        ("random_state", -1),
        ("propensity_clip", 0),
        ("propensity_clip", 0.5),
        ("censoring_clip", np.nan),
        ("censoring_clip", 1),
        ("store_diagnostics", "yes"),
    ],
)
def test_invalid_configuration(option, value):
    with pytest.raises(ValueError, match=option):
        IPCW(**{option: value}).fit(IPCWData.from_df(_frame()))


def test_lifecycle_support_and_probability_validation():
    data = IPCWData.from_df(_frame())
    with pytest.raises(RuntimeError, match="fit"):
        IPCW().estimate(n_bootstrap=0)
    with pytest.raises(ValueError, match="IPCWData"):
        IPCW().fit()
    with pytest.raises(ValueError, match="n_folds"):
        IPCW(n_folds=40).fit(data)
    with pytest.raises(ValueError, match="predict_proba"):
        IPCW(ml_m=object()).fit(data)
    with pytest.raises(ValueError, match="probabilities"):
        IPCW(ml_m=BrokenProbability()).fit(data)
    model = _model(data).fit()
    for repetitions in (1, -1, 2.5, True):
        with pytest.raises(ValueError, match="n_bootstrap"):
            model.estimate(n_bootstrap=repetitions)
    for alpha in (0, 1, np.nan):
        with pytest.raises(ValueError, match="alpha"):
            model.estimate(alpha=alpha, n_bootstrap=0)


def test_no_usable_controls_has_clear_error():
    frame = _frame(horizon=2)
    controls = frame.baseline_d == 0
    frame.loc[controls & (frame.followup_month == 1), "censor_now"] = True
    frame.loc[controls & (frame.followup_month == 1), "outcome_usable"] = False
    with pytest.raises(ValueError, match="support|usable controls|only events"):
        _model(IPCWData.from_df(frame)).fit()
