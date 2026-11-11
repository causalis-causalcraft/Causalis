"""Independent score and sample-lifecycle checks for one-way clustered IRM."""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import norm, t
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from statsmodels.stats.sandwich_covariance import cov_cluster

from causalis.data_contracts.causaldata import CausalData
from causalis.scenarios.gate.model import estimate_gate_from_irm, estimate_gatet_from_irm
from causalis.scenarios.unconfoundedness.model import IRM


class _FeaturePropensity(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        propensity = np.asarray(X)[:, 1]
        return np.column_stack((1 - propensity, propensity))


def _data(*, clustered=True, sizes=(7, 10, 8, 9, 6, 11, 8, 7, 12, 9, 6, 10), baseline=20.0):
    rng = np.random.default_rng(196)
    cluster = np.repeat(np.arange(len(sizes)), sizes)
    n = len(cluster)
    treatment = np.concatenate([np.arange(size) % 2 for size in sizes])
    x = rng.normal(size=n)
    propensity = rng.uniform(0.2, 0.8, size=n)
    shock = rng.normal(scale=1.8, size=len(sizes))[cluster]
    outcome = baseline + 2 * treatment + 0.7 * x + shock + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame(
        {"y": outcome, "d": treatment, "x": x, "p": propensity,
         "uid": [f"u{i}" for i in range(n)], "cluster": cluster},
        index=pd.Index(np.arange(n) // 2, name="nonunique_row_index"),
    )
    return _contract(frame, clustered=clustered)


def _contract(frame, *, clustered=True):
    return CausalData.from_df(
        frame, "d", "y", ["x", "p"], "uid",
        cluster="cluster" if clustered else None,
    )


def _model(data, **kwargs):
    options = dict(ml_g=LinearRegression(), ml_m=DummyClassifier(strategy="prior"),
                   n_folds=3, random_state=17)
    options.update(kwargs)
    return IRM(data, **options)


def _scores(model, score):
    """Canonical unweighted scores, derived directly from the AIPW formulas."""
    y, d = model._y, model._d
    g0, g1, p = model.g0_hat_, model.g1_hat_, model.m_hat_
    if score == "ATE":
        signal = g1 - g0 + d * (y - g1) / p - (1 - d) * (y - g0) / (1 - p)
        baseline_signal = g0 + (1 - d) * (y - g0) / (1 - p)
        weight = np.ones(len(y))
    else:
        treated_share = d.mean()
        control_correction = (1 - d) * p * (y - g0) / (1 - p)
        signal = (d * (y - g0) - control_correction) / treated_share
        baseline_signal = (d * g0 + control_correction) / treated_share
        weight = d / treated_share
    effect, baseline = signal.mean(), baseline_signal.mean()
    return effect, baseline, signal - weight * effect, baseline_signal - weight * baseline


def _cluster_variance(influence, codes):
    # Deliberately use direct cluster selection, independent of production bincount.
    sums = np.array([influence[codes == group].sum() for group in np.unique(codes)])
    count = len(sums)
    return count / (count - 1) * (sums @ sums) / len(influence) ** 2


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_cluster_absolute_inference_matches_scores_and_statsmodels(score):
    data = _data()
    model = _model(data).fit()
    result = model.estimate(score=score, alpha=0.1)
    effect, _, influence, _ = _scores(model, score)
    codes = pd.factorize(data.cluster, sort=False)[0]
    expected_var = _cluster_variance(influence, codes)
    oracle = sm.OLS(influence, np.ones((len(influence), 1))).fit()
    np.testing.assert_allclose(cov_cluster(oracle, codes)[0, 0], expected_var, rtol=1e-12)
    np.testing.assert_allclose(result.value, effect, rtol=1e-12)
    np.testing.assert_allclose(model.psi_, influence, atol=1e-12)
    np.testing.assert_allclose(model.se_[0] ** 2, expected_var, rtol=1e-12)
    se = np.sqrt(expected_var)
    df = len(np.unique(codes)) - 1
    critical = t.ppf(0.95, df)
    np.testing.assert_allclose([result.ci_lower_absolute, result.ci_upper_absolute],
                               [effect - critical * se, effect + critical * se])
    np.testing.assert_allclose(result.p_value, 2 * t.sf(abs(effect / se), df))
    np.testing.assert_allclose(model.confint(alpha=0.2).iloc[0],
                               [effect - t.ppf(0.9, df) * se, effect + t.ppf(0.9, df) * se])
    metadata = result.model_options
    assert metadata["cov_type"] == "cluster"
    assert metadata["cluster_name"] == "cluster"
    assert metadata["n_clusters"] == metadata["n_clusters_fit"] == len(np.unique(codes))
    assert metadata["inference_distribution"] == "t"
    assert metadata["inference_df"] == df
    assert metadata["covariance_correction"] == pytest.approx((df + 1) / df)
    assert result.summary().loc["cov_type", "value"] == "cluster"


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
@pytest.mark.parametrize("baseline", [20.0, -20.0])
def test_cluster_relative_inference_preserves_joint_covariance(score, baseline):
    model = _model(_data(baseline=baseline)).fit()
    result = model.estimate(score=score, alpha=0.1)
    effect, baseline, influence, baseline_influence = _scores(model, score)
    relative_influence = 100 * (influence / baseline - effect * baseline_influence / baseline**2)
    expected_se = np.sqrt(_cluster_variance(relative_influence, model.cluster_codes_))
    relative = 100 * effect / baseline
    critical = t.ppf(0.95, model.n_clusters_ - 1)
    np.testing.assert_allclose(result.value_relative, relative)
    np.testing.assert_allclose(model.se_relative_[0], expected_se, rtol=1e-12)
    np.testing.assert_allclose(result.model_options["std_error_relative"], expected_se)
    np.testing.assert_allclose([result.ci_lower_relative, result.ci_upper_relative],
                               [relative - critical * expected_se, relative + critical * expected_se])


def test_iid_atte_relative_inference_accounts_for_treated_normalizer():
    model = _model(_data(clustered=False)).fit()
    result = model.estimate(score="ATTE")
    effect, baseline, influence, baseline_influence = _scores(model, "ATTE")
    relative_influence = 100 * (influence / baseline - effect * baseline_influence / baseline**2)
    expected_se = np.std(relative_influence, ddof=1) / np.sqrt(len(influence))
    np.testing.assert_allclose(model.se_relative_[0], expected_se, rtol=1e-12)
    critical = norm.ppf(0.975)
    np.testing.assert_allclose([result.ci_lower_relative, result.ci_upper_relative],
                               [result.value_relative - critical * expected_se,
                                result.value_relative + critical * expected_se])


def test_relative_baseline_guard_uses_cluster_variance():
    values = np.repeat(np.array([-3, -2, -1, 1, 2, 3]) + 0.8, 10)
    frame = pd.DataFrame({"x": values, "p": np.linspace(0.2, 0.8, 60),
                          "d": np.tile([0, 1], 30), "cluster": np.repeat(np.arange(6), 10),
                          "uid": [f"u{i}" for i in range(60)]})
    frame["y"] = frame.x + 2 * frame.d
    clustered = _model(_contract(frame)).fit()
    iid = _model(_contract(frame, clustered=False)).fit()
    with pytest.warns(RuntimeWarning, match="Relative effect baseline"):
        result = clustered.estimate()
    assert np.isnan(result.value_relative)
    with warnings.catch_warnings(record=True) as recorded:
        iid_result = iid.estimate()
    assert not any("Relative effect baseline" in str(item.message) for item in recorded)
    assert np.isfinite(iid_result.value_relative)


def test_cluster_folds_are_disjoint_reproducible_and_parallel_equivalent():
    data = _data()
    sequential = _model(data).fit()
    repeated = _model(data).fit()
    parallel = _model(data, n_jobs=2).fit()
    for label in pd.unique(data.cluster):
        assert np.unique(sequential.folds_[data.cluster.to_numpy() == label]).size == 1
    cluster_folds = sequential.folds_[np.r_[0, np.cumsum(np.bincount(sequential.cluster_codes_))[:-1]]]
    counts = np.bincount(cluster_folds)
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(sequential.folds_, repeated.folds_)
    np.testing.assert_array_equal(sequential.folds_, parallel.folds_)
    for name in ("g0_hat_", "g1_hat_", "m_hat_"):
        np.testing.assert_allclose(getattr(sequential, name), getattr(parallel, name))
    np.testing.assert_allclose(sequential.estimate().value, parallel.estimate().value)
    np.testing.assert_allclose(sequential.se_, parallel.se_)


@pytest.mark.parametrize("kind", ["string", "mixed", "tuple", "categorical"])
def test_opaque_cluster_labels_do_not_change_estimation(kind):
    numeric = _data()
    frame = numeric.df.copy()
    ids = frame.cluster.to_numpy()
    unique = pd.unique(ids)
    if kind == "string":
        labels = [f"cluster-{index}" for index in unique]
    elif kind == "mixed":
        labels = [index if index % 2 else f"c{index}" for index in unique]
    elif kind == "tuple":
        labels = [("cluster", index) for index in unique]
    else:
        labels = [f"c{index}" for index in unique]
    values = pd.Series([labels[index] for index in ids], dtype=object)
    if kind == "categorical":
        values = pd.Series(pd.Categorical(values, categories=labels + ["unused"]))
    frame["cluster"] = values.array
    expected = _model(numeric).fit()
    actual = _model(_contract(frame)).fit()
    np.testing.assert_array_equal(actual.cluster_codes_, expected.cluster_codes_)
    np.testing.assert_array_equal(actual.folds_, expected.folds_)
    np.testing.assert_allclose(actual.estimate().value, expected.estimate().value)
    np.testing.assert_allclose(actual.se_, expected.se_)


def test_fit_rejects_more_folds_than_clusters_before_training(monkeypatch):
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid cluster support must fail before fitting a learner")
    monkeypatch.setattr(IRM, "_fit_nuisances_for_fold", unexpected_fit)
    with pytest.raises(ValueError, match="(?i)cluster"):
        _model(_data(), n_folds=13).fit()


def test_fit_rejects_treatment_arm_confined_to_one_cluster(monkeypatch):
    data = _data()
    data.df["d"] = (data.df.cluster == 0).astype(int)
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid cluster support must fail before fitting a learner")
    monkeypatch.setattr(IRM, "_fit_nuisances_for_fold", unexpected_fit)
    with pytest.raises(ValueError, match="(?i)cluster"):
        _model(data).fit()


def test_fixed_folds_cannot_split_a_cluster():
    data = _data()
    model = _model(data)
    model._fixed_fold_assignments_ = data.cluster.to_numpy() % 3
    model._fixed_fold_assignments_[0] = 1
    with pytest.raises(ValueError, match="(?i)cluster"):
        model.fit()


def test_fixed_folds_validate_training_treatment_support_before_learning(monkeypatch):
    data = _data(sizes=(6, 6, 6, 6))
    data.df["d"] = (data.df.cluster < 2).astype(int)
    model = _model(data)
    model._fixed_fold_assignments_ = np.repeat([0, 0, 1, 2], 6)
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid training support must fail before fitting a learner")
    monkeypatch.setattr(IRM, "_fit_nuisances_for_fold", unexpected_fit)
    with pytest.raises(ValueError, match="(?i)training|fold|treatment"):
        model.fit()


def test_heldout_fold_does_not_need_both_treatment_arms():
    data = _data(sizes=(6, 6, 6, 6))
    data.df["d"] = (data.df.cluster < 2).astype(int)
    model = _model(data, n_folds=4).fit()
    assert all(np.unique(model._d[model.folds_ == fold]).size == 1 for fold in range(4))
    assert np.isfinite(model.estimate().model_options["std_error"])


def test_cluster_folds_do_not_require_minority_row_count_to_exceed_fold_count():
    data = _data(sizes=(6, 6, 6, 6))
    data.df["d"] = 0
    data.df.iloc[[0, 6], data.df.columns.get_loc("d")] = 1
    model = _model(data, n_folds=4).fit()
    assert model._d.sum() == 2
    assert np.isfinite(model.estimate(score="ATTE").model_options["std_error"])


def test_unlucky_cluster_partition_fails_before_learning_without_resampling(monkeypatch):
    data = _data(sizes=(6, 6, 6, 6))
    data.df["d"] = (data.df.cluster < 2).astype(int)
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid training support must fail before fitting a learner")
    monkeypatch.setattr(IRM, "_fit_nuisances_for_fold", unexpected_fit)
    with pytest.raises(ValueError, match="random_state"):
        _model(data, n_folds=2, random_state=0).fit()


@pytest.mark.parametrize("invalid", [None, ["unhashable"]], ids=["missing", "unhashable"])
def test_fit_revalidates_mutated_cluster_labels(invalid):
    data = _data()
    labels = data.cluster.to_numpy(dtype=object, copy=True)
    labels[0] = invalid
    data.df["cluster"] = labels
    with pytest.raises(ValueError, match="(?i)cluster"):
        _model(data).fit()


def test_sample_fingerprint_detects_changed_cluster_membership():
    data = _data()
    model = _model(data).fit()
    data.df.iloc[0, data.df.columns.get_loc("cluster")] = 1
    X, y, d, _ = model._check_data()
    with pytest.raises(RuntimeError, match="changed after fit"):
        model._validate_current_data_matches_fit(X=X, y=y, d=d)


def test_cluster_snapshot_is_readonly_and_diagnostics_mode_independent():
    data = _data()
    full = _model(data).fit()
    light = _model(data, store_diagnostics=False).fit()
    expected = full.estimate(score="ATTE")
    actual = light.estimate(score="ATTE")
    np.testing.assert_array_equal(full.cluster_codes_, light.cluster_codes_)
    assert full.cluster_codes_.flags.writeable is False
    assert light.cluster_codes_.flags.writeable is False
    assert np.shares_memory(expected.diagnostic_data.cluster_codes, full.cluster_codes_)
    assert expected.diagnostic_data.cluster_name == "cluster"
    assert actual.diagnostic_data is None
    np.testing.assert_allclose(actual.value, expected.value)
    np.testing.assert_allclose(light.se_, full.se_)
    data.df["cluster"] = "changed-after-fit"
    repeated = light.estimate(score="ATTE")
    np.testing.assert_allclose(repeated.value, expected.value)
    np.testing.assert_allclose(light.se_, full.se_)
    assert repeated.model_options["n_clusters"] == 12


def test_overlap_drop_keeps_cluster_codes_aligned_and_recounts_clusters():
    data = _data()
    values = data.df["p"].to_numpy().copy()
    original = data.cluster.to_numpy()
    values[original == 0] = 0.01
    values[np.flatnonzero(original == 1)[:3]] = 0.99
    data.df["p"] = values
    model = _model(data, ml_m=_FeaturePropensity(), overlap_policy="drop", overlap_threshold=0.1).fit()
    retained = (values > 0.1) & (values < 0.9)
    expected_codes = pd.factorize(original[retained], sort=False)[0]
    np.testing.assert_array_equal(model.cluster_codes_, expected_codes)
    np.testing.assert_array_equal(model.overlap_mask_, retained)
    assert model.n_clusters_fit_ == 12
    assert model.n_clusters_ == 11
    result = model.estimate()
    assert result.model_options["n_clusters"] == 11
    assert result.model_options["inference_df"] == 10
    assert len(result.diagnostic_data.cluster_codes) == retained.sum()


def test_overlap_drop_rejects_only_one_retained_cluster():
    data = _data()
    data.df["p"] = np.where(data.cluster == 0, 0.5, 0.01)
    with pytest.raises(ValueError, match="(?i)cluster"):
        _model(data, ml_m=_FeaturePropensity(), overlap_policy="drop", overlap_threshold=0.1).fit()


@pytest.mark.parametrize("score", ["GATE", "GATET"])
@pytest.mark.parametrize("entrypoint", ["estimate", "convenience", "direct"])
def test_clustered_subgroup_inference_is_explicitly_unsupported(score, entrypoint):
    model = _model(_data()).fit()
    with pytest.raises(NotImplementedError, match="(?i)cluster"):
        if entrypoint == "estimate":
            model.estimate(score=score)
        elif entrypoint == "convenience":
            getattr(model, score.lower())(groups=None)
        else:
            function = estimate_gate_from_irm if score == "GATE" else estimate_gatet_from_irm
            function(model, groups=None)


def test_clustered_weighted_hajek_inference_preserves_approximation_flags():
    data = _data()
    weights = np.linspace(0.5, 1.5, len(data.df))
    model = _model(data, weights=weights, normalize_ipw=True).fit()
    with pytest.warns(RuntimeWarning) as recorded:
        result = model.estimate()
    messages = [str(item.message) for item in recorded]
    assert any("Hájek" in message for message in messages)
    assert any("normalized by sample mean" in message for message in messages)
    assert result.model_options["se_approx_hajek"] is True
    assert result.model_options["se_approx_weight_norm"] is True
    np.testing.assert_allclose(model.se_[0] ** 2,
                               _cluster_variance(model.psi_, model.cluster_codes_))


def test_refit_without_clusters_clears_cluster_state_and_restores_iid_inference():
    model = _model(_data()).fit()
    model.estimate()
    iid = _data(clustered=False)
    model.fit(iid)
    result = model.estimate()
    assert model.cluster_codes_ is None
    assert model.n_clusters_ is None
    assert model.n_clusters_fit_ is None
    assert model._fit_cluster_name_ is None
    expected = np.std(model.psi_, ddof=1) / np.sqrt(len(model.psi_))
    np.testing.assert_allclose(model.se_[0], expected)
    np.testing.assert_allclose(result.ci_lower_absolute, result.value - norm.ppf(0.975) * expected)
    assert result.diagnostic_data.cluster_codes is None


def test_singleton_clusters_reproduce_iid_variance_for_identical_folds():
    data = _data()
    data.df["cluster"] = np.arange(len(data.df))
    clustered = _model(data).fit()
    iid_data = _contract(data.df, clustered=False)
    iid = _model(iid_data)
    iid._fixed_fold_assignments_ = clustered._full_sample_folds_.copy()
    iid.fit()
    clustered_result = clustered.estimate()
    iid_result = iid.estimate()
    np.testing.assert_allclose(clustered_result.value, iid_result.value)
    np.testing.assert_allclose(clustered.se_, iid.se_, rtol=1e-12)
    assert clustered_result.ci_upper_absolute > iid_result.ci_upper_absolute


@pytest.mark.parametrize("failure", ["support", "training", "overlap"])
def test_failed_refit_invalidates_old_predictions_and_inference(failure, monkeypatch):
    model = _model(_data()).fit()
    model.estimate()
    replacement = _data()
    if failure == "support":
        replacement.df["cluster"] %= 2
        expected_error = ValueError
    elif failure == "training":
        def fail_training(**kwargs):
            raise RuntimeError("deliberate learner failure")
        monkeypatch.setattr(model, "_fit_nuisances_for_fold", fail_training)
        expected_error = RuntimeError
    else:
        replacement.df["p"] = np.where(replacement.cluster == 0, 0.5, 0.01)
        model.ml_m = _FeaturePropensity()
        model.overlap_policy = "drop"
        model.overlap_threshold = 0.1
        expected_error = ValueError
    assert len(replacement.df) == len(model.data.df)
    with pytest.raises(expected_error):
        model.fit(replacement)
    with pytest.raises(NotFittedError):
        model.estimate()
    with pytest.raises(NotFittedError):
        model.confint()


@pytest.mark.parametrize("destination", ["iid", "clustered"])
def test_successful_refit_requires_fresh_estimate_before_confint(destination):
    model = _model(_data()).fit()
    model.estimate()
    replacement = _data(clustered=destination == "clustered")
    if destination == "clustered":
        replacement.df["cluster"] %= 6
    model.fit(replacement)
    assert model.n_clusters_ == (6 if destination == "clustered" else None)
    with pytest.raises(NotFittedError):
        model.confint()
    result = model.estimate()
    np.testing.assert_allclose(model.confint().iloc[0],
                               [result.ci_lower_absolute, result.ci_upper_absolute])
