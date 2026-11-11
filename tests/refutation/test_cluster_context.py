"""Independent covariance and saved-sample identity checks."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.dummy import DummyClassifier, DummyRegressor
from statsmodels.stats.sandwich_covariance import cov_cluster

from causalis.data_contracts import CausalData
from causalis.dgp import generate_obs_clustered
from causalis.scenarios.unconfoundedness import IRM
from causalis.scenarios.unconfoundedness._cluster_utils import compact_cluster_codes, mean_covariance
from causalis.scenarios.unconfoundedness.refutation._shared import _aligned_data_array, _resolve_cluster_context


@pytest.mark.parametrize("sizes", [(1,) * 30, (4, 13, 2, 7, 19, 5)])
def test_covariance_matches_independent_group_sums_and_statsmodels(sizes):
    rng = np.random.default_rng(216)
    codes = np.repeat(np.arange(len(sizes)), sizes)
    values = rng.normal(size=(len(codes), 3)) + [1., -2., 3.]
    centered = values - values.mean(axis=0)
    sums = np.array([centered[codes == g].sum(axis=0) for g in range(len(sizes))])
    expected = len(sizes) / (len(sizes) - 1) * sums.T @ sums / len(codes)**2
    actual = mean_covariance(*values.T, cluster_codes=codes)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    oracle = sm.OLS(values[:, 0], np.ones((len(codes), 1))).fit()
    np.testing.assert_allclose(actual[0, 0], cov_cluster(oracle, codes)[0, 0])
    if len(sizes) == len(codes):
        np.testing.assert_allclose(actual, mean_covariance(*values.T), atol=1e-14)
    order = rng.permutation(len(codes))
    np.testing.assert_allclose(actual, mean_covariance(*values[order].T, cluster_codes=codes[order]), atol=1e-14)


@pytest.mark.parametrize("codes", [[-1, 0], [0., 1.], [[0, 1]], [True, False], [0]])
def test_invalid_codes_fail_before_reduction(codes):
    with pytest.raises(ValueError, match="cluster_codes"):
        compact_cluster_codes(codes, 2)


def test_sparse_codes_and_unavailable_single_cluster():
    codes = compact_cluster_codes(np.array([10**15, 1, 10**15]), 3)
    np.testing.assert_array_equal(codes, [0, 1, 0])
    assert not codes.flags.writeable
    assert np.isnan(mean_covariance(np.arange(3.), cluster_codes=np.zeros(3, dtype=int))).all()


@pytest.fixture
def fitted_cluster():
    data = generate_obs_clustered(n=600, n_clusters=30, seed=19)
    model = IRM(data, ml_g=DummyRegressor(), ml_m=DummyClassifier(strategy="prior"),
                n_folds=3, random_state=19).fit()
    return data, model, model.estimate()


def _rebuild(data, frame):
    return CausalData.from_df(frame, data.treatment_name, data.outcome_name,
                              list(data.confounders), cluster=data.cluster_name)


def test_snapshot_accepts_relabeling_and_rejects_changed_partition(fitted_cluster):
    data, _, estimate = fitted_cluster
    frame = data.df.copy()
    frame["cluster"] = pd.Categorical([f"market-{x}" for x in frame.cluster])
    actual = _resolve_cluster_context(_rebuild(data, frame), estimate, len(data.df))
    np.testing.assert_array_equal(actual.codes, estimate.diagnostic_data.cluster_codes)
    assert actual.labels[0] == "market-0"
    frame["cluster"] = frame.cluster.astype(object)
    frame.iloc[0, frame.columns.get_loc("cluster")] = "market-1"
    with pytest.raises(ValueError, match="partition"):
        _resolve_cluster_context(_rebuild(data, frame), estimate, len(data.df))


def test_snapshot_rejects_reordering_and_changed_outcomes(fitted_cluster):
    data, _, estimate = fitted_cluster
    for frame in (data.df.iloc[::-1].copy(), data.df.assign(y=data.df.y + 1)):
        with pytest.raises(ValueError, match="sample"):
            _resolve_cluster_context(_rebuild(data, frame), estimate, len(data.df))


def test_grouping_does_not_follow_a_refitted_or_missing_model(fitted_cluster):
    data, model, estimate = fitted_cluster
    expected = estimate.diagnostic_data.cluster_codes.copy()
    model.cluster_codes_ = np.zeros(len(data.df), dtype=int)
    np.testing.assert_array_equal(_resolve_cluster_context(None, estimate, len(data.df)).codes, expected)
    estimate.diagnostic_data._model = None
    np.testing.assert_array_equal(_resolve_cluster_context(None, estimate, len(data.df)).codes, expected)
    assert len(estimate.diagnostic_data.sample_fingerprint) < 15


def test_drop_mapping_and_finite_mask_recount(fitted_cluster):
    data, _, estimate = fitted_cluster
    keep = np.asarray(data.cluster) >= 2
    old = estimate.diagnostic_data
    fields = {key: getattr(old, key)[keep] for key in
              ("d", "y", "x", "m_hat", "g0_hat", "g1_hat", "cluster_codes")}
    fields["overlap_mask"] = keep
    diagnostic = old.model_copy(update=fields)
    result = estimate.model_copy(update={"diagnostic_data": diagnostic})
    context = _resolve_cluster_context(data, result, int(keep.sum()))
    np.testing.assert_array_equal(context.row_positions, np.flatnonzero(keep))
    np.testing.assert_array_equal(_aligned_data_array(data, diagnostic, "y"), data.df.y.to_numpy()[keep])
    assert context.metadata["n_clusters"] == 28
    assert context.subset(context.codes == 0).metadata["inferential_available"] is False


def test_missing_grouping_never_silently_uses_iid(fitted_cluster):
    data, _, estimate = fitted_cluster
    diagnostic = estimate.diagnostic_data.model_copy(update={"cluster_codes": None})
    result = estimate.model_copy(update={"diagnostic_data": diagnostic})
    with pytest.raises(ValueError, match="membership is missing"):
        _resolve_cluster_context(None, result, len(data.df))
    np.testing.assert_array_equal(_resolve_cluster_context(data, result, len(data.df)).codes,
                                  estimate.diagnostic_data.cluster_codes)
    inconsistent = estimate.model_copy(update={"model_options": {"cov_type": "iid"}})
    with pytest.raises(ValueError, match="iid"):
        _resolve_cluster_context(None, inconsistent, len(data.df))


def test_frozen_user_ids_are_checked_without_changing_model_fingerprints(fitted_cluster):
    data, _, _ = fitted_cluster
    frame = data.df.assign(uid=[f"user-{i}" for i in range(len(data.df))])
    identified = CausalData.from_df(frame, "d", "y", list(data.confounders),
                                   user_id="uid", cluster="cluster")
    model = IRM(identified, ml_g=DummyRegressor(), ml_m=DummyClassifier(strategy="prior"),
                n_folds=3, random_state=19).fit()
    result = model.estimate()
    identified.df.loc[identified.df.index[0], "uid"] = "different"
    with pytest.raises(ValueError, match="sample contents"):
        _resolve_cluster_context(identified, result, len(identified.df))
    # The snapshot remains sufficient without external mutable labels.
    assert _resolve_cluster_context(None, result, len(identified.df)).metadata["n_clusters"] == 30
