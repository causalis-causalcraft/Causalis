"""Compare one-way IRM inference with the pinned DoubleML 0.11.4 formulas.

These tests isolate inference from learner/split randomness. DoubleML is not a
runtime dependency: the small reference below follows the published source.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from causalis.data_contracts.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM


def _doubleml_one_way_reference(psi_a, psi_b, clusters, folds):
    """Reproduce DoubleML's cluster coefficient and asymptotic variance.

    Pinned implementations:
    https://github.com/DoubleML/doubleml-for-py/blob/0.11.4/doubleml/double_ml_score_mixins.py
    https://github.com/DoubleML/doubleml-for-py/blob/0.11.4/doubleml/utils/_estimation.py

    DoubleML weights fold totals by inverse held-out cluster count and uses no
    finite-cluster correction. Squaring a cluster score sum is algebraically
    equivalent to its sum of the within-cluster outer-product matrix.
    """
    fold_ids = np.unique(folds)
    cluster_ids = np.unique(clusters)
    a_total = 0.0
    b_total = 0.0
    for fold in fold_ids:
        rows = folds == fold
        fold_cluster_count = np.unique(clusters[rows]).size
        a_total += np.sum(psi_a[rows]) / fold_cluster_count
        b_total += np.sum(psi_b[rows]) / fold_cluster_count
    theta = -b_total / a_total
    score = psi_a * theta + psi_b
    meat = 0.0
    for fold in fold_ids:
        held_out_clusters = np.unique(clusters[folds == fold])
        for cluster in held_out_clusters:
            cluster_score = np.sum(score[clusters == cluster])
            meat += cluster_score**2 / held_out_clusters.size
    jacobian = a_total / fold_ids.size
    gamma = meat / fold_ids.size
    variance = gamma / (cluster_ids.size * jacobian**2)
    return theta, np.sqrt(variance)


def _fixed_prediction_model(monkeypatch, cluster_fold_ids):
    """Use unequal cluster row counts and fixed out-of-fold predictions."""
    sizes = np.array([4, 7, 5, 8, 6, 9])[: len(cluster_fold_ids)]
    clusters = np.repeat(np.arange(len(sizes)), sizes)
    n = clusters.size
    x = np.linspace(-1.5, 2.0, n)
    d = np.arange(n) % 2
    g0 = 9.6 + 0.3 * x
    g1 = g0 + 0.8 + 0.02 * x
    m = 0.25 + 0.5 * (np.sin(x) + 1.0) / 2.0
    y = (
        10.0 + 0.4 * x + d * (0.5 + 0.07 * clusters)
        + 0.2 * clusters + 0.3 * np.cos(np.arange(n))
    )
    data = CausalData(
        df=pd.DataFrame({"y": y, "d": d, "x": x, "cluster": clusters}),
        outcome="y",
        treatment="d",
        confounders=["x"],
        cluster="cluster",
    )
    folds = np.asarray(cluster_fold_ids)[clusters]

    def fixed_predictions(self, X, y, d, y_is_binary):
        return g0.copy(), g1.copy(), m.copy(), folds.copy(), None

    monkeypatch.setattr(IRM, "_cross_fit_nuisances", fixed_predictions)
    model = IRM(
        data,
        ml_g=DummyRegressor(),
        ml_m=DummyClassifier(),
        n_folds=np.unique(folds).size,
        store_diagnostics=False,
        random_state=19,
    ).fit()
    return model, y, d, g0, g1, m, clusters, folds


def _canonical_score(score, y, d, g0, g1, m):
    # Deliberately independent of IRM's score construction and cached psi arrays.
    if score == "ATE":
        psi_a = -np.ones(y.size)
        psi_b = g1 - g0 + d * (y - g1) / m - (1 - d) * (y - g0) / (1 - m)
    else:
        treated_share = np.mean(d)
        psi_a = -d / treated_share
        psi_b = (d * (y - g0) - (1 - d) * m * (y - g0) / (1 - m)) / treated_share
    return psi_a, psi_b


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_balanced_cluster_folds_match_doubleml_before_finite_correction(monkeypatch, score):
    model, y, d, g0, g1, m, clusters, folds = _fixed_prediction_model(
        monkeypatch, [0, 0, 1, 1, 2, 2]
    )
    psi_a, psi_b = _canonical_score(score, y, d, g0, g1, m)
    reference_theta, reference_se = _doubleml_one_way_reference(psi_a, psi_b, clusters, folds)

    result = model.estimate(score=score)
    correction = len(np.unique(clusters)) / (len(np.unique(clusters)) - 1)

    assert result.value == pytest.approx(reference_theta, rel=1e-13, abs=1e-13)
    assert model.se_[0] / np.sqrt(correction) == pytest.approx(reference_se, rel=1e-13)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_unequal_cluster_fold_counts_keep_pooled_observation_moment(monkeypatch, score):
    model, y, d, g0, g1, m, clusters, folds = _fixed_prediction_model(
        monkeypatch, [0, 0, 1, 1, 2]
    )
    psi_a, psi_b = _canonical_score(score, y, d, g0, g1, m)
    reference_theta, _ = _doubleml_one_way_reference(psi_a, psi_b, clusters, folds)
    pooled_theta = -np.sum(psi_b) / np.sum(psi_a)
    residual_score = psi_a * pooled_theta + psi_b
    cluster_sums = np.array([
        np.sum(residual_score[clusters == cluster]) for cluster in np.unique(clusters)
    ])
    G = cluster_sums.size
    pooled_variance = (G / (G - 1)) * np.sum(cluster_sums**2) / np.sum(psi_a)**2

    result = model.estimate(score=score)

    assert result.value == pytest.approx(pooled_theta, rel=1e-13, abs=1e-13)
    assert model.se_[0] ** 2 == pytest.approx(pooled_variance, rel=1e-13)
    assert not np.isclose(result.value, reference_theta, rtol=1e-5, atol=1e-5)
