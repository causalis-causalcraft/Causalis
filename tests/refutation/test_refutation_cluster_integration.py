"""Exercise the public binary refutation API on genuinely clustered DGP fits."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from causalis.dgp.causaldata.clustered import generate_obs_clustered
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness import refutation as ref
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness.sensitivity import compute_bias_aware_ci


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
@pytest.mark.parametrize("policy", ["clip", "drop"])
@pytest.mark.parametrize("treatment_correlation", [0.0, 1.0])
def test_all_refutation_entrypoints_on_clustered_generator(monkeypatch, score, policy, treatment_correlation):
    data = generate_obs_clustered(n=600, n_clusters=30, seed=19,
                                  treatment_correlation=treatment_correlation)
    if policy == "drop":
        # Fit real forests and retain their feature importance. Force the same
        # known row removals in long and short refits to exercise alignment,
        # including complete removal of a cluster, without changing estimands
        # between sensitivity benchmarks.
        original_cross_fit = IRM._cross_fit_nuisances

        def cross_fit(self, X, y, d, y_is_binary):
            g0, g1, p, folds, importance = original_cross_fit(self, X, y, d, y_is_binary)
            positions = np.arange(len(d))
            remove = (positions < 20) | (positions % 17 == 0)
            p = np.where(remove, .01, np.clip(p, .2, .8))
            return g0, g1, p, folds, importance

        monkeypatch.setattr(IRM, "_cross_fit_nuisances", cross_fit)
    model = IRM(
        data, n_folds=3, random_state=19, overlap_policy=policy, overlap_threshold=.1,
        ml_g=RandomForestRegressor(n_estimators=5, max_depth=3, min_samples_leaf=5, random_state=19),
        ml_m=RandomForestClassifier(n_estimators=5, max_depth=3, min_samples_leaf=5, random_state=19),
    ).fit()
    estimate = model.estimate(score=score, alpha=.1)
    expected_n = int(np.sum(model.overlap_mask_))
    expected_clusters = 29 if policy == "drop" else 30
    if policy == "drop":
        assert expected_n < 600
        assert expected_n == len(estimate.diagnostic_data.cluster_codes)
    overlap = ref.run_overlap_diagnostics(data, estimate)
    balance = ref.run_unconfoundedness_diagnostics(data, estimate)
    scores = ref.run_score_diagnostics(data, estimate)
    for report in (overlap, balance, scores):
        assert report.get("n", report["meta"].get("n")) == expected_n
        assert report["meta"]["n_clusters"] == expected_clusters
        assert report["meta"]["cov_type"] == "cluster"
    assert scores["influence_diagnostics"]["se_plugin"] == pytest.approx(model.se_[0])
    assert scores["oos_moment_test"]["available"] is False
    assert scores["oos_moment_test"]["descriptive_available"] is True
    for code in np.unique(model.cluster_codes_):
        assert np.unique(model.folds_[model.cluster_codes_ == code]).size == 1

    figures = [
        ref.plot_m_overlap(estimate, dpi=60),
        ref.plot_propensity_reliability(estimate, data=data, dpi=60),
        ref.plot_feature_importance(estimate, top_k=3, dpi=60),
        ref.love_plot(data, estimate, dpi=60),
        ref.plot_influence_instability(estimate, data=data, top_k=4, dpi=60),
        ref.plot_influence_instability(estimate, influence_level="row", top_k=4, dpi=60),
        ref.plot_residual_diagnostics(estimate, data=data, max_scatter_points=50, random_state=2, dpi=60),
    ]
    assert all(isinstance(figure, plt.Figure) and figure.axes for figure in figures)
    assert "clusters" in figures[4].axes[0].get_ylabel()
    assert all("cluster=" in label.get_text() for label in figures[4].axes[0].get_yticklabels())

    detached_diagnostic = estimate.diagnostic_data.model_copy()
    detached_diagnostic._model = None
    detached = estimate.model_copy(update={"diagnostic_data": detached_diagnostic})
    detached_scores = ref.run_score_diagnostics(None, detached)
    pd.testing.assert_frame_equal(scores["orthogonality_derivatives"], detached_scores["orthogonality_derivatives"])
    for rho in (0., 1.):
        sensitivity = ref.sensitivity_analysis(detached, r2_y=.02, r2_d=.02, rho=rho, alpha=.1)
        computed = compute_bias_aware_ci(detached, r2_y=.02, r2_d=.02, rho=rho, alpha=.1)
        np.testing.assert_allclose(sensitivity["bias_aware_ci"], computed["bias_aware_ci"])
        assert computed["inference_df"] == expected_clusters-1
        assert isinstance(ref.get_sensitivity_summary(detached), str)
        interpreted = ref.interpret_sensitivity_analysis(detached, r2_y=.02, r2_d=.02, rho=rho, alpha=.1)
        assert "summary" in interpreted and "interpretation" in interpreted
        if rho == 0:
            np.testing.assert_allclose(computed["sampling_ci"], [estimate.ci_lower_absolute, estimate.ci_upper_absolute])

    singleton_benchmark = ref.sensitivity_benchmark(estimate, data, ["age_years"])
    grouped_benchmark = ref.sensitivity_benchmark_group(estimate, data, ["age_years", "tenure_months"])
    assert len(singleton_benchmark) == len(grouped_benchmark) == 1
    assert np.isfinite(singleton_benchmark.loc[0, "theta_short"])
    assert np.isfinite(grouped_benchmark.loc[0, "theta_short"])
    protocol = ref.run_sensitivity_protocol(
        estimate, data, benchmark_groups={"age": ["age_years"]},
        decision_threshold=0., direction="positive", alpha=.1,
    )
    assert protocol["status"] in {"PASS", "FAIL"}
    assert len(protocol["benchmarks"]) == 1
    for figure in figures:
        plt.close(figure)
