"""Observation-weighted descriptive diagnostics with frozen cluster membership."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from causalis.data_contracts import CausalData
from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata.clustered import generate_obs_clustered
from causalis.scenarios.unconfoundedness.refutation import (
    love_plot,
    plot_m_overlap,
    plot_propensity_reliability,
    plot_residual_diagnostics,
    run_overlap_diagnostics,
    run_unconfoundedness_diagnostics,
)
from causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation import _auc_mann_whitney
from causalis.scenarios.unconfoundedness.refutation.overlap.reliability_plot import _resolve_calibration_payload
from causalis.scenarios.unconfoundedness.refutation.score.residual_plots import (
    _binned_summary,
    _resolve_residual_inputs,
)


def _estimate(data, diagnostic):
    d = np.asarray(diagnostic.d)
    return CausalEstimate(
        estimand=diagnostic.score, model="IRM", value=1.0,
        ci_upper_absolute=1.2, ci_lower_absolute=0.8, alpha=0.05,
        is_significant=True, n_treated=int(d.sum()), n_control=int((1 - d).sum()),
        treatment_mean=1.0, control_mean=0.0, outcome="y", treatment="d",
        confounders=list(data.confounders), diagnostic_data=diagnostic,
        model_options={"cov_type": "cluster", "cluster_name": "cluster"},
    )


def _oracle(*, score="ATE", n=690, groups=30, rho=0.5, sizes=None):
    frame = generate_obs_clustered(
        n=n, n_clusters=groups, cluster_sizes=sizes, seed=19,
        treatment_correlation=rho, return_causal_data=False,
    )
    confounders = [name for name in frame if name not in {
        "y", "d", "cluster", "m", "m_obs", "g0", "g1", "cate", "tau_link",
    }]
    data = CausalData(df=frame, treatment="d", outcome="y", confounders=confounders, cluster="cluster")
    diagnostic = UnconfoundednessDiagnosticData(
        m_hat=frame.m.to_numpy(), d=frame.d.to_numpy(), y=frame.y.to_numpy(),
        x=frame[confounders].to_numpy(), g0_hat=frame.g0.to_numpy(),
        g1_hat=frame.g1.to_numpy(), cluster_codes=frame.cluster.to_numpy(),
        cluster_name="cluster", score=score,
    )
    return data, _estimate(data, diagnostic)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_cluster_descriptive_values_preserve_observation_weighting(score):
    data, estimate = _oracle(score=score, n=735, sizes=list(range(10, 40)))
    plain = CausalData(df=data.df, treatment="d", outcome="y", confounders=data.confounders)
    iid_diag = estimate.diagnostic_data.model_copy(update={"cluster_codes": None, "cluster_name": None})
    iid = estimate.model_copy(update={"diagnostic_data": iid_diag, "model_options": {"cov_type": "iid"}})
    clustered_overlap = run_overlap_diagnostics(data, estimate)
    plain_overlap = run_overlap_diagnostics(plain, iid)
    for key in ["ks", "auc", "support_rate", "p1"]:
        assert clustered_overlap[key] == pytest.approx(plain_overlap[key])
    for key in ["ece", "brier"]:
        assert clustered_overlap["calibration"][key] == pytest.approx(plain_overlap["calibration"][key])
    pd.testing.assert_series_equal(
        run_unconfoundedness_diagnostics(data, estimate)["balance"]["smd"],
        run_unconfoundedness_diagnostics(plain, iid)["balance"]["smd"],
    )
    assert clustered_overlap["meta"]["n_clusters"] == 30
    assert clustered_overlap["meta"]["weighting"] == "observation"


def test_cluster_weight_concentration_matches_direct_sums_and_pure_arm_support():
    data, estimate = _oracle(rho=1.0)
    result = run_overlap_diagnostics(data, estimate)
    diag = estimate.diagnostic_data
    codes, d, m = diag.cluster_codes, diag.d, diag.m_hat
    expected_weights = {
        "w1": d / m, "w0": (1 - d) / (1 - m),
        "att_w1": d, "att_w0": (1 - d) * m / (1 - m),
    }
    for name, weights in expected_weights.items():
        sums = np.array([weights[codes == group].sum() for group in np.unique(codes)])
        expected = sums.sum() ** 2 / np.dot(sums, sums)
        assert result["cluster_weight_concentration"][name] == pytest.approx(expected)
    treated_counts = np.array([d[codes == group].sum() for group in np.unique(codes)])
    sizes = np.bincount(codes)
    support = result["cluster_support"]
    assert support["n_treated_only_clusters"] == np.count_nonzero(treated_counts == sizes)
    assert support["n_control_only_clusters"] == np.count_nonzero(treated_counts == 0)
    assert support["n_treated_only_clusters"] + support["n_control_only_clusters"] > 0


@pytest.mark.parametrize("kind", ["string", "categorical"])
def test_cluster_relabeling_preserves_descriptive_reports(kind):
    data, estimate = _oracle()
    frame = data.df.copy()
    labels = frame.cluster.map(lambda value: f"market-{value}")
    frame["cluster"] = pd.Categorical(labels) if kind == "categorical" else labels
    relabeled = CausalData(df=frame, treatment="d", outcome="y", confounders=data.confounders, cluster="cluster")
    assert run_overlap_diagnostics(relabeled, estimate)["cluster_support"]["n_clusters"] == 30


def test_drop_fallback_arrays_select_saved_estimation_rows():
    data, estimate = _oracle()
    original = estimate.diagnostic_data
    mask = np.arange(original.d.size) % 3 != 0
    updates = {name: getattr(original, name)[mask] for name in ["m_hat", "d", "y", "x", "g0_hat", "g1_hat", "cluster_codes"]}
    retained = original.model_copy(update={**updates, "overlap_mask": None})
    reduced_data = CausalData(df=data.df.iloc[mask], treatment="d", outcome="y", confounders=data.confounders, cluster="cluster")
    expected_estimate = estimate.model_copy(update={"diagnostic_data": retained})
    fallback = retained.model_copy(update={"overlap_mask": mask, "d": None, "x": None})
    fallback_estimate = estimate.model_copy(update={"diagnostic_data": fallback})
    expected = run_unconfoundedness_diagnostics(reduced_data, expected_estimate)
    actual = run_unconfoundedness_diagnostics(data, fallback_estimate)
    pd.testing.assert_series_equal(actual["balance"]["smd"], expected["balance"]["smd"])
    assert run_overlap_diagnostics(data, fallback_estimate)["n"] == int(mask.sum())
    assert _resolve_calibration_payload(fallback_estimate, data, n_bins=10)["metadata"]["n_clusters"] == 30


def test_nonfinite_treatment_is_dropped_before_binary_conversion_and_cache_reuse():
    data, estimate = _oracle()
    d = estimate.diagnostic_data.d.astype(float).copy()
    d[0] = np.nan
    diagnostic = estimate.diagnostic_data.model_copy(update={"d": d})
    estimate = estimate.model_copy(update={"diagnostic_data": diagnostic})
    assert run_overlap_diagnostics(data, estimate)["n"] == d.size - 1
    assert _resolve_calibration_payload(estimate, None, n_bins=10)["table"]["count"].sum() == d.size - 1
    first = _resolve_residual_inputs(estimate, None, clip_propensity=1e-6)
    second = _resolve_residual_inputs(estimate, None, clip_propensity=1e-6)
    assert first["d"].size == d.size - 1
    np.testing.assert_array_equal(first["d"], second["d"])
    assert first["metadata"] == second["metadata"]
    cache_only = diagnostic.model_copy(update={
        "m_hat": None, "g0_hat": None, "g1_hat": None, "y": None, "d": None,
    })
    cached_estimate = estimate.model_copy(update={"diagnostic_data": cache_only})
    cached = _resolve_residual_inputs(cached_estimate, None, clip_propensity=1e-6)
    np.testing.assert_array_equal(first["d"], cached["d"])


def test_infinite_balance_smd_is_a_failure():
    frame = pd.DataFrame({"d": [1, 1, 1, 1, 0, 0, 0, 0], "x": [2, 2, 2, 2, -1, -1, -1, -1], "y": np.arange(8.0), "cluster": np.repeat(np.arange(4), 2)})
    data = CausalData(df=frame, treatment="d", outcome="y", confounders=["x"], cluster="cluster")
    diag = UnconfoundednessDiagnosticData(m_hat=np.full(8, 0.5), d=frame.d.to_numpy(), y=frame.y.to_numpy(), x=frame[["x"]].to_numpy(), cluster_codes=frame.cluster.to_numpy(), cluster_name="cluster", score="ATE")
    result = run_unconfoundedness_diagnostics(data, _estimate(data, diag))
    assert np.isinf(result["balance"]["smd_max"])
    assert result["balance"]["frac_violations"] == 1.0
    assert result["balance"]["pass"] is False
    assert result["overall_flag"] == "RED"


@pytest.mark.parametrize("seed", [19, 20])
def test_vectorized_auc_matches_independent_tie_aware_reference(seed):
    rng = np.random.default_rng(seed)
    scores = np.round(rng.random(501), 1)
    labels = rng.integers(0, 2, scores.size)
    assert _auc_mann_whitney(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_vectorized_binned_summary_matches_interval_reference():
    x = np.repeat(np.linspace(-1, 1, 15), np.arange(1, 16))
    y = np.sin(np.arange(x.size))
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    expected = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= lo) & ((x < hi) if index < edges.size - 2 else (x <= hi))
        if mask.any():
            expected.append((x[mask].mean(), y[mask].mean(), mask.sum()))
    np.testing.assert_allclose(np.asarray(_binned_summary(x, y, n_bins=10)).T, expected)


def test_clustered_descriptive_plot_entrypoints():
    data, estimate = _oracle(rho=1.0)
    figures = [plot_m_overlap(estimate), plot_propensity_reliability(estimate, data), love_plot(data, estimate), plot_residual_diagnostics(estimate, data)]
    assert [len(figure.axes) for figure in figures] == [1, 1, 1, 3]
    for figure in figures:
        assert not plt.fignum_exists(figure.number)


def test_residual_default_scatter_is_bounded_and_reproducible():
    data, estimate = _oracle(n=60_000, groups=100)
    first = plot_residual_diagnostics(estimate, data)
    second = plot_residual_diagnostics(estimate, data)
    all_points = plot_residual_diagnostics(estimate, data, max_scatter_points=None)
    for index in [0, 1]:
        offsets = first.axes[index].collections[0].get_offsets()
        assert len(offsets) <= 20_000
        np.testing.assert_array_equal(offsets, second.axes[index].collections[0].get_offsets())
        assert len(all_points.axes[index].collections[0].get_offsets()) >= len(offsets)
        np.testing.assert_allclose(first.axes[index].lines[0].get_ydata(), all_points.axes[index].lines[0].get_ydata())
