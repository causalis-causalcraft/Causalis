"""Independent covariance, numerical derivative and clustered score regressions."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causaldata import CausalData
from causalis.dgp.causaldata.clustered import generate_obs_clustered
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.score.influence_plot import plot_influence_instability
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import (
    _orthogonality_derivatives_ate,
    _orthogonality_derivatives_atte,
    run_score_diagnostics,
)


def _variance(values, codes=None):
    centered = values - np.mean(values)
    if codes is None:
        return np.dot(centered, centered) / (len(values) * (len(values) - 1))
    sums = np.array([centered[codes == code].sum() for code in pd.unique(codes)])
    return len(sums) / (len(sums) - 1) * np.dot(sums, sums) / len(values)**2


def _oracle_estimate():
    frame = generate_obs_clustered(n=360, n_clusters=18, seed=37, return_causal_data=False)
    features = [name for name in frame if name not in {"y", "d", "cluster", "m", "m_obs", "g0", "g1", "cate", "tau_link"}]
    data = CausalData.from_df(frame, treatment="d", outcome="y", confounders=features, cluster="cluster")
    y, d, p = frame.y.to_numpy(), frame.d.to_numpy(), frame.m.to_numpy()
    signal = frame.cate.to_numpy() + d * (y-frame.g1.to_numpy()) / p - (1-d)*(y-frame.g0.to_numpy())/(1-p)
    theta = float(np.mean(signal))
    diagnostic = UnconfoundednessDiagnosticData(
        y=y, d=d, x=data.X.to_numpy(dtype=float), m_hat=p,
        g0_hat=frame.g0.to_numpy(), g1_hat=frame.g1.to_numpy(), psi=signal-theta,
        psi_b=signal, folds=frame.cluster.to_numpy() % 3,
        cluster_codes=frame.cluster.to_numpy(), cluster_name="cluster", score="ATE",
    )
    estimate = CausalEstimate(
        estimand="ATE", model="IRM", model_options={"cov_type": "cluster"},
        value=theta, ci_lower_absolute=theta-1, ci_upper_absolute=theta+1,
        alpha=.05, p_value=.5, is_significant=False, n_treated=int(d.sum()),
        n_control=int((1-d).sum()), treatment_mean=float(y[d == 1].mean()),
        control_mean=float(y[d == 0].mean()), outcome="y", treatment="d",
        confounders=features, diagnostic_data=diagnostic,
    )
    return data, estimate


def test_cluster_score_covariance_and_default_influence_units():
    data, estimate = _oracle_estimate()
    report = run_score_diagnostics(data, estimate)
    psi, codes = estimate.diagnostic_data.psi, estimate.diagnostic_data.cluster_codes
    assert report["influence_diagnostics"]["se_plugin"]**2 == pytest.approx(_variance(psi, codes))
    assert report["meta"]["n_clusters"] == 18
    assert report["meta"]["inference_df"] == 17
    assert report["influence_diagnostics"]["influence_level"] == "cluster"
    top = report["influence_diagnostics"]["top_influential"]
    sums = np.array([psi[codes == code].sum() for code in range(18)])
    for row in top.itertuples():
        assert row.psi == pytest.approx(sums[row.cluster_code])
        assert row.contribution == pytest.approx(row.psi / len(psi))
        assert row.variance_share == pytest.approx(row.psi**2 / np.dot(sums, sums))
    row_report = run_score_diagnostics(data, estimate, influence_level="row")
    assert row_report["influence_diagnostics"]["n_units"] == len(psi)
    assert row_report["influence_diagnostics"]["se_plugin"] == report["influence_diagnostics"]["se_plugin"]
    pd.testing.assert_frame_equal(report["orthogonality_derivatives"], row_report["orthogonality_derivatives"])


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_clustered_dgp_score_plugin_matches_estimator(score):
    data = generate_obs_clustered(n=600, n_clusters=30, seed=19, sigma_cluster=2)
    model = IRM(data, ml_g=LinearRegression(), ml_m=DummyClassifier(strategy="prior"), n_folds=3, random_state=4).fit()
    estimate = model.estimate(score=score)
    report = run_score_diagnostics(data, estimate)
    assert report["influence_diagnostics"]["se_plugin"] == pytest.approx(model.se_[0], rel=1e-12)
    saved = run_score_diagnostics(None, estimate)
    pd.testing.assert_frame_equal(saved["orthogonality_derivatives"], report["orthogonality_derivatives"])
    assert saved["oos_moment_test"]["available"] is False
    assert saved["oos_moment_test"]["descriptive_available"] is True


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_ratio_derivatives_and_standard_errors_match_numerical_perturbations(clustered, score):
    rng = np.random.default_rng(28)
    n = 24
    d = np.tile([0., 1., 1.], n // 3)
    p, y, g0, g1 = rng.uniform(.2, .8, n), rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    w, wbar = rng.uniform(.5, 1.5, n), rng.uniform(.5, 1.5, n)
    basis = np.column_stack((np.ones(n), rng.normal(size=n)))
    codes = np.repeat(np.arange(8), 3) if clustered else None
    probabilities = np.full(n, 1/n)

    def score_mean(prob, g0_value, g1_value, propensity):
        if score == "ATTE":
            numerator = d*(y-g0_value) - (1-d)*propensity/(1-propensity)*(y-g0_value)
            return np.dot(prob, numerator) / np.dot(prob, d)
        treated, control = d/propensity, (1-d)/(1-propensity)
        return (np.dot(prob, w*(g1_value-g0_value))
                + np.dot(prob, wbar*(y-g1_value)*treated)/np.dot(prob, treated)
                - np.dot(prob, wbar*(y-g0_value)*control)/np.dot(prob, control))

    if score == "ATE":
        table = _orthogonality_derivatives_ate(basis, y, d, g0, g1, p, .01,
                                             normalize_ipw=True, w=w, w_bar=wbar, cluster_codes=codes)
        nuisances = ["g0", "g1", "m"]
    else:
        table = _orthogonality_derivatives_atte(basis, y, d, g0, p, .01, cluster_codes=codes)
        nuisances = ["g0", "m"]
    for column in range(basis.shape[1]):
        for nuisance in nuisances:
            def derivative(prob):
                args_plus, args_minus = [g0, g1, p], [g0, g1, p]
                index = {"g0": 0, "g1": 1, "m": 2}[nuisance]
                step = 1e-4
                args_plus[index] = args_plus[index] + step*basis[:, column]
                args_minus[index] = args_minus[index] - step*basis[:, column]
                return (score_mean(prob, *args_plus)-score_mean(prob, *args_minus))/(2*step)
            expected = derivative(probabilities)
            numerical_if = np.empty(n)
            for i in range(n):
                direction = -probabilities.copy()
                direction[i] += 1
                step = 1e-4
                numerical_if[i] = (derivative(probabilities+step*direction)-derivative(probabilities-step*direction))/(2*step)
            assert table.loc[column, "d_"+nuisance] == pytest.approx(expected, rel=2e-6, abs=2e-6)
            assert table.loc[column, "se_"+nuisance]**2 == pytest.approx(_variance(numerical_if, codes), rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_single_retained_cluster_has_descriptive_influence_and_no_inference(score):
    data, estimate = _oracle_estimate()
    psi = estimate.diagnostic_data.psi.copy()
    psi[estimate.diagnostic_data.cluster_codes != 0] = np.nan
    diagnostic = estimate.diagnostic_data.model_copy(update={"psi": psi, "score": score})
    report = run_score_diagnostics(None, estimate.model_copy(update={"diagnostic_data": diagnostic}))
    assert report["meta"]["n_clusters"] == 1
    assert report["meta"]["inferential_available"] is False
    assert np.isnan(report["influence_diagnostics"]["se_plugin"])
    assert report["influence_diagnostics"]["top_influential"].shape[0] == 1
    assert report["flags"]["ortho_max_|t|_m"] == "NA"
    for nuisance in ("g1", "g0", "m"):
        assert report["orthogonality_derivatives"][[f"se_{nuisance}", f"t_{nuisance}"]].isna().all().all()
        assert report["flags"][f"ortho_max_|t|_{nuisance}"] == "NA"


def test_cluster_influence_plot_cache_and_explicit_row_level():
    data, estimate = _oracle_estimate()
    cluster_figure = plot_influence_instability(estimate, data=data, top_k=5)
    cached_figure = plot_influence_instability(estimate, top_k=5)
    assert len(cluster_figure.axes) == 2
    assert "Cluster" in cluster_figure.axes[0].get_title()
    assert all("cluster=" in label.get_text() for label in cluster_figure.axes[0].get_yticklabels())
    np.testing.assert_allclose([bar.get_width() for bar in cluster_figure.axes[0].patches],
                               [bar.get_width() for bar in cached_figure.axes[0].patches])
    scatter = cluster_figure.axes[1].collections[0].get_offsets()
    psi, codes = estimate.diagnostic_data.psi, estimate.diagnostic_data.cluster_codes
    sums = np.array([psi[codes == group].sum() for group in np.unique(codes)])
    ranked = np.argsort(-np.abs(sums))[:5]
    np.testing.assert_allclose(scatter[:, 0], np.bincount(codes)[ranked])
    np.testing.assert_allclose(scatter[:, 1], np.abs(sums[ranked]))
    assert cluster_figure.axes[1].get_xlabel() == "Cluster size (observations)"
    assert not cluster_figure.axes[1].lines
    row_figure = plot_influence_instability(estimate, top_k=5, influence_level="row")
    assert row_figure.axes[0].get_ylabel() == "Ranked observations"


def test_cluster_influence_requires_membership():
    _, estimate = _oracle_estimate()
    diagnostic = estimate.diagnostic_data.model_copy(update={"cluster_codes": None, "cluster_name": None})
    estimate = estimate.model_copy(update={"diagnostic_data": diagnostic, "model_options": {}})
    with pytest.raises(ValueError, match="Cluster influence"):
        run_score_diagnostics(None, estimate, influence_level="cluster")


@pytest.mark.parametrize("normalize", [False, True])
def test_propensity_derivative_respects_clipping(normalize):
    from causalis.scenarios.unconfoundedness.refutation.score.score_validation import _aipw_score_ate
    rng = np.random.default_rng(9)
    n = 20
    d = np.tile([0., 1.], n//2)
    p = np.linspace(.01, .99, n)
    y, g0, g1 = (rng.normal(size=n) for _ in range(3))
    basis = rng.normal(size=(n, 1))
    ones = np.ones(n)
    table = _orthogonality_derivatives_ate(basis, y, d, g0, g1, p, .1,
                                         normalize_ipw=normalize, w=ones, w_bar=ones)
    step = 1e-6
    kwargs = dict(y=y, d=d, g0=g0, g1=g1, theta=0., overlap_threshold=.1,
                  normalize_ipw=normalize, w=ones, w_bar=ones)
    finite = (_aipw_score_ate(m=p+step*basis[:, 0], **kwargs).mean()
              - _aipw_score_ate(m=p-step*basis[:, 0], **kwargs).mean())/(2*step)
    assert table.loc[0, "d_m"] == pytest.approx(finite, rel=1e-7)


def test_missing_custom_weights_never_reload_a_live_models_weights():
    from types import SimpleNamespace
    _, estimate = _oracle_estimate()
    diagnostic = estimate.diagnostic_data.model_copy(update={"w": None, "w_bar": None, "score_plot_cache": None})
    def forbidden(**kwargs):
        pytest.fail("A saved estimate must not adopt weights from a refitted model")
    diagnostic._model = SimpleNamespace(_fit_weights_used_=True, weights=np.ones(len(diagnostic.d)), _get_weights=forbidden)
    estimate = estimate.model_copy(update={"diagnostic_data": diagnostic})
    with pytest.raises(ValueError, match="saved w and w_bar"):
        run_score_diagnostics(None, estimate)
    with pytest.raises(ValueError, match="saved w and w_bar"):
        plot_influence_instability(estimate, use_estimator_psi=False)
    # Stored estimator influence is sufficient for plotting without reweighting.
    figure = plot_influence_instability(estimate)
    assert len(figure.axes[0].patches) == 18


def test_cluster_variance_shares_use_centered_influence_after_filtering():
    from causalis.scenarios.unconfoundedness.refutation.score.score_validation import _aggregate_cluster_influence
    psi = np.array([1., 2., 4., 8., 16.])
    codes = np.array([0, 0, 1, 1, 1])
    units = _aggregate_cluster_influence(psi, np.ones(5), np.full(5, .5), codes)
    centered_sums = np.array([(psi[codes == group]-psi.mean()).sum() for group in (0, 1)])
    np.testing.assert_allclose(units["variance_share"], centered_sums**2/np.dot(centered_sums, centered_sums))
    np.testing.assert_allclose(units["psi"], [3., 28.])


@pytest.mark.parametrize("field", ["psi", "m_hat"])
def test_all_nonfinite_saved_score_rows_raise_actionable_error(field):
    _, estimate = _oracle_estimate()
    diagnostic = estimate.diagnostic_data.model_copy(update={field: np.full(360, np.nan)})
    detached = estimate.model_copy(update={"diagnostic_data": diagnostic})
    with pytest.raises(ValueError, match="No finite observations available for score diagnostics"):
        run_score_diagnostics(None, detached)
