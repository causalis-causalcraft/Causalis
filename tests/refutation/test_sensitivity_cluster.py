"""Independent cluster-moment checks and clustered DGP sensitivity routes."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t
from sklearn.dummy import DummyClassifier, DummyRegressor

from causalis.data_contracts import CausalData
from causalis.dgp.causaldata.clustered import generate_obs_clustered
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import sensitivity as s
from causalis.scenarios.unconfoundedness.refutation._shared import _resolve_cluster_context


def _cluster_variance(values, codes):
    centered = values - np.mean(values)
    sums = np.array([centered[codes == code].sum() for code in np.unique(codes)])
    groups = sums.size
    return groups / (groups - 1) * np.dot(sums, sums) / values.size**2


@pytest.mark.parametrize("rho", [-1.0, -0.4, 0.0, 0.4])
def test_cluster_sensitivity_matches_direct_bound_influence_variance(rho):
    codes = np.repeat([0, 1, 2, 3], [2, 5, 3, 7])
    x = np.linspace(-1, 1, codes.size)
    elements = dict(
        sigma2=1.3, nu2=3.0, psi=0.2 * x + 0.1,
        psi_sigma2=0.4 * x, psi_nu2=np.cos(np.arange(x.size)),
        cluster_codes=codes,
    )
    model = SimpleNamespace(
        coef_=np.array([0.7]),
        se_=np.array([np.sqrt(_cluster_variance(elements["psi"], codes))]),
        cluster_codes_=codes,
        _sensitivity_element_est=lambda: elements,
    )
    out = s.compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.3, rho=rho, alpha=0.1)
    base = np.sqrt(elements["sigma2"] * elements["nu2"])
    bias_if = (
        elements["nu2"] * elements["psi_sigma2"]
        + elements["sigma2"] * elements["psi_nu2"]
    ) / (2 * base)
    strength = abs(rho) * np.sqrt(0.2 * 0.3 / 0.7)
    expected = []
    for side in [-1, 1]:
        se = np.sqrt(_cluster_variance(elements["psi"] + side * strength * bias_if, codes))
        expected.append(0.7 + side * (strength * base + t.ppf(0.95, 3) * se))
    np.testing.assert_allclose(out["bias_aware_ci"], expected, rtol=1e-12)
    assert out["n_clusters"] == 4
    assert out["inference_distribution"] == "t"
    assert out["inference_df"] == 3
    if rho:
        root = s.compute_bias_aware_ci(model, r2_y=out["rva"], r2_d=out["rva"], rho=rho, alpha=0.1)
        assert root["bias_aware_ci"][0] == pytest.approx(0.0, abs=1e-10)


@pytest.fixture
def clustered():
    data = generate_obs_clustered(n=600, n_clusters=30, seed=19)
    model = IRM(
        data=data, ml_g=DummyRegressor(), ml_m=DummyClassifier(strategy="prior"),
        n_folds=3, random_state=19,
    ).fit()
    return data, model


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
@pytest.mark.parametrize("route", ["estimate", "model", "dict", "detached"])
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_zero_confounding_preserves_clustered_ci(clustered, score, route, alpha):
    _, model = clustered
    estimate = model.estimate(score=score, alpha=alpha)
    effect = estimate
    if route == "model":
        effect = model
    elif route == "dict":
        effect = {"model": model}
    elif route == "detached":
        effect = SimpleNamespace(
            value=estimate.value, alpha=estimate.alpha,
            ci_lower_absolute=estimate.ci_lower_absolute,
            ci_upper_absolute=estimate.ci_upper_absolute,
            model_options=dict(estimate.model_options),
            diagnostic_data=estimate.diagnostic_data,
        )
    out = s.compute_bias_aware_ci(effect, r2_y=0.0, r2_d=0.2, alpha=alpha)
    expected = (estimate.ci_lower_absolute, estimate.ci_upper_absolute)
    np.testing.assert_allclose(out["sampling_ci"], expected, rtol=1e-12)
    np.testing.assert_allclose(out["bias_aware_ci"], expected, rtol=1e-12)
    assert out["cov_type"] == "cluster"
    assert out["inference_df"] == 29


def test_cluster_ci_se_reconstruction_uses_original_t_quantile(clustered):
    _, model = clustered
    estimate = model.estimate(alpha=0.1)
    detached = SimpleNamespace(
        value=estimate.value, alpha=estimate.alpha,
        ci_lower_absolute=estimate.ci_lower_absolute,
        ci_upper_absolute=estimate.ci_upper_absolute,
        model_options={key: value for key, value in estimate.model_options.items() if key != "std_error"},
        diagnostic_data=estimate.diagnostic_data,
    )
    out = s.compute_bias_aware_ci(detached, r2_y=0, r2_d=0, alpha=0.05)
    assert out["se"] == pytest.approx(model.se_[0])
    np.testing.assert_allclose(out["sampling_ci"], model.confint(alpha=0.05).iloc[0])


def test_cluster_benchmark_preserves_membership_and_full_folds(clustered, monkeypatch):
    data, model = clustered
    estimate = model.estimate()
    expected_codes = model.cluster_codes_.copy()
    expected_folds = model._full_sample_folds_.copy()
    original_fit = IRM.fit
    observed = []

    def fit(short, *args, **kwargs):
        np.testing.assert_array_equal(short._fixed_fold_assignments_, expected_folds)
        assert short.data.cluster_name == data.cluster_name
        result = original_fit(short, *args, **kwargs)
        np.testing.assert_array_equal(short.cluster_codes_, expected_codes)
        for code in np.unique(expected_codes):
            assert np.unique(short._full_sample_folds_[expected_codes == code]).size == 1
        observed.append(short)
        return result

    monkeypatch.setattr(IRM, "fit", fit)
    out = s.sensitivity_benchmark_group(estimate, data, ["age_years"])
    assert len(observed) == 1
    assert observed[0]._inference_metadata()["cov_type"] == "cluster"
    assert np.isfinite(out.loc[0, "theta_short"])


@pytest.mark.parametrize("change", ["membership", "declaration", "mutate_original"])
def test_cluster_benchmark_rejects_changed_partition(clustered, change):
    data, model = clustered
    estimate = model.estimate()
    frame = data.df.copy()
    if change == "membership":
        frame.loc[0, "cluster"] = frame.loc[30, "cluster"]
    other = CausalData(
        df=frame, treatment=data.treatment_name, outcome=data.outcome_name,
        confounders=data.confounders, cluster=None if change == "declaration" else "cluster",
    )
    if change == "mutate_original":
        data.df.loc[0, "cluster"] = data.df.loc[30, "cluster"]
        other = data
    with pytest.raises(ValueError, match="cluster|changed after fit"):
        s.sensitivity_benchmark_group(estimate, other, ["age_years"])


def test_cluster_protocol_agrees_with_standalone_analysis(clustered):
    data, model = clustered
    estimate = model.estimate()
    report = s.run_sensitivity_protocol(
        estimate, data, decision_threshold=0.0, alpha=0.1,
        benchmark_groups={"age": ["age_years"]},
    )
    for raw in report["details"]["age"].values():
        expected = s.compute_bias_aware_ci(estimate, alpha=0.1, H0=0.0, **raw["params"])
        np.testing.assert_allclose(raw["bias_aware_ci"], expected["bias_aware_ci"])
        assert raw["inference_df"] == 29


def test_cluster_drop_benchmark_reuses_full_folds_and_retained_groups(monkeypatch):
    data = generate_obs_clustered(n=600, n_clusters=30, seed=19)
    original = IRM._cross_fit_nuisances
    fitted = []

    def cross_fit(self, X, y, d, y_is_binary):
        g0, g1, _, folds, importance = original(self, X, y, d, y_is_binary)
        positions = np.arange(len(d))
        propensity = np.where((positions < 40) | (positions % 7 == 0), 0.01, 0.5)
        fitted.append(self)
        return g0, g1, propensity, folds, importance

    monkeypatch.setattr(IRM, "_cross_fit_nuisances", cross_fit)
    model = IRM(
        data, ml_g=DummyRegressor(), ml_m=DummyClassifier(strategy="prior"),
        n_folds=3, random_state=19, overlap_policy="drop", overlap_threshold=0.1,
    ).fit()
    estimate = model.estimate()
    out = s.compute_bias_aware_ci(estimate, r2_y=0, r2_d=0)
    assert out["n_clusters"] == 28
    np.testing.assert_allclose(out["bias_aware_ci"], [estimate.ci_lower_absolute, estimate.ci_upper_absolute])
    s.sensitivity_benchmark_group(estimate, data, ["income_monthly"])
    assert len(fitted) == 2
    short = fitted[1]
    np.testing.assert_array_equal(short._full_sample_folds_, model._full_sample_folds_)
    np.testing.assert_array_equal(short.cluster_codes_, model.cluster_codes_)
    assert short.n_clusters_ == 28


def test_single_retained_cluster_is_rejected_for_sensitivity():
    codes = np.zeros(8, dtype=int)
    elements = dict(sigma2=1.0, nu2=2.0, psi=np.arange(8.),
                    psi_sigma2=np.arange(8.), psi_nu2=np.arange(8.), cluster_codes=codes)
    model = SimpleNamespace(coef_=np.array([1.]), se_=np.array([0.1]),
                            _sensitivity_element_est=lambda: elements)
    with pytest.raises(ValueError, match="at least two"):
        s.compute_bias_aware_ci(model, r2_y=0.1, r2_d=0.1)


def test_legacy_sensitivity_recovery_rejects_a_refitted_model(clustered):
    _, model = clustered
    estimate = model.estimate()
    estimate.diagnostic_data.sigma2 = None
    expected = s.compute_bias_aware_ci(model, r2_y=0.1, r2_d=0.1)
    actual = s.compute_bias_aware_ci(estimate, r2_y=0.1, r2_d=0.1)
    np.testing.assert_allclose(actual["bias_aware_ci"], expected["bias_aware_ci"])
    model.coef_[0] += 0.5
    with pytest.raises(ValueError, match="model state"):
        s.compute_bias_aware_ci(estimate, r2_y=0.1, r2_d=0.1)


def test_legacy_shared_alignment_preserves_finite_value_checks(clustered):
    data, model = clustered
    estimate = model.estimate()
    old = estimate.diagnostic_data
    y, x = old.y.copy(), old.x.copy()
    y[0] = np.nan
    x[1, 0] = np.nan
    diagnostic = old.model_copy(update={"sample_fingerprint": None, "y": y, "x": x})
    legacy = estimate.model_copy(update={"diagnostic_data": diagnostic})
    context = _resolve_cluster_context(data, legacy, len(data.df))
    assert context.metadata["n_clusters"] == 30
    frame = data.df.copy()
    frame.loc[2, data.confounders[0]] += 1
    changed = CausalData(
        df=frame, treatment=data.treatment_name, outcome=data.outcome_name,
        confounders=data.confounders, cluster="cluster",
    )
    with pytest.raises(ValueError, match="saved estimation rows"):
        _resolve_cluster_context(changed, legacy, len(frame))


def test_frozen_membership_is_authority_when_covariance_metadata_is_absent(clustered):
    data, model = clustered
    estimate = model.estimate()
    incomplete = estimate.model_copy(update={"model_options": {}})
    context = _resolve_cluster_context(data, incomplete, len(data.df))
    assert context.metadata["cov_type"] == "cluster"
    assert context.metadata["inference_df"] == 29
    conflicting = estimate.model_copy(update={"model_options": {"cov_type": "iid"}})
    with pytest.raises(ValueError, match="iid"):
        s.compute_bias_aware_ci(conflicting, r2_y=0.1, r2_d=0.1)
