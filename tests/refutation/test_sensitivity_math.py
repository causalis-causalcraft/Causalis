"""Independent numerical checks for binary IRM sensitivity inference."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import (
    sensitivity as s,
    _sensitivity_benchmark as benchmark,
)


def _model(theta=0.7):
    x = np.linspace(-1.0, 1.0, 32)
    psi = 0.2 * x
    psi_nu2 = np.cos(np.arange(x.size)) - np.cos(np.arange(x.size)).mean()
    elements = dict(sigma2=1.3, nu2=3.0, psi=psi, psi_sigma2=0.4 * x, psi_nu2=psi_nu2)
    return (
        SimpleNamespace(
            coef_=np.array([theta]),
            se_=np.array([np.sqrt(np.var(psi, ddof=1) / x.size)]),
            _sensitivity_element_est=lambda: elements,
        ),
        elements,
    )


@pytest.mark.parametrize("score", ["ATE", "ATTE", "weighted"])
def test_irm_moments_use_the_linear_functional_weights(score):
    d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    m = np.repeat([0.25, 0.75], 4)
    w = np.ones(8) if score == "ATE" else d / d.mean()
    wb = w if score == "ATE" else m / d.mean()
    if score == "weighted":
        w, wb = np.linspace(1.0, 2.0, 8), np.linspace(1.2, 1.8, 8)
    y = np.arange(8) / 8
    result = s.compute_irm_sensitivity_elements(
        model=None,
        y=y,
        d=d,
        g0=np.zeros(8),
        g1=np.ones(8),
        m_hat=m,
        w=w,
        w_bar=wb,
        score="ATTE" if score == "ATTE" else "ATE",
    )
    ma = w * wb * (1 / m + 1 / (1 - m))
    rr = wb * (d / m - (1 - d) / (1 - m))
    score_nu2 = 2 * ma - rr**2
    np.testing.assert_allclose(result["m_alpha"], ma)
    np.testing.assert_allclose(result["riesz_rep"], rr)
    assert result["nu2"] == pytest.approx(score_nu2.mean())
    np.testing.assert_allclose(result["psi_nu2"], score_nu2 - score_nu2.mean())
    assert result["sigma2"] == pytest.approx(np.mean((y - d) ** 2))


def test_atte_nu2_is_orthogonal_to_propensity_perturbations():
    # Exact empirical conditional treatment frequencies equal the oracle m.
    d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    oracle = np.repeat([0.25, 0.75], 4)

    def nu2(shift):
        m = oracle + shift
        return s.compute_irm_sensitivity_elements(
            model=None,
            y=np.arange(8) / 8,
            d=d,
            g0=np.zeros(8),
            g1=np.ones(8),
            m_hat=m,
            w=d / d.mean(),
            w_bar=m / d.mean(),
            score="ATTE",
        )["nu2"]

    epsilon = 1e-6
    assert (nu2(epsilon) - nu2(-epsilon)) / (2 * epsilon) == pytest.approx(
        0.0, abs=1e-7
    )


@pytest.mark.parametrize("rho", [-1.0, -0.4, 0.0, 0.4, 1.0])
@pytest.mark.parametrize(
    "r2_y,r2_d",
    [(0.2, 0.3), (0.0, 0.3), (0.2, 0.0), (1.0 - 1e-12, 0.3)],
)
def test_scalar_covariance_ci_matches_direct_influence_arrays(rho, r2_y, r2_d):
    model, elements = _model()
    out = s.compute_bias_aware_ci(model, r2_y=r2_y, r2_d=r2_d, rho=rho)
    base = np.sqrt(elements["sigma2"] * elements["nu2"])
    factor = np.sqrt(r2_y * r2_d / (1 - r2_d))
    bias_if = (
        elements["sigma2"] * elements["psi_nu2"]
        + elements["nu2"] * elements["psi_sigma2"]
    ) / (2 * base)
    width = abs(rho) * factor * base
    expected = []
    for side in [-1, 1]:
        scores = elements["psi"] + side * abs(rho) * factor * bias_if
        se = np.sqrt(np.var(scores, ddof=1) / scores.size)
        expected.append(model.coef_[0] + side * (width + norm.ppf(0.975) * se))
    assert out["bound_width"] == pytest.approx(width)
    np.testing.assert_allclose(out["bias_aware_ci"], expected, rtol=1e-12)


@pytest.mark.parametrize("theta", [0.7, -0.3])
@pytest.mark.parametrize("rho", [1.0, -0.5])
def test_robustness_values_reach_the_actual_bounds_and_ci(theta, rho):
    model, _ = _model(theta)
    h0 = 0.2
    out = s.compute_bias_aware_ci(model, r2_y=0.1, r2_d=0.2, rho=rho, H0=h0)
    side = 0 if theta > h0 else 1
    for key, bounds_key in [
        ("rv", "theta_bounds_cofounding"),
        ("rva", "bias_aware_ci"),
    ]:
        r = out[key]
        at_root = s.compute_bias_aware_ci(model, r2_y=r, r2_d=r, rho=rho, H0=h0)
        assert at_root[bounds_key][side] == pytest.approx(h0, abs=1e-10)
        below = s.compute_bias_aware_ci(
            model, r2_y=r - 1e-5, r2_d=r - 1e-5, rho=rho, H0=h0
        )
        assert (below[bounds_key][side] - h0) * (theta - h0) > 0


def test_zero_adversity_and_initially_non_significant_robustness():
    model, _ = _model()
    out = s.compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.2, rho=0.0)
    assert np.isnan(out["rv"]) and np.isnan(out["rva"])
    np.testing.assert_allclose(out["bias_aware_ci"], out["sampling_ci"])
    out = s.compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.2, H0=model.coef_[0])
    assert out["rv"] == out["rva"] == 0.0


def test_zero_residual_variance_requires_a_zero_residual_influence_function():
    model, elements = _model()
    elements["sigma2"] = 0.0
    with pytest.raises(ValueError, match="zero psi_sigma2"):
        s.compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.2)
    elements["psi_sigma2"] = np.zeros(32)
    out = s.compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.2)
    assert out["bound_width"] == 0.0
    np.testing.assert_allclose(out["bias_aware_ci"], out["sampling_ci"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r2_y": np.nan},
        {"r2_y": -0.1},
        {"r2_y": 1.0},
        {"r2_y": 1.01},
        {"r2_d": np.inf},
        {"r2_d": 1.0},
        {"r2_d": -0.1},
        {"rho": 2.0},
        {"rho": np.nan},
        {"H0": np.inf},
        {"alpha": np.nan},
        {"use_signed_rr": True},
    ],
)
def test_invalid_sensitivity_parameters_are_rejected(kwargs):
    model, _ = _model()
    parameters = dict(r2_y=0.2, r2_d=0.2)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        s.compute_bias_aware_ci(model, **parameters)


@pytest.mark.parametrize("strength_name", ["cf_y", "cf_d"])
def test_unit_strength_boundary_is_rejected_with_canonical_names(strength_name):
    model, _ = _model()
    parameters = dict(cf_y=0.2, cf_d=0.2)
    parameters[strength_name] = 1.0
    with pytest.raises(ValueError, match=rf"{strength_name} must be finite and in \[0, 1\)"):
        s.compute_bias_aware_ci(model, **parameters)


@pytest.mark.parametrize(
    "name,value",
    [
        ("sigma2", -1.0),
        ("sigma2", np.nan),
        ("nu2", -1.0),
        ("nu2", 0.0),
        ("nu2", np.inf),
        ("psi", None),
        ("psi_sigma2", None),
        ("psi_nu2", np.zeros((32, 1))),
        ("psi", np.zeros(31)),
        ("psi_nu2", np.full(32, np.nan)),
    ],
)
def test_invalid_elements_are_rejected(name, value):
    model, elements = _model()
    elements[name] = value
    with pytest.raises(ValueError):
        s.compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.2)


@pytest.mark.parametrize("se", [np.nan, np.inf, -0.1])
def test_invalid_standard_error_is_rejected(se):
    model, _ = _model()
    model.se_[0] = se
    with pytest.raises(ValueError, match="standard error"):
        s.compute_bias_aware_ci(model, r2_y=0.2, r2_d=0.2)


@pytest.mark.parametrize("stored_se", [True, False])
def test_requested_alpha_rebuilds_sampling_ci(stored_se):
    model, elements = _model()
    theta, se = model.coef_[0], model.se_[0]
    estimate = SimpleNamespace(
        value=theta,
        alpha=0.05,
        model_options={"std_error": se} if stored_se else {},
        ci_lower_absolute=theta - norm.ppf(0.975) * se,
        ci_upper_absolute=theta + norm.ppf(0.975) * se,
        diagnostic_data=SimpleNamespace(**elements),
    )
    out = s.compute_bias_aware_ci(estimate, r2_y=0.0, r2_d=0.0, alpha=0.1)
    np.testing.assert_allclose(
        out["sampling_ci"], [theta - norm.ppf(0.95) * se, theta + norm.ppf(0.95) * se]
    )
    np.testing.assert_allclose(out["sampling_ci"], out["bias_aware_ci"])


@pytest.fixture
def fitted():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(500, 3))
    d = rng.binomial(1, 1 / (1 + np.exp(-x[:, 0])))
    y = 5 + 2 * d + x[:, 0] + 0.3 * x[:, 1] + rng.normal(size=500)
    data = CausalData(
        df=pd.DataFrame(x, columns=["x0", "x1", "x2"]).assign(y=y, d=d),
        treatment="d",
        outcome="y",
        confounders=["x0", "x1", "x2"],
    )
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=2,
    ).fit()
    return data, model, model.estimate()


def test_protocol_reuses_long_moments_and_unique_refits(fitted, monkeypatch):
    data, model, _ = fitted
    counts = dict(long=0, short=0, fit=0, preparation=0)
    original_elements, original_fit = IRM._sensitivity_element_est, IRM.fit
    original_prepare = benchmark._prepare_benchmark_context

    def elements(self, *args, **kwargs):
        counts["long" if self is model else "short"] += 1
        return original_elements(self, *args, **kwargs)

    def fit(self, *args, **kwargs):
        counts["fit"] += 1
        return original_fit(self, *args, **kwargs)

    def prepare(*args, **kwargs):
        counts["preparation"] += 1
        return original_prepare(*args, **kwargs)

    def unused_interpretation(*args, **kwargs):
        pytest.fail("Protocol scenarios must not build text interpretations.")

    effect = {"model": model}
    with monkeypatch.context() as patch:
        patch.setattr(IRM, "_sensitivity_element_est", elements)
        patch.setattr(IRM, "fit", fit)
        patch.setattr(benchmark, "_prepare_benchmark_context", prepare)
        patch.setattr(s, "interpret_sensitivity_analysis", unused_interpretation)
        report = s.run_sensitivity_protocol(
            effect,
            data,
            decision_threshold=0.2,
            benchmark_groups={
                "a": ["x0", "x1"],
                "b": ["x0"],
                "again": ["x1", "x0", "x1"],
            },
        )
    assert counts == dict(long=1, short=2, fit=2, preparation=1)
    assert report["benchmarks"]["benchmark_group"].tolist() == [
        ("x0", "x1"),
        ("x0",),
        ("x1", "x0"),
    ]
    for scenarios in report["details"].values():
        for raw in scenarios.values():
            assert "params" in raw
            expected = s.compute_bias_aware_ci(model, H0=0.2, **raw["params"])
            np.testing.assert_allclose(raw["bias_aware_ci"], expected["bias_aware_ci"])
    assert (
        effect["bias_aware"]["params"]
        == report["details"]["again"]["primary"]["params"]
    )


@pytest.mark.parametrize("se", [np.nan, -0.1])
def test_real_protocol_fails_closed_for_invalid_inference(fitted, se):
    data, _, estimate = fitted
    estimate.model_options["std_error"] = se
    report = s.run_sensitivity_protocol(
        estimate, data, decision_threshold=0.0, benchmark_groups={"g": ["x1"]}
    )
    assert report["status"] == "FAIL"
    assert not report["scenarios"]["scenario_valid"].any()
    assert not report["scenarios"]["passed"].any()
    assert report["scenarios"]["scenario_issue"].str.contains("standard error").all()


def test_benchmark_rejects_changed_estimand_and_incompatible_override(fitted):
    data, model, estimate = fitted
    with pytest.raises(ValueError, match="estimand/score"):
        s.sensitivity_benchmark_group(model, data, ["x1"], fit_args={"score": "ATTE"})
    model.estimate(score="ATTE")
    with pytest.raises(ValueError, match="estimand/score"):
        s.sensitivity_benchmark_group(estimate, data, ["x1"])


def test_benchmark_rejects_changed_model_estimate(fitted):
    data, model, estimate = fitted
    model.coef_[0] += 0.1
    with pytest.raises(ValueError, match="long-model state"):
        s.sensitivity_benchmark_group(estimate, data, ["x1"])


def test_benchmark_rejects_mutated_fit_sample(fitted):
    data, model, _ = fitted
    data.df.loc[0, "y"] += 1.0
    with pytest.raises(ValueError, match="changed after fit"):
        s.sensitivity_benchmark_group(model, data, ["x1"])


def test_protocol_reports_refit_failure_and_keeps_other_benchmarks(fitted, monkeypatch):
    data, model, _ = fitted
    original = benchmark._benchmark_group_refit

    def refit(context, group):
        if group == ("x0",):
            raise ValueError("learner could not fit")
        return original(context, group)

    monkeypatch.setattr(benchmark, "_benchmark_group_refit", refit)
    report = s.run_sensitivity_protocol(
        model,
        data,
        decision_threshold=0.0,
        benchmark_groups={"bad": ["x0"], "good": ["x1"]},
    )
    assert report["status"] == "FAIL"
    assert report["primary"]["scenario_valid"].tolist() == [False, True]
    assert "learner could not fit" in report["primary"].loc[0, "scenario_issue"]


def test_protocol_rejects_ambiguous_labels_before_refitting(fitted, monkeypatch):
    data, model, _ = fitted

    def refit(*args, **kwargs):
        pytest.fail("Invalid labels must be rejected before refitting.")

    monkeypatch.setattr(benchmark, "_benchmark_group_refit", refit)
    with pytest.raises(ValueError, match="unique"):
        s.run_sensitivity_protocol(
            model,
            data,
            decision_threshold=0.0,
            benchmark_groups={"g": ["x0"], " g ": ["x1"]},
        )


def test_drop_benchmark_fails_closed_for_different_populations():
    rng = np.random.default_rng(23)
    x1 = np.linspace(-4.0, 4.0, 600)
    x2 = rng.normal(size=600)
    d = rng.binomial(1, 1 / (1 + np.exp(-1.8 * x1)))
    y = 5 + d + x1 + rng.normal(size=600)
    data = CausalData(
        df=pd.DataFrame(dict(x1=x1, x2=x2, y=y, d=d)),
        treatment="d",
        outcome="y",
        confounders=["x1", "x2"],
    )
    model = IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=17,
        overlap_policy="drop",
        overlap_threshold=0.1,
    ).fit()
    model.estimate()
    report = s.run_sensitivity_protocol(
        model, data, decision_threshold=0.0, benchmark_groups={"g": ["x1"]}
    )
    assert report["status"] == "FAIL"
    assert not report["benchmarks"].loc[0, "calibration_valid"]
    assert "overlap masks differ" in report["benchmarks"].loc[0, "calibration_issue"]
    assert not report["scenarios"]["scenario_valid"].any()
