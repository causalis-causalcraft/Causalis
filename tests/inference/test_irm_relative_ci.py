import numpy as np
import pandas as pd
import pytest

from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.model import IRM


def _make_linear_cd(n=600, seed=0, baseline=1.5, tau=0.8, noise=0.5):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.4 * x1 - 0.2 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, p)
    y = baseline + tau * d + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=noise, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return df


def test_relative_ci_delta_method_matches_formula_with_weights():
    df = _make_linear_cd(n=700, seed=42, baseline=2.0, tau=0.6, noise=0.4)
    rng = np.random.default_rng(42)
    weights = 0.5 + rng.random(len(df))
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])

    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        weights=weights,
        random_state=7,
    )
    res = irm.fit().estimate(score="ATE", alpha=0.1)

    n = len(df)
    w, w_bar = irm._get_weights(n=n, m_hat_adj=irm.m_hat_, d=irm._d, score="ATE")
    y = irm._y
    d = irm._d
    u0 = y - irm.g0_hat_
    _, h0 = irm._normalize_ipw_terms(d, irm.m_hat_, score="ATE", warn=False)
    psi_mu_c = w * irm.g0_hat_ + w_bar * (u0 * h0)
    mu_c = float(np.mean(psi_mu_c))
    tau_rel_expected = 100.0 * irm.coef_[0] / mu_c

    assert np.isclose(res.value_relative, tau_rel_expected, rtol=1e-10, atol=1e-12)

    IF_mu = psi_mu_c - mu_c
    IF_rel = 100.0 * (irm.psi_ / mu_c - (irm.coef_[0] * IF_mu) / (mu_c ** 2))
    var_rel = float(np.var(IF_rel, ddof=1)) / n
    se_rel = float(np.sqrt(max(var_rel, 0.0)))
    z = norm.ppf(1 - 0.1 / 2.0)
    ci_low_expected = tau_rel_expected - z * se_rel
    ci_high_expected = tau_rel_expected + z * se_rel
    if ci_low_expected > ci_high_expected:
        ci_low_expected, ci_high_expected = ci_high_expected, ci_low_expected

    assert np.isclose(res.ci_lower_relative, ci_low_expected, rtol=1e-10, atol=1e-10)
    assert np.isclose(res.ci_upper_relative, ci_high_expected, rtol=1e-10, atol=1e-10)


def test_relative_ci_guardrail_warns_and_sets_nan():
    df = _make_linear_cd(n=400, seed=11, baseline=1e-6, tau=0.2, noise=1e-6)
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])

    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        relative_baseline_min=1e-2,
        random_state=11,
    )

    with pytest.warns(RuntimeWarning, match="Relative effect baseline"):
        res = irm.fit().estimate(score="ATE")

    assert np.isnan(res.value_relative)
    assert np.isnan(res.ci_lower_relative)
    assert np.isnan(res.ci_upper_relative)


def test_relative_ci_ordered_when_baseline_negative():
    df = _make_linear_cd(n=500, seed=21, baseline=-2.5, tau=0.4, noise=0.5)
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])

    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        random_state=21,
    )
    res = irm.fit().estimate(score="ATE", alpha=0.05)

    w, w_bar = irm._get_weights(n=len(df), m_hat_adj=irm.m_hat_, d=irm._d, score="ATE")
    y = irm._y
    d = irm._d
    u0 = y - irm.g0_hat_
    _, h0 = irm._normalize_ipw_terms(d, irm.m_hat_, score="ATE", warn=False)
    psi_mu_c = w * irm.g0_hat_ + w_bar * (u0 * h0)
    mu_c = float(np.mean(psi_mu_c))

    assert mu_c < 0.0
    assert res.ci_lower_relative <= res.ci_upper_relative


def test_ate_hajek_and_weight_norm_mark_approximate_inference():
    df = _make_linear_cd(n=650, seed=123, baseline=2.0, tau=0.5, noise=0.4)
    rng = np.random.default_rng(123)
    weights = 0.5 + rng.random(len(df))
    cd = CausalData(df=df, treatment="d", outcome="y", confounders=["x1", "x2"])

    irm = IRM(
        data=cd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=2000),
        n_folds=3,
        normalize_ipw=True,
        weights=weights,
        random_state=123,
    )

    with pytest.warns(RuntimeWarning) as rec:
        res = irm.fit().estimate(score="ATE")

    messages = [str(w.message) for w in rec]
    assert any("Hájek" in msg for msg in messages)
    assert any("normalized by sample mean" in msg for msg in messages)
    assert res.model_options["se_approx_hajek"] is True
    assert res.model_options["se_approx_weight_norm"] is True
