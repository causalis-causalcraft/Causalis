import numpy as np
import pandas as pd
import pytest
import inspect
from types import SimpleNamespace

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.model import MultiTreatmentIRM
from causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.sensitivity import (
    sensitivity_analysis,
    sensitivity_benchmark,
    get_sensitivity_summary,
)


def _make_synthetic_multi(n: int = 450, seed: int = 42, user_id: str | None = None) -> MultiCausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    logits = np.column_stack(
        [
            np.zeros(n),
            1.1 * x1 + 0.2 * x2,
            -0.9 * x1 + 0.3 * x2,
        ]
    )
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    labels = np.array([rng.choice(3, p=probs[i]) for i in range(n)], dtype=int)
    d = np.eye(3, dtype=int)[labels]

    effects = np.array([0.0, 0.6, -0.4], dtype=float)
    y = 1.0 + effects[labels] + 0.9 * x1 + 0.4 * x2 + rng.normal(scale=0.6, size=n)

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "d0": d[:, 0],
            "d1": d[:, 1],
            "d2": d[:, 2],
        }
    )
    if user_id is not None:
        df[user_id] = [f"obs_{i}" for i in rng.permutation(n)]
        df.index = pd.Index(rng.permutation(n) + 1000, name="row")
    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1", "d2"],
        confounders=["x1", "x2"],
        control_treatment="d0",
        user_id=user_id,
    )


def _fit_multi_irm(data: MultiCausalData) -> MultiTreatmentIRM:
    lr_sig = inspect.signature(LogisticRegression)
    lr_kwargs = {"max_iter": 1500}
    if "multi_class" in lr_sig.parameters:
        lr_kwargs["multi_class"] = "multinomial"

    model = MultiTreatmentIRM(
        data=data,
        ml_g=RandomForestRegressor(n_estimators=80, random_state=7),
        ml_m=LogisticRegression(**lr_kwargs),
        n_folds=3,
        normalize_ipw=False,
        random_state=7,
    )
    model.fit().estimate(score="ATE", diagnostic_data=True)
    return model


def _supports_r2_y_alias() -> bool:
    fn_sig = inspect.signature(sensitivity_analysis)
    model_sig = inspect.signature(MultiTreatmentIRM.sensitivity_analysis)
    return ("r2_y" in fn_sig.parameters) and ("r2_y" in model_sig.parameters)


def _dummy_estimate_for_sensitivity(
    *,
    theta: np.ndarray,
    se: np.ndarray,
    sigma2: float,
    nu2: np.ndarray,
) -> SimpleNamespace:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    se = np.asarray(se, dtype=float).reshape(-1)
    z = 1.959963984540054
    ci_low = theta - z * se
    ci_high = theta + z * se

    diag = SimpleNamespace(
        sigma2=float(sigma2),
        nu2=np.asarray(nu2, dtype=float).reshape(-1),
        psi_sigma2=None,
        psi_nu2=None,
        riesz_rep=None,
        m_alpha=None,
        psi=None,
        sensitivity_analysis=None,
    )
    return SimpleNamespace(
        value=theta,
        ci_lower_absolute=ci_low,
        ci_upper_absolute=ci_high,
        model_options={"std_error": se},
        diagnostic_data=diag,
    )


@pytest.mark.parametrize("user_id", [None, "observation_id"])
def test_multi_sensitivity_benchmark_basic_with_estimate_input(user_id):
    data = _make_synthetic_multi(user_id=user_id)
    original_df = data.df.copy(deep=True)
    model = _fit_multi_irm(data)
    dml_result = model.estimate(score="ATE", diagnostic_data=True)

    res = sensitivity_benchmark(dml_result, benchmarking_set=["x1"])

    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == 2
    for col in [
        "cf_y", "cf_d", "rho", "theta_long", "theta_short", "delta",
        "r2_y", "r2_d", "gain_y", "gain_d", "cf_y_raw", "cf_d_raw",
        "sigma2_long", "sigma2_short", "nu2_long", "nu2_short",
    ]:
        assert col in res.columns
    assert list(res.columns[:10]) == [
        "cf_y", "cf_d", "rho", "theta_long", "theta_short", "delta",
        "r2_y", "r2_d", "gain_y", "gain_d",
    ]
    assert np.all(np.isfinite(res["theta_short"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(res["delta"].to_numpy(dtype=float)))
    assert np.any(np.abs(res["delta"].to_numpy(dtype=float)) > 0.0)
    pd.testing.assert_frame_equal(data.df, original_df)

    # Gain-statistics calibration: strengths in [0, 1], with valid rows below 1.
    cf_y = res["cf_y"].to_numpy(dtype=float)
    cf_d = res["cf_d"].to_numpy(dtype=float)
    rho = res["rho"].to_numpy(dtype=float)
    assert np.all((cf_y >= 0.0) & (cf_y <= 1.0))
    assert np.all((cf_d >= 0.0) & (cf_d <= 1.0))
    valid = res["calibration_valid"].to_numpy(dtype=bool)
    assert np.all(cf_y[valid] < 1.0)
    assert np.all(cf_d[valid] < 1.0)
    assert np.all(np.abs(rho) <= 1.0)
    np.testing.assert_allclose(res["r2_y"].to_numpy(dtype=float), cf_y)
    np.testing.assert_allclose(res["r2_d"].to_numpy(dtype=float), cf_d)

    # Strengths follow from the long/short sensitivity elements of each contrast.
    elems_long = model._sensitivity_element_est()
    sigma2_long = float(elems_long["sigma2"])
    nu2_long = np.asarray(elems_long["nu2"], dtype=float).reshape(-1)
    np.testing.assert_allclose(res["sigma2_long"].to_numpy(dtype=float), sigma2_long)
    np.testing.assert_allclose(res["nu2_long"].to_numpy(dtype=float), nu2_long)
    sigma2_short = res["sigma2_short"].to_numpy(dtype=float)
    nu2_short = res["nu2_short"].to_numpy(dtype=float)
    np.testing.assert_allclose(
        res["cf_y_raw"].to_numpy(dtype=float), (sigma2_short - sigma2_long) / sigma2_long
    )
    np.testing.assert_allclose(
        res["cf_d_raw"].to_numpy(dtype=float), (nu2_long - nu2_short) / nu2_short
    )
    np.testing.assert_allclose(cf_y, np.clip(res["cf_y_raw"].to_numpy(dtype=float), 0.0, 1.0))
    np.testing.assert_allclose(cf_d, np.clip(res["cf_d_raw"].to_numpy(dtype=float), 0.0, 1.0))


def test_multi_sensitivity_benchmark_input_validation():
    data = _make_synthetic_multi(seed=123)
    model = _fit_multi_irm(data)
    effect = {"model": model}

    with pytest.raises(TypeError):
        sensitivity_benchmark(effect, benchmarking_set="x1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sensitivity_benchmark(effect, benchmarking_set=[])
    with pytest.raises(ValueError):
        sensitivity_benchmark(effect, benchmarking_set=["not_a_feature"])


def test_multi_sensitivity_analysis_r2_aliases_map_one_to_one_with_warning():
    if not _supports_r2_y_alias():
        pytest.skip("r2_y alias is not available in this implementation.")

    data = _make_synthetic_multi(seed=777)
    model = _fit_multi_irm(data)
    dml_result = model.estimate(score="ATE", diagnostic_data=True)

    with pytest.warns(DeprecationWarning):
        out = sensitivity_analysis(dml_result, data, r2_y=0.2, r2_d=0.1, rho=0.7, alpha=0.05)
    # Aliases are used as-is (no odds conversion).
    assert np.isclose(float(out["params"]["cf_y"]), 0.2, atol=1e-12, rtol=0.0)
    assert np.allclose(np.asarray(out["params"]["cf_d"], dtype=float), 0.1, atol=1e-12, rtol=0.0)

    out_same = sensitivity_analysis(
        dml_result,
        data,
        cf_y=0.2,
        cf_d=0.1,
        rho=0.7,
        alpha=0.05,
    )
    assert np.allclose(
        np.asarray(out["theta_bounds_cofounding"], dtype=float),
        np.asarray(out_same["theta_bounds_cofounding"], dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    summary_df = out.summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert list(summary_df.columns) == ["statistics", "value"]
    assert summary_df["statistics"].tolist() == [
        "d1 vs d0 | bias_aware_ci",
        "d1 vs d0 | theta",
        "d1 vs d0 | sampling_ci",
        "d1 vs d0 | rv",
        "d1 vs d0 | rva",
        "d1 vs d0 | se",
        "d1 vs d0 | max_bias",
        "d1 vs d0 | max_bias_base",
        "d1 vs d0 | bound_width",
        "d1 vs d0 | sigma2",
        "d1 vs d0 | nu2",
        "d2 vs d0 | bias_aware_ci",
        "d2 vs d0 | theta",
        "d2 vs d0 | sampling_ci",
        "d2 vs d0 | rv",
        "d2 vs d0 | rva",
        "d2 vs d0 | se",
        "d2 vs d0 | max_bias",
        "d2 vs d0 | max_bias_base",
        "d2 vs d0 | bound_width",
        "d2 vs d0 | sigma2",
        "d2 vs d0 | nu2",
    ]
    assert summary_df.iloc[1]["value"] == [
        round(out["theta_bounds_cofounding"][0][0], 4),
        round(out["theta"][0], 4),
        round(out["theta_bounds_cofounding"][0][1], 4),
    ]
    assert summary_df.iloc[12]["value"] == [
        round(out["theta_bounds_cofounding"][1][0], 4),
        round(out["theta"][1], 4),
        round(out["theta_bounds_cofounding"][1][1], 4),
    ]
    assert "Bias-aware Interval" in str(out)
    assert "Bias-aware Interval" in repr(out)
    assert isinstance(get_sensitivity_summary(out), str)


def test_multi_sensitivity_analysis_rejects_name_and_alias_together():
    if not _supports_r2_y_alias():
        pytest.skip("r2_y alias is not available in this implementation.")

    data = _make_synthetic_multi(seed=888)
    model = _fit_multi_irm(data)
    dml_result = model.estimate(score="ATE", diagnostic_data=True)

    with pytest.raises(ValueError, match="not both"):
        sensitivity_analysis(dml_result, data, cf_y=0.01, r2_y=0.2, cf_d=0.1)
    with pytest.raises(ValueError, match="not both"):
        sensitivity_analysis(dml_result, data, cf_y=0.2, cf_d=0.1, r2_d=0.1)


def test_model_sensitivity_analysis_accepts_cf_and_deprecated_r2_aliases():
    if not _supports_r2_y_alias():
        pytest.skip("r2_y alias is not available in this implementation.")

    data = _make_synthetic_multi(seed=999)
    model = _fit_multi_irm(data)
    returned = model.sensitivity_analysis(cf_y=0.1, cf_d=0.1, rho=1.0, alpha=0.05)
    assert returned is model
    assert np.isclose(float(model.sensitivity_result["params"]["cf_y"]), 0.1)

    with pytest.warns(DeprecationWarning):
        returned = model.sensitivity_analysis(r2_y=0.1, r2_d=0.1, rho=1.0, alpha=0.05)
    assert returned is model
    assert isinstance(model.sensitivity_summary, str)
    assert isinstance(model.sensitivity_result, dict)
    assert isinstance(model.sensitivity_result.summary(), pd.DataFrame)


def test_multi_sensitivity_is_stored_on_diagnostic_data_only():
    data = _make_synthetic_multi(seed=2026)
    model = _fit_multi_irm(data)
    dml_result = model.estimate(score="ATE", diagnostic_data=True)

    _ = sensitivity_analysis(dml_result, data, cf_y=0.1, cf_d=0.1, rho=0.5, alpha=0.05)

    diag_payload = None
    if dml_result.diagnostic_data is not None:
        diag_payload = getattr(dml_result.diagnostic_data, "sensitivity_analysis", None)
    estimate_payload = getattr(dml_result, "sensitivity_analysis", None)
    assert isinstance(diag_payload, dict) or isinstance(estimate_payload, dict)


def test_multi_rv_formula_uses_common_strength_solution():
    est = _dummy_estimate_for_sensitivity(
        theta=np.array([0.2]),
        se=np.array([0.01]),
        sigma2=1.0,
        nu2=np.array([4.0]),
    )
    out = sensitivity_analysis(est, cf_y=0.25, cf_d=0.2, rho=1.0, H0=0.0, alpha=0.05)

    max_bias_base = np.sqrt(1.0 * 4.0)  # 2.0
    den = 1.0 * max_bias_base  # 2.0 (RV does not depend on the supplied cf_y/cf_d)
    D2 = (0.2 / den) ** 2  # 0.01
    expected_rv = (np.sqrt(D2 * D2 + 4.0 * D2) - D2) / 2.0
    assert np.isclose(float(out["rv"]), expected_rv, atol=1e-12, rtol=0.0)
    # scale * v / sqrt(1 - v) == delta
    v = float(out["rv"])
    assert np.isclose(den * v / np.sqrt(1.0 - v), 0.2, atol=1e-12, rtol=0.0)

    delta_a = max(0.2 - 1.959963984540054 * 0.01, 0.0)
    D2a = (delta_a / den) ** 2
    expected_rva = (np.sqrt(D2a * D2a + 4.0 * D2a) - D2a) / 2.0
    assert np.isclose(float(out["rva"]), expected_rva, atol=1e-12, rtol=0.0)


def test_multi_sensitivity_accepts_vector_cf_d_and_rho():
    est = _dummy_estimate_for_sensitivity(
        theta=np.array([0.2, -0.3]),
        se=np.array([0.01, 0.02]),
        sigma2=1.0,
        nu2=np.array([1.0, 4.0]),
    )
    cf_d = np.array([0.1, 0.3], dtype=float)
    rho = np.array([1.0, 0.5], dtype=float)
    cf_y = 0.2
    out = sensitivity_analysis(est, cf_y=cf_y, cf_d=cf_d, rho=rho, alpha=0.05)

    expected_bound = np.abs(rho) * np.sqrt(np.array([1.0, 4.0])) * np.sqrt(cf_y * cf_d / (1.0 - cf_d))
    assert np.allclose(np.asarray(out["bound_width"], dtype=float), expected_bound, atol=1e-12, rtol=0.0)
    assert np.allclose(np.asarray(out["params"]["cf_d"], dtype=float), cf_d, atol=1e-12, rtol=0.0)
    assert "r2_d" not in out["params"]
    assert np.allclose(np.asarray(out["params"]["rho"], dtype=float), rho, atol=1e-12, rtol=0.0)


def test_multi_normalized_ipw_sensitivity_elements_match_normalized_representer():
    data = _make_synthetic_multi(seed=3030)
    lr_sig = inspect.signature(LogisticRegression)
    lr_kwargs = {"max_iter": 1500}
    if "multi_class" in lr_sig.parameters:
        lr_kwargs["multi_class"] = "multinomial"

    model = MultiTreatmentIRM(
        data=data,
        ml_g=RandomForestRegressor(n_estimators=60, random_state=31),
        ml_m=LogisticRegression(**lr_kwargs),
        n_folds=3,
        normalize_ipw=True,
        random_state=31,
    ).fit()
    est = model.estimate(score="ATE", diagnostic_data=True)
    diag = est.diagnostic_data
    assert diag is not None

    d = np.asarray(diag.d, dtype=float)
    m_hat = np.asarray(diag.m_hat, dtype=float)
    rr = np.asarray(diag.riesz_rep, dtype=float)
    m_alpha = np.asarray(diag.m_alpha, dtype=float)

    h_raw = d / m_hat
    h_mean = np.mean(h_raw, axis=0)
    h_mean = np.where(np.isfinite(h_mean) & (np.abs(h_mean) > 1e-12), h_mean, 1.0)
    expected_rr = d[:, 1:] / m_hat[:, 1:] / h_mean[1:] - d[:, [0]] / m_hat[:, [0]] / h_mean[0]
    expected_m_alpha = (1.0 / m_hat[:, 1:]) / h_mean[1:] + (1.0 / m_hat[:, [0]]) / h_mean[0]

    assert np.allclose(rr, expected_rr, atol=1e-10, rtol=0.0)
    assert np.allclose(m_alpha, expected_m_alpha, atol=1e-10, rtol=0.0)
