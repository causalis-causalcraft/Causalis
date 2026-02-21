import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.model import MultiTreatmentIRM


def _make_multi_causal_data(n: int = 180, seed: int = 42) -> MultiCausalData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)

    labels = np.tile(np.array([0, 1, 2], dtype=int), int(np.ceil(n / 3)))[:n]
    rng.shuffle(labels)
    d = np.eye(3, dtype=int)[labels]

    effects = np.array([0.0, -0.5, 0.8], dtype=float)
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + effects[labels] + rng.normal(0.0, 0.1, size=n)

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

    return MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1", "d2"],
        confounders=["x1", "x2"],
        control_treatment="d0",
    )


def test_multi_treatment_irm_returns_multicausal_estimate():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    )

    result = model.fit().estimate(alpha=0.05)

    assert isinstance(result, MultiCausalEstimate)
    assert result.value.shape == (2,)
    assert result.p_value.shape == (2,)
    assert result.n_control == int(np.sum(data.get_df()["d0"].to_numpy() == 1))
    assert result.n_treated == int(np.sum(data.get_df()[["d1", "d2"]].to_numpy() == 1))
    assert result.contrast_labels == ["d1 vs d0", "d2 vs d0"]
    assert np.array_equal(
        result.n_treated_by_arm,
        np.array(
            [
                int(np.sum(data.get_df()["d1"].to_numpy() == 1)),
                int(np.sum(data.get_df()["d2"].to_numpy() == 1)),
            ]
        ),
    )
    assert result.diagnostic_data is not None
    assert result.diagnostic_data.m_hat_raw is not None
    assert result.diagnostic_data.m_hat_raw.shape == result.diagnostic_data.m_hat.shape


def test_multi_treatment_irm_summary_uses_causalestimate_style_with_one_column_per_contrast():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    )
    result = model.fit().estimate(alpha=0.05)
    summary = result.summary()

    assert summary.index.name == "field"
    assert summary.columns.tolist() == ["d1 vs d0", "d2 vs d0"]
    assert summary.index.tolist() == [
        "estimand",
        "model",
        "value",
        "value_relative",
        "alpha",
        "p_value",
        "is_significant",
        "n_treated",
        "n_control",
        "treatment_mean",
        "control_mean",
        "time",
    ]
    assert "ci_abs:" in summary.loc["value", "d1 vs d0"]
    assert "ci_abs:" in summary.loc["value", "d2 vs d0"]
    assert summary.loc["n_treated", "d1 vs d0"] == int(np.sum(data.get_df()["d1"].to_numpy() == 1))
    assert summary.loc["n_treated", "d2 vs d0"] == int(np.sum(data.get_df()["d2"].to_numpy() == 1))
    assert summary.loc["n_control", "d1 vs d0"] == int(np.sum(data.get_df()["d0"].to_numpy() == 1))
    assert summary.loc["n_control", "d2 vs d0"] == int(np.sum(data.get_df()["d0"].to_numpy() == 1))


def test_multi_treatment_irm_fit_requires_multicausaldata():
    model = MultiTreatmentIRM(
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=2,
    )
    with pytest.raises(TypeError, match="MultiCausalData"):
        model.fit(data="not_multicausaldata")


def test_multi_treatment_irm_enforces_probabilistic_classifier_for_ml_m():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=DummyRegressor(strategy="mean"),
        n_folds=2,
    )
    with pytest.raises(ValueError, match="ml_m must be a classifier"):
        model.fit()


def test_multi_treatment_irm_validates_alpha():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    ).fit()

    with pytest.raises(ValueError, match="alpha must be in"):
        model.estimate(alpha=1.0)


def test_multi_treatment_irm_rejects_too_many_folds():
    data = _make_multi_causal_data(n=9, seed=7)
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=4,
        random_state=1,
    )
    with pytest.raises(ValueError, match="minimum treatment class count"):
        model.fit()


def test_multi_treatment_irm_handles_single_class_binary_outcome_arm_fold():
    n = 90
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(1.0, -1.0, n)
    labels = np.tile(np.array([0, 1, 2], dtype=int), n // 3)
    d = np.eye(3, dtype=int)[labels]

    # Binary outcome with one treatment arm always zero -> per-arm folds can be single-class.
    y = np.zeros(n, dtype=int)
    y[labels == 2] = 1
    # Break exact equality with treatment columns while keeping arm-level degeneracy.
    y[0] = 1

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
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d0", "d1", "d2"],
        confounders=["x1", "x2"],
        control_treatment="d0",
    )

    model = MultiTreatmentIRM(
        data=data,
        ml_g=LogisticRegression(max_iter=1000),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=0,
    )
    result = model.fit().estimate()
    assert isinstance(result, MultiCausalEstimate)
    assert np.all(np.isfinite(model.g_hat_))


def test_multi_treatment_irm_trimmed_propensity_rows_sum_to_one():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        random_state=1,
    ).fit()

    row_sums = model.m_hat_.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10, rtol=0.0)
    assert np.all(model.m_hat_ >= model.trimming_threshold - 1e-12)


def test_multi_treatment_irm_rejects_trimming_threshold_above_one_over_k():
    data = _make_multi_causal_data()
    model = MultiTreatmentIRM(
        data=data,
        ml_g=DummyRegressor(strategy="mean"),
        ml_m=LogisticRegression(max_iter=1000),
        n_folds=3,
        trimming_threshold=0.34,  # K=3 -> must be < 1/3
    )
    with pytest.raises(ValueError, match="1/K"):
        model.fit()
