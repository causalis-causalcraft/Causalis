import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.special import expit

from causalis.data_contracts import RctCausalData
from causalis.dgp import generate_rct_causal_data
from causalis.scenarios.cuped import CUPEDModel


def test_defaults_reproducible_and_raw_parity():
    data = generate_rct_causal_data(n=1000, seed=7)
    assert isinstance(data, RctCausalData)
    assert data.Y.shape == (1000, 3)
    assert data.D.shape == (1000, 3)
    assert data.confounders == ["x1", "x2"]
    assert data.control_treatment == "d_0"
    assert (data.D.sum(axis=1) == 1).all()
    assert data.user_id.is_unique
    assert_frame_equal(data.df, generate_rct_causal_data(n=1000, seed=7).df)
    raw = generate_rct_causal_data(n=1000, seed=7, return_causal_data=False)
    assert_frame_equal(data.df, raw[list(data.df)])
    assert data.df.attrs["rct"] == raw.attrs["rct"]
    assert not data.df.equals(generate_rct_causal_data(n=1000, seed=8).df)


def test_binary_no_covariates_no_ids():
    data = generate_rct_causal_data(n=1000, n_treatments=2, treatment_encoding="binary",
                                    d_names=["treated"], k=0, n_outcomes=1, include_user_id=False)
    assert data.treatment_name == "treated"
    assert data.control_treatment is None
    assert data.X.shape == (1000, 0)
    assert data.user_id.empty
    assert set(data.treatment) == {0, 1}
    estimate = CUPEDModel().fit(data, covariates=[], run_checks=False).estimate()
    assert estimate.n_control + estimate.n_treated == 1000


def test_mixed_families_oracles_and_config_unchanged():
    specs = [dict(name="revenue", outcome_type="gamma", alpha_y=1., theta=[0.2, 0.4, 0.6]),
             dict(name="conversion", outcome_type="binary", alpha_y=-0.5, theta=[0.4, 0.8]),
             dict(name="orders", outcome_type="poisson", theta=0.3),
             dict(name="score", outcome_type="normal", theta=-0.5)]
    raw = generate_rct_causal_data(n=3000, outcome_specs=specs, d_names=["control", "a", "b"],
                                   k=0, include_oracle=True, return_causal_data=False)
    assert specs[0]["theta"] == [0.2, 0.4, 0.6]
    assert (raw.revenue > 0).all()
    assert set(raw.conversion) == {0, 1}
    assert (raw.orders >= 0).all()
    assert (raw.orders % 1 == 0).all()
    np.testing.assert_allclose(raw.g_revenue_control, np.exp(1.2))
    np.testing.assert_allclose(raw.g_revenue_b, np.exp(1.6))
    np.testing.assert_allclose(raw.g_conversion_a, expit(-0.1))
    np.testing.assert_allclose(raw.g_orders_a, np.exp(0.3))
    for name in [s["name"] for s in specs]:
        for arm in ["a", "b"]:
            np.testing.assert_allclose(raw[f"cate_{name}_{arm}"], raw[f"g_{name}_{arm}"] - raw[f"g_{name}_control"])
    np.testing.assert_allclose(raw[["m_control", "m_a", "m_b"]].sum(axis=1), 1)
    assert raw.attrs["rct"]["outcomes"]["score"]["outcome_type"] == "continuous"
    np.testing.assert_allclose(raw.attrs["rct"]["outcomes"]["revenue"]["effects_link"], [0, .2, .4])
    contract = generate_rct_causal_data(n=3000, outcome_specs=specs, k=0, include_oracle=True)
    assert not any(name.startswith(("g_", "cate_", "m_")) for name in contract.df)


def test_covariate_schema_expansion():
    data = generate_rct_causal_data(
        n=1000, confounder_specs=[
            {"name": "pre_spend", "dist": "gamma", "shape": 2, "mean": 2},
            {"name": "platform", "dist": "categorical", "categories": [0, 1, 2]},
        ], outcome_specs=[{"beta_y": [0.5, 1., -1.]}],
    )
    assert data.confounders == ["pre_spend", "platform_1", "platform_2"]
    assert (data.X.pre_spend > 0).all()


def test_assignment_is_shared_randomized_and_outcome_independent():
    options = dict(n=20_000, target_d_rate=[1, 2, 7], seed=11)
    first = generate_rct_causal_data(**options, outcome_specs=[{"theta": .1}])
    second = generate_rct_causal_data(**options, outcome_specs=[{"theta": 10.}, {}, {}])
    assert_frame_equal(first.D, second.D)
    assert_frame_equal(first.X, second.X)
    np.testing.assert_allclose(first.D.mean(), [.1, .2, .7], atol=.015)
    assert abs(np.corrcoef(first.df.x1, first.df.d_1)[0, 1]) < .03
    raw = generate_rct_causal_data(**options, outcome_specs=[{"theta": .1}], include_oracle=True, return_causal_data=False)
    np.testing.assert_array_equal(first.df.y1, raw.y1)


def test_cuped_recovers_known_effects_and_reduces_variance():
    data = generate_rct_causal_data(n=12000, seed=92, k=1, outcome_specs=[
        {"name": "spend", "alpha_y": 20, "beta_y": [3], "theta": [1, 2], "sigma_y": .5},
        {"name": "visits", "alpha_y": 10, "beta_y": [2], "theta": [-.5, 1.5], "sigma_y": .5},
    ])
    estimates = CUPEDModel().fit(data, covariates=["x1"], run_checks=False).estimate()
    for name, effects in [("spend", [1, 2]), ("visits", [-.5, 1.5])]:
        for arm, expected in zip(["d_1", "d_2"], effects):
            assert estimates[name][arm].value == pytest.approx(expected, abs=.05)
            assert estimates[name][arm].diagnostic_data.variance_reduction_pct_same_cov > 90


@pytest.mark.parametrize("kwargs,message", [
    ({"n": 1}, "n must"), ({"n": 2, "n_treatments": 3}, "n must"),
    ({"n": 2.5}, "integer"), ({"n_treatments": 1}, "integer"),
    ({"n_outcomes": 0}, "integer"), ({"k": -1}, "integer"),
    ({"treatment_encoding": "bad"}, "treatment_encoding"),
    ({"treatment_encoding": "binary"}, "n_treatments=2"),
    ({"d_names": ["a", "b"]}, "length"),
    ({"d_names": ["a", "a", "b"]}, "duplicate"),
    ({"target_d_rate": [0, .5, .5]}, "positive"),
    ({"target_d_rate": [1, np.nan, 1]}, "finite"),
    ({"target_d_rate": [1, 1]}, "per arm"),
    ({"outcome_specs": []}, "non-empty"),
    ({"outcome_specs": [{"oops": 1}]}, "Unknown"),
    ({"outcome_specs": [{"name": "x1"}]}, "disjoint"),
    ({"outcome_specs": [{"name": "user_id"}]}, "disjoint"),
    ({"outcome_specs": [{"name": "a"}, {"name": "a"}]}, "duplicate"),
    ({"outcome_specs": [{"outcome_type": "bad"}]}, "outcome_type"),
    ({"outcome_specs": [{"theta": [1, 2, 3, 4]}]}, "theta"),
    ({"outcome_specs": [{"theta": np.inf}]}, "finite"),
    ({"outcome_specs": [{"beta_y": [1]}]}, "beta_y"),
    ({"outcome_specs": [{"sigma_y": -1}]}, "sigma_y"),
    ({"outcome_specs": [{"gamma_shape": 0}]}, "gamma_shape"),
])
def test_invalid_configuration(kwargs, message):
    options = {"n": 1000, **kwargs}
    with pytest.raises(ValueError, match=message):
        generate_rct_causal_data(**options)


def test_absent_arms_are_not_fabricated():
    with pytest.raises(ValueError, match="Not all treatment arms"):
        generate_rct_causal_data(n=100, target_d_rate=[1, 1e-100, 1e-100])


def test_oracle_collision_and_binary_oracle_names():
    with pytest.raises(ValueError, match="disjoint"):
        generate_rct_causal_data(n=1000, outcome_specs=[{"name": "m_d_0"}], include_oracle=True)
    raw = generate_rct_causal_data(n=1000, n_treatments=2, treatment_encoding="binary",
                                  n_outcomes=1, include_oracle=True, return_causal_data=False)
    assert {"m_0", "m_1", "g_y1_0", "g_y1_1", "cate_y1_1"}.issubset(raw)


def test_no_global_rng_mutation_and_many_outcomes():
    state = np.random.get_state()
    data = generate_rct_causal_data(n=1000, n_outcomes=100)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert data.Y.shape == (1000, 100)
