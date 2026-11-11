"""Clustered observational samples and their use with one-way IRM inference."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causalis.data_contracts import CausalData
from causalis.scenarios.unconfoundedness.dgp import generate_obs_clustered
from causalis.scenarios.unconfoundedness.model import IRM

ORACLE_COLUMNS = {"m", "m_obs", "tau_link", "g0", "g1", "cate"}
CONFOUNDERS = [
    "age_years",
    "tenure_months",
    "avg_sessions_week",
    "spend_last_month",
    "income_monthly",
    "premium_user",
    "mobile_user",
    "market_median_income",
]


def test_clustered_generator_exports_and_causal_data_roles():
    from causalis.dgp import generate_obs_clustered as public_generator
    from causalis.dgp.causaldata import generate_obs_clustered as tabular_generator

    assert public_generator is generate_obs_clustered
    assert tabular_generator is generate_obs_clustered
    data = generate_obs_clustered(n=203, n_clusters=10, seed=19)

    assert isinstance(data, CausalData)
    assert data.cluster_name == "cluster"
    assert data.confounders_names == CONFOUNDERS
    assert data.cluster.nunique() == 10
    assert list(data.X.columns) == CONFOUNDERS
    assert ORACLE_COLUMNS.isdisjoint(data.df.columns)
    sizes = data.cluster.value_counts()
    assert len(data.df) == 203
    assert sizes.max() - sizes.min() == 1


def test_clustered_generator_unequal_sizes_and_cluster_covariate():
    sizes = [1, 3, 8, 13, 2]
    frame = generate_obs_clustered(
        n=sum(sizes),
        n_clusters=len(sizes),
        cluster_sizes=sizes,
        seed=19,
        return_causal_data=False,
    )

    np.testing.assert_array_equal(frame.groupby("cluster", sort=False).size(), sizes)
    assert (frame.groupby("cluster").market_median_income.nunique() == 1).all()
    assert frame.groupby("cluster").market_median_income.first().nunique() == len(sizes)


def test_clustered_generator_customer_features_have_realistic_support():
    frame = generate_obs_clustered(
        n=10_000, n_clusters=100, seed=19, return_causal_data=False
    )

    assert np.isfinite(frame[CONFOUNDERS].to_numpy()).all()
    for column, lower, upper in (
        ("age_years", 18, 80),
        ("tenure_months", 1, 120),
        ("avg_sessions_week", 0, 60),
        ("spend_last_month", 0, 2000),
        ("income_monthly", 500, 30_000),
        ("market_median_income", 20_000, 125_000),
    ):
        assert frame[column].between(lower, upper).all(), column
    for column in ("age_years", "tenure_months"):
        np.testing.assert_array_equal(frame[column], np.rint(frame[column]))
    monthly_sessions = 4 * frame.avg_sessions_week
    np.testing.assert_array_equal(monthly_sessions, np.rint(monthly_sessions))
    for column in ("premium_user", "mobile_user"):
        assert set(frame[column].unique()) == {0, 1}

    # Recent spending includes inactive customers and a right-skewed positive part.
    assert 0.05 < (frame.spend_last_month == 0).mean() < 0.5
    positive_spend = frame.loc[frame.spend_last_month > 0, "spend_last_month"]
    assert positive_spend.mean() > positive_spend.median()


def test_clustered_generator_customer_features_have_plausible_relationships():
    frame = generate_obs_clustered(
        n=10_000, n_clusters=100, seed=19, return_causal_data=False
    )

    assert (
        frame.avg_sessions_week.corr(frame.spend_last_month, method="spearman") > 0.15
    )
    cluster_means = frame.groupby("cluster")[
        ["market_median_income", "income_monthly"]
    ].mean()
    assert cluster_means.market_median_income.corr(cluster_means.income_monthly) > 0.5
    assert (
        frame.loc[frame.premium_user == 1, "income_monthly"].mean()
        > frame.loc[frame.premium_user == 0, "income_monthly"].mean()
    )
    assert (
        frame.loc[frame.mobile_user == 1, "age_years"].mean()
        < frame.loc[frame.mobile_user == 0, "age_years"].mean()
    )


def test_clustered_generator_reproducible_and_oracle_toggle_keeps_observations():
    options = dict(n=300, n_clusters=20, seed=19, return_causal_data=False)
    frame = generate_obs_clustered(**options)
    repeated = generate_obs_clustered(**options)
    observed = generate_obs_clustered(**options, include_oracle=False)
    changed_seed = generate_obs_clustered(**{**options, "seed": 20})

    pd.testing.assert_frame_equal(frame, repeated)
    assert ORACLE_COLUMNS.issubset(frame.columns)
    assert ORACLE_COLUMNS.isdisjoint(observed.columns)
    pd.testing.assert_frame_equal(frame.drop(columns=list(ORACLE_COLUMNS)), observed)
    assert not frame.equals(changed_seed)


def test_clustered_generator_oracle_effects_and_noiseless_outcomes():
    frame = generate_obs_clustered(
        n=300,
        n_clusters=20,
        theta=1.7,
        heterogeneity=0.6,
        sigma_y=0.0,
        sigma_cluster=0.0,
        seed=19,
        return_causal_data=False,
    )

    np.testing.assert_allclose(frame.cate, frame.g1 - frame.g0, atol=1e-12)
    market_score = np.log(frame.market_median_income / 50_000) / 0.3
    np.testing.assert_allclose(frame.cate, 1.7 + 0.6 * np.tanh(market_score))
    np.testing.assert_allclose(frame.tau_link, frame.cate)
    np.testing.assert_allclose(frame.y, frame.g0 + frame.d * frame.cate, atol=1e-12)
    assert ((frame.m > 0) & (frame.m < 1)).all()
    assert ((frame.m_obs >= 0) & (frame.m_obs <= 1)).all()
    assert set(frame.d.unique()) == {0.0, 1.0}


def test_clustered_generator_outcome_shocks_are_shared_within_clusters():
    options = dict(n=300, n_clusters=20, sigma_y=0.0, seed=19, return_causal_data=False)
    frame = generate_obs_clustered(**options, sigma_cluster=1.3)
    noiseless = generate_obs_clustered(**options, sigma_cluster=0.0)
    residual = frame.y - frame.g0 - frame.d * frame.cate

    # Only the shared shock remains when observation-level noise is zero.
    within_cluster_ranges = residual.groupby(frame.cluster).agg(
        lambda x: x.max() - x.min()
    )
    np.testing.assert_allclose(within_cluster_ranges, 0.0, atol=1e-12)
    assert residual.groupby(frame.cluster).first().std() > 0.5
    pd.testing.assert_frame_equal(frame.drop(columns="y"), noiseless.drop(columns="y"))


def test_clustered_generator_treatment_correlation_endpoints():
    options = dict(n=400, n_clusters=20, seed=19, return_causal_data=False)
    independent = generate_obs_clustered(**options, treatment_correlation=0.0)
    shared = generate_obs_clustered(**options, treatment_correlation=1.0)

    np.testing.assert_allclose(independent.m_obs, independent.m)
    np.testing.assert_array_equal(shared.m_obs, shared.d)
    # Perfect latent correlation gives each cluster a single propensity threshold.
    for _, group in shared.groupby("cluster"):
        assert (np.diff(group.sort_values("m").d) >= 0).all()
    pd.testing.assert_frame_equal(
        independent[CONFOUNDERS + ["cluster", "m", "g0", "g1", "cate"]],
        shared[CONFOUNDERS + ["cluster", "m", "g0", "g1", "cate"]],
    )


def test_clustered_generator_induces_dependence_in_oracle_ate_scores():
    frame = generate_obs_clustered(
        n=4000,
        n_clusters=200,
        treatment_correlation=0.8,
        sigma_cluster=2.0,
        sigma_y=0.3,
        heterogeneity=0.0,
        seed=19,
        return_causal_data=False,
    )
    # With constant effects, this is the oracle ATE influence function.
    influence = frame.d * (frame.y - frame.g1) / frame.m - (1 - frame.d) * (
        frame.y - frame.g0
    ) / (1 - frame.m)
    influence -= influence.mean()
    cluster_sums = influence.groupby(frame.cluster).sum()
    clustered_variance = (200 / 199) * np.sum(cluster_sums**2) / len(frame) ** 2
    iid_variance = np.var(influence, ddof=1) / len(frame)

    # Shared outcome shocks alone need not correlate ATE scores; shared
    # assignment shocks make this sample useful for clustered-inference tests.
    assert clustered_variance > 2 * iid_variance


@pytest.mark.parametrize(
    "options, parameter",
    [
        ({"n": 0}, "n"),
        ({"n": True}, "n"),
        ({"n": 100.5}, "n"),
        ({"n_clusters": 1}, "n_clusters"),
        ({"n_clusters": True}, "n_clusters"),
        ({"n_clusters": 2.5}, "n_clusters"),
        ({"n_clusters": 101}, "n_clusters"),
        ({"cluster_sizes": [50, 50]}, "cluster_sizes"),
        ({"cluster_sizes": [10] * 9 + [0]}, "cluster_sizes"),
        ({"cluster_sizes": [10] * 9 + [11]}, "cluster_sizes"),
        ({"cluster_sizes": [10.0] * 10}, "cluster_sizes"),
        ({"cluster_sizes": [True] * 10}, "cluster_sizes"),
        ({"cluster_sizes": [[10] * 10]}, "cluster_sizes"),
        ({"theta": np.nan}, "theta"),
        ({"heterogeneity": np.inf}, "heterogeneity"),
        ({"sigma_y": -1}, "sigma_y"),
        ({"sigma_y": np.nan}, "sigma_y"),
        ({"sigma_cluster": -1}, "sigma_cluster"),
        ({"sigma_cluster": np.inf}, "sigma_cluster"),
        ({"treatment_correlation": -0.1}, "treatment_correlation"),
        ({"treatment_correlation": 1.1}, "treatment_correlation"),
        ({"treatment_correlation": np.nan}, "treatment_correlation"),
    ],
)
def test_clustered_generator_rejects_invalid_configuration(options, parameter):
    configuration = dict(n=100, n_clusters=10, return_causal_data=False)
    configuration.update(options)
    with pytest.raises(ValueError, match=parameter):
        generate_obs_clustered(**configuration)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_clustered_generator_supports_irm_estimation_and_grouped_folds(score):
    # Exact linear outcome regressions isolate DGP integration from learner error.
    data = generate_obs_clustered(
        n=600,
        n_clusters=30,
        theta=1.7,
        heterogeneity=0.0,
        sigma_y=0.0,
        sigma_cluster=0.0,
        seed=19,
    )
    model = IRM(
        data,
        ml_g=make_pipeline(StandardScaler(), LinearRegression()),
        ml_m=make_pipeline(StandardScaler(), LogisticRegression()),
        n_folds=3,
        random_state=19,
    ).fit()
    result = model.estimate(score=score)

    assert result.value == pytest.approx(1.7, abs=1e-10)
    assert np.isfinite(model.se_[0])
    assert result.model_options["cov_type"] == "cluster"
    assert result.model_options["cluster_name"] == "cluster"
    assert result.model_options["n_clusters"] == 30
    assert result.model_options["inference_df"] == 29
    fold_counts = pd.Series(model.folds_).groupby(data.cluster.to_numpy()).nunique()
    assert (fold_counts == 1).all()
