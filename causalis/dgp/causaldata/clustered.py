"""Clustered observational samples for testing unconfoundedness estimators."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from causalis.data_contracts.causaldata import CausalData
from causalis.dgp.base import _sigmoid
from .base import CausalDatasetGenerator

_CONFOUNDER_NAMES = [
    "age_years",
    "tenure_months",
    "avg_sessions_week",
    "spend_last_month",
    "income_monthly",
    "premium_user",
    "mobile_user",
    "market_median_income",
]


def _sample_customer_confounders(
    rng: np.random.Generator, clusters: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Sample customers with related demographics, usage, and purchasing power."""
    n = len(clusters)
    market_income = (
        50_000
        * np.exp(np.clip(0.3 * rng.normal(size=n_clusters), -np.log(2.5), np.log(2.5)))[
            clusters
        ]
    )
    # These latent factors only generate observed covariates; they do not enter
    # assignment or outcome equations as unmeasured common causes.
    engagement = rng.normal(size=n)
    affluence = rng.normal(size=n)
    age = np.rint(np.clip(rng.normal(38 + 2 * affluence, 12, n), 18, 80))
    tenure = np.rint(
        np.clip(
            rng.gamma(shape=2.0, scale=12.0, size=n)
            * np.exp(0.15 * engagement + 0.1 * (age - 38) / 12),
            1,
            120,
        )
    )
    income = np.clip(
        rng.lognormal(
            mean=np.log(market_income / 12) + 0.35 * affluence + 0.1 * (age - 38) / 12,
            sigma=0.4,
            size=n,
        ),
        500,
        30_000,
    )
    premium = rng.binomial(
        1,
        _sigmoid(
            -1.5
            + 0.5 * np.log(income / 4_000)
            + 0.25 * np.log1p(tenure / 12)
            + 0.3 * engagement
        ),
    )
    mobile = rng.binomial(1, _sigmoid(1.0 - 0.35 * (age - 38) / 12 + 0.25 * engagement))
    weekly_mean = 5 * np.exp(0.35 * engagement + 0.25 * premium + 0.1 * mobile)
    # Four-week counts are overdispersed, with averages in quarter-session steps.
    sessions = np.minimum(rng.negative_binomial(2, 2 / (2 + 4 * weekly_mean)) / 4, 60)
    active_buyer = rng.binomial(1, _sigmoid(1.2 + 0.45 * engagement + 0.3 * premium))
    spend = active_buyer * np.minimum(
        rng.lognormal(
            mean=np.log(60)
            + 0.4 * np.log(income / 4_000)
            + 0.35 * (np.log1p(sessions) - np.log(6))
            + 0.25 * premium,
            sigma=0.7,
            size=n,
        ),
        2_000,
    )
    return np.column_stack(
        (age, tenure, sessions, spend, income, premium, mobile, market_income)
    )


def _customer_treatment_score(x: np.ndarray) -> np.ndarray:
    """Assign treatment from observed customer and market characteristics."""
    age, tenure, sessions, spend, income, premium, mobile, market_income = x.T
    return (
        -0.3
        + 0.1 * (age - 38) / 12
        + 0.15 * (np.log1p(tenure) - np.log(25))
        + 0.4 * (np.log1p(sessions) - np.log(6))
        + 0.2 * (np.log1p(spend) - np.log(61))
        + 0.3 * np.log(income / 4_000)
        + 0.55 * premium
        + 0.15 * mobile
        + 0.5 * np.log(market_income / 50_000) / 0.3
    )


def generate_obs_clustered(
    n: int = 10_000,
    n_clusters: int = 100,
    *,
    cluster_sizes: Optional[Sequence[int]] = None,
    theta: float = 1.0,
    heterogeneity: float = 0.5,
    sigma_y: float = 1.0,
    sigma_cluster: float = 1.0,
    treatment_correlation: float = 0.5,
    seed: Optional[int] = 42,
    include_oracle: bool = True,
    return_causal_data: bool = True,
) -> Union[pd.DataFrame, CausalData]:
    r"""Generate binary treatment and a continuous outcome in independent clusters.

    Clusters represent local customer markets. Observed pre-treatment
    confounders are ``age_years`` (18--80), ``tenure_months`` (1--120),
    ``avg_sessions_week`` (0--60), ``spend_last_month`` (0--2000),
    ``income_monthly`` (500--30000), binary ``premium_user`` and ``mobile_user``,
    and cluster-level annual ``market_median_income`` (20000--125000).
    Income and positive spending are right-skewed; spending includes nonbuyers.
    Session averages come from overdispersed monthly counts. Engagement links
    tenure, premium status, activity and spending; market income shifts customer
    incomes, and mobile usage decreases with age. These are illustrative
    synthetic relationships, without calibration to a particular population.

    With X in the column order above, treatment and outcomes follow

    .. math::

        m(X) &= \operatorname{expit}\{2\tanh[s(X)/2]\}, \\
        g_0(X) &= 5 + X\beta_y, \\
        \tau(X) &= \theta + h\tanh\{\log(I_c/50000)/0.3\}, \\
        Y_i &= g_0(X_i) + D_i\tau(X_i) + B_{c(i)} + \epsilon_i.

    Here ``I_c`` is observed market median income, ``h = heterogeneity``,
    and ``beta_y = (0.01, 0.02, 0.12, 0.0075, 0.00008, 0.8, 0.2, 0.000015)``.
    The treatment score ``s(X)`` uses centered age, log tenure, log sessions,
    log spending, log income, premium/mobile indicators and observed market
    income. It depends only on observed pre-treatment confounders.
    Outcome shocks satisfy ``B_c ~ N(0, sigma_cluster**2)`` and
    ``epsilon_i ~ N(0, sigma_y**2)``. Treatment uses a Gaussian copula:
    ``D_i = 1[Phi(sqrt(rho)*Z_c + sqrt(1-rho)*Z_i) < m(X_i)]``.
    Assignment shocks, outcome shocks, and confounders use independent random
    streams. Thus treatment and outcome can both depend within clusters while
    ``(Y(0), Y(1))`` remains independent of treatment conditional on observed X.
    There are no spillovers between observations.

    Parameters
    ----------
    n : int, default=10000
        Total number of observations; must be at least ``n_clusters``.
    n_clusters : int, default=100
        Number of independent clusters; must be at least two.
    cluster_sizes : sequence of int, optional
        Positive sizes in cluster order, with length ``n_clusters`` and sum
        ``n``. Otherwise sizes differ by at most one, with larger clusters first.
        Sizes are fixed independently of all sampled variables.
    theta : float, default=1.0
        Population ATE. The sample oracle ATE is ``df.cate.mean()``; the sample
        oracle ATTE is ``df.loc[df.d == 1, 'cate'].mean()``. Both weight rows,
        including when cluster sizes differ.
    heterogeneity : float, default=0.5
        Coefficient of the bounded, centered log market-income signal in the
        treatment effect. Set to zero for a constant effect.
    sigma_y : float, default=1.0
        Nonnegative standard deviation of individual outcome noise.
    sigma_cluster : float, default=1.0
        Nonnegative standard deviation of the independent cluster outcome shock.
        Conditional outcome residual ICC is
        ``sigma_cluster**2 / (sigma_cluster**2 + sigma_y**2)`` when defined.
    treatment_correlation : float, default=0.5
        Correlation ``rho`` of the latent treatment normals within a cluster,
        in [0, 1]. This is not the ICC of the binary treatment. Zero gives
        conditionally independent assignments; one uses a shared threshold.
    seed : int, optional, default=42
        Seed for reproducible independent random streams. None uses fresh entropy.
    include_oracle : bool, default=True
        Include ``m``, ``m_obs``, ``g0``, ``g1``, ``cate``, and ``tau_link`` in
        DataFrame output. ``m`` is the marginal propensity given X; ``m_obs``
        additionally conditions on the latent cluster assignment shock.
        Outcome means marginalize over the independent cluster outcome shock.
    return_causal_data : bool, default=True
        Return ``CausalData`` with ``cluster='cluster'`` and the eight observed
        confounders. This contract retains only analysis columns;
        request False to inspect oracle columns.

    Returns
    -------
    pandas.DataFrame or CausalData
        Rows ordered by integer cluster labels ``0, ..., n_clusters - 1``.
        Treatment and binary confounder support are stochastic; very small
        samples may lack one arm or have constant confounders and cannot be
        wrapped as ``CausalData`` or fitted by ``IRM``.

    Examples
    --------
    >>> from causalis.dgp import generate_obs_clustered
    >>> data = generate_obs_clustered(n=600, n_clusters=30, seed=19)
    >>> data.cluster_name, data.cluster.nunique()
    ('cluster', 30)
    >>> frame = generate_obs_clustered(
    ...     n=600, n_clusters=30, seed=19, return_causal_data=False
    ... )
    >>> bool(np.allclose(frame.g1 - frame.g0, frame.cate))
    True
    """
    for name, value in (("n", n), ("n_clusters", n_clusters)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least two.")
    if n < n_clusters:
        raise ValueError("n must be at least n_clusters.")

    for name, value in (
        ("theta", theta),
        ("heterogeneity", heterogeneity),
        ("sigma_y", sigma_y),
        ("sigma_cluster", sigma_cluster),
        ("treatment_correlation", treatment_correlation),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real number.")
    if sigma_y < 0 or sigma_cluster < 0:
        raise ValueError("sigma_y and sigma_cluster must be nonnegative.")
    if not 0 <= treatment_correlation <= 1:
        raise ValueError("treatment_correlation must be in [0, 1].")

    if cluster_sizes is None:
        sizes = np.full(n_clusters, n // n_clusters, dtype=np.intp)
        sizes[: n % n_clusters] += 1
    else:
        sizes = np.asarray(cluster_sizes, dtype=object)
        if sizes.ndim != 1 or len(sizes) != n_clusters:
            raise ValueError("cluster_sizes must have length n_clusters.")
        if any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, Integral)
            or size < 1
            for size in sizes
        ):
            raise ValueError("cluster_sizes must contain positive integers.")
        if sum(int(size) for size in sizes) != n:
            raise ValueError("cluster_sizes must sum to n.")
        sizes = sizes.astype(np.intp)

    x_seed, base_seed, assignment_seed, outcome_seed = np.random.SeedSequence(
        seed
    ).spawn(4)
    x_rng = np.random.default_rng(x_seed)
    assignment_rng = np.random.default_rng(assignment_seed)
    outcome_rng = np.random.default_rng(outcome_seed)
    clusters = np.repeat(np.arange(n_clusters), sizes)
    x = _sample_customer_confounders(x_rng, clusters, n_clusters)
    confounders = _CONFOUNDER_NAMES.copy()

    # Reuse the tabular DGP's bounded propensity, continuous model and oracle
    # definitions, then couple assignments and outcome residuals within groups.
    generator = CausalDatasetGenerator(
        confounder_specs=[{"name": name} for name in confounders],
        x_sampler=lambda _n, _k, _seed: x,
        alpha_y=5.0,
        beta_y=np.array([0.01, 0.02, 0.12, 0.0075, 0.00008, 0.8, 0.2, 0.000015]),
        g_d=_customer_treatment_score,
        tau=lambda X: theta + heterogeneity * np.tanh(np.log(X[:, 7] / 50_000) / 0.3),
        sigma_y=sigma_y,
        score_bounding=2.0,
        include_oracle=True,
        seed=int(base_seed.generate_state(1)[0]),
    )
    df = generator.generate(n)
    rho = float(treatment_correlation)
    shared_assignment = assignment_rng.normal(size=n_clusters)[clusters]
    individual_assignment = assignment_rng.normal(size=n)
    threshold = ndtri(df["m"].to_numpy())
    latent_assignment = (
        np.sqrt(rho) * shared_assignment + np.sqrt(1 - rho) * individual_assignment
    )
    treatment = (latent_assignment < threshold).astype(float)
    df["y"] += df["cate"] * (treatment - df["d"].to_numpy())
    df["y"] += outcome_rng.normal(scale=sigma_cluster, size=n_clusters)[clusters]
    df["d"] = treatment
    df["cluster"] = clusters

    if return_causal_data:
        return CausalData(
            df=df,
            treatment="d",
            outcome="y",
            confounders=confounders,
            cluster="cluster",
        )
    if include_oracle:
        df["m_obs"] = (
            ndtr((threshold - np.sqrt(rho) * shared_assignment) / np.sqrt(1 - rho))
            if rho < 1
            else treatment
        )
    else:
        df = df.drop(columns=["m", "m_obs", "tau_link", "g0", "g1", "cate"])
    return df


__all__ = ["generate_obs_clustered"]
