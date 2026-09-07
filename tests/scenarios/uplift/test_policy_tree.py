"""Policy learning, independent evaluation, and client export contracts."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.exceptions import NotFittedError

from causalis.data_contracts import CausalData
from causalis.scenarios.unconfoundedness import IRM
from causalis.scenarios.uplift import UpliftPolicyEvaluation, UpliftPolicyTree


def make_irm(
    prefix="train",
    *,
    tau=None,
    noise=0.0,
    seed=42,
    x_values=(-2.0, -1.0, 1.0, 2.0),
    separated=False,
    store_diagnostics=True,
    **kwargs,
):
    rng = np.random.default_rng(seed)
    x = np.repeat(x_values, 30)
    z = rng.normal(size=len(x))
    d = (x > 0).astype(int) if separated else np.tile([0, 1], len(x) // 2)
    effect = 2 * x if tau is None else np.full(len(x), tau)
    y = 10 + 0.7 * x + 0.2 * z + effect * d + noise * rng.normal(size=len(x))
    frame = pd.DataFrame(
        {
            "client_id": [f"{prefix}_{i}" for i in range(len(x))],
            "x": x,
            "z": z,
            "d": d,
            "y": y,
        },
        index=np.arange(len(x)) * 3 + 7,
    )
    data = CausalData.from_df(
        frame, outcome="y", treatment="d", confounders=["x", "z"], user_id="client_id"
    )
    return IRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        n_folds=2,
        random_state=seed,
        store_diagnostics=store_diagnostics,
        **kwargs,
    ).fit()


@pytest.fixture
def train():
    return make_irm()


def fit_policy(train, **kwargs):
    return UpliftPolicyTree(min_samples_leaf=20, **kwargs).fit(
        train, policy_features=["x"]
    )


def manual_signal(irm):
    y, d = irm.data.outcome.to_numpy(), irm.data.treatment.to_numpy()
    g0, g1, p = irm.g0_hat_, irm.g1_hat_, irm.m_hat_
    return g1 - g0 + d * (y - g1) / p - (1 - d) * (y - g0) / (1 - p)


def test_discovers_correct_regions_and_exact_boundary(train):
    policy = fit_policy(train)
    rules = policy.rules()
    assert len(rules) == 2
    assert rules.action.tolist() == [0, 1]
    assert rules.predicates.tolist() == [(("x", "<=", 0.0),), (("x", ">", 0.0),)]
    np.testing.assert_allclose(rules.train_mean_uplift_descriptive, [-3, 3])
    clients = pd.DataFrame(
        {"id": ["a", "b", "c"], "x": [-1.0, 0.0, 1.0]}, index=[9, 3, 9]
    )
    result = policy.assign(clients, user_id="id")
    assert result.action.tolist() == [0, 0, 1]
    assert result.index.tolist() == [9, 3, 9]
    assert result.id.tolist() == ["a", "b", "c"]
    assert "z" not in clients  # Adjustment-only features are not needed to assign.


def test_independent_evaluation_manual_values_intervals_and_gate(train):
    policy = fit_policy(train)
    eval_irm = make_irm("eval", noise=0.5, seed=12)
    before_rules = policy.rules()
    before_assignments = policy.assign(eval_irm.data.df, user_id="client_id")
    phi = manual_signal(eval_irm)
    a = before_assignments.action.to_numpy()
    # The cached score is deliberately ATTE; neither fit nor evaluate may use it.
    train.estimate(score="ATTE")
    eval_irm.estimate(score="ATTE")
    old_scores = (train.score, eval_irm.score)
    report = policy.evaluate(eval_irm, alpha=0.1)
    assert isinstance(report, UpliftPolicyEvaluation)
    assert (train.score, eval_irm.score) == old_scores
    for row, signal in zip(
        report.summary().itertuples(), [a * phi, (a - 1) * phi, phi]
    ):
        expected_se = np.std(signal, ddof=1) / np.sqrt(len(signal))
        assert row.value == pytest.approx(np.mean(signal))
        assert row.std_error == pytest.approx(expected_se)
        assert row.ci_lower == pytest.approx(
            np.mean(signal) - norm.ppf(0.95) * expected_se
        )
        assert row.ci_upper == pytest.approx(
            np.mean(signal) + norm.ppf(0.95) * expected_se
        )
        assert row.treatment_fraction == 0.5
        assert row.n_obs == len(phi)
    for row in report.rules_summary().itertuples():
        values = phi[before_assignments.rule_id.to_numpy() == row.rule_id]
        se = np.sqrt(np.sum((values - values.mean()) ** 2)) / (len(values) - 1)
        assert row.value == pytest.approx(np.mean(values))
        assert row.std_error == pytest.approx(se)
        assert row.status == "ok"
    groups = pd.Series(
        before_assignments.rule_id.to_numpy(), index=eval_irm.data.user_id
    )
    gate = eval_irm.gate(groups=groups, alpha=0.1).summary()
    np.testing.assert_allclose(report.rules_summary().value, gate.value)
    np.testing.assert_allclose(report.rules_summary().ci_lower, gate.ci_lower)
    pd.testing.assert_frame_equal(policy.rules(), before_rules)
    pd.testing.assert_frame_equal(
        policy.assign(eval_irm.data.df, user_id="client_id"), before_assignments
    )
    again = fit_policy(train)
    pd.testing.assert_frame_equal(again.rules(), before_rules)


@pytest.mark.parametrize("tau,action", [(2.0, 1), (-2.0, 0)])
def test_constant_effect_policy(tau, action):
    policy = fit_policy(make_irm(tau=tau))
    assert len(policy.rules()) == 1
    assert policy.rules().action.item() == action
    assert policy.rules().conditions.item() == "All clients"


def test_zero_signal_ties_do_not_treat(train):
    # Exact signals isolate the tree's tie convention from regression roundoff.
    train.g0_hat_ = train.data.outcome.to_numpy().copy()
    train.g1_hat_ = train.g0_hat_.copy()
    policy = fit_policy(train)
    assert policy.rules().action.tolist() == [0]


@pytest.mark.parametrize(
    "settings",
    [dict(max_depth=0), dict(min_samples_leaf=100), dict(min_samples_per_arm=40)],
)
def test_split_constraints_preserve_root(train, settings):
    policy = UpliftPolicyTree(**settings).fit(train)
    assert len(policy.rules()) == 1


def test_ties_follow_feature_order_and_threshold_order(train):
    # Duplicate signals at multiple thresholds; x and 3*x give identical partitions.
    frame = train.data.df.copy()
    frame["z"] = 3 * frame.x
    data = CausalData.from_df(
        frame, outcome="y", treatment="d", confounders=["x", "z"], user_id="client_id"
    )
    irm = IRM(data, ml_g=LinearRegression(), ml_m=LogisticRegression(), n_folds=2).fit()
    phi = np.repeat([-1.0, 0.0, 0.0, 1.0], 30)
    y, d = irm.data.outcome.to_numpy(), irm.data.treatment.to_numpy()
    irm.g0_hat_ = y - d * phi
    irm.g1_hat_ = y + (1 - d) * phi
    policy = UpliftPolicyTree(min_samples_leaf=20).fit(irm, policy_features=["z", "x"])
    assert policy.rules().predicates.iloc[0] == (("z", "<=", -4.5),)
    assert policy.rules().rule_id.tolist() == ["rule_1", "rule_2"]


def test_empty_and_unsupported_evaluation_leaves(train):
    policy = fit_policy(train)
    empty = policy.evaluate(make_irm("positive", x_values=(1.0, 2.0)))
    row = empty.rules_summary().iloc[0]
    assert row.status == "empty" and row.n_eval == 0 and np.isnan(row.value)
    unsupported_irm = make_irm("unsupported", separated=True)
    report = policy.evaluate(unsupported_irm)
    assert report.rules_summary().status.tolist() == ["insufficient_arm_support"] * 2
    assert report.rules_summary().value.isna().all()
    assert report.summary().value.notna().all()
    assert report.summary().n_obs.eq(len(unsupported_irm.data.df)).all()


def test_clipping_diagnostics_without_storing_diagnostics():
    # An extreme propensity prediction is clipped and counted even with storage off.
    irm = make_irm(separated=True, overlap_threshold=0.49, store_diagnostics=False)
    policy = fit_policy(irm)
    assert irm.m_hat_raw_ is None
    assert policy.training_diagnostics_["n_clipped"] > 0
    assert policy.training_diagnostics_["overlap_threshold"] == 0.49
    report = policy.evaluate(make_irm("eval", store_diagnostics=False))
    assert report.diagnostics["training"] == policy.training_diagnostics_
    assert report.diagnostics["evaluation"]["n_obs"] == 120


def test_csv_exports_partition_clients_and_preserve_id_dtype(train, tmp_path):
    clients = train.data.df.iloc[::-1].copy()
    clients.client_id = np.arange(len(clients), dtype=np.int64)
    policy = fit_policy(train)
    assignments = policy.assign(clients, user_id="client_id")
    assert assignments.client_id.dtype == clients.client_id.dtype
    exported = []
    for action, name in [(1, "target"), (0, "do_not_target")]:
        path = tmp_path / f"{name}.csv"
        assignments.loc[assignments.action == action, ["client_id"]].to_csv(
            path, index=False
        )
        exported.append(set(pd.read_csv(path).client_id))
    assert not exported[0] & exported[1]
    assert exported[0] | exported[1] == set(clients.client_id)
    assert assignments.client_id.tolist() == clients.client_id.tolist()
    empty = policy.assign(clients.iloc[:0], user_id="client_id")
    assert empty.columns.tolist() == ["client_id", "rule_id", "action"]
    assert empty.empty


def test_rejects_overlapping_ids(train):
    with pytest.raises(ValueError, match="disjoint"):
        fit_policy(train).evaluate(train)


@pytest.mark.parametrize("change", ["x", "y", "d", "ids", "row_order", "roles"])
def test_rejects_changed_fit_data(train, change):
    if change in {"x", "y", "d"}:
        train.data.df.loc[train.data.df.index[0], change] += 1
    elif change == "ids":
        train.data.df.loc[train.data.df.index[0], "client_id"] = "changed"
    elif change == "row_order":
        train.data.df = train.data.df.iloc[::-1]
    else:
        train.data.confounders_names = ["z", "x"]
    with pytest.raises((ValueError, RuntimeError)):
        fit_policy(train)


@pytest.mark.parametrize("change", ["weights", "drop", "threshold"])
def test_rejects_unsupported_or_changed_settings(train, change):
    if change == "weights":
        train.weights = np.ones(len(train.data.df))
    elif change == "drop":
        train.overlap_policy = "drop"
    else:
        train.overlap_threshold = 0.2
    with pytest.raises(ValueError):
        fit_policy(train)


def test_rejects_fit_with_dropping():
    irm = make_irm(overlap_policy="drop", overlap_threshold=0)
    with pytest.raises(ValueError, match="overlap_policy"):
        fit_policy(irm)


def test_rejects_fit_weighting_even_if_current_weights_removed():
    irm = make_irm(weights=np.ones(120))
    irm.weights = None
    with pytest.raises(ValueError, match="unweighted"):
        fit_policy(irm)


def test_rejects_mismatched_evaluation_roles(train):
    policy = fit_policy(train)
    evaluation = make_irm("eval")
    frame = evaluation.data.df.rename(columns={"y": "other_y"})
    evaluation.fit(
        CausalData.from_df(
            frame,
            outcome="other_y",
            treatment="d",
            confounders=["x", "z"],
            user_id="client_id",
        )
    )
    with pytest.raises(ValueError, match="definitions must match"):
        policy.evaluate(evaluation)


@pytest.mark.parametrize("feature_value", [np.nan, np.inf, "bad", 1j])
def test_rejects_invalid_scoring_features(train, feature_value):
    policy = fit_policy(train)
    with pytest.raises(ValueError, match="numeric|finite"):
        policy.assign(pd.DataFrame({"id": [1], "x": [feature_value]}), user_id="id")


@pytest.mark.parametrize("ids", [[1, 1], [1, None]])
def test_rejects_invalid_scoring_and_fit_ids(train, ids):
    policy = fit_policy(train)
    with pytest.raises(ValueError, match="unique and nonmissing"):
        policy.assign(pd.DataFrame({"id": ids, "x": [0, 1]}), user_id="id")
    train.data.df["client_id"] = train.data.df.client_id.astype(object)
    train.data.df.iloc[:2, train.data.df.columns.get_loc("client_id")] = ids
    with pytest.raises(ValueError, match="unique and nonmissing"):
        fit_policy(train)


def test_missing_features_ids_and_unfitted_models(train):
    policy = fit_policy(train)
    with pytest.raises(ValueError, match="Missing policy features"):
        policy.assign(pd.DataFrame({"id": [1]}), user_id="id")
    with pytest.raises(ValueError, match="user_id column"):
        policy.assign(pd.DataFrame({"x": [1]}), user_id="id")
    with pytest.raises(ValueError, match="reserved"):
        policy.assign(pd.DataFrame({"x": [1], "action": [1]}), user_id="action")
    with pytest.raises(NotFittedError):
        UpliftPolicyTree().rules()
    with pytest.raises(NotFittedError):
        UpliftPolicyTree().fit(IRM())
    with pytest.raises(TypeError, match="IRM"):
        UpliftPolicyTree().fit(object())


@pytest.mark.parametrize("features", [[], ["d"], ["client_id"], ["x", "x"], "x"])
def test_invalid_policy_features(train, features):
    with pytest.raises(ValueError, match="policy_features"):
        UpliftPolicyTree().fit(train, policy_features=features)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_depth=-1),
        dict(max_depth=1.5),
        dict(min_samples_leaf=0),
        dict(min_samples_per_arm=True),
    ],
)
def test_invalid_tree_configuration(train, kwargs):
    with pytest.raises(ValueError, match="integer"):
        UpliftPolicyTree(**kwargs).fit(train)


@pytest.mark.parametrize("alpha", [0, 1, np.nan])
def test_invalid_alpha(train, alpha):
    with pytest.raises(ValueError, match="alpha"):
        fit_policy(train).evaluate(train, alpha=alpha)


def test_reports_are_defensive_copies(train):
    policy = fit_policy(train)
    original = policy.rules()
    table = policy.rules()
    table.loc[0, "action"] = 9
    pd.testing.assert_frame_equal(original, policy.rules())
    evaluation = policy.evaluate(make_irm("eval"))
    summary = evaluation.summary()
    summary.loc[0, "value"] = 999
    assert evaluation.summary().value.iloc[0] != 999
    evaluation.diagnostics["training"]["n_obs"] = 0
    assert policy.training_diagnostics_["n_obs"] == 120


def test_irm_metadata_refreshes_after_refit(train):
    train.data.df.client_id = "new_" + train.data.df.client_id
    train.overlap_threshold = 0.49
    train.fit()
    policy = fit_policy(train)
    assert policy.training_diagnostics_["overlap_threshold"] == 0.49
    assert policy._training_ids_[0].startswith("new_")


def test_decisions_do_not_depend_on_outcome_units(train):
    original = fit_policy(train).rules()
    train.data.df.y *= 1e-15
    train.fit()
    scaled = fit_policy(train).rules()
    assert scaled.predicates.tolist() == original.predicates.tolist()
    assert scaled.action.tolist() == original.action.tolist()


@pytest.mark.parametrize(
    "tau,comparison", [(2.0, "policy_vs_all"), (-2.0, "policy_vs_none")]
)
def test_constant_policy_has_exact_zero_gain_against_itself(tau, comparison):
    policy = fit_policy(make_irm(tau=tau))
    summary = (
        policy.evaluate(make_irm("eval", noise=0.5)).summary().set_index("comparison")
    )
    assert (
        summary.loc[comparison, ["value", "std_error", "ci_lower", "ci_upper"]]
        .eq(0)
        .all()
    )


def test_binary_outcome_policy_and_evaluation():
    fits = []
    for prefix, seed in [("train", 17), ("eval", 23)]:
        irm = make_irm(prefix, seed=seed)
        p = 0.5 + 0.3 * np.sign(irm.data.df.x) * irm.data.df.d
        irm.data.df.y = np.random.default_rng(seed).binomial(1, p)
        irm.fit()
        fits.append(irm)
    policy = fit_policy(fits[0])
    assert policy.rules().action.tolist() == [0, 1]
    assert policy.evaluate(fits[1]).summary().value.notna().all()


def test_evaluation_rejects_data_changed_after_fit(train):
    policy = fit_policy(train)
    evaluation = make_irm("eval")
    evaluation.data.df.loc[evaluation.data.df.index[0], "client_id"] = "different"
    with pytest.raises(ValueError, match="changed after fit"):
        policy.evaluate(evaluation)
