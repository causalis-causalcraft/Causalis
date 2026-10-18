"""Honest enrichment: separate users, no holdout-label leakage, and RCT reuse."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from causalis.data_contracts import CausalData, RctCausalData
from causalis.scenarios.classic_rct import DiffInMeans
from causalis.shared import check_srm
from causalis.shared.rct_design import EnrichmentDiagnostics, PrognosticEnrichment


def contract(frame):
    return CausalData.from_df(
        frame,
        treatment="treated",
        outcome="utilized",
        confounders=["activity", "spend"],
        user_id="customer_id",
    )


@pytest.fixture
def frame():
    rng = np.random.default_rng(462)
    n = 1600
    activity, spend = rng.normal(size=(2, n))
    treated = rng.binomial(1, 0.5, n)
    return pd.DataFrame(
        {
            "customer_id": [f"customer-{i}" for i in range(n)],
            "treated": treated,
            "activity": activity,
            "spend": spend,
            "utilized": rng.binomial(
                1, expit(-1.5 + activity + 0.4 * spend + 0.2 * treated)
            ),
        },
        index=np.arange(n) // 2,
    )  # IDs, not possibly duplicate row indices, define users.


def fit(frame, **options):
    return PrognosticEnrichment(
        LogisticRegression(max_iter=500),
        **options,
    ).fit(contract(frame), features=["activity", "spend"])


@pytest.mark.parametrize("source,arm", [("treatment", 1), ("control", 0)])
def test_fit_uses_only_requested_training_arm_and_selects_only_holdout(
    frame, source, arm
):
    original = frame.copy(deep=True)
    estimator = LogisticRegression(max_iter=500)
    rule = PrognosticEnrichment(estimator, training_source=source).fit(
        contract(frame),
        features=["activity", "spend"],
    )
    partition = rule.get_partition()
    train = partition["sample"].to_numpy() == "train"
    trained = train & (frame.treated.to_numpy() == arm)
    expected_model = LogisticRegression(max_iter=500).fit(
        frame.iloc[np.flatnonzero(trained)][["activity", "spend"]],
        frame.iloc[np.flatnonzero(trained)].utilized,
    )
    np.testing.assert_allclose(rule.model_.coef_, expected_model.coef_)
    np.testing.assert_allclose(rule.model_.intercept_, expected_model.intercept_)
    assert not hasattr(estimator, "classes_")
    assert rule.model_ is not estimator
    assert rule.threshold_ == pytest.approx(
        np.quantile(partition.score.to_numpy()[train], 0.8)
    )
    assert not partition.loc[train, "selected"].any()
    expected_selection = ~train & (partition.score.to_numpy() >= rule.threshold_)
    np.testing.assert_array_equal(partition.selected, expected_selection)
    selected = rule.select()
    assert set(selected.user_id).isdisjoint(partition.loc[train, "user_id"])
    expected_frame = contract(frame).df.iloc[np.flatnonzero(expected_selection)]
    pd.testing.assert_frame_equal(selected.df, expected_frame)
    pd.testing.assert_frame_equal(frame, original)
    result = DiffInMeans().fit(selected).estimate(method="conversion_ztest")
    assert result.value == pytest.approx(
        selected.df.loc[selected.df.treated == 1, "utilized"].mean()
        - selected.df.loc[selected.df.treated == 0, "utilized"].mean()
    )


def test_holdout_outcomes_and_unused_training_arm_do_not_change_model_or_selection(
    frame,
):
    rule = fit(frame)
    partition = rule.get_partition()
    train = partition["sample"].to_numpy() == "train"
    # Only treatment training outcomes are allowed to affect the model/cutoff.
    changed = frame.copy(deep=True)
    unused = ~train | (frame.treated.to_numpy() == 0)
    changed.iloc[np.flatnonzero(unused), changed.columns.get_loc("utilized")] = (
        1 - changed.iloc[np.flatnonzero(unused)].utilized.to_numpy()
    )
    refit = fit(changed)
    np.testing.assert_allclose(rule.model_.coef_, refit.model_.coef_)
    assert rule.threshold_ == refit.threshold_
    pd.testing.assert_frame_equal(partition, refit.get_partition())


def test_threshold_does_not_use_evaluation_features_or_arm_labels(frame):
    rule = fit(frame)
    changed = frame.copy(deep=True)
    evaluation = rule.get_partition()["sample"].to_numpy() == "evaluation"
    positions = np.flatnonzero(evaluation)
    changed.iloc[positions, changed.columns.get_indexer(["activity", "spend"])] += 10
    changed.iloc[positions, changed.columns.get_loc("treated")] = (
        1 - changed.iloc[positions].treated.to_numpy()
    )
    refit = fit(changed)
    assert rule.threshold_ == refit.threshold_
    np.testing.assert_allclose(rule.model_.coef_, refit.model_.coef_)
    np.testing.assert_array_equal(
        rule.get_partition()["sample"], refit.get_partition()["sample"]
    )


def test_split_is_stable_under_row_reordering_and_returned_frames_are_copies(frame):
    rule = fit(frame)
    original = rule.get_partition()
    reordered = fit(frame.sample(frac=1, random_state=24)).get_partition()
    pd.testing.assert_frame_equal(
        original.set_index("user_id").sort_index(),
        reordered.set_index("user_id").sort_index(),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    partition = rule.get_partition()
    partition["selected"] = True
    selected = rule.get_selected_df()
    selected["utilized"] = 0
    pd.testing.assert_frame_equal(rule.get_partition(), original)
    assert rule.get_selected_df().utilized.sum() > 0


def test_diagnostics_use_planned_allocation_and_include_score(frame):
    rule = fit(frame)
    report = rule.diagnostics(target_allocation={0: 0.4, 1: 0.6})
    assert isinstance(report, EnrichmentDiagnostics)
    evaluation = rule.get_partition()["sample"].to_numpy() == "evaluation"
    expected = check_srm(frame.treated.to_numpy()[evaluation], {0: 0.4, 1: 0.6})
    assert report.srm_evaluation.chi2 == expected.chi2
    assert report.threshold == rule.threshold_
    assert report.selected_fraction == pytest.approx(
        rule.get_partition().selected.sum() / evaluation.sum()
    )
    assert set(report.balance.confounders) == {"activity", "spend", "prognostic_score"}
    for sample, mask in (
        ("train", ~evaluation),
        ("evaluation", evaluation),
        ("selected", rule.get_partition().selected.to_numpy()),
    ):
        for arm in (0, 1):
            row = report.counts.loc[
                (report.counts["sample"] == sample) & (report.counts.treatment == arm)
            ].iloc[0]
            source = frame.iloc[
                np.flatnonzero(mask & (frame.treated.to_numpy() == arm))
            ]
            assert row.n_users == len(source)
            assert row.n_events == source.utilized.sum()


class ReversedClassesClassifier(BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, X):
        positive = expit(X.iloc[:, 0].to_numpy())
        return np.column_stack([positive, 1 - positive])


def test_positive_class_is_resolved_by_label_and_ties_use_common_threshold(frame):
    rule = PrognosticEnrichment(ReversedClassesClassifier()).fit(
        contract(frame), features=["activity"]
    )
    np.testing.assert_allclose(rule.get_partition().score, expit(frame.activity))
    tied = PrognosticEnrichment(DummyClassifier(strategy="prior")).fit(
        contract(frame),
        features=["activity"],
    )
    partition = tied.get_partition()
    np.testing.assert_array_equal(
        partition.selected, partition["sample"] == "evaluation"
    )
    assert tied.diagnostics(target_allocation={0: 0.5, 1: 0.5}).selected_fraction == 1


def test_explicit_threshold_and_empty_selection_remain_inspectable(frame):
    rule = fit(frame, top_fraction=None, threshold=1)
    assert not rule.get_partition().selected.any()
    report = rule.diagnostics(target_allocation={0: 0.5, 1: 0.5})
    assert report.srm_selected is None
    assert report.selected_fraction == 0
    assert report.balance.empty
    assert (
        report.counts.loc[report.counts["sample"] == "selected", "n_users"].sum() == 0
    )
    with pytest.raises(ValueError, match="both treatment arms"):
        rule.select()


def test_constant_selected_covariate_is_kept_in_diagnostics_and_omitted_from_contract(
    frame,
):
    changed = frame.copy(deep=True)
    changed["activity"] = (changed.activity > 0).astype(int)
    rule = PrognosticEnrichment(
        ReversedClassesClassifier(),
        top_fraction=None,
        threshold=0.6,
    ).fit(contract(changed), features=["activity"])
    assert rule.get_selected_df().activity.eq(1).all()
    assert rule.select().confounders_names == ["spend"]
    report = rule.diagnostics(target_allocation={0: 0.5, 1: 0.5})
    assert "activity" in report.balance.confounders.to_list()


class InvalidProbabilityClassifier(ReversedClassesClassifier):
    def predict_proba(self, X):
        return np.full((len(X), 2), 0.8)


def test_invalid_probability_predictions_are_rejected(frame):
    with pytest.raises(ValueError, match="summing to 1"):
        PrognosticEnrichment(InvalidProbabilityClassifier()).fit(
            contract(frame), features=["activity"]
        )


def test_user_id_names_cannot_collide_with_partition_columns(frame):
    changed = frame.rename(columns={"customer_id": "sample"})
    data = CausalData.from_df(
        changed, "treated", "utilized", ["activity", "spend"], "sample"
    )
    rule = PrognosticEnrichment(LogisticRegression()).fit(data, features=["activity"])
    assert rule.get_partition().user_id.tolist() == changed["sample"].tolist()
    assert set(rule.get_partition()["sample"]) == {"train", "evaluation"}


def test_zero_evaluation_events_do_not_change_selection_and_report_sparse_events(frame):
    rule = fit(frame)
    evaluation = rule.get_partition()["sample"].to_numpy() == "evaluation"
    changed = frame.copy(deep=True)
    changed.iloc[np.flatnonzero(evaluation), changed.columns.get_loc("utilized")] = 0
    refit = fit(changed)
    pd.testing.assert_frame_equal(rule.get_partition(), refit.get_partition())
    assert any(
        "few events" in warning
        for warning in refit.diagnostics(target_allocation={0: 0.5, 1: 0.5}).warnings
    )
    with pytest.raises(ValueError, match="outcome is constant"):
        refit.select()


def test_single_arm_selected_is_reported_not_repaired(frame):
    rule = fit(frame, top_fraction=None, threshold=0)
    changed = frame.copy(deep=True)
    evaluation = rule.get_partition()["sample"].to_numpy() == "evaluation"
    changed.iloc[np.flatnonzero(evaluation), changed.columns.get_loc("treated")] = 1
    refit = fit(changed, top_fraction=None, threshold=0)
    report = refit.diagnostics(target_allocation={0: 0.5, 1: 0.5})
    assert report.srm_selected.is_srm
    assert report.balance.empty
    with pytest.raises(ValueError, match="both treatment arms"):
        refit.select()


def test_rct_contract_requires_outcome_and_binary_treatment(frame):
    changed = frame.copy(deep=True)
    changed["second_outcome"] = 1 - changed.utilized
    data = RctCausalData.from_df(
        changed,
        "treated",
        ["utilized", "second_outcome"],
        ["activity", "spend"],
        "customer_id",
    )
    rule = PrognosticEnrichment(LogisticRegression())
    with pytest.raises(ValueError, match="Select one outcome"):
        rule.fit(data, features=["activity"])
    rule.fit(data, features=["activity"], outcome="utilized")
    assert rule.select().outcome_name == "utilized"
    changed["control"] = 1 - changed.treated
    multi = RctCausalData.from_df(
        changed,
        ["control", "treated"],
        ["utilized"],
        ["activity", "spend"],
        "customer_id",
        control_treatment="control",
    )
    with pytest.raises(ValueError, match="single binary treatment"):
        rule.fit(multi, features=["activity"])
    with pytest.raises(RuntimeError, match="Call fit"):
        rule.select()  # A failed refit cannot return stale selected users.


@pytest.mark.parametrize(
    "options",
    [
        {"train_fraction": 0},
        {"train_fraction": 1},
        {"train_fraction": np.nan},
        {"top_fraction": 0},
        {"top_fraction": True},
        {"top_fraction": None},
        {"threshold": 0.4},
        {"top_fraction": None, "threshold": 2},
        {"split_seed": True},
        {"split_seed": 1.5},
        {"training_source": "both"},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        PrognosticEnrichment(LogisticRegression(), **options)


@pytest.mark.parametrize(
    "features",
    [
        [],
        "activity",
        ["treated"],
        ["utilized"],
        ["customer_id"],
        ["activity", "activity"],
    ],
)
def test_features_must_explicitly_select_baseline_columns(frame, features):
    with pytest.raises(ValueError, match="features"):
        PrognosticEnrichment(LogisticRegression()).fit(
            contract(frame), features=features
        )


def test_requires_ids_binary_outcome_and_training_events(frame):
    no_id = CausalData.from_df(frame, "treated", "utilized", ["activity"])
    with pytest.raises(ValueError, match="requires user_id"):
        PrognosticEnrichment(LogisticRegression()).fit(no_id, features=["activity"])
    changed = frame.copy(deep=True)
    changed["utilized"] = changed.utilized + 1
    with pytest.raises(ValueError, match="binary utilization outcome"):
        fit(changed)
    changed = frame.copy(deep=True)
    changed.loc[changed.treated == 1, "utilized"] = 0
    with pytest.raises(ValueError, match="both binary outcome classes"):
        fit(changed)


def test_methods_require_fit_and_allocation_is_explicit(frame):
    rule = PrognosticEnrichment(LogisticRegression())
    for method in (rule.select, rule.get_partition, rule.get_selected_df):
        with pytest.raises(RuntimeError, match="Call fit"):
            method()
    fitted = fit(frame)
    with pytest.raises(ValueError, match="original probabilities"):
        fitted.diagnostics(target_allocation={"control": 0.5, "treatment": 0.5})
