import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from causalis.data_contracts.causaldata import CausalData


@pytest.fixture
def cluster_df():
    return pd.DataFrame(
        {
            "y": [1.2, 3.0, -0.4, 2.1, 4.0, 0.5, 2.8, 1.6],
            "d": [0, 0, 1, 1, 0, 1, 0, 1],
            "x": [8, 3, 6, 1, 5, 4, 2, 7],
            "uid": [f"u{i}" for i in range(8)],
            "group": ["c", "a", "c", "b", "a", "c", "c", "d"],
            "unused": [np.nan] * 8,
        },
        index=pd.Index([20, 10, 20, 0, 9, 8, 7, 2], name="row"),
    )


def make_data(df, **kwargs):
    return CausalData.from_df(df, "d", "y", ["x"], "uid", **kwargs)


def test_cluster_is_optional_and_skips_validation(cluster_df, monkeypatch):
    def unexpected_validation(series):
        pytest.fail("No cluster validation should run when cluster is omitted.")

    monkeypatch.setattr(CausalData, "_validate_cluster_values", staticmethod(unexpected_validation))
    omitted = make_data(cluster_df)
    explicit_none = make_data(cluster_df, cluster=None)

    assert omitted.cluster_name is None
    assert omitted.cluster.empty
    assert "group" not in omitted.df
    assert "cluster=" not in repr(omitted)
    pd.testing.assert_frame_equal(omitted.df, explicit_none.df)
    pd.testing.assert_frame_equal(omitted.get_df(), omitted.get_df(include_cluster=True))


def test_cluster_constructor_alias_and_canonical_name(cluster_df):
    aliased = make_data(cluster_df, cluster="group")
    canonical = CausalData(
        df=cluster_df,
        treatment_name="d",
        outcome_name="y",
        confounders_names=["x"],
        user_id_name="uid",
        cluster_name="group",
    )

    assert aliased.cluster_name == "group"
    assert "cluster='group'" in repr(aliased)
    pd.testing.assert_frame_equal(aliased.df, canonical.df)
    pd.testing.assert_series_equal(aliased.cluster, cluster_df["group"])
    assert aliased.df.index.equals(cluster_df.index)
    assert "unused" not in aliased.df


@pytest.mark.parametrize(
    "labels",
    [
        pd.Series([10, 20, 10, 30, 20, 10, 10, 40], dtype="Int64"),
        pd.Series(["c", "a", "c", "b", "a", "c", "c", "d"], dtype="string"),
        pd.Series(pd.Categorical(["b", "b", "a", "a", "b", "a", "b", "a"], categories=["unused", "a", "b"])),
        pd.Series([True, True, False, False, True, False, True, False], dtype=bool),
        pd.Series([True, True, False, False, True, False, True, False], dtype="boolean"),
        pd.Series([0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        pd.Series(["a", 1, "a", 1, "a", 1, "a", 1], dtype=object),
        pd.Series([("a", 1), ("b", 2)] * 4),
        pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"] * 4)),
    ],
    ids=["nullable-int", "string", "categorical", "bool", "nullable-bool", "float", "mixed", "tuple", "datetime"],
)
def test_cluster_preserves_labels_dtype_and_observation_order(cluster_df, labels):
    cluster_df["group"] = labels.array
    data = make_data(cluster_df, cluster="group")

    pd.testing.assert_series_equal(data.cluster, cluster_df["group"])
    assert data.treatment.dtype == np.dtype("int8")


@pytest.mark.parametrize("source", ["y", "d", "x", "uid"])
def test_cluster_values_can_equal_other_roles(cluster_df, source):
    cluster_df["group"] = cluster_df[source].copy()
    data = make_data(cluster_df, cluster="group")
    pd.testing.assert_series_equal(data.cluster, cluster_df["group"])


@pytest.mark.parametrize(
    "labels",
    [
        list(range(8)),  # all singleton clusters
        ["large"] * 7 + ["small"],  # unbalanced sizes
        ["control"] * 2 + ["treated"] * 2 + ["mixed"] * 4,
    ],
    ids=["singletons", "unbalanced", "constant-treatment-within-some-clusters"],
)
def test_cluster_accepts_singletons_unbalanced_sizes_and_cluster_treatment(cluster_df, labels):
    cluster_df["group"] = labels
    data = make_data(cluster_df, cluster="group")
    assert len(data.cluster) == len(cluster_df)


@pytest.mark.parametrize("name", ["", " ", "\t\n", 1, True, ["group"], ("group",), b"group"])
def test_cluster_requires_one_nonempty_string_column_name(cluster_df, name):
    with pytest.raises(ValidationError):
        make_data(cluster_df, cluster=name)


def test_cluster_column_must_exist(cluster_df):
    with pytest.raises(ValueError, match="specified as cluster does not exist"):
        make_data(cluster_df, cluster="missing")


@pytest.mark.parametrize("role", ["y", "d", "x", "uid"])
def test_cluster_column_name_must_be_disjoint_from_other_roles(cluster_df, role):
    with pytest.raises(ValueError, match="cannot be both|disjoint"):
        make_data(cluster_df, cluster=role)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT])
def test_cluster_rejects_missing_labels(cluster_df, missing):
    cluster_df.iloc[3, cluster_df.columns.get_loc("group")] = missing
    with pytest.raises(ValueError, match="NaN values in used columns"):
        make_data(cluster_df, cluster="group")


@pytest.mark.parametrize("bad_label", [["a"], {"a": 1}, {"a"}, np.array([1, 2])])
def test_cluster_rejects_unhashable_labels(cluster_df, bad_label):
    labels = cluster_df["group"].tolist()
    labels[0] = bad_label
    cluster_df["group"] = pd.Series(labels, dtype=object, index=cluster_df.index)
    with pytest.raises(ValueError, match="cluster must contain hashable identifiers"):
        make_data(cluster_df, cluster="group")


@pytest.mark.parametrize("categorical", [False, True])
def test_cluster_requires_two_observed_clusters(cluster_df, categorical):
    labels = ["only"] * len(cluster_df)
    cluster_df["group"] = pd.Categorical(labels, categories=["only", "unused"]) if categorical else labels
    with pytest.raises(ValueError, match="at least two observed clusters"):
        make_data(cluster_df, cluster="group")


def test_cluster_is_metadata_and_get_df_can_include_it(cluster_df):
    data = make_data(cluster_df, cluster="group")
    pd.testing.assert_frame_equal(data.X, cluster_df[["x"]])
    assert list(data.get_df().columns) == ["y", "x", "d"]
    assert list(data.get_df(include_cluster=True).columns) == ["y", "x", "d", "group"]
    assert list(data.get_df(columns=["group"], include_cluster=True).columns).count("group") == 1

    cluster_only = data.get_df(
        include_treatment=False, include_outcome=False, include_confounders=False, include_cluster=True
    )
    pd.testing.assert_frame_equal(cluster_only, cluster_df[["group"]])
    cluster_only.iloc[0, 0] = "changed"
    assert data.cluster.iloc[0] == "c"

    empty = data.get_df(include_treatment=False, include_outcome=False, include_confounders=False)
    assert empty.shape == (len(cluster_df), 0)
    assert empty.index.equals(cluster_df.index)


def test_cluster_storage_is_independent_of_input(cluster_df):
    original = cluster_df.copy(deep=True)
    data = make_data(cluster_df, cluster="group")
    pd.testing.assert_frame_equal(cluster_df, original)
    cluster_df.iloc[0, cluster_df.columns.get_loc("group")] = "input-changed"
    assert data.cluster.iloc[0] == "c"
    data.df.iloc[1, data.df.columns.get_loc("group")] = "stored-changed"
    assert cluster_df["group"].iloc[1] == "a"


def test_cluster_labels_are_not_fingerprinted_for_duplicate_screening(cluster_df, monkeypatch):
    original = CausalData._column_value_signatures
    checked_columns = []

    def record_columns(df, cols):
        checked_columns.extend(cols)
        return original(df, cols)

    monkeypatch.setattr(CausalData, "_column_value_signatures", staticmethod(record_columns))
    make_data(cluster_df, cluster="group")
    assert "group" not in checked_columns
    assert set(checked_columns) == {"y", "d", "x", "uid"}
