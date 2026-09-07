import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from causalis.data_contracts import CausalData, RctCausalData


@pytest.fixture
def frame():
    return pd.DataFrame({
        "uid": ["a", "b", "c", "d"], "d": [False, True, False, True],
        "y1": [2., 4., 8., 3.], "y2": [True, False, False, True],
        "x": [10, 20, 15, 30], "unused": [None] * 4,
    }, index=[8, 3, 3, 1])


def make(frame, **kwargs):
    return RctCausalData.from_df(
        frame, "d", ["y1", "y2"], **kwargs,
    )


def test_construction_access_and_isolation(frame):
    original = frame.copy(deep=True)
    data = make(frame, confounders=["x", "x"], user_id="uid")
    assert list(data.df) == ["uid", "y1", "y2", "d", "x"]
    assert data.confounders == ["x"]
    assert data.treatment.dtype == np.int8
    assert data.df.y2.dtype == np.int8
    assert_frame_equal(data.Y, data.outcomes)
    assert_frame_equal(data.X, frame[["x"]])
    assert_series_equal(data.user_id, frame.uid)
    assert_frame_equal(frame, original)
    for result in (data.Y, data.X, data.get_df()):
        result.iloc[0, 0] = -999
    assert data.df.y1.iloc[0] == 2
    assert data.df.x.iloc[0] == 10
    frame.iloc[0, frame.columns.get_loc("y1")] = 999
    assert data.df.y1.iloc[0] == 2
    data.df.iloc[1, data.df.columns.get_loc("x")] = -1
    assert frame.x.iloc[1] == 20


def test_names_aliases_single_outcome_and_empty_features(frame):
    data = RctCausalData(
        df=frame, treatment_name="d", outcome_names="y1", confounders_names=None,
    )
    assert data.Y.shape == (4, 1)
    assert data.X.shape == (4, 0)
    assert data.X.index.equals(frame.index)
    assert data.user_id.empty
    repeated = RctCausalData.from_df(frame, "d", ["y2", "y1", "y2"])
    assert repeated.outcome_names == ["y2", "y1"]
    assert "df=(4, 3)" in repr(repeated)


@pytest.mark.parametrize("outcomes", [[], None, 12, [1], [""], [" "]])
def test_bad_outcome_names(frame, outcomes):
    with pytest.raises(ValueError):
        RctCausalData.from_df(frame, "d", outcomes)


@pytest.mark.parametrize("column,values,message", [
    ("y2", [1, 1, 1, 1], "constant"),
    ("y2", [1, 2, np.nan, 3], "NaN"),
    ("y2", [1, 2, np.inf, 3], "finite"),
    ("y2", ["a", "b", "c", "d"], "real int"),
    ("y2", [1j, 2j, 3j, 4j], "real int"),
    ("d", [0, 1, 2, 1], "binary"),
    ("d", [0, 0, 0, 0], "constant"),
    ("x", [0, 0, 0, 0], "constant"),
    ("uid", ["a", "a", "b", "c"], "duplicate values"),
    ("uid", ["a", None, "b", "c"], "NaN"),
])
def test_invalid_values(frame, column, values, message):
    frame[column] = values
    with pytest.raises(ValueError, match=message):
        make(frame, confounders="x", user_id="uid")


@pytest.mark.parametrize("kwargs", [
    {"confounders": "y2"}, {"user_id": "d"}, {"user_id": "y1"},
])
def test_roles_disjoint(frame, kwargs):
    with pytest.raises(ValueError, match="cannot be both"):
        make(frame, **kwargs)


def test_missing_duplicate_columns_empty_and_extra(frame):
    with pytest.raises(ValueError, match="does not exist"):
        make(frame.drop(columns="y2"))
    with pytest.raises(ValueError, match="duplicate column names"):
        make(pd.concat([frame, frame[["x"]]], axis=1))
    with pytest.raises(ValueError, match="constant"):
        make(frame.iloc[:0])
    with pytest.raises(ValueError, match="Extra inputs"):
        make(frame, surprise=True)


def test_duplicate_values_across_roles_and_dtypes(frame):
    for name, source in [("y2", "y1"), ("x", "y1"), ("y2", "d")]:
        duplicate = frame.copy()
        duplicate[name] = duplicate[source].astype(float)
        with pytest.raises(ValueError, match="identical values"):
            make(duplicate, confounders="x")


def test_sampling_does_not_determine_equality():
    n = 1000
    frame = pd.DataFrame({"d": np.arange(n) % 2, "y1": np.arange(n) + 10})
    frame["y2"] = frame.y1.copy()
    frame.loc[1, "y2"] = -5  # outside the evenly spaced sample
    assert make(frame).df.shape == (n, 3)
    frame.loc[1, "y2"] = frame.loc[1, "y1"]
    with pytest.raises(ValueError, match="identical values"):
        make(frame)


def test_fingerprint_collisions_are_checked_exactly(monkeypatch, frame):
    monkeypatch.setattr(CausalData, "_column_value_signature", staticmethod(lambda s: ("all", len(s), "same")))
    assert make(frame).Y.shape == (4, 2)
    frame["y2"] = frame.y1
    with pytest.raises(ValueError, match="identical values"):
        make(frame)


def test_large_integer_precision():
    values = np.array([2**60, 2**60 + 2, 2**60 + 4], dtype=np.int64)
    frame = pd.DataFrame({"d": [0, 1, 0], "y1": values, "y2": values + 1})
    data = make(frame)
    assert_series_equal(data.df.y2, frame.y2)


def test_nullable_dtypes(frame):
    frame["y1"] = frame.y1.astype("Float64")
    frame["x"] = frame.x.astype("Int64")
    frame["d"] = frame.d.astype("boolean")
    data = make(frame, confounders="x")
    assert str(data.df.y1.dtype) == "Float64"
    assert str(data.X.x.dtype) == "Int64"
    frame.loc[8, "y1"] = pd.NA
    with pytest.raises(ValueError, match="NaN"):
        make(frame)


def test_selection_and_single_outcome_conversion(frame):
    data = make(frame, confounders="x", user_id="uid")
    assert list(data.get_df()) == ["y1", "y2", "x", "d"]
    flags = dict(include_outcomes=False, include_treatment=False, include_confounders=False)
    assert data.get_df(**flags).shape == (4, 0)
    assert list(data.get_df(["y2", "y2"], **flags)) == ["y2"]
    assert list(data.get_df(include_user_id=True))[-1] == "uid"
    with pytest.raises(ValueError, match="do not exist"):
        data.get_df(["missing"])
    single = data.for_outcome("y1")
    assert isinstance(single, CausalData)
    assert_series_equal(single.outcome, data.df.y1)
    assert_frame_equal(single.X, data.X)
    single.df.iloc[0, single.df.columns.get_loc("y1")] = -9
    assert data.df.y1.iloc[0] == 2
    with pytest.raises(ValueError, match="not a declared outcome"):
        data.for_outcome("x")


def test_wide_dataset_validates_without_single_outcome_construction(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("must not construct a CausalData per outcome")
    monkeypatch.setattr(CausalData, "from_df", unexpected)
    rng = np.random.default_rng(3)
    names = [f"y{i}" for i in range(200)]
    frame = pd.DataFrame(rng.normal(size=(1000, 200)), columns=names)
    frame["d"] = np.arange(1000) % 2
    data = RctCausalData.from_df(frame, "d", names)
    assert data.Y.shape == (1000, 200)


@pytest.fixture
def multi_frame():
    return pd.DataFrame({
        "control": [1, 0, 0, 1, 0, 0],
        "a": [0, 1, 0, 0, 1, 0],
        "b": [0, 0, 1, 0, 0, 1],
        "y1": [3., 7., 9., 2., 8., 1.],
        "y2": [2., 3., 8., 4., 1., 9.],
        "x": [10, 30, 20, 15, 50, 60],
        "uid": list("abcdef"),
    }, index=[9, 2, 2, 6, 4, 1])


def make_multi(frame, **kwargs):
    options = dict(treatment_names=["a", "control", "b"], control_treatment="control")
    options.update(kwargs)
    return RctCausalData.from_df(frame, outcomes=["y1", "y2"], **options)


def test_multiple_treatments_and_conversion(multi_frame):
    from causalis.data_contracts import MultiCausalData
    data = make_multi(multi_frame, confounders="x", user_id="uid")
    assert data.treatment_names == ["control", "a", "b"]
    assert list(data.D) == data.treatment_names
    assert_frame_equal(data.D, data.treatments)
    assert_frame_equal(data.treatment, data.D)
    assert (data.D.dtypes == np.int8).all()
    assert data.D.index.equals(multi_frame.index)
    assert list(data.get_df()) == ["y1", "y2", "x", "control", "a", "b"]
    assert list(data.get_df(include_treatment=False)) == ["y1", "y2", "x"]
    matrix = data.D
    matrix.iloc[0, 0] = 0
    assert data.df.control.iloc[0] == 1
    single = data.for_outcome("y2")
    assert isinstance(single, MultiCausalData)
    assert single.treatment_names == data.treatment_names
    assert single.control_treatment == "control"
    assert_series_equal(single.df.y2, data.df.y2)
    single.df.iloc[0, single.df.columns.get_loc("y2")] = -10
    assert data.df.y2.iloc[0] == 2
    with pytest.raises(ValueError, match="Multiple treatment arms"):
        _ = data.treatment_name


@pytest.mark.parametrize("alias", ["treatment", "treatments", "treatment_names"])
def test_treatment_aliases(multi_frame, alias):
    data = RctCausalData.from_df(
        multi_frame, outcomes="y1", control_treatment="control",
        **{alias: ["a", "b", "control"]},
    )
    assert data.treatment_names == ["control", "a", "b"]


@pytest.mark.parametrize("options,message", [
    ({"control_treatment": None}, "control_treatment"),
    ({"control_treatment": "missing"}, "control_treatment"),
    ({"treatment_names": []}, "at least one"),
    ({"treatment_names": ["a", "a"]}, "duplicate names"),
    ({"treatment_names": ["control", "missing"]}, "does not exist"),
    ({"treatment_names": ["control", "y1"]}, "cannot be both"),
    ({"confounders": "a"}, "cannot be both"),
    ({"user_id": "b"}, "cannot be both"),
])
def test_invalid_multiple_treatment_metadata(multi_frame, options, message):
    with pytest.raises(ValueError, match=message):
        make_multi(multi_frame, **options)


@pytest.mark.parametrize("column,values,message", [
    ("a", [1, 1, 0, 0, 1, 0], "one-hot"),
    ("b", [0, 0, 0, 0, 0, 1], "one-hot"),
    ("b", [0, 0, 2, 0, 0, 1], "binary"),
    ("b", [0, 0, np.nan, 0, 0, 1], "NaN"),
    ("b", [0, 0, np.inf, 0, 0, 1], "finite"),
    ("b", [0, 0, 0, 0, 0, 0], "constant"),
])
def test_invalid_multiple_treatment_values(multi_frame, column, values, message):
    multi_frame[column] = values
    with pytest.raises(ValueError, match=message):
        make_multi(multi_frame)


def test_nullable_treatment_arms_and_input_isolation(multi_frame):
    multi_frame["control"] = multi_frame.control.astype("boolean")
    multi_frame["a"] = multi_frame.a.astype("Int64")
    multi_frame["b"] = multi_frame.b.astype("Float64")
    original = multi_frame.copy(deep=True)
    data = make_multi(multi_frame)
    assert_frame_equal(multi_frame, original)
    data.df.iloc[0, data.df.columns.get_loc("control")] = 0
    assert multi_frame.control.iloc[0]


def test_binary_control_and_matrix(frame):
    data = make(frame)
    assert data.treatment_name == "d"
    assert data.D.shape == (4, 1)
    with pytest.raises(ValueError, match="omit control_treatment"):
        make(frame, control_treatment="d")


def test_many_arms_use_non_overflowing_counter():
    n = 260
    names = [f"arm{i}" for i in range(n)]
    frame = pd.DataFrame(np.eye(n, dtype=np.int8), columns=names)
    frame["y"] = np.arange(n) + 5
    data = RctCausalData.from_df(frame, names, "y", control_treatment=names[0])
    assert data.D.shape == (n, n)
    frame.loc[0, names] = 1  # must not wrap the active-arm count
    with pytest.raises(ValueError, match="one-hot"):
        RctCausalData.from_df(frame, names, "y", control_treatment=names[0])
