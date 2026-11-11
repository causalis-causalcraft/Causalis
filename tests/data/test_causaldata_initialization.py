"""Initialization rejects malformed analysis data without rejecting metadata."""

import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import CausalData, IVCausalData


@pytest.fixture
def analysis_frame():
    return pd.DataFrame(
        {
            "y": [1, 4, 2, 9, 5, 12, 8, 15],
            "d": [0, 1, 0, 1, 0, 1, 0, 1],
            "x": [11, 13, 17, 19, 23, 29, 31, 37],
            "z": [1, 1, 0, 0, 1, 1, 0, 0],
            "uid": [f"u{i}" for i in range(8)],
            "group": ["a", "a", "b", "b", "c", "c", "d", "d"],
        },
        index=[9, 3, 7, 1, 5, 2, 8, 4],
    )


def make_data(frame, role):
    if role == "z":
        return IVCausalData.from_df(frame, "d", "y", "z", ["x"])
    return CausalData.from_df(frame, "d", "y", ["x"])


@pytest.mark.parametrize("role", ["y", "d", "x", "z"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_numeric_roles_rejected(analysis_frame, role, value):
    analysis_frame[role] = analysis_frame[role].astype(float)
    analysis_frame.loc[9, role] = value

    with pytest.raises(ValueError, match="finite"):
        make_data(analysis_frame, role)


@pytest.mark.parametrize("role", ["y", "d", "x", "z"])
@pytest.mark.parametrize("imaginary", [0.0, 2.0])
def test_complex_numeric_roles_rejected_before_cast(analysis_frame, role, imaginary):
    analysis_frame[role] = analysis_frame[role].to_numpy(dtype=complex) + imaginary * 1j

    with pytest.raises(ValueError, match="real|complex"):
        make_data(analysis_frame, role)


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_nullable_real_numeric_roles_supported(analysis_frame, dtype):
    for name in ["y", "d", "x", "z"]:
        analysis_frame[name] = analysis_frame[name].astype(dtype)

    data = IVCausalData.from_df(analysis_frame, "d", "y", "z", ["x"])

    assert data.df["d"].dtype == np.dtype("int8")
    for name in ["y", "x", "z"]:
        pd.testing.assert_series_equal(data.df[name], analysis_frame[name])
    pd.testing.assert_index_equal(data.df.index, analysis_frame.index)


@pytest.mark.parametrize("dtype", ["float32", "Float32", "UInt64", pd.SparseDtype("float64", 0.0)])
def test_other_real_numeric_dtypes_remain_supported(analysis_frame, dtype):
    analysis_frame["x"] = analysis_frame["x"].astype(dtype)

    data = CausalData.from_df(analysis_frame, "d", "y", ["x"])

    pd.testing.assert_series_equal(data.df["x"], analysis_frame["x"])


def test_nullable_boolean_numeric_roles_canonicalized(analysis_frame):
    values = {
        "y": [False, False, True, True, False, True, False, True],
        "d": analysis_frame["d"].astype(bool),
        "x": [False, True, True, False, True, False, False, True],
        "z": analysis_frame["z"].astype(bool),
    }
    for name, column in values.items():
        analysis_frame[name] = pd.array(column, dtype="boolean")

    data = IVCausalData.from_df(analysis_frame, "d", "y", "z", ["x"])

    for name in ["y", "d", "x", "z"]:
        pd.testing.assert_series_equal(data.df[name], analysis_frame[name].astype("int8"))


@pytest.mark.parametrize("role", ["y", "d", "x", "z"])
@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_nullable_numeric_roles_still_reject_missing(analysis_frame, role, dtype):
    values = analysis_frame[role]
    if dtype == "boolean":
        values = values % 2 == 0
    analysis_frame[role] = values.astype(dtype)
    analysis_frame.loc[9, role] = pd.NA

    with pytest.raises(ValueError, match="NaN|missing"):
        make_data(analysis_frame, role)


@pytest.mark.parametrize("constructor", [CausalData, IVCausalData])
def test_empty_data_rejected_explicitly(analysis_frame, constructor):
    options = {"instruments": "z"} if constructor is IVCausalData else {}

    with pytest.raises(ValueError, match="empty|at least one (row|observation)"):
        constructor(
            df=analysis_frame.iloc[:0], treatment="d", outcome="y",
            confounders=["x"], **options,
        )


@pytest.mark.parametrize(
    "role,column",
    [("outcome", "y"), ("treatment", "d"), ("confounders", "x"),
     ("instruments", "z"), ("user_id", "uid"), ("cluster", "group")],
)
@pytest.mark.parametrize("name", ["", " ", "\t\n"])
def test_blank_column_names_rejected(analysis_frame, role, column, name):
    frame = analysis_frame.rename(columns={column: name})
    options = {
        "outcome": "y", "treatment": "d", "confounders": ["x"],
        "instruments": "z", "user_id": "uid", "cluster": "group",
    }
    options[role] = [name] if role == "confounders" else name

    with pytest.raises(ValueError, match="non-empty|blank|whitespace"):
        IVCausalData(df=frame, **options)


def test_nonblank_column_names_preserved_exactly(analysis_frame):
    names = {name: f" {name} " for name in analysis_frame.columns}
    frame = analysis_frame.rename(columns=names)

    data = IVCausalData.from_df(
        frame, treatment=names["d"], outcome=names["y"],
        instruments=names["z"], confounders=[names["x"]],
        user_id=names["uid"], cluster=names["group"],
    )

    assert data.outcome_name == names["y"]
    assert data.treatment_name == names["d"]
    assert data.confounders_names == [names["x"]]
    assert data.instruments_names == [names["z"]]
    assert data.user_id_name == names["uid"]
    assert data.cluster_name == names["group"]
    assert set(data.df.columns) == set(names.values())


def test_invalid_unused_columns_ignored(analysis_frame):
    analysis_frame["unused_inf"] = np.inf
    analysis_frame["unused_null"] = pd.NA
    analysis_frame["unused_complex"] = 1 + 2j
    analysis_frame["unused_unhashable"] = [[i] for i in range(len(analysis_frame))]

    data = CausalData.from_df(analysis_frame, "d", "y", ["x"])

    assert list(data.df.columns) == ["y", "d", "x"]


def test_numeric_identifier_labels_are_not_treated_as_features(analysis_frame):
    analysis_frame["uid"] = [np.inf, -np.inf, 1, 2, 3, 4, 5, 6]
    analysis_frame["group"] = [np.inf, np.inf, -np.inf, -np.inf, 1, 1, 2, 2]

    data = CausalData.from_df(
        analysis_frame, "d", "y", ["x"], user_id="uid", cluster="group",
    )

    pd.testing.assert_series_equal(data.user_id, analysis_frame["uid"])
    pd.testing.assert_series_equal(data.cluster, analysis_frame["group"])


@pytest.mark.parametrize("confounders", [None, []])
def test_confounders_remain_optional(analysis_frame, confounders):
    data = CausalData.from_df(analysis_frame, "d", "y", confounders)

    assert data.confounders_names == []
    assert data.X.shape == (len(analysis_frame), 0)


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="platform has no floating type with range wider than float64",
)
@pytest.mark.parametrize("role", ["y", "x"])
def test_finite_values_that_overflow_float64_rejected(analysis_frame, role):
    values = analysis_frame[role].to_numpy(dtype=np.longdouble)
    values[0] = np.longdouble(np.finfo(np.float64).max) * 2
    assert np.isfinite(values).all()
    analysis_frame[role] = values

    with pytest.raises(ValueError, match="float64|finite"):
        make_data(analysis_frame, role)
