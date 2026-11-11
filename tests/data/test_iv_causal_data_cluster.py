import pandas as pd
import pytest

from causalis.data_contracts import IVCausalData


@pytest.fixture
def iv_frame():
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "d": [0, 1, 0, 1],
            "z": [1, 1, 0, 0],
            "x": [0.1, 0.2, 0.3, 0.4],
            "group": pd.Categorical(["a", "a", "b", "b"]),
        },
        index=[4, 1, 9, 6],
    )


@pytest.mark.parametrize("constructor", [IVCausalData, IVCausalData.from_df])
def test_iv_cluster_metadata_preserved(iv_frame, constructor):
    data = constructor(
        df=iv_frame, treatment="d", outcome="y", instruments="z",
        confounders="x", cluster="group",
    )

    assert data.cluster_name == "group"
    pd.testing.assert_series_equal(data.cluster, iv_frame["group"])
    assert "group" in data.df.columns
    assert list(data.get_df().columns) == ["y", "x", "d", "z"]
    assert list(data.X.columns) == ["x"]
    assert list(data.Z.columns) == ["z"]
    assert "cluster='group'" in repr(data)


def test_iv_cluster_role_disjoint_from_instrument(iv_frame):
    with pytest.raises(ValueError, match="cannot be both"):
        IVCausalData.from_df(
            iv_frame, treatment="d", outcome="y", instruments="z", cluster="z",
        )


def test_iv_get_df_cluster_only(iv_frame):
    data = IVCausalData.from_df(
        iv_frame, treatment="d", outcome="y", instruments="z", cluster="group",
    )
    selected = data.get_df(
        include_treatment=False, include_outcome=False,
        include_confounders=False, include_instruments=False,
        include_cluster=True,
    )

    pd.testing.assert_frame_equal(selected, iv_frame[["group"]])
    selected.iloc[0, 0] = "b"
    assert data.cluster.iloc[0] == "a"


def test_iv_get_df_absent_cluster_returns_empty(iv_frame):
    data = IVCausalData.from_df(iv_frame, treatment="d", outcome="y", instruments="z")

    selected = data.get_df(
        include_treatment=False, include_outcome=False,
        include_confounders=False, include_instruments=False,
        include_cluster=True,
    )

    assert selected.shape == (4, 0)
    pd.testing.assert_index_equal(selected.index, iv_frame.index)
    assert "cluster=" not in repr(data)
