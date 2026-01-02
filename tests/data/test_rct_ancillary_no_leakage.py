import numpy as np


from causalis.data.dgps import generate_rct


def test_rct_ancillary_columns_do_not_depend_on_realized_y_when_seed_fixed():
    """Regression test: ancillary features must not be generated from the realized outcome.

    With fixed random_state, changing only the outcome parameters should change y (and g0/g1)
    but must not change ancillary columns.
    """

    common_kwargs = dict(
        n=2000,
        split=0.5,
        random_state=123,
        outcome_type="normal",
        k=3,
        add_ancillary=True,
        deterministic_ids=True,
    )

    df_a = generate_rct(
        **common_kwargs,
        outcome_params={"mean": {"A": 0.0, "B": 0.2}, "std": 1.0},
    )
    df_b = generate_rct(
        **common_kwargs,
        outcome_params={"mean": {"A": 0.0, "B": 1.2}, "std": 1.0},
    )

    # Sanity: y must change when only outcome parameters change
    assert float(np.mean(np.abs(df_a["y"].to_numpy() - df_b["y"].to_numpy()))) > 1e-3

    ancillary_cols = [
        "user_id",
        "age",
        "cnt_trans",
        "platform_Android",
        "platform_iOS",
        "invited_friend",
    ]

    for c in ancillary_cols:
        assert df_a[c].equals(df_b[c]), f"Ancillary column '{c}' should be invariant to outcome params under fixed seed."


def test_rct_ancillary_columns_do_not_depend_on_treatment_split_when_seed_fixed():
    """Ancillary columns should be baseline-only: invariant to treatment-rate (split).

    With fixed random_state, changing split should change treatment assignment,
    but must not change baseline ancillary columns.
    """

    common_kwargs = dict(
        n=2000,
        random_state=321,
        outcome_type="binary",
        k=4,
        add_ancillary=True,
        deterministic_ids=True,
    )

    df_a = generate_rct(**common_kwargs, split=0.3)
    df_b = generate_rct(**common_kwargs, split=0.8)

    # Sanity: treatment should change when split changes
    assert float(np.mean(np.abs(df_a["d"].to_numpy() - df_b["d"].to_numpy()))) > 1e-3

    ancillary_cols = [
        "user_id",
        "age",
        "cnt_trans",
        "platform_Android",
        "platform_iOS",
        "invited_friend",
    ]

    for c in ancillary_cols:
        assert df_a[c].equals(df_b[c]), f"Ancillary column '{c}' should be invariant to split under fixed seed."
