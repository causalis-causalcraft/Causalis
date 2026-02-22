import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts import PanelDataSCM
from causalis.scenarios.synthetic_control import ASCM, RSCM, SCM, SyntheticControl


def _make_panel_with_effect(effect: float = 2.5) -> pd.DataFrame:
    rows = []
    for t in [1, 2, 3, 4, 5, 6]:
        y_c1 = 10.0 + 0.5 * t
        y_c2 = 12.0 + 0.2 * t
        y_treat = 0.65 * y_c1 + 0.35 * y_c2
        if t >= 4:
            y_treat += effect

        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": y_treat},
                {"unit_id": "C1", "time_id": t, "y": y_c1},
                {"unit_id": "C2", "time_id": t, "y": y_c2},
            ]
        )
    return pd.DataFrame(rows)


def test_scm_defaults_to_ascm_when_fully_observed():
    df = _make_panel_with_effect(effect=2.0)
    data = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=4)

    estimate_auto = SyntheticControl(lambda_aug=0.5).fit(data).estimate()
    estimate_ascm = ASCM(lambda_aug=0.5).fit(data).estimate()

    assert estimate_auto.estimand == "ATTE"
    assert estimate_auto.model == "AugmentedSyntheticControl"
    assert abs(float(estimate_auto.att) - float(estimate_ascm.att)) < 1e-9
    assert abs(float(estimate_auto.att_sc) - float(estimate_ascm.att_sc)) < 1e-9
    assert estimate_auto.diagnostics["selected_model"] == "AugmentedSyntheticControl"


def test_scm_forces_rscm_when_missing_outcomes_present():
    df = _make_panel_with_effect(effect=3.0)
    df = df[~((df["unit_id"] == "C2") & (df["time_id"] == 2))].copy()
    df.loc[(df["unit_id"] == "C1") & (df["time_id"] == 3), "y"] = np.nan
    data = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=df, treated_unit="T", intervention_time=4, allow_missing_outcome=True)

    estimate_auto = SyntheticControl(lambda_aug=0.5, completion_max_iter=250).fit(data).estimate()
    estimate_rscm = RSCM(lambda_aug=0.5, completion_max_iter=250).fit(data).estimate()

    assert estimate_auto.model == "RobustSyntheticControl"
    assert abs(float(estimate_auto.att) - float(estimate_rscm.att)) < 1e-9
    assert abs(float(estimate_auto.att_sc) - float(estimate_rscm.att_sc)) < 1e-9
    assert estimate_auto.diagnostics["selected_model"] == "RobustSyntheticControl"


def test_scm_alias_and_not_fitted_guard():
    model = SCM()
    with pytest.raises(RuntimeError, match="fit"):
        model.estimate()


def test_scm_contract_rejects_missing_treated_post_outcomes():
    df = _make_panel_with_effect(effect=2.0)
    df.loc[(df["unit_id"] == "T") & (df["time_id"] == 6), "y"] = np.nan
    with pytest.raises(ValueError, match="treated_unit must have observed outcomes"):
        PanelDataSCM(
            unit_id="unit_id",
            time_id="time_id",
            y="y",
            df=df,
            treated_unit="T",
            intervention_time=4,
            allow_missing_outcome=True,
        )


def test_scm_failed_refit_does_not_return_stale_estimate():
    valid_df = _make_panel_with_effect(effect=2.0)
    valid_data = PanelDataSCM(unit_id="unit_id", time_id="time_id", y="y", df=valid_df, treated_unit="T", intervention_time=4)

    model = SyntheticControl(lambda_aug=0.5).fit(valid_data)
    assert model.estimate().model == "AugmentedSyntheticControl"

    invalid_df = _make_panel_with_effect(effect=2.0)
    invalid_data = PanelDataSCM(
        unit_id="unit_id",
        time_id="time_id",
        y="y",
        df=invalid_df,
        treated_unit="T",
        intervention_time=10,
    )

    with pytest.raises(ValueError, match="post-treatment"):
        model.fit(invalid_data)

    with pytest.raises(RuntimeError, match="fit"):
        model.estimate()
