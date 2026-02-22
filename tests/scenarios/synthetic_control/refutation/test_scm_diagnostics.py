import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd

from causalis.data_contracts import PanelDataSCM
from causalis.scenarios.synthetic_control import ASCM, RSCM, run_scm_diagnostics


def _make_panel_with_effect(effect: float = 2.5) -> pd.DataFrame:
    rows = []
    for t in [1, 2, 3, 4, 5, 6]:
        y_c1 = 10.0 + 0.5 * t
        y_c2 = 12.0 + 0.2 * t
        y_c3 = 9.0 + 0.3 * t
        y_treat = 0.5 * y_c1 + 0.3 * y_c2 + 0.2 * y_c3
        if t >= 4:
            y_treat += effect

        rows.extend(
            [
                {"unit_id": "T", "time_id": t, "y": y_treat},
                {"unit_id": "C1", "time_id": t, "y": y_c1},
                {"unit_id": "C2", "time_id": t, "y": y_c2},
                {"unit_id": "C3", "time_id": t, "y": y_c3},
            ]
        )
    return pd.DataFrame(rows)


def test_run_scm_diagnostics_writes_three_plots_and_metrics(tmp_path):
    df = _make_panel_with_effect(effect=3.0)
    data = PanelDataSCM(
        unit_id="unit_id",
        time_id="time_id",
        y="y",
        df=df,
        treated_unit="T",
        intervention_time=4,
    )
    estimate = ASCM(lambda_aug=0.5).fit(data).estimate()

    out = run_scm_diagnostics(
        estimate=estimate,
        paneldata=data,
        output_dir=tmp_path,
        filename_prefix="ascm",
    )
    metrics = out["metrics"]
    plots = out["plots"]

    expected_metric_keys = {
        "n_donors",
        "n_pre",
        "n_post",
        "pre_rmse_sc",
        "pre_rmse_aug",
        "att_sc",
        "att_aug",
        "max_weight_sc",
        "max_abs_weight_aug",
        "l1_norm_weight_aug",
        "cond_augmented_gram",
        "n_placebos",
        "min_possible_p",
        "p_value_att",
        "ci_low_abs",
        "ci_high_abs",
        "placebo_ci_is_unbounded",
        "missing_cell_fraction",
        "completion_converged",
        "completion_effective_rank",
    }
    assert expected_metric_keys.issubset(metrics.keys())

    assert metrics["n_donors"] == len(data.donor_pool())
    assert metrics["n_pre"] == len(estimate.pre_times)
    assert metrics["n_post"] == len(estimate.post_times)
    assert np.isfinite(metrics["pre_rmse_sc"])
    assert np.isfinite(metrics["pre_rmse_aug"])
    assert metrics["n_placebos"] >= 1
    assert np.isclose(
        float(metrics["min_possible_p"]),
        1.0 / float(metrics["n_placebos"] + 1),
        atol=1e-12,
    )

    assert set(plots.keys()) == {
        "observed_vs_synthetic",
        "gap_over_time",
        "placebo_att_histogram",
    }
    paths = [Path(path) for path in plots.values()]
    assert len(set(paths)) == 3
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_run_scm_diagnostics_reports_robust_completion_fields(tmp_path):
    df = _make_panel_with_effect(effect=2.0)
    df = df[~((df["unit_id"] == "C3") & (df["time_id"] == 2))].copy()
    df.loc[(df["unit_id"] == "C2") & (df["time_id"] == 3), "y"] = np.nan
    data = PanelDataSCM(
        unit_id="unit_id",
        time_id="time_id",
        y="y",
        df=df,
        treated_unit="T",
        intervention_time=4,
        allow_missing_outcome=True,
    )
    estimate = RSCM(lambda_aug=0.5, completion_max_iter=250).fit(data).estimate()

    out = run_scm_diagnostics(
        estimate=estimate,
        paneldata=data,
        output_dir=tmp_path,
        filename_prefix="rscm",
    )
    metrics = out["metrics"]

    assert metrics["missing_cell_fraction"] is not None
    assert float(metrics["missing_cell_fraction"]) > 0.0
    assert metrics["completion_converged"] in {True, False}
    assert metrics["completion_effective_rank"] is not None
    assert int(metrics["completion_effective_rank"]) >= 1


def test_synthetic_control_namespace_exposes_scm_diagnostics():
    import causalis.scenarios.synthetic_control as scm

    assert hasattr(scm, "run_scm_diagnostics")
