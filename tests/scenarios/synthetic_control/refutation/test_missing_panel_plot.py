import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causalis.data_contracts import PanelDataSCM
from causalis.scenarios.synthetic_control import missing_panel_plot


def _panel_with_structured_missingness() -> PanelDataSCM:
    df = pd.DataFrame(
        {
            "unit_id": ["T", "T", "T", "C1", "C1", "C1", "C2", "C2"],
            "time_id": [1, 2, 3, 1, 2, 3, 1, 3],  # C2 at t=2 is absent.
            "y": [10.0, 11.0, 12.0, np.nan, 9.0, 10.0, 8.0, 8.5],
            "observed": [1, 1, 1, 0, 1, 1, 1, 1],
        }
    )
    return PanelDataSCM(
        unit_id="unit_id",
        time_id="time_id",
        y="y",
        df=df,
        treated_unit="T",
        intervention_time=3,
        donor_units=["C1", "C2"],
        observed_col="observed",
        allow_missing_outcome=True,
    )


def test_missing_panel_plot_unit_by_time_heatmap():
    panel = _panel_with_structured_missingness()

    fig = missing_panel_plot(panel)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.lines]
    heat = np.asarray(ax.images[0].get_array(), dtype=float)

    expected = np.array(
        [
            [0.0, 0.0, 0.0],  # treated
            [1.0, 0.0, 0.0],  # C1 has explicit missing at t=1
            [0.0, 1.0, 0.0],  # C2 is missing row at t=2
        ]
    )

    assert isinstance(fig, mpl_figure.Figure)
    assert not plt.fignum_exists(fig.number)
    assert ax.get_title() == "Panel Missingness by Unit and Time"
    assert ax.get_xlabel() == "time_id"
    assert ax.get_ylabel() == "unit_id"
    assert "Intervention" in labels
    assert heat.shape == expected.shape
    np.testing.assert_allclose(heat, expected)


def test_synthetic_control_namespace_exposes_missing_panel_plot():
    import causalis.scenarios.synthetic_control as scm

    assert hasattr(scm, "missing_panel_plot")
