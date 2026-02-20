import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causalis.scenarios.unconfoundedness.refutation.overlap.reliability_plot import (
    plot_propensity_reliability,
)
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate


def _make_data_and_estimate(seed: int = 123):
    rng = np.random.default_rng(seed)
    n = 700
    x = rng.normal(size=n)
    m = 1.0 / (1.0 + np.exp(-(0.8 * x)))
    d = rng.binomial(1, m)
    return make_overlap_data_and_estimate(m_hat=m, d=d)


def test_reliability_plot_from_estimate():
    data, estimate = _make_data_and_estimate(seed=11)
    fig = plot_propensity_reliability(estimate, data=data, n_bins=10)

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert "Reliability" in ax.get_title()
    assert len(ax.collections) >= 1  # scatter points
    labels = [line.get_label() for line in ax.lines]
    assert "Perfect calibration" in labels

    plt.close(fig)


def test_reliability_plot_falls_back_to_data_for_d_without_recalibration():
    data, estimate = _make_data_and_estimate(seed=22)
    diag_without_d = estimate.diagnostic_data.model_copy(update={"d": None})
    estimate_without_d = estimate.model_copy(update={"diagnostic_data": diag_without_d})

    fig = plot_propensity_reliability(
        estimate_without_d,
        data=data,
        n_bins=10,
        show_recalibration=False,
        annotate_metrics=False,
    )

    assert fig is not None
    assert len(fig.axes) == 1
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "Perfect calibration" in labels
    assert all("Logit recalibration" not in label for label in labels)

    plt.close(fig)
