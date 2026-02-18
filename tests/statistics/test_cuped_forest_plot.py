import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.cuped import CUPEDModel, cuped_forest_plot


@pytest.fixture
def sample_data():
    np.random.seed(123)
    n = 300
    x = np.random.normal(10.0, 1.5, size=n)
    d = np.random.binomial(1, 0.5, size=n)
    y = 2.0 + 3.0 * d + 1.2 * x + np.random.normal(0, 1.0, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x1": x})
    return CausalData(df=df, treatment="d", outcome="y", confounders=["x1"])


def test_cuped_forest_plot_uses_diagnostic_naive(sample_data):
    model = CUPEDModel(cov_type="HC3", alpha=0.05)
    estimate = model.fit(sample_data, covariates=["x1"]).estimate()

    fig = cuped_forest_plot(estimate)
    ax = fig.axes[0]

    assert fig is not None
    assert ax.get_title() == "Estimate and Absolute CI: CUPED vs Non-CUPED"
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["Without CUPED", "With CUPED"]


def test_cuped_forest_plot_accepts_explicit_without_cuped(sample_data):
    model = CUPEDModel(cov_type="HC3", alpha=0.05)
    estimate_with = model.fit(sample_data, covariates=["x1"]).estimate()

    estimate_without = estimate_with.model_copy(
        update={
            "value": estimate_with.diagnostic_data.ate_naive,
            "ci_lower_absolute": estimate_with.diagnostic_data.ate_naive - 0.5,
            "ci_upper_absolute": estimate_with.diagnostic_data.ate_naive + 0.5,
        }
    )

    fig = cuped_forest_plot(estimate_with, estimate_without_cuped=estimate_without)
    ax = fig.axes[0]

    assert fig is not None
    assert len(ax.lines) >= 3  # includes errorbar artists + reference line


def test_cuped_forest_plot_raises_without_diagnostics(sample_data):
    model = CUPEDModel(cov_type="HC3", alpha=0.05)
    estimate = model.fit(sample_data, covariates=["x1"]).estimate(diagnostic_data=False)

    with pytest.raises(ValueError, match="diagnostic_data"):
        cuped_forest_plot(estimate)
