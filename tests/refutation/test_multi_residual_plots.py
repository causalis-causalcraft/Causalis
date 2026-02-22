import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts.causal_diagnostic_data import MultiUnconfoundednessDiagnosticData
from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.refutation.score.residual_plots import (
    plot_residual_diagnostics,
)


def _build_data_and_estimate(*, include_yd_in_diag: bool = True) -> tuple[MultiCausalData, MultiCausalEstimate]:
    n = 240
    rng = np.random.default_rng(1234)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    logits = np.column_stack(
        [
            np.zeros(n, dtype=float),
            0.6 * x1 - 0.2 * x2,
            -0.3 * x1 + 0.5 * x2,
        ]
    )
    logits = logits - logits.max(axis=1, keepdims=True)
    m = np.exp(logits)
    m = m / m.sum(axis=1, keepdims=True)

    labels = np.array([rng.choice(3, p=m_i) for m_i in m], dtype=int)
    d = np.eye(3, dtype=float)[labels]

    g0 = 0.2 + 0.4 * x1 - 0.1 * x2
    g1 = g0 + 0.8
    g2 = g0 - 0.5 + 0.2 * x2
    g_hat = np.column_stack([g0, g1, g2])
    y = g_hat[np.arange(n), labels] + rng.normal(scale=0.25, size=n)

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "d_0": d[:, 0],
            "d_1": d[:, 1],
            "d_2": d[:, 2],
        }
    )
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=["d_0", "d_1", "d_2"],
        confounders=["x1", "x2"],
        control_treatment="d_0",
    )

    diag = MultiUnconfoundednessDiagnosticData(
        m_hat=m,
        d=d,
        y=y,
        x=df[["x1", "x2"]].to_numpy(dtype=float),
        g_hat=g_hat,
        score="ATE",
    )
    if not include_yd_in_diag:
        diag = diag.model_copy(update={"d": None, "y": None})

    zeros = np.zeros(2, dtype=float)
    estimate = MultiCausalEstimate(
        estimand="ATE",
        model="MultiTreatmentIRM",
        model_options={"normalize_ipw": False, "trimming_threshold": 1e-3},
        value=zeros,
        ci_upper_absolute=zeros,
        ci_lower_absolute=zeros,
        alpha=0.05,
        p_value=np.ones(2, dtype=float),
        is_significant=[False, False],
        n_treated=int(np.sum(d[:, 1:] == 1.0)),
        n_control=int(np.sum(d[:, 0] == 1.0)),
        outcome="y",
        treatment=["d_0", "d_1", "d_2"],
        contrast_labels=["d_1 vs d_0", "d_2 vs d_0"],
        confounders=["x1", "x2"],
        diagnostic_data=diag,
    )
    return data, estimate


def test_multi_plot_residual_diagnostics_basic():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    fig = plot_residual_diagnostics(estimate=estimate, data=data)

    assert fig is not None
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == 4
    titles = [ax.get_title() for ax in fig.axes]
    assert any("d_0: Residual vs Fitted" in title for title in titles)
    assert any("d_1: Residual vs Fitted" in title for title in titles)
    assert any("d_2: Residual vs Fitted" in title for title in titles)
    assert any("Calibration Error by Propensity Bin" in title for title in titles)

    plt.close(fig)


def test_multi_plot_residual_diagnostics_fallback_for_y_d():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=False)
    fig = plot_residual_diagnostics(estimate=estimate, data=data, n_bins=12)

    assert fig is not None
    assert not plt.fignum_exists(fig.number)
    assert len(fig.axes) == 4
    xlabels = [ax.get_xlabel() for ax in fig.axes]
    assert any(r"\hat g_{0}" in lbl for lbl in xlabels)
    assert any(r"\hat g_{1}" in lbl for lbl in xlabels)
    assert any(r"\hat g_{2}" in lbl for lbl in xlabels)
    assert any(r"\hat m_k" in lbl for lbl in xlabels)

    plt.close(fig)


def test_multi_plot_residual_diagnostics_uses_cached_inputs_without_data():
    _, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    diag = estimate.diagnostic_data
    residual_cache = {
        "y": np.asarray(diag.y, dtype=float).ravel(),
        "d": np.asarray(diag.d, dtype=float),
        "g": np.asarray(diag.g_hat, dtype=float),
        "m": np.asarray(diag.m_hat, dtype=float),
        "treatment_names": list(estimate.treatment),
    }
    diag_cached_only = diag.model_copy(
        update={
            "m_hat": None,
            "g_hat": None,
            "y": None,
            "d": None,
            "residual_plot_cache": residual_cache,
        }
    )
    estimate_cached_only = estimate.model_copy(update={"diagnostic_data": diag_cached_only})

    fig = plot_residual_diagnostics(estimate=estimate_cached_only, data=None)
    assert fig is not None
    assert len(fig.axes) == 4
    plt.close(fig)


def test_multi_plot_residual_diagnostics_populates_cache_on_first_call():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    diag_no_cache = estimate.diagnostic_data.model_copy(update={"residual_plot_cache": None})
    estimate_no_cache = estimate.model_copy(update={"diagnostic_data": diag_no_cache})

    fig = plot_residual_diagnostics(estimate=estimate_no_cache, data=data)
    assert fig is not None
    assert isinstance(estimate_no_cache.diagnostic_data.residual_plot_cache, dict)
    plt.close(fig)


def test_multi_plot_residual_diagnostics_requires_g_hat():
    data, estimate = _build_data_and_estimate(include_yd_in_diag=True)
    diag_missing_g = estimate.diagnostic_data.model_copy(update={"g_hat": None})
    estimate_missing_g = estimate.model_copy(update={"diagnostic_data": diag_missing_g})

    with pytest.raises(ValueError, match="g_hat"):
        plot_residual_diagnostics(estimate=estimate_missing_g, data=data)


def test_multi_refutation_namespace_exposes_residual_plot():
    import causalis.scenarios.multi_unconfoundedness.refutation as ref

    assert hasattr(ref, "plot_residual_diagnostics")
