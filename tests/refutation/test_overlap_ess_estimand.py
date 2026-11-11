"""ESS must describe the weights for the requested target population."""
import numpy as np
import pytest

from causalis.scenarios.unconfoundedness.refutation import run_overlap_diagnostics
from causalis.scenarios.multi_unconfoundedness.refutation.overlap.overlap_validation import (
    run_overlap_diagnostics as run_multi_overlap,
)
from tests.refutation._overlap_test_utils import make_overlap_data_and_estimate
from tests.refutation.test_multi_unconfoundedness_diagnostics import (
    _make_manual_multi_data, _make_manual_estimate,
)


def kish(w):
    return np.sum(w) ** 2 / np.dot(w, w)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
@pytest.mark.parametrize("hajek", [False, True])
def test_binary_ess_uses_estimand_weights(score, hajek):
    # ATE control weights [1.01, 1.11, 2, 10] hide the much greater
    # concentration of ATT odds weights [0.01, 0.11, 1, 9].
    m = np.array([.01, .1, .5, .9, .02, .2, .6, .8])
    d = np.array([0] * 4 + [1] * 4)
    data, estimate = make_overlap_data_and_estimate(m, d)
    estimate = estimate.model_copy(update={"estimand": score})
    report = run_overlap_diagnostics(
        data, estimate, use_hajek=hajek,
        thresholds={"ess_ratio_warn": .4, "ess_ratio_strong": .2},
    )
    expected_control = kish(m[:4] / (1 - m[:4])) / 4
    expected_treated = 1.0
    if score == "ATE":
        expected_control = kish(1 / (1 - m[:4])) / 4
        expected_treated = kish(1 / m[4:]) / 4
    summary = report["summary"].set_index("metric")
    assert summary.loc["ESS_control_ratio", "value"] == pytest.approx(expected_control)
    assert summary.loc["ESS_treated_ratio", "value"] == pytest.approx(expected_treated)
    assert report["flags"]["ess_w0"] == ("YELLOW" if score == "ATTE" else "GREEN")
    assert summary.loc["ESS_control_ratio", "flag"] == report["flags"]["ess_w0"]
    # The existing explicitly ATE-labelled field keeps its public meaning.
    assert report["ate_ess"]["ess_w1"] == pytest.approx(kish(1 / m[4:]))
    if score == "ATTE":
        assert report["att_ess"]["ess_w1"] == 4
        assert report["att_ess"]["ess_w0"] == pytest.approx(4 * expected_control)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
@pytest.mark.parametrize("hajek", [False, True])
def test_multi_ess_uses_generalized_propensity_odds(score, hajek):
    d = np.eye(3)[np.repeat(np.arange(3), 4)]
    # Third-arm mass varies: pair-normalized inverse weights are not ATE weights.
    m = np.array([[.8, .01, .19], [.6, .1, .3], [.3, .5, .2], [.05, .9, .05],
                  [.1, .2, .7], [.2, .3, .5], [.1, .8, .1], [.3, .6, .1],
                  [.2, .1, .7], [.3, .2, .5], [.05, .1, .85], [.5, .2, .3]])
    data = _make_manual_multi_data(d=d, treatment_names=["d0", "d1", "d2"])
    estimate = _make_manual_estimate(data=data, d=d, m_hat=m)
    diag = estimate.diagnostic_data.model_copy(update={"score": score})
    estimate = estimate.model_copy(update={"estimand": score, "diagnostic_data": diag})
    report = run_multi_overlap(data, estimate, use_hajek=hajek)
    for k, row in enumerate(report["overlap"]["by_comparison"].itertuples(), start=1):
        wt = np.ones(4) if score == "ATTE" else 1 / m[d[:, k] == 1, k]
        wb = m[:4, k] / m[:4, 0] if score == "ATTE" else 1 / m[:4, 0]
        assert row.ess_ratio_treated == pytest.approx(kish(wt) / 4)
        assert row.ess_ratio_baseline == pytest.approx(kish(wb) / 4)


@pytest.mark.parametrize("raw", [False, True])
def test_att_ess_uses_same_propensity_source_as_overlap_report(raw):
    m = np.array([.01, .1, .5, .9, .02, .2, .6, .8])
    d = np.array([0] * 4 + [1] * 4)
    data, estimate = make_overlap_data_and_estimate(np.clip(m, .2, .8), d)
    diag = estimate.diagnostic_data.model_copy(update={
        "score": "ATTE", "m_hat_raw": m if raw else None,
    })
    # Diagnostic score takes precedence, as in existing overlap diagnostics.
    estimate = estimate.model_copy(update={"diagnostic_data": diag})
    report = run_overlap_diagnostics(data, estimate)
    source = m if raw else np.clip(m, .2, .8)
    assert report["att_ess"]["ess_w0"] == pytest.approx(
        kish(source[:4] / (1 - source[:4]))
    )
