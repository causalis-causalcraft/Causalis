import numpy as np

from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.data_contracts.causal_estimate import CausalEstimate


def _estimate(model_options):
    return CausalEstimate(
        estimand="ATE",
        model="IRM",
        model_options=model_options,
        value=0.4,
        ci_lower_absolute=0.1,
        ci_upper_absolute=0.7,
        alpha=0.05,
        p_value=0.01,
        is_significant=True,
        n_treated=20,
        n_control=20,
        treatment_mean=2.4,
        control_mean=2.0,
        outcome="y",
        treatment="d",
    )


def test_cluster_summary_exposes_inference_metadata_without_changing_estimate_schema():
    options = {
        "cov_type": "cluster",
        "cluster_name": "school",
        "n_clusters_fit": 12,
        "n_clusters": 10,
        "covariance_correction": 10 / 9,
        "inference_distribution": "t",
        "inference_df": 9,
        "std_error": 0.123456,
        "std_error_relative": 6.123456,
    }
    estimate = _estimate(options)
    summary = estimate.summary()["value"]

    for key, value in options.items():
        expected = f"{value:.4f}" if key.startswith("std_error") else value
        assert summary[key] == expected
    assert "cov_type" not in CausalEstimate.model_fields
    assert estimate.model_options == options


def test_unclustered_summary_retains_existing_fields():
    summary = _estimate({"std_error": 0.123456}).summary()

    assert "cov_type" not in summary.index
    assert "cluster_name" not in summary.index
    assert "std_error" not in summary.index
    assert "inference_df" not in summary.index


def test_cluster_diagnostic_shares_immutable_codes_and_allows_legacy_payloads():
    codes = np.array([0, 1, 0, 1], dtype=np.intp)
    codes.setflags(write=False)
    common = {"m_hat": np.full(4, 0.5), "d": np.array([0, 1, 1, 0])}
    diagnostic = UnconfoundednessDiagnosticData(
        **common, cluster_codes=codes, cluster_name="school"
    )

    assert diagnostic.cluster_codes is codes
    assert not diagnostic.cluster_codes.flags.writeable
    assert diagnostic.cluster_name == "school"
    legacy = UnconfoundednessDiagnosticData(**common)
    assert legacy.cluster_codes is None
    assert legacy.cluster_name is None
