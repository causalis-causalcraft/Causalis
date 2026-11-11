"""Compatibility checks for sensitivity adapters and public entry points."""

from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest

from causalis.scenarios.unconfoundedness import _sensitivity_inputs as inputs
from causalis.scenarios.unconfoundedness.refutation import unconfoundedness
from causalis.scenarios.unconfoundedness.refutation.unconfoundedness import sensitivity


def _model():
    elements = dict(
        sigma2=1.0,
        nu2=4.0,
        psi=np.array([-0.1, 0.1]),
        psi_sigma2=np.zeros(2),
        psi_nu2=np.zeros(2),
    )
    return (
        SimpleNamespace(
            coef_=np.array([0.7]),
            se_=np.array([0.1]),
            score="ATE",
            data=SimpleNamespace(treatment="d"),
            _sensitivity_element_est=lambda: elements,
        ),
        elements,
    )


@pytest.mark.parametrize("name", sensitivity.__all__ + ["compute_bias_aware_ci"])
def test_public_exports_and_annotations_remain_usable(name):
    entrypoint = getattr(sensitivity, name)
    assert getattr(unconfoundedness, name) is entrypoint
    assert get_type_hints(entrypoint)


def test_irm_element_adapter_keeps_the_original_import_path():
    assert (
        sensitivity.compute_irm_sensitivity_elements
        is inputs.compute_irm_sensitivity_elements
    )


@pytest.mark.parametrize("kind", ["model", "dict", "estimate"])
def test_resolved_estimate_normalizes_supported_containers(kind):
    model, elements = _model()
    source = model
    if kind == "dict":
        source = {"model": model, "coefficient": 0.7, "std_error": 0.1}
    elif kind == "estimate":
        source = SimpleNamespace(
            value=0.7,
            model_options={"std_error": 0.1},
            ci_lower_absolute=0.5,
            ci_upper_absolute=0.9,
            alpha=0.05,
            treatment="d",
            estimand="ATE",
            diagnostic_data=SimpleNamespace(**elements, _model=model, score="ATE"),
        )
    resolved = inputs.resolve_estimate(source)
    assert inputs.resolve_estimate(resolved) is resolved
    assert resolved.model is model
    assert resolved.theta == 0.7
    assert resolved.se == 0.1
    assert resolved.score == "ATE"
    assert resolved.label == "d"
    np.testing.assert_array_equal(resolved.elements()["psi"], elements["psi"])


def test_inference_is_extracted_once_per_resolved_estimate(monkeypatch):
    model, _ = _model()
    original = inputs._pull_theta_se_ci
    calls = []

    def pull(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(inputs, "_pull_theta_se_ci", pull)
    estimate = inputs.resolve_estimate(model)
    assert estimate.theta == 0.7
    assert estimate.se == 0.1
    model.coef_[0] = 1.0
    assert estimate.theta == 0.7
    assert len(calls) == 1


def test_cached_report_does_not_require_a_live_model_or_inference(monkeypatch):
    model, _ = _model()
    result = sensitivity.sensitivity_analysis(model, r2_y=0.1, r2_d=0.1)

    def unused(*args, **kwargs):
        pytest.fail("A stored report must not recalculate inference.")

    monkeypatch.setattr(inputs, "_pull_theta_se_ci", unused)
    stored = {"model": None, "bias_aware": result}
    assert sensitivity.get_sensitivity_summary(stored) == result.text_summary()


def test_invalid_inference_is_rejected_before_nuisance_evaluation():
    model, _ = _model()
    model.se_[0] = np.nan

    def unused():
        pytest.fail("Nuisance elements must not be computed for an invalid SE.")

    model._sensitivity_element_est = unused
    with pytest.raises(ValueError, match="standard error"):
        sensitivity.compute_bias_aware_ci(model, r2_y=0.1, r2_d=0.1)
