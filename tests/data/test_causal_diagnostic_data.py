import numpy as np
import pytest
from pydantic import ValidationError

from causalis.data.causal_diagnostic_data import DiagnosticData, UnconfoundednessDiagnosticData
from causalis.data.causal_estimate import CausalEstimate


def test_diagnostic_data_instantiation():
    diag = DiagnosticData()
    assert isinstance(diag, DiagnosticData)


def test_unconfoundedness_diagnostic_data_instantiation():
    m_hat = np.array([0.1, 0.2, 0.3])
    d = np.array([0, 1, 0])
    diag = UnconfoundednessDiagnosticData(m_hat=m_hat, d=d)
    
    assert np.array_equal(diag.m_hat, m_hat)
    assert np.array_equal(diag.d, d)
    assert diag.y is None
    assert diag.x is None
    assert diag.trimming_threshold == 0.0


def test_unconfoundedness_diagnostic_data_full_instantiation():
    m_hat = np.array([0.1, 0.2, 0.3])
    d = np.array([0, 1, 0])
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([[1, 2], [3, 4], [5, 6]])
    diag = UnconfoundednessDiagnosticData(
        m_hat=m_hat, d=d, y=y, x=x, trimming_threshold=0.1
    )
    
    assert np.array_equal(diag.m_hat, m_hat)
    assert np.array_equal(diag.d, d)
    assert np.array_equal(diag.y, y)
    assert np.array_equal(diag.x, x)
    assert diag.trimming_threshold == 0.1


def test_unconfoundedness_diagnostic_data_missing_fields():
    with pytest.raises(ValidationError):
        UnconfoundednessDiagnosticData(m_hat=np.array([0.1]))


def test_causal_estimate_with_diagnostic_data():
    m_hat = np.array([0.1, 0.2, 0.3])
    d = np.array([0, 1, 0])
    diag = UnconfoundednessDiagnosticData(m_hat=m_hat, d=d)
    
    estimate = CausalEstimate(
        estimand="ATE",
        model="test_model",
        value=1.0,
        ci_upper_absolute=1.5,
        ci_lower_absolute=0.5,
        alpha=0.05,
        is_significant=True,
        n_treated=10,
        n_control=10,
        outcome="y",
        treatment="d",
        diagnostic_data=diag
    )
    
    assert estimate.diagnostic_data == diag
    assert isinstance(estimate.diagnostic_data, UnconfoundednessDiagnosticData)
    assert np.array_equal(estimate.diagnostic_data.m_hat, m_hat)


def test_causal_estimate_without_diagnostic_data():
    estimate = CausalEstimate(
        estimand="ATE",
        model="test_model",
        value=1.0,
        ci_upper_absolute=1.5,
        ci_lower_absolute=0.5,
        alpha=0.05,
        is_significant=True,
        n_treated=10,
        n_control=10,
        outcome="y",
        treatment="d"
    )
    
    assert estimate.diagnostic_data is None


def test_causal_estimate_with_empty_dict_diagnostic_data():
    estimate = CausalEstimate(
        estimand="ATE",
        model="test_model",
        value=1.0,
        ci_upper_absolute=1.5,
        ci_lower_absolute=0.5,
        alpha=0.05,
        is_significant=True,
        n_treated=10,
        n_control=10,
        outcome="y",
        treatment="d",
        diagnostic_data={}
    )
    
    assert isinstance(estimate.diagnostic_data, DiagnosticData)
    assert not isinstance(estimate.diagnostic_data, UnconfoundednessDiagnosticData)
