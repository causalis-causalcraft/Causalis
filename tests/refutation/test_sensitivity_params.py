import warnings

import numpy as np
import pytest

from causalis.shared.sensitivity_params import (
    bias_factor,
    resolve_strength_aliases,
    robustness_value,
)


def test_aliases_map_one_to_one_and_warn():
    with pytest.warns(DeprecationWarning, match="r2_y is deprecated"):
        cf_y, cf_d = resolve_strength_aliases(cf_y=None, cf_d=0.3, r2_y=0.2)
    assert cf_y == 0.2  # no odds conversion
    assert cf_d == 0.3

    with pytest.warns(DeprecationWarning, match="r2_d is deprecated"):
        cf_y, cf_d = resolve_strength_aliases(cf_y=0.1, cf_d=None, r2_d=0.4)
    assert cf_y == 0.1
    assert cf_d == 0.4

    with pytest.warns(DeprecationWarning) as record:
        cf_y, cf_d = resolve_strength_aliases(cf_y=None, cf_d=None, r2_y=0.25, r2_d=0.5)
    assert (cf_y, cf_d) == (0.25, 0.5)
    assert len(record) == 2


def test_names_without_alias_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_strength_aliases(cf_y=0.1, cf_d=0.2) == (0.1, 0.2)


def test_both_name_and_alias_raise_value_error():
    with pytest.raises(ValueError, match="not both"):
        resolve_strength_aliases(cf_y=0.1, cf_d=0.2, r2_y=0.1)
    with pytest.raises(ValueError, match="not both"):
        resolve_strength_aliases(cf_y=0.1, cf_d=0.2, r2_d=0.2)


def test_missing_without_default_raises_type_error():
    with pytest.raises(TypeError, match="cf_y"):
        resolve_strength_aliases(cf_y=None, cf_d=0.2)
    with pytest.raises(TypeError, match="cf_d"):
        resolve_strength_aliases(cf_y=0.2, cf_d=None)


def test_missing_with_default_uses_default():
    assert resolve_strength_aliases(cf_y=None, cf_d=None, default=0.0) == (0.0, 0.0)
    assert resolve_strength_aliases(cf_y=0.3, cf_d=None, default=0.0) == (0.3, 0.0)


def test_bias_factor_value():
    assert float(bias_factor(0.2, 0.2)) == pytest.approx(np.sqrt(0.2 * 0.2 / 0.8))
    assert float(bias_factor(0.0, 0.5)) == 0.0
    assert float(bias_factor(0.5, 0.0)) == 0.0
    out = bias_factor(np.array([0.1, 0.4]), np.array([0.5, 0.2]))
    np.testing.assert_allclose(out, np.sqrt(np.array([0.1 * 1.0, 0.4 * 0.25])))


@pytest.mark.parametrize(
    ("cf_y", "cf_d"),
    [
        (0.1, 1.0),
        (1.0, 0.1),
        (0.1, -0.1),
        (-0.1, 0.1),
        (np.nan, 0.1),
        (0.1, np.inf),
        (0.1, 1.5),
    ],
)
def test_bias_factor_rejects_invalid_strengths(cf_y, cf_d):
    with pytest.raises(ValueError):
        bias_factor(cf_y, cf_d)


@pytest.mark.parametrize("delta", [0.05, 0.3, 1.0, 5.0, 100.0])
@pytest.mark.parametrize("scale", [0.1, 0.5, 2.0])
def test_robustness_value_solves_bound_equation(delta, scale):
    v = float(robustness_value(delta, scale))
    assert 0.0 < v < 1.0
    assert v / np.sqrt(1.0 - v) == pytest.approx(delta / scale, rel=1e-10)
    # Equivalent to the bound at cf_y = cf_d = v
    assert scale * float(bias_factor(v, v)) == pytest.approx(delta, rel=1e-10)


def test_robustness_value_uses_absolute_delta():
    assert float(robustness_value(-0.3, 0.5)) == pytest.approx(
        float(robustness_value(0.3, 0.5))
    )


@pytest.mark.parametrize(
    ("delta", "scale", "expected"),
    [(1e-200, 1.0, 1e-200), (1e100, 1.0, 1.0), (1e308, 1e-15, 1.0)],
)
def test_robustness_value_is_stable_for_extreme_ratios(delta, scale, expected):
    with np.errstate(over="raise", invalid="raise"):
        value = float(robustness_value(delta, scale))
    assert value == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("delta", [np.nan, np.inf, -np.inf])
def test_robustness_value_nonfinite_delta_is_undefined(delta):
    assert np.isnan(float(robustness_value(delta, 1.0)))


def test_robustness_value_zero_delta_and_nonpositive_scale():
    assert float(robustness_value(0.0, 0.5)) == 0.0
    assert np.isnan(float(robustness_value(0.5, 0.0)))
    assert np.isnan(float(robustness_value(0.5, -1.0)))
    assert np.isnan(float(robustness_value(0.5, np.nan)))
    # delta == 0 takes precedence even when scale is zero
    assert float(robustness_value(0.0, 0.0)) == 0.0


def test_robustness_value_vectorised_with_broadcasting():
    delta = np.array([0.0, 0.2, 1.0])
    scale = np.array([1.0, 0.0, 0.5])
    out = robustness_value(delta, scale)
    assert out.shape == (3,)
    assert out[0] == 0.0
    assert np.isnan(out[1])
    assert out[2] / np.sqrt(1.0 - out[2]) == pytest.approx(2.0, rel=1e-10)
