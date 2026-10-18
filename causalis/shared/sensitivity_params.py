r"""
Shared parametrization of omitted-confounder strength for DML sensitivity analysis.

Follows Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), "Long Story
Short: Omitted Variable Bias in Causal Machine Learning", and the DoubleML
implementation's sensitivity bound. For IRM, confounder strength is expressed
through the outcome regression and the Riesz representer, rather than a
partial R^2 of the treatment:

- ``cf_y`` = :math:`R^2_{Y - g_s \sim g - g_s}`: share of the residual outcome
  variance of the short model explained by the omitted confounder.
- ``cf_d`` = :math:`1 - R^2_{\alpha \sim \alpha_s}`: share of the variance of the
  long Riesz representer not explained by the short one.

The bias bound is

.. math::

    |\theta_s - \theta| \le |\rho| \sqrt{cf_y \frac{cf_d}{1 - cf_d}} \sqrt{\sigma^2 \nu^2}.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "resolve_strength_aliases",
    "bias_factor",
    "robustness_value",
]


def resolve_strength_aliases(
    *,
    cf_y: Optional[Any],
    cf_d: Optional[Any],
    r2_y: Optional[Any] = None,
    r2_d: Optional[Any] = None,
    default: Optional[float] = None,
) -> Tuple[Any, Any]:
    """Resolve ``cf_y``/``cf_d`` from the deprecated ``r2_y``/``r2_d`` aliases.

    Aliases are mapped one-to-one (no odds conversion). Passing both a name and
    its alias raises ``ValueError``.
    """
    resolved = []
    for name, value, alias_name, alias in (
        ("cf_y", cf_y, "r2_y", r2_y),
        ("cf_d", cf_d, "r2_d", r2_d),
    ):
        if alias is not None:
            if value is not None:
                raise ValueError(
                    f"Pass either {name} or the deprecated alias {alias_name}, not both."
                )
            warnings.warn(
                f"{alias_name} is deprecated; use {name}. The value is now used as {name} "
                "directly (Chernozhukov et al., 2022 parametrization), without odds conversion.",
                DeprecationWarning,
                stacklevel=3,
            )
            value = alias
        if value is None:
            if default is None:
                raise TypeError(f"Missing required sensitivity parameter {name}.")
            value = default
        resolved.append(value)
    return resolved[0], resolved[1]


def bias_factor(cf_y: Any, cf_d: Any) -> np.ndarray:
    """Return ``sqrt(cf_y * cf_d / (1 - cf_d))`` after validating both lie in [0, 1)."""
    cf_y_arr = np.asarray(cf_y, dtype=float)
    cf_d_arr = np.asarray(cf_d, dtype=float)
    for name, arr in (("cf_y", cf_y_arr), ("cf_d", cf_d_arr)):
        if np.any(~np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr >= 1.0):
            raise ValueError(f"{name} must be finite and in [0, 1).")
    return np.sqrt(cf_y_arr * cf_d_arr / (1.0 - cf_d_arr))


def robustness_value(delta: Any, scale: Any) -> np.ndarray:
    """Robustness value: the ``v = cf_y = cf_d`` at which the bound reaches ``delta``.

    Solves ``scale * v / sqrt(1 - v) = delta`` with ``scale = |rho| * sqrt(sigma2 * nu2)``,
    i.e. ``v = (-D^2 + sqrt(D^4 + 4 D^2)) / 2`` for ``D = delta / scale``.
    Uses a rationalized form to avoid cancellation and overflow. Returns 0
    where ``delta == 0`` and NaN for non-finite inputs or unusable scales.
    """
    delta_arr = np.abs(np.asarray(delta, dtype=float))
    scale_arr = np.asarray(scale, dtype=float)
    delta_arr, scale_arr = np.broadcast_arrays(delta_arr, scale_arr)
    out = np.full(delta_arr.shape, np.nan, dtype=float)
    good = (
        np.isfinite(delta_arr)
        & np.isfinite(scale_arr)
        & (scale_arr > 1e-16)
        & (delta_arr > 0.0)
    )
    small = good & (delta_arr <= scale_arr)
    d = delta_arr[small] / scale_arr[small]
    out[small] = 2.0 * d / (np.hypot(d, 2.0) + d)
    large = good & ~small
    inverse_d = scale_arr[large] / delta_arr[large]
    out[large] = 2.0 / (1.0 + np.hypot(1.0, 2.0 * inverse_d))
    out[delta_arr == 0.0] = 0.0
    return out
