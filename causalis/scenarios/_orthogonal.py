"""Canonical cross-fitted signals shared by subgroup and policy estimation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _compute_dr_signal_from_irm(
    irm_model: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the canonical unnormalized DR signal from fitted IRM nuisances."""
    y, d, g0_hat, g1_hat, m_hat = _resolve_irm_signal_inputs(irm_model)
    # Canonical DR signal for subgroup effects uses Horvitz-Thompson IPW terms.
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = d / m_hat
        h0 = (1.0 - d) / (1.0 - m_hat)
        phi = (g1_hat - g0_hat) + (y - g1_hat) * h1 - (y - g0_hat) * h0
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("Computed DR orthogonal signal contains non-finite values.")

    return phi, d, m_hat


def _resolve_irm_signal_inputs(
    irm_model: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve aligned outcome, treatment, and nuisance arrays used by subgroup scores."""
    if hasattr(irm_model, "_resolve_estimation_targets"):
        y_raw, d_raw = irm_model._resolve_estimation_targets()
    elif hasattr(irm_model, "_resolve_estimation_sample"):
        _, y_raw, d_raw = irm_model._resolve_estimation_sample()
    else:
        y_raw = getattr(irm_model, "_y", None)
        d_raw = getattr(irm_model, "_d", None)
        if y_raw is None or d_raw is None:
            raise RuntimeError(
                "IRM does not expose estimate-time sample arrays. Refit with matching data available."
            )

    y = np.asarray(y_raw, dtype=float).reshape(-1)
    d = np.asarray(d_raw, dtype=float).reshape(-1)
    g0_hat = np.asarray(irm_model.g0_hat_, dtype=float).reshape(-1)
    g1_hat = np.asarray(irm_model.g1_hat_, dtype=float).reshape(-1)
    m_hat = np.asarray(irm_model.m_hat_, dtype=float).reshape(-1)

    n = y.shape[0]
    if not (d.shape[0] == n == g0_hat.shape[0] == g1_hat.shape[0] == m_hat.shape[0]):
        raise RuntimeError(
            "Stored IRM arrays have inconsistent lengths; refit the model."
        )

    return y, d, g0_hat, g1_hat, m_hat
