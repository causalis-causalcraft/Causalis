"""Numeric helpers for unconfoundedness IRM models."""
from __future__ import annotations

from typing import Any, Optional
import warnings

import numpy as np
from sklearn.base import is_classifier


def _is_binary(values: np.ndarray) -> bool:
    """Check if an array contains only binary values (0 and 1)."""
    uniq = np.unique(values)
    return np.array_equal(np.sort(uniq), np.array([0, 1])) or np.array_equal(np.sort(uniq), np.array([0.0, 1.0]))


def _safe_is_classifier(estimator) -> bool:
    """Safely check if an estimator is a classifier."""
    try:
        return is_classifier(estimator)
    except (AttributeError, TypeError):
        return getattr(estimator, "_estimator_type", None) == "classifier"


def _binary_label_is_one(label: Any) -> Optional[bool]:
    """Map a binary-like class label to {False, True}, if possible."""
    if isinstance(label, (bool, np.bool_)):
        return bool(label)
    try:
        val = float(label)
    except (TypeError, ValueError):
        return None
    if np.isclose(val, 1.0):
        return True
    if np.isclose(val, 0.0):
        return False
    return None


def _predict_prob_or_value(model, X: np.ndarray, is_propensity: bool = False) -> np.ndarray:
    """Predict probabilities or values using a model."""
    if _safe_is_classifier(model) and hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 1:
            # Assume this is already P(class=1).
            res = proba.ravel()
        elif proba.shape[1] == 1:
            # Can happen if the training fold has a single class.
            # Resolve P(class=1) from classes_ when available.
            classes = np.asarray(getattr(model, "classes_", [])).ravel()
            if classes.size == 1:
                class_is_one = _binary_label_is_one(classes[0])
                if class_is_one is True:
                    res = proba[:, 0]
                elif class_is_one is False:
                    res = np.zeros(proba.shape[0], dtype=float)
                else:
                    # Unknown class label semantics; fall back to available column.
                    res = proba[:, 0]
            else:
                # No reliable class metadata; infer from hard labels when possible.
                if hasattr(model, "predict"):
                    pred = np.asarray(model.predict(X)).ravel()
                    try:
                        pred_f = pred.astype(float)
                        res = np.where(np.isclose(pred_f, 1.0), 1.0, 0.0)
                    except (TypeError, ValueError):
                        res = proba[:, 0]
                else:
                    res = proba[:, 0]
        else:
            classes = np.asarray(getattr(model, "classes_", [])).ravel()
            pos_idx = None
            if classes.size == proba.shape[1]:
                for i, cls in enumerate(classes):
                    if _binary_label_is_one(cls) is True:
                        pos_idx = i
                        break
            if pos_idx is None:
                # Fallback to the second column when binary classes metadata is missing.
                pos_idx = 1
            res = proba[:, pos_idx]
    else:
        res = model.predict(X)

    res = np.asarray(res, dtype=float).ravel()
    if is_propensity:
        if np.any((res < -1e-12) | (res > 1.0 + 1e-12)):
            warnings.warn(
                "Propensity model produced values outside [0, 1]. "
                "Consider using a classifier or a model with a logistic link.",
                RuntimeWarning,
            )
        res = np.clip(res, 0.0, 1.0)
    return res


def _clip_propensity(p: np.ndarray, thr: float) -> np.ndarray:
    """Clip propensity scores to be within [thr, 1 - thr]."""
    thr = float(thr)
    if not np.isfinite(thr) or thr < 0.0 or thr >= 0.5:
        raise ValueError("trimming_threshold must be finite and in [0, 0.5).")
    return np.clip(p, thr, 1.0 - thr)

