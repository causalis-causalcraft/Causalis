"""Adapters from fitted estimates and model state to sensitivity inputs."""

from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional
import numpy as np
from . import _sensitivity_core as core
from ._sensitivity_core import _finite_vector
from ._score_utils import _compute_ipw_components
from ._cluster_utils import compact_cluster_codes, inference_metadata


def _diagnostic_value(diag: Any, key: str, default: Any = None) -> Any:
    return diag.get(key, default) if isinstance(diag, dict) else getattr(diag, key, default)


def _sensitivity_reference(effect: Any):
    """Recover the fitted sampling reference without reloading mutable data."""
    from scipy.stats import norm, t

    model = _sensitivity_model(effect)
    diag = (
        effect.get("diagnostic_data")
        if isinstance(effect, dict)
        else getattr(effect, "diagnostic_data", None)
    )
    options = (
        effect.get("model_options", {})
        if isinstance(effect, dict)
        else getattr(effect, "model_options", {})
    )
    options = options or {}
    has_saved_membership = (
        isinstance(diag, dict) and "cluster_codes" in diag
    ) or hasattr(diag, "cluster_codes")
    if has_saved_membership:
        codes = _diagnostic_value(diag, "cluster_codes")
    else:
        codes = getattr(model, "cluster_codes_", None)
    if codes is not None:
        if options.get("cov_type") == "iid":
            raise ValueError("The estimate's iid covariance metadata conflicts with cluster membership.")
        codes = compact_cluster_codes(codes, len(codes))
        metadata = inference_metadata(codes)
        if not metadata.get("inferential_available", True):
            raise ValueError("Clustered sensitivity inference requires at least two retained clusters.")
        return t(df=metadata["inference_df"]), metadata
    metadata = dict(options)
    if isinstance(effect, dict):
        metadata.update({
            key: effect[key]
            for key in ("cov_type", "inference_distribution", "inference_df", "n_clusters")
            if key in effect
        })
    declared_cluster = _diagnostic_value(diag, "cluster_name") is not None
    if not has_saved_membership:
        declared_cluster = declared_cluster or getattr(model, "_fit_cluster_name_", None) is not None
    if declared_cluster or metadata.get("cov_type") == "cluster" or metadata.get("inference_distribution") == "t":
        df = metadata.get("inference_df")
        if df is None and metadata.get("n_clusters") is not None:
            df = metadata["n_clusters"] - 1
        if df is None or not np.isfinite(df) or df < 1:
            raise ValueError("Clustered sampling inference requires the fitted cluster degrees of freedom.")
        return t(df=df), dict(
            cov_type="cluster", n_clusters=df + 1,
            inference_distribution="t", inference_df=df,
            covariance_correction=(df + 1) / df,
            inferential_available=True, inference_reason=None,
        )
    return norm, inference_metadata(None)


def _sensitivity_model(effect: Any) -> Any:
    if isinstance(effect, dict):
        return effect.get("model")
    if hasattr(effect, "coef_"):
        return effect
    return getattr(effect, "_model", None) or getattr(
        getattr(effect, "diagnostic_data", None), "_model", None
    )


def _resolve_sensitivity_label(
    effect_estimation: Dict[str, Any] | Any,
    *,
    model: Any | None = None,
) -> str:
    """Resolve the row label used for single-treatment sensitivity summaries."""
    treatment = getattr(effect_estimation, "treatment", None)
    if isinstance(treatment, str):
        return treatment

    if isinstance(effect_estimation, dict):
        estimate = effect_estimation.get("estimate")
        treatment = getattr(estimate, "treatment", None)
        if isinstance(treatment, str):
            return treatment
        model = effect_estimation.get("model", model)

    data_obj = getattr(model, "data", getattr(model, "data_contracts", None))
    treatment = getattr(data_obj, "treatment", None)
    if hasattr(treatment, "name"):
        return str(treatment.name)
    if isinstance(treatment, str):
        return treatment
    return "theta"


def _normalize_ate_atte_score(score: Any) -> str:
    """Normalize score aliases and enforce ATE/ATTE-only semantics."""
    score_u = str(score or "ATE").upper()
    if "ATT" in score_u:
        return "ATTE"
    if score_u == "ATE":
        return "ATE"
    raise ValueError(
        "Sensitivity analysis supports only score='ATE' or score='ATTE'. "
        f"Got {score!r}."
    )


def _resolve_sensitivity_score(
    effect_estimation: Any, model: Any, explicit_score: Any | None
) -> str:
    """Resolve benchmark/sensitivity score without coupling to model.score=GATE state."""
    if explicit_score is not None:
        return _normalize_ate_atte_score(explicit_score)

    score_candidates: list[Any] = []

    if hasattr(effect_estimation, "estimand"):
        score_candidates.append(getattr(effect_estimation, "estimand", None))

    diag = getattr(effect_estimation, "diagnostic_data", None)
    if diag is not None and hasattr(diag, "score"):
        score_candidates.append(getattr(diag, "score", None))

    if hasattr(model, "score"):
        score_candidates.append(getattr(model, "score", None))

    for cand in score_candidates:
        if cand is None:
            continue
        try:
            return _normalize_ate_atte_score(cand)
        except ValueError:
            continue

    return "ATE"


def _pull_theta_se_ci(
    effect_estimation: Any, alpha: float
) -> tuple[float, float, tuple[float, float]]:
    """Extract the estimate and SE, rebuilding the Wald CI at the requested alpha."""
    reference, _ = _sensitivity_reference(effect_estimation)

    if hasattr(effect_estimation, "value") and hasattr(
        effect_estimation, "ci_lower_absolute"
    ):
        theta = float(effect_estimation.value)
        options = getattr(effect_estimation, "model_options", {})
        if "std_error" in options:
            se = float(options["std_error"])
        else:
            original_alpha = float(getattr(effect_estimation, "alpha", alpha))
            if not 0.0 < original_alpha < 1.0:
                raise ValueError("The original estimate alpha must be in (0, 1).")
            se = (
                float(effect_estimation.ci_upper_absolute)
                - float(effect_estimation.ci_lower_absolute)
            ) / (2.0 * reference.ppf(1.0 - original_alpha / 2.0))
    elif isinstance(effect_estimation, dict):
        model = effect_estimation.get("model")
        theta = float(
            effect_estimation["coefficient"]
            if "coefficient" in effect_estimation
            else model.coef_[0]
        )
        if "std_error" in effect_estimation:
            se = float(effect_estimation["std_error"])
        elif model is not None and hasattr(model, "se_"):
            se = float(model.se_[0])
        elif "confidence_interval" in effect_estimation:
            lo, hi = map(float, effect_estimation["confidence_interval"])
            original_alpha = float(effect_estimation.get("alpha", alpha))
            if not 0.0 < original_alpha < 1.0:
                raise ValueError("The original estimate alpha must be in (0, 1).")
            se = (hi - lo) / (2.0 * reference.ppf(1.0 - original_alpha / 2.0))
        else:
            raise ValueError("A standard error is required for sensitivity inference.")
    elif hasattr(effect_estimation, "coef_") and hasattr(effect_estimation, "se_"):
        theta, se = float(effect_estimation.coef_[0]), float(effect_estimation.se_[0])
    else:
        raise TypeError(
            "Pass a CausalEstimate, result dictionary, or fitted IRM model."
        )
    z = float(reference.ppf(1.0 - alpha / 2.0))
    return theta, se, (theta - z * se, theta + z * se)


@dataclass(frozen=True)
class ResolvedEstimate:
    """One normalized view of a result dictionary, CausalEstimate, or IRM."""

    source: Any
    model: Any
    diagnostic_data: Any
    alpha: float
    score: str
    label: str

    @cached_property
    def inference(self) -> tuple[float, float, tuple[float, float]]:
        # Stored reports can be rendered even after the fitted model is gone.
        return _pull_theta_se_ci(self.source, self.alpha)

    @property
    def theta(self) -> float:
        return self.inference[0]

    @property
    def se(self) -> float:
        return self.inference[1]

    def elements(self) -> dict[str, Any]:
        diag = self.diagnostic_data
        if diag is not None and _diagnostic_value(diag, "sigma2") is not None:
            elements = {
                key: _diagnostic_value(diag, key)
                for key in (
                    "sigma2",
                    "nu2",
                    "psi_sigma2",
                    "psi_nu2",
                    "psi",
                    "riesz_rep",
                    "m_alpha",
                    "cluster_codes",
                    "cluster_name",
                )
            }
            _, metadata = _sensitivity_reference(self.source)
            elements["cov_type"] = metadata["cov_type"]
            return elements
        if self.model is None or not hasattr(self.model, "_sensitivity_element_est"):
            raise ValueError(
                "Sensitivity elements are unavailable; pass a fitted IRM with diagnostics."
            )
        if diag is not None:
            # Legacy snapshots may need to recover sensitivity moments from a
            # model reference. A later refit must not supply another fit's IFs.
            if not np.isclose(self.theta, float(self.model.coef_[0]), rtol=1e-12, atol=1e-14):
                raise ValueError("Saved diagnostics do not match the fitted model state; refit with diagnostics.")
            saved_fingerprint = _diagnostic_value(diag, "sample_fingerprint")
            current_fingerprint = getattr(self.model, "_fit_sample_fingerprint_", None)
            if saved_fingerprint is not None and current_fingerprint is not None:
                if any(saved_fingerprint.get(key) != value for key, value in current_fingerprint.items()):
                    raise ValueError("Saved diagnostics do not match the fitted model sample; refit with diagnostics.")
            for saved_key, current_key in (
                ("psi", "psi_"), ("m_hat", "m_hat_"),
                ("overlap_mask", "overlap_mask_"), ("cluster_codes", "cluster_codes_"),
            ):
                saved = _diagnostic_value(diag, saved_key)
                current = getattr(self.model, current_key, None)
                if saved is not None and (current is None or not np.array_equal(saved, current, equal_nan=True)):
                    raise ValueError("Saved diagnostics do not match the fitted model state; refit with diagnostics.")
            if getattr(self.model, "_fit_weights_used_", False):
                raise ValueError("Legacy weighted estimates require saved sensitivity elements; refit with diagnostics.")
        elements = self.model._sensitivity_element_est()
        if isinstance(elements, dict):
            elements = dict(elements)
            elements.setdefault("cluster_codes", getattr(self.model, "cluster_codes_", None))
            elements.setdefault("cluster_name", getattr(self.model, "_fit_cluster_name_", None))
        return elements

    @property
    def cached_result(self) -> Any:
        if isinstance(self.source, dict):
            return self.source.get("bias_aware")
        if not hasattr(self.source, "value"):
            return None
        diag = self.diagnostic_data
        if isinstance(diag, dict):
            return diag.get("sensitivity_analysis") or diag.get("bias_aware")
        return getattr(diag, "sensitivity_analysis", None) or getattr(
            diag, "bias_aware", None
        )

    def cache_result(self, result: Any) -> None:
        """Explicitly persist a completed analysis at the application boundary."""
        if isinstance(self.source, dict):
            self.source["bias_aware"] = result
        if isinstance(self.diagnostic_data, dict):
            self.diagnostic_data["sensitivity_analysis"] = result
        elif self.diagnostic_data is not None:
            try:
                self.diagnostic_data.sensitivity_analysis = result
            except Exception:
                pass


def resolve_estimate(effect: Any, alpha: float = 0.05) -> ResolvedEstimate:
    """Resolve supported containers once; numerical code receives only scalars and arrays."""
    if isinstance(effect, ResolvedEstimate):
        return effect
    if not (
        isinstance(effect, dict)
        or (hasattr(effect, "value") and hasattr(effect, "ci_lower_absolute"))
        or (hasattr(effect, "coef_") and hasattr(effect, "se_"))
    ):
        raise TypeError(
            "Pass a CausalEstimate, result dictionary, or fitted IRM model."
        )
    model = _sensitivity_model(effect)
    diag = (
        effect.get("diagnostic_data")
        if isinstance(effect, dict)
        else getattr(effect, "diagnostic_data", None)
    )
    return ResolvedEstimate(
        source=effect,
        model=model,
        diagnostic_data=diag,
        alpha=float(alpha),
        score=_resolve_sensitivity_score(effect, model, None),
        label=_resolve_sensitivity_label(effect, model=model),
    )


def compute_irm_sensitivity_elements(
    *,
    model: Any,
    y: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    m_hat: np.ndarray,
    w: Optional[np.ndarray] = None,
    w_bar: Optional[np.ndarray] = None,
    psi: Optional[np.ndarray] = None,
    inv_m: Optional[np.ndarray] = None,
    inv_1m: Optional[np.ndarray] = None,
    score: Any = "ATE",
) -> dict[str, Any]:
    """Compute DoubleML-style sensitivity elements for binary-treatment IRM."""
    y_arr = np.asarray(y, dtype=float).ravel()
    d_arr = np.asarray(d, dtype=float).ravel()
    g0_arr = np.asarray(g0, dtype=float).ravel()
    g1_arr = np.asarray(g1, dtype=float).ravel()
    m_hat_arr = np.asarray(m_hat, dtype=float).ravel()

    n = y_arr.size
    if any(arr.size != n for arr in (d_arr, g0_arr, g1_arr, m_hat_arr)):
        raise ValueError("y, d, g0, g1, and m_hat must share the same sample size.")
    for name, arr in (
        ("y", y_arr),
        ("d", d_arr),
        ("g0", g0_arr),
        ("g1", g1_arr),
        ("m_hat", m_hat_arr),
    ):
        _finite_vector(arr, name, n)
    if not np.isin(d_arr, [0.0, 1.0]).all():
        raise ValueError("d must contain only binary treatment indicators.")
    if not ((m_hat_arr > 0.0) & (m_hat_arr < 1.0)).all():
        raise ValueError("m_hat must lie strictly between zero and one.")

    if w is None or w_bar is None:
        if model is None or not hasattr(model, "_get_weights"):
            raise RuntimeError(
                "IRM sensitivity elements require model._get_weights when weights are omitted."
            )
        w, w_bar = model._get_weights(n=n, m_hat_adj=m_hat_arr, d=d_arr, score=score)

    w_arr = np.asarray(w, dtype=float).ravel()
    w_bar_arr = np.asarray(w_bar, dtype=float).ravel()
    if w_arr.size != n or w_bar_arr.size != n:
        raise ValueError("w and w_bar must share the same sample size as y.")

    if psi is None and model is not None:
        psi = getattr(model, "psi_", None)
    psi_arr = None if psi is None else _finite_vector(psi, "psi", n)

    if inv_m is None or inv_1m is None:
        normalize_ipw = (
            bool(getattr(model, "normalize_ipw", False)) if model is not None else False
        )
        _, _, inv_m_arr, inv_1m_arr = _compute_ipw_components(
            d=d_arr,
            m_hat=m_hat_arr,
            normalize_ipw=normalize_ipw,
            score=score,
            warn=False,
        )
    else:
        inv_m_arr = np.asarray(inv_m, dtype=float).ravel()
        inv_1m_arr = np.asarray(inv_1m, dtype=float).ravel()

    if inv_m_arr.size != n or inv_1m_arr.size != n:
        raise ValueError("inv_m and inv_1m must share the same sample size as y.")
    for name, arr in (
        ("w", w_arr),
        ("w_bar", w_bar_arr),
        ("inv_m", inv_m_arr),
        ("inv_1m", inv_1m_arr),
    ):
        _finite_vector(arr, name, n)

    elements = core.compute_sensitivity_elements(
        y=y_arr,
        d=d_arr,
        g0=g0_arr,
        g1=g1_arr,
        w=w_arr,
        w_bar=w_bar_arr,
        inv_m=inv_m_arr,
        inv_1m=inv_1m_arr,
        psi=psi_arr,
    )
    elements["cluster_codes"] = getattr(model, "cluster_codes_", None)
    elements["cluster_name"] = getattr(model, "_fit_cluster_name_", None)
    return elements
