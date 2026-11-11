"""Cross-fitted IPCW for one fixed-horizon, artificially censored trial.

Baseline assignment odds adjust controls to the treated population. Sequential
inverse retention probabilities adjust their observed outcome prefixes. This is
a weighting estimator, not an orthogonal or doubly robust estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from causalis.data_contracts.ipcw_data import IPCWData
from causalis.data_contracts.ipcw_estimate import IPCWDiagnosticData, IPCWEstimate


@dataclass(frozen=True)
class _PreparedTrial:
    data: IPCWData
    frame: pd.DataFrame
    baseline: pd.DataFrame
    unit_ids: np.ndarray
    d: np.ndarray
    y: np.ndarray
    usable: np.ndarray
    risk: np.ndarray
    events: np.ndarray
    x_baseline: np.ndarray
    x_censoring: np.ndarray
    folds: np.ndarray
    seed: int


@dataclass(frozen=True)
class _TrialFit:
    treated_mean: np.ndarray
    control_mean: np.ndarray
    propensity_raw: np.ndarray
    propensity: np.ndarray
    hazard_raw: np.ndarray
    retention: np.ndarray
    survival_raw: np.ndarray
    survival: np.ndarray
    control_weight: np.ndarray

    @property
    def effects(self) -> np.ndarray:
        return self.treated_mean - self.control_mean


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _clip_parameter(value: Any, name: str, maximum: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and in (0, {maximum}).")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in (0, {maximum}).") from exc
    if not np.isfinite(out) or not 0 < out < maximum:
        raise ValueError(f"{name} must be finite and in (0, {maximum}).")
    return out


def _numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.loc[:, columns].to_numpy(dtype=float, na_value=np.nan)


def _probability(
    prototype: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    multiplicity: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit a fresh learner, repeating training rows by bootstrap multiplicity."""
    if x_train.shape[1] == 0:
        return np.full(len(x_test), np.average(y_train, weights=multiplicity))
    # CatBoost rejects an entirely constant design; its intercept-only limit is
    # the observed class probability, including bootstrap multiplicity.
    if isinstance(prototype, CatBoostClassifier) and np.all(
        np.ptp(x_train, axis=0) == 0
    ):
        return np.full(len(x_test), np.average(y_train, weights=multiplicity))
    learner = clone(prototype)
    params = learner.get_params()
    for name in ("random_state", "random_seed"):
        if name in params and params[name] is None:
            learner.set_params(**{name: seed})
    repeat = np.repeat(np.arange(len(y_train)), multiplicity)
    learner.fit(x_train[repeat], y_train[repeat])
    probabilities = np.asarray(learner.predict_proba(x_test), dtype=float)
    classes = np.asarray(getattr(learner, "classes_", []))
    if (
        classes.ndim != 1
        or classes.size != 2
        or np.count_nonzero(classes == 0) != 1
        or np.count_nonzero(classes == 1) != 1
    ):
        raise ValueError(
            "The fitted probability learner must expose binary classes_ {0, 1}."
        )
    positive = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape != (len(x_test), len(classes)):
        raise ValueError(
            "predict_proba must return one column per class and one row per observation."
        )
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("predict_proba must return finite probabilities in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("predict_proba rows must sum to one.")
    return probabilities[:, positive[0]]


def _fit_trial(
    prepared: _PreparedTrial,
    baseline_learner: Any,
    censoring_learner: Any,
    propensity_clip: float,
    censoring_clip: float,
    multiplicity: np.ndarray,
) -> _TrialFit:
    n, horizon = prepared.y.shape
    active = multiplicity > 0
    if not (np.any(active & (prepared.d == 1)) and np.any(active & (prepared.d == 0))):
        raise ValueError(
            "Both baseline treatment groups must have positive bootstrap support."
        )
    m_raw = np.full(n, np.nan)
    hazard_raw = np.zeros((n, horizon), dtype=float)
    for fold in np.unique(prepared.folds):
        test = active & (prepared.folds == fold)
        if not test.any():
            continue
        train = active & (prepared.folds != fold)
        if np.unique(prepared.d[train]).size != 2:
            raise ValueError(
                f"Baseline training fold {fold} must contain both treatment groups."
            )
        m_raw[test] = _probability(
            baseline_learner,
            prepared.x_baseline[train],
            prepared.d[train],
            prepared.x_baseline[test],
            multiplicity[train],
            seed=prepared.seed,
        )
        test_risk = prepared.risk & test[:, None]
        if not test_risk.any():
            continue
        train_risk = prepared.risk & train[:, None]
        if not train_risk.any():
            raise ValueError(
                f"Censoring training fold {fold} has no at-risk observations."
            )
        targets = prepared.events[train_risk].astype(int)
        if not targets.any():
            continue  # Exact retention=1 when this training fold has no events.
        if targets.all():
            raise ValueError(
                f"Censoring training fold {fold} contains only events; retention has no support."
            )
        row_counts = np.broadcast_to(multiplicity[:, None], (n, horizon))
        hazard_raw[test_risk] = _probability(
            censoring_learner,
            prepared.x_censoring[train_risk],
            targets,
            prepared.x_censoring[test_risk],
            row_counts[train_risk],
            seed=prepared.seed,
        )

    propensity = np.clip(m_raw, propensity_clip, 1.0 - propensity_clip)
    retention = np.maximum(1.0 - hazard_raw, censoring_clip)
    survival = np.cumprod(retention, axis=1)
    survival_raw = np.cumprod(1.0 - hazard_raw, axis=1)
    control_usable = active[:, None] & (prepared.d[:, None] == 0) & prepared.usable
    if (survival[control_usable] <= 0).any():
        raise ValueError(
            "Cumulative retention underflowed; the requested horizon has insufficient support."
        )
    weights = np.zeros((n, horizon), dtype=float)
    odds = propensity / (1.0 - propensity)
    numerator = np.broadcast_to(odds[:, None], weights.shape)
    weights[control_usable] = numerator[control_usable] / survival[control_usable]
    if not np.isfinite(weights).all():
        raise ValueError(
            "IPCW weights must be finite; check positivity and the follow-up horizon."
        )
    weighted = weights * multiplicity[:, None]
    denominator = weighted.sum(axis=0)
    if (denominator <= 0).any():
        month = int(np.flatnonzero(denominator <= 0)[0])
        raise ValueError(
            f"No usable controls with positive weight at followup_month={month}."
        )
    treated_counts = multiplicity * prepared.d
    treated_mean = (prepared.y * treated_counts[:, None]).sum(
        axis=0
    ) / treated_counts.sum()
    control_mean = (prepared.y * weighted).sum(axis=0) / denominator
    return _TrialFit(
        treated_mean,
        control_mean,
        m_raw,
        propensity,
        hazard_raw,
        retention,
        survival_raw,
        survival,
        weights,
    )


def _bootstrap_replication(
    index: int,
    prepared: _PreparedTrial,
    baseline_learner: Any,
    censoring_learner: Any,
    propensity_clip: float,
    censoring_clip: float,
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([prepared.seed, index, 1701]))
    n = len(prepared.d)
    counts = rng.multinomial(n, np.full(n, 1.0 / n))
    try:
        fitted = _fit_trial(
            prepared,
            baseline_learner,
            censoring_learner,
            propensity_clip,
            censoring_clip,
            counts,
        )
    except Exception as exc:
        raise RuntimeError(
            f"IPCW bootstrap replication {index + 1} failed: {exc}"
        ) from exc
    return fitted.effects


def _inference(
    values: np.ndarray, draws: np.ndarray | None, alpha: float
) -> dict[str, Any]:
    if draws is None:
        return {
            key: [None] * len(values)
            for key in (
                "std_error",
                "ci_lower_absolute",
                "ci_upper_absolute",
                "p_value",
                "is_significant",
            )
        }
    se = np.std(draws, axis=0, ddof=1)
    critical = norm.ppf(1.0 - alpha / 2.0)
    p_value = np.empty(len(values))
    positive = se > 0
    p_value[positive] = 2.0 * norm.sf(np.abs(values[positive] / se[positive]))
    p_value[~positive] = np.where(values[~positive] == 0, 1.0, 0.0)
    return {
        "std_error": se,
        "ci_lower_absolute": values - critical * se,
        "ci_upper_absolute": values + critical * se,
        "p_value": p_value,
        "is_significant": p_value < alpha,
    }


class IPCW(BaseEstimator):
    """IPCW ATT for a single trial with artificial censoring of controls.

    ``ml_m`` estimates baseline assignment; ``ml_c`` estimates the censoring
    event hazard on subsequent at-risk rows. Both accept sklearn-compatible
    classifiers with ``fit`` and ``predict_proba``. Default learners are
    CatBoost classifiers (200 iterations, depth 6, one thread).

    Identification requires baseline exchangeability, sequential conditional
    exchangeability for censoring, and positivity. The estimate is a sum of
    monthly effects, using separately normalized control weights each month.
    Cross-fitting does not make pure IPCW doubly robust. Inference uses a client
    bootstrap that refits both probability models, conditional on the original
    fold assignment; validity still depends on the nuisance estimation method.

    ``n_jobs`` parallelizes bootstrap replications. ``store_diagnostics=False``
    omits detailed diagnostic payloads, while retaining the fit snapshot needed
    to refit models during bootstrap. It does not change point estimates or CIs.
    """

    def __init__(
        self,
        data: IPCWData | None = None,
        ml_m: Any = None,
        ml_c: Any = None,
        *,
        n_folds: int = 4,
        propensity_clip: float = 0.01,
        censoring_clip: float = 0.01,
        random_state: int | None = None,
        n_jobs: int = 1,
        store_diagnostics: bool = True,
    ) -> None:
        self.data = data
        self.ml_m = ml_m
        self.ml_c = ml_c
        self.n_folds = n_folds
        self.propensity_clip = propensity_clip
        self.censoring_clip = censoring_clip
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.store_diagnostics = store_diagnostics

    def _validate_config(self) -> None:
        _integer(self.n_folds, "n_folds", 2)
        _clip_parameter(self.propensity_clip, "propensity_clip", 0.5)
        _clip_parameter(self.censoring_clip, "censoring_clip", 1.0)
        if self.random_state is not None:
            _integer(self.random_state, "random_state", 0)
        jobs = _integer(self.n_jobs, "n_jobs", -1)
        if jobs == 0:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if not isinstance(self.store_diagnostics, (bool, np.bool_)):
            raise ValueError("store_diagnostics must be boolean.")
        for name in ("ml_m", "ml_c"):
            learner = getattr(self, name)
            if learner is not None:
                if not all(
                    callable(getattr(learner, attr, None))
                    for attr in ("fit", "predict_proba")
                ):
                    raise ValueError(f"{name} must implement fit and predict_proba.")
                clone(learner)

    def fit(self, data: IPCWData | None = None) -> "IPCW":
        """Snapshot the validated trial and cross-fit both probability models."""
        self._prepared_ = None
        self._fit_result_ = None
        self._validate_config()
        if data is not None:
            self.data = data
        if not isinstance(self.data, IPCWData):
            raise ValueError("Provide IPCWData in the constructor or .fit(data).")
        snapshot = self.data.model_copy(deep=True)
        frame = snapshot.get_df()
        codes, units = pd.factorize(frame[snapshot.unit_col], sort=False)
        order = np.lexsort((frame[snapshot.followup_col].to_numpy(), codes))
        frame = frame.iloc[order].reset_index(drop=True)
        baseline = frame.loc[frame[snapshot.followup_col] == 0].reset_index(drop=True)
        n, horizon = len(baseline), snapshot.horizon
        d = baseline[snapshot.treatment].to_numpy(dtype=int)
        if np.bincount(d, minlength=2).min() < self.n_folds:
            raise ValueError(
                "Each baseline treatment group must contain at least n_folds clients."
            )
        seed = int(np.random.SeedSequence(self.random_state).generate_state(1)[0])
        folds = np.empty(n, dtype=int)
        splitter = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=seed
        )
        for fold, (_, heldout) in enumerate(splitter.split(np.zeros(n), d)):
            folds[heldout] = fold
        usable = (
            frame[snapshot.outcome_usable_col].to_numpy(dtype=bool).reshape(n, horizon)
        )
        risk = frame[snapshot.at_risk_col].to_numpy(dtype=bool).reshape(n, horizon)
        events = frame[snapshot.censor_col].to_numpy(dtype=bool).reshape(n, horizon)
        # Never read unused outcomes or post-censoring predictors into the fit.
        y = np.zeros((n, horizon))
        y[usable] = pd.to_numeric(frame.loc[usable.ravel(), snapshot.outcome]).to_numpy(
            dtype=float
        )
        x_censoring = np.zeros((n, horizon, len(snapshot.censoring_covariates)))
        x_censoring[risk] = _numeric_matrix(
            frame.loc[risk.ravel()], snapshot.censoring_covariates
        )
        prepared = _PreparedTrial(
            snapshot,
            frame,
            baseline,
            np.asarray(units),
            d,
            y,
            usable,
            risk,
            events,
            _numeric_matrix(baseline, snapshot.baseline_covariates),
            x_censoring,
            folds,
            seed,
        )

        def default_learner() -> CatBoostClassifier:
            return CatBoostClassifier(
                iterations=200,
                depth=6,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
                random_seed=seed,
            )

        baseline_learner = (
            clone(self.ml_m) if self.ml_m is not None else default_learner()
        )
        censoring_learner = (
            clone(self.ml_c) if self.ml_c is not None else default_learner()
        )
        fitted = _fit_trial(
            prepared,
            baseline_learner,
            censoring_learner,
            float(self.propensity_clip),
            float(self.censoring_clip),
            np.ones(n, dtype=int),
        )
        self._prepared_ = prepared
        self._fit_result_ = fitted
        self._baseline_learner_ = baseline_learner
        self._censoring_learner_ = censoring_learner
        self._fit_propensity_clip_ = float(self.propensity_clip)
        self._fit_censoring_clip_ = float(self.censoring_clip)
        self._fit_store_diagnostics_ = bool(self.store_diagnostics)
        return self

    def _diagnostics(
        self, prepared: _PreparedTrial, fitted: _TrialFit
    ) -> IPCWDiagnosticData:
        from .refutation.diagnostics import (
            _support_table_from_rows,
            _weight_summary_from_rows,
        )

        data, frame = prepared.data, prepared.frame
        _, horizon = prepared.y.shape
        rows = pd.DataFrame(
            {
                "unit_id": frame[data.unit_col].to_numpy(),
                "trial_id": frame[data.trial_col].to_numpy(),
                "time": frame[data.time_col].to_numpy(),
                "followup_month": frame[data.followup_col].to_numpy(),
                "treatment": np.repeat(prepared.d, horizon),
                "at_risk": prepared.risk.ravel(),
                "censor_now": prepared.events.ravel(),
                "outcome_usable": prepared.usable.ravel(),
                "fold": np.repeat(prepared.folds, horizon),
                "propensity_raw": np.repeat(fitted.propensity_raw, horizon),
                "propensity": np.repeat(fitted.propensity, horizon),
                "hazard_raw": fitted.hazard_raw.ravel(),
                "hazard": (1.0 - fitted.retention).ravel(),
                "retention": fitted.retention.ravel(),
                "survival_raw": fitted.survival_raw.ravel(),
                "survival": fitted.survival.ravel(),
                "control_weight": fitted.control_weight.ravel(),
                "propensity_clipped": np.repeat(
                    fitted.propensity != fitted.propensity_raw, horizon
                ),
                "censoring_clipped": (
                    fitted.retention != 1.0 - fitted.hazard_raw
                ).ravel(),
            }
        )
        baseline = rows.loc[rows.followup_month == 0].reset_index(drop=True).copy()
        for name in data.baseline_covariates:
            baseline[f"feature:{name}"] = prepared.baseline[name].to_numpy(copy=True)
        return IPCWDiagnosticData(
            row_data=rows,
            baseline_data=baseline,
            support=_support_table_from_rows(rows),
            weight_summary=_weight_summary_from_rows(rows),
        )

    def estimate(self, *, alpha: float = 0.05, n_bootstrap: int = 199) -> IPCWEstimate:
        """Return horizon-sum ATT with joint monthly and cumulative inference.

        ``n_bootstrap=0`` requests point estimates only; otherwise at least two
        replications are required. Full client refits can be expensive; use
        499 or more replications for more stable final inference. Failed draws
        raise with their replication number and are never silently discarded.
        """
        if getattr(self, "_fit_result_", None) is None:
            raise RuntimeError("Call .fit(data) before .estimate().")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be finite and in (0, 1).")
        replications = _integer(n_bootstrap, "n_bootstrap", 0)
        if replications == 1:
            raise ValueError("n_bootstrap must be 0 or an integer >= 2.")
        jobs = _integer(self.n_jobs, "n_jobs", -1)
        if jobs == 0:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        prepared, fitted = self._prepared_, self._fit_result_
        draws = None
        if replications:
            draws = np.asarray(
                Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(_bootstrap_replication)(
                        b,
                        prepared,
                        self._baseline_learner_,
                        self._censoring_learner_,
                        self._fit_propensity_clip_,
                        self._fit_censoring_clip_,
                    )
                    for b in range(replications)
                )
            )
        data, horizon = prepared.data, prepared.data.horizon
        period = pd.DataFrame(
            {
                "time": prepared.frame.iloc[:horizon][data.time_col].to_numpy(),
                "followup_month": np.arange(horizon),
                "value": fitted.effects,
                "treatment_mean": fitted.treated_mean,
                "control_mean": fitted.control_mean,
                **_inference(fitted.effects, draws, float(alpha)),
            }
        )
        cumulative = period.copy(deep=True)
        for column in ("value", "treatment_mean", "control_mean"):
            cumulative[column] = period[column].cumsum()
        cumulative_draws = None if draws is None else np.cumsum(draws, axis=1)
        for column, values in _inference(
            cumulative.value.to_numpy(), cumulative_draws, float(alpha)
        ).items():
            cumulative[column] = values
        total = cumulative.iloc[-1]
        diagnostic_data = (
            self._diagnostics(prepared, fitted)
            if self._fit_store_diagnostics_
            else None
        )
        return IPCWEstimate(
            value=float(total.value),
            std_error=total.std_error,
            ci_lower_absolute=total.ci_lower_absolute,
            ci_upper_absolute=total.ci_upper_absolute,
            p_value=total.p_value,
            is_significant=total.is_significant,
            alpha=float(alpha),
            n_treated=int(prepared.d.sum()),
            n_control=int((prepared.d == 0).sum()),
            horizon=horizon,
            trial_id=data.trial_id,
            treatment_mean=float(cumulative.treatment_mean.iloc[-1]),
            control_mean=float(cumulative.control_mean.iloc[-1]),
            outcome=data.outcome,
            treatment=data.treatment,
            baseline_covariates=list(data.baseline_covariates),
            censoring_covariates=list(data.censoring_covariates),
            model_options={
                "n_folds": len(np.unique(prepared.folds)),
                "propensity_clip": self._fit_propensity_clip_,
                "censoring_clip": self._fit_censoring_clip_,
                "n_bootstrap": replications,
                "inference": "client_refit_bootstrap_wald" if replications else "none",
                "bootstrap_fixed_folds": True,
                "seed": prepared.seed,
                "ml_m": self._baseline_learner_.get_params(),
                "ml_c": self._censoring_learner_.get_params(),
            },
            period_effects=period,
            cumulative_effects=cumulative,
            diagnostic_data=diagnostic_data,
        )
