r"""
IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

from sklearn.base import is_classifier, clone, BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from scipy.stats import norm
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import UnconfoundednessDiagnosticData
from causalis.scenarios.cate.blp import BLP




def _is_binary(values: np.ndarray) -> bool:
    """Check if an array contains only binary values (0 and 1).

    Parameters
    ----------
    values : np.ndarray
        The array to check.

    Returns
    -------
    bool
        True if the array is binary, False otherwise.
    """
    uniq = np.unique(values)
    return np.array_equal(np.sort(uniq), np.array([0, 1])) or np.array_equal(np.sort(uniq), np.array([0.0, 1.0]))


def _safe_is_classifier(estimator) -> bool:
    """Safely check if an estimator is a classifier.

    Parameters
    ----------
    estimator : estimator
        The estimator to check.

    Returns
    -------
    bool
        True if the estimator is a classifier, False otherwise.
    """
    try:
        return is_classifier(estimator)
    except (AttributeError, TypeError):
        return getattr(estimator, "_estimator_type", None) == "classifier"


def _predict_prob_or_value(model, X: np.ndarray, is_propensity: bool = False) -> np.ndarray:
    """Predict probabilities or values using a model.

    Parameters
    ----------
    model : estimator
        The fitted model to use for prediction.
    X : np.ndarray
        The input features.
    is_propensity : bool, default False
        Whether the prediction is for a propensity score. If True, values are clipped to [0, 1].

    Returns
    -------
    np.ndarray
        The predicted values or probabilities.
    """
    if _safe_is_classifier(model) and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] == 1:
            res = proba.ravel()
        else:
            res = proba[:, 1]
    else:
        res = model.predict(X)

    res = np.asarray(res, dtype=float).ravel()
    if is_propensity:
        if np.any((res < -1e-12) | (res > 1.0 + 1e-12)):
            warnings.warn("Propensity model produced values outside [0, 1]. "
                          "Consider using a classifier or a model with a logistic link.", RuntimeWarning)
        res = np.clip(res, 0.0, 1.0)
    return res


def _clip_propensity(p: np.ndarray, thr: float) -> np.ndarray:
    """Clip propensity scores to be within [thr, 1 - thr].

    Parameters
    ----------
    p : np.ndarray
        The propensity scores to clip.
    thr : float
        The threshold for clipping.

    Returns
    -------
    np.ndarray
        The clipped propensity scores.
    """
    thr = float(thr)
    return np.clip(p, thr, 1.0 - thr)


# IRMResults removed, functionality replaced by CausalEstimate


class IRM(BaseEstimator):
    """Interactive Regression Model (IRM) with cross-fitting using CausalData.

    Parameters
    ----------
    data : CausalData
        Data container with outcome, binary treatment (0/1), and confounders.
    ml_g : estimator
        Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
    ml_m : classifier
        Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
    n_folds : int, default 5
        Number of cross-fitting folds.
    n_rep : int, default 1
        Number of repetitions of sample splitting. Currently only 1 is supported.
    normalize_ipw : bool, default False
        Whether to normalize IPW terms within the score.
    trimming_rule : {"truncate"}, default "truncate"
        Trimming approach for propensity scores.
    trimming_threshold : float, default 1e-2
        Threshold for trimming if rule is "truncate".
    weights : Optional[np.ndarray or Dict], default None
        Optional weights.
        - If array of shape (n,), used as ATE weights (w). Assumed E[w|X] = w.
        - If dict, can contain 'weights' (w) and 'weights_bar' (E[w|X]).
        - For ATTE, computed internally (w=D/P(D=1), w_bar=m(X)/P(D=1)).
        Note: If weights depend on treatment or outcome, E[w|X] must be provided for correct sensitivity analysis.
    relative_baseline_min : float, default 1e-8
        Minimum absolute baseline value used for relative effects. If |mu_c| is below this
        threshold, relative estimates are set to NaN with a warning.
    random_state : Optional[int], default None
        Random seed for fold creation.
    """

    def __init__(
        self,
        data: Optional[CausalData] = None,
        ml_g: Any = None,
        ml_m: Any = None,
        *,
        n_folds: int = 5,
        n_rep: int = 1,
        normalize_ipw: bool = False,
        trimming_rule: str = "truncate",
        trimming_threshold: float = 1e-2,
        weights: Optional[np.ndarray | Dict[str, Any]] = None,
        relative_baseline_min: float = 1e-8,
        random_state: Optional[int] = None,
    ) -> None:
        self.data = data
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        self.score = "ATE"
        self.normalize_ipw = bool(normalize_ipw)
        self.trimming_rule = str(trimming_rule)
        self.trimming_threshold = float(trimming_threshold)
        self.weights = weights
        self.relative_baseline_min = float(relative_baseline_min)
        self.random_state = random_state

        # Initialize default learners if not provided
        if HAS_CATBOOST:
            if self.ml_m is None:
                self.ml_m = CatBoostClassifier(
                    thread_count=-1,
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
            if self.ml_g is None and self.data is not None:
                y_is_binary = False
                try:
                    df_tmp = self.data.get_df()
                    y_tmp = df_tmp[self.data.outcome.name].to_numpy()
                    y_is_binary = _is_binary(y_tmp)
                except (AttributeError, KeyError, ValueError):
                    pass

                if y_is_binary:
                    self.ml_g = CatBoostClassifier(
                        thread_count=-1,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
                else:
                    self.ml_g = CatBoostRegressor(
                        thread_count=-1,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
        
        # If ml_g is still None and HAS_CATBOOST is True, it means data was not provided.
        # It will be initialized in fit().
        if self.relative_baseline_min < 0.0:
            raise ValueError("relative_baseline_min must be non-negative.")

    # --------- Helpers ---------
    def _check_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Check and prepare data for IRM.

        Ensures treatment is binary, confounders are present, and returns relevant arrays.

        Returns
        -------
        X : np.ndarray
            Confounders matrix.
        y : np.ndarray
            Outcome array.
        d : np.ndarray
            Treatment array.
        y_is_binary : bool
            Whether the outcome is binary.
        """
        df = self.data.get_df().copy()
        y = df[self.data.outcome.name].to_numpy(dtype=float)
        d = df[self.data.treatment.name].to_numpy()
        # Ensure binary 0/1
        if df[self.data.treatment.name].dtype == bool:
            d = d.astype(int)
        if not _is_binary(d):
            raise ValueError("Treatment must be binary 0/1 or boolean.")
        d = d.astype(int)

        x_cols = list(self.data.confounders)
        if len(x_cols) == 0:
            raise ValueError("CausalData must include non-empty confounders.")
        X = df[x_cols].to_numpy(dtype=float)

        y_is_binary = _is_binary(y)
        return X, y, d, y_is_binary

    def _get_weights(self, n: int, m_hat_adj: Optional[np.ndarray], d: np.ndarray, score: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weights for the IRM score.

        Parameters
        ----------
        n : int
            Number of observations.
        m_hat_adj : Optional[np.ndarray]
            Adjusted propensity scores.
        d : np.ndarray
            Treatment indicators.
        score : Optional[str], default None
            Target estimand. If None, uses self.score.

        Returns
        -------
        w : np.ndarray
            Weights for the outcome terms.
        w_bar : np.ndarray
            Weights for the representer terms.
        """
        if score is None:
            score = self.score
        score = score.upper()

        if self.weights is not None and score != "ATE":
            raise ValueError(f"weights are only supported for score='ATE', but got score='{score}'")

        # Standard ATE
        if score in {"ATE", "CATE"}:
            if self.weights is None:
                w = np.ones(n, dtype=float)
            elif isinstance(self.weights, np.ndarray):
                if self.weights.shape[0] != n:
                    raise ValueError("weights array must have shape (n,)")
                w = np.asarray(self.weights, dtype=float)
            elif isinstance(self.weights, dict):
                w = np.asarray(self.weights.get("weights"), dtype=float)
                if w.shape[0] != n:
                    raise ValueError("weights['weights'] must have shape (n,)")
            else:
                raise TypeError("weights must be None, np.ndarray, or dict")
            w_bar = w
            if isinstance(self.weights, dict) and "weights_bar" in self.weights:
                w_bar = np.asarray(self.weights["weights_bar"], dtype=float)
                if w_bar.ndim == 2:
                    # choose first repetition for now
                    w_bar = w_bar[:, 0]
        # ATTE requires m_hat
        elif score == "ATTE":
            if m_hat_adj is None:
                raise ValueError("m_hat required for ATTE weights computation")
            w = d.astype(float)
            w_bar = m_hat_adj.astype(float)
        else:
            raise ValueError("score must be 'ATE', 'ATTE' or 'CATE'")

        # Central weight normalization to ensure mean(w) == 1.
        # This ensures that both estimation and sensitivity components use the same scale.
        mean_w = float(np.mean(w))
        if mean_w > 1e-12:
            w = w / mean_w
            w_bar = w_bar / mean_w
        return w, w_bar

    def _normalize_ipw_terms(self, d: np.ndarray, m_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and optionally normalize IPW terms.

        Parameters
        ----------
        d : np.ndarray
            Treatment indicators.
        m_hat : np.ndarray
            Propensity scores.

        Returns
        -------
        h1 : np.ndarray
            IPW term for treated units (d / m_hat).
        h0 : np.ndarray
            IPW term for control units ((1 - d) / (1 - m_hat)).
        """
        # Compute IPW terms and optionally normalize to mean 1
        h1 = d / m_hat
        h0 = (1 - d) / (1 - m_hat)
        if self.normalize_ipw:
            h1_mean = np.mean(h1)
            h0_mean = np.mean(h0)
            # Avoid division by zero
            h1 = h1 / (h1_mean if h1_mean != 0 else 1.0)
            h0 = h0 / (h0_mean if h0_mean != 0 else 1.0)
        return h1, h0

    # --------- API ---------
    def fit(self, data: Optional[CausalData] = None) -> "IRM":
        """Fit nuisance models via cross-fitting.

        Parameters
        ----------
        data : Optional[CausalData], default None
            CausalData container. If None, uses self.data.

        Returns
        -------
        self : IRM
            Fitted estimator.
        """
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("Model must be provided with CausalData either in __init__ or in .fit(data_contracts).")
        X, y, d, y_is_binary = self._check_data()

        # Initialize default learners if not provided and data is now available
        if HAS_CATBOOST:
            if self.ml_m is None:
                self.ml_m = CatBoostClassifier(
                    thread_count=-1,
                    verbose=False,
                    allow_writing_files=False,
                    random_seed=self.random_state,
                )
            if self.ml_g is None:
                if y_is_binary:
                    self.ml_g = CatBoostClassifier(
                        thread_count=-1,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )
                else:
                    self.ml_g = CatBoostRegressor(
                        thread_count=-1,
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=self.random_state,
                    )

        if self.ml_g is None or self.ml_m is None:
            raise ValueError("ml_g and ml_m must be provided (either as defaults or in __init__).")
        # Cache for sensitivity analysis and effect calculation
        self._y = y.copy()
        self._d = d.copy()
        n = X.shape[0]

        # Enforce valid propensity model: must expose predict_proba when classifier
        if _safe_is_classifier(self.ml_m) and not hasattr(self.ml_m, "predict_proba"):
            raise ValueError("ml_m must support predict_proba() to produce valid propensity probabilities.")
        # For binary outcomes, require probabilistic outcome models when using classifiers
        if y_is_binary and _safe_is_classifier(self.ml_g) and not hasattr(self.ml_g, "predict_proba"):
            raise ValueError("Binary outcome: ml_g is a classifier but does not expose predict_proba(). Use a probabilistic classifier or calibrate it.")

        if self.n_rep != 1:
            raise NotImplementedError("IRM currently supports n_rep=1 only.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.trimming_rule not in {"truncate"}:
            raise ValueError("Only trimming_rule='truncate' is supported")

        g0_hat = np.full(n, np.nan, dtype=float)
        g1_hat = np.full(n, np.nan, dtype=float)
        m_hat = np.full(n, np.nan, dtype=float)
        folds = np.full(n, -1, dtype=int)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, d)):
            folds[test_idx] = i
            # Outcome models trained on respective treatment groups in the train fold
            X_tr, y_tr, d_tr = X[train_idx], y[train_idx], d[train_idx]
            X_te = X[test_idx]

            # g0
            model_g0 = clone(self.ml_g)
            mask0 = (d_tr == 0)
            if not np.any(mask0):
                raise RuntimeError("IRM: A CV fold has no controls in the training split. "
                                   "This violates the IRM nuisance definition. "
                                   "Consider reducing n_folds or increasing sample size.")
            else:
                X_g0, y_g0 = X_tr[mask0], y_tr[mask0]
            model_g0.fit(X_g0, y_g0)
            if y_is_binary and _safe_is_classifier(model_g0) and hasattr(model_g0, "predict_proba"):
                pred_g0 = model_g0.predict_proba(X_te)
                pred_g0 = pred_g0[:, 1] if pred_g0.ndim == 2 else pred_g0.ravel()
            else:
                pred_g0 = model_g0.predict(X_te)
            pred_g0 = np.asarray(pred_g0, dtype=float).ravel()
            if y_is_binary:
                pred_g0 = np.clip(pred_g0, 1e-12, 1 - 1e-12)
            g0_hat[test_idx] = pred_g0

            # g1
            model_g1 = clone(self.ml_g)
            mask1 = (d_tr == 1)
            if not np.any(mask1):
                raise RuntimeError("IRM: A CV fold has no treated units in the training split. "
                                   "This violates the IRM nuisance definition. "
                                   "Consider reducing n_folds or increasing sample size.")
            else:
                X_g1, y_g1 = X_tr[mask1], y_tr[mask1]
            model_g1.fit(X_g1, y_g1)
            if y_is_binary and _safe_is_classifier(model_g1) and hasattr(model_g1, "predict_proba"):
                pred_g1 = model_g1.predict_proba(X_te)
                pred_g1 = pred_g1[:, 1] if pred_g1.ndim == 2 else pred_g1.ravel()
            else:
                pred_g1 = model_g1.predict(X_te)
            pred_g1 = np.asarray(pred_g1, dtype=float).ravel()
            if y_is_binary:
                pred_g1 = np.clip(pred_g1, 1e-12, 1 - 1e-12)
            g1_hat[test_idx] = pred_g1

            # m
            model_m = clone(self.ml_m)
            model_m.fit(X_tr, d_tr)
            m_pred = _predict_prob_or_value(model_m, X_te, is_propensity=True)
            m_hat[test_idx] = m_pred

        # Trimming/clipping propensity
        if np.any(np.isnan(m_hat)) or np.any(np.isnan(g0_hat)) or np.any(np.isnan(g1_hat)):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        m_hat = _clip_propensity(m_hat, self.trimming_threshold)
        self.folds_ = folds

        self.g0_hat_ = g0_hat
        self.g1_hat_ = g1_hat
        self.m_hat_ = m_hat

        return self

    def estimate(
        self, score: str = "ATE", alpha: float = 0.05, diagnostic_data: bool = True
    ) -> CausalEstimate:
        """Compute treatment effects using stored nuisance predictions.

        Parameters
        ----------
        score : {"ATE", "ATTE", "CATE"}, default "ATE"
            Target estimand.
        alpha : float, default 0.05
            Significance level for intervals.
        diagnostic_data : bool, default True
            Whether to include diagnostic data_contracts in the result.

        Returns
        -------
        CausalEstimate
            Result container for the estimated effect.
        """
        check_is_fitted(self, attributes=["g0_hat_", "g1_hat_", "m_hat_"])
        score = score.upper()
        self.score = score

        y, d = self._y, self._d
        n = len(y)
        g0_hat, g1_hat, m_hat = self.g0_hat_, self.g1_hat_, self.m_hat_

        # Score elements
        u0 = y - g0_hat
        u1 = y - g1_hat
        h1, h0 = self._normalize_ipw_terms(d, m_hat)

        # weights
        w, w_bar = self._get_weights(n, m_hat, d, score=score)

        # psi elements
        psi_b = w * (g1_hat - g0_hat) + w_bar * (u1 * h1 - u0 * h0)

        # Jacobian E[psi_a] and point estimate
        if score == "ATE":
            psi_a = -np.ones(n)
        elif score == "ATTE":
            psi_a = -w
        else:
            # For CATE or custom scores, we use -w as well
            psi_a = -w

        J = float(np.mean(psi_a))

        if abs(J) < 1e-16:
            theta_hat = np.nan
            IF = np.zeros(n)
            se = np.nan
        else:
            theta_hat = -float(np.mean(psi_b) / J)
            # Moment residual psi = psi_b + psi_a * theta_hat
            psi_res = psi_b + psi_a * theta_hat
            # Influence function IF = -psi_res / J
            IF = -psi_res / J
            var = float(np.var(IF, ddof=1)) / n
            se = float(np.sqrt(max(var, 0.0)))

        t_stat = theta_hat / se if se > 0 else np.nan
        pval = 2 * (1 - norm.cdf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
        z = norm.ppf(1 - alpha / 2.0)
        ci_low, ci_high = theta_hat - z * se, theta_hat + z * se

        # Cache for sensitivity and other diagnostics
        self.coef_ = np.array([theta_hat])
        self.se_ = np.array([se])
        self.psi_ = IF  # Store the influence function used for inference
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b

        # Baseline mean for relative effect (weighted to match estimand)
        mu_c = float(np.mean(w * g0_hat))
        mu_c_var = float(np.var(w * g0_hat, ddof=1)) / n if n > 1 else 0.0
        mu_c_se = float(np.sqrt(max(mu_c_var, 0.0)))
        tau_rel = np.nan
        ci_low_rel = np.nan
        ci_high_rel = np.nan
        se_rel = np.nan

        baseline_too_small = abs(mu_c) < self.relative_baseline_min
        baseline_low_signal = np.isfinite(mu_c_se) and mu_c_se > 0.0 and abs(mu_c) < z * mu_c_se

        if np.isfinite(mu_c) and not (baseline_too_small or baseline_low_signal):
            tau_rel = 100.0 * theta_hat / mu_c
            # Delta method for relative effect: tau_rel = 100 * theta / mu_c
            IF_mu = w * g0_hat - mu_c
            with np.errstate(divide="ignore", invalid="ignore"):
                IF_rel = 100.0 * (IF / mu_c - (theta_hat * IF_mu) / (mu_c ** 2))
            var_rel = float(np.var(IF_rel, ddof=1)) / n
            se_rel = float(np.sqrt(max(var_rel, 0.0)))
            ci_low_rel = tau_rel - z * se_rel
            ci_high_rel = tau_rel + z * se_rel
            if ci_low_rel > ci_high_rel:
                ci_low_rel, ci_high_rel = ci_high_rel, ci_low_rel
        else:
            reasons = []
            if not np.isfinite(mu_c):
                reasons.append("is not finite")
            if baseline_too_small:
                reasons.append(f"is below relative_baseline_min={self.relative_baseline_min:.3e}")
            if baseline_low_signal:
                reasons.append(f"is within {z:.2f} SE of 0 (SE={mu_c_se:.3e})")
            reason_str = "; ".join(reasons) if reasons else "is too small"
            warnings.warn(
                f"Relative effect baseline |mu_c|={abs(mu_c):.3e} {reason_str}. "
                "Relative estimates are set to NaN.",
                RuntimeWarning,
            )
        self.mu_c_ = mu_c
        self.se_relative_ = np.array([se_rel])
        self.confint_relative_ = np.array([[ci_low_rel, ci_high_rel]])

        treatment_mean = float(np.mean(y[d == 1])) if np.any(d == 1) else np.nan
        control_mean = float(np.mean(y[d == 0])) if np.any(d == 0) else np.nan

        diag = None
        if diagnostic_data:
            # Calculate sensitivity elements
            sens_elements = self._sensitivity_element_est(
                y=y, d=d, g0=g0_hat, g1=g1_hat, m_hat=m_hat, w=w, w_bar=w_bar, psi=IF
            )

            diag = UnconfoundednessDiagnosticData(
                m_hat=m_hat,
                d=d,
                y=y,
                x=self.data.get_df()[list(self.data.confounders)].to_numpy(dtype=float),
                g0_hat=g0_hat,
                g1_hat=g1_hat,
                psi_b=psi_b,
                folds=self.folds_,
                trimming_threshold=self.trimming_threshold,
                score=score,
                **sens_elements
            )
            diag._model = self

        results = CausalEstimate(
            estimand=score,
            model="IRM",
            model_options={
                "n_folds": self.n_folds,
                "n_rep": self.n_rep,
                "normalize_ipw": self.normalize_ipw,
                "trimming_rule": self.trimming_rule,
                "trimming_threshold": self.trimming_threshold,
                "random_state": self.random_state,
                "std_error": se,
                "t_stat": t_stat,
            },
            value=theta_hat,
            ci_upper_absolute=ci_high,
            ci_lower_absolute=ci_low,
            value_relative=tau_rel,
            ci_upper_relative=ci_high_rel,
            ci_lower_relative=ci_low_rel,
            alpha=alpha,
            p_value=pval,
            is_significant=bool(pval < alpha) if np.isfinite(pval) else False,
            n_treated=int(np.sum(d == 1)),
            n_control=int(np.sum(d == 0)),
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            outcome=self.data.outcome.name,
            treatment=self.data.treatment.name,
            confounders=list(self.data.confounders),
            time=datetime.now().strftime("%Y-%m-%d"),
            diagnostic_data=diag,
        )

        # Update internal state
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b
        self.psi_ = IF
        self.coef_ = np.array([theta_hat])
        self.se_ = np.array([se])
        self.t_stat_ = np.array([t_stat])
        self.pval_ = np.array([pval])
        self.confint_ = np.array([[ci_low, ci_high]])
        self.summary_ = results.summary()

        return results

    @property
    def diagnostics_(self) -> Dict[str, Any]:
        """Return diagnostic data.

        Returns
        -------
        dict
            Dictionary containing 'm_hat', 'g0_hat', 'g1_hat', and 'folds'.
        """
        check_is_fitted(self, attributes=["m_hat_", "g0_hat_", "g1_hat_"])
        return {
            "m_hat": self.m_hat_,
            "g0_hat": self.g0_hat_,
            "g1_hat": self.g1_hat_,
            "folds": self.folds_,
        }

    # Convenience properties
    @property
    def coef(self) -> np.ndarray:
        """Return the estimated coefficient.

        Returns
        -------
        np.ndarray
            The estimated coefficient.
        """
        check_is_fitted(self, attributes=["coef_"])
        return self.coef_

    @property
    def se(self) -> np.ndarray:
        """Return the standard error of the estimate.

        Returns
        -------
        np.ndarray
            The standard error.
        """
        check_is_fitted(self, attributes=["se_"])
        return self.se_

    @property
    def pvalues(self) -> np.ndarray:
        """Return the p-values for the estimate.

        Returns
        -------
        np.ndarray
            The p-values.
        """
        check_is_fitted(self, attributes=["pval_"])
        return self.pval_

    @property
    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of the results.

        Returns
        -------
        pd.DataFrame
            The results summary.
        """
        check_is_fitted(self, attributes=["summary_"])
        return self.summary_

    @property
    def orth_signal(self) -> np.ndarray:
        """Return the cross-fitted orthogonal signal (psi_b).

        Returns
        -------
        np.ndarray
            The orthogonal signal.
        """
        check_is_fitted(self, attributes=["psi_b_"])
        return self.psi_b_

    def gate(self, groups: pd.DataFrame | pd.Series, alpha: float = 0.05) -> BLP:
        """
        Estimate Group Average Treatment Effects via BLP on orthogonal signal.

        Parameters
        ----------
        groups : pd.DataFrame or pd.Series
            Group indicators or labels.
            - If a single column (Series or 1-col DataFrame) with non-boolean values,
              it is treated as categorical labels and one-hot encoded.
            - If multiple columns or boolean/int indicators, it is used as the basis directly.
        alpha : float
            Significance level for intervals (passed to BLP).

        Returns
        -------
        BLP
            Fitted Best Linear Predictor model.
        """
        check_is_fitted(self, attributes=["psi_b_"])

        if isinstance(groups, pd.Series):
            groups = groups.to_frame()

        # Prepare basis
        if groups.shape[1] == 1:
            col = groups.iloc[:, 0]
            # If single column is not boolean, assume it's categorical labels -> one-hot encode
            # Even if it is boolean, get_dummies creates False/True columns which is a valid partition
            # We use prefix to ensure unique column names
            basis = pd.get_dummies(col, prefix=col.name, dtype=int)
        else:
            # Assume multiple columns are already indicators (dummy basis)
            basis = groups.astype(int)

        # Instantiate and fit BLP using the orthogonal signal
        # We use the existing BLP class from causalis.scenarios.cate.blp
        blp_model = BLP(
            orth_signal=self.orth_signal,
            basis=basis,
            is_gate=True
        )
        # BLP.fit() uses HC0 covariance by default, which is correct for DML
        blp_model.fit()
        
        return blp_model

    # --------- Sensitivity ---------
    def _sensitivity_element_est(
        self,
        y: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        g0: Optional[np.ndarray] = None,
        g1: Optional[np.ndarray] = None,
        m_hat: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        w_bar: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute elements needed for sensitivity bias bounds.

        Mirrors a standard IRM sensitivity element computation using fitted nuisances.
        Requires fit() to have been called.

        Parameters
        ----------
        y : Optional[np.ndarray], default None
            Outcomes. If None, uses cached outcomes.
        d : Optional[np.ndarray], default None
            Treatment indicators. If None, uses cached indicators.
        g0 : Optional[np.ndarray], default None
            Outcome predictions under control. If None, uses fitted g0_hat_.
        g1 : Optional[np.ndarray], default None
            Outcome predictions under treatment. If None, uses fitted g1_hat_.
        m_hat : Optional[np.ndarray], default None
            Propensity scores. If None, uses fitted m_hat_.
        w : Optional[np.ndarray], default None
            Outcome weights. If None, computed internally.
        w_bar : Optional[np.ndarray], default None
            Representer weights. If None, computed internally.
        psi : Optional[np.ndarray], default None
            Score values. If None, uses fitted psi_.

        Returns
        -------
        dict
            Sensitivity elements including 'sigma2', 'nu2', 'psi_sigma2', 'psi_nu2',
            'riesz_rep', 'm_alpha', and 'psi'.
        """
        if any(getattr(self, attr) is None for attr in ["g0_hat_", "g1_hat_", "m_hat_"]):
            raise RuntimeError("IRM model must be fitted before sensitivity analysis.")

        if y is None:
            y = self._y
        if d is None:
            d = self._d

        if y is None or d is None:
            # fallback to current data_contracts
            df = self.data.get_df()
            y = df[self.data.outcome.name].to_numpy(dtype=float)
            d = df[self.data.treatment.name].to_numpy(dtype=int)

        if m_hat is None:
            m_hat = np.asarray(self.m_hat_, dtype=float)
        if g0 is None:
            g0 = np.asarray(self.g0_hat_, dtype=float)
        if g1 is None:
            g1 = np.asarray(self.g1_hat_, dtype=float)

        if w is None or w_bar is None:
            w, w_bar = self._get_weights(n=len(y), m_hat_adj=m_hat, d=d)

        if psi is None:
            psi = self.psi_

        # sigma^2
        sigma2_score_element = np.square(y - d * g1 - (1.0 - d) * g0)
        sigma2 = float(np.mean(sigma2_score_element))
        psi_sigma2 = sigma2_score_element - sigma2

        # Riesz representer and m_alpha
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_m = 1.0 / m_hat
            inv_1m = 1.0 / (1.0 - m_hat)
        # If IPW terms are normalized in the score, mirror that normalization here.
        if self.normalize_ipw:
            h1_raw = d * inv_m
            h0_raw = (1.0 - d) * inv_1m
            c1 = float(np.mean(h1_raw))
            c0 = float(np.mean(h0_raw))
            if not (np.isfinite(c1) and abs(c1) > 1e-12):
                c1 = 1.0
            if not (np.isfinite(c0) and abs(c0) > 1e-12):
                c0 = 1.0
            inv_m = inv_m / c1
            inv_1m = inv_1m / c0
        m_alpha = (w_bar ** 2) * (inv_m + inv_1m)
        rr = w_bar * (d * inv_m - (1.0 - d) * inv_1m)

        nu2_score_element = 2.0 * m_alpha - np.square(rr)
        nu2 = float(np.mean(nu2_score_element))
        psi_nu2 = nu2_score_element - nu2

        return {
            "sigma2": sigma2,
            "nu2": nu2,
            "psi_sigma2": psi_sigma2,
            "psi_nu2": psi_nu2,
            "riesz_rep": rr,
            "m_alpha": m_alpha,
            "psi": psi,
        }

    def sensitivity_analysis(self, r2_y: float, r2_d: float, rho: float = 1.0, H0: float = 0.0, alpha: float = 0.05) -> "IRM":
        """Compute a sensitivity analysis following Chernozhukov et al. (2022).
        
        Parameters
        ----------
        r2_y : float
            Sensitivity parameter for outcome equation (R^2 form, R_Y^2; converted to odds form internally).
        r2_d : float
            Sensitivity parameter for treatment equation (R^2 form, R_D^2).
        rho : float, default 1.0
            Correlation between unobserved components.
        H0 : float, default 0.0
            Null hypothesis for robustness values.
        alpha : float, default 0.05
            Significance level for CI bounds.
        """
        from causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity import (
            sensitivity_analysis as sa_fn,
            get_sensitivity_summary
        )
        check_is_fitted(self, attributes=["coef_", "se_", "psi_"])

        # Execute sensitivity analysis using the centralized module logic
        res = sa_fn(self, r2_y=r2_y, r2_d=r2_d, rho=rho, H0=H0, alpha=alpha)

        # Cache the summary string for display
        self.sensitivity_summary = get_sensitivity_summary({"model": self, "bias_aware": res})

        return self

    def confint(self, alpha: float = 0.05) -> pd.DataFrame:
        """Compute confidence intervals for the estimated coefficient.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        pd.DataFrame
            DataFrame with confidence intervals.
        """
        check_is_fitted(self, attributes=["coef_", "se_"])
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        z = norm.ppf(1 - alpha / 2.0)
        ci_low = self.coef_[0] - z * self.se_[0]
        ci_high = self.coef_[0] + z * self.se_[0]
        return pd.DataFrame(
            {f"{alpha/2*100:.1f} %": [ci_low], f"{(1-alpha/2)*100:.1f} %": [ci_high]},
            index=[self.data.treatment.name],
        )
