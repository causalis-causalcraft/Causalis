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

from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.causal_diagnostic_data import MultiUnconfoundednessDiagnosticData
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
    # нужно подкорректировать
    if _safe_is_classifier(model) and hasattr(model, "predict_proba"):
        res = model.predict_proba(X)
    else:
        res = model.predict(X)

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


class MultiTreatmentIRM(BaseEstimator):
    """Interactive Regression Model with multi_uncofoundedness (Multi treatment IRM) with DoubleML-style cross-fitting using CausalData.
       Model supports >= 2 treatments.

        Parameters
        ----------
        data : MultiCausalData
            Data container with outcome, binary treatment (0/1), and confounders.
        ml_g : estimator
            Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
        ml_m : classifier
            Learner for E[D|X] (generelized propensity score). Must support predict_proba() or predict() in (0,1).
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
        random_state : Optional[int], default None
            Random seed for fold creation.
        """
    def __init__(self,
                 data: Optional[MultiCausalData] = None,
                 ml_g: Any = None,
                 ml_m: Any = None,
                 *,
                 n_folds: int = 5,
                 n_rep: int = 1,
                 normalize_ipw: bool = False,
                 trimming_rule: str = "truncate",
                 trimming_threshold: float = 1e-2,
                 random_state: Optional[int] = None
    ):
        self.data = data
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.n_folds = int(n_folds)
        self.n_rep = int(n_rep)
        self.score = "ATE"
        self.normalize_ipw = bool(normalize_ipw)
        self.trimming_rule = str(trimming_rule)
        self.trimming_threshold = float(trimming_threshold)
        self.random_state = random_state

        # Initialize default learners if not provided
        if HAS_CATBOOST:
            if self.ml_m is None:
                self.ml_m = CatBoostClassifier(
                    loss_function="MultiClass",
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

    def _check_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = self.data.get_df().copy()
        y = df[self.data.outcome.name].to_numpy(dtype=float)
        d = df[self.data.treatments.columns].to_numpy()

        n_treatments_ = len(self.data.treatments.columns)
        if n_treatments_ < 3:
            raise ValueError("Need at least 2 treatments and 1 control variations")

        x_cols = list(self.data.confounders_names)
        if len(x_cols) == 0:
            raise ValueError("MultiCausalData must include non-empty confounders.")
        X = df[x_cols].to_numpy(dtype=float)
        y_is_binary = _is_binary(y)

        return X, y, d, y_is_binary, n_treatments_

    def _normalize_ipw_terms(self, d: np.ndarray, m_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = np.asarray(d, dtype=float)
        m_hat = np.asarray(m_hat, dtype=float)

        h = d / m_hat

        if self.normalize_ipw:
            h_mean = h.mean(axis=0, keepdims=True)  # (1, D)
            h = h / np.where(h_mean != 0, h_mean, 1.0)

        return h

    def fit(self, data: Optional[CausalData] = None) -> "MultiTreatmentIRM":
        if data is not None:
            self.data = data
        if self.data is None:
            raise ValueError("Model must be provided with MultiCausalData either in __init__ or in .fit(data_contracts).")
        X, y, d, y_is_binary, n_treatments = self._check_data()

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

        self.n_treatments = n_treatments

        n = X.shape[0]

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

        g_hat = np.full((n, self.n_treatments), np.nan, dtype=float)
        m_hat = np.full((n, self.n_treatments), np.nan, dtype=float)
        folds = np.full(n, -1, dtype=int)

        d_strat = d.argmax(axis=1)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, d_strat)):
            folds[test_idx] = i
            X_tr, y_tr, d_tr = X[train_idx], y[train_idx], d_strat[train_idx]
            X_te = X[test_idx]

            # m model
            model_m = clone(self.ml_m)
            model_m.fit(X_tr, d_tr)
            m_pred = _predict_prob_or_value(model_m, X_te, is_propensity=True)
            m_hat[test_idx] = m_pred

            # g models
            for k in range(self.n_treatments):
                model_g = clone(self.ml_g)

                # Here we assume that the lack of treatment is set in a separate column t_0
                treatment_mask_train = (d[train_idx, k] == 1)
                train_treatment_idx = train_idx[treatment_mask_train]

                if not np.any(treatment_mask_train):
                    raise RuntimeError("IRM: A CV fold has no controls in the training split. "
                                       "This violates the IRM nuisance definition. "
                                       "Consider reducing n_folds or increasing sample size.")
                else:
                    X_g, y_g = X[train_treatment_idx], y[train_treatment_idx]

                model_g.fit(X_g, y_g)

                if y_is_binary and _safe_is_classifier(model_g) and hasattr(model_g, "predict_proba"):
                    pred_g = model_g.predict_proba(X_te)
                    pred_g = pred_g[:, 1] if pred_g.ndim == 2 else pred_g.ravel()
                else:
                    pred_g = model_g.predict(X_te)
                pred_g = np.asarray(pred_g, dtype=float).ravel()
                if y_is_binary:
                    pred_g = np.clip(pred_g, 1e-12, 1 - 1e-12)
                g_hat[test_idx, k] = pred_g

        if np.any(np.isnan(m_hat)) or np.any(np.isnan(g_hat)):
            raise RuntimeError("Cross-fitted predictions contain NaN values.")
        m_hat = _clip_propensity(m_hat, self.trimming_threshold)

        self.folds_ = folds

        self.g_hat_ = g_hat
        self.m_hat_ = m_hat

        return self

    def estimate(
            self, score: str = "ATE", alpha: float = 0.05, diagnostic_data: bool = True
    ) -> CausalEstimate:
        check_is_fitted(self, attributes=["g_hat_", "m_hat_"])
        score = score.upper()
        self.score = score

        y, d = self._y, self._d
        y = y.reshape(-1, 1)
        n = len(y)
        g_hat, m_hat = self.g_hat_, self.m_hat_

        # Score elements

        u = y - g_hat
        h = self._normalize_ipw_terms(d, m_hat)

        # psi elements
        psi_b = (
                (g_hat[:, 1:] - g_hat[:, [0]])
                + (u[:, 1:] * h[:, 1:])
                - (u[:, [0]] * h[:, [0]])
        )

        # Jacobian E[psi_a] and point estimate
        if score == "ATE":
            psi_a = -np.ones(n)
        else:
            raise RuntimeError("Only ATE is supported")

        J = float(np.mean(psi_a))

        if abs(J) < 1e-16:
            theta_hat = np.nan
            IF = np.zeros(n)
            se = np.nan
        else:
            theta_hat = -np.mean(psi_b, axis=0) / J

            psi_res = psi_b + psi_a[:, None] * theta_hat[None, :]
            IF = -psi_res / J

            var = np.var(IF, axis=0, ddof=1) / n
            se = np.sqrt(np.maximum(var, 0.0))

        t_stat = np.where(se > 0, theta_hat / se, np.nan)
        pval = np.full_like(t_stat, np.nan, dtype=float)
        mask = np.isfinite(t_stat)
        pval[mask] = 2 * (1 - norm.cdf(np.abs(t_stat[mask])))
        z = norm.ppf(1 - alpha / 2.0)
        ci_low, ci_high = theta_hat - z * se, theta_hat + z * se

        # Cache for sensitivity and other diagnostics
        self.coef_ = np.array(theta_hat)
        self.se_ = np.array(se)
        self.psi_ = IF  # Store the influence function used for inference
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b

        # Baseline mean for corrected relative effect
        psi_mu_c = (g_hat[:, [0]] + u[:, [0]] * h[:, [0]]).ravel()  # (n,)
        mu_c = float(psi_mu_c.mean())

        se_mu_c = psi_mu_c.std(ddof=1) / np.sqrt(n)
        psi_mu_c_centered = psi_mu_c - mu_c
        psi_b_centered = psi_b - theta_hat  # broadcasting -> (n, D-1)
        cov_theta_mu_c = (psi_b_centered * psi_mu_c_centered[:, None]).mean(axis=0) / n  # (D-1,)

        if mu_c == 0:
            tau_rel = np.full_like(theta_hat, np.nan, dtype=float)
            ci_low_rel = np.full_like(theta_hat, np.nan, dtype=float)
            ci_high_rel = np.full_like(theta_hat, np.nan, dtype=float)
        else:
            tau_rel = 100.0 * theta_hat / mu_c

            d_theta = 100.0 / mu_c
            d_mu = -100.0 * theta_hat / (mu_c ** 2)

            var_rel = (d_theta ** 2) * (se ** 2) + (d_mu ** 2) * (se_mu_c ** 2) + 2.0 * d_theta * d_mu * cov_theta_mu_c
            se_rel = np.sqrt(np.maximum(var_rel, 0.0))

            ci_low_rel = tau_rel - z * se_rel
            ci_high_rel = tau_rel + z * se_rel


        diag = None
        if diagnostic_data:
            # Calculate sensitivity elements
            sens_elements = self._sensitivity_element_est(
                y=y, d=d, g_hat=g_hat, m_hat=m_hat, psi=IF
            )

            diag = MultiUnconfoundednessDiagnosticData(
                m_hat=m_hat,
                d=d,
                y=y,
                x=self.data.get_df()[list(self.data.confounders_names)].to_numpy(dtype=float),
                g_hat=g_hat,
                psi_b=psi_b,
                folds=self.folds_,
                trimming_threshold=self.trimming_threshold,
                score=score,
                **sens_elements
            )
            diag._model = self

        results = MultiCausalEstimate(
            estimand=score,
            model="MultiTreatmentIRM",
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
            is_significant=np.where(np.isfinite(pval), pval < alpha / self.n_treatments, False),
            n_treated=int(np.sum(d == 1)),
            n_control=int(np.sum(d == 0)),
            outcome=self.data.outcome.name,
            treatment=list(self.data.treatments.columns),
            confounders=list(self.data.confounders_names),
            time=datetime.now().strftime("%Y-%m-%d"),
            diagnostic_data=diag,
        )

        # Update internal state
        self.psi_a_ = psi_a
        self.psi_b_ = psi_b
        self.psi_ = IF
        self.coef_ = np.array(theta_hat)
        self.se_ = np.array(se)
        self.t_stat_ = np.array(t_stat)
        self.pval_ = np.array(pval)
        self.confint_ = np.array([ci_low, ci_high])
        self.summary_ = results.summary()

        return results

    @property
    def diagnostics_(self) -> Dict[str, Any]:
        """Return diagnostic data.

                Returns
                -------
                dict
                    Dictionary containing 'm_hat', 'g_hat' and 'folds'.
                """
        check_is_fitted(self, attributes=["m_hat_", "g_hat_"])
        return {
            "m_hat": self.m_hat_,
            "g_hat": self.g_hat_,
            "folds": self.folds_,
        }

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

    def _sensitivity_element_est(
            self,
            y: Optional[np.ndarray] = None,
            d: Optional[np.ndarray] = None,
            g_hat: Optional[np.ndarray] = None,
            m_hat: Optional[np.ndarray] = None,
            psi: Optional[np.ndarray] = None,
    ) -> dict:
        if any(getattr(self, attr, None) is None for attr in ["g_hat_", "m_hat_"]):
            raise RuntimeError("Model must be fitted before sensitivity analysis.")

        # --- fetch data ---
        if y is None:
            y = getattr(self, "_y", None)
        if d is None:
            d = getattr(self, "_d", None)

        if y is None or d is None:
            df = self.data.get_df()
            y = df[self.data.outcome.name].to_numpy(dtype=float)
            d = df[self.data.treatment.name].to_numpy()

        # --- get fitted nuisances ---
        if m_hat is None:
            m_hat = np.asarray(self.m_hat_, dtype=float)
        else:
            m_hat = np.asarray(m_hat, dtype=float)

        if g_hat is None:
            g_hat = np.asarray(self.g_hat_, dtype=float)
        else:
            g_hat = np.asarray(g_hat, dtype=float)

        if psi is None:
            psi = getattr(self, "psi_", None)
        psi = None if psi is None else np.asarray(psi, dtype=float)

        y = np.asarray(y, dtype=float).reshape(-1)
        d = np.asarray(d)

        # --- shape checks / normalization ---
        n = y.shape[0]

        # ожидаем one-hot (n, K)
        if d.ndim != 2:
            raise ValueError(
                "Expected d as one-hot matrix of shape (n, K) for multi-treatment. "
                "If you store d as labels (n,), tell me and I'll add conversion."
            )
        if d.shape[0] != n:
            raise ValueError("y and d must have same number of rows.")
        if m_hat.shape[0] != n or g_hat.shape[0] != n:
            raise ValueError("m_hat and g_hat must have same number of rows as y.")
        if m_hat.shape[1] != d.shape[1] or g_hat.shape[1] != d.shape[1]:
            raise ValueError("d, m_hat, g_hat must have same number of treatments (columns).")

        K = d.shape[1]
        if K < 2:
            raise ValueError("Need at least two treatments for pairwise contrasts 0 vs k.")

        g_obs = np.sum(d * g_hat, axis=1)
        sigma2_score_element = (y - g_obs) ** 2
        sigma2 = float(np.mean(sigma2_score_element))
        psi_sigma2 = sigma2_score_element - sigma2

        w_bar = np.ones(n, dtype=float)

        inv_p = 1.0 / m_hat                               # (n, K)

        d0 = d[:, [0]]                                          # (n, 1)
        inv_p0 = inv_p[:, [0]]                                  # (n, 1)

        dk = d[:, 1:]                                           # (n, K-1)
        inv_pk = inv_p[:, 1:]                                   # (n, K-1)

        rr = w_bar[:, None] * (dk * inv_pk - d0 * inv_p0)        # (n, K-1)

        m_alpha = (w_bar[:, None] ** 2) * (inv_pk + inv_p0)      # (n, K-1)

        nu2_score_element = 2.0 * m_alpha - rr ** 2              # (n, K-1)
        nu2 = np.mean(nu2_score_element, axis=0)                 # (K-1,)
        psi_nu2 = nu2_score_element - nu2[None, :]               # (n, K-1)

        if psi is not None:
            if psi.shape[0] != n:
                raise ValueError("psi must have same number of rows as y.")
            if psi.ndim == 1 and (K - 1) == 1:
                psi = psi.reshape(-1, 1)
            if psi.ndim != 2 or psi.shape[1] != (K - 1):
                raise ValueError(f"psi must have shape (n, K-1) = (n, {K-1}).")

        return {
            "sigma2": sigma2,
            "psi_sigma2": psi_sigma2,
            "nu2": nu2,
            "psi_nu2": psi_nu2,
            "riesz_rep": rr,
            "m_alpha": m_alpha,
            "psi": psi,
        }

    def sensitivity_analysis(self, cf_y: float, r2_d: float, rho: float = 1.0, H0: float = 0.0,
                             alpha: float = 0.05) -> "MultiTreatmentIRM":
        from causalis.scenarios.multi_uncofoundedness.refutation.unconfoundedness.sensitivity import (
            sensitivity_analysis as sa_fn,
            get_sensitivity_summary
        )

        res = sa_fn(self, cf_y=cf_y, r2_d=r2_d, rho=rho, H0=H0, alpha=alpha)

        self.sensitivity_summary = get_sensitivity_summary({"model": self, "bias_aware": res})

        return self

    def confint(self) -> pd.DataFrame:
        check_is_fitted(self, attributes=["confint_"])

        return self.confint_
