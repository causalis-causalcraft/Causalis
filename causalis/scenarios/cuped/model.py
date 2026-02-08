from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List, Literal

import numpy as np

from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import CUPEDDiagnosticData

try:
    import statsmodels.api as sm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "CUPEDModel requires statsmodels. Install with: pip install statsmodels"
    ) from e


class CUPEDModel:
    """
    CUPED estimator for ATE/ITT in randomized experiments.

    Fits an outcome regression with pre-treatment covariates (always centered)
    using the Lin (2013) fully interacted adjustment:

        Y ~ 1 + D + X^c + D * X^c

    The reported effect is the coefficient on D, with robust covariance as requested.
    This specification ensures the coefficient on D is the ATE/ITT even if the
    treatment effect is heterogeneous with respect to covariates.

    Parameters
    ----------
    cov_type : str, default="HC3"
        Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
        Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    strict_binary_treatment : bool, default=True
        If True, require treatment to be binary {0,1}.
    use_t : bool, default=True
        Passed to statsmodels `.fit(..., use_t=use_t)`. If False, inference is based on
        normal approximation (common asymptotic choice for robust covariances).

    Notes
    -----
    - Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
    - The Lin (2013) specification is recommended as a robust regression-adjustment default
      in RCTs.
    """

    def __init__(
        self,
        cov_type: str = "HC3",
        alpha: float = 0.05,
        strict_binary_treatment: bool = True,
        use_t: bool = True,
    ) -> None:
        self.cov_type = str(cov_type)
        self.alpha = float(alpha)
        self.center_covariates = True
        self.strict_binary_treatment = bool(strict_binary_treatment)
        self.adjustment: Literal["lin"] = "lin"
        self.use_t = bool(use_t)

        self._is_fitted: bool = False
        self._result: Any = None
        self._result_naive: Any = None
        self._covariate_names: List[str] = []
        self._p: int = 0  # number of covariates used
        self._data: Optional[CausalData] = None

    def fit(self, data: CausalData, covariates: Optional[Sequence[str]] = None) -> CUPEDModel:
        """
        Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.

        Parameters
        ----------
        data : CausalData
            Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
        covariates : sequence of str, optional
            Subset of `data_contracts.confounders_names` to use as CUPED covariates.
            If None, uses all confounders from the object.

        Returns
        -------
        CUPEDModel
            Fitted estimator.

        Raises
        ------
        ValueError
            If requested covariates are missing, not in `data_contracts.confounders_names`,
            or treatment is not binary when `strict_binary_treatment=True`.
        """
        df = data.df
        y_name = data.outcome_name
        t_name = data.treatment_name

        # Choose covariates used for adjustment
        if covariates is None:
            x_names = list(data.confounders_names)
        else:
            covariates = list(covariates)
            missing = [c for c in covariates if c not in df.columns]
            if missing:
                raise ValueError(f"CUPED covariates not found in data_contracts.df: {missing}")
            not_in_contract = [c for c in covariates if c not in set(data.confounders_names)]
            if not_in_contract:
                raise ValueError(
                    "CUPED covariates must be a subset of data_contracts.confounders_names; "
                    f"not allowed: {not_in_contract}"
                )
            x_names = covariates

        y = df[y_name].to_numpy(dtype=float)
        d = df[t_name].to_numpy(dtype=float)

        if self.strict_binary_treatment:
            uniq = np.unique(d)
            uniq_r = set(np.round(uniq, 12).tolist())
            if not (len(uniq_r) == 2 and uniq_r.issubset({0.0, 1.0})):
                raise ValueError(
                    "Treatment must be binary {0,1} when strict_binary_treatment=True. "
                    f"Found unique values: {uniq.tolist()}"
                )

        n = len(y)

        # Build covariate matrix
        if len(x_names) > 0:
            X = df[x_names].to_numpy(dtype=float)
            Xc = X - X.mean(axis=0, keepdims=True)
            p = Xc.shape[1]
        else:
            Xc = np.zeros((n, 0), dtype=float)
            p = 0

        # Design matrix: Lin interacted specification [1, D, Xc, D*Xc]
        design_parts = [np.ones((n, 1), dtype=float), d.reshape(-1, 1)]
        if p > 0:
            design_parts.append(Xc)
            DXc = Xc * d.reshape(-1, 1)  # elementwise interaction
            design_parts.append(DXc)

        design = np.column_stack(design_parts)

        # Fit adjusted model with requested covariance estimator
        model = sm.OLS(y, design)
        self._result = model.fit(cov_type=self.cov_type, use_t=self.use_t)

        # Fit naive model: Y ~ 1 + D
        model_naive = sm.OLS(y, np.column_stack([np.ones(n, dtype=float), d]))
        self._result_naive = model_naive.fit(cov_type=self.cov_type, use_t=self.use_t)

        self._covariate_names = x_names
        self._p = p
        self._data = data
        self._is_fitted = True
        return self

    def estimate(self, alpha: Optional[float] = None, diagnostic_data: bool = True) -> CausalEstimate:
        """
        Return the adjusted ATE/ITT estimate and inference.

        Parameters
        ----------
        alpha : float, optional
            Override the instance significance level for confidence intervals.
        diagnostic_data : bool, default True
            Whether to include diagnostic data_contracts in the result.

        Returns
        -------
        CausalEstimate
            A results object containing effect estimates and inference.
        """
        self._require_fitted()

        a = float(self.alpha if alpha is None else alpha)

        # Coef index: 0 intercept, 1 treatment, then covariates / interactions
        tau = float(self._result.params[1])
        se = float(self._result.bse[1])
        t_stat = float(self._result.tvalues[1])
        p_value = float(self._result.pvalues[1])

        ci = self._result.conf_int(alpha=a)
        ci_low = float(ci[1, 0])
        ci_high = float(ci[1, 1])

        # CI relative calculation
        # self._result.model.endog is Y, self._result.model.exog[:, 1] is D
        y_internal = self._result.model.endog
        d_internal = self._result.model.exog[:, 1]
        mu_t = float(np.mean(y_internal[d_internal == 1]))
        mu_c = float(np.mean(y_internal[d_internal == 0]))

        if mu_c != 0:
            tau_rel = (tau / mu_c) * 100
            ci_low_rel = (ci_low / mu_c) * 100
            ci_high_rel = (ci_high / mu_c) * 100
        else:
            tau_rel = np.nan
            ci_low_rel = np.nan
            ci_high_rel = np.nan

        diag = None
        if diagnostic_data:
            ate_naive = float(self._result_naive.params[1])
            se_naive = float(self._result_naive.bse[1])

            # Variance reduction calculation.
            # We use non-robust SEs for this metric because they directly reflect the reduction in 
            # residual variance due to covariates. Robust SEs (like HC3) might increase 
            # if heteroscedasticity is detected in the adjusted model, even if variance is reduced.
            if self.cov_type == "nonrobust":
                se_naive_nonrobust = float(self._result_naive.bse[1])
                se_adjusted_nonrobust = float(self._result.bse[1])
            else:
                # Statsmodels OLS results have 'ssr' (sum of squared residuals) and 'df_resid'
                # Standard Error for OLS: sqrt( (SSR/df_resid) * (X'X)^-1 [j,j] )
                # But even simpler, we can just use the scale (sigma^2) which is SSR/df_resid
                # and (X'X)^-1 is available in normalized_cov_params.
                
                # Naive
                scale_naive = self._result_naive.scale
                var_tau_naive = scale_naive * self._result_naive.normalized_cov_params[1, 1]
                se_naive_nonrobust = np.sqrt(var_tau_naive)
                
                # Adjusted
                scale_adj = self._result.scale
                var_tau_adj = scale_adj * self._result.normalized_cov_params[1, 1]
                se_adjusted_nonrobust = np.sqrt(var_tau_adj)

            if se_naive_nonrobust > 0:
                var_red = 1.0 - (se_adjusted_nonrobust**2) / (se_naive_nonrobust**2)
                var_red_pct = float(100.0 * var_red)
            else:
                var_red_pct = np.nan

            p = self._p
            if p == 0:
                beta_cov = np.zeros((0,), dtype=float)
                gamma_cov = np.zeros((0,), dtype=float)
            else:
                # Lin interacted model:
                # params[2:2+p] are main effects (beta_cov)
                # params[2+p:2+2p] are interaction effects (gamma_cov)
                beta_cov = np.asarray(self._result.params[2 : 2 + p], dtype=float)
                gamma_cov = np.asarray(self._result.params[2 + p : 2 + 2 * p], dtype=float)

            diag = CUPEDDiagnosticData(
                ate_naive=ate_naive,
                se_naive=se_naive,
                variance_reduction_pct=float(var_red_pct) if np.isfinite(var_red_pct) else np.nan,
                beta_covariates=beta_cov,
                gamma_interactions=gamma_cov,
                covariates=list(self._covariate_names),
                adj_type=self.adjustment,
            )

        return CausalEstimate(
            estimand="ATE",
            model="CUPEDModel",
            model_options={
                "cov_type": self.cov_type,
                "use_t": self.use_t,
            },
            value=tau,
            ci_upper_absolute=ci_high,
            ci_lower_absolute=ci_low,
            value_relative=tau_rel,
            ci_upper_relative=ci_high_rel,
            ci_lower_relative=ci_low_rel,
            alpha=a,
            p_value=p_value,
            is_significant=bool(p_value < a),
            n_treated=int(np.sum(self._result.model.exog[:, 1] == 1)),
            n_control=int(np.sum(self._result.model.exog[:, 1] == 0)),
            treatment_mean=mu_t,
            control_mean=mu_c,
            outcome=str(self._result.model.endog_names) if hasattr(self._result.model, "endog_names") else "outcome",
            treatment=str(self._result.model.exog_names[1]) if hasattr(self._result.model, "exog_names") else "treatment",
            confounders=list(self._covariate_names),
            diagnostic_data=diag,
        )

    def summary_dict(self, alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Convenience JSON/logging output.

        Parameters
        ----------
        alpha : float, optional
            Override the instance significance level for confidence intervals.

        Returns
        -------
        dict
            Dictionary with estimates, inference, and diagnostics.
        """
        eff = self.estimate(alpha=alpha)
        diag: CUPEDDiagnosticData = eff.diagnostic_data
        return {
            "method": "Lin (2013) interacted adjustment",
            "adjustment": diag.adj_type,
            "ate": eff.value,
            "ate_relative_%": eff.value_relative,
            "p_value": eff.p_value,
            "ci_low": eff.ci_lower_absolute,
            "ci_high": eff.ci_upper_absolute,
            "ci_low_relative": eff.ci_lower_relative,
            "ci_high_relative": eff.ci_upper_relative,
            "alpha": eff.alpha,
            "nobs": eff.n_treated + eff.n_control,
            "cov_type": self.cov_type,
            "use_t": self.use_t,
            "ate_naive": diag.ate_naive,
            "se_naive": diag.se_naive,
            "variance_reduction_pct": diag.variance_reduction_pct,
            "covariates": diag.covariates,
            "beta_covariates": diag.beta_covariates.tolist(),
            "gamma_interactions": diag.gamma_interactions.tolist(),
        }

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._result is None:
            raise RuntimeError("CUPEDModel is not fitted. Call .fit(causaldata) first.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}("
            f"cov_type='{self.cov_type}', alpha={self.alpha}, "
            f"strict_binary_treatment={self.strict_binary_treatment}, "
            f"use_t={self.use_t}, status='{status}')"
        )
