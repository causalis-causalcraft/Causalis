from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, List, Literal

import numpy as np
import pandas as pd

from causalis.data.causaldata import CausalData

try:
    import statsmodels.api as sm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "CUPEDModel requires statsmodels. Install with: pip install statsmodels"
    ) from e


@dataclass(frozen=True)
class CUPEDResults:
    """
    Result container for CUPED / ANCOVA (and optional Lin-interacted) ATE/ITT estimate.

    Attributes
    ----------
    ate : float
        Estimated ATE/ITT (coefficient on treatment indicator D).
    se : float
        Standard error of `ate` under the requested covariance estimator.
    t_stat : float
        Test statistic for H0: ate = 0 (as reported by statsmodels; depends on `use_t`).
    p_value : float
        Two-sided p-value (as reported by statsmodels; depends on `use_t`).
    ci_low : float
        Lower bound of (1 - alpha) confidence interval.
    ci_high : float
        Upper bound of (1 - alpha) confidence interval.
    alpha : float
        Significance level used for CI.
    nobs : int
        Number of observations.
    cov_type : str
        Covariance estimator name (e.g., "HC3").
    use_t : bool
        Whether inference used t-distribution (True) or normal approximation (False),
        as configured in the fitted statsmodels result.
    adjustment : {"ancova", "lin"}
        Which adjustment was used: plain ANCOVA or Lin (fully interacted).

    ate_naive : float
        Unadjusted difference-in-means estimate (via Y ~ 1 + D).
    se_naive : float
        Standard error of `ate_naive` under the same covariance estimator.
    variance_reduction_pct : float
        100 * (1 - se(ate)^2 / se(ate_naive)^2). Can be negative.

    covariates : list[str]
        Names of covariates used for adjustment.
    beta_covariates : np.ndarray
        Estimated coefficients on centered covariates X^c (empty if no covariates).
    gamma_interactions : np.ndarray
        Estimated coefficients on interactions D * X^c (empty unless adjustment="lin").
    """
    ate: float
    se: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float
    nobs: int
    cov_type: str
    use_t: bool
    adjustment: Literal["ancova", "lin"]

    # Diagnostics
    ate_naive: float
    se_naive: float
    variance_reduction_pct: float

    # Coefficients
    covariates: List[str]
    beta_covariates: np.ndarray
    gamma_interactions: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        d = asdict(self)
        # Add some standardized keys for compatibility with IRM/DML
        d["coefficient"] = self.ate
        d["stderr"] = self.se
        d["p_val"] = self.p_value
        d["conf_int"] = [self.ci_low, self.ci_high]
        return d

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of the results."""
        return pd.DataFrame(
            {
                "coefficient": [self.ate],
                "stderr": [self.se],
                "t_stat": [self.t_stat],
                "p_val": [self.p_value],
                "lower_ci": [self.ci_low],
                "upper_ci": [self.ci_high],
            }
        )


class CUPEDModel:
    """
    CUPED / ANCOVA estimator for ATE/ITT in randomized experiments.

    Fits an outcome regression with pre-treatment covariates (optionally centered):

        ANCOVA (classic CUPED form):
            Y ~ 1 + D + X^c

        Lin (2013) fully interacted adjustment (optional, best-practice default in many RCTs):
            Y ~ 1 + D + X^c + D * X^c

    The reported effect is the coefficient on D, with robust covariance as requested.

    Parameters
    ----------
    cov_type : str, default="HC3"
        Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
        Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    center_covariates : bool, default=True
        If True, center covariates at their sample mean (X^c = X - mean(X)).
        This matches the classic CUPED adjusted-outcome form and improves numerical stability.
    strict_binary_treatment : bool, default=True
        If True, require treatment to be binary {0,1}.
    adjustment : {"ancova", "lin"}, default="ancova"
        - "ancova": Y ~ 1 + D + X^c
        - "lin":    Y ~ 1 + D + X^c + D*X^c
    use_t : bool, default=True
        Passed to statsmodels `.fit(..., use_t=use_t)`. If False, inference is based on
        normal approximation (common asymptotic choice for robust covariances).

    Notes
    -----
    - Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
    - The Lin (2013) specification is often recommended as a more robust regression-adjustment default
      in RCTs (allows different covariate slopes by treatment arm).
    """

    def __init__(
        self,
        cov_type: str = "HC3",
        alpha: float = 0.05,
        center_covariates: bool = True,
        strict_binary_treatment: bool = True,
        adjustment: Literal["ancova", "lin"] = "ancova",
        use_t: bool = True,
    ) -> None:
        self.cov_type = str(cov_type)
        self.alpha = float(alpha)
        self.center_covariates = bool(center_covariates)
        self.strict_binary_treatment = bool(strict_binary_treatment)
        if adjustment not in ("ancova", "lin"):
            raise ValueError("adjustment must be one of {'ancova','lin'}.")
        self.adjustment: Literal["ancova", "lin"] = adjustment
        self.use_t = bool(use_t)

        self._is_fitted: bool = False
        self._result: Any = None
        self._result_naive: Any = None
        self._covariate_names: List[str] = []
        self._p: int = 0  # number of covariates used

    def fit(self, data: CausalData, covariates: Optional[Sequence[str]] = None) -> CUPEDModel:
        """
        Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.

        Parameters
        ----------
        data : CausalData
            Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
        covariates : sequence of str, optional
            Subset of `data.confounders_names` to use as CUPED covariates.
            If None, uses all confounders from the object.

        Returns
        -------
        CUPEDModel
            Fitted estimator.

        Raises
        ------
        ValueError
            If requested covariates are missing, not in `data.confounders_names`,
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
                raise ValueError(f"CUPED covariates not found in data.df: {missing}")
            not_in_contract = [c for c in covariates if c not in set(data.confounders_names)]
            if not_in_contract:
                raise ValueError(
                    "CUPED covariates must be a subset of data.confounders_names; "
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
            if self.center_covariates:
                Xc = X - X.mean(axis=0, keepdims=True)
            else:
                Xc = X
            p = Xc.shape[1]
        else:
            Xc = np.zeros((n, 0), dtype=float)
            p = 0

        # Design matrix
        # Base: [1, D, Xc...]
        design_parts = [np.ones((n, 1), dtype=float), d.reshape(-1, 1)]
        if p > 0:
            design_parts.append(Xc)

        # Optional Lin interactions: add D * Xc
        if self.adjustment == "lin" and p > 0:
            DXc = Xc * d.reshape(-1, 1)  # elementwise
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
        self._is_fitted = True
        return self

    def estimate(self, alpha: Optional[float] = None) -> CUPEDResults:
        """
        Return the adjusted ATE/ITT estimate and inference.

        Parameters
        ----------
        alpha : float, optional
            Override the instance significance level for confidence intervals.

        Returns
        -------
        CUPEDResults
            Effect estimate (coefficient on D), standard error, test statistic, p-value,
            confidence interval, naive comparator, and variance-reduction diagnostic.
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

        ate_naive = float(self._result_naive.params[1])
        se_naive = float(self._result_naive.bse[1])

        # SE-based variance reduction (proxy). Guard division by 0.
        if se_naive > 0:
            var_red = 1.0 - (se * se) / (se_naive * se_naive)
            var_red_pct = float(100.0 * var_red)
        else:
            var_red_pct = np.nan

        p = self._p
        if p == 0:
            beta_cov = np.zeros((0,), dtype=float)
            gamma_cov = np.zeros((0,), dtype=float)
        else:
            # ANCOVA: params[2:2+p] are betas
            beta_cov = np.asarray(self._result.params[2 : 2 + p], dtype=float)

            # Lin: params[2+p:2+2p] are gammas
            if self.adjustment == "lin":
                gamma_cov = np.asarray(self._result.params[2 + p : 2 + 2 * p], dtype=float)
            else:
                gamma_cov = np.zeros((0,), dtype=float)

        return CUPEDResults(
            ate=tau,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            alpha=a,
            nobs=int(self._result.nobs),
            cov_type=str(self.cov_type),
            use_t=bool(getattr(self._result, "use_t", self.use_t)),
            adjustment=self.adjustment,
            ate_naive=ate_naive,
            se_naive=se_naive,
            variance_reduction_pct=float(var_red_pct) if np.isfinite(var_red_pct) else np.nan,
            covariates=list(self._covariate_names),
            beta_covariates=beta_cov,
            gamma_interactions=gamma_cov,
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
        return {
            "method": "CUPED/ANCOVA" if eff.adjustment == "ancova" else "Lin (2013) interacted adjustment",
            "adjustment": eff.adjustment,
            "ate": eff.ate,
            "se": eff.se,
            "t_stat": eff.t_stat,
            "p_value": eff.p_value,
            "ci_low": eff.ci_low,
            "ci_high": eff.ci_high,
            "alpha": eff.alpha,
            "nobs": eff.nobs,
            "cov_type": eff.cov_type,
            "use_t": eff.use_t,
            "ate_naive": eff.ate_naive,
            "se_naive": eff.se_naive,
            "variance_reduction_pct": eff.variance_reduction_pct,
            "covariates": eff.covariates,
            "beta_covariates": eff.beta_covariates.tolist(),
            "gamma_interactions": eff.gamma_interactions.tolist(),
        }

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._result is None:
            raise RuntimeError("CUPEDModel is not fitted. Call .fit(causaldata) first.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}("
            f"cov_type='{self.cov_type}', alpha={self.alpha}, "
            f"center_covariates={self.center_covariates}, "
            f"strict_binary_treatment={self.strict_binary_treatment}, "
            f"adjustment='{self.adjustment}', use_t={self.use_t}, "
            f"status='{status}')"
        )
