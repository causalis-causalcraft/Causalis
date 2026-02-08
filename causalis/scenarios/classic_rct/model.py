from __future__ import annotations

from typing import Any, Optional, Literal

from causalis.dgp.causaldata import CausalData
from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.data_contracts.causal_diagnostic_data import DiffInMeansDiagnosticData
from .inference import ttest, bootstrap_diff_means, conversion_z_test


class DiffInMeans:
    """
    Difference-in-means model for CausalData.
    Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.
    """

    def __init__(self) -> None:
        self.data: Optional[CausalData] = None
        self._is_fitted: bool = False

    def fit(self, data: CausalData) -> DiffInMeans:
        """
        Fit the model by storing the CausalData object.

        Parameters
        ----------
        data : CausalData
            The CausalData object containing treatment and outcome variables.

        Returns
        -------
        DiffInMeans
            The fitted model.
        """
        if not isinstance(data, CausalData):
            raise ValueError("Input must be a CausalData object.")
        self.data = data
        self._is_fitted = True
        return self

    def estimate(
        self,
        method: Literal["ttest", "bootstrap", "conversion_ztest"] = "ttest",
        alpha: float = 0.05,
        diagnostic_data: bool = True,
        **kwargs: Any,
    ) -> CausalEstimate:
        """
        Compute the treatment effect using the specified method.

        Parameters
        ----------
        method : {"ttest", "bootstrap", "conversion_ztest"}, default "ttest"
            The inference method to use.
            - "ttest": Standard independent two-sample t-test.
            - "bootstrap": Bootstrap-based inference for difference in means.
            - "conversion_ztest": Two-proportion z-test for binary outcomes.
        alpha : float, default 0.05
            The significance level for calculating confidence intervals.
        diagnostic_data : bool, default True
            Whether to include diagnostic data_contracts in the result.
        **kwargs : Any
            Additional arguments passed to the underlying inference function.
            - For "bootstrap": can pass `n_simul` (default 10000).

        Returns
        -------
        CausalEstimate
            A results object containing effect estimates and inference.
        """
        if not self._is_fitted or self.data is None:
            raise RuntimeError(
                "Model must be fitted with .fit(data_contracts) before calling .estimate()."
            )

        a = float(alpha)

        if method == "ttest":
            res = ttest(self.data, alpha=a)
        elif method in ["bootstrap", "bootsrap"]:
            n_simul = kwargs.get("n_simul", 10000)
            res = bootstrap_diff_means(
                self.data, alpha=a, n_simul=n_simul
            )
        elif method in ["conversion_ztest", "coversion_ztest"]:
            res = conversion_z_test(self.data, alpha=a, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Supported: 'ttest', 'bootstrap', 'conversion_ztest'."
            )

        diag_data = None
        if diagnostic_data and self.data.confounders:
            diag_data = DiffInMeansDiagnosticData()

        return CausalEstimate(
            estimand="ATE",
            model="DiffInMeans",
            model_options={"method": method, "alpha": a, **kwargs},
            value=res["absolute_difference"],
            ci_upper_absolute=res["absolute_ci"][1],
            ci_lower_absolute=res["absolute_ci"][0],
            value_relative=res.get("relative_difference"),
            ci_upper_relative=res.get("relative_ci")[1] if isinstance(res.get("relative_ci"), (tuple, list)) else None,
            ci_lower_relative=res.get("relative_ci")[0] if isinstance(res.get("relative_ci"), (tuple, list)) else None,
            alpha=a,
            p_value=res["p_value"],
            is_significant=bool(res["p_value"] < a),
            n_treated=int(self.data.outcome[self.data.treatment == 1].shape[0]),
            n_control=int(self.data.outcome[self.data.treatment == 0].shape[0]),
            outcome=self.data.outcome_name,
            treatment=self.data.treatment_name,
            confounders=self.data.confounders,
            diagnostic_data=diag_data,
        )


    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"DiffInMeans(status='{status}')"
