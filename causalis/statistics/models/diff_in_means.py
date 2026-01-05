from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from causalis.data.causaldata import CausalData
from causalis.scenarios.rct import ttest
from causalis.scenarios.rct import bootstrap_diff_means
from causalis.scenarios.rct import conversion_z_test


class DiffInMeans:
    """
    Difference-in-means model for CausalData.
    Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.

    Parameters
    ----------
    confidence_level : float, default 0.95
        The confidence level for calculating confidence intervals (between 0 and 1).
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = float(confidence_level)
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

    def effect(
        self,
        method: Literal["ttest", "bootstrap", "conversion_ztest"] = "ttest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute the treatment effect using the specified method.

        Parameters
        ----------
        method : {"ttest", "bootstrap", "conversion_ztest"}, default "ttest"
            The inference method to use.
            - "ttest": Standard independent two-sample t-test.
            - "bootstrap": Bootstrap-based inference for difference in means.
            - "conversion_ztest": Two-proportion z-test for binary outcomes.
        **kwargs : Any
            Additional arguments passed to the underlying inference function.
            - For "bootstrap": can pass `n_simul` (default 10000).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - p_value: The p-value from the test
            - absolute_difference: The absolute difference between treatment and control means
            - absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI
            - relative_difference: The relative difference (%) between treatment and control means
            - relative_ci: Tuple of (lower, upper) bounds for the relative difference CI
        """
        if not self._is_fitted or self.data is None:
            raise RuntimeError(
                "Model must be fitted with .fit(data) before calling .effect()."
            )

        if method == "ttest":
            return ttest(self.data, confidence_level=self.confidence_level)
        elif method in ["bootstrap", "bootsrap"]:
            n_simul = kwargs.get("n_simul", 10000)
            return bootstrap_diff_means(
                self.data, confidence_level=self.confidence_level, n_simul=n_simul
            )
        elif method in ["conversion_ztest", "coversion_ztest"]:
            return conversion_z_test(self.data, confidence_level=self.confidence_level)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Supported: 'ttest', 'bootstrap', 'conversion_ztest'."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"DiffInMeans(confidence_level={self.confidence_level}, status='{status}')"
