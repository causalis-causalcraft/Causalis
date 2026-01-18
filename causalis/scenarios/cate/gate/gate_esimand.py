"""
Group Average Treatment Effect (GATE) estimation using local DML IRM and BLP.
"""

from typing import Any, Optional, Union

import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness.irm import IRM


def gate_esimand(
    data: CausalData,
    groups: Optional[Union[pd.Series, pd.DataFrame]] = None,
    n_groups: int = 5,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate Group Average Treatment Effects (GATEs).

    If `groups` is None, observations are grouped by quantiles of the
    plugin CATE proxy (g1_hat - g0_hat).
    """
    # 1. Define defaults
    if ml_g is None:
        ml_g = CatBoostRegressor(thread_count=-1, verbose=False, allow_writing_files=False)
    if ml_m is None:
        ml_m = CatBoostClassifier(thread_count=-1, verbose=False, allow_writing_files=False)

    # 2. Fit IRM model
    # We use the local IRM implementation which exposes .gate()
    irm = IRM(
        data=data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
    )
    irm.fit().estimate(score="ATE")  # GATE uses ATE orthogonal signal

    # 3. Prepare groups
    if groups is None:
        # Construct groups based on CATE proxy
        # We use the plug-in estimator g1 - g0 for sorting to avoid overfitting to Y noise
        if irm.g1_hat_ is None or irm.g0_hat_ is None:
            raise RuntimeError("IRM model did not produce g1/g0 estimates.")
            
        cate_proxy = irm.g1_hat_ - irm.g0_hat_
        
        # Create quantile groups
        try:
            q = pd.qcut(cate_proxy, n_groups, labels=False, duplicates="drop")
        except ValueError:
            # Fallback for ties
            q = pd.cut(cate_proxy, n_groups, labels=False, duplicates="drop")
            
        # Create a DataFrame with a clear name
        groups_df = pd.DataFrame({"Group": q})
    else:
        groups_df = groups.copy()
        if isinstance(groups_df, pd.Series):
            groups_df = groups_df.to_frame()

    # 4. Run GATE via BLP
    # This returns a fitted BLP object
    gate_model = irm.gate(groups_df, alpha=alpha)

    # 5. Format results
    # Retrieve summary stats and confidence intervals
    summary = gate_model.summary
    ci_df = gate_model.confint(alpha=alpha)
    
    # Calculate group sizes (n) from the basis used in BLP
    # basis columns correspond to the groups
    counts = gate_model.basis.sum(axis=0).astype(int)

    # Construct final DataFrame
    # Note: summary index matches counts index and ci_df index
    results = pd.DataFrame({
        "group": summary.index,
        "n": counts.values,
        "theta": summary["coef"].values,
        "std_error": summary["std err"].values,
        "p_value": summary["P>|t|"].values,
        # CI columns in ci_df are [lower, effect, upper]
        "ci_lower": ci_df.iloc[:, 0].values,
        "ci_upper": ci_df.iloc[:, 2].values,
    })
    
    # Sort by group name/label for consistency
    results = results.sort_values("group").reset_index(drop=True)

    return results
