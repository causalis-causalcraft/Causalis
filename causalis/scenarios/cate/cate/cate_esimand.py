"""
IRM-based implementation for estimating CATE (per-observation orthogonal signals).

This module provides a function that, given a CausalData object, fits the internal IRM
model and augments the data with a new column 'cate' that contains the orthogonal
signals (an estimate of the conditional average treatment effect for each unit).
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier

from causalis.dgp.causaldata import CausalData
from causalis.scenarios.cate.blp import BLP
from causalis.scenarios.unconfoundedness.model import IRM


def cate_esimand(
    data: CausalData,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    use_blp: bool = False,
    X_new: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estimate per-observation CATEs using IRM and return a DataFrame with a new 'cate' column.

    Parameters
    ----------
    data : CausalData
        A CausalData object with defined outcome (outcome), treatment (binary 0/1), and confounders.
    ml_g : estimator, optional
        ML learner for outcome regression g(D, X) = E[Y | D, X] supporting fit/predict.
        Defaults to CatBoostRegressor if None.
    ml_m : classifier, optional
        ML learner for propensity m(X) = P[D=1 | X] supporting fit/predict_proba.
        Defaults to CatBoostClassifier if None.
    n_folds : int, default 5
        Number of folds for cross-fitting.
    n_rep : int, default 1
        Number of repetitions for sample splitting.
    use_blp : bool, default False
        If True, and X_new is provided, fits a BLP on the orthogonal signal and predicts CATE for X_new.
        If False (default), uses the in-sample orthogonal signal and appends to data.
    X_new : pd.DataFrame, optional
        New covariate matrix for out-of-sample CATE prediction via best linear predictor.
        Must contain the same feature columns as the confounders in `data_contracts`.

    Returns
    -------
    pd.DataFrame
        If use_blp is False: returns a copy of data with a new column 'cate'.
        If use_blp is True and X_new is provided: returns a DataFrame with 'cate' column for X_new rows.

    Raises
    ------
    ValueError
        If treatment is not binary 0/1 or required metadata is missing.
    """
    # Basic validation similar to ATE/ATT implementations for consistency
    if data.treatment is None:
        raise ValueError("CausalData object must have a treatment variable defined")
    if data.outcome is None:
        raise ValueError("CausalData object must have a outcome variable defined")
    if data.confounders is None:
        raise ValueError("CausalData object must have confounders variables defined")

    # Check binary treatment with explicit {0,1}
    unique_treatments = pd.Series(data.treatment).unique()
    if len(unique_treatments) != 2:
        raise ValueError("Treatment variable must be binary (have exactly 2 unique values)")
    if set(unique_treatments) != {0, 1}:
        raise ValueError("Treatment variable must have values 0 and 1")

    # Defaults for learners
    if ml_g is None:
        ml_g = CatBoostRegressor(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)
    if ml_m is None:
        ml_m = CatBoostClassifier(iterations=100, depth=5, min_data_in_leaf=2, thread_count=-1, verbose=False)

    # Prepare data
    df = data.get_df()
    irm = IRM(
        data=data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
    ).fit()

    if use_blp:
        # Out-of-sample via best linear predictor; require X_new
        if X_new is None:
            raise ValueError("X_new must be provided when use_blp=True")
        # Ensure columns match the confounders order
        X_cols = list(data.confounders)
        missing = [c for c in X_cols if c not in X_new.columns]
        if missing:
            raise ValueError(f"X_new is missing required columns: {missing}")
        if len(X_cols) == 0:
            raise ValueError("CATE BLP prediction requires at least one confounder.")
        basis = df[X_cols].copy()
        blp_model = BLP(orth_signal=irm.orth_signal, basis=basis).fit()
        cate_hat = blp_model.blp_model.predict(X_new[X_cols])
        return pd.DataFrame({"cate": np.asarray(cate_hat).reshape(-1)})

    # In-sample orthogonal signals as CATE estimates
    cate_hat = np.asarray(irm.orth_signal).reshape(-1)

    out = df.copy()
    out["cate"] = np.asarray(cate_hat).reshape(-1)
    return out
