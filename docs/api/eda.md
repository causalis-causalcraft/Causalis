# EDA Module

The `causalis.eda` module provides exploratory diagnostics for causal designs with binary treatment. It helps assess treatment predictability, overlap/positivity, covariate balance, and outcome modeling quality before running inference.

## Overview

Key components:

- `CausalEDA`: High-level interface for EDA on CausalData or a lightweight container
- `CausalDataLite`: Minimal data container compatible with CausalEDA

Main capabilities:
- Outcome group statistics by treatment
- Cross-validated propensity scores with ROC AUC and positivity checks
- Covariate balance diagnostics (means, absolute diffs, and standardized mean differences)
- Outcome model fit diagnostics (RMSE, MAE) and SHAP-based feature attributions for CatBoost models
- Visualization helpers (propensity score overlap, distributions and boxplots)

## API Reference

```{eval-rst}
.. currentmodule:: causalis.eda

.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   eda
   confounders_balance
```