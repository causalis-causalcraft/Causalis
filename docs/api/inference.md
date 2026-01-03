# Inference Module

The `causalis.inference` package provides statistical inference tools for causal analysis across several estimands:
- ATT: Average Treatment effect on the Treated
- ATE: Average Treatment Effect
- CATE: Conditional Average Treatment Effect (per-observation signals)
- GATE: Grouped Average Treatment Effects

## Overview

At a glance:
- Simple tests for A/B outcomes (t-test, two-proportion z-test)
- DoubleML-based estimators for ATE and ATT
- DoubleML-based CATE signals and GATE grouping/intervals

## API Reference

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   causalis.inference
```