# Refutation Module

The `causalis.refutation` package provides robustness and refutation utilities to stress-test causal estimates by perturbing data, checking identifying assumptions, and running sensitivity analyses.

## Overview

Key utilities:
- Placebo tests (randomize outcome or treatment, subsample)
- Sensitivity analysis for unobserved cofounding (including set-based benchmarking)
- Orthogonality/IRM moment checks with out-of-sample (OOS) diagnostics

## API Reference

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:
   :nosignatures:

   causalis.refutation
```