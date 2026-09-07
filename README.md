# Causalis
[![PyPI version](https://img.shields.io/pypi/v/causalis.svg)](https://pypi.org/project/causalis/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/causalis?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/causalis)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Code quality](https://img.shields.io/badge/code%20quality-A-brightgreen)
[![Docs](https://img.shields.io/badge/docs-causalis.causalcraft.com-blue)](https://causalis.causalcraft.com/)

<a href="https://causalis.causalcraft.com/"><img src="https://raw.githubusercontent.com/causalis-causalcraft/Causalis/main/notebooks/new_logo_big.svg" alt="Causalis logo" width="80" style="float: left; margin-right: 10px;" /></a>

Robust causal inference for experiments and observational studies in Python, organized around **scenarios** (e.g., Classic RCT, CUPED, Unconfoundedness) with a consistent `fit() → estimate()` workflow.

- 📚 Documentation & notebooks: https://causalis.causalcraft.com/
- 🔎 API reference: https://causalis.causalcraft.com/api-reference

## Why Causalis?
Causalis focuses on:
- Scenario-first workflows (you pick the study design; Causalis provides best-practice defaults).
- Extensive robustness tests that reveal issues in the study design or model specification
- Pydantic data contracts 
- An advanced DGP (Data Generating Process) with heterogeneous treatment effects, latent variables, and correlated confounders
- A website with notebooks based on real-world cases

## Installation
### Recommended
```bash
pip install causalis
```

# Quickstart: Classic RCT (difference in means + inference)

```python
from causalis.dgp import generate_classic_rct_26
from causalis.scenarios.classic_rct import DiffInMeans, check_srm

# Synthetic RCT data as a validated CausalData object
data = generate_classic_rct_26(seed=42, return_causal_data=True)

# Optional: Sample Ratio Mismatch check
srm = check_srm(data, target_allocation={0: 0.5, 1: 0.5}, alpha=1e-3)
print("SRM detected?", srm.is_srm, "p=", srm.p_value, "chi2=", srm.chi2)

# Estimate treatment effect with t-test inference (or bootstrap / conversion_ztest)
result = DiffInMeans().fit(data).estimate(method="ttest", alpha=0.05)
result.summary()
```
# Quickstart: Observational study (Unconfoundedness / DML IRM)
```python
from causalis.scenarios.unconfoundedness.dgp import generate_obs_hte_26
from causalis.scenarios.unconfoundedness import IRM
from causalis.data_contracts import CausalData

causaldata = generate_obs_hte_26(return_causal_data=True, include_oracle=False)

from causalis.scenarios.unconfoundedness import IRM

model = IRM().fit(causaldata)
result = model.estimate(score='ATTE')
result.summary()
```

## RCT data with multiple outcomes and treatment arms

```python
from causalis.data_contracts import RctCausalData

data = RctCausalData.from_df(
    df,
    treatment="treated",
    outcomes=["revenue", "purchases", "retention"],
    confounders=["age", "prior_spend"],
    user_id="customer_id",  # optional
)
Y = data.Y  # DataFrame, including when only one outcome is specified
X = data.X
revenue_data = data.for_outcome("revenue")  # CausalData for existing estimators
```

For multiple arms, supply one-hot columns including the control arm:

```python
data = RctCausalData.from_df(
    df,
    treatment_names=["variant_a", "control", "variant_b"],
    control_treatment="control",
    outcomes=["revenue", "purchases", "retention"],
    confounders=["age", "prior_spend"],
)
D = data.D  # treatment matrix, with control first
revenue_data = data.for_outcome("revenue")  # MultiCausalData for multiple arms
```

Every row must belong to exactly one arm, and every arm must be represented.
A single binary column uses 0 as control; omit `control_treatment` in that case.

This contract stores one independent copy of the selected columns and validates
shared treatment/confounders once. It checks missing values column by column and
screens duplicate columns with fingerprints before exact comparisons. Outcomes
and confounders must be finite, real numeric or boolean, and non-constant;
each treatment column must contain both 0 and 1. Identical columns are rejected.
`Y`, `D`, `X`, and `get_df()` return copies; `for_outcome()` copies and validates a
single-outcome subset using the destination contract's constraints, including
`MultiCausalData`'s limit of 15 arms. Existing estimators consume that
`CausalData` or `MultiCausalData` subset. Validation checks data structure;
it does not establish that treatment assignment was randomized.

To measure construction time and retained data size on your machine, run
`.venv/bin/python benchmarks/rct_causal_data.py --rows 1000000 --outcomes 32 --arms 3`.

### CUPED with RCT data

Generate a reproducible experiment using the internal DGP outcome families and
covariate samplers:

```python
from causalis.dgp import generate_rct_causal_data

data = generate_rct_causal_data(
    n=20_000,
    n_treatments=3,
    d_names=["control", "variant_a", "variant_b"],
    confounder_specs=[{"name": "prior_spend", "dist": "normal"}],
    outcome_specs=[
        {"name": "revenue", "alpha_y": 20, "beta_y": [3], "theta": [1, 2]},
        {"name": "purchases", "alpha_y": 10, "beta_y": [2], "theta": [0.5, 1]},
    ],
    seed=42,
)
```

Each outcome supports `continuous`, `binary`, `poisson`, or `gamma`. `theta`
sets arm effects on the family's link scale; the example uses continuous
outcomes, so revenue effects are exactly 1 and 2. For a binary treatment column,
set `n_treatments=2, treatment_encoding="binary"`. Use
`return_causal_data=False, include_oracle=True` for a DataFrame containing
allocation probabilities, potential-outcome means, and natural-scale effects.
Random assignment is shared across outcomes; rare missing arms raise an error
instead of changing sampled assignments.

```python
from causalis.scenarios.cuped import CUPEDModel

model = CUPEDModel().fit(data, covariates=["prior_spend"])
estimates = model.estimate()  # estimates[outcome][active_arm]
estimates.summary(outcome="revenue")  # formatted comparisons, side by side
revenue_a = estimates["revenue"]["variant_a"]

# Fit just one comparison when needed:
revenue_model = CUPEDModel().fit(
    data, covariates=["prior_spend"], outcome="revenue", treatment="variant_a",
)
revenue_a = revenue_model.estimate()  # CausalEstimate
```

CUPED compares each active arm only against the declared control and centers
covariates over those two arms. It shares the design matrix and decomposition
across outcomes within each comparison. Pass `covariates=[]` for unadjusted
estimates. One fitted comparison returns a `CausalEstimate`; multiple comparisons
return `RctEstimates`, preserving nested outcome/arm dictionary access.
`estimates.summary(outcome="revenue")` displays all arms for an outcome using
the same field and confidence-interval formatting as `CausalEstimate.summary()`.
Omit `outcome` to show every outcome, or add `treatment` to select an arm.
The same selectors work in `estimate()` and
`summary_dict()`. Batch `assumptions_table()` adds outcome and treatment columns.
Confidence intervals and p-values are per comparison, without multiplicity
adjustment. Existing `CausalData` inputs retain their single-estimate behavior.

CUPED keeps `treatment_mean` and `control_mean` as raw observed group means.
The additional `adjusted_control_mean` (regression intercept) and
`adjusted_treatment_mean` (intercept plus treatment coefficient) are evaluated
at the comparison sample's mean covariates. Their difference equals the adjusted
ATE; the difference of raw means may not. Both pairs appear in the summary.
The default relative effect is `100 * ATE / adjusted_control_mean`.
Choose `relative_denominator="raw_control"` explicitly to divide by the observed
control mean instead; the summary reports which denominator was selected.
CUPED preserves consistency and asymptotic unbiasedness under randomization
and standard regularity conditions; finite-sample bias need not be zero.

The comparison benchmark is
`.venv/bin/python benchmarks/cuped_rct.py --rows 100000 --outcomes 8`.
It disables optional regression checks with `run_checks=False`; ordinary fits
keep their configured checks.

### Plan an experiment: sample size and MDE

Use a past `RctCausalData` to estimate future sample size or sensitivity from
**historical control rows only**. Both calculators require one explicit `outcome`.
Omitted covariates, `None`, and `[]` use classic unadjusted planning, even when
the data declares confounders. Pass an explicit list to enable CUPED. With the
`data` from the example above:

```python
from causalis.shared.rct_design import calculate_mde, calculate_sample_size

allocation = {"control": 0.5, "variant_a": 0.25, "variant_b": 0.25}

# Classic planning: separate experiments for relative MDEs [0.5, 1, 5, 10, 20]%.
sizes = calculate_sample_size(data, outcome="revenue", allocation=allocation)

# Custom relative targets: 1 means 1%, not 100%.
custom = calculate_sample_size(
    data, outcome="revenue", scenario="user", mde_type="relative",
    mde=[1, 3, 5], allocation=allocation,
)

# CUPED sample-size planning with absolute effects in revenue units.
adjusted_sizes = calculate_sample_size(
    data, outcome="revenue", covariates=["prior_spend"],
    scenario="user", mde_type="absolute", mde=[0.5, 1, 2],
    allocation=allocation, include_details=True,
)

# Classic sensitivity for an exact total audience of 30,000 participants.
sensitivity = calculate_mde(
    data, outcome="revenue", sample_size=30_000, allocation=allocation,
)

# CUPED sensitivity for the same audience and allocation.
adjusted_sensitivity = calculate_mde(
    data, outcome="revenue", sample_size=30_000, covariates=["prior_spend"],
    allocation=allocation, include_details=True,
)

# Format for presentation without rounding the underlying numeric results.
print(sizes.to_string(index=False, formatters={
    "mde_relative": "{:.1f}%".format,
    "mde_absolute": "{:.3f}".format,
    "n_control": "{:,.0f}".format,
    "n_treatment": "{:,.0f}".format,
    "n_total": "{:,.0f}".format,
}))
```

`calculate_sample_size` takes MDE targets and returns required sample sizes.
`scenario="default"` uses the preset percentages and rejects custom `mde` or
`mde_type`. `scenario="user"` requires both a type and a nonempty sequence of
finite positive targets; input order and duplicates are preserved.
`calculate_mde` takes one positive integer `sample_size` across all future groups
and returns absolute and relative MDE for each active arm.

Both return unrounded numeric DataFrames with compact columns `outcome`,
`control`, `treatment`, `mde_relative`, `mde_absolute`, `n_control`, `n_treatment`,
and `n_total`. The MDE columns describe **requested targets** in sample-size
results and **calculated sensitivity** in MDE results. Relative values are
percentages of the raw historical control mean: with baseline 10, a relative
input of 1 corresponds to an absolute difference of 0.1. Relative target planning
requires a positive mean. Absolute target planning and MDE calculation remain
available for nonpositive means, with relative output set to NaN.

`include_details=True` appends `baseline_mean`, `variance_raw`, `variance_used`,
`variance_reduction_pct`, `n_reference`, `alpha`, and `power`. Sample-size results
also include `achieved_power` after rounding. Classic planning uses raw sample
variance (`ddof=1`) without fitting a regression; `variance_used` equals
`variance_raw` and variance reduction is zero. CUPED uses only explicitly selected
pre-treatment confounders and residual variance `SSE / (n_reference - design_rank)`.
Historical variance is estimated once for all targets.

All declared groups participate in each future experiment. Allocation defaults
to equal shares; custom shares must be positive and sum to one. For a binary
treatment column `d`, use keys `"d=0"` and `"d=1"`. For each sample-size target,
take the largest continuous requirement across active arms and round every group
up. MDE calculations preserve the supplied total using largest remainders, with
ties resolved in contract order, control first. Every group needs at least one
participant. `n_total` counts shared control once; do not sum it across rows.

Planning uses deterministic, two-sided normal power and assumes independently
randomized units with the same historical-control variance in every future arm.
Binary outcomes also use this fixed-variance approximation. `alpha` and `power`
are per comparison, without multiple-testing correction or a joint detection
guarantee. Control-constant covariates are dropped with a warning; singular CUPED
fits, insufficient residual degrees of freedom, and zero or non-finite variance
raise errors. Estimated variance reduction can be negative and is not clipped.

**Breaking API change:** the old `calculate_cuped_mde` and
`calculate_cuped_sample_size` names and CUPED design module are removed. Use the
shared calculators above; there are no aliases or deprecation wrappers. The old
shared summary-statistics `calculate_mde` API (`baseline_rate`, `variance`,
`ratio`, and `data_type`) is also removed. Both new APIs require `RctCausalData`
and a single `outcome`; the previous multi-outcome mapping interface is removed.

## Binary sensitivity protocol for observational DML/IRM

Pre-specify a practically meaningful effect boundary and one or more
domain-justified groups of observed **pre-treatment** confounders. The primary
decision uses element-based long/short gain statistics for the benchmark
group. Its `r2_y`, `r2_d`, and `rho` are calibrated jointly from the outcome
variance, Riesz-representer variance, and actual effect shift. The 2× strength
and forced `rho=1` scenarios are reported as secondary stress tests.

```python
from causalis.scenarios.unconfoundedness.refutation import run_sensitivity_protocol

# Example only: replace with a domain-justified, pre-specified group.
primary_group = list(causaldata.confounders[:2])

protocol = run_sensitivity_protocol(
    model,
    causaldata,
    benchmark_groups={"primary_domain_benchmark": primary_group},
    decision_threshold=0.0,  # replace with the minimum practical effect
    direction="auto",  # default: infer direction relative to the threshold
    preconditions_passed=True,  # causal set, overlap, nuisance quality, stability
)

print(protocol["status"])
print(protocol["summary"])
protocol["primary"]
protocol["stress"]
protocol["adversarial"]
```

`PASS` means every primary benchmark's bias-aware confidence interval remains
strictly beyond `decision_threshold` in the requested direction. `RV` and
`RVa` are reported as robustness diagnostics, not compared with universal
cutoffs. An empty benchmark set, failed external preconditions, unavailable
sensitivity elements, or strengths outside the finite sensitivity domain
produce `FAIL`.

By default, `direction="auto"` selects positive when the original estimate is
at or above `decision_threshold`, and negative otherwise. It uses this same
direction for every scenario and returns it in `protocol["direction"]`, with
an inference warning in `protocol["warnings"]`. For a negative estimate at a
zero threshold, every primary CI must have `ci_upper < 0`; touching or crossing
zero still fails. Thresholds retain their supplied sign, and an estimate equal
to the threshold does not pass. Use explicit `direction="positive"` or
`direction="negative"` for a pre-specified directional claim; these choices are
never overridden. Significance before sensitivity analysis alone does not
guarantee a pass.

Benchmark boundary handling matches DoubleML: raw `cf_y` and `cf_d` are
clipped to `[0, 1]`. If either long/short gain is not strictly positive,
primary and stress use `rho=sign(theta_short-theta_long)`; the adversarial
scenario still forces `rho=1`. `protocol["benchmarks"]` retains raw gains,
clipping/fallback flags, long/short elements, and warnings for auditability.
In particular, a negative `cf_d_raw` becomes a numerical `cf_d=0` boundary
benchmark rather than a missing scenario.

# Pick your scenario

| Scenario                                                                                   | Estimator                                                 | Assumptions                                                                                                                     |
|--------------------------------------------------------------------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [Classic RCT](https://causalis.causalcraft.com/articles/classic_rct)                       | Difference in means (ttest, ztest, welch_permutation_t_test)             | Random assignment, no sample ratio mismatch, SUTVA                                                                              |
| [CUPED](https://causalis.causalcraft.com/articles/cuped)                                   | CUPED-adjusted difference in means with Lin specification | Random assignment, no sample ratio mismatch, SUTVA, valid pre-period metrics                                                    |
| [Unconfoundedness](https://causalis.causalcraft.com/articles/unconfoundedness)             | DML IRM                                                   | Unconfoundedness, Overlap, SUTVA, No leakage, Score stability                                                                   |
| [GATE](https://causalis.causalcraft.com/articles/gate)                                     | DML IRM (GATE and GATET)                                  | Same assumptions as unconfoundedness, plus meaningful pre-specified or validated subgroup definitions.                          |
| [Multi Unconfoundedness](https://causalis.causalcraft.com/articles/multi_unconfoundedness) | Multi DML IRM                                             | Unconfoundedness, Multi class Overlap, SUTVA, No leakage, Score stability                                                       |
| [Synthetic Control](https://causalis.causalcraft.com/articles/synthetic_control)           | ASCM                                                      | No interference / spillovers, No anticipation, The treated unit’s untreated outcome path is well approximated by the donor pool |
| [Difference in Difference](https://causalis.causalcraft.com/articles/did)                  | CallawaySantAnnaDID                                       | Parallel trends, no anticipation, stable group composition, no spillovers between treated and control groups.                   |
| [IV](https://causalis.causalcraft.com/articles/iv)                                         | DML IV                                                    | First-stage strength, Reduced form, Instrument balance by Z, Instrument propensity / predictability                             |
| [Uplift / CATE scoring](https://causalis.causalcraft.com/articles/uplift)                  | DML IRM (CATE)                                            | Identified treatment effects from randomized or unconfounded data, overlap, calibrated individual-level predictions.            |

[Introduction to Causal Inference](https://causalis.causalcraft.com/articles/introduction-to-causal-inference): guide

See scenario notebooks: https://causalis.causalcraft.com/explore-scenarios

# [Contributing guidelines](https://github.com/causalis-causalcraft/Causalis?tab=contributing-ov-file)

# Maintainers

[Ioann Martynov](https://www.linkedin.com/in/ioannmartynov/)

# References

https://github.com/DoubleML/doubleml-for-py

## Search terms / supported methods

Causalis covers methods often searched as:

- causal inference Python
- causal machine learning Python
- treatment effect estimation
- A/B testing Python
- randomized controlled trial analysis
- CUPED Python
- Double Machine Learning Python
- DML / IRM
- CATE estimation
- uplift modeling
- propensity score diagnostics
- synthetic control Python
- difference-in-differences Python
