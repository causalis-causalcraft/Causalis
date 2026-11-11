# IPCW for a single organic-activation trial

`IPCW` estimates `ATT_no_activation`: activate in month `g` versus remain
unactivated throughout the follow-up horizon, among clients who activate in
`g`. Its main estimate is the **total effect over the horizon**, in units of
the outcome summed across months. Monthly effects and effects accumulated from
baseline are also available.

## Generate data and estimate the effect

```python
from causalis.scenarios.ipcw import IPCW, generate_ipcw_data
from causalis.scenarios.ipcw.refutation import (
    plot_ipcw_effects,
    plot_ipcw_weights,
    run_ipcw_diagnostics,
)

data, oracle, metadata = generate_ipcw_data(
    n_clients=5_000,
    n_periods=24,
    g=10,
    horizon=6,
    scenario="baseline",
    seed=42,
)

model = IPCW(data=data, random_state=42).fit()

# Start with a point estimate: no bootstrap fits or confidence intervals.
point = model.estimate(n_bootstrap=0)
print(point.value)
print(point.summary())
print(point.period_effects)
print(point.cumulative_effects)

# Full client bootstrap: refit both nuisance models for every replicate.
# The default is 199; consider 499 or more for a final analysis if practical.
result = model.estimate(alpha=0.05, n_bootstrap=199)
print(result.summary())

truth = oracle["att_aggregate"]["att_no_activation"].sum()
print("Estimated total ATT:", result.value)
print("Realized oracle total ATT:", truth)

checks = run_ipcw_diagnostics(result)
print(checks)
plot_ipcw_weights(result)
plot_ipcw_effects(result)
```

Both plotting functions return a matplotlib `Figure`; display the returned
object in a notebook or call its `savefig()` method to export it.

The generator wraps the existing organic-activation DGP and fixes
`main_cohorts=(g,)`, so its aggregate oracle refers to this trial. It returns
`(data, oracle, metadata)` and keeps potential outcomes and latent variables out
of the estimator's data. Use `return_ipcw_data=False` to receive the trial
DataFrame instead. Additional keyword arguments configure the underlying DGP;
an incompatible `main_cohorts` override raises an error.

## Prepare an observed panel

```python
from causalis.data_contracts import IPCWData
from causalis.dgp.organic_activation import make_ipcw_view
from causalis.scenarios.ipcw import IPCW

view = make_ipcw_view(df, g=10, horizon=6)
data = IPCWData.from_df(view)
result = IPCW(random_state=42).fit(data).estimate(n_bootstrap=0)
```

`make_ipcw_view` is intended for panels with the organic-activation column
schema. For other panels, create the corresponding trial rows and pass custom
column-role names and covariate lists to `IPCWData.from_df`. Baseline and
censoring covariates must be explicit or supplied by
`view.attrs["model_features"]`; the estimator does not infer them from all
numeric columns. Keep postdecision variables and oracle information out of
these feature lists.

Every baseline client must have every follow-up month from `0` to `horizon-1`.
Baseline assignment and baseline features remain fixed. The contract checks
the artificial-censoring indicators, requires finite outcomes on usable rows,
and requires current censoring features on at-risk rows. Outcomes on censored
rows can be missing because they do not contribute to the estimate.

## Estimation, assumptions, and diagnostics

Baseline propensity uses one observation per client. The censoring model uses
pooled at-risk control observations after baseline, including each censoring
event row. A control is censored when it first activates: its event-month and
later outcomes are excluded, while earlier outcomes remain usable. Treated
clients are never artificially censored.

For each usable control outcome in month `t`, the weight is the baseline
propensity odds multiplied by the inverse probability of remaining
uncensored from `g+1` through `t`. The survival product equals one at baseline.
Control weights are normalized separately in each month (a Hájek mean), and
the total ATT is the sum of treated-minus-weighted-control monthly means.
The final accumulated estimate equals `result.value`.

Both nuisance classifiers default to CatBoost with 200 iterations and depth 6.
You can supply cloneable classifiers with `fit` and `predict_proba` via `ml_m`
and `ml_c`. Cross-fitting assigns whole clients to folds, stratified by baseline
treatment. Each baseline group must contain at least `n_folds` clients.

Identification requires baseline exchangeability for activation at `g`,
sequential exchangeability of artificial censoring given the observed history,
and positivity for both assignment and remaining uncensored. Features used
for each censoring decision must be available before that decision.
Cross-fitting does **not** make this weighting estimator doubly robust.
Scenarios with unobserved selection can violate these assumptions; agreement
with an oracle is a benchmark, not an identification guarantee.

Propensities are clipped away from zero and one; conditional probabilities of
remaining uncensored are clipped only from below. Inspect censoring counts,
available controls, effective sample sizes, weight distributions, and clipping
fractions with the refutation functions. Their checks use `GREEN`, `YELLOW`,
and `RED` flags; these are support and stability checks, not proofs of validity.
Even the correctly specified identification scenarios can produce noisy
estimates at the default sample size: a small treated cohort or a few large
weights can make the effective control sample much smaller than its row count.
Compare simulation performance over several seeds and sample sizes, and inspect
ESS before interpreting agreement or disagreement with the oracle.

Bootstrap samples whole clients and refits both nuisance models using the
original client folds. Repeated clients contribute their multiplicity in
training and weighted means. All monthly, accumulated, and total intervals
use the same replicates and Wald inference. Refitting can be expensive;
`n_jobs` controls bootstrap parallelism. The estimator performs exactly the
requested number of replicates and raises a numbered error if a replicate
lacks support or model fitting fails, without replacing it with another draw.
`estimate(n_bootstrap=0)` remains available for point estimates; `1` is invalid.

This first version fits one trial baseline `g` at a time. It handles artificial
censoring of controls after activation, requires complete administrative
follow-up, and does not model natural loss to observation, survival outcomes,
multiple-trial pooling, ATE, or a doubly robust extension.
