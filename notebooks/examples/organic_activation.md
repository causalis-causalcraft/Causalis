# Organic activation and ecosystem LTV

This generator creates one client-by-calendar-month panel for comparing causal
designs with staggered, organic product activation. It provides separate oracle
benchmarks for two interventions, six selection scenarios, and data preparation
helpers for DiD, DML/IRM, and an observational target trial with artificial
censoring. It does not fit these estimators.

## Generate the panel and three analysis views

```python
from causalis.dgp import generate_organic_activation_data
from causalis.dgp.organic_activation import (
    OrganicActivationConfig,
    generate_data,
    make_did_pair,
    make_dml_view,
    make_ipcw_view,
    summarize_data,
    validate_data,
)

# Use n_clients=50_000 for the default benchmark size.
df, oracle, metadata = generate_organic_activation_data(
    n_clients=5_000,
    n_periods=24,
    horizon=6,
    scenario="baseline",
    seed=42,
)

df_did = make_did_pair(df, g=10, t=15)
df_dml = make_dml_view(df, g=10, horizon=6)
df_ipcw = make_ipcw_view(df, g=10, horizon=6)

checks = validate_data(df, oracle, metadata)
diagnostics = summarize_data(df, oracle, metadata)

print(df.head())
print(checks)
print(diagnostics["overview"])
print(diagnostics["att_aggregate"])
print(metadata["scenario_table"].to_string(index=False))
```

`generate_data` is the same generator entry point inside the dedicated package.
The defaults are 50,000 clients, 24 months, a six-month horizon, the `baseline`
scenario, and seed 42. Additional keyword arguments configure the DGP through
`OrganicActivationConfig`; its default entry periods are 0 through 9,
`main_cohorts=tuple(range(1, 11))`, and `calendar_start="2024-01"`.
Only the requested scenario is generated. Results stay in memory; no files are
saved automatically. Repeating a call with the same configuration and seed
reproduces its data and oracle.

## DGP and timing

Client entry $E_i$ is drawn before activation from months 0 through 9. Client
$i$ has rows only for $t\geq E_i$, through the administrative panel end.
There is no observation dropout. At the beginning of month $t$, tenure is
$t-E_i$. The first month is ineligible; the first possible activation is
$E_i+1$. An activated client remains treated in every later month.

Each month follows this order:

1. Generate the observed state $L_{it}$, available before the decision.
2. Compute the activation hazard among eligible, previously untreated clients.
3. Realize first activation using the pre-generated uniform innovation.
4. Generate the monthly monetary contribution, including any treatment effect.
5. Add that contribution to cumulative LTV.

Static observed $X_i$ describe income, salary receipt, segment, digital
propensity, and two noise features. The state `l_activity_pre` follows an AR(1)
process with a mean depending on $X_i$. The latent level $U_i$, hidden
time-varying factor $V_{it}$, state innovations, activation uniforms, and
outcome innovations are generated before the structural replay. State
innovations are independent of $U_i$. Treatment affects outcomes only, so
state paths remain the same under all interventions. Activation uniforms are
independent across calendar months and shared across interventions; the current
decision therefore does not reuse a latent draw governing future decisions.

The structural equations have the form

$$
L_{it}=\rho L_{i,t-1}+a(X_i)+\nu_{it},
\qquad
p_{it}=\operatorname{expit}\{a_t+q(X_i,t-E_i)
 +\gamma_L L_{it}+\gamma_U U_i+\gamma_V V_{it}\},
$$

$$
Y_{it}(0)=f(X_i,t-E_i)+\lambda_t+\beta_t(X_i)
 +b_U U_i+b_L L_{it}+\kappa V_{it}+\varepsilon_{it},
$$

$$
Y_{it}=Y_{it}(0)+\mathbf 1\{t\geq G_i\}\tau(X_i,t-G_i),
\qquad
C_{it}=\sum_{s=E_i}^{t}Y_{is}.
$$

The implementation uses smooth nonlinearities and interactions in the
selection, outcome, and heterogeneous treatment-effect functions. Calendar
seasonality and observed-covariate trends create outcome dynamics. Monetary
outcomes are in rubles; innovations allow negative contributions and finite
variance heavy tails. The effect changes with event time and equals zero before
activation. There is no discounting. Cumulative LTV may decrease when a monthly
contribution is negative.

More specifically, income enters through the bounded transform
$z_i=\tanh\{(\log x_{\mathrm{income},i}-\mu_{\log\mathrm{income}})
/\sigma_{\log\mathrm{income}}\}$. The activation score combines this
transform, salary, segment, digital propensity, and their salary/digital
interaction. Applying $2\tanh(\mathrm{score}/2)$ bounds this covariate
component before the scenario-specific strength multiplier. The hazard also
includes log tenure and annual seasonality. This is a structural score
definition, not clipping of probabilities or observations.

For $h\geq0$, the treatment effect is

$$
\tau(X_i,h)=\theta_0
 (1+\theta_z z_i+\theta_s x_{\mathrm{salary},i}
   +\theta_d x_{\mathrm{digital},i})
 \{1-\exp[-(h+1)/r]\}(1+s h),
$$

with parameters `effect_base`, `effect_income`, `effect_salary`,
`effect_digital`, `effect_ramp`, and `effect_slope`. The default base scale is
220 rubles, with ramp 2.5 and slope 0.02. `null_effect` sets the entire function
to exactly zero. Outcome innovations combine a standardized Student-t draw
(default five degrees of freedom) and centered lognormal noise. The
configuration requires degrees of freedom above two to retain finite variance.

Activation occurs at the beginning of its month. For cohort $g$, the last full
untreated month is $g-1$, the first treated event month is $h=t-g=0$, and a
horizon $H$ contains months $g,\ldots,g+H-1$. With the default entry scheme,
the main cohorts $g=1,\ldots,10$ naturally have between one and ten full
pre-treatment observations. Later cohorts remain in the source panel and can
serve as earlier not-yet-treated controls.

The main panel contains the following design and outcome fields:

| Columns | Meaning |
|---|---|
| `client_id`, `period`, `date` | Unique client/month key, zero-based calendar month, and month-start timestamp |
| `entry_period`, `tenure_months`, `n_pre_periods_available` | Entry month and observed history before the current decision |
| `first_activation_period` | Factual first activation month; nullable integer for clients never activated in the observed panel |
| `activation_now`, `treated`, `eligible_to_activate` | First activation this month, absorbing treatment status, and risk-set eligibility immediately before the decision |
| `event_time` | `period - first_activation_period`, nullable for never-activated clients |
| `x_income`, `x_salary`, `x_segment`, `x_digital`, `x_noise_1`, `x_noise_2` | Numeric static observed covariates |
| `l_activity_pre` | Observed financial activity measured before the current decision |
| `ltv_monthly`, `ltv_cumulative` | Monthly contribution in rubles and its cumulative sum since entry |

## Two oracle effects

Among clients who actually first activate in $g$, the generator evaluates

$$
ATT_{\mathrm{no\ activation}}(g,H)=
E\!\left[\sum_{h=0}^{H-1}
 \{Y_{i,g+h}^{\mathrm{start\ at}\ g}
  -Y_{i,g+h}^{\mathrm{no\ activation}}\}\mid G_i=g\right],
$$

and

$$
ATT_{\mathrm{defer\ natural}}(g,H)=
E\!\left[\sum_{h=0}^{H-1}
 \{Y_{i,g+h}^{\mathrm{start\ at}\ g}
  -Y_{i,g+h}^{\mathrm{skip}\ g,\ \mathrm{then\ natural}}\}
 \mid G_i=g\right].
$$

The first intervention prevents activation throughout follow-up. The second
forbids activation only in $g$, then recomputes future first activation using
the organic hazard and the same external innovations. It never substitutes the
client's factual activation date for that counterfactual date.

The oracle reports individual and cohort effects by event month, their running
sums, and an aggregate over the configured main cohorts. Cohort weights are
their factual treated counts divided by the total treated count across those
cohorts. The same weights apply at every event month, so summing the aggregate
monthly curve equals aggregating the cohort cumulative effects.
These are realized sample ATTs for the simulated treated clients and fixed
external innovations; they are not analytically integrated population means.

The package keeps these quantities separate from observed data:

| Object | Contents |
|---|---|
| `df` | Observed outcomes, treatment/design columns, static observed `x_*` features, and pre-decision `l_*_pre` states |
| `oracle["potential_outcomes"]` | Row-aligned untreated and factual outcomes, factual effect, true hazard, and hidden `U`/`V` |
| `oracle["individual_effects"]` | Client, factual cohort `g`, event month, `y_start`, `y_no_activation`, `y_defer_natural`, and both effect paths with cumulative effects |
| `oracle["att_by_cohort"]` | Cohort/event-month effects, `n_treated`, and explicit cohort `weight` |
| `oracle["att_aggregate"]` | Aggregate monthly and cumulative curves for both estimands |
| `oracle["skip_activation_period"]` | Replayed future activation date after skipping the factual activation decision, for clients in the configured main cohorts |
| `oracle["exogenous"]` | Pre-generated external innovations used for structural replay |
| `metadata["model_features"]` | Allowed feature lists by scenario and design |
| `metadata["scenario_table"]` | Scenario-specific assumptions and identification statements |

`first_activation_period` and `event_time` in `df` organize analysis groups;
they reveal future information and must not be propensity-model predictors.
Never build a feature matrix by automatically taking every numeric column.

## Analysis views and identification

All three helpers return new dataframes and leave the source `df` unchanged.

`make_did_pair(df, g, t)` pairs the monthly outcomes at $g-1$ and $t$ for
cohort $G_i=g$ and controls with $G_i>t$ or no observed activation. Both
groups must be eligible at the baseline decision and observed at both outcome
times. Its adjustment features come from pre-treatment information. Estimate
the monthly cohort-time effect, then sum monthly effects over the desired
horizon using fixed cohort weights. The long source panel also remains
available for a staggered cohort-by-time estimator. Conditional parallel trends
can allow covariates to create differing unconditional trends, as described by
[Callaway and Sant'Anna](https://arxiv.org/abs/1803.09015).

Use **monthly** LTV for this DiD comparison. In
`latent_level_confounding`, the additive monthly $U_i$ cancels from
$Y_{it}(0)-Y_{i,g-1}(0)$. A change in cumulative LTV instead contains a
multiple of $U_i$, so it does not inherit this cancellation. The valid
conditioning set is the observed static $X_i$ together with entry/tenure
information; do not add noisy outcome lags to that identification claim.
Conditioning on a lag containing both $U_i$ and noise can create a
regression-to-the-mean problem after selection on $U_i$.

`make_dml_view(df, g=10, horizon=6)` takes the eligible untreated risk set
immediately before the decision in $g$, freezes the available baseline
features, sets `d=activation_now`, and sums factual monthly contributions into
`y_ltv_h`. Clients who activate later remain in baseline `d=0`. Its ATT
benchmark is therefore `ATT_defer_natural`. An IRM with an ATT score needs
conditional exchangeability and overlap at this baseline; orthogonal scores
and cross-fitting do not remove hidden confounding. See
[Chernozhukov et al.](https://arxiv.org/abs/1701.08687) for DML treatment-effect
estimation.

`make_ipcw_view(df, g=10, horizon=6)` expands the same baseline risk set into
client-by-follow-up-month rows. It freezes baseline treatment and covariates,
and provides current pre-decision history for a censoring model. A baseline
control is artificially censored at its first subsequent activation. The
censoring-event row remains available for fitting that model, while its monthly
outcome is unusable as an untreated outcome because activation preceded that
outcome. All subsequent control outcomes are also unusable. Baseline treated
clients are not artificially censored. The source `df` retains every row.

For an ATT, the intended control weight is

$$
w_{ig,t}=\frac{e_g(\mathcal H_{ig})}{1-e_g(\mathcal H_{ig})}
 \prod_{s=g+1}^{t}
 \frac{1}{P(A_{is}=0\mid G_i\geq s,\mathcal H_{is})}.
$$

The first factor addresses baseline confounding. The product addresses
selection from later artificial censoring and is empty at $t=g$. Do not
include baseline assignment a second time in that product. Censoring predictors
must be available before the current decision; the current outcome is not
available. Target-trial emulation requires aligned eligibility, assignment, and
follow-up, plus adjustment for baseline confounding and censoring selection;
see [Hernán et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5124536/).

Feature lists are explicit and depend on the scenario:

```python
features = metadata["model_features"]["baseline"]
did_features = features["did"]
dml_features = features["dml"]
ipcw_baseline_features = features["ipcw_baseline"]
ipcw_censoring_features = features["ipcw_censoring"]

print(did_features)
print(dml_features)
```

In `observed_timevarying_confounding`, baseline DML and the baseline IPCW model
need the current pre-decision state; later censoring models need the evolving
pre-decision state. Frozen baseline names and current-history names in the
trial view distinguish those two roles. Missing history must retain its
availability and window-length information rather than being silently replaced
with zero. If pooling multiple baseline trials, group cross-fitting and
bootstrap resampling by `client_id` because a client can enter several trials.

Each view also carries its predictor manifest and target in `DataFrame.attrs`:

```python
X_did = df_did.loc[:, df_did.attrs["model_features"]]
X_dml = df_dml.loc[:, df_dml.attrs["model_features"]]
trial_features = df_ipcw.attrs["model_features"]

# Fit only later-decision censoring hazards on these rows, including events.
censoring_sample = df_ipcw.loc[df_ipcw["at_risk_of_censoring"]]
X_censoring = censoring_sample.loc[:, trial_features["censoring"]]

# Retained post-censoring rows must not contribute outcome observations.
usable_outcomes = df_ipcw.loc[df_ipcw["outcome_usable"]]
print(df_did.attrs["estimand"], df_dml.attrs["estimand"])
```

DiD returns `y_pre`, `y_post`, and `delta_y` with `baseline_period=g-1`.
DML has `baseline_period=g` and outcome `y_ltv_h`. IPCW uses
`baseline_period=g`, `baseline_d`, zero-based `followup_month`, and `trial_id=g`.
Its frozen baseline feature names start with `b_`; the current censoring-model
features retain their original names. `at_risk_of_censoring` is false at
baseline and after the censoring event. `censor_now` marks that event, and
`uncensored_through_t` and `outcome_usable` mark usable follow-up outcomes.
Every baseline client retains all $H$ trial rows, including unusable outcomes.

| Scenario | Selection and outcome mechanism | DiD: `ATT_no_activation` | DML/IRM: `ATT_defer_natural` | IPCW: `ATT_no_activation` |
|---|---|---|---|---|
| `baseline` | Hazard uses observed static features, calendar time, and tenure | Conditional parallel trends given static features and entry/tenure | Exchangeability given sufficient observed baseline features | Baseline adjustment and observed censoring history suffice |
| `latent_level_confounding` | Hidden constant `U` affects selection and monthly outcome level | Valid for monthly changes using static features and entry/tenure; no noisy outcome lags required | Hidden baseline confounding can bias the estimate | Baseline and censoring adjustment on observed features cannot guarantee identification |
| `observed_timevarying_confounding` | Current observed `L` affects selection and outcome | Static-only parallel trends can fail | Valid with sufficient current pre-decision state and baseline features | Valid with both baseline adjustment and sufficient evolving pre-decision history |
| `hidden_timevarying_confounding` | Hidden evolving `V` affects selection and outcome dynamics | No observed-feature identification guarantee | No observed-feature identification guarantee | No observed-feature identification guarantee |
| `null_effect` | Baseline selection, identically zero treatment effect | Valid under baseline conditions; oracle is exactly zero | Valid under baseline conditions; oracle is exactly zero | Valid under baseline conditions; oracle is exactly zero |
| `weak_overlap` | Baseline structure with extreme but interior hazards | Conditional trends hold; finite-sample overlap can be poor | Exchangeability holds; extreme propensities impair estimation | Identification conditions retain positivity; accumulated weights can be unstable |

All validity statements also require the design's relevant overlap, consistency,
and no anticipation conditions. The stress scenarios are defined by structural
equations, not by selecting seeds or a non-significant pre-trend test. Organic
decisions are observational; these comparisons are not randomized ITT effects.
The generator neither trims clients nor clips estimated weights automatically.

## Validation and compact diagnostics

`validate_data` checks key uniqueness, at most one activation, absorbing
treatment, activation eligibility, pre-history support, cumulative accounting,
zero pre-activation effects, factual consistency, the exact null effect,
censoring timing, and feature leakage. It returns named boolean checks and raises
`ValueError` for structural violations. It also reports sample support for
treated, later-treated, and never-treated clients. Support is a report rather
than a reason to resample or silently alter the simulated population; very
small configurations can have empty cohorts.

`summarize_data` returns in-memory diagnostic objects:

```python
print(diagnostics["activation_counts"])
print(diagnostics["pre_history_counts"])
print(diagnostics["control_support"])
print(diagnostics["dml_later_activation"])
print(diagnostics["hazard_quantiles"])
print(diagnostics["att_by_cohort"])
print(diagnostics["att_aggregate"])
```

Inspect both ATT columns in the oracle tables: deferral may lead to treatment
within the horizon, so its effect generally differs from preventing activation
throughout the horizon. Hazard quantiles are calculated inside the eligible
activation risk set. In `weak_overlap`, these diagnostics expose extreme
probabilities rather than concealing them through filtering.
