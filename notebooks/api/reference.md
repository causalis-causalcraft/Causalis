## `causalis`

Causalis: A Python package for causal inference.

**Modules:**

- [**data**](#causalis.data) – Data generation utilities for causal inference tasks.
- [**eda**](#causalis.eda) –
- [**refutation**](#causalis.refutation) – Refutation and robustness utilities for Causalis.
- [**scenarios**](#causalis.scenarios) –
- [**statistics**](#causalis.statistics) –

### `causalis.data`

Data generation utilities for causal inference tasks.

**Modules:**

- [**causal_diagnostic_data**](#causalis.data.causal_diagnostic_data) –
- [**causal_estimate**](#causalis.data.causal_estimate) –
- [**causaldata**](#causalis.data.causaldata) – Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.
- [**causaldata_instrumental**](#causalis.data.causaldata_instrumental) –
- [**dgps**](#causalis.data.dgps) –

**Classes:**

- [**CausalData**](#causalis.data.CausalData) – Container for causal inference datasets.
- [**CausalDataInstrumental**](#causalis.data.CausalDataInstrumental) – Container for causal inference datasets with causaldata_instrumental variables.
- [**CausalDatasetGenerator**](#causalis.data.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,
- [**CausalEstimate**](#causalis.data.CausalEstimate) – Result container for causal effect estimates.
- [**DiagnosticData**](#causalis.data.DiagnosticData) – Base class for all diagnostic data.
- [**SmokingDGP**](#causalis.data.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.
- [**UnconfoundednessDiagnosticData**](#causalis.data.UnconfoundednessDiagnosticData) – Fields common to all models assuming unconfoundedness.

**Functions:**

- [**generate_classic_rct**](#causalis.data.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.data.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_rct**](#causalis.data.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_gold_linear**](#causalis.data.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.data.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.data.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

#### `causalis.data.CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data contract.

**Attributes:**

- [**df**](#causalis.data.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.data.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.data.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.data.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.data.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.data.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.data.CausalData.get_df) – Get a DataFrame with specified columns.

##### `causalis.data.CausalData.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

##### `causalis.data.CausalData.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

##### `causalis.data.CausalData.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

##### `causalis.data.CausalData.df`

```python
df: pd.DataFrame
```

##### `causalis.data.CausalData.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, **kwargs)
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A validated CausalData instance.

##### `causalis.data.CausalData.get_df`

```python
get_df(columns=None, include_treatment=True, include_outcome=True, include_confounders=True, include_user_id=False)
```

Get a DataFrame with specified columns.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

##### `causalis.data.CausalData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

##### `causalis.data.CausalData.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

##### `causalis.data.CausalData.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

##### `causalis.data.CausalData.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

##### `causalis.data.CausalData.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

##### `causalis.data.CausalData.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

##### `causalis.data.CausalData.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data.CausalDataInstrumental`

Bases: <code>[CausalData](#causalis.data.causaldata.CausalData)</code>

Container for causal inference datasets with causaldata_instrumental variables.

**Attributes:**

- [**instrument_name**](#causalis.data.CausalDataInstrumental.instrument_name) (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.

**Functions:**

- [**from_df**](#causalis.data.CausalDataInstrumental.from_df) – Friendly constructor for CausalDataInstrumental.
- [**get_df**](#causalis.data.CausalDataInstrumental.get_df) – Get a DataFrame with specified columns including instrument.

##### `causalis.data.CausalDataInstrumental.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

##### `causalis.data.CausalDataInstrumental.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

##### `causalis.data.CausalDataInstrumental.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

##### `causalis.data.CausalDataInstrumental.df`

```python
df: pd.DataFrame
```

##### `causalis.data.CausalDataInstrumental.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, instrument=None, **kwargs)
```

Friendly constructor for CausalDataInstrumental.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- **instrument** (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalDataInstrumental](#causalis.data.causaldata_instrumental.CausalDataInstrumental)</code> – A validated CausalDataInstrumental instance.

##### `causalis.data.CausalDataInstrumental.get_df`

```python
get_df(columns=None, include_treatment=True, include_outcome=True, include_confounders=True, include_user_id=False, include_instrument=False)
```

Get a DataFrame with specified columns including instrument.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.
- **include_instrument** (<code>[bool](#bool)</code>) – Whether to include the instrument column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

##### `causalis.data.CausalDataInstrumental.instrument`

```python
instrument: pd.Series
```

instrument column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The instrument column.

##### `causalis.data.CausalDataInstrumental.instrument_name`

```python
instrument_name: str = Field(alias='instrument')
```

##### `causalis.data.CausalDataInstrumental.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

##### `causalis.data.CausalDataInstrumental.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

##### `causalis.data.CausalDataInstrumental.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

##### `causalis.data.CausalDataInstrumental.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

##### `causalis.data.CausalDataInstrumental.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

##### `causalis.data.CausalDataInstrumental.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

##### `causalis.data.CausalDataInstrumental.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, include_oracle=True, seed=None)
```

Generate synthetic causal inference datasets with controllable confounding,
treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

**Data model (high level)**

- confounders X ∈ R^k are drawn from user-specified distributions.
- Binary treatment D is assigned by a logistic model:
  D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
  where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
- Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
  outcome_type = "continuous":
  Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε, ε ~ N(0, sigma_y^2)
  outcome_type = "binary":
  logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "poisson":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

**Returned columns**

- y: outcome
- d: binary treatment (0/1)
- x1..xk (or user-provided names)
- m: true propensity P(T=1 | X) marginalized over U
- m_obs: realized propensity P(T=1 | X, U)
- tau_link: tau(X) on the structural (link) scale
- g0: E[Y | X, T=0] on the natural outcome scale marginalized over U
- g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
- cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

Notes on effect scale:

- For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
- For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
- For "poisson", tau acts on the *log-mean* scale. `cate` is reported on the mean (rate) scale.

**Parameters:**

- **theta** (<code>[float](#float)</code>) – Constant treatment effect used if `tau` is None.
- **tau** (<code>[callable](#callable)</code>) – Function tau(X) -> array-like shape (n,) for heterogeneous effects.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the outcome baseline f_y(X).
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the treatment score f_d(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
- **g_d** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson).
- **alpha_d** (<code>[float](#float)</code>) – Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
- **sigma_y** (<code>[float](#float)</code>) – Std. dev. of the Gaussian noise for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Outcome family and link as defined above.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders. See `_gaussian_copula` for details.
- **k** (<code>[int](#int)</code>) – Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
- **use_copula** (<code>[bool](#bool)</code>) – If True and `confounder_specs` provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
- **u_strength_d** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in treatment assignment.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in the outcome.
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales the X-driven treatment score to adjust positivity difficulty.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Attributes:**

- [**rng**](#causalis.data.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.data.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

##### `causalis.data.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

##### `causalis.data.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

##### `causalis.data.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

##### `causalis.data.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

##### `causalis.data.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

##### `causalis.data.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

##### `causalis.data.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data.CausalDatasetGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

##### `causalis.data.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

##### `causalis.data.CausalDatasetGenerator.k`

```python
k: int = 5
```

##### `causalis.data.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

##### `causalis.data.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

##### `causalis.data.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

##### `causalis.data.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

##### `causalis.data.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

##### `causalis.data.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

##### `causalis.data.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

##### `causalis.data.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

##### `causalis.data.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

##### `causalis.data.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

##### `causalis.data.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

##### `causalis.data.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

##### `causalis.data.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

#### `causalis.data.CausalEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result container for causal effect estimates.

**Parameters:**

- **estimand** (<code>[str](#str)</code>) – The estimand being estimated (e.g., 'ATE', 'ATTE', 'CATE').
- **model** (<code>[str](#str)</code>) – The name of the model used for estimation.
- **model_options** (<code>[dict](#dict)</code>) – Options passed to the model.
- **value** (<code>[float](#float)</code>) – The estimated absolute effect.
- **ci_upper_absolute** (<code>[float](#float)</code>) – Upper bound of the absolute confidence interval.
- **ci_lower_absolute** (<code>[float](#float)</code>) – Lower bound of the absolute confidence interval.
- **value_relative** (<code>[float](#float)</code>) – The estimated relative effect.
- **ci_upper_relative** (<code>[float](#float)</code>) – Upper bound of the relative confidence interval.
- **ci_lower_relative** (<code>[float](#float)</code>) – Lower bound of the relative confidence interval.
- **alpha** (<code>[float](#float)</code>) – The significance level (e.g., 0.05).
- **p_value** (<code>[float](#float)</code>) – The p-value from the test.
- **is_significant** (<code>[bool](#bool)</code>) – Whether the result is statistically significant at alpha.
- **n_treated** (<code>[int](#int)</code>) – Number of units in the treatment group.
- **n_control** (<code>[int](#int)</code>) – Number of units in the control group.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome variable.
- **treatment** (<code>[str](#str)</code>) – The name of the treatment variable.
- **confounders** (<code>list of str</code>) – The names of the confounders used in the model.
- **instrument** (<code>list of str</code>) – The names of the instruments used in the model.
- **time** (<code>[datetime](#datetime.datetime)</code>) – The date and time when the estimate was created.
- **diagnostic_data** (<code>[DiagnosticData](#causalis.data.causal_diagnostic_data.DiagnosticData)</code>) – Additional diagnostic data.
- **sensitivity_analysis** (<code>[dict](#dict)</code>) – Results from sensitivity analysis.

**Functions:**

- [**summary**](#causalis.data.CausalEstimate.summary) – Return a summary DataFrame of the results.

**Attributes:**

- [**alpha**](#causalis.data.CausalEstimate.alpha) (<code>[float](#float)</code>) –
- [**ci_lower_absolute**](#causalis.data.CausalEstimate.ci_lower_absolute) (<code>[float](#float)</code>) –
- [**ci_lower_relative**](#causalis.data.CausalEstimate.ci_lower_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**ci_upper_absolute**](#causalis.data.CausalEstimate.ci_upper_absolute) (<code>[float](#float)</code>) –
- [**ci_upper_relative**](#causalis.data.CausalEstimate.ci_upper_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**confounders**](#causalis.data.CausalEstimate.confounders) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**diagnostic_data**](#causalis.data.CausalEstimate.diagnostic_data) (<code>[Optional](#typing.Optional)\[[DiagnosticData](#causalis.data.causal_diagnostic_data.DiagnosticData)\]</code>) –
- [**estimand**](#causalis.data.CausalEstimate.estimand) (<code>[str](#str)</code>) –
- [**instrument**](#causalis.data.CausalEstimate.instrument) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**is_significant**](#causalis.data.CausalEstimate.is_significant) (<code>[bool](#bool)</code>) –
- [**model**](#causalis.data.CausalEstimate.model) (<code>[str](#str)</code>) –
- [**model_config**](#causalis.data.CausalEstimate.model_config) –
- [**model_options**](#causalis.data.CausalEstimate.model_options) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**n_control**](#causalis.data.CausalEstimate.n_control) (<code>[int](#int)</code>) –
- [**n_treated**](#causalis.data.CausalEstimate.n_treated) (<code>[int](#int)</code>) –
- [**outcome**](#causalis.data.CausalEstimate.outcome) (<code>[str](#str)</code>) –
- [**p_value**](#causalis.data.CausalEstimate.p_value) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**sensitivity_analysis**](#causalis.data.CausalEstimate.sensitivity_analysis) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**time**](#causalis.data.CausalEstimate.time) (<code>[datetime](#datetime.datetime)</code>) –
- [**treatment**](#causalis.data.CausalEstimate.treatment) (<code>[str](#str)</code>) –
- [**value**](#causalis.data.CausalEstimate.value) (<code>[float](#float)</code>) –
- [**value_relative**](#causalis.data.CausalEstimate.value_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –

##### `causalis.data.CausalEstimate.alpha`

```python
alpha: float
```

##### `causalis.data.CausalEstimate.ci_lower_absolute`

```python
ci_lower_absolute: float
```

##### `causalis.data.CausalEstimate.ci_lower_relative`

```python
ci_lower_relative: Optional[float] = None
```

##### `causalis.data.CausalEstimate.ci_upper_absolute`

```python
ci_upper_absolute: float
```

##### `causalis.data.CausalEstimate.ci_upper_relative`

```python
ci_upper_relative: Optional[float] = None
```

##### `causalis.data.CausalEstimate.confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

##### `causalis.data.CausalEstimate.diagnostic_data`

```python
diagnostic_data: Optional[DiagnosticData] = None
```

##### `causalis.data.CausalEstimate.estimand`

```python
estimand: str
```

##### `causalis.data.CausalEstimate.instrument`

```python
instrument: List[str] = Field(default_factory=list)
```

##### `causalis.data.CausalEstimate.is_significant`

```python
is_significant: bool
```

##### `causalis.data.CausalEstimate.model`

```python
model: str
```

##### `causalis.data.CausalEstimate.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `causalis.data.CausalEstimate.model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

##### `causalis.data.CausalEstimate.n_control`

```python
n_control: int
```

##### `causalis.data.CausalEstimate.n_treated`

```python
n_treated: int
```

##### `causalis.data.CausalEstimate.outcome`

```python
outcome: str
```

##### `causalis.data.CausalEstimate.p_value`

```python
p_value: Optional[float] = None
```

##### `causalis.data.CausalEstimate.sensitivity_analysis`

```python
sensitivity_analysis: Dict[str, Any] = Field(default_factory=dict)
```

##### `causalis.data.CausalEstimate.summary`

```python
summary()
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Summary DataFrame.

##### `causalis.data.CausalEstimate.time`

```python
time: datetime = Field(default_factory=(datetime.now))
```

##### `causalis.data.CausalEstimate.treatment`

```python
treatment: str
```

##### `causalis.data.CausalEstimate.value`

```python
value: float
```

##### `causalis.data.CausalEstimate.value_relative`

```python
value_relative: Optional[float] = None
```

#### `causalis.data.DiagnosticData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Base class for all diagnostic data.

**Attributes:**

- [**model_config**](#causalis.data.DiagnosticData.model_config) –

##### `causalis.data.DiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

#### `causalis.data.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.data.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.data.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.data.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.data.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.data.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.data.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.data.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.data.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.data.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**include_oracle**](#causalis.data.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.data.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**outcome_type**](#causalis.data.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.data.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.data.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**seed**](#causalis.data.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.data.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.data.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.data.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.data.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.data.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.data.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.data.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.data.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

##### `causalis.data.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

##### `causalis.data.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

##### `causalis.data.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

##### `causalis.data.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

##### `causalis.data.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

##### `causalis.data.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

##### `causalis.data.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data.SmokingDGP.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

##### `causalis.data.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

##### `causalis.data.SmokingDGP.k`

```python
k: int = 5
```

##### `causalis.data.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

##### `causalis.data.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

##### `causalis.data.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

##### `causalis.data.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

##### `causalis.data.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

##### `causalis.data.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

##### `causalis.data.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

##### `causalis.data.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data.SmokingDGP.theta`

```python
theta: float = 1.0
```

##### `causalis.data.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

##### `causalis.data.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

##### `causalis.data.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

##### `causalis.data.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

##### `causalis.data.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

#### `causalis.data.UnconfoundednessDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data.causal_diagnostic_data.DiagnosticData)</code>

Fields common to all models assuming unconfoundedness.

**Attributes:**

- [**d**](#causalis.data.UnconfoundednessDiagnosticData.d) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**m_hat**](#causalis.data.UnconfoundednessDiagnosticData.m_hat) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**model_config**](#causalis.data.UnconfoundednessDiagnosticData.model_config) –
- [**trimming_threshold**](#causalis.data.UnconfoundednessDiagnosticData.trimming_threshold) (<code>[float](#float)</code>) –
- [**x**](#causalis.data.UnconfoundednessDiagnosticData.x) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**y**](#causalis.data.UnconfoundednessDiagnosticData.y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –

##### `causalis.data.UnconfoundednessDiagnosticData.d`

```python
d: np.ndarray
```

##### `causalis.data.UnconfoundednessDiagnosticData.m_hat`

```python
m_hat: np.ndarray
```

##### `causalis.data.UnconfoundednessDiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `causalis.data.UnconfoundednessDiagnosticData.trimming_threshold`

```python
trimming_threshold: float = 0.0
```

##### `causalis.data.UnconfoundednessDiagnosticData.x`

```python
x: Optional[np.ndarray] = None
```

##### `causalis.data.UnconfoundednessDiagnosticData.y`

```python
y: Optional[np.ndarray] = None
```

#### `causalis.data.causal_diagnostic_data`

**Classes:**

- [**DiagnosticData**](#causalis.data.causal_diagnostic_data.DiagnosticData) – Base class for all diagnostic data.
- [**UnconfoundednessDiagnosticData**](#causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData) – Fields common to all models assuming unconfoundedness.

##### `causalis.data.causal_diagnostic_data.DiagnosticData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Base class for all diagnostic data.

**Attributes:**

- [**model_config**](#causalis.data.causal_diagnostic_data.DiagnosticData.model_config) –

###### `causalis.data.causal_diagnostic_data.DiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data.causal_diagnostic_data.DiagnosticData)</code>

Fields common to all models assuming unconfoundedness.

**Attributes:**

- [**d**](#causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.d) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**m_hat**](#causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.m_hat) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**model_config**](#causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.model_config) –
- [**trimming_threshold**](#causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.trimming_threshold) (<code>[float](#float)</code>) –
- [**x**](#causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.x) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**y**](#causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –

###### `causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.d`

```python
d: np.ndarray
```

###### `causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.m_hat`

```python
m_hat: np.ndarray
```

###### `causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.trimming_threshold`

```python
trimming_threshold: float = 0.0
```

###### `causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.x`

```python
x: Optional[np.ndarray] = None
```

###### `causalis.data.causal_diagnostic_data.UnconfoundednessDiagnosticData.y`

```python
y: Optional[np.ndarray] = None
```

#### `causalis.data.causal_estimate`

**Classes:**

- [**CausalEstimate**](#causalis.data.causal_estimate.CausalEstimate) – Result container for causal effect estimates.

##### `causalis.data.causal_estimate.CausalEstimate`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Result container for causal effect estimates.

**Parameters:**

- **estimand** (<code>[str](#str)</code>) – The estimand being estimated (e.g., 'ATE', 'ATTE', 'CATE').
- **model** (<code>[str](#str)</code>) – The name of the model used for estimation.
- **model_options** (<code>[dict](#dict)</code>) – Options passed to the model.
- **value** (<code>[float](#float)</code>) – The estimated absolute effect.
- **ci_upper_absolute** (<code>[float](#float)</code>) – Upper bound of the absolute confidence interval.
- **ci_lower_absolute** (<code>[float](#float)</code>) – Lower bound of the absolute confidence interval.
- **value_relative** (<code>[float](#float)</code>) – The estimated relative effect.
- **ci_upper_relative** (<code>[float](#float)</code>) – Upper bound of the relative confidence interval.
- **ci_lower_relative** (<code>[float](#float)</code>) – Lower bound of the relative confidence interval.
- **alpha** (<code>[float](#float)</code>) – The significance level (e.g., 0.05).
- **p_value** (<code>[float](#float)</code>) – The p-value from the test.
- **is_significant** (<code>[bool](#bool)</code>) – Whether the result is statistically significant at alpha.
- **n_treated** (<code>[int](#int)</code>) – Number of units in the treatment group.
- **n_control** (<code>[int](#int)</code>) – Number of units in the control group.
- **outcome** (<code>[str](#str)</code>) – The name of the outcome variable.
- **treatment** (<code>[str](#str)</code>) – The name of the treatment variable.
- **confounders** (<code>list of str</code>) – The names of the confounders used in the model.
- **instrument** (<code>list of str</code>) – The names of the instruments used in the model.
- **time** (<code>[datetime](#datetime.datetime)</code>) – The date and time when the estimate was created.
- **diagnostic_data** (<code>[DiagnosticData](#causalis.data.causal_diagnostic_data.DiagnosticData)</code>) – Additional diagnostic data.
- **sensitivity_analysis** (<code>[dict](#dict)</code>) – Results from sensitivity analysis.

**Functions:**

- [**summary**](#causalis.data.causal_estimate.CausalEstimate.summary) – Return a summary DataFrame of the results.

**Attributes:**

- [**alpha**](#causalis.data.causal_estimate.CausalEstimate.alpha) (<code>[float](#float)</code>) –
- [**ci_lower_absolute**](#causalis.data.causal_estimate.CausalEstimate.ci_lower_absolute) (<code>[float](#float)</code>) –
- [**ci_lower_relative**](#causalis.data.causal_estimate.CausalEstimate.ci_lower_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**ci_upper_absolute**](#causalis.data.causal_estimate.CausalEstimate.ci_upper_absolute) (<code>[float](#float)</code>) –
- [**ci_upper_relative**](#causalis.data.causal_estimate.CausalEstimate.ci_upper_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**confounders**](#causalis.data.causal_estimate.CausalEstimate.confounders) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**diagnostic_data**](#causalis.data.causal_estimate.CausalEstimate.diagnostic_data) (<code>[Optional](#typing.Optional)\[[DiagnosticData](#causalis.data.causal_diagnostic_data.DiagnosticData)\]</code>) –
- [**estimand**](#causalis.data.causal_estimate.CausalEstimate.estimand) (<code>[str](#str)</code>) –
- [**instrument**](#causalis.data.causal_estimate.CausalEstimate.instrument) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**is_significant**](#causalis.data.causal_estimate.CausalEstimate.is_significant) (<code>[bool](#bool)</code>) –
- [**model**](#causalis.data.causal_estimate.CausalEstimate.model) (<code>[str](#str)</code>) –
- [**model_config**](#causalis.data.causal_estimate.CausalEstimate.model_config) –
- [**model_options**](#causalis.data.causal_estimate.CausalEstimate.model_options) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**n_control**](#causalis.data.causal_estimate.CausalEstimate.n_control) (<code>[int](#int)</code>) –
- [**n_treated**](#causalis.data.causal_estimate.CausalEstimate.n_treated) (<code>[int](#int)</code>) –
- [**outcome**](#causalis.data.causal_estimate.CausalEstimate.outcome) (<code>[str](#str)</code>) –
- [**p_value**](#causalis.data.causal_estimate.CausalEstimate.p_value) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**sensitivity_analysis**](#causalis.data.causal_estimate.CausalEstimate.sensitivity_analysis) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**time**](#causalis.data.causal_estimate.CausalEstimate.time) (<code>[datetime](#datetime.datetime)</code>) –
- [**treatment**](#causalis.data.causal_estimate.CausalEstimate.treatment) (<code>[str](#str)</code>) –
- [**value**](#causalis.data.causal_estimate.CausalEstimate.value) (<code>[float](#float)</code>) –
- [**value_relative**](#causalis.data.causal_estimate.CausalEstimate.value_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –

###### `causalis.data.causal_estimate.CausalEstimate.alpha`

```python
alpha: float
```

###### `causalis.data.causal_estimate.CausalEstimate.ci_lower_absolute`

```python
ci_lower_absolute: float
```

###### `causalis.data.causal_estimate.CausalEstimate.ci_lower_relative`

```python
ci_lower_relative: Optional[float] = None
```

###### `causalis.data.causal_estimate.CausalEstimate.ci_upper_absolute`

```python
ci_upper_absolute: float
```

###### `causalis.data.causal_estimate.CausalEstimate.ci_upper_relative`

```python
ci_upper_relative: Optional[float] = None
```

###### `causalis.data.causal_estimate.CausalEstimate.confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

###### `causalis.data.causal_estimate.CausalEstimate.diagnostic_data`

```python
diagnostic_data: Optional[DiagnosticData] = None
```

###### `causalis.data.causal_estimate.CausalEstimate.estimand`

```python
estimand: str
```

###### `causalis.data.causal_estimate.CausalEstimate.instrument`

```python
instrument: List[str] = Field(default_factory=list)
```

###### `causalis.data.causal_estimate.CausalEstimate.is_significant`

```python
is_significant: bool
```

###### `causalis.data.causal_estimate.CausalEstimate.model`

```python
model: str
```

###### `causalis.data.causal_estimate.CausalEstimate.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `causalis.data.causal_estimate.CausalEstimate.model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

###### `causalis.data.causal_estimate.CausalEstimate.n_control`

```python
n_control: int
```

###### `causalis.data.causal_estimate.CausalEstimate.n_treated`

```python
n_treated: int
```

###### `causalis.data.causal_estimate.CausalEstimate.outcome`

```python
outcome: str
```

###### `causalis.data.causal_estimate.CausalEstimate.p_value`

```python
p_value: Optional[float] = None
```

###### `causalis.data.causal_estimate.CausalEstimate.sensitivity_analysis`

```python
sensitivity_analysis: Dict[str, Any] = Field(default_factory=dict)
```

###### `causalis.data.causal_estimate.CausalEstimate.summary`

```python
summary()
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Summary DataFrame.

###### `causalis.data.causal_estimate.CausalEstimate.time`

```python
time: datetime = Field(default_factory=(datetime.now))
```

###### `causalis.data.causal_estimate.CausalEstimate.treatment`

```python
treatment: str
```

###### `causalis.data.causal_estimate.CausalEstimate.value`

```python
value: float
```

###### `causalis.data.causal_estimate.CausalEstimate.value_relative`

```python
value_relative: Optional[float] = None
```

#### `causalis.data.causaldata`

Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.

**Classes:**

- [**CausalData**](#causalis.data.causaldata.CausalData) – Container for causal inference datasets.

##### `causalis.data.causaldata.CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data contract.

**Attributes:**

- [**df**](#causalis.data.causaldata.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.data.causaldata.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.data.causaldata.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.data.causaldata.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.data.causaldata.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.data.causaldata.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.data.causaldata.CausalData.get_df) – Get a DataFrame with specified columns.

###### `causalis.data.causaldata.CausalData.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

###### `causalis.data.causaldata.CausalData.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

###### `causalis.data.causaldata.CausalData.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

###### `causalis.data.causaldata.CausalData.df`

```python
df: pd.DataFrame
```

###### `causalis.data.causaldata.CausalData.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, **kwargs)
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A validated CausalData instance.

###### `causalis.data.causaldata.CausalData.get_df`

```python
get_df(columns=None, include_treatment=True, include_outcome=True, include_confounders=True, include_user_id=False)
```

Get a DataFrame with specified columns.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

###### `causalis.data.causaldata.CausalData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

###### `causalis.data.causaldata.CausalData.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

###### `causalis.data.causaldata.CausalData.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

###### `causalis.data.causaldata.CausalData.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

###### `causalis.data.causaldata.CausalData.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

###### `causalis.data.causaldata.CausalData.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

###### `causalis.data.causaldata.CausalData.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data.causaldata_instrumental`

**Classes:**

- [**CausalDataInstrumental**](#causalis.data.causaldata_instrumental.CausalDataInstrumental) – Container for causal inference datasets with causaldata_instrumental variables.

##### `causalis.data.causaldata_instrumental.CausalDataInstrumental`

Bases: <code>[CausalData](#causalis.data.causaldata.CausalData)</code>

Container for causal inference datasets with causaldata_instrumental variables.

**Attributes:**

- [**instrument_name**](#causalis.data.causaldata_instrumental.CausalDataInstrumental.instrument_name) (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.

**Functions:**

- [**from_df**](#causalis.data.causaldata_instrumental.CausalDataInstrumental.from_df) – Friendly constructor for CausalDataInstrumental.
- [**get_df**](#causalis.data.causaldata_instrumental.CausalDataInstrumental.get_df) – Get a DataFrame with specified columns including instrument.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.df`

```python
df: pd.DataFrame
```

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, instrument=None, **kwargs)
```

Friendly constructor for CausalDataInstrumental.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- **instrument** (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalDataInstrumental](#causalis.data.causaldata_instrumental.CausalDataInstrumental)</code> – A validated CausalDataInstrumental instance.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.get_df`

```python
get_df(columns=None, include_treatment=True, include_outcome=True, include_confounders=True, include_user_id=False, include_instrument=False)
```

Get a DataFrame with specified columns including instrument.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.
- **include_instrument** (<code>[bool](#bool)</code>) – Whether to include the instrument column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.instrument`

```python
instrument: pd.Series
```

instrument column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The instrument column.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.instrument_name`

```python
instrument_name: str = Field(alias='instrument')
```

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

###### `causalis.data.causaldata_instrumental.CausalDataInstrumental.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data.dgps`

**Modules:**

- [**base**](#causalis.data.dgps.base) –
- [**causaldata**](#causalis.data.dgps.causaldata) –
- [**causaldata_instrumental**](#causalis.data.dgps.causaldata_instrumental) –

**Classes:**

- [**CausalDatasetGenerator**](#causalis.data.dgps.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,
- [**SmokingDGP**](#causalis.data.dgps.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.

**Functions:**

- [**generate_classic_rct**](#causalis.data.dgps.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.data.dgps.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_iv_data**](#causalis.data.dgps.generate_iv_data) – Generate synthetic dataset with instrumental variables.
- [**generate_rct**](#causalis.data.dgps.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_gold_linear**](#causalis.data.dgps.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.data.dgps.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.data.dgps.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

##### `causalis.data.dgps.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, include_oracle=True, seed=None)
```

Generate synthetic causal inference datasets with controllable confounding,
treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

**Data model (high level)**

- confounders X ∈ R^k are drawn from user-specified distributions.
- Binary treatment D is assigned by a logistic model:
  D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
  where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
- Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
  outcome_type = "continuous":
  Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε, ε ~ N(0, sigma_y^2)
  outcome_type = "binary":
  logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "poisson":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

**Returned columns**

- y: outcome
- d: binary treatment (0/1)
- x1..xk (or user-provided names)
- m: true propensity P(T=1 | X) marginalized over U
- m_obs: realized propensity P(T=1 | X, U)
- tau_link: tau(X) on the structural (link) scale
- g0: E[Y | X, T=0] on the natural outcome scale marginalized over U
- g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
- cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

Notes on effect scale:

- For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
- For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
- For "poisson", tau acts on the *log-mean* scale. `cate` is reported on the mean (rate) scale.

**Parameters:**

- **theta** (<code>[float](#float)</code>) – Constant treatment effect used if `tau` is None.
- **tau** (<code>[callable](#callable)</code>) – Function tau(X) -> array-like shape (n,) for heterogeneous effects.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the outcome baseline f_y(X).
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the treatment score f_d(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
- **g_d** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson).
- **alpha_d** (<code>[float](#float)</code>) – Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
- **sigma_y** (<code>[float](#float)</code>) – Std. dev. of the Gaussian noise for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Outcome family and link as defined above.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders. See `_gaussian_copula` for details.
- **k** (<code>[int](#int)</code>) – Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
- **use_copula** (<code>[bool](#bool)</code>) – If True and `confounder_specs` provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
- **u_strength_d** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in treatment assignment.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in the outcome.
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales the X-driven treatment score to adjust positivity difficulty.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Attributes:**

- [**rng**](#causalis.data.dgps.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.data.dgps.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.dgps.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.dgps.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

###### `causalis.data.dgps.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

###### `causalis.data.dgps.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

###### `causalis.data.dgps.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

###### `causalis.data.dgps.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

###### `causalis.data.dgps.CausalDatasetGenerator.k`

```python
k: int = 5
```

###### `causalis.data.dgps.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

###### `causalis.data.dgps.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

###### `causalis.data.dgps.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

###### `causalis.data.dgps.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

###### `causalis.data.dgps.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

###### `causalis.data.dgps.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.data.dgps.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

###### `causalis.data.dgps.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

###### `causalis.data.dgps.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

###### `causalis.data.dgps.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

###### `causalis.data.dgps.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

###### `causalis.data.dgps.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `causalis.data.dgps.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.data.dgps.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.dgps.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.dgps.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.data.dgps.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.data.dgps.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.data.dgps.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.data.dgps.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.data.dgps.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.data.dgps.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.data.dgps.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.data.dgps.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**include_oracle**](#causalis.data.dgps.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.data.dgps.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**outcome_type**](#causalis.data.dgps.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.data.dgps.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.data.dgps.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**seed**](#causalis.data.dgps.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.data.dgps.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.data.dgps.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.data.dgps.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.data.dgps.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.data.dgps.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.data.dgps.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.data.dgps.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.data.dgps.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

###### `causalis.data.dgps.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

###### `causalis.data.dgps.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

###### `causalis.data.dgps.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

###### `causalis.data.dgps.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

###### `causalis.data.dgps.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

###### `causalis.data.dgps.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

###### `causalis.data.dgps.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.data.dgps.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.data.dgps.SmokingDGP.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

###### `causalis.data.dgps.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

###### `causalis.data.dgps.SmokingDGP.k`

```python
k: int = 5
```

###### `causalis.data.dgps.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

###### `causalis.data.dgps.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

###### `causalis.data.dgps.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

###### `causalis.data.dgps.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

###### `causalis.data.dgps.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

###### `causalis.data.dgps.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

###### `causalis.data.dgps.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

###### `causalis.data.dgps.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.data.dgps.SmokingDGP.theta`

```python
theta: float = 1.0
```

###### `causalis.data.dgps.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

###### `causalis.data.dgps.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

###### `causalis.data.dgps.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

###### `causalis.data.dgps.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

###### `causalis.data.dgps.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `causalis.data.dgps.base`

##### `causalis.data.dgps.causaldata`

**Modules:**

- [**base**](#causalis.data.dgps.causaldata.base) –
- [**functional**](#causalis.data.dgps.causaldata.functional) –
- [**gold_library**](#causalis.data.dgps.causaldata.gold_library) –

**Classes:**

- [**CausalDatasetGenerator**](#causalis.data.dgps.causaldata.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,
- [**SmokingDGP**](#causalis.data.dgps.causaldata.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.

**Functions:**

- [**generate_classic_rct**](#causalis.data.dgps.causaldata.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.data.dgps.causaldata.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_rct**](#causalis.data.dgps.causaldata.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_gold_linear**](#causalis.data.dgps.causaldata.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.data.dgps.causaldata.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.data.dgps.causaldata.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

###### `causalis.data.dgps.causaldata.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, include_oracle=True, seed=None)
```

Generate synthetic causal inference datasets with controllable confounding,
treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

**Data model (high level)**

- confounders X ∈ R^k are drawn from user-specified distributions.
- Binary treatment D is assigned by a logistic model:
  D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
  where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
- Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
  outcome_type = "continuous":
  Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε, ε ~ N(0, sigma_y^2)
  outcome_type = "binary":
  logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "poisson":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

**Returned columns**

- y: outcome
- d: binary treatment (0/1)
- x1..xk (or user-provided names)
- m: true propensity P(T=1 | X) marginalized over U
- m_obs: realized propensity P(T=1 | X, U)
- tau_link: tau(X) on the structural (link) scale
- g0: E[Y | X, T=0] on the natural outcome scale marginalized over U
- g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
- cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

Notes on effect scale:

- For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
- For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
- For "poisson", tau acts on the *log-mean* scale. `cate` is reported on the mean (rate) scale.

**Parameters:**

- **theta** (<code>[float](#float)</code>) – Constant treatment effect used if `tau` is None.
- **tau** (<code>[callable](#callable)</code>) – Function tau(X) -> array-like shape (n,) for heterogeneous effects.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the outcome baseline f_y(X).
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the treatment score f_d(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
- **g_d** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson).
- **alpha_d** (<code>[float](#float)</code>) – Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
- **sigma_y** (<code>[float](#float)</code>) – Std. dev. of the Gaussian noise for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Outcome family and link as defined above.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders. See `_gaussian_copula` for details.
- **k** (<code>[int](#int)</code>) – Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
- **use_copula** (<code>[bool](#bool)</code>) – If True and `confounder_specs` provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
- **u_strength_d** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in treatment assignment.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in the outcome.
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales the X-driven treatment score to adjust positivity difficulty.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Attributes:**

- [**rng**](#causalis.data.dgps.causaldata.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.data.dgps.causaldata.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.dgps.causaldata.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.dgps.causaldata.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.k`

```python
k: int = 5
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

####### `causalis.data.dgps.causaldata.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

###### `causalis.data.dgps.causaldata.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.data.dgps.causaldata.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.dgps.causaldata.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.dgps.causaldata.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.data.dgps.causaldata.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.data.dgps.causaldata.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.data.dgps.causaldata.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.data.dgps.causaldata.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.data.dgps.causaldata.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.data.dgps.causaldata.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.data.dgps.causaldata.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.data.dgps.causaldata.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**include_oracle**](#causalis.data.dgps.causaldata.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.data.dgps.causaldata.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**outcome_type**](#causalis.data.dgps.causaldata.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.data.dgps.causaldata.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.data.dgps.causaldata.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**seed**](#causalis.data.dgps.causaldata.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.data.dgps.causaldata.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.data.dgps.causaldata.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.data.dgps.causaldata.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.data.dgps.causaldata.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.data.dgps.causaldata.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.data.dgps.causaldata.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.data.dgps.causaldata.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.data.dgps.causaldata.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

####### `causalis.data.dgps.causaldata.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

####### `causalis.data.dgps.causaldata.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

####### `causalis.data.dgps.causaldata.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

####### `causalis.data.dgps.causaldata.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

####### `causalis.data.dgps.causaldata.SmokingDGP.k`

```python
k: int = 5
```

####### `causalis.data.dgps.causaldata.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

####### `causalis.data.dgps.causaldata.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

####### `causalis.data.dgps.causaldata.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

####### `causalis.data.dgps.causaldata.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

####### `causalis.data.dgps.causaldata.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

####### `causalis.data.dgps.causaldata.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.data.dgps.causaldata.SmokingDGP.theta`

```python
theta: float = 1.0
```

####### `causalis.data.dgps.causaldata.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

####### `causalis.data.dgps.causaldata.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

####### `causalis.data.dgps.causaldata.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

####### `causalis.data.dgps.causaldata.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

####### `causalis.data.dgps.causaldata.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

###### `causalis.data.dgps.causaldata.base`

**Classes:**

- [**CausalDatasetGenerator**](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,

####### `causalis.data.dgps.causaldata.base.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, include_oracle=True, seed=None)
```

Generate synthetic causal inference datasets with controllable confounding,
treatment prevalence, noise, and (optionally) heterogeneous treatment effects.

**Data model (high level)**

- confounders X ∈ R^k are drawn from user-specified distributions.
- Binary treatment D is assigned by a logistic model:
  D ~ Bernoulli( sigmoid(alpha_d + f_d(X) + u_strength_d * U) ),
  where f_d(X) = (X @ beta_d + g_d(X)) * propensity_sharpness, and U ~ N(0,1) is an optional unobserved confounder.
- Outcome Y depends on treatment and confounders with link determined by `outcome_type`:
  outcome_type = "continuous":
  Y = alpha_y + f_y(X) + u_strength_y * U + T * tau(X) + ε, ε ~ N(0, sigma_y^2)
  outcome_type = "binary":
  logit P(Y=1|T,X,U) = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  outcome_type = "poisson":
  log E[Y|T,X,U] = alpha_y + f_y(X) + u_strength_y * U + T * tau(X)
  where f_y(X) = X @ beta_y + g_y(X), and tau(X) is either constant `theta` or a user function.

**Returned columns**

- y: outcome
- d: binary treatment (0/1)
- x1..xk (or user-provided names)
- m: true propensity P(T=1 | X) marginalized over U
- m_obs: realized propensity P(T=1 | X, U)
- tau_link: tau(X) on the structural (link) scale
- g0: E[Y | X, T=0] on the natural outcome scale marginalized over U
- g1: E[Y | X, T=1] on the natural outcome scale marginalized over U
- cate: g1 - g0 (conditional average treatment effect on the natural outcome scale)

Notes on effect scale:

- For "continuous", `theta` (or tau(X)) is an additive mean difference, so `tau_link == cate`.
- For "binary", tau acts on the *log-odds* scale. `cate` is reported as a risk difference.
- For "poisson", tau acts on the *log-mean* scale. `cate` is reported on the mean (rate) scale.

**Parameters:**

- **theta** (<code>[float](#float)</code>) – Constant treatment effect used if `tau` is None.
- **tau** (<code>[callable](#callable)</code>) – Function tau(X) -> array-like shape (n,) for heterogeneous effects.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the outcome baseline f_y(X).
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients of confounders in the treatment score f_d(X).
- **g_y** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_y(X) -> (n,) added to the outcome baseline.
- **g_d** (<code>[callable](#callable)</code>) – Nonlinear/additive function g_d(X) -> (n,) added to the treatment score.
- **alpha_y** (<code>[float](#float)</code>) – Outcome intercept (natural scale for continuous; log-odds for binary; log-mean for Poisson).
- **alpha_d** (<code>[float](#float)</code>) – Treatment intercept (log-odds). If `target_d_rate` is set, `alpha_d` is auto-calibrated.
- **sigma_y** (<code>[float](#float)</code>) – Std. dev. of the Gaussian noise for continuous outcomes.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Outcome family and link as defined above.
- **confounder_specs** (<code>list of dict</code>) – Schema for generating confounders. See `_gaussian_copula` for details.
- **k** (<code>[int](#int)</code>) – Number of confounders when `confounder_specs` is None. Defaults to independent N(0,1).
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler (n, k, seed) -> X ndarray of shape (n,k). Overrides `confounder_specs`.
- **use_copula** (<code>[bool](#bool)</code>) – If True and `confounder_specs` provided, use Gaussian copula for X.
- **copula_corr** (<code>[array](#array) - [like](#like)</code>) – Correlation matrix for copula.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean). Calibrates `alpha_d`.
- **u_strength_d** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in treatment assignment.
- **u_strength_y** (<code>[float](#float)</code>) – Strength of the unobserved confounder U in the outcome.
- **propensity_sharpness** (<code>[float](#float)</code>) – Scales the X-driven treatment score to adjust positivity difficulty.
- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Attributes:**

- [**rng**](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.k`

```python
k: int = 5
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

######## `causalis.data.dgps.causaldata.base.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

###### `causalis.data.dgps.causaldata.functional`

**Functions:**

- [**generate_classic_rct**](#causalis.data.dgps.causaldata.functional.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_rct**](#causalis.data.dgps.causaldata.functional.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**obs_linear_effect**](#causalis.data.dgps.causaldata.functional.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

####### `causalis.data.dgps.causaldata.functional.generate_classic_rct`

```python
generate_classic_rct(n=10000, split=0.5, random_state=42, outcome_params=None, add_pre=False, beta_y=None, outcome_depends_on_x=True, prognostic_scale=1.0, pre_corr=0.7, return_causal_data=False, **kwargs)
```

Generate a classic RCT dataset with three binary confounders:
platform_ios, country_usa, and source_paid.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal (passed to generate_rct).
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre (passed to generate_rct).
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- \*\***kwargs** – Additional arguments passed to `generate_rct`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

####### `causalis.data.dgps.causaldata.functional.generate_rct`

```python
generate_rct(n=20000, split=0.5, random_state=42, outcome_type='binary', outcome_params=None, confounder_specs=None, k=0, x_sampler=None, add_ancillary=True, deterministic_ids=False, add_pre=True, pre_name='y_pre', pre_corr=0.7, prognostic_scale=1.0, beta_y=None, g_y=None, use_prognostic=None, include_oracle=True, return_causal_data=False)
```

Generate an RCT dataset with randomized treatment assignment.

Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
Specifically designed for benchmarking variance reduction techniques like CUPED.

**Notes on effect scale**

How `outcome_params` maps into the structural effect:

- outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
- outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
- outcome_type="poisson": treatment shifts the log-mean by log(lam_B / lam_A).

Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
avoiding outcome leakage and post-treatment adjustment issues.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_type** (<code>('binary', 'normal', 'poisson')</code>) – Distribution family of the outcome.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.12}} for binary.
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add descriptive columns like 'age', 'platform', etc.
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation between `y_pre` and the outcome Y in the control group.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of the prognostic signal derived from confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic RCT dataset.

####### `causalis.data.dgps.causaldata.functional.obs_linear_effect`

```python
obs_linear_effect(n=10000, theta=1.0, outcome_type='continuous', sigma_y=1.0, target_d_rate=None, confounder_specs=None, beta_y=None, beta_d=None, random_state=42, k=0, x_sampler=None, include_oracle=True, add_ancillary=False, deterministic_ids=False)
```

Generate an observational dataset with linear effects of confounders and a constant treatment effect.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **theta** (<code>[float](#float)</code>) – Constant treatment effect.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Family of the outcome distribution.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the treatment model.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **add_ancillary** (<code>[bool](#bool)</code>) – If True, adds standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – If True, generates deterministic user IDs.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic observational dataset.

###### `causalis.data.dgps.causaldata.generate_classic_rct`

```python
generate_classic_rct(n=10000, split=0.5, random_state=42, outcome_params=None, add_pre=False, beta_y=None, outcome_depends_on_x=True, prognostic_scale=1.0, pre_corr=0.7, return_causal_data=False, **kwargs)
```

Generate a classic RCT dataset with three binary confounders:
platform_ios, country_usa, and source_paid.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal (passed to generate_rct).
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre (passed to generate_rct).
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- \*\***kwargs** – Additional arguments passed to `generate_rct`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

###### `causalis.data.dgps.causaldata.generate_classic_rct_26`

```python
generate_classic_rct_26(seed=42, add_pre=False, beta_y=None, outcome_depends_on_x=True, include_oracle=False, return_causal_data=True)
```

A pre-configured classic RCT dataset with 3 binary confounders.
n=10000, split=0.5, outcome is conversion (binary), real effect = 0.01.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

###### `causalis.data.dgps.causaldata.generate_rct`

```python
generate_rct(n=20000, split=0.5, random_state=42, outcome_type='binary', outcome_params=None, confounder_specs=None, k=0, x_sampler=None, add_ancillary=True, deterministic_ids=False, add_pre=True, pre_name='y_pre', pre_corr=0.7, prognostic_scale=1.0, beta_y=None, g_y=None, use_prognostic=None, include_oracle=True, return_causal_data=False)
```

Generate an RCT dataset with randomized treatment assignment.

Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
Specifically designed for benchmarking variance reduction techniques like CUPED.

**Notes on effect scale**

How `outcome_params` maps into the structural effect:

- outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
- outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
- outcome_type="poisson": treatment shifts the log-mean by log(lam_B / lam_A).

Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
avoiding outcome leakage and post-treatment adjustment issues.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_type** (<code>('binary', 'normal', 'poisson')</code>) – Distribution family of the outcome.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.12}} for binary.
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add descriptive columns like 'age', 'platform', etc.
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation between `y_pre` and the outcome Y in the control group.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of the prognostic signal derived from confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic RCT dataset.

###### `causalis.data.dgps.causaldata.gold_library`

**Classes:**

- [**SmokingDGP**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.

**Functions:**

- [**generate_classic_rct_26**](#causalis.data.dgps.causaldata.gold_library.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**make_gold_linear**](#causalis.data.dgps.causaldata.gold_library.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.data.dgps.causaldata.gold_library.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.

####### `causalis.data.dgps.causaldata.gold_library.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.data.dgps.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**include_oracle**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**outcome_type**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**seed**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.data.dgps.causaldata.gold_library.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.k`

```python
k: int = 5
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.theta`

```python
theta: float = 1.0
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

######## `causalis.data.dgps.causaldata.gold_library.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

####### `causalis.data.dgps.causaldata.gold_library.generate_classic_rct_26`

```python
generate_classic_rct_26(seed=42, add_pre=False, beta_y=None, outcome_depends_on_x=True, include_oracle=False, return_causal_data=True)
```

A pre-configured classic RCT dataset with 3 binary confounders.
n=10000, split=0.5, outcome is conversion (binary), real effect = 0.01.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

####### `causalis.data.dgps.causaldata.gold_library.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

####### `causalis.data.dgps.causaldata.gold_library.obs_linear_26_dataset`

```python
obs_linear_26_dataset(n=10000, seed=42, include_oracle=True, return_causal_data=True)
```

A pre-configured observational linear dataset with 5 standard confounders.
Based on the scenario in docs/cases/dml_ate.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

###### `causalis.data.dgps.causaldata.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

###### `causalis.data.dgps.causaldata.obs_linear_26_dataset`

```python
obs_linear_26_dataset(n=10000, seed=42, include_oracle=True, return_causal_data=True)
```

A pre-configured observational linear dataset with 5 standard confounders.
Based on the scenario in docs/cases/dml_ate.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

###### `causalis.data.dgps.causaldata.obs_linear_effect`

```python
obs_linear_effect(n=10000, theta=1.0, outcome_type='continuous', sigma_y=1.0, target_d_rate=None, confounder_specs=None, beta_y=None, beta_d=None, random_state=42, k=0, x_sampler=None, include_oracle=True, add_ancillary=False, deterministic_ids=False)
```

Generate an observational dataset with linear effects of confounders and a constant treatment effect.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **theta** (<code>[float](#float)</code>) – Constant treatment effect.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Family of the outcome distribution.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the treatment model.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **add_ancillary** (<code>[bool](#bool)</code>) – If True, adds standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – If True, generates deterministic user IDs.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic observational dataset.

##### `causalis.data.dgps.causaldata_instrumental`

**Modules:**

- [**base**](#causalis.data.dgps.causaldata_instrumental.base) –
- [**functional**](#causalis.data.dgps.causaldata_instrumental.functional) –

**Classes:**

- [**InstrumentalGenerator**](#causalis.data.dgps.causaldata_instrumental.InstrumentalGenerator) – Generator for synthetic causal inference datasets with instrumental variables.

**Functions:**

- [**generate_iv_data**](#causalis.data.dgps.causaldata_instrumental.generate_iv_data) – Generate synthetic dataset with instrumental variables.

###### `causalis.data.dgps.causaldata_instrumental.InstrumentalGenerator`

```python
InstrumentalGenerator(seed=None)
```

Generator for synthetic causal inference datasets with instrumental variables.

Placeholder implementation for future use.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Functions:**

- [**generate**](#causalis.data.dgps.causaldata_instrumental.InstrumentalGenerator.generate) – Draw a synthetic dataset of size `n`.

**Attributes:**

- [**seed**](#causalis.data.dgps.causaldata_instrumental.InstrumentalGenerator.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –

####### `causalis.data.dgps.causaldata_instrumental.InstrumentalGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – An empty DataFrame (placeholder).

####### `causalis.data.dgps.causaldata_instrumental.InstrumentalGenerator.seed`

```python
seed: Optional[int] = None
```

###### `causalis.data.dgps.causaldata_instrumental.base`

**Classes:**

- [**InstrumentalGenerator**](#causalis.data.dgps.causaldata_instrumental.base.InstrumentalGenerator) – Generator for synthetic causal inference datasets with instrumental variables.

####### `causalis.data.dgps.causaldata_instrumental.base.InstrumentalGenerator`

```python
InstrumentalGenerator(seed=None)
```

Generator for synthetic causal inference datasets with instrumental variables.

Placeholder implementation for future use.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Functions:**

- [**generate**](#causalis.data.dgps.causaldata_instrumental.base.InstrumentalGenerator.generate) – Draw a synthetic dataset of size `n`.

**Attributes:**

- [**seed**](#causalis.data.dgps.causaldata_instrumental.base.InstrumentalGenerator.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –

######## `causalis.data.dgps.causaldata_instrumental.base.InstrumentalGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – An empty DataFrame (placeholder).

######## `causalis.data.dgps.causaldata_instrumental.base.InstrumentalGenerator.seed`

```python
seed: Optional[int] = None
```

###### `causalis.data.dgps.causaldata_instrumental.functional`

**Functions:**

- [**generate_iv_data**](#causalis.data.dgps.causaldata_instrumental.functional.generate_iv_data) – Generate synthetic dataset with instrumental variables.

####### `causalis.data.dgps.causaldata_instrumental.functional.generate_iv_data`

```python
generate_iv_data(n=1000)
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

###### `causalis.data.dgps.causaldata_instrumental.generate_iv_data`

```python
generate_iv_data(n=1000)
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

##### `causalis.data.dgps.generate_classic_rct`

```python
generate_classic_rct(n=10000, split=0.5, random_state=42, outcome_params=None, add_pre=False, beta_y=None, outcome_depends_on_x=True, prognostic_scale=1.0, pre_corr=0.7, return_causal_data=False, **kwargs)
```

Generate a classic RCT dataset with three binary confounders:
platform_ios, country_usa, and source_paid.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal (passed to generate_rct).
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre (passed to generate_rct).
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- \*\***kwargs** – Additional arguments passed to `generate_rct`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

##### `causalis.data.dgps.generate_classic_rct_26`

```python
generate_classic_rct_26(seed=42, add_pre=False, beta_y=None, outcome_depends_on_x=True, include_oracle=False, return_causal_data=True)
```

A pre-configured classic RCT dataset with 3 binary confounders.
n=10000, split=0.5, outcome is conversion (binary), real effect = 0.01.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

##### `causalis.data.dgps.generate_iv_data`

```python
generate_iv_data(n=1000)
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

##### `causalis.data.dgps.generate_rct`

```python
generate_rct(n=20000, split=0.5, random_state=42, outcome_type='binary', outcome_params=None, confounder_specs=None, k=0, x_sampler=None, add_ancillary=True, deterministic_ids=False, add_pre=True, pre_name='y_pre', pre_corr=0.7, prognostic_scale=1.0, beta_y=None, g_y=None, use_prognostic=None, include_oracle=True, return_causal_data=False)
```

Generate an RCT dataset with randomized treatment assignment.

Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
Specifically designed for benchmarking variance reduction techniques like CUPED.

**Notes on effect scale**

How `outcome_params` maps into the structural effect:

- outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
- outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
- outcome_type="poisson": treatment shifts the log-mean by log(lam_B / lam_A).

Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
avoiding outcome leakage and post-treatment adjustment issues.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_type** (<code>('binary', 'normal', 'poisson')</code>) – Distribution family of the outcome.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.12}} for binary.
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add descriptive columns like 'age', 'platform', etc.
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation between `y_pre` and the outcome Y in the control group.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of the prognostic signal derived from confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic RCT dataset.

##### `causalis.data.dgps.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

##### `causalis.data.dgps.obs_linear_26_dataset`

```python
obs_linear_26_dataset(n=10000, seed=42, include_oracle=True, return_causal_data=True)
```

A pre-configured observational linear dataset with 5 standard confounders.
Based on the scenario in docs/cases/dml_ate.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

##### `causalis.data.dgps.obs_linear_effect`

```python
obs_linear_effect(n=10000, theta=1.0, outcome_type='continuous', sigma_y=1.0, target_d_rate=None, confounder_specs=None, beta_y=None, beta_d=None, random_state=42, k=0, x_sampler=None, include_oracle=True, add_ancillary=False, deterministic_ids=False)
```

Generate an observational dataset with linear effects of confounders and a constant treatment effect.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **theta** (<code>[float](#float)</code>) – Constant treatment effect.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Family of the outcome distribution.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the treatment model.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **add_ancillary** (<code>[bool](#bool)</code>) – If True, adds standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – If True, generates deterministic user IDs.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic observational dataset.

#### `causalis.data.generate_classic_rct`

```python
generate_classic_rct(n=10000, split=0.5, random_state=42, outcome_params=None, add_pre=False, beta_y=None, outcome_depends_on_x=True, prognostic_scale=1.0, pre_corr=0.7, return_causal_data=False, **kwargs)
```

Generate a classic RCT dataset with three binary confounders:
platform_ios, country_usa, and source_paid.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.15}} for binary.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of nonlinear prognostic signal (passed to generate_rct).
- **pre_corr** (<code>[float](#float)</code>) – Target correlation for y_pre (passed to generate_rct).
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.
- \*\***kwargs** – Additional arguments passed to `generate_rct`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

#### `causalis.data.generate_classic_rct_26`

```python
generate_classic_rct_26(seed=42, add_pre=False, beta_y=None, outcome_depends_on_x=True, include_oracle=False, return_causal_data=True)
```

A pre-configured classic RCT dataset with 3 binary confounders.
n=10000, split=0.5, outcome is conversion (binary), real effect = 0.01.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate ('y_pre') and include prognostic signal from X.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **outcome_depends_on_x** (<code>[bool](#bool)</code>) – Whether to add default effects for confounders if beta_y is None.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

#### `causalis.data.generate_rct`

```python
generate_rct(n=20000, split=0.5, random_state=42, outcome_type='binary', outcome_params=None, confounder_specs=None, k=0, x_sampler=None, add_ancillary=True, deterministic_ids=False, add_pre=True, pre_name='y_pre', pre_corr=0.7, prognostic_scale=1.0, beta_y=None, g_y=None, use_prognostic=None, include_oracle=True, return_causal_data=False)
```

Generate an RCT dataset with randomized treatment assignment.

Uses `CausalDatasetGenerator` internally, ensuring treatment is independent of X.
Specifically designed for benchmarking variance reduction techniques like CUPED.

**Notes on effect scale**

How `outcome_params` maps into the structural effect:

- outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
- outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
- outcome_type="poisson": treatment shifts the log-mean by log(lam_B / lam_A).

Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only,
avoiding outcome leakage and post-treatment adjustment issues.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **split** (<code>[float](#float)</code>) – Proportion of samples assigned to the treatment group.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **outcome_type** (<code>('binary', 'normal', 'poisson')</code>) – Distribution family of the outcome.
- **outcome_params** (<code>[dict](#dict)</code>) – Parameters defining baseline rates/means and treatment effects.
  e.g., {"p": {"A": 0.1, "B": 0.12}} for binary.
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **add_ancillary** (<code>[bool](#bool)</code>) – Whether to add descriptive columns like 'age', 'platform', etc.
- **deterministic_ids** (<code>[bool](#bool)</code>) – Whether to generate deterministic user IDs.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to generate a pre-period covariate (`y_pre`).
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_corr** (<code>[float](#float)</code>) – Target correlation between `y_pre` and the outcome Y in the control group.
- **prognostic_scale** (<code>[float](#float)</code>) – Scale of the prognostic signal derived from confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a `CausalData` object instead of a `pandas.DataFrame`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.data.causaldata.CausalData)</code> – Synthetic RCT dataset.

#### `causalis.data.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

#### `causalis.data.obs_linear_26_dataset`

```python
obs_linear_26_dataset(n=10000, seed=42, include_oracle=True, return_causal_data=True)
```

A pre-configured observational linear dataset with 5 standard confounders.
Based on the scenario in docs/cases/dml_ate.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

#### `causalis.data.obs_linear_effect`

```python
obs_linear_effect(n=10000, theta=1.0, outcome_type='continuous', sigma_y=1.0, target_d_rate=None, confounder_specs=None, beta_y=None, beta_d=None, random_state=42, k=0, x_sampler=None, include_oracle=True, add_ancillary=False, deterministic_ids=False)
```

Generate an observational dataset with linear effects of confounders and a constant treatment effect.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **theta** (<code>[float](#float)</code>) – Constant treatment effect.
- **outcome_type** (<code>('continuous', 'binary', 'poisson')</code>) – Family of the outcome distribution.
- **sigma_y** (<code>[float](#float)</code>) – Noise level for continuous outcomes.
- **target_d_rate** (<code>[float](#float)</code>) – Target treatment prevalence (propensity mean).
- **confounder_specs** (<code>list of dict</code>) – Schema for confounder distributions.
- **beta_y** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the outcome model.
- **beta_d** (<code>[array](#array) - [like](#like)</code>) – Linear coefficients for confounders in the treatment model.
- **random_state** (<code>[int](#int)</code>) – Random seed for reproducibility.
- **k** (<code>[int](#int)</code>) – Number of confounders if specs not provided.
- **x_sampler** (<code>[callable](#callable)</code>) – Custom sampler for confounders.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'm', etc.
- **add_ancillary** (<code>[bool](#bool)</code>) – If True, adds standard ancillary columns (age, platform, etc.).
- **deterministic_ids** (<code>[bool](#bool)</code>) – If True, generates deterministic user IDs.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic observational dataset.

### `causalis.eda`

**Modules:**

- [**cb_module**](#causalis.eda.cb_module) –
- [**confounders_balance**](#causalis.eda.confounders_balance) –
- [**eda**](#causalis.eda.eda) – EDA utilities for causal analysis (propensity, overlap, balance, weights).
- [**rct_design**](#causalis.eda.rct_design) – Design module for experimental rct_design utilities.

**Classes:**

- [**CausalDataLite**](#causalis.eda.CausalDataLite) – A minimal container for dataset roles used by CausalEDA.
- [**CausalEDA**](#causalis.eda.CausalEDA) – Exploratory diagnostics for causal designs with binary treatment.

#### `causalis.eda.CausalDataLite`

```python
CausalDataLite(df, treatment, outcome, confounders)
```

A minimal container for dataset roles used by CausalEDA.

Attributes

- df: The full pandas DataFrame containing treatment, outcome and covariates.
- treatment: Column name of the binary treatment indicator (0/1).
- outcome: Column name of the outcome variable.
- confounders: List of covariate column names used to model treatment.

**Attributes:**

- [**confounders**](#causalis.eda.CausalDataLite.confounders) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**df**](#causalis.eda.CausalDataLite.df) (<code>[DataFrame](#pandas.DataFrame)</code>) –
- [**outcome**](#causalis.eda.CausalDataLite.outcome) (<code>[str](#str)</code>) –
- [**treatment**](#causalis.eda.CausalDataLite.treatment) (<code>[str](#str)</code>) –

##### `causalis.eda.CausalDataLite.confounders`

```python
confounders: List[str]
```

##### `causalis.eda.CausalDataLite.df`

```python
df: pd.DataFrame
```

##### `causalis.eda.CausalDataLite.outcome`

```python
outcome: str
```

##### `causalis.eda.CausalDataLite.treatment`

```python
treatment: str
```

#### `causalis.eda.CausalEDA`

```python
CausalEDA(data, ps_model=None, n_splits=5, random_state=42)
```

Exploratory diagnostics for causal designs with binary treatment.

The class exposes methods to:

- Summarize outcome by treatment and naive mean difference.
- Estimate cross-validated propensity scores and assess treatment
  predictability (AUC) and positivity/overlap.
- Inspect covariate balance via standardized mean differences (SMD)
  before/after IPTW weighting; visualize with a love plot.
- Inspect weight distributions and effective sample size (ESS).

**Functions:**

- [**auc_m**](#causalis.eda.CausalEDA.auc_m) – Compute ROC AUC of treatment assignment vs. m(x).
- [**confounders_means**](#causalis.eda.CausalEDA.confounders_means) – Comprehensive confounders balance assessment with means by treatment group.
- [**confounders_roc_auc**](#causalis.eda.CausalEDA.confounders_roc_auc) –
- [**data_shape**](#causalis.eda.CausalEDA.data_shape) – Return the shape information of the causal dataset.
- [**fit_m**](#causalis.eda.CausalEDA.fit_m) – Estimate cross-validated m(x) = P(D=1|X).
- [**fit_propensity**](#causalis.eda.CausalEDA.fit_propensity) –
- [**m_features**](#causalis.eda.CausalEDA.m_features) – Return feature attribution from the fitted m(x) model.
- [**outcome_boxplot**](#causalis.eda.CausalEDA.outcome_boxplot) – Prettified boxplot of the outcome by treatment.
- [**outcome_fit**](#causalis.eda.CausalEDA.outcome_fit) – Fit a regression model to predict outcome from confounders only.
- [**outcome_hist**](#causalis.eda.CausalEDA.outcome_hist) – Plot the distribution of the outcome for each treatment on a single, pretty plot.
- [**outcome_plots**](#causalis.eda.CausalEDA.outcome_plots) – Plot the distribution of the outcome for every treatment on one plot,
- [**plot_m_overlap**](#causalis.eda.CausalEDA.plot_m_overlap) – Plot overlaid histograms of m(x) for treated vs control.
- [**plot_ps_overlap**](#causalis.eda.CausalEDA.plot_ps_overlap) –
- [**positivity_check**](#causalis.eda.CausalEDA.positivity_check) –
- [**positivity_check_m**](#causalis.eda.CausalEDA.positivity_check_m) – Check overlap/positivity for m(x) based on thresholds.
- [**treatment_features**](#causalis.eda.CausalEDA.treatment_features) –

**Attributes:**

- [**cat_features**](#causalis.eda.CausalEDA.cat_features) –
- [**d**](#causalis.eda.CausalEDA.d) –
- [**n_splits**](#causalis.eda.CausalEDA.n_splits) –
- [**preproc**](#causalis.eda.CausalEDA.preproc) –
- [**ps_model**](#causalis.eda.CausalEDA.ps_model) –
- [**ps_pipe**](#causalis.eda.CausalEDA.ps_pipe) –
- [**random_state**](#causalis.eda.CausalEDA.random_state) –

##### `causalis.eda.CausalEDA.auc_m`

```python
auc_m(m=None)
```

Compute ROC AUC of treatment assignment vs. m(x).

##### `causalis.eda.CausalEDA.cat_features`

```python
cat_features = None
```

##### `causalis.eda.CausalEDA.confounders_means`

```python
confounders_means()
```

Comprehensive confounders balance assessment with means by treatment group.

Returns a DataFrame with detailed balance information including:

- Mean values of each confounder for control group (treatment=0)
- Mean values of each confounder for treated group (treatment=1)
- Absolute difference between treatment groups
- Standardized Mean Difference (SMD) for formal balance assessment
- Kolmogorov–Smirnov test p-value (ks_pvalue) for distributional differences

This method provides a comprehensive view of confounder balance by showing
the actual mean values alongside the standardized differences, making it easier
to understand both the magnitude and direction of imbalances.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with confounders as index and the following columns:
- mean_d_0: mean value for control group (treatment=0)
- mean_d_1: mean value for treated group (treatment=1)
- abs_diff: absolute difference abs(mean_d_1 - mean_d_0)
- smd: standardized mean difference (Cohen's d)
- ks_pvalue: p-value of the KS test

<details class="note" open markdown="1">
<summary>Notes</summary>

SMD values > 0.1 in absolute value typically indicate meaningful imbalance.
Categorical variables are automatically converted to dummy variables.

</details>

**Examples:**

```pycon
>>> eda = CausalEDA(causal_data)
>>> balance = eda.confounders_means()
>>> print(balance.head())
             mean_d_0  mean_d_1  abs_diff       smd
confounders                                       
age              29.5      31.2      1.7     0.085
income        45000.0   47500.0   2500.0     0.125
education         0.25      0.35      0.1     0.215
```

##### `causalis.eda.CausalEDA.confounders_roc_auc`

```python
confounders_roc_auc(ps=None)
```

##### `causalis.eda.CausalEDA.d`

```python
d = CausalDataLite(df=(roles['df']), treatment=(roles['treatment']), outcome=(roles['outcome']), confounders=(roles['confounders']))
```

##### `causalis.eda.CausalEDA.data_shape`

```python
data_shape()
```

Return the shape information of the causal dataset.

Returns a dict with:

- n_rows: number of rows (observations) in the dataset
- n_columns: number of columns (features) in the dataset

This provides a quick overview of the dataset dimensions for
exploratory analysis and reporting purposes.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [int](#int)\]</code> – Dictionary containing 'n_rows' and 'n_columns' keys with
  corresponding integer values representing the dataset dimensions.

**Examples:**

```pycon
>>> eda = CausalEDA(causal_data)
>>> shape_info = eda.data_shape()
>>> print(f"Dataset has {shape_info['n_rows']} rows and {shape_info['n_columns']} columns")
```

##### `causalis.eda.CausalEDA.fit_m`

```python
fit_m()
```

Estimate cross-validated m(x) = P(D=1|X).

Uses a preprocessing + classifier setup with stratified K-fold to generate
out-of-fold probabilities. For CatBoost, data are one-hot encoded via the
configured ColumnTransformer before fitting. Returns a PropensityModel.

##### `causalis.eda.CausalEDA.fit_propensity`

```python
fit_propensity()
```

##### `causalis.eda.CausalEDA.m_features`

```python
m_features()
```

Return feature attribution from the fitted m(x) model.

- CatBoost path: SHAP attributions with columns 'shap_mean' and 'shap_mean_abs',
  sorted by 'shap_mean_abs'. Uses transformed feature names from the preprocessor.
- Sklearn path (LogisticRegression): absolute coefficients reported as 'coef_abs'.

##### `causalis.eda.CausalEDA.n_splits`

```python
n_splits = n_splits
```

##### `causalis.eda.CausalEDA.outcome_boxplot`

```python
outcome_boxplot(treatment=None, outcome=None, figsize=(9, 5.5), dpi=220, font_scale=1.15, showfliers=True, patch_artist=True, save=None, save_dpi=None, transparent=False)
```

Prettified boxplot of the outcome by treatment.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI figure, scalable fonts
- Soft modern color styling (default Matplotlib palette)
- Optional outliers, gentle transparency
- Optional save to PNG/SVG/PDF

</details>

##### `causalis.eda.CausalEDA.outcome_fit`

```python
outcome_fit(outcome_model=None)
```

Fit a regression model to predict outcome from confounders only.

Uses a preprocessing+CatBoost regressor pipeline with K-fold
cross_val_predict to generate out-of-fold predictions. CatBoost uses
all available threads and handles categorical features natively. Returns an
OutcomeModel instance containing predicted outcomes and diagnostic methods.

The outcome model predicts the baseline outcome from confounders only,
excluding treatment. This is essential for proper causal analysis.

**Parameters:**

- **outcome_model** (<code>[Optional](#typing.Optional)\[[Any](#typing.Any)\]</code>) – Custom regression model to use. If None, uses CatBoostRegressor.

**Returns:**

- <code>[OutcomeModel](#causalis.eda.eda.OutcomeModel)</code> – An OutcomeModel instance with methods for:
- scores: RMSE and MAE regression metrics
- shap: SHAP values DataFrame property for outcome prediction

##### `causalis.eda.CausalEDA.outcome_hist`

```python
outcome_hist(treatment=None, outcome=None, bins='fd', density=True, alpha=0.45, sharex=True, kde=True, clip=(0.01, 0.99), figsize=(9, 5.5), dpi=220, font_scale=1.15, save=None, save_dpi=None, transparent=False)
```

Plot the distribution of the outcome for each treatment on a single, pretty plot.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI canvas + scalable fonts
- Default Matplotlib colors; KDE & mean lines match their histogram colors
- Numeric outcomes: shared x-range (optional), optional KDE, quantile clipping
- Categorical outcomes: normalized grouped bars by treatment
- Optional hi-res export (PNG/SVG/PDF)

</details>

##### `causalis.eda.CausalEDA.outcome_plots`

```python
outcome_plots(treatment=None, outcome=None, bins=30, density=True, alpha=0.5, figsize=(7, 4), sharex=True)
```

Plot the distribution of the outcome for every treatment on one plot,
and also produce a boxplot by treatment to visualize outliers.

**Parameters:**

- **treatment** (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) – Treatment column name. Defaults to the treatment stored in the CausalEDA data.
- **outcome** (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) – Outcome column name. Defaults to the outcome stored in the CausalEDA data.
- **bins** (<code>[int](#int)</code>) – Number of bins for histograms when the outcome is numeric.
- **density** (<code>[bool](#bool)</code>) – Whether to normalize histograms to form a density.
- **alpha** (<code>[float](#float)</code>) – Transparency for overlaid histograms.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size for the plots.
- **sharex** (<code>[bool](#bool)</code>) – If True and the outcome is numeric, use the same x-limits across treatments.

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[Figure](#matplotlib.figure.Figure), [Figure](#matplotlib.figure.Figure)\]</code> – (fig_distribution, fig_boxplot)

##### `causalis.eda.CausalEDA.plot_m_overlap`

```python
plot_m_overlap(m=None)
```

Plot overlaid histograms of m(x) for treated vs control.

##### `causalis.eda.CausalEDA.plot_ps_overlap`

```python
plot_ps_overlap(ps=None)
```

##### `causalis.eda.CausalEDA.positivity_check`

```python
positivity_check(ps=None, bounds=(0.05, 0.95))
```

##### `causalis.eda.CausalEDA.positivity_check_m`

```python
positivity_check_m(m=None, bounds=(0.05, 0.95))
```

Check overlap/positivity for m(x) based on thresholds.

##### `causalis.eda.CausalEDA.preproc`

```python
preproc = ColumnTransformer(transformers=[('num', num_transformer, num), ('cat', OneHotEncoder(handle_unknown='ignore', drop=None, sparse_output=False), cat)], remainder='drop')
```

##### `causalis.eda.CausalEDA.ps_model`

```python
ps_model = ps_model or CatBoostClassifier(thread_count=(-1), random_seed=random_state, verbose=False, allow_writing_files=False)
```

##### `causalis.eda.CausalEDA.ps_pipe`

```python
ps_pipe = Pipeline([('prep', self.preproc), ('clf', self.ps_model)])
```

##### `causalis.eda.CausalEDA.random_state`

```python
random_state = random_state
```

##### `causalis.eda.CausalEDA.treatment_features`

```python
treatment_features()
```

#### `causalis.eda.cb_module`

**Functions:**

- [**confounders_balance**](#causalis.eda.cb_module.confounders_balance) – Compute balance diagnostics for confounders between treatment groups.

##### `causalis.eda.cb_module.confounders_balance`

```python
confounders_balance(data)
```

Compute balance diagnostics for confounders between treatment groups.

Produces a DataFrame indexed by expanded confounder columns (after one-hot
encoding categorical variables if present) with:

- mean_d_0: mean value for control group (t=0)
- mean_d_1: mean value for treated group (t=1)
- abs_diff: abs(mean_d_1 - mean_d_0)
- smd: standardized mean difference (Cohen's d using pooled std)
- ks_pvalue: p-value for the KS test (rounded to 5 decimal places, non-scientific)

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causal dataset containing the dataframe, treatment, and confounders.
  Accepts CausalData or any object with `df`, `treatment`, and `confounders`
  attributes/properties.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Balance table sorted by |smd| (descending), index named 'confounders'.

#### `causalis.eda.confounders_balance`

**Functions:**

- [**confounders_balance**](#causalis.eda.confounders_balance.confounders_balance) – Compute balance diagnostics for confounders between treatment groups.

##### `causalis.eda.confounders_balance.confounders_balance`

```python
confounders_balance(data)
```

Compute balance diagnostics for confounders between treatment groups.

Produces a DataFrame indexed by expanded confounder columns (after one-hot
encoding categorical variables if present) with:

- mean_d_0: mean value for control group (t=0)
- mean_d_1: mean value for treated group (t=1)
- abs_diff: abs(mean_d_1 - mean_d_0)
- smd: standardized mean difference (Cohen's d using pooled std)
- ks_pvalue: p-value for the KS test (rounded to 5 decimal places, non-scientific)

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causal dataset containing the dataframe, treatment, and confounders.
  Accepts CausalData or any object with `df`, `treatment`, and `confounders`
  attributes/properties.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Balance table sorted by |smd| (descending), index named 'confounders'.

#### `causalis.eda.eda`

EDA utilities for causal analysis (propensity, overlap, balance, weights).

This module provides a lightweight CausalEDA class to quickly assess whether a
binary treatment problem is suitable for causal effect estimation. The outputs
focus on interpretability: treatment predictability, overlap/positivity,
covariate balance before/after weighting, and basic data health.

What the main outputs mean

- fit_propensity(): Numpy array of cross-validated propensity scores P(T=1|X).
- confounders_roc_auc(): Float ROC AUC of treatment vs. propensity score.
  Higher AUC implies treatment is predictable from confounders (more cofounding risk).
- positivity_check(): Dict with bounds, share_below, share_above, and flag.
  It reports what share of units have PS outside [low, high]; a large share
  signals poor overlap (violated positivity).
- plot_ps_overlap(): Overlaid histograms of PS for treated vs control.
- confounders_means(): DataFrame with comprehensive balance assessment including
  means by treatment group, absolute differences, and standardized mean differences (SMD).

Note: The class accepts either the project’s CausalData object (duck-typed) or a
CausalDataLite with explicit fields.

**Classes:**

- [**CausalDataLite**](#causalis.eda.eda.CausalDataLite) – A minimal container for dataset roles used by CausalEDA.
- [**CausalEDA**](#causalis.eda.eda.CausalEDA) – Exploratory diagnostics for causal designs with binary treatment.
- [**OutcomeModel**](#causalis.eda.eda.OutcomeModel) – A model for outcome prediction and related diagnostics.
- [**PropensityModel**](#causalis.eda.eda.PropensityModel) – A model for m(x) = P(D=1|X) and related diagnostics.

##### `causalis.eda.eda.CausalDataLite`

```python
CausalDataLite(df, treatment, outcome, confounders)
```

A minimal container for dataset roles used by CausalEDA.

Attributes

- df: The full pandas DataFrame containing treatment, outcome and covariates.
- treatment: Column name of the binary treatment indicator (0/1).
- outcome: Column name of the outcome variable.
- confounders: List of covariate column names used to model treatment.

**Attributes:**

- [**confounders**](#causalis.eda.eda.CausalDataLite.confounders) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**df**](#causalis.eda.eda.CausalDataLite.df) (<code>[DataFrame](#pandas.DataFrame)</code>) –
- [**outcome**](#causalis.eda.eda.CausalDataLite.outcome) (<code>[str](#str)</code>) –
- [**treatment**](#causalis.eda.eda.CausalDataLite.treatment) (<code>[str](#str)</code>) –

###### `causalis.eda.eda.CausalDataLite.confounders`

```python
confounders: List[str]
```

###### `causalis.eda.eda.CausalDataLite.df`

```python
df: pd.DataFrame
```

###### `causalis.eda.eda.CausalDataLite.outcome`

```python
outcome: str
```

###### `causalis.eda.eda.CausalDataLite.treatment`

```python
treatment: str
```

##### `causalis.eda.eda.CausalEDA`

```python
CausalEDA(data, ps_model=None, n_splits=5, random_state=42)
```

Exploratory diagnostics for causal designs with binary treatment.

The class exposes methods to:

- Summarize outcome by treatment and naive mean difference.
- Estimate cross-validated propensity scores and assess treatment
  predictability (AUC) and positivity/overlap.
- Inspect covariate balance via standardized mean differences (SMD)
  before/after IPTW weighting; visualize with a love plot.
- Inspect weight distributions and effective sample size (ESS).

**Functions:**

- [**auc_m**](#causalis.eda.eda.CausalEDA.auc_m) – Compute ROC AUC of treatment assignment vs. m(x).
- [**confounders_means**](#causalis.eda.eda.CausalEDA.confounders_means) – Comprehensive confounders balance assessment with means by treatment group.
- [**confounders_roc_auc**](#causalis.eda.eda.CausalEDA.confounders_roc_auc) –
- [**data_shape**](#causalis.eda.eda.CausalEDA.data_shape) – Return the shape information of the causal dataset.
- [**fit_m**](#causalis.eda.eda.CausalEDA.fit_m) – Estimate cross-validated m(x) = P(D=1|X).
- [**fit_propensity**](#causalis.eda.eda.CausalEDA.fit_propensity) –
- [**m_features**](#causalis.eda.eda.CausalEDA.m_features) – Return feature attribution from the fitted m(x) model.
- [**outcome_boxplot**](#causalis.eda.eda.CausalEDA.outcome_boxplot) – Prettified boxplot of the outcome by treatment.
- [**outcome_fit**](#causalis.eda.eda.CausalEDA.outcome_fit) – Fit a regression model to predict outcome from confounders only.
- [**outcome_hist**](#causalis.eda.eda.CausalEDA.outcome_hist) – Plot the distribution of the outcome for each treatment on a single, pretty plot.
- [**outcome_plots**](#causalis.eda.eda.CausalEDA.outcome_plots) – Plot the distribution of the outcome for every treatment on one plot,
- [**plot_m_overlap**](#causalis.eda.eda.CausalEDA.plot_m_overlap) – Plot overlaid histograms of m(x) for treated vs control.
- [**plot_ps_overlap**](#causalis.eda.eda.CausalEDA.plot_ps_overlap) –
- [**positivity_check**](#causalis.eda.eda.CausalEDA.positivity_check) –
- [**positivity_check_m**](#causalis.eda.eda.CausalEDA.positivity_check_m) – Check overlap/positivity for m(x) based on thresholds.
- [**treatment_features**](#causalis.eda.eda.CausalEDA.treatment_features) –

**Attributes:**

- [**cat_features**](#causalis.eda.eda.CausalEDA.cat_features) –
- [**d**](#causalis.eda.eda.CausalEDA.d) –
- [**n_splits**](#causalis.eda.eda.CausalEDA.n_splits) –
- [**preproc**](#causalis.eda.eda.CausalEDA.preproc) –
- [**ps_model**](#causalis.eda.eda.CausalEDA.ps_model) –
- [**ps_pipe**](#causalis.eda.eda.CausalEDA.ps_pipe) –
- [**random_state**](#causalis.eda.eda.CausalEDA.random_state) –

###### `causalis.eda.eda.CausalEDA.auc_m`

```python
auc_m(m=None)
```

Compute ROC AUC of treatment assignment vs. m(x).

###### `causalis.eda.eda.CausalEDA.cat_features`

```python
cat_features = None
```

###### `causalis.eda.eda.CausalEDA.confounders_means`

```python
confounders_means()
```

Comprehensive confounders balance assessment with means by treatment group.

Returns a DataFrame with detailed balance information including:

- Mean values of each confounder for control group (treatment=0)
- Mean values of each confounder for treated group (treatment=1)
- Absolute difference between treatment groups
- Standardized Mean Difference (SMD) for formal balance assessment
- Kolmogorov–Smirnov test p-value (ks_pvalue) for distributional differences

This method provides a comprehensive view of confounder balance by showing
the actual mean values alongside the standardized differences, making it easier
to understand both the magnitude and direction of imbalances.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with confounders as index and the following columns:
- mean_d_0: mean value for control group (treatment=0)
- mean_d_1: mean value for treated group (treatment=1)
- abs_diff: absolute difference abs(mean_d_1 - mean_d_0)
- smd: standardized mean difference (Cohen's d)
- ks_pvalue: p-value of the KS test

<details class="note" open markdown="1">
<summary>Notes</summary>

SMD values > 0.1 in absolute value typically indicate meaningful imbalance.
Categorical variables are automatically converted to dummy variables.

</details>

**Examples:**

```pycon
>>> eda = CausalEDA(causal_data)
>>> balance = eda.confounders_means()
>>> print(balance.head())
             mean_d_0  mean_d_1  abs_diff       smd
confounders                                       
age              29.5      31.2      1.7     0.085
income        45000.0   47500.0   2500.0     0.125
education         0.25      0.35      0.1     0.215
```

###### `causalis.eda.eda.CausalEDA.confounders_roc_auc`

```python
confounders_roc_auc(ps=None)
```

###### `causalis.eda.eda.CausalEDA.d`

```python
d = CausalDataLite(df=(roles['df']), treatment=(roles['treatment']), outcome=(roles['outcome']), confounders=(roles['confounders']))
```

###### `causalis.eda.eda.CausalEDA.data_shape`

```python
data_shape()
```

Return the shape information of the causal dataset.

Returns a dict with:

- n_rows: number of rows (observations) in the dataset
- n_columns: number of columns (features) in the dataset

This provides a quick overview of the dataset dimensions for
exploratory analysis and reporting purposes.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [int](#int)\]</code> – Dictionary containing 'n_rows' and 'n_columns' keys with
  corresponding integer values representing the dataset dimensions.

**Examples:**

```pycon
>>> eda = CausalEDA(causal_data)
>>> shape_info = eda.data_shape()
>>> print(f"Dataset has {shape_info['n_rows']} rows and {shape_info['n_columns']} columns")
```

###### `causalis.eda.eda.CausalEDA.fit_m`

```python
fit_m()
```

Estimate cross-validated m(x) = P(D=1|X).

Uses a preprocessing + classifier setup with stratified K-fold to generate
out-of-fold probabilities. For CatBoost, data are one-hot encoded via the
configured ColumnTransformer before fitting. Returns a PropensityModel.

###### `causalis.eda.eda.CausalEDA.fit_propensity`

```python
fit_propensity()
```

###### `causalis.eda.eda.CausalEDA.m_features`

```python
m_features()
```

Return feature attribution from the fitted m(x) model.

- CatBoost path: SHAP attributions with columns 'shap_mean' and 'shap_mean_abs',
  sorted by 'shap_mean_abs'. Uses transformed feature names from the preprocessor.
- Sklearn path (LogisticRegression): absolute coefficients reported as 'coef_abs'.

###### `causalis.eda.eda.CausalEDA.n_splits`

```python
n_splits = n_splits
```

###### `causalis.eda.eda.CausalEDA.outcome_boxplot`

```python
outcome_boxplot(treatment=None, outcome=None, figsize=(9, 5.5), dpi=220, font_scale=1.15, showfliers=True, patch_artist=True, save=None, save_dpi=None, transparent=False)
```

Prettified boxplot of the outcome by treatment.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI figure, scalable fonts
- Soft modern color styling (default Matplotlib palette)
- Optional outliers, gentle transparency
- Optional save to PNG/SVG/PDF

</details>

###### `causalis.eda.eda.CausalEDA.outcome_fit`

```python
outcome_fit(outcome_model=None)
```

Fit a regression model to predict outcome from confounders only.

Uses a preprocessing+CatBoost regressor pipeline with K-fold
cross_val_predict to generate out-of-fold predictions. CatBoost uses
all available threads and handles categorical features natively. Returns an
OutcomeModel instance containing predicted outcomes and diagnostic methods.

The outcome model predicts the baseline outcome from confounders only,
excluding treatment. This is essential for proper causal analysis.

**Parameters:**

- **outcome_model** (<code>[Optional](#typing.Optional)\[[Any](#typing.Any)\]</code>) – Custom regression model to use. If None, uses CatBoostRegressor.

**Returns:**

- <code>[OutcomeModel](#causalis.eda.eda.OutcomeModel)</code> – An OutcomeModel instance with methods for:
- scores: RMSE and MAE regression metrics
- shap: SHAP values DataFrame property for outcome prediction

###### `causalis.eda.eda.CausalEDA.outcome_hist`

```python
outcome_hist(treatment=None, outcome=None, bins='fd', density=True, alpha=0.45, sharex=True, kde=True, clip=(0.01, 0.99), figsize=(9, 5.5), dpi=220, font_scale=1.15, save=None, save_dpi=None, transparent=False)
```

Plot the distribution of the outcome for each treatment on a single, pretty plot.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI canvas + scalable fonts
- Default Matplotlib colors; KDE & mean lines match their histogram colors
- Numeric outcomes: shared x-range (optional), optional KDE, quantile clipping
- Categorical outcomes: normalized grouped bars by treatment
- Optional hi-res export (PNG/SVG/PDF)

</details>

###### `causalis.eda.eda.CausalEDA.outcome_plots`

```python
outcome_plots(treatment=None, outcome=None, bins=30, density=True, alpha=0.5, figsize=(7, 4), sharex=True)
```

Plot the distribution of the outcome for every treatment on one plot,
and also produce a boxplot by treatment to visualize outliers.

**Parameters:**

- **treatment** (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) – Treatment column name. Defaults to the treatment stored in the CausalEDA data.
- **outcome** (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) – Outcome column name. Defaults to the outcome stored in the CausalEDA data.
- **bins** (<code>[int](#int)</code>) – Number of bins for histograms when the outcome is numeric.
- **density** (<code>[bool](#bool)</code>) – Whether to normalize histograms to form a density.
- **alpha** (<code>[float](#float)</code>) – Transparency for overlaid histograms.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size for the plots.
- **sharex** (<code>[bool](#bool)</code>) – If True and the outcome is numeric, use the same x-limits across treatments.

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[Figure](#matplotlib.figure.Figure), [Figure](#matplotlib.figure.Figure)\]</code> – (fig_distribution, fig_boxplot)

###### `causalis.eda.eda.CausalEDA.plot_m_overlap`

```python
plot_m_overlap(m=None)
```

Plot overlaid histograms of m(x) for treated vs control.

###### `causalis.eda.eda.CausalEDA.plot_ps_overlap`

```python
plot_ps_overlap(ps=None)
```

###### `causalis.eda.eda.CausalEDA.positivity_check`

```python
positivity_check(ps=None, bounds=(0.05, 0.95))
```

###### `causalis.eda.eda.CausalEDA.positivity_check_m`

```python
positivity_check_m(m=None, bounds=(0.05, 0.95))
```

Check overlap/positivity for m(x) based on thresholds.

###### `causalis.eda.eda.CausalEDA.preproc`

```python
preproc = ColumnTransformer(transformers=[('num', num_transformer, num), ('cat', OneHotEncoder(handle_unknown='ignore', drop=None, sparse_output=False), cat)], remainder='drop')
```

###### `causalis.eda.eda.CausalEDA.ps_model`

```python
ps_model = ps_model or CatBoostClassifier(thread_count=(-1), random_seed=random_state, verbose=False, allow_writing_files=False)
```

###### `causalis.eda.eda.CausalEDA.ps_pipe`

```python
ps_pipe = Pipeline([('prep', self.preproc), ('clf', self.ps_model)])
```

###### `causalis.eda.eda.CausalEDA.random_state`

```python
random_state = random_state
```

###### `causalis.eda.eda.CausalEDA.treatment_features`

```python
treatment_features()
```

##### `causalis.eda.eda.OutcomeModel`

```python
OutcomeModel(predicted_outcomes, actual_outcomes, fitted_model, feature_names, X_for_shap=None, cat_features_for_shap=None)
```

A model for outcome prediction and related diagnostics.

This class encapsulates outcome predictions and provides methods for:

- Computing RMSE and MAE regression metrics
- Extracting SHAP values for outcome prediction

The class is returned by CausalEDA.outcome_fit() and provides a cleaner
interface for outcome model analysis.

**Functions:**

- [**from_kfold**](#causalis.eda.eda.OutcomeModel.from_kfold) – Estimate an outcome model via K-fold and return an OutcomeModel.

**Attributes:**

- [**X_for_shap**](#causalis.eda.eda.OutcomeModel.X_for_shap) –
- [**actual_outcomes**](#causalis.eda.eda.OutcomeModel.actual_outcomes) –
- [**cat_features_for_shap**](#causalis.eda.eda.OutcomeModel.cat_features_for_shap) –
- [**feature_names**](#causalis.eda.eda.OutcomeModel.feature_names) –
- [**fitted_model**](#causalis.eda.eda.OutcomeModel.fitted_model) –
- [**predicted_outcomes**](#causalis.eda.eda.OutcomeModel.predicted_outcomes) –
- [**scores**](#causalis.eda.eda.OutcomeModel.scores) (<code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code>) – Compute regression metrics (RMSE and MAE) for outcome predictions.
- [**shap**](#causalis.eda.eda.OutcomeModel.shap) (<code>[DataFrame](#pandas.DataFrame)</code>) – Return SHAP values from the fitted outcome prediction model.

**Parameters:**

- **predicted_outcomes** (<code>[ndarray](#numpy.ndarray)</code>) – Array of predicted outcome values
- **actual_outcomes** (<code>[ndarray](#numpy.ndarray)</code>) – Array of actual outcome values
- **fitted_model** (<code>[Any](#typing.Any)</code>) – The fitted outcome prediction model
- **feature_names** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of features used in the model (confounders only)
- **X_for_shap** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) – Preprocessed feature matrix for SHAP computation
- **cat_features_for_shap** (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[int](#int)\]\]</code>) – Categorical feature indices for SHAP computation

###### `causalis.eda.eda.OutcomeModel.X_for_shap`

```python
X_for_shap = X_for_shap
```

###### `causalis.eda.eda.OutcomeModel.actual_outcomes`

```python
actual_outcomes = actual_outcomes
```

###### `causalis.eda.eda.OutcomeModel.cat_features_for_shap`

```python
cat_features_for_shap = cat_features_for_shap
```

###### `causalis.eda.eda.OutcomeModel.feature_names`

```python
feature_names = feature_names
```

###### `causalis.eda.eda.OutcomeModel.fitted_model`

```python
fitted_model = fitted_model
```

###### `causalis.eda.eda.OutcomeModel.from_kfold`

```python
from_kfold(X, y, model=None, n_splits=5, random_state=42, preprocessor=None)
```

Estimate an outcome model via K-fold and return an OutcomeModel.

Produces out-of-fold predictions using KFold. If no model is provided,
a fast LinearRegression is used. Categorical features are one-hot
encoded and numeric features standardized by default.

###### `causalis.eda.eda.OutcomeModel.predicted_outcomes`

```python
predicted_outcomes = predicted_outcomes
```

###### `causalis.eda.eda.OutcomeModel.scores`

```python
scores: Dict[str, float]
```

Compute regression metrics (RMSE and MAE) for outcome predictions.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code> – Dictionary containing:
- 'rmse': Root Mean Squared Error
- 'mae': Mean Absolute Error

###### `causalis.eda.eda.OutcomeModel.shap`

```python
shap: pd.DataFrame
```

Return SHAP values from the fitted outcome prediction model.

SHAP values show the directional contribution of each feature to
outcome prediction, where positive values increase the predicted
outcome and negative values decrease it.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – For CatBoost models: DataFrame with columns 'feature' and 'shap_mean',
  where 'shap_mean' represents the mean SHAP value across all samples.

For sklearn models: DataFrame with columns 'feature' and 'importance'
(absolute coefficient values, for backward compatibility).

**Raises:**

- <code>[RuntimeError](#RuntimeError)</code> – If the fitted model does not support SHAP values extraction.

##### `causalis.eda.eda.PropensityModel`

```python
PropensityModel(m=None, d=None, fitted_model=None, feature_names=None, X_for_shap=None, cat_features_for_shap=None, propensity_scores=None, treatment_values=None)
```

A model for m(x) = P(D=1|X) and related diagnostics.

This class encapsulates propensity m and provides methods for:

- Computing ROC AUC (AUC of D vs m)
- Extracting SHAP values
- Plotting m overlap
- Checking positivity/overlap

The class is returned by CausalEDA.fit_m() and provides a cleaner
interface for propensity analysis.

**Functions:**

- [**from_kfold**](#causalis.eda.eda.PropensityModel.from_kfold) – Estimate m(x) via K-fold and return a PropensityModel.
- [**plot_m_overlap**](#causalis.eda.eda.PropensityModel.plot_m_overlap) – Overlap plot for m(x)=P(D=1|X) with high-res rendering.
- [**positivity_check**](#causalis.eda.eda.PropensityModel.positivity_check) –
- [**positivity_check_m**](#causalis.eda.eda.PropensityModel.positivity_check_m) – Check overlap/positivity for m(x) based on thresholds.
- [**ps_graph**](#causalis.eda.eda.PropensityModel.ps_graph) –

**Attributes:**

- [**X_for_shap**](#causalis.eda.eda.PropensityModel.X_for_shap) –
- [**cat_features_for_shap**](#causalis.eda.eda.PropensityModel.cat_features_for_shap) –
- [**d**](#causalis.eda.eda.PropensityModel.d) –
- [**feature_names**](#causalis.eda.eda.PropensityModel.feature_names) –
- [**fitted_model**](#causalis.eda.eda.PropensityModel.fitted_model) –
- [**m**](#causalis.eda.eda.PropensityModel.m) –
- [**propensity_scores**](#causalis.eda.eda.PropensityModel.propensity_scores) –
- [**roc_auc**](#causalis.eda.eda.PropensityModel.roc_auc) (<code>[float](#float)</code>) – Compute ROC AUC of treatment assignment vs. m(x).
- [**shap**](#causalis.eda.eda.PropensityModel.shap) (<code>[DataFrame](#pandas.DataFrame)</code>) – Return feature attribution from the fitted propensity model.

**Parameters:**

- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Array of m(x) = P(D=1|X)
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Array of actual treatment assignments (0/1)
- **fitted_model** (<code>[Any](#typing.Any)</code>) – The fitted propensity model
- **feature_names** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of features used in the model
- **X_for_shap** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) – Preprocessed feature matrix for SHAP computation
- **cat_features_for_shap** (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[int](#int)\]\]</code>) – Categorical feature indices for SHAP computation
- **propensity_scores** (<code>Optional legacy aliases accepted for back-compat</code>) –
- **treatment_values** (<code>Optional legacy aliases accepted for back-compat</code>) –

###### `causalis.eda.eda.PropensityModel.X_for_shap`

```python
X_for_shap = X_for_shap
```

###### `causalis.eda.eda.PropensityModel.cat_features_for_shap`

```python
cat_features_for_shap = cat_features_for_shap
```

###### `causalis.eda.eda.PropensityModel.d`

```python
d = np.asarray(d) if d is not None else None
```

###### `causalis.eda.eda.PropensityModel.feature_names`

```python
feature_names = feature_names
```

###### `causalis.eda.eda.PropensityModel.fitted_model`

```python
fitted_model = fitted_model
```

###### `causalis.eda.eda.PropensityModel.from_kfold`

```python
from_kfold(X, t, model=None, n_splits=5, random_state=42, preprocessor=None)
```

Estimate m(x) via K-fold and return a PropensityModel.

Produces out-of-fold m estimates using StratifiedKFold. If no
model is provided, a fast LogisticRegression is used. Categorical
features are one-hot encoded and numeric features standardized by
default using a ColumnTransformer.

###### `causalis.eda.eda.PropensityModel.m`

```python
m = np.asarray(m) if m is not None else None
```

###### `causalis.eda.eda.PropensityModel.plot_m_overlap`

```python
plot_m_overlap(clip=(0.01, 0.99), bins='fd', kde=True, shade_overlap=True, ax=None, figsize=(9, 5.5), dpi=220, font_scale=1.15, save=None, save_dpi=None, transparent=False, color_t=None, color_c=None)
```

Overlap plot for m(x)=P(D=1|X) with high-res rendering.

- x in [0,1]
- Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
- Uses Matplotlib default colors unless color_t/color_c are provided

###### `causalis.eda.eda.PropensityModel.positivity_check`

```python
positivity_check(*args, **kwargs)
```

###### `causalis.eda.eda.PropensityModel.positivity_check_m`

```python
positivity_check_m(bounds=(0.05, 0.95))
```

Check overlap/positivity for m(x) based on thresholds.

###### `causalis.eda.eda.PropensityModel.propensity_scores`

```python
propensity_scores
```

###### `causalis.eda.eda.PropensityModel.ps_graph`

```python
ps_graph(*args, **kwargs)
```

###### `causalis.eda.eda.PropensityModel.roc_auc`

```python
roc_auc: float
```

Compute ROC AUC of treatment assignment vs. m(x).

Higher AUC means treatment is more predictable from confounders,
indicating stronger systematic differences between groups (potential
cofounding). Values near 0.5 suggest random-like assignment.

**Returns:**

- <code>[float](#float)</code> – ROC AUC score between 0 and 1

###### `causalis.eda.eda.PropensityModel.shap`

```python
shap: pd.DataFrame
```

Return feature attribution from the fitted propensity model.

For CatBoost models: returns SHAP-based attributions with both signed mean (shap_mean)
and magnitude (shap_mean_abs), sorted by shap_mean_abs.

For sklearn linear models (LogisticRegression): returns absolute coefficients under
column 'coef_abs' (not SHAP), sorted descending.

#### `causalis.eda.rct_design`

Design module for experimental rct_design utilities.

**Classes:**

- [**SRMResult**](#causalis.eda.rct_design.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**assign_variants_df**](#causalis.eda.rct_design.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.
- [**calculate_mde**](#causalis.eda.rct_design.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data.
- [**check_srm**](#causalis.eda.rct_design.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

##### `causalis.eda.rct_design.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.eda.rct_design.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.eda.rct_design.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.eda.rct_design.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.eda.rct_design.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.eda.rct_design.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.eda.rct_design.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.eda.rct_design.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.eda.rct_design.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `causalis.eda.rct_design.SRMResult.alpha`

```python
alpha: float
```

###### `causalis.eda.rct_design.SRMResult.chi2`

```python
chi2: float
```

###### `causalis.eda.rct_design.SRMResult.df`

```python
df: int
```

###### `causalis.eda.rct_design.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

###### `causalis.eda.rct_design.SRMResult.is_srm`

```python
is_srm: bool
```

###### `causalis.eda.rct_design.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

###### `causalis.eda.rct_design.SRMResult.p_value`

```python
p_value: float
```

###### `causalis.eda.rct_design.SRMResult.warning`

```python
warning: str | None = None
```

##### `causalis.eda.rct_design.assign_variants_df`

```python
assign_variants_df(df, id_col, experiment_id, variants, *, salt='global_ab_salt', layer_id='default', variant_col='variant')
```

Deterministically assign variants for each row in df based on id_col.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – Input DataFrame with an identifier column.
- **id_col** (<code>[str](#str)</code>) – Column name in df containing entity identifiers (user_id, session_id, etc.).
- **experiment_id** (<code>[str](#str)</code>) – Unique identifier for the experiment (versioned for reruns).
- **variants** (<code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code>) – Mapping from variant name to weight (coverage). Weights must be non-negative
  and their sum must be in (0, 1\]. If the sum is < 1, the remaining mass
  corresponds to "not in experiment" and the assignment will be None.
- **salt** (<code>[str](#str)</code>) – Secret string to de-correlate from other hash uses and make assignments
  non-gameable.
- **layer_id** (<code>[str](#str)</code>) – Identifier for mutual exclusivity layer or surface. In this case work like
  another random
- **variant_col** (<code>[str](#str)</code>) – Name of output column to store assigned variant labels.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of df with an extra column `variant_col`.
  Entities outside experiment coverage will have None in the variant column.

##### `causalis.eda.rct_design.calculate_mde`

```python
calculate_mde(sample_size, baseline_rate=None, variance=None, alpha=0.05, power=0.8, data_type='conversion', ratio=0.5)
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data) or baseline mean (for continuous data).
  Required for conversion data.
- **variance** (<code>float or tuple of float</code>) – Variance of the data. For conversion data, this is calculated from the baseline rate if not provided.
  For continuous data, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data. Either 'conversion' for binary/conversion data or 'continuous' for continuous data.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

##### `causalis.eda.rct_design.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.data.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.scenarios.rct.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
- Each p must be > 0.
- Sum of all p must be 1.0 (within numerical tolerance).

Examples:
{"control": 0.5, "treatment": 0.5}
{"A": 0.2, "B": 0.3, "C": 0.5}

- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.scenarios.rct.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

### `causalis.refutation`

Refutation and robustness utilities for Causalis.

Importing this package exposes the public functions from all refutation
submodules (overlap, score, uncofoundedness, sutva) so you can access
commonly used helpers directly via `causalis.refutation`.

**Modules:**

- [**overlap**](#causalis.refutation.overlap) –
- [**score**](#causalis.refutation.score) –
- [**sutva**](#causalis.refutation.sutva) –
- [**uncofoundedness**](#causalis.refutation.uncofoundedness) –

**Classes:**

- [**CausalData**](#causalis.refutation.CausalData) – Container for causal inference datasets.

**Functions:**

- [**add_score_flags**](#causalis.refutation.add_score_flags) – Augment run_score_diagnostics(...) dict with:
- [**aipw_score_ate**](#causalis.refutation.aipw_score_ate) – Efficient influence function (EIF) for ATE.
- [**aipw_score_atte**](#causalis.refutation.aipw_score_atte) – Efficient influence function (EIF) for ATTE under IRM/AIPW.
- [**att_overlap_tests**](#causalis.refutation.att_overlap_tests) – Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.
- [**att_weight_sum_identity**](#causalis.refutation.att_weight_sum_identity) – ATT weight-sum identity check (un-normalized IPW form).
- [**auc_for_m**](#causalis.refutation.auc_for_m) – ROC AUC using scores m_hat vs labels D.
- [**calibration_report_m**](#causalis.refutation.calibration_report_m) – Propensity calibration report for cross-fitted propensities m_hat against treatment D.
- [**ece_binary**](#causalis.refutation.ece_binary) – Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].
- [**edge_mass**](#causalis.refutation.edge_mass) – Edge mass diagnostics.
- [**ess_per_group**](#causalis.refutation.ess_per_group) – Effective sample size (ESS) for ATE-style inverse-probability weights per arm.
- [**extract_nuisances**](#causalis.refutation.extract_nuisances) – Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.
- [**get_sensitivity_summary**](#causalis.refutation.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**influence_summary**](#causalis.refutation.influence_summary) – Compute influence diagnostics showing where uncertainty comes from.
- [**ks_distance**](#causalis.refutation.ks_distance) – Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.
- [**oos_moment_check**](#causalis.refutation.oos_moment_check) – Out-of-sample moment check to avoid tautological results (legacy/simple version).
- [**oos_moment_check_from_psi**](#causalis.refutation.oos_moment_check_from_psi) – OOS moment check using cached ψ_a, ψ_b only.
- [**oos_moment_check_with_fold_nuisances**](#causalis.refutation.oos_moment_check_with_fold_nuisances) – Out-of-sample moment check using fold-specific nuisances to avoid tautological results.
- [**orthogonality_derivatives**](#causalis.refutation.orthogonality_derivatives) – Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
- [**orthogonality_derivatives_atte**](#causalis.refutation.orthogonality_derivatives_atte) – Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.
- [**overlap_diagnostics_atte**](#causalis.refutation.overlap_diagnostics_atte) – Key overlap metrics for ATTE: availability of suitable controls.
- [**overlap_report_from_result**](#causalis.refutation.overlap_report_from_result) – High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.
- [**positivity_overlap_checks**](#causalis.refutation.positivity_overlap_checks) – Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
- [**print_sutva_questions**](#causalis.refutation.print_sutva_questions) – Print the SUTVA validation questions.
- [**refute_irm_orthogonality**](#causalis.refutation.refute_irm_orthogonality) – Comprehensive AIPW orthogonality diagnostics for IRM models.
- [**refute_placebo_outcome**](#causalis.refutation.refute_placebo_outcome) – Generate random outcome variables while keeping treatment
- [**refute_placebo_treatment**](#causalis.refutation.refute_placebo_treatment) – Generate random binary treatment variables while keeping outcome and
- [**refute_subset**](#causalis.refutation.refute_subset) – Re-estimate the effect on a random subset (default 80 %)
- [**run_overlap_diagnostics**](#causalis.refutation.run_overlap_diagnostics) – Single entry-point for overlap / positivity / calibration diagnostics.
- [**run_score_diagnostics**](#causalis.refutation.run_score_diagnostics) – Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.
- [**run_uncofoundedness_diagnostics**](#causalis.refutation.run_uncofoundedness_diagnostics) – Uncofoundedness diagnostics focused on balance (SMD).
- [**sensitivity_analysis**](#causalis.refutation.sensitivity_analysis) – Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.
- [**sensitivity_benchmark**](#causalis.refutation.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model
- [**trim_sensitivity_curve_ate**](#causalis.refutation.trim_sensitivity_curve_ate) – Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).
- [**trim_sensitivity_curve_atte**](#causalis.refutation.trim_sensitivity_curve_atte) – Re-estimate θ while progressively trimming CONTROLS with large m(X).
- [**validate_uncofoundedness_balance**](#causalis.refutation.validate_uncofoundedness_balance) – Assess covariate balance under the uncofoundedness assumption by computing

**Attributes:**

- [**DEFAULT_THRESHOLDS**](#causalis.refutation.DEFAULT_THRESHOLDS) –
- [**QUESTIONS**](#causalis.refutation.QUESTIONS) (<code>[Iterable](#typing.Iterable)\[[str](#str)\]</code>) –
- [**ResultLike**](#causalis.refutation.ResultLike) –

#### `causalis.refutation.CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data contract.

**Attributes:**

- [**df**](#causalis.refutation.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.refutation.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.refutation.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.refutation.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.refutation.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.refutation.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.refutation.CausalData.get_df) – Get a DataFrame with specified columns.

##### `causalis.refutation.CausalData.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

##### `causalis.refutation.CausalData.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

##### `causalis.refutation.CausalData.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

##### `causalis.refutation.CausalData.df`

```python
df: pd.DataFrame
```

##### `causalis.refutation.CausalData.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, **kwargs)
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A validated CausalData instance.

##### `causalis.refutation.CausalData.get_df`

```python
get_df(columns=None, include_treatment=True, include_outcome=True, include_confounders=True, include_user_id=False)
```

Get a DataFrame with specified columns.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

##### `causalis.refutation.CausalData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

##### `causalis.refutation.CausalData.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

##### `causalis.refutation.CausalData.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

##### `causalis.refutation.CausalData.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

##### `causalis.refutation.CausalData.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

##### `causalis.refutation.CausalData.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

##### `causalis.refutation.CausalData.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.refutation.DEFAULT_THRESHOLDS`

```python
DEFAULT_THRESHOLDS = dict(edge_mass_warn_001=0.02, edge_mass_strong_001=0.05, edge_mass_warn_002=0.05, edge_mass_strong_002=0.1, ks_warn=0.3, ks_strong=0.4, auc_warn=0.8, auc_strong=0.9, ipw_relerr_warn=0.05, ipw_relerr_strong=0.1, ess_ratio_warn=0.3, ess_ratio_strong=0.15, clip_share_warn=0.02, clip_share_strong=0.05, tail_vs_med_warn=10.0)
```

#### `causalis.refutation.QUESTIONS`

```python
QUESTIONS: Iterable[str] = ('1.) Are your clients independent (i)?', '2.) Do you measure confounders, treatment, and outcome in the same intervals?', '3.) Do you measure confounders before treatment and outcome after?', '4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?')
```

#### `causalis.refutation.ResultLike`

```python
ResultLike = Dict[str, Any] | Any
```

#### `causalis.refutation.add_score_flags`

```python
add_score_flags(rep_score, thresholds=None, *, effect_size_guard=0.02, oos_gate=True, se_rule=None, se_ref=None)
```

Augment run_score_diagnostics(...) dict with:

- rep['flags'] (per-metric flags)
- rep['thresholds'] (the cutoffs used)
- rep['summary'] with a new 'flag' column
- rep['overall_flag'] (rollup)

Additional logic:

- Practical effect-size guard: if the constant-basis derivative magnitude is tiny
  (\<= effect_size_guard), then downgrade an orthogonality RED to GREEN (if OOS is GREEN)
  or to YELLOW (otherwise). Controlled by `oos_gate`.
- Huge-n relaxation: for very large n (>= 200k), relax tail/kurtosis flags slightly
  under specified value gates.

#### `causalis.refutation.aipw_score_ate`

```python
aipw_score_ate(y, d, g0, g1, m, theta, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATE.
Uses IRM naming: g0,g1 are outcome regressions E[Y|X,D=0/1], m is propensity P(D=1|X).

#### `causalis.refutation.aipw_score_atte`

```python
aipw_score_atte(y, d, g0, g1, m, theta, p_treated=None, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATTE under IRM/AIPW.

ψ_ATTE(W; θ, η) = \[ D\*(Y - g0(X) - θ) - (1-D)*{ m(X)/(1-m(X)) }*(Y - g0(X)) \] / E[D]

Notes:

- Matches DoubleML's `score='ATTE'` (weights ω=D/E[D], ar{ω}=m(X)/E[D]).
- g1 enters only via θ; ∂ψ/∂g1 = 0.

#### `causalis.refutation.att_overlap_tests`

```python
att_overlap_tests(dml_att_result, epsilon_list=(0.01, 0.02))
```

Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.

Inputs expected in result\['diagnostic_data'\]:

- m_hat: np.ndarray of cross-fitted propensity scores Pr(D=1|X)
- d: np.ndarray of treatment indicators {0,1}

Returns:
dict with keys:
\- edge_mass: {'eps': {eps: {'share_below': float, 'share_above': float, 'warn': bool}}}
\- ks: {'value': float, 'warn': bool}
\- auc: {'value': float or nan, 'flag': str} # 'GREEN'/'YELLOW'/'RED' or 'NA' if undefined
\- ess: {'treated': {'ess': float, 'n': int, 'ratio': float, 'flag': str},
'control': {'ess': float, 'n': int, 'ratio': float, 'flag': str}}
\- att_weight_identity: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float, 'flag': str}

#### `causalis.refutation.att_weight_sum_identity`

```python
att_weight_sum_identity(m_hat, D)
```

ATT weight-sum identity check (un-normalized IPW form).

Math:
w1_i = D_i / p1, w0_i = (1 - D_i) * m_hat_i / ((1 - m_hat_i) * p1), where p1 = (1/n) sum_i D_i.
Sum check: sum_i (1 - D_i) * m_hat_i / (1 - m_hat_i) ?≈ sum_i D_i.

Returns: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float}

#### `causalis.refutation.auc_for_m`

```python
auc_for_m(m_hat, D)
```

ROC AUC using scores m_hat vs labels D.

Math (Mann–Whitney relation):
AUC = P(m_i^+ > m_j^-) + 0.5 P(m_i^+ = m_j^-)

#### `causalis.refutation.calibration_report_m`

```python
calibration_report_m(m_hat, D, n_bins=10, *, thresholds=None)
```

Propensity calibration report for cross-fitted propensities m_hat against treatment D.

Returns a dictionary with:

- auc: ROC AUC of m_hat vs D (Mann–Whitney)
- brier: Brier score (mean squared error)
- ece: Expected Calibration Error (equal-width bins)
- reliability_table: pd.DataFrame with per-bin stats
- recalibration: {'intercept': alpha, 'slope': beta} from logistic recalibration
- flags: {'ece': ..., 'slope': ..., 'intercept': ...} using GREEN/YELLOW/RED

#### `causalis.refutation.ece_binary`

```python
ece_binary(p, y, n_bins=10)
```

Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].

**Parameters:**

- **p** (<code>[ndarray](#numpy.ndarray)</code>) – Predicted probabilities in [0,1]. Will be clipped to [0,1].
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Binary labels {0,1}.
- **n_bins** (<code>[int](#int)</code>) – Number of bins.

**Returns:**

- <code>[float](#float)</code> – ECE value in [0,1].

#### `causalis.refutation.edge_mass`

```python
edge_mass(m_hat, eps=0.01)
```

Edge mass diagnostics.

Math:
share_below = (1/n) * sum_i 1{ m_hat_i < ε }
share_above = (1/n) * sum_i 1{ m_hat_i > 1 - ε }

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Array of propensities m_hat in [0,1].
- **eps** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – A single ε or a sequence of ε values.

**Returns:**

- <code>[dict](#dict)</code> – - If eps is a scalar: {'eps': ε, 'share_below': float, 'share_above': float}
- If eps is a sequence: {ε: {'share_below': float, 'share_above': float}, ...}

#### `causalis.refutation.ess_per_group`

```python
ess_per_group(m_hat, D)
```

Effective sample size (ESS) for ATE-style inverse-probability weights per arm.

Weights:
w1_i = D_i / m_hat_i,
w0_i = (1 - D_i) / (1 - m_hat_i).

ESS:
ESS(w_g) = (sum_i w\_{gi})^2 / sum_i w\_{gi}^2.

Returns dict with ess and ratios (ESS / group size).

#### `causalis.refutation.extract_nuisances`

```python
extract_nuisances(model, test_indices=None)
```

Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.

Tries several backends for robustness:

1. IRM attributes: m_hat\_, g0_hat\_, g1_hat\_
1. model.predictions dict with keys: 'ml_m','ml_g0','ml_g1'
1. Direct attributes: ml_m, ml_g0, ml_g1

**Parameters:**

- **model** (<code>[object](#object)</code>) – Fitted internal IRM estimator (causalis.statistics.models.IRM) or a compatible dummy model
- **test_indices** (<code>[ndarray](#numpy.ndarray)</code>) – If provided, extract predictions only for these indices

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]</code> – (m, g0, g1) where:
- m: propensity scores P(D=1|X)
- g0: outcome predictions E[Y|X,D=0]
- g1: outcome predictions E[Y|X,D=1]

#### `causalis.refutation.get_sensitivity_summary`

```python
get_sensitivity_summary(effect_estimation, *, label=None)
```

Render a single, unified bias-aware summary string.
If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

#### `causalis.refutation.influence_summary`

```python
influence_summary(y, d, g0, g1, m, theta_hat, k=10, score='ATE', trimming_threshold=0.01)
```

Compute influence diagnostics showing where uncertainty comes from.

**Parameters:**

- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **theta_hat** (<code>[float](#float)</code>) – Estimated treatment effect
- **k** (<code>[int](#int)</code>) – Number of top influential observations to return

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Influence diagnostics including SE, heavy-tail metrics, and top-k cases

#### `causalis.refutation.ks_distance`

```python
ks_distance(m_hat, D)
```

Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.

Math:
KS = sup_t | F\_{m|D=1}(t) - F\_{m|D=0}(t) |

#### `causalis.refutation.oos_moment_check`

```python
oos_moment_check(fold_thetas, fold_indices, y, d, g0, g1, m, score_fn=None)
```

Out-of-sample moment check to avoid tautological results (legacy/simple version).

For each fold k, evaluates the AIPW score using θ fitted on other folds,
then tests if the combined moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

#### `causalis.refutation.oos_moment_check_from_psi`

```python
oos_moment_check_from_psi(psi_a, psi_b, fold_indices, *, strict=False)
```

OOS moment check using cached ψ_a, ψ_b only.
Returns (fold-wise DF, t_fold_agg, t_strict if requested).

#### `causalis.refutation.oos_moment_check_with_fold_nuisances`

```python
oos_moment_check_with_fold_nuisances(fold_thetas, fold_indices, fold_nuisances, y, d, score_fn=None)
```

Out-of-sample moment check using fold-specific nuisances to avoid tautological results.

For each fold k, evaluates the AIPW score using θ fitted on other folds and
nuisance predictions from the fold-specific model, then tests if the combined
moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **fold_nuisances** (<code>[List](#typing.List)\[[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]\]</code>) – Fold-specific nuisance predictions (m, g0, g1) for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

#### `causalis.refutation.orthogonality_derivatives`

```python
orthogonality_derivatives(X_basis, y, d, g0, g1, m, trimming_threshold=0.01)
```

Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
Uses IRM naming: g0,g1 outcomes; m propensity.

#### `causalis.refutation.orthogonality_derivatives_atte`

```python
orthogonality_derivatives_atte(X_basis, y, d, g0, m, p_treated, trimming_threshold=0.01)
```

Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.

For ψ_ATTE = \[ D\*(Y - g0 - θ) - (1-D)*(m/(1-m))*(Y - g0) \] / p_treated:

∂\_{g0}[h] : (1/n) Σ h(X_i) * \[ ((1-D_i)*m_i/(1-m_i) - D_i) / p_treated \]
∂\_{m}[s] : (1/n) Σ s(X_i) * \[ -(1-D_i)*(Y_i - g0_i) / ( p_treated * (1-m_i)^2 ) \]

Both have 0 expectation at the truth (Neyman orthogonality).

#### `causalis.refutation.overlap`

**Functions:**

- [**att_overlap_tests**](#causalis.refutation.overlap.att_overlap_tests) – Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.
- [**att_weight_sum_identity**](#causalis.refutation.overlap.att_weight_sum_identity) – ATT weight-sum identity check (un-normalized IPW form).
- [**auc_for_m**](#causalis.refutation.overlap.auc_for_m) – ROC AUC using scores m_hat vs labels D.
- [**calibration_report_m**](#causalis.refutation.overlap.calibration_report_m) – Propensity calibration report for cross-fitted propensities m_hat against treatment D.
- [**ece_binary**](#causalis.refutation.overlap.ece_binary) – Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].
- [**edge_mass**](#causalis.refutation.overlap.edge_mass) – Edge mass diagnostics.
- [**ess_per_group**](#causalis.refutation.overlap.ess_per_group) – Effective sample size (ESS) for ATE-style inverse-probability weights per arm.
- [**ks_distance**](#causalis.refutation.overlap.ks_distance) – Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.
- [**overlap_report_from_result**](#causalis.refutation.overlap.overlap_report_from_result) – High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.
- [**positivity_overlap_checks**](#causalis.refutation.overlap.positivity_overlap_checks) – Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
- [**run_overlap_diagnostics**](#causalis.refutation.overlap.run_overlap_diagnostics) – Single entry-point for overlap / positivity / calibration diagnostics.

**Attributes:**

- [**DEFAULT_THRESHOLDS**](#causalis.refutation.overlap.DEFAULT_THRESHOLDS) –

##### `causalis.refutation.overlap.DEFAULT_THRESHOLDS`

```python
DEFAULT_THRESHOLDS = dict(edge_mass_warn_001=0.02, edge_mass_strong_001=0.05, edge_mass_warn_002=0.05, edge_mass_strong_002=0.1, ks_warn=0.3, ks_strong=0.4, auc_warn=0.8, auc_strong=0.9, ipw_relerr_warn=0.05, ipw_relerr_strong=0.1, ess_ratio_warn=0.3, ess_ratio_strong=0.15, clip_share_warn=0.02, clip_share_strong=0.05, tail_vs_med_warn=10.0)
```

##### `causalis.refutation.overlap.att_overlap_tests`

```python
att_overlap_tests(dml_att_result, epsilon_list=(0.01, 0.02))
```

Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.

Inputs expected in result\['diagnostic_data'\]:

- m_hat: np.ndarray of cross-fitted propensity scores Pr(D=1|X)
- d: np.ndarray of treatment indicators {0,1}

Returns:
dict with keys:
\- edge_mass: {'eps': {eps: {'share_below': float, 'share_above': float, 'warn': bool}}}
\- ks: {'value': float, 'warn': bool}
\- auc: {'value': float or nan, 'flag': str} # 'GREEN'/'YELLOW'/'RED' or 'NA' if undefined
\- ess: {'treated': {'ess': float, 'n': int, 'ratio': float, 'flag': str},
'control': {'ess': float, 'n': int, 'ratio': float, 'flag': str}}
\- att_weight_identity: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float, 'flag': str}

##### `causalis.refutation.overlap.att_weight_sum_identity`

```python
att_weight_sum_identity(m_hat, D)
```

ATT weight-sum identity check (un-normalized IPW form).

Math:
w1_i = D_i / p1, w0_i = (1 - D_i) * m_hat_i / ((1 - m_hat_i) * p1), where p1 = (1/n) sum_i D_i.
Sum check: sum_i (1 - D_i) * m_hat_i / (1 - m_hat_i) ?≈ sum_i D_i.

Returns: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float}

##### `causalis.refutation.overlap.auc_for_m`

```python
auc_for_m(m_hat, D)
```

ROC AUC using scores m_hat vs labels D.

Math (Mann–Whitney relation):
AUC = P(m_i^+ > m_j^-) + 0.5 P(m_i^+ = m_j^-)

##### `causalis.refutation.overlap.calibration_report_m`

```python
calibration_report_m(m_hat, D, n_bins=10, *, thresholds=None)
```

Propensity calibration report for cross-fitted propensities m_hat against treatment D.

Returns a dictionary with:

- auc: ROC AUC of m_hat vs D (Mann–Whitney)
- brier: Brier score (mean squared error)
- ece: Expected Calibration Error (equal-width bins)
- reliability_table: pd.DataFrame with per-bin stats
- recalibration: {'intercept': alpha, 'slope': beta} from logistic recalibration
- flags: {'ece': ..., 'slope': ..., 'intercept': ...} using GREEN/YELLOW/RED

##### `causalis.refutation.overlap.ece_binary`

```python
ece_binary(p, y, n_bins=10)
```

Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].

**Parameters:**

- **p** (<code>[ndarray](#numpy.ndarray)</code>) – Predicted probabilities in [0,1]. Will be clipped to [0,1].
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Binary labels {0,1}.
- **n_bins** (<code>[int](#int)</code>) – Number of bins.

**Returns:**

- <code>[float](#float)</code> – ECE value in [0,1].

##### `causalis.refutation.overlap.edge_mass`

```python
edge_mass(m_hat, eps=0.01)
```

Edge mass diagnostics.

Math:
share_below = (1/n) * sum_i 1{ m_hat_i < ε }
share_above = (1/n) * sum_i 1{ m_hat_i > 1 - ε }

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Array of propensities m_hat in [0,1].
- **eps** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – A single ε or a sequence of ε values.

**Returns:**

- <code>[dict](#dict)</code> – - If eps is a scalar: {'eps': ε, 'share_below': float, 'share_above': float}
- If eps is a sequence: {ε: {'share_below': float, 'share_above': float}, ...}

##### `causalis.refutation.overlap.ess_per_group`

```python
ess_per_group(m_hat, D)
```

Effective sample size (ESS) for ATE-style inverse-probability weights per arm.

Weights:
w1_i = D_i / m_hat_i,
w0_i = (1 - D_i) / (1 - m_hat_i).

ESS:
ESS(w_g) = (sum_i w\_{gi})^2 / sum_i w\_{gi}^2.

Returns dict with ess and ratios (ESS / group size).

##### `causalis.refutation.overlap.ks_distance`

```python
ks_distance(m_hat, D)
```

Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.

Math:
KS = sup_t | F\_{m|D=1}(t) - F\_{m|D=0}(t) |

##### `causalis.refutation.overlap.overlap_report_from_result`

```python
overlap_report_from_result(res, *, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
dml_ate(normalize_ipw=True) get meaningful ipw_sum\_\* checks without extra arguments.

##### `causalis.refutation.overlap.positivity_overlap_checks`

```python
positivity_overlap_checks(m_hat, D, *, m_clipped_from=None, g_clipped_share=None, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
Inputs are cross-fitted m̂ and treatment D (0/1). Returns a structured report with GREEN/YELLOW/RED flags.

##### `causalis.refutation.overlap.run_overlap_diagnostics`

```python
run_overlap_diagnostics(res=None, *, m_hat=None, D=None, thresholds=DEFAULT_THRESHOLDS, n_bins=10, use_hajek=None, m_clipped_from=None, g_clipped_share=None, return_summary=True, cal_thresholds=None, auc_flip_margin=0.05)
```

Single entry-point for overlap / positivity / calibration diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_overlap_diagnostics(m_hat=..., D=...)
B) With a model/result:
run_overlap_diagnostics(res=\<dml_ate/dml_att result dict or IRM/DoubleML-like model>)

The function:

- Auto-extracts (m_hat, D, trimming_threshold) from `res` if provided.
- Auto-detects Hájek normalization if available on `res` (normalize_ipw).
- Runs positivity/overlap checks (edge mass, KS, AUC, ESS, tails, ATT identity),
  clipping audit, and calibration (ECE + logistic recalibration).
- Returns a dict with full details and, optionally, a compact summary DataFrame.

**Returns:**

- <code>[dict](#dict)</code> – A dictionary with keys including:
  - n, n_treated, p1
  - edge_mass, edge_mass_by_arm, ks, auc
  - ate_ipw, ate_ess, ate_tails
  - att_weights, att_ess
  - clipping
  - calibration (with reliability_table)
  - flags (GREEN/YELLOW/RED/NA)
  - summary (pd.DataFrame) if return_summary=True
  - meta (use_hajek, thresholds)

#### `causalis.refutation.overlap_diagnostics_atte`

```python
overlap_diagnostics_atte(m, d, eps_list=[0.95, 0.97, 0.98, 0.99])
```

Key overlap metrics for ATTE: availability of suitable controls.
Reports conditional shares: among CONTROLS, fraction with m(X) ≥ threshold; among TREATED, fraction with m(X) ≤ 1 - threshold.

#### `causalis.refutation.overlap_report_from_result`

```python
overlap_report_from_result(res, *, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
dml_ate(normalize_ipw=True) get meaningful ipw_sum\_\* checks without extra arguments.

#### `causalis.refutation.positivity_overlap_checks`

```python
positivity_overlap_checks(m_hat, D, *, m_clipped_from=None, g_clipped_share=None, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
Inputs are cross-fitted m̂ and treatment D (0/1). Returns a structured report with GREEN/YELLOW/RED flags.

#### `causalis.refutation.print_sutva_questions`

```python
print_sutva_questions()
```

Print the SUTVA validation questions.

Just prints questions, nothing more.

#### `causalis.refutation.refute_irm_orthogonality`

```python
refute_irm_orthogonality(inference_fn, data, trim_propensity=(0.02, 0.98), n_basis_funcs=None, n_folds_oos=4, score=None, trimming_threshold=0.01, strict_oos=True, **inference_kwargs)
```

Comprehensive AIPW orthogonality diagnostics for IRM models.

Implements three key diagnostic approaches based on the efficient influence function (EIF):

1. Out-of-sample moment check (non-tautological)
1. Orthogonality (Gateaux derivative) tests
1. Influence diagnostics

**Parameters:**

- **inference_fn** (<code>[Callable](#typing.Callable)</code>) – The inference function (dml_ate or dml_att)
- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causal data object
- **trim_propensity** (<code>[Tuple](#typing.Tuple)\[[float](#float), [float](#float)\]</code>) – Propensity score trimming bounds (min, max) to avoid extreme weights
- **n_basis_funcs** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Number of basis functions for orthogonality derivative tests (constant + covariates).
  If None, defaults to the number of confounders in `data` plus 1 for the constant term.
- **n_folds_oos** (<code>[int](#int)</code>) – Number of folds for out-of-sample moment check
- \*\***inference_kwargs** (<code>[dict](#dict)</code>) – Additional arguments passed to inference_fn

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary containing:
- oos_moment_test: Out-of-sample moment condition results
- orthogonality_derivatives: Gateaux derivative test results
- influence_diagnostics: Influence function diagnostics
- theta: Original treatment effect estimate
- trimmed_diagnostics: Results on trimmed sample
- overall_assessment: Summary diagnostic assessment

**Examples:**

```pycon
>>> from causalis.refutation.orthogonality import refute_irm_orthogonality
>>> from causalis.scenarios.unconfoundedness.ate.dml_ate import dml_ate
>>> 
>>> # Comprehensive orthogonality check
>>> ortho_results = refute_irm_orthogonality(dml_ate, causal_data)
>>> 
>>> # Check key diagnostics
>>> print(f"OOS moment t-stat: {ortho_results['oos_moment_test']['tstat']:.3f}")
>>> print(f"Assessment: {ortho_results['overall_assessment']}")
```

#### `causalis.refutation.refute_placebo_outcome`

```python
refute_placebo_outcome(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random outcome variables while keeping treatment
and covariates intact. For binary outcomes, generates random binary
variables with the same proportion. For continuous outcomes, generates
random variables from a normal distribution fitted to the original data.
A valid causal design should now yield θ ≈ 0 and a large p-value.

#### `causalis.refutation.refute_placebo_treatment`

```python
refute_placebo_treatment(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random binary treatment variables while keeping outcome and
covariates intact. Generates random binary treatment with the same
proportion as the original treatment. Breaks the treatment–outcome link.

#### `causalis.refutation.refute_subset`

```python
refute_subset(inference_fn, data, fraction=0.8, random_state=None, **inference_kwargs)
```

Re-estimate the effect on a random subset (default 80 %)
to check sample-stability of the estimate.

#### `causalis.refutation.run_overlap_diagnostics`

```python
run_overlap_diagnostics(res=None, *, m_hat=None, D=None, thresholds=DEFAULT_THRESHOLDS, n_bins=10, use_hajek=None, m_clipped_from=None, g_clipped_share=None, return_summary=True, cal_thresholds=None, auc_flip_margin=0.05)
```

Single entry-point for overlap / positivity / calibration diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_overlap_diagnostics(m_hat=..., D=...)
B) With a model/result:
run_overlap_diagnostics(res=\<dml_ate/dml_att result dict or IRM/DoubleML-like model>)

The function:

- Auto-extracts (m_hat, D, trimming_threshold) from `res` if provided.
- Auto-detects Hájek normalization if available on `res` (normalize_ipw).
- Runs positivity/overlap checks (edge mass, KS, AUC, ESS, tails, ATT identity),
  clipping audit, and calibration (ECE + logistic recalibration).
- Returns a dict with full details and, optionally, a compact summary DataFrame.

**Returns:**

- <code>[dict](#dict)</code> – A dictionary with keys including:
  - n, n_treated, p1
  - edge_mass, edge_mass_by_arm, ks, auc
  - ate_ipw, ate_ess, ate_tails
  - att_weights, att_ess
  - clipping
  - calibration (with reliability_table)
  - flags (GREEN/YELLOW/RED/NA)
  - summary (pd.DataFrame) if return_summary=True
  - meta (use_hajek, thresholds)

#### `causalis.refutation.run_score_diagnostics`

```python
run_score_diagnostics(res=None, *, y=None, d=None, g0=None, g1=None, m=None, theta=None, score=None, trimming_threshold=0.01, n_basis_funcs=None, return_summary=True)
```

Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_score_diagnostics(y=..., d=..., g0=..., g1=..., m=..., theta=...)
B) With a model/result:
run_score_diagnostics(res=\<dml_ate/dml_att result dict or IRM-like model>)

Returns a dictionary with:

- params (score, trimming_threshold)
- oos_moment_test (if fast-path caches available on model; else omitted)
- orthogonality_derivatives (DataFrame)
- influence_diagnostics (full_sample)
- summary (compact DataFrame) if return_summary=True
- meta

#### `causalis.refutation.run_uncofoundedness_diagnostics`

```python
run_uncofoundedness_diagnostics(*, res=None, X=None, d=None, m_hat=None, names=None, score=None, normalize=None, threshold=0.1, eps_overlap=0.01, return_summary=True)
```

Uncofoundedness diagnostics focused on balance (SMD).

Inputs:

- Either a result/model via `res`, or raw arrays X, d, m_hat (+ optional names, score, normalize).

Returns a dictionary:
{
"params": {"score", "normalize", "smd_threshold"},
"balance": {"smd", "smd_unweighted", "smd_max", "frac_violations", "pass", "worst_features"},
"flags": {"balance_max_smd", "balance_violations"},
"overall_flag": max severity across balance flags,
"summary": pd.DataFrame with balance rows only
}

#### `causalis.refutation.score`

**Modules:**

- [**score_validation**](#causalis.refutation.score.score_validation) – AIPW orthogonality diagnostics for IRM-based models.

##### `causalis.refutation.score.score_validation`

AIPW orthogonality diagnostics for IRM-based models.

This module implements comprehensive orthogonality diagnostics for AIPW/IRM-based
models like dml_ate and dml_att to validate the key assumptions required
for valid causal inference. Based on the efficient influence function (EIF) framework.

Key diagnostics implemented:

- Out-of-sample moment check (non-tautological)
- Orthogonality (Gateaux derivative) tests
- Influence diagnostics

**Functions:**

- [**add_score_flags**](#causalis.refutation.score.score_validation.add_score_flags) – Augment run_score_diagnostics(...) dict with:
- [**aipw_score_ate**](#causalis.refutation.score.score_validation.aipw_score_ate) – Efficient influence function (EIF) for ATE.
- [**aipw_score_atte**](#causalis.refutation.score.score_validation.aipw_score_atte) – Efficient influence function (EIF) for ATTE under IRM/AIPW.
- [**extract_nuisances**](#causalis.refutation.score.score_validation.extract_nuisances) – Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.
- [**influence_summary**](#causalis.refutation.score.score_validation.influence_summary) – Compute influence diagnostics showing where uncertainty comes from.
- [**oos_moment_check**](#causalis.refutation.score.score_validation.oos_moment_check) – Out-of-sample moment check to avoid tautological results (legacy/simple version).
- [**oos_moment_check_from_psi**](#causalis.refutation.score.score_validation.oos_moment_check_from_psi) – OOS moment check using cached ψ_a, ψ_b only.
- [**oos_moment_check_with_fold_nuisances**](#causalis.refutation.score.score_validation.oos_moment_check_with_fold_nuisances) – Out-of-sample moment check using fold-specific nuisances to avoid tautological results.
- [**orthogonality_derivatives**](#causalis.refutation.score.score_validation.orthogonality_derivatives) – Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
- [**orthogonality_derivatives_atte**](#causalis.refutation.score.score_validation.orthogonality_derivatives_atte) – Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.
- [**overlap_diagnostics_atte**](#causalis.refutation.score.score_validation.overlap_diagnostics_atte) – Key overlap metrics for ATTE: availability of suitable controls.
- [**refute_irm_orthogonality**](#causalis.refutation.score.score_validation.refute_irm_orthogonality) – Comprehensive AIPW orthogonality diagnostics for IRM models.
- [**refute_placebo_outcome**](#causalis.refutation.score.score_validation.refute_placebo_outcome) – Generate random outcome variables while keeping treatment
- [**refute_placebo_treatment**](#causalis.refutation.score.score_validation.refute_placebo_treatment) – Generate random binary treatment variables while keeping outcome and
- [**refute_subset**](#causalis.refutation.score.score_validation.refute_subset) – Re-estimate the effect on a random subset (default 80 %)
- [**run_score_diagnostics**](#causalis.refutation.score.score_validation.run_score_diagnostics) – Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.
- [**trim_sensitivity_curve_ate**](#causalis.refutation.score.score_validation.trim_sensitivity_curve_ate) – Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).
- [**trim_sensitivity_curve_atte**](#causalis.refutation.score.score_validation.trim_sensitivity_curve_atte) – Re-estimate θ while progressively trimming CONTROLS with large m(X).

**Attributes:**

- [**ResultLike**](#causalis.refutation.score.score_validation.ResultLike) –

###### `causalis.refutation.score.score_validation.ResultLike`

```python
ResultLike = Dict[str, Any] | Any
```

###### `causalis.refutation.score.score_validation.add_score_flags`

```python
add_score_flags(rep_score, thresholds=None, *, effect_size_guard=0.02, oos_gate=True, se_rule=None, se_ref=None)
```

Augment run_score_diagnostics(...) dict with:

- rep['flags'] (per-metric flags)
- rep['thresholds'] (the cutoffs used)
- rep['summary'] with a new 'flag' column
- rep['overall_flag'] (rollup)

Additional logic:

- Practical effect-size guard: if the constant-basis derivative magnitude is tiny
  (\<= effect_size_guard), then downgrade an orthogonality RED to GREEN (if OOS is GREEN)
  or to YELLOW (otherwise). Controlled by `oos_gate`.
- Huge-n relaxation: for very large n (>= 200k), relax tail/kurtosis flags slightly
  under specified value gates.

###### `causalis.refutation.score.score_validation.aipw_score_ate`

```python
aipw_score_ate(y, d, g0, g1, m, theta, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATE.
Uses IRM naming: g0,g1 are outcome regressions E[Y|X,D=0/1], m is propensity P(D=1|X).

###### `causalis.refutation.score.score_validation.aipw_score_atte`

```python
aipw_score_atte(y, d, g0, g1, m, theta, p_treated=None, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATTE under IRM/AIPW.

ψ_ATTE(W; θ, η) = \[ D\*(Y - g0(X) - θ) - (1-D)*{ m(X)/(1-m(X)) }*(Y - g0(X)) \] / E[D]

Notes:

- Matches DoubleML's `score='ATTE'` (weights ω=D/E[D], ar{ω}=m(X)/E[D]).
- g1 enters only via θ; ∂ψ/∂g1 = 0.

###### `causalis.refutation.score.score_validation.extract_nuisances`

```python
extract_nuisances(model, test_indices=None)
```

Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.

Tries several backends for robustness:

1. IRM attributes: m_hat\_, g0_hat\_, g1_hat\_
1. model.predictions dict with keys: 'ml_m','ml_g0','ml_g1'
1. Direct attributes: ml_m, ml_g0, ml_g1

**Parameters:**

- **model** (<code>[object](#object)</code>) – Fitted internal IRM estimator (causalis.statistics.models.IRM) or a compatible dummy model
- **test_indices** (<code>[ndarray](#numpy.ndarray)</code>) – If provided, extract predictions only for these indices

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]</code> – (m, g0, g1) where:
- m: propensity scores P(D=1|X)
- g0: outcome predictions E[Y|X,D=0]
- g1: outcome predictions E[Y|X,D=1]

###### `causalis.refutation.score.score_validation.influence_summary`

```python
influence_summary(y, d, g0, g1, m, theta_hat, k=10, score='ATE', trimming_threshold=0.01)
```

Compute influence diagnostics showing where uncertainty comes from.

**Parameters:**

- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **theta_hat** (<code>[float](#float)</code>) – Estimated treatment effect
- **k** (<code>[int](#int)</code>) – Number of top influential observations to return

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Influence diagnostics including SE, heavy-tail metrics, and top-k cases

###### `causalis.refutation.score.score_validation.oos_moment_check`

```python
oos_moment_check(fold_thetas, fold_indices, y, d, g0, g1, m, score_fn=None)
```

Out-of-sample moment check to avoid tautological results (legacy/simple version).

For each fold k, evaluates the AIPW score using θ fitted on other folds,
then tests if the combined moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

###### `causalis.refutation.score.score_validation.oos_moment_check_from_psi`

```python
oos_moment_check_from_psi(psi_a, psi_b, fold_indices, *, strict=False)
```

OOS moment check using cached ψ_a, ψ_b only.
Returns (fold-wise DF, t_fold_agg, t_strict if requested).

###### `causalis.refutation.score.score_validation.oos_moment_check_with_fold_nuisances`

```python
oos_moment_check_with_fold_nuisances(fold_thetas, fold_indices, fold_nuisances, y, d, score_fn=None)
```

Out-of-sample moment check using fold-specific nuisances to avoid tautological results.

For each fold k, evaluates the AIPW score using θ fitted on other folds and
nuisance predictions from the fold-specific model, then tests if the combined
moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **fold_nuisances** (<code>[List](#typing.List)\[[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]\]</code>) – Fold-specific nuisance predictions (m, g0, g1) for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

###### `causalis.refutation.score.score_validation.orthogonality_derivatives`

```python
orthogonality_derivatives(X_basis, y, d, g0, g1, m, trimming_threshold=0.01)
```

Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
Uses IRM naming: g0,g1 outcomes; m propensity.

###### `causalis.refutation.score.score_validation.orthogonality_derivatives_atte`

```python
orthogonality_derivatives_atte(X_basis, y, d, g0, m, p_treated, trimming_threshold=0.01)
```

Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.

For ψ_ATTE = \[ D\*(Y - g0 - θ) - (1-D)*(m/(1-m))*(Y - g0) \] / p_treated:

∂\_{g0}[h] : (1/n) Σ h(X_i) * \[ ((1-D_i)*m_i/(1-m_i) - D_i) / p_treated \]
∂\_{m}[s] : (1/n) Σ s(X_i) * \[ -(1-D_i)*(Y_i - g0_i) / ( p_treated * (1-m_i)^2 ) \]

Both have 0 expectation at the truth (Neyman orthogonality).

###### `causalis.refutation.score.score_validation.overlap_diagnostics_atte`

```python
overlap_diagnostics_atte(m, d, eps_list=[0.95, 0.97, 0.98, 0.99])
```

Key overlap metrics for ATTE: availability of suitable controls.
Reports conditional shares: among CONTROLS, fraction with m(X) ≥ threshold; among TREATED, fraction with m(X) ≤ 1 - threshold.

###### `causalis.refutation.score.score_validation.refute_irm_orthogonality`

```python
refute_irm_orthogonality(inference_fn, data, trim_propensity=(0.02, 0.98), n_basis_funcs=None, n_folds_oos=4, score=None, trimming_threshold=0.01, strict_oos=True, **inference_kwargs)
```

Comprehensive AIPW orthogonality diagnostics for IRM models.

Implements three key diagnostic approaches based on the efficient influence function (EIF):

1. Out-of-sample moment check (non-tautological)
1. Orthogonality (Gateaux derivative) tests
1. Influence diagnostics

**Parameters:**

- **inference_fn** (<code>[Callable](#typing.Callable)</code>) – The inference function (dml_ate or dml_att)
- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causal data object
- **trim_propensity** (<code>[Tuple](#typing.Tuple)\[[float](#float), [float](#float)\]</code>) – Propensity score trimming bounds (min, max) to avoid extreme weights
- **n_basis_funcs** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Number of basis functions for orthogonality derivative tests (constant + covariates).
  If None, defaults to the number of confounders in `data` plus 1 for the constant term.
- **n_folds_oos** (<code>[int](#int)</code>) – Number of folds for out-of-sample moment check
- \*\***inference_kwargs** (<code>[dict](#dict)</code>) – Additional arguments passed to inference_fn

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary containing:
- oos_moment_test: Out-of-sample moment condition results
- orthogonality_derivatives: Gateaux derivative test results
- influence_diagnostics: Influence function diagnostics
- theta: Original treatment effect estimate
- trimmed_diagnostics: Results on trimmed sample
- overall_assessment: Summary diagnostic assessment

**Examples:**

```pycon
>>> from causalis.refutation.orthogonality import refute_irm_orthogonality
>>> from causalis.scenarios.unconfoundedness.ate.dml_ate import dml_ate
>>> 
>>> # Comprehensive orthogonality check
>>> ortho_results = refute_irm_orthogonality(dml_ate, causal_data)
>>> 
>>> # Check key diagnostics
>>> print(f"OOS moment t-stat: {ortho_results['oos_moment_test']['tstat']:.3f}")
>>> print(f"Assessment: {ortho_results['overall_assessment']}")
```

###### `causalis.refutation.score.score_validation.refute_placebo_outcome`

```python
refute_placebo_outcome(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random outcome variables while keeping treatment
and covariates intact. For binary outcomes, generates random binary
variables with the same proportion. For continuous outcomes, generates
random variables from a normal distribution fitted to the original data.
A valid causal design should now yield θ ≈ 0 and a large p-value.

###### `causalis.refutation.score.score_validation.refute_placebo_treatment`

```python
refute_placebo_treatment(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random binary treatment variables while keeping outcome and
covariates intact. Generates random binary treatment with the same
proportion as the original treatment. Breaks the treatment–outcome link.

###### `causalis.refutation.score.score_validation.refute_subset`

```python
refute_subset(inference_fn, data, fraction=0.8, random_state=None, **inference_kwargs)
```

Re-estimate the effect on a random subset (default 80 %)
to check sample-stability of the estimate.

###### `causalis.refutation.score.score_validation.run_score_diagnostics`

```python
run_score_diagnostics(res=None, *, y=None, d=None, g0=None, g1=None, m=None, theta=None, score=None, trimming_threshold=0.01, n_basis_funcs=None, return_summary=True)
```

Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_score_diagnostics(y=..., d=..., g0=..., g1=..., m=..., theta=...)
B) With a model/result:
run_score_diagnostics(res=\<dml_ate/dml_att result dict or IRM-like model>)

Returns a dictionary with:

- params (score, trimming_threshold)
- oos_moment_test (if fast-path caches available on model; else omitted)
- orthogonality_derivatives (DataFrame)
- influence_diagnostics (full_sample)
- summary (compact DataFrame) if return_summary=True
- meta

###### `causalis.refutation.score.score_validation.trim_sensitivity_curve_ate`

```python
trim_sensitivity_curve_ate(m_hat, D, Y, g0_hat, g1_hat, eps_grid=(0.0, 0.005, 0.01, 0.02, 0.05))
```

Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).

For each epsilon in eps_grid, compute the AIPW/IRM ATE estimate using
m_clipped = clip(m_hat, eps, 1-eps) over the full sample and report
the plug-in standard error from the EIF.

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **D** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **Y** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g0_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g1_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **eps_grid** (<code>[tuple](#tuple)\[[float](#float), ...\]</code>) – Sequence of clipping thresholds ε to evaluate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Columns: ['trim_eps','n','pct_clipped','theta','se'].
  pct_clipped is the percent of observations with m outside [ε,1-ε].

###### `causalis.refutation.score.score_validation.trim_sensitivity_curve_atte`

```python
trim_sensitivity_curve_atte(inference_fn, data, m, d, thresholds=np.linspace(0.9, 0.995, 12), **inference_kwargs)
```

Re-estimate θ while progressively trimming CONTROLS with large m(X).

#### `causalis.refutation.sensitivity_analysis`

```python
sensitivity_analysis(effect_estimation, *, cf_y, cf_d, rho=1.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.

Returns a dict with:

- theta, se, alpha, z
- sampling_ci
- theta_bounds_cofounding = (theta - max_bias, theta + max_bias)
- bias_aware_ci = \[theta - (max_bias + z*se), theta + (max_bias + z*se)\]
- max_bias and components (sigma2, nu2)
- params (cf_y, cf_d, rho, use_signed_rr)

#### `causalis.refutation.sensitivity_benchmark`

```python
sensitivity_benchmark(effect_estimation, benchmarking_set, fit_args=None)
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing cf_y, cf_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Additional keyword arguments for the IRM.fit() method of the short model.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- cf_y, cf_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

#### `causalis.refutation.sutva`

**Modules:**

- [**sutva_validation**](#causalis.refutation.sutva.sutva_validation) – SUTVA validation helper.

##### `causalis.refutation.sutva.sutva_validation`

SUTVA validation helper.

This module provides a simple function to print four SUTVA-related
questions for the user to consider. It has no side effects on import.

**Functions:**

- [**print_sutva_questions**](#causalis.refutation.sutva.sutva_validation.print_sutva_questions) – Print the SUTVA validation questions.

**Attributes:**

- [**QUESTIONS**](#causalis.refutation.sutva.sutva_validation.QUESTIONS) (<code>[Iterable](#typing.Iterable)\[[str](#str)\]</code>) –

###### `causalis.refutation.sutva.sutva_validation.QUESTIONS`

```python
QUESTIONS: Iterable[str] = ('1.) Are your clients independent (i)?', '2.) Do you measure confounders, treatment, and outcome in the same intervals?', '3.) Do you measure confounders before treatment and outcome after?', '4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?')
```

###### `causalis.refutation.sutva.sutva_validation.print_sutva_questions`

```python
print_sutva_questions()
```

Print the SUTVA validation questions.

Just prints questions, nothing more.

#### `causalis.refutation.trim_sensitivity_curve_ate`

```python
trim_sensitivity_curve_ate(m_hat, D, Y, g0_hat, g1_hat, eps_grid=(0.0, 0.005, 0.01, 0.02, 0.05))
```

Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).

For each epsilon in eps_grid, compute the AIPW/IRM ATE estimate using
m_clipped = clip(m_hat, eps, 1-eps) over the full sample and report
the plug-in standard error from the EIF.

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **D** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **Y** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g0_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g1_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **eps_grid** (<code>[tuple](#tuple)\[[float](#float), ...\]</code>) – Sequence of clipping thresholds ε to evaluate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Columns: ['trim_eps','n','pct_clipped','theta','se'].
  pct_clipped is the percent of observations with m outside [ε,1-ε].

#### `causalis.refutation.trim_sensitivity_curve_atte`

```python
trim_sensitivity_curve_atte(inference_fn, data, m, d, thresholds=np.linspace(0.9, 0.995, 12), **inference_kwargs)
```

Re-estimate θ while progressively trimming CONTROLS with large m(X).

#### `causalis.refutation.uncofoundedness`

**Modules:**

- [**sensitivity**](#causalis.refutation.uncofoundedness.sensitivity) – Sensitivity functions refactored into a dedicated module.
- [**uncofoundedness_validation**](#causalis.refutation.uncofoundedness.uncofoundedness_validation) – Uncofoundedness validation module

##### `causalis.refutation.uncofoundedness.sensitivity`

Sensitivity functions refactored into a dedicated module.

This module centralizes bias-aware sensitivity helpers and the public
entry points used by refutation utilities for uncofoundedness.

**Functions:**

- [**get_sensitivity_summary**](#causalis.refutation.uncofoundedness.sensitivity.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**sensitivity_analysis**](#causalis.refutation.uncofoundedness.sensitivity.sensitivity_analysis) – Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.
- [**sensitivity_benchmark**](#causalis.refutation.uncofoundedness.sensitivity.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model

###### `causalis.refutation.uncofoundedness.sensitivity.get_sensitivity_summary`

```python
get_sensitivity_summary(effect_estimation, *, label=None)
```

Render a single, unified bias-aware summary string.
If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

###### `causalis.refutation.uncofoundedness.sensitivity.sensitivity_analysis`

```python
sensitivity_analysis(effect_estimation, *, cf_y, cf_d, rho=1.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.

Returns a dict with:

- theta, se, alpha, z
- sampling_ci
- theta_bounds_cofounding = (theta - max_bias, theta + max_bias)
- bias_aware_ci = \[theta - (max_bias + z*se), theta + (max_bias + z*se)\]
- max_bias and components (sigma2, nu2)
- params (cf_y, cf_d, rho, use_signed_rr)

###### `causalis.refutation.uncofoundedness.sensitivity.sensitivity_benchmark`

```python
sensitivity_benchmark(effect_estimation, benchmarking_set, fit_args=None)
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing cf_y, cf_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Additional keyword arguments for the IRM.fit() method of the short model.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- cf_y, cf_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

##### `causalis.refutation.uncofoundedness.uncofoundedness_validation`

Uncofoundedness validation module

**Functions:**

- [**run_uncofoundedness_diagnostics**](#causalis.refutation.uncofoundedness.uncofoundedness_validation.run_uncofoundedness_diagnostics) – Uncofoundedness diagnostics focused on balance (SMD).
- [**validate_uncofoundedness_balance**](#causalis.refutation.uncofoundedness.uncofoundedness_validation.validate_uncofoundedness_balance) – Assess covariate balance under the uncofoundedness assumption by computing

###### `causalis.refutation.uncofoundedness.uncofoundedness_validation.run_uncofoundedness_diagnostics`

```python
run_uncofoundedness_diagnostics(*, res=None, X=None, d=None, m_hat=None, names=None, score=None, normalize=None, threshold=0.1, eps_overlap=0.01, return_summary=True)
```

Uncofoundedness diagnostics focused on balance (SMD).

Inputs:

- Either a result/model via `res`, or raw arrays X, d, m_hat (+ optional names, score, normalize).

Returns a dictionary:
{
"params": {"score", "normalize", "smd_threshold"},
"balance": {"smd", "smd_unweighted", "smd_max", "frac_violations", "pass", "worst_features"},
"flags": {"balance_max_smd", "balance_violations"},
"overall_flag": max severity across balance flags,
"summary": pd.DataFrame with balance rows only
}

###### `causalis.refutation.uncofoundedness.uncofoundedness_validation.validate_uncofoundedness_balance`

```python
validate_uncofoundedness_balance(effect_estimation, *, threshold=0.1, normalize=None)
```

Assess covariate balance under the uncofoundedness assumption by computing
standardized mean differences (SMD) both before weighting (raw groups) and
after weighting using the IPW / ATT weights implied by the DML/IRM estimation.

This function expects the result dictionary returned by dml_ate() or dml_att(),
which includes a fitted IRM model and a 'diagnostic_data' entry with the
necessary arrays.

We compute, for each confounder X_j:

- For ATE (weighted): w1 = D/m_hat, w0 = (1-D)/(1-m_hat).
- For ATTE (weighted): Treated weight = 1 for D=1; Control weight w0 = m_hat/(1-m_hat) for D=0.
- If estimation used normalized IPW (normalize_ipw=True), we scale the corresponding
  weights by their sample mean (as done in IRM) before computing balance.

The SMD is defined as |mu1 - mu0| / s_pooled, where mu_g are (weighted) means in the
(pseudo-)populations and s_pooled is the square root of the average of the (weighted)
variances in the two groups.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Output dict from dml_ate() or dml_att(). Must contain 'model' and 'diagnostic_data'.
- **threshold** (<code>[float](#float)</code>) – Threshold for SMD; values below indicate acceptable balance for most use cases.
- **normalize** (<code>[Optional](#typing.Optional)\[[bool](#bool)\]</code>) – Whether to use normalized weights. If None, inferred from effect_estimation['diagnostic_data']['normalize_ipw'].

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary with keys:
- 'smd': pd.Series of weighted SMD values indexed by confounder names
- 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
- 'score': 'ATE' or 'ATTE'
- 'normalized': bool used for weighting
- 'threshold': float
- 'pass': bool indicating whether all weighted SMDs are below threshold

#### `causalis.refutation.validate_uncofoundedness_balance`

```python
validate_uncofoundedness_balance(effect_estimation, *, threshold=0.1, normalize=None)
```

Assess covariate balance under the uncofoundedness assumption by computing
standardized mean differences (SMD) both before weighting (raw groups) and
after weighting using the IPW / ATT weights implied by the DML/IRM estimation.

This function expects the result dictionary returned by dml_ate() or dml_att(),
which includes a fitted IRM model and a 'diagnostic_data' entry with the
necessary arrays.

We compute, for each confounder X_j:

- For ATE (weighted): w1 = D/m_hat, w0 = (1-D)/(1-m_hat).
- For ATTE (weighted): Treated weight = 1 for D=1; Control weight w0 = m_hat/(1-m_hat) for D=0.
- If estimation used normalized IPW (normalize_ipw=True), we scale the corresponding
  weights by their sample mean (as done in IRM) before computing balance.

The SMD is defined as |mu1 - mu0| / s_pooled, where mu_g are (weighted) means in the
(pseudo-)populations and s_pooled is the square root of the average of the (weighted)
variances in the two groups.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Output dict from dml_ate() or dml_att(). Must contain 'model' and 'diagnostic_data'.
- **threshold** (<code>[float](#float)</code>) – Threshold for SMD; values below indicate acceptable balance for most use cases.
- **normalize** (<code>[Optional](#typing.Optional)\[[bool](#bool)\]</code>) – Whether to use normalized weights. If None, inferred from effect_estimation['diagnostic_data']['normalize_ipw'].

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary with keys:
- 'smd': pd.Series of weighted SMD values indexed by confounder names
- 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
- 'score': 'ATE' or 'ATTE'
- 'normalized': bool used for weighting
- 'threshold': float
- 'pass': bool indicating whether all weighted SMDs are below threshold

### `causalis.scenarios`

**Modules:**

- [**rct**](#causalis.scenarios.rct) –
- [**unconfoundedness**](#causalis.scenarios.unconfoundedness) –

#### `causalis.scenarios.rct`

**Modules:**

- [**rct_design**](#causalis.scenarios.rct.rct_design) – Design module for experimental rct_design utilities.
- [**srm**](#causalis.scenarios.rct.srm) – Sample Ratio Mismatch (SRM) utilities for randomized experiments.

**Classes:**

- [**SRMResult**](#causalis.scenarios.rct.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**bootstrap_diff_means**](#causalis.scenarios.rct.bootstrap_diff_means) – Bootstrap inference for difference in means between treated (T=1) and control (T=0).
- [**check_srm**](#causalis.scenarios.rct.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.
- [**conversion_z_test**](#causalis.scenarios.rct.conversion_z_test) – Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).
- [**ttest**](#causalis.scenarios.rct.ttest) – Perform a t-test on a CausalData object to compare the outcome variable between

##### `causalis.scenarios.rct.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.scenarios.rct.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.scenarios.rct.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.scenarios.rct.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.scenarios.rct.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.scenarios.rct.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.scenarios.rct.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.scenarios.rct.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.scenarios.rct.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `causalis.scenarios.rct.SRMResult.alpha`

```python
alpha: float
```

###### `causalis.scenarios.rct.SRMResult.chi2`

```python
chi2: float
```

###### `causalis.scenarios.rct.SRMResult.df`

```python
df: int
```

###### `causalis.scenarios.rct.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

###### `causalis.scenarios.rct.SRMResult.is_srm`

```python
is_srm: bool
```

###### `causalis.scenarios.rct.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

###### `causalis.scenarios.rct.SRMResult.p_value`

```python
p_value: float
```

###### `causalis.scenarios.rct.SRMResult.warning`

```python
warning: str | None = None
```

##### `causalis.scenarios.rct.bootstrap_diff_means`

```python
bootstrap_diff_means(data, alpha=0.05, n_simul=10000)
```

Bootstrap inference for difference in means between treated (T=1) and control (T=0).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary with p_value, absolute_difference, absolute_ci, relative_difference, relative_ci
  (matching the structure of inference.atte.ttest).

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

##### `causalis.scenarios.rct.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.data.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.scenarios.rct.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
- Each p must be > 0.
- Sum of all p must be 1.0 (within numerical tolerance).

Examples:
{"control": 0.5, "treatment": 0.5}
{"A": 0.2, "B": 0.3, "C": 0.5}

- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.scenarios.rct.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

##### `causalis.scenarios.rct.conversion_z_test`

```python
conversion_z_test(data, alpha=0.05, ci_method='newcombe', se_for_test='pooled')
```

Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **ci_method** (<code>([newcombe](#newcombe), [wald_unpooled](#wald_unpooled), [wald_pooled](#wald_pooled))</code>) – Method for calculating the confidence interval for the absolute difference.
  "newcombe" is the most robust default for conversion rates.
- **se_for_test** (<code>([pooled](#pooled), [unpooled](#unpooled))</code>) – Method for calculating the standard error for the z-test p-value.
  "pooled" (score test) is generally preferred for testing equality of proportions.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value from the z-test
- absolute_difference: Difference in conversion rates (treated - control)
- absolute_ci: Tuple (lower, upper) for the absolute difference CI
- relative_difference: Percentage change relative to control rate
- relative_ci: Tuple (lower, upper) for the relative difference CI

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment/outcome are missing, treatment is not binary, outcome is not binary,
  groups are empty, or alpha is outside (0, 1).

##### `causalis.scenarios.rct.rct_design`

Design module for experimental rct_design utilities.

**Modules:**

- [**mde**](#causalis.scenarios.rct.rct_design.mde) – Utility functions for calculating Minimum Detectable Effect (MDE) for experimental rct_design.
- [**split**](#causalis.scenarios.rct.rct_design.split) – Split (assignment) utilities for randomized controlled experiments.

**Classes:**

- [**SRMResult**](#causalis.scenarios.rct.rct_design.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**assign_variants_df**](#causalis.scenarios.rct.rct_design.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.
- [**calculate_mde**](#causalis.scenarios.rct.rct_design.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data.
- [**check_srm**](#causalis.scenarios.rct.rct_design.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

###### `causalis.scenarios.rct.rct_design.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.scenarios.rct.rct_design.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.scenarios.rct.rct_design.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.scenarios.rct.rct_design.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.scenarios.rct.rct_design.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.scenarios.rct.rct_design.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.scenarios.rct.rct_design.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.scenarios.rct.rct_design.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.scenarios.rct.rct_design.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

####### `causalis.scenarios.rct.rct_design.SRMResult.alpha`

```python
alpha: float
```

####### `causalis.scenarios.rct.rct_design.SRMResult.chi2`

```python
chi2: float
```

####### `causalis.scenarios.rct.rct_design.SRMResult.df`

```python
df: int
```

####### `causalis.scenarios.rct.rct_design.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

####### `causalis.scenarios.rct.rct_design.SRMResult.is_srm`

```python
is_srm: bool
```

####### `causalis.scenarios.rct.rct_design.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

####### `causalis.scenarios.rct.rct_design.SRMResult.p_value`

```python
p_value: float
```

####### `causalis.scenarios.rct.rct_design.SRMResult.warning`

```python
warning: str | None = None
```

###### `causalis.scenarios.rct.rct_design.assign_variants_df`

```python
assign_variants_df(df, id_col, experiment_id, variants, *, salt='global_ab_salt', layer_id='default', variant_col='variant')
```

Deterministically assign variants for each row in df based on id_col.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – Input DataFrame with an identifier column.
- **id_col** (<code>[str](#str)</code>) – Column name in df containing entity identifiers (user_id, session_id, etc.).
- **experiment_id** (<code>[str](#str)</code>) – Unique identifier for the experiment (versioned for reruns).
- **variants** (<code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code>) – Mapping from variant name to weight (coverage). Weights must be non-negative
  and their sum must be in (0, 1\]. If the sum is < 1, the remaining mass
  corresponds to "not in experiment" and the assignment will be None.
- **salt** (<code>[str](#str)</code>) – Secret string to de-correlate from other hash uses and make assignments
  non-gameable.
- **layer_id** (<code>[str](#str)</code>) – Identifier for mutual exclusivity layer or surface. In this case work like
  another random
- **variant_col** (<code>[str](#str)</code>) – Name of output column to store assigned variant labels.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of df with an extra column `variant_col`.
  Entities outside experiment coverage will have None in the variant column.

###### `causalis.scenarios.rct.rct_design.calculate_mde`

```python
calculate_mde(sample_size, baseline_rate=None, variance=None, alpha=0.05, power=0.8, data_type='conversion', ratio=0.5)
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data) or baseline mean (for continuous data).
  Required for conversion data.
- **variance** (<code>float or tuple of float</code>) – Variance of the data. For conversion data, this is calculated from the baseline rate if not provided.
  For continuous data, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data. Either 'conversion' for binary/conversion data or 'continuous' for continuous data.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

###### `causalis.scenarios.rct.rct_design.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.data.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.scenarios.rct.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
- Each p must be > 0.
- Sum of all p must be 1.0 (within numerical tolerance).

Examples:
{"control": 0.5, "treatment": 0.5}
{"A": 0.2, "B": 0.3, "C": 0.5}

- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.scenarios.rct.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

###### `causalis.scenarios.rct.rct_design.mde`

Utility functions for calculating Minimum Detectable Effect (MDE) for experimental rct_design.

**Functions:**

- [**calculate_mde**](#causalis.scenarios.rct.rct_design.mde.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data.

####### `causalis.scenarios.rct.rct_design.mde.calculate_mde`

```python
calculate_mde(sample_size, baseline_rate=None, variance=None, alpha=0.05, power=0.8, data_type='conversion', ratio=0.5)
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data) or baseline mean (for continuous data).
  Required for conversion data.
- **variance** (<code>float or tuple of float</code>) – Variance of the data. For conversion data, this is calculated from the baseline rate if not provided.
  For continuous data, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data. Either 'conversion' for binary/conversion data or 'continuous' for continuous data.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

###### `causalis.scenarios.rct.rct_design.split`

Split (assignment) utilities for randomized controlled experiments.

This module provides deterministic assignment of variants to entities based
on hashing a composite key (salt | layer_id | experiment_id | entity_id)
into the unit interval and mapping it to cumulative variant weights.

The implementation mirrors the reference notebook in docs/cases/rct_design.ipynb.

**Functions:**

- [**assign_variants_df**](#causalis.scenarios.rct.rct_design.split.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.

####### `causalis.scenarios.rct.rct_design.split.assign_variants_df`

```python
assign_variants_df(df, id_col, experiment_id, variants, *, salt='global_ab_salt', layer_id='default', variant_col='variant')
```

Deterministically assign variants for each row in df based on id_col.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – Input DataFrame with an identifier column.
- **id_col** (<code>[str](#str)</code>) – Column name in df containing entity identifiers (user_id, session_id, etc.).
- **experiment_id** (<code>[str](#str)</code>) – Unique identifier for the experiment (versioned for reruns).
- **variants** (<code>[Dict](#typing.Dict)\[[str](#str), [float](#float)\]</code>) – Mapping from variant name to weight (coverage). Weights must be non-negative
  and their sum must be in (0, 1\]. If the sum is < 1, the remaining mass
  corresponds to "not in experiment" and the assignment will be None.
- **salt** (<code>[str](#str)</code>) – Secret string to de-correlate from other hash uses and make assignments
  non-gameable.
- **layer_id** (<code>[str](#str)</code>) – Identifier for mutual exclusivity layer or surface. In this case work like
  another random
- **variant_col** (<code>[str](#str)</code>) – Name of output column to store assigned variant labels.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of df with an extra column `variant_col`.
  Entities outside experiment coverage will have None in the variant column.

##### `causalis.scenarios.rct.srm`

Sample Ratio Mismatch (SRM) utilities for randomized experiments.

This module implements a chi-square goodness-of-fit SRM check mirroring the
reference implementation demonstrated in docs/cases/rct_design.ipynb.

**Classes:**

- [**SRMResult**](#causalis.scenarios.rct.srm.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**check_srm**](#causalis.scenarios.rct.srm.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

###### `causalis.scenarios.rct.srm.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.scenarios.rct.srm.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.scenarios.rct.srm.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.scenarios.rct.srm.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.scenarios.rct.srm.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.scenarios.rct.srm.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.scenarios.rct.srm.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.scenarios.rct.srm.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.scenarios.rct.srm.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

####### `causalis.scenarios.rct.srm.SRMResult.alpha`

```python
alpha: float
```

####### `causalis.scenarios.rct.srm.SRMResult.chi2`

```python
chi2: float
```

####### `causalis.scenarios.rct.srm.SRMResult.df`

```python
df: int
```

####### `causalis.scenarios.rct.srm.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

####### `causalis.scenarios.rct.srm.SRMResult.is_srm`

```python
is_srm: bool
```

####### `causalis.scenarios.rct.srm.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

####### `causalis.scenarios.rct.srm.SRMResult.p_value`

```python
p_value: float
```

####### `causalis.scenarios.rct.srm.SRMResult.warning`

```python
warning: str | None = None
```

###### `causalis.scenarios.rct.srm.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.data.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.scenarios.rct.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
- Each p must be > 0.
- Sum of all p must be 1.0 (within numerical tolerance).

Examples:
{"control": 0.5, "treatment": 0.5}
{"A": 0.2, "B": 0.3, "C": 0.5}

- **alpha** (<code>[float](#float)</code>) – Significance level. Use strict values like 1e-3 or 1e-4 in production.
- **min_expected** (<code>[float](#float)</code>) – If any expected count < min_expected, a warning is attached.
- **strict_variants** (<code>[bool](#bool)</code>) – - True: fail if observed variants differ from target keys.
- False: drop unknown variants and test only on declared ones.

**Returns:**

- <code>[SRMResult](#causalis.scenarios.rct.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

##### `causalis.scenarios.rct.ttest`

```python
ttest(data, alpha=0.05)
```

Perform a t-test on a CausalData object to compare the outcome variable between
treated (T=1) and control (T=0) groups. Returns differences and confidence intervals.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: The p-value from the t-test
- absolute_difference: The absolute difference between treatment and control means
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference confidence interval
- relative_difference: The relative difference (percentage change) between treatment and control means
- relative_ci: Tuple of (lower, upper) bounds for the relative difference confidence interval

**Raises:**

- <code>[ValueError](#ValueError)</code> – If the CausalData object doesn't have both treatment and outcome variables defined,
  or if the treatment variable is not binary.

#### `causalis.scenarios.unconfoundedness`

**Modules:**

- [**ate**](#causalis.scenarios.unconfoundedness.ate) – Average Treatment Effect (ATE) inference methods for causalis.
- [**atte**](#causalis.scenarios.unconfoundedness.atte) – Average Treatment Effect on the Treated (ATT) inference methods for causalis.
- [**cate**](#causalis.scenarios.unconfoundedness.cate) – Conditional Average Treatment Effect (CATE) inference methods for causalis.
- [**gate**](#causalis.scenarios.unconfoundedness.gate) – Group Average Treatment Effect (GATE) inference methods for causalis.
- [**refutation**](#causalis.scenarios.unconfoundedness.refutation) – Refutation and robustness utilities for Causalis.

##### `causalis.scenarios.unconfoundedness.ate`

Average Treatment Effect (ATE) inference methods for causalis.

This module provides methods for estimating average treatment effects.

**Modules:**

- [**dml_ate**](#causalis.scenarios.unconfoundedness.ate.dml_ate) – IRM implementation for estimating average treatment effects.
- [**dml_ate_source**](#causalis.scenarios.unconfoundedness.ate.dml_ate_source) – DoubleML implementation for estimating average treatment effects.

###### `causalis.scenarios.unconfoundedness.ate.dml_ate`

IRM implementation for estimating average treatment effects.

This module provides a function dml_ate to estimate average treatment effects using
our internal DoubleML-style IRM estimator that consumes CausalData directly.

**Functions:**

- [**dml_ate**](#causalis.scenarios.unconfoundedness.ate.dml_ate.dml_ate) – Estimate average treatment effects using the internal IRM estimator.

####### `causalis.scenarios.unconfoundedness.ate.dml_ate.dml_ate`

```python
dml_ate(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, score='ATE', alpha=0.05, normalize_ipw=False, trimming_rule='truncate', trimming_threshold=0.01, random_state=None, store_diagnostic_data=True)
```

Estimate average treatment effects using the internal IRM estimator.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causaldata object containing treatment, outcome, and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for g(D,X)=E[Y|X,D]. If outcome is binary and learner is classifier,
  predict_proba will be used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for m(X)=E[D|X] (propensity). If None, a CatBoostClassifier is used.
- **n_folds** (<code>[int](#int)</code>) – Number of folds for cross-fitting.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions (currently only 1 supported by IRM).
- **score** (<code>('ATE', 'ATTE')</code>) – Target estimand.
- **alpha** (<code>[float](#float)</code>) – Significance level for CI in (0,1).
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score.
- **trimming_rule** (<code>[str](#str)</code>) – Trimming approach for propensity (only "truncate" supported).
- **trimming_threshold** (<code>[float](#float)</code>) – Trimming threshold for propensity.
- **random_state** (<code>[int](#int)</code>) – Random seed for fold creation.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Keys: coefficient, std_error, p_value, confidence_interval, model

<details class="note" open markdown="1">
<summary>Notes</summary>

By default, this function stores a comprehensive 'diagnostic_data' dictionary in the result.
You can disable this by setting store_diagnostic_data=False.

</details>

###### `causalis.scenarios.unconfoundedness.ate.dml_ate_source`

DoubleML implementation for estimating average treatment effects.

This module provides a function to estimate average treatment effects using DoubleML.

**Functions:**

- [**dml_ate_source**](#causalis.scenarios.unconfoundedness.ate.dml_ate_source.dml_ate_source) – Estimate average treatment effects using DoubleML's interactive regression model (IRM).

####### `causalis.scenarios.unconfoundedness.ate.dml_ate_source.dml_ate_source`

```python
dml_ate_source(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, score='ATE', alpha=0.05)
```

Estimate average treatment effects using DoubleML's interactive regression model (IRM).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causaldata object containing treatment, target, and confounders variables.
- **ml_g** (<code>[estimator](#estimator)</code>) – A machine learner implementing `fit()` and `predict()` methods for the nuisance function g_0(D,X) = E[Y|X,D].
  If None, a CatBoostRegressor configured to use all CPU cores is used.
- **ml_m** (<code>[classifier](#classifier)</code>) – A machine learner implementing `fit()` and `predict_proba()` methods for the nuisance function m_0(X) = E[D|X].
  If None, a CatBoostClassifier configured to use all CPU cores is used.
- **n_folds** (<code>[int](#int)</code>) – Number of folds for cross-fitting.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions for the sample splitting.
- **score** (<code>[str](#str)</code>) – A str ("ATE" or "ATTE") specifying the score function.
- **alpha** (<code>[float](#float)</code>) – Significance level for CI in (0,1).

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- coefficient: The estimated average treatment effect
- std_error: The standard error of the estimate
- p_value: The p-value for the null hypothesis that the effect is zero
- confidence_interval: Tuple of (lower, upper) bounds for the confidence interval
- model: The fitted DoubleMLIRM object

**Raises:**

- <code>[ValueError](#ValueError)</code> – If the causaldata object doesn't have treatment, target, and confounders variables defined,
  or if the treatment variable is not binary.

**Examples:**

```pycon
>>> from causalis.data import generate_rct
>>> from causalis.data import CausalData
>>> from causalis.scenarios.unconfoundedness.ate import dml_ate_source
>>> 
>>> # Generate data
>>> df = generate_rct()
>>> 
>>> # Create causaldata object
>>> causal_data = CausalData(
...     df=df,
...     outcome='outcome',
...     treatment='treatment',
...     confounders=['age', 'invited_friend']
... )
>>> 
>>> # Estimate ATE using DoubleML
>>> results = dml_ate_source(causal_data)
>>> print(f"ATE: {results['coefficient']:.4f}")
>>> print(f"Standard Error: {results['std_error']:.4f}")
>>> print(f"P-value: {results['p_value']:.4f}")
>>> print(f"Confidence Interval: {results['confidence_interval']}")
```

##### `causalis.scenarios.unconfoundedness.atte`

Average Treatment Effect on the Treated (ATT) inference methods for causalis.

This module provides methods for estimating average treatment effects on the treated.

**Modules:**

- [**dml_atte**](#causalis.scenarios.unconfoundedness.atte.dml_atte) – Simple IRM-based implementation for estimating ATT (Average Treatment effect on the Treated).
- [**dml_atte_source**](#causalis.scenarios.unconfoundedness.atte.dml_atte_source) – DoubleML implementation for estimating average treatment effects on the treated.

**Functions:**

- [**bootstrap_diff_means**](#causalis.scenarios.unconfoundedness.atte.bootstrap_diff_means) – Bootstrap inference for difference in means between treated (T=1) and control (T=0).
- [**conversion_z_test**](#causalis.scenarios.unconfoundedness.atte.conversion_z_test) – Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).
- [**ttest**](#causalis.scenarios.unconfoundedness.atte.ttest) – Perform a t-test on a CausalData object to compare the outcome variable between

###### `causalis.scenarios.unconfoundedness.atte.bootstrap_diff_means`

```python
bootstrap_diff_means(data, alpha=0.05, n_simul=10000)
```

Bootstrap inference for difference in means between treated (T=1) and control (T=0).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary with p_value, absolute_difference, absolute_ci, relative_difference, relative_ci
  (matching the structure of inference.atte.ttest).

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

###### `causalis.scenarios.unconfoundedness.atte.conversion_z_test`

```python
conversion_z_test(data, alpha=0.05, ci_method='newcombe', se_for_test='pooled')
```

Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **ci_method** (<code>([newcombe](#newcombe), [wald_unpooled](#wald_unpooled), [wald_pooled](#wald_pooled))</code>) – Method for calculating the confidence interval for the absolute difference.
  "newcombe" is the most robust default for conversion rates.
- **se_for_test** (<code>([pooled](#pooled), [unpooled](#unpooled))</code>) – Method for calculating the standard error for the z-test p-value.
  "pooled" (score test) is generally preferred for testing equality of proportions.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value from the z-test
- absolute_difference: Difference in conversion rates (treated - control)
- absolute_ci: Tuple (lower, upper) for the absolute difference CI
- relative_difference: Percentage change relative to control rate
- relative_ci: Tuple (lower, upper) for the relative difference CI

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment/outcome are missing, treatment is not binary, outcome is not binary,
  groups are empty, or alpha is outside (0, 1).

###### `causalis.scenarios.unconfoundedness.atte.dml_atte`

Simple IRM-based implementation for estimating ATT (Average Treatment effect on the Treated).

This module provides a function dml_att_s to estimate ATT using our internal
DoubleML-style IRM estimator that consumes CausalData directly (not DoubleML).

**Functions:**

- [**dml_atte**](#causalis.scenarios.unconfoundedness.atte.dml_atte.dml_atte) – Estimate average treatment effect on the treated (ATT) using the internal IRM estimator.

####### `causalis.scenarios.unconfoundedness.atte.dml_atte.dml_atte`

```python
dml_atte(data, ml_g=None, ml_m=None, n_folds=4, n_rep=1, alpha=0.05, normalize_ipw=False, trimming_rule='truncate', trimming_threshold=0.01, random_state=None, store_diagnostic_data=True)
```

Estimate average treatment effect on the treated (ATT) using the internal IRM estimator.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment, outcome, and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for g(D,X)=E[Y|X,D]. If outcome is binary and learner is classifier,
  predict_proba will be used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for m(X)=E[D|X] (propensity). If None, a CatBoostClassifier is used.
- **n_folds** (<code>[int](#int)</code>) – Number of folds for cross-fitting.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions (currently only 1 supported by IRM).
- **alpha** (<code>[float](#float)</code>) – Significance level for CI in (0,1).
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score.
- **trimming_rule** (<code>[str](#str)</code>) – Trimming approach for propensity (only "truncate" supported).
- **trimming_threshold** (<code>[float](#float)</code>) – Trimming threshold for propensity.
- **random_state** (<code>[int](#int)</code>) – Random seed for fold creation.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Keys: coefficient, std_error, p_value, confidence_interval, model

<details class="note" open markdown="1">
<summary>Notes</summary>

By default, this function stores a comprehensive 'diagnostic_data' dictionary in the result.
You can disable this by setting store_diagnostic_data=False.

</details>

###### `causalis.scenarios.unconfoundedness.atte.dml_atte_source`

DoubleML implementation for estimating average treatment effects on the treated.

This module provides a function to estimate average treatment effects on the treated using DoubleML.

**Functions:**

- [**dml_atte_source**](#causalis.scenarios.unconfoundedness.atte.dml_atte_source.dml_atte_source) – Estimate average treatment effects on the treated using DoubleML's interactive regression model (IRM).

####### `causalis.scenarios.unconfoundedness.atte.dml_atte_source.dml_atte_source`

```python
dml_atte_source(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, alpha=0.05)
```

Estimate average treatment effects on the treated using DoubleML's interactive regression model (IRM).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causaldata object containing treatment, target, and confounders variables.
- **ml_g** (<code>[estimator](#estimator)</code>) – A machine learner implementing `fit()` and `predict()` methods for the nuisance function g_0(D,X) = E[Y|X,D].
  If None, a CatBoostRegressor configured to use all CPU cores is used.
- **ml_m** (<code>[classifier](#classifier)</code>) – A machine learner implementing `fit()` and `predict_proba()` methods for the nuisance function m_0(X) = E[D|X].
  If None, a CatBoostClassifier configured to use all CPU cores is used.
- **n_folds** (<code>[int](#int)</code>) – Number of folds for cross-fitting.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions for the sample splitting.
- **alpha** (<code>[float](#float)</code>) – Significance level for CI in (0,1).

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- coefficient: The estimated average treatment effect on the treated
- std_error: The standard error of the estimate
- p_value: The p-value for the null hypothesis that the effect is zero
- confidence_interval: Tuple of (lower, upper) bounds for the confidence interval
- model: The fitted DoubleMLIRM object

**Raises:**

- <code>[ValueError](#ValueError)</code> – If the causaldata object doesn't have treatment, target, and confounders variables defined,
  or if the treatment variable is not binary.

**Examples:**

```pycon
>>> from causalis.data import generate_rct
>>> from causalis.data import CausalData
>>> from causalis.scenarios.unconfoundedness.atte import dml_atte_source
>>> 
>>> # Generate data
>>> df = generate_rct()
>>> 
>>> # Create causaldata object
>>> ck = CausalData(
...     df=df,
...     outcome='outcome',
...     treatment='treatment',
...     confounders=['age', 'invited_friend']
... )
>>> 
>>> # Estimate ATT using DoubleML
>>> results = dml_atte_source(ck)
>>> print(f"ATT: {results['coefficient']:.4f}")
>>> print(f"Standard Error: {results['std_error']:.4f}")
>>> print(f"P-value: {results['p_value']:.4f}")
>>> print(f"Confidence Interval: {results['confidence_interval']}")
```

###### `causalis.scenarios.unconfoundedness.atte.ttest`

```python
ttest(data, alpha=0.05)
```

Perform a t-test on a CausalData object to compare the outcome variable between
treated (T=1) and control (T=0) groups. Returns differences and confidence intervals.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: The p-value from the t-test
- absolute_difference: The absolute difference between treatment and control means
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference confidence interval
- relative_difference: The relative difference (percentage change) between treatment and control means
- relative_ci: Tuple of (lower, upper) bounds for the relative difference confidence interval

**Raises:**

- <code>[ValueError](#ValueError)</code> – If the CausalData object doesn't have both treatment and outcome variables defined,
  or if the treatment variable is not binary.

##### `causalis.scenarios.unconfoundedness.cate`

Conditional Average Treatment Effect (CATE) inference methods for causalis.

This submodule provides methods for estimating conditional average treatment effects.

**Modules:**

- [**cate_esimand**](#causalis.scenarios.unconfoundedness.cate.cate_esimand) – DoubleML implementation for estimating CATE (per-observation orthogonal signals).

###### `causalis.scenarios.unconfoundedness.cate.cate_esimand`

DoubleML implementation for estimating CATE (per-observation orthogonal signals).

This module provides a function that, given a CausalData object, fits a DoubleML IRM
model and augments the data with a new column 'cate' that contains the orthogonal
signals (an estimate of the conditional average treatment effect for each unit).

**Functions:**

- [**cate_esimand**](#causalis.scenarios.unconfoundedness.cate.cate_esimand.cate_esimand) – Estimate per-observation CATEs using DoubleML IRM and return a DataFrame with a new 'cate' column.

####### `causalis.scenarios.unconfoundedness.cate.cate_esimand.cate_esimand`

```python
cate_esimand(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, use_blp=False, X_new=None)
```

Estimate per-observation CATEs using DoubleML IRM and return a DataFrame with a new 'cate' column.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – A CausalData object with defined outcome (outcome), treatment (binary 0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – ML learner for outcome regression g(D, X) = E[Y | D, X] supporting fit/predict.
  Defaults to CatBoostRegressor if None.
- **ml_m** (<code>[classifier](#classifier)</code>) – ML learner for propensity m(X) = P[D=1 | X] supporting fit/predict_proba.
  Defaults to CatBoostClassifier if None.
- **n_folds** (<code>[int](#int)</code>) – Number of folds for cross-fitting.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions for sample splitting.
- **use_blp** (<code>[bool](#bool)</code>) – If True, and X_new is provided, returns cate from obj.blp_predict(X_new) aligned to X_new.
  If False (default), uses obj.\_orthogonal_signals (in-sample estimates) and appends to data.
- **X_new** (<code>[DataFrame](#pandas.DataFrame)</code>) – New covariate matrix for out-of-sample CATE prediction via best linear predictor.
  Must contain the same feature columns as the confounders in `data`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – If use_blp is False: returns a copy of data.df with a new column 'cate'.
  If use_blp is True and X_new is provided: returns a DataFrame with 'cate' column for X_new rows.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment is not binary 0/1 or required metadata is missing.

##### `causalis.scenarios.unconfoundedness.gate`

Group Average Treatment Effect (GATE) inference methods for causalis.

This submodule provides methods for estimating group average treatment effects.

**Modules:**

- [**gate_esimand**](#causalis.scenarios.unconfoundedness.gate.gate_esimand) – Group Average Treatment Effect (GATE) estimation using local DML IRM and BLP.

###### `causalis.scenarios.unconfoundedness.gate.gate_esimand`

Group Average Treatment Effect (GATE) estimation using local DML IRM and BLP.

**Functions:**

- [**gate_esimand**](#causalis.scenarios.unconfoundedness.gate.gate_esimand.gate_esimand) – Estimate Group Average Treatment Effects (GATEs).

####### `causalis.scenarios.unconfoundedness.gate.gate_esimand.gate_esimand`

```python
gate_esimand(data, groups=None, n_groups=5, ml_g=None, ml_m=None, n_folds=5, n_rep=1, alpha=0.05)
```

Estimate Group Average Treatment Effects (GATEs).

If `groups` is None, observations are grouped by quantiles of the
plugin CATE proxy (g1_hat - g0_hat).

##### `causalis.scenarios.unconfoundedness.refutation`

Refutation and robustness utilities for Causalis.

Importing this package exposes the public functions from all refutation
submodules (overlap, score, uncofoundedness, sutva) so you can access
commonly used helpers directly via `causalis.refutation`.

**Modules:**

- [**overlap**](#causalis.scenarios.unconfoundedness.refutation.overlap) –
- [**score**](#causalis.scenarios.unconfoundedness.refutation.score) –
- [**sutva**](#causalis.scenarios.unconfoundedness.refutation.sutva) –
- [**uncofoundedness**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness) –

**Classes:**

- [**CausalData**](#causalis.scenarios.unconfoundedness.refutation.CausalData) – Container for causal inference datasets.

**Functions:**

- [**add_score_flags**](#causalis.scenarios.unconfoundedness.refutation.add_score_flags) – Augment run_score_diagnostics(...) dict with:
- [**aipw_score_ate**](#causalis.scenarios.unconfoundedness.refutation.aipw_score_ate) – Efficient influence function (EIF) for ATE.
- [**aipw_score_atte**](#causalis.scenarios.unconfoundedness.refutation.aipw_score_atte) – Efficient influence function (EIF) for ATTE under IRM/AIPW.
- [**att_overlap_tests**](#causalis.scenarios.unconfoundedness.refutation.att_overlap_tests) – Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.
- [**att_weight_sum_identity**](#causalis.scenarios.unconfoundedness.refutation.att_weight_sum_identity) – ATT weight-sum identity check (un-normalized IPW form).
- [**auc_for_m**](#causalis.scenarios.unconfoundedness.refutation.auc_for_m) – ROC AUC using scores m_hat vs labels D.
- [**calibration_report_m**](#causalis.scenarios.unconfoundedness.refutation.calibration_report_m) – Propensity calibration report for cross-fitted propensities m_hat against treatment D.
- [**ece_binary**](#causalis.scenarios.unconfoundedness.refutation.ece_binary) – Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].
- [**edge_mass**](#causalis.scenarios.unconfoundedness.refutation.edge_mass) – Edge mass diagnostics.
- [**ess_per_group**](#causalis.scenarios.unconfoundedness.refutation.ess_per_group) – Effective sample size (ESS) for ATE-style inverse-probability weights per arm.
- [**extract_nuisances**](#causalis.scenarios.unconfoundedness.refutation.extract_nuisances) – Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.
- [**get_sensitivity_summary**](#causalis.scenarios.unconfoundedness.refutation.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**influence_summary**](#causalis.scenarios.unconfoundedness.refutation.influence_summary) – Compute influence diagnostics showing where uncertainty comes from.
- [**ks_distance**](#causalis.scenarios.unconfoundedness.refutation.ks_distance) – Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.
- [**oos_moment_check**](#causalis.scenarios.unconfoundedness.refutation.oos_moment_check) – Out-of-sample moment check to avoid tautological results (legacy/simple version).
- [**oos_moment_check_from_psi**](#causalis.scenarios.unconfoundedness.refutation.oos_moment_check_from_psi) – OOS moment check using cached ψ_a, ψ_b only.
- [**oos_moment_check_with_fold_nuisances**](#causalis.scenarios.unconfoundedness.refutation.oos_moment_check_with_fold_nuisances) – Out-of-sample moment check using fold-specific nuisances to avoid tautological results.
- [**orthogonality_derivatives**](#causalis.scenarios.unconfoundedness.refutation.orthogonality_derivatives) – Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
- [**orthogonality_derivatives_atte**](#causalis.scenarios.unconfoundedness.refutation.orthogonality_derivatives_atte) – Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.
- [**overlap_diagnostics_atte**](#causalis.scenarios.unconfoundedness.refutation.overlap_diagnostics_atte) – Key overlap metrics for ATTE: availability of suitable controls.
- [**overlap_report_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap_report_from_result) – High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.
- [**positivity_overlap_checks**](#causalis.scenarios.unconfoundedness.refutation.positivity_overlap_checks) – Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
- [**print_sutva_questions**](#causalis.scenarios.unconfoundedness.refutation.print_sutva_questions) – Print the SUTVA validation questions.
- [**refute_irm_orthogonality**](#causalis.scenarios.unconfoundedness.refutation.refute_irm_orthogonality) – Comprehensive AIPW orthogonality diagnostics for IRM models.
- [**refute_placebo_outcome**](#causalis.scenarios.unconfoundedness.refutation.refute_placebo_outcome) – Generate random outcome variables while keeping treatment
- [**refute_placebo_treatment**](#causalis.scenarios.unconfoundedness.refutation.refute_placebo_treatment) – Generate random binary treatment variables while keeping outcome and
- [**refute_subset**](#causalis.scenarios.unconfoundedness.refutation.refute_subset) – Re-estimate the effect on a random subset (default 80 %)
- [**run_overlap_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_overlap_diagnostics) – Single entry-point for overlap / positivity / calibration diagnostics.
- [**run_score_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_score_diagnostics) – Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.
- [**run_uncofoundedness_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_uncofoundedness_diagnostics) – Uncofoundedness diagnostics focused on balance (SMD).
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.sensitivity_analysis) – Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.
- [**sensitivity_benchmark**](#causalis.scenarios.unconfoundedness.refutation.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model
- [**trim_sensitivity_curve_ate**](#causalis.scenarios.unconfoundedness.refutation.trim_sensitivity_curve_ate) – Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).
- [**trim_sensitivity_curve_atte**](#causalis.scenarios.unconfoundedness.refutation.trim_sensitivity_curve_atte) – Re-estimate θ while progressively trimming CONTROLS with large m(X).
- [**validate_uncofoundedness_balance**](#causalis.scenarios.unconfoundedness.refutation.validate_uncofoundedness_balance) – Assess covariate balance under the uncofoundedness assumption by computing

**Attributes:**

- [**DEFAULT_THRESHOLDS**](#causalis.scenarios.unconfoundedness.refutation.DEFAULT_THRESHOLDS) –
- [**QUESTIONS**](#causalis.scenarios.unconfoundedness.refutation.QUESTIONS) (<code>[Iterable](#typing.Iterable)\[[str](#str)\]</code>) –
- [**ResultLike**](#causalis.scenarios.unconfoundedness.refutation.ResultLike) –

###### `causalis.scenarios.unconfoundedness.refutation.CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data contract.

**Attributes:**

- [**df**](#causalis.scenarios.unconfoundedness.refutation.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.scenarios.unconfoundedness.refutation.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.scenarios.unconfoundedness.refutation.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.scenarios.unconfoundedness.refutation.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.scenarios.unconfoundedness.refutation.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.scenarios.unconfoundedness.refutation.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.scenarios.unconfoundedness.refutation.CausalData.get_df) – Get a DataFrame with specified columns.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.df`

```python
df: pd.DataFrame
```

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, **kwargs)
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data.causaldata.CausalData)</code> – A validated CausalData instance.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.get_df`

```python
get_df(columns=None, include_treatment=True, include_outcome=True, include_confounders=True, include_user_id=False)
```

Get a DataFrame with specified columns.

**Parameters:**

- **columns** (<code>[List](#typing.List)\[[str](#str)\]</code>) – Specific column names to include.
- **include_treatment** (<code>[bool](#bool)</code>) – Whether to include the treatment column.
- **include_outcome** (<code>[bool](#bool)</code>) – Whether to include the outcome column.
- **include_confounders** (<code>[bool](#bool)</code>) – Whether to include confounder columns.
- **include_user_id** (<code>[bool](#bool)</code>) – Whether to include the user_id column.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A copy of the internal DataFrame with selected columns.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If any specified columns do not exist.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

####### `causalis.scenarios.unconfoundedness.refutation.CausalData.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

###### `causalis.scenarios.unconfoundedness.refutation.DEFAULT_THRESHOLDS`

```python
DEFAULT_THRESHOLDS = dict(edge_mass_warn_001=0.02, edge_mass_strong_001=0.05, edge_mass_warn_002=0.05, edge_mass_strong_002=0.1, ks_warn=0.3, ks_strong=0.4, auc_warn=0.8, auc_strong=0.9, ipw_relerr_warn=0.05, ipw_relerr_strong=0.1, ess_ratio_warn=0.3, ess_ratio_strong=0.15, clip_share_warn=0.02, clip_share_strong=0.05, tail_vs_med_warn=10.0)
```

###### `causalis.scenarios.unconfoundedness.refutation.QUESTIONS`

```python
QUESTIONS: Iterable[str] = ('1.) Are your clients independent (i)?', '2.) Do you measure confounders, treatment, and outcome in the same intervals?', '3.) Do you measure confounders before treatment and outcome after?', '4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?')
```

###### `causalis.scenarios.unconfoundedness.refutation.ResultLike`

```python
ResultLike = Dict[str, Any] | Any
```

###### `causalis.scenarios.unconfoundedness.refutation.add_score_flags`

```python
add_score_flags(rep_score, thresholds=None, *, effect_size_guard=0.02, oos_gate=True, se_rule=None, se_ref=None)
```

Augment run_score_diagnostics(...) dict with:

- rep['flags'] (per-metric flags)
- rep['thresholds'] (the cutoffs used)
- rep['summary'] with a new 'flag' column
- rep['overall_flag'] (rollup)

Additional logic:

- Practical effect-size guard: if the constant-basis derivative magnitude is tiny
  (\<= effect_size_guard), then downgrade an orthogonality RED to GREEN (if OOS is GREEN)
  or to YELLOW (otherwise). Controlled by `oos_gate`.
- Huge-n relaxation: for very large n (>= 200k), relax tail/kurtosis flags slightly
  under specified value gates.

###### `causalis.scenarios.unconfoundedness.refutation.aipw_score_ate`

```python
aipw_score_ate(y, d, g0, g1, m, theta, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATE.
Uses IRM naming: g0,g1 are outcome regressions E[Y|X,D=0/1], m is propensity P(D=1|X).

###### `causalis.scenarios.unconfoundedness.refutation.aipw_score_atte`

```python
aipw_score_atte(y, d, g0, g1, m, theta, p_treated=None, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATTE under IRM/AIPW.

ψ_ATTE(W; θ, η) = \[ D\*(Y - g0(X) - θ) - (1-D)*{ m(X)/(1-m(X)) }*(Y - g0(X)) \] / E[D]

Notes:

- Matches DoubleML's `score='ATTE'` (weights ω=D/E[D], ar{ω}=m(X)/E[D]).
- g1 enters only via θ; ∂ψ/∂g1 = 0.

###### `causalis.scenarios.unconfoundedness.refutation.att_overlap_tests`

```python
att_overlap_tests(dml_att_result, epsilon_list=(0.01, 0.02))
```

Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.

Inputs expected in result\['diagnostic_data'\]:

- m_hat: np.ndarray of cross-fitted propensity scores Pr(D=1|X)
- d: np.ndarray of treatment indicators {0,1}

Returns:
dict with keys:
\- edge_mass: {'eps': {eps: {'share_below': float, 'share_above': float, 'warn': bool}}}
\- ks: {'value': float, 'warn': bool}
\- auc: {'value': float or nan, 'flag': str} # 'GREEN'/'YELLOW'/'RED' or 'NA' if undefined
\- ess: {'treated': {'ess': float, 'n': int, 'ratio': float, 'flag': str},
'control': {'ess': float, 'n': int, 'ratio': float, 'flag': str}}
\- att_weight_identity: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float, 'flag': str}

###### `causalis.scenarios.unconfoundedness.refutation.att_weight_sum_identity`

```python
att_weight_sum_identity(m_hat, D)
```

ATT weight-sum identity check (un-normalized IPW form).

Math:
w1_i = D_i / p1, w0_i = (1 - D_i) * m_hat_i / ((1 - m_hat_i) * p1), where p1 = (1/n) sum_i D_i.
Sum check: sum_i (1 - D_i) * m_hat_i / (1 - m_hat_i) ?≈ sum_i D_i.

Returns: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float}

###### `causalis.scenarios.unconfoundedness.refutation.auc_for_m`

```python
auc_for_m(m_hat, D)
```

ROC AUC using scores m_hat vs labels D.

Math (Mann–Whitney relation):
AUC = P(m_i^+ > m_j^-) + 0.5 P(m_i^+ = m_j^-)

###### `causalis.scenarios.unconfoundedness.refutation.calibration_report_m`

```python
calibration_report_m(m_hat, D, n_bins=10, *, thresholds=None)
```

Propensity calibration report for cross-fitted propensities m_hat against treatment D.

Returns a dictionary with:

- auc: ROC AUC of m_hat vs D (Mann–Whitney)
- brier: Brier score (mean squared error)
- ece: Expected Calibration Error (equal-width bins)
- reliability_table: pd.DataFrame with per-bin stats
- recalibration: {'intercept': alpha, 'slope': beta} from logistic recalibration
- flags: {'ece': ..., 'slope': ..., 'intercept': ...} using GREEN/YELLOW/RED

###### `causalis.scenarios.unconfoundedness.refutation.ece_binary`

```python
ece_binary(p, y, n_bins=10)
```

Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].

**Parameters:**

- **p** (<code>[ndarray](#numpy.ndarray)</code>) – Predicted probabilities in [0,1]. Will be clipped to [0,1].
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Binary labels {0,1}.
- **n_bins** (<code>[int](#int)</code>) – Number of bins.

**Returns:**

- <code>[float](#float)</code> – ECE value in [0,1].

###### `causalis.scenarios.unconfoundedness.refutation.edge_mass`

```python
edge_mass(m_hat, eps=0.01)
```

Edge mass diagnostics.

Math:
share_below = (1/n) * sum_i 1{ m_hat_i < ε }
share_above = (1/n) * sum_i 1{ m_hat_i > 1 - ε }

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Array of propensities m_hat in [0,1].
- **eps** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – A single ε or a sequence of ε values.

**Returns:**

- <code>[dict](#dict)</code> – - If eps is a scalar: {'eps': ε, 'share_below': float, 'share_above': float}
- If eps is a sequence: {ε: {'share_below': float, 'share_above': float}, ...}

###### `causalis.scenarios.unconfoundedness.refutation.ess_per_group`

```python
ess_per_group(m_hat, D)
```

Effective sample size (ESS) for ATE-style inverse-probability weights per arm.

Weights:
w1_i = D_i / m_hat_i,
w0_i = (1 - D_i) / (1 - m_hat_i).

ESS:
ESS(w_g) = (sum_i w\_{gi})^2 / sum_i w\_{gi}^2.

Returns dict with ess and ratios (ESS / group size).

###### `causalis.scenarios.unconfoundedness.refutation.extract_nuisances`

```python
extract_nuisances(model, test_indices=None)
```

Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.

Tries several backends for robustness:

1. IRM attributes: m_hat\_, g0_hat\_, g1_hat\_
1. model.predictions dict with keys: 'ml_m','ml_g0','ml_g1'
1. Direct attributes: ml_m, ml_g0, ml_g1

**Parameters:**

- **model** (<code>[object](#object)</code>) – Fitted internal IRM estimator (causalis.statistics.models.IRM) or a compatible dummy model
- **test_indices** (<code>[ndarray](#numpy.ndarray)</code>) – If provided, extract predictions only for these indices

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]</code> – (m, g0, g1) where:
- m: propensity scores P(D=1|X)
- g0: outcome predictions E[Y|X,D=0]
- g1: outcome predictions E[Y|X,D=1]

###### `causalis.scenarios.unconfoundedness.refutation.get_sensitivity_summary`

```python
get_sensitivity_summary(effect_estimation, *, label=None)
```

Render a single, unified bias-aware summary string.
If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

###### `causalis.scenarios.unconfoundedness.refutation.influence_summary`

```python
influence_summary(y, d, g0, g1, m, theta_hat, k=10, score='ATE', trimming_threshold=0.01)
```

Compute influence diagnostics showing where uncertainty comes from.

**Parameters:**

- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **theta_hat** (<code>[float](#float)</code>) – Estimated treatment effect
- **k** (<code>[int](#int)</code>) – Number of top influential observations to return

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Influence diagnostics including SE, heavy-tail metrics, and top-k cases

###### `causalis.scenarios.unconfoundedness.refutation.ks_distance`

```python
ks_distance(m_hat, D)
```

Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.

Math:
KS = sup_t | F\_{m|D=1}(t) - F\_{m|D=0}(t) |

###### `causalis.scenarios.unconfoundedness.refutation.oos_moment_check`

```python
oos_moment_check(fold_thetas, fold_indices, y, d, g0, g1, m, score_fn=None)
```

Out-of-sample moment check to avoid tautological results (legacy/simple version).

For each fold k, evaluates the AIPW score using θ fitted on other folds,
then tests if the combined moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

###### `causalis.scenarios.unconfoundedness.refutation.oos_moment_check_from_psi`

```python
oos_moment_check_from_psi(psi_a, psi_b, fold_indices, *, strict=False)
```

OOS moment check using cached ψ_a, ψ_b only.
Returns (fold-wise DF, t_fold_agg, t_strict if requested).

###### `causalis.scenarios.unconfoundedness.refutation.oos_moment_check_with_fold_nuisances`

```python
oos_moment_check_with_fold_nuisances(fold_thetas, fold_indices, fold_nuisances, y, d, score_fn=None)
```

Out-of-sample moment check using fold-specific nuisances to avoid tautological results.

For each fold k, evaluates the AIPW score using θ fitted on other folds and
nuisance predictions from the fold-specific model, then tests if the combined
moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **fold_nuisances** (<code>[List](#typing.List)\[[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]\]</code>) – Fold-specific nuisance predictions (m, g0, g1) for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

###### `causalis.scenarios.unconfoundedness.refutation.orthogonality_derivatives`

```python
orthogonality_derivatives(X_basis, y, d, g0, g1, m, trimming_threshold=0.01)
```

Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
Uses IRM naming: g0,g1 outcomes; m propensity.

###### `causalis.scenarios.unconfoundedness.refutation.orthogonality_derivatives_atte`

```python
orthogonality_derivatives_atte(X_basis, y, d, g0, m, p_treated, trimming_threshold=0.01)
```

Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.

For ψ_ATTE = \[ D\*(Y - g0 - θ) - (1-D)*(m/(1-m))*(Y - g0) \] / p_treated:

∂\_{g0}[h] : (1/n) Σ h(X_i) * \[ ((1-D_i)*m_i/(1-m_i) - D_i) / p_treated \]
∂\_{m}[s] : (1/n) Σ s(X_i) * \[ -(1-D_i)*(Y_i - g0_i) / ( p_treated * (1-m_i)^2 ) \]

Both have 0 expectation at the truth (Neyman orthogonality).

###### `causalis.scenarios.unconfoundedness.refutation.overlap`

**Modules:**

- [**overlap_validation**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation) – Overlap validation module

**Functions:**

- [**att_overlap_tests**](#causalis.scenarios.unconfoundedness.refutation.overlap.att_overlap_tests) – Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.
- [**att_weight_sum_identity**](#causalis.scenarios.unconfoundedness.refutation.overlap.att_weight_sum_identity) – ATT weight-sum identity check (un-normalized IPW form).
- [**auc_for_m**](#causalis.scenarios.unconfoundedness.refutation.overlap.auc_for_m) – ROC AUC using scores m_hat vs labels D.
- [**calibration_report_m**](#causalis.scenarios.unconfoundedness.refutation.overlap.calibration_report_m) – Propensity calibration report for cross-fitted propensities m_hat against treatment D.
- [**ece_binary**](#causalis.scenarios.unconfoundedness.refutation.overlap.ece_binary) – Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].
- [**edge_mass**](#causalis.scenarios.unconfoundedness.refutation.overlap.edge_mass) – Edge mass diagnostics.
- [**ess_per_group**](#causalis.scenarios.unconfoundedness.refutation.overlap.ess_per_group) – Effective sample size (ESS) for ATE-style inverse-probability weights per arm.
- [**ks_distance**](#causalis.scenarios.unconfoundedness.refutation.overlap.ks_distance) – Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.
- [**overlap_report_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_report_from_result) – High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.
- [**positivity_overlap_checks**](#causalis.scenarios.unconfoundedness.refutation.overlap.positivity_overlap_checks) – Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
- [**run_overlap_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.overlap.run_overlap_diagnostics) – Single entry-point for overlap / positivity / calibration diagnostics.

**Attributes:**

- [**DEFAULT_THRESHOLDS**](#causalis.scenarios.unconfoundedness.refutation.overlap.DEFAULT_THRESHOLDS) –

####### `causalis.scenarios.unconfoundedness.refutation.overlap.DEFAULT_THRESHOLDS`

```python
DEFAULT_THRESHOLDS = dict(edge_mass_warn_001=0.02, edge_mass_strong_001=0.05, edge_mass_warn_002=0.05, edge_mass_strong_002=0.1, ks_warn=0.3, ks_strong=0.4, auc_warn=0.8, auc_strong=0.9, ipw_relerr_warn=0.05, ipw_relerr_strong=0.1, ess_ratio_warn=0.3, ess_ratio_strong=0.15, clip_share_warn=0.02, clip_share_strong=0.05, tail_vs_med_warn=10.0)
```

####### `causalis.scenarios.unconfoundedness.refutation.overlap.att_overlap_tests`

```python
att_overlap_tests(dml_att_result, epsilon_list=(0.01, 0.02))
```

Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.

Inputs expected in result\['diagnostic_data'\]:

- m_hat: np.ndarray of cross-fitted propensity scores Pr(D=1|X)
- d: np.ndarray of treatment indicators {0,1}

Returns:
dict with keys:
\- edge_mass: {'eps': {eps: {'share_below': float, 'share_above': float, 'warn': bool}}}
\- ks: {'value': float, 'warn': bool}
\- auc: {'value': float or nan, 'flag': str} # 'GREEN'/'YELLOW'/'RED' or 'NA' if undefined
\- ess: {'treated': {'ess': float, 'n': int, 'ratio': float, 'flag': str},
'control': {'ess': float, 'n': int, 'ratio': float, 'flag': str}}
\- att_weight_identity: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float, 'flag': str}

####### `causalis.scenarios.unconfoundedness.refutation.overlap.att_weight_sum_identity`

```python
att_weight_sum_identity(m_hat, D)
```

ATT weight-sum identity check (un-normalized IPW form).

Math:
w1_i = D_i / p1, w0_i = (1 - D_i) * m_hat_i / ((1 - m_hat_i) * p1), where p1 = (1/n) sum_i D_i.
Sum check: sum_i (1 - D_i) * m_hat_i / (1 - m_hat_i) ?≈ sum_i D_i.

Returns: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float}

####### `causalis.scenarios.unconfoundedness.refutation.overlap.auc_for_m`

```python
auc_for_m(m_hat, D)
```

ROC AUC using scores m_hat vs labels D.

Math (Mann–Whitney relation):
AUC = P(m_i^+ > m_j^-) + 0.5 P(m_i^+ = m_j^-)

####### `causalis.scenarios.unconfoundedness.refutation.overlap.calibration_report_m`

```python
calibration_report_m(m_hat, D, n_bins=10, *, thresholds=None)
```

Propensity calibration report for cross-fitted propensities m_hat against treatment D.

Returns a dictionary with:

- auc: ROC AUC of m_hat vs D (Mann–Whitney)
- brier: Brier score (mean squared error)
- ece: Expected Calibration Error (equal-width bins)
- reliability_table: pd.DataFrame with per-bin stats
- recalibration: {'intercept': alpha, 'slope': beta} from logistic recalibration
- flags: {'ece': ..., 'slope': ..., 'intercept': ...} using GREEN/YELLOW/RED

####### `causalis.scenarios.unconfoundedness.refutation.overlap.ece_binary`

```python
ece_binary(p, y, n_bins=10)
```

Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].

**Parameters:**

- **p** (<code>[ndarray](#numpy.ndarray)</code>) – Predicted probabilities in [0,1]. Will be clipped to [0,1].
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Binary labels {0,1}.
- **n_bins** (<code>[int](#int)</code>) – Number of bins.

**Returns:**

- <code>[float](#float)</code> – ECE value in [0,1].

####### `causalis.scenarios.unconfoundedness.refutation.overlap.edge_mass`

```python
edge_mass(m_hat, eps=0.01)
```

Edge mass diagnostics.

Math:
share_below = (1/n) * sum_i 1{ m_hat_i < ε }
share_above = (1/n) * sum_i 1{ m_hat_i > 1 - ε }

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Array of propensities m_hat in [0,1].
- **eps** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – A single ε or a sequence of ε values.

**Returns:**

- <code>[dict](#dict)</code> – - If eps is a scalar: {'eps': ε, 'share_below': float, 'share_above': float}
- If eps is a sequence: {ε: {'share_below': float, 'share_above': float}, ...}

####### `causalis.scenarios.unconfoundedness.refutation.overlap.ess_per_group`

```python
ess_per_group(m_hat, D)
```

Effective sample size (ESS) for ATE-style inverse-probability weights per arm.

Weights:
w1_i = D_i / m_hat_i,
w0_i = (1 - D_i) / (1 - m_hat_i).

ESS:
ESS(w_g) = (sum_i w\_{gi})^2 / sum_i w\_{gi}^2.

Returns dict with ess and ratios (ESS / group size).

####### `causalis.scenarios.unconfoundedness.refutation.overlap.ks_distance`

```python
ks_distance(m_hat, D)
```

Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.

Math:
KS = sup_t | F\_{m|D=1}(t) - F\_{m|D=0}(t) |

####### `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_report_from_result`

```python
overlap_report_from_result(res, *, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
dml_ate(normalize_ipw=True) get meaningful ipw_sum\_\* checks without extra arguments.

####### `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation`

Overlap validation module

**Functions:**

- [**att_overlap_tests**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.att_overlap_tests) – Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.
- [**att_weight_sum_identity**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.att_weight_sum_identity) – ATT weight-sum identity check (un-normalized IPW form).
- [**auc_for_m**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.auc_for_m) – ROC AUC using scores m_hat vs labels D.
- [**calibration_report_m**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.calibration_report_m) – Propensity calibration report for cross-fitted propensities m_hat against treatment D.
- [**ece_binary**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ece_binary) – Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].
- [**edge_mass**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.edge_mass) – Edge mass diagnostics.
- [**ess_per_group**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ess_per_group) – Effective sample size (ESS) for ATE-style inverse-probability weights per arm.
- [**extract_diag_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.extract_diag_from_result) – Extract m_hat, D, and trimming epsilon from dml_ate/dml_att result dict or model.
- [**ks_distance**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ks_distance) – Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.
- [**overlap_report_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.overlap_report_from_result) – High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.
- [**positivity_overlap_checks**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.positivity_overlap_checks) – Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
- [**run_overlap_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.run_overlap_diagnostics) – Single entry-point for overlap / positivity / calibration diagnostics.

**Attributes:**

- [**CAL_THRESHOLDS**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.CAL_THRESHOLDS) –
- [**DEFAULT_THRESHOLDS**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.DEFAULT_THRESHOLDS) –
- [**ResultLike**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ResultLike) –

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.CAL_THRESHOLDS`

```python
CAL_THRESHOLDS = dict(ece_warn=0.1, ece_strong=0.2, slope_warn_lo=0.8, slope_warn_hi=1.2, slope_strong_lo=0.6, slope_strong_hi=1.4, intercept_warn=0.2, intercept_strong=0.4)
```

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.DEFAULT_THRESHOLDS`

```python
DEFAULT_THRESHOLDS = dict(edge_mass_warn_001=0.02, edge_mass_strong_001=0.05, edge_mass_warn_002=0.05, edge_mass_strong_002=0.1, ks_warn=0.3, ks_strong=0.4, auc_warn=0.8, auc_strong=0.9, ipw_relerr_warn=0.05, ipw_relerr_strong=0.1, ess_ratio_warn=0.3, ess_ratio_strong=0.15, clip_share_warn=0.02, clip_share_strong=0.05, tail_vs_med_warn=10.0)
```

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ResultLike`

```python
ResultLike = Union[Dict[str, Any], Any]
```

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.att_overlap_tests`

```python
att_overlap_tests(dml_att_result, epsilon_list=(0.01, 0.02))
```

Compute ATT overlap/weight diagnostics from a dml_att(\_source) result dict.

Inputs expected in result\['diagnostic_data'\]:

- m_hat: np.ndarray of cross-fitted propensity scores Pr(D=1|X)
- d: np.ndarray of treatment indicators {0,1}

Returns:
dict with keys:
\- edge_mass: {'eps': {eps: {'share_below': float, 'share_above': float, 'warn': bool}}}
\- ks: {'value': float, 'warn': bool}
\- auc: {'value': float or nan, 'flag': str} # 'GREEN'/'YELLOW'/'RED' or 'NA' if undefined
\- ess: {'treated': {'ess': float, 'n': int, 'ratio': float, 'flag': str},
'control': {'ess': float, 'n': int, 'ratio': float, 'flag': str}}
\- att_weight_identity: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float, 'flag': str}

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.att_weight_sum_identity`

```python
att_weight_sum_identity(m_hat, D)
```

ATT weight-sum identity check (un-normalized IPW form).

Math:
w1_i = D_i / p1, w0_i = (1 - D_i) * m_hat_i / ((1 - m_hat_i) * p1), where p1 = (1/n) sum_i D_i.
Sum check: sum_i (1 - D_i) * m_hat_i / (1 - m_hat_i) ?≈ sum_i D_i.

Returns: {'lhs_sum': float, 'rhs_sum': float, 'rel_err': float}

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.auc_for_m`

```python
auc_for_m(m_hat, D)
```

ROC AUC using scores m_hat vs labels D.

Math (Mann–Whitney relation):
AUC = P(m_i^+ > m_j^-) + 0.5 P(m_i^+ = m_j^-)

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.calibration_report_m`

```python
calibration_report_m(m_hat, D, n_bins=10, *, thresholds=None)
```

Propensity calibration report for cross-fitted propensities m_hat against treatment D.

Returns a dictionary with:

- auc: ROC AUC of m_hat vs D (Mann–Whitney)
- brier: Brier score (mean squared error)
- ece: Expected Calibration Error (equal-width bins)
- reliability_table: pd.DataFrame with per-bin stats
- recalibration: {'intercept': alpha, 'slope': beta} from logistic recalibration
- flags: {'ece': ..., 'slope': ..., 'intercept': ...} using GREEN/YELLOW/RED

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ece_binary`

```python
ece_binary(p, y, n_bins=10)
```

Expected Calibration Error (ECE) for binary labels using equal-width bins on [0,1].

**Parameters:**

- **p** (<code>[ndarray](#numpy.ndarray)</code>) – Predicted probabilities in [0,1]. Will be clipped to [0,1].
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Binary labels {0,1}.
- **n_bins** (<code>[int](#int)</code>) – Number of bins.

**Returns:**

- <code>[float](#float)</code> – ECE value in [0,1].

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.edge_mass`

```python
edge_mass(m_hat, eps=0.01)
```

Edge mass diagnostics.

Math:
share_below = (1/n) * sum_i 1{ m_hat_i < ε }
share_above = (1/n) * sum_i 1{ m_hat_i > 1 - ε }

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Array of propensities m_hat in [0,1].
- **eps** (<code>[float](#float) or [array](#array) - [like](#like)</code>) – A single ε or a sequence of ε values.

**Returns:**

- <code>[dict](#dict)</code> – - If eps is a scalar: {'eps': ε, 'share_below': float, 'share_above': float}
- If eps is a sequence: {ε: {'share_below': float, 'share_above': float}, ...}

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ess_per_group`

```python
ess_per_group(m_hat, D)
```

Effective sample size (ESS) for ATE-style inverse-probability weights per arm.

Weights:
w1_i = D_i / m_hat_i,
w0_i = (1 - D_i) / (1 - m_hat_i).

ESS:
ESS(w_g) = (sum_i w\_{gi})^2 / sum_i w\_{gi}^2.

Returns dict with ess and ratios (ESS / group size).

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.extract_diag_from_result`

```python
extract_diag_from_result(res)
```

Extract m_hat, D, and trimming epsilon from dml_ate/dml_att result dict or model.
Accepts:

- dict returned by dml_ate/dml_att (prefers key 'diagnostic_data'; otherwise uses 'model'), or
- a fitted IRM/DoubleMLIRM-like model instance with a .data attribute.
  Returns (m_hat, D, trimming_threshold_if_any).

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ks_distance`

```python
ks_distance(m_hat, D)
```

Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.

Math:
KS = sup_t | F\_{m|D=1}(t) - F\_{m|D=0}(t) |

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.overlap_report_from_result`

```python
overlap_report_from_result(res, *, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
dml_ate(normalize_ipw=True) get meaningful ipw_sum\_\* checks without extra arguments.

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.positivity_overlap_checks`

```python
positivity_overlap_checks(m_hat, D, *, m_clipped_from=None, g_clipped_share=None, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
Inputs are cross-fitted m̂ and treatment D (0/1). Returns a structured report with GREEN/YELLOW/RED flags.

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.run_overlap_diagnostics`

```python
run_overlap_diagnostics(res=None, *, m_hat=None, D=None, thresholds=DEFAULT_THRESHOLDS, n_bins=10, use_hajek=None, m_clipped_from=None, g_clipped_share=None, return_summary=True, cal_thresholds=None, auc_flip_margin=0.05)
```

Single entry-point for overlap / positivity / calibration diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_overlap_diagnostics(m_hat=..., D=...)
B) With a model/result:
run_overlap_diagnostics(res=\<dml_ate/dml_att result dict or IRM/DoubleML-like model>)

The function:

- Auto-extracts (m_hat, D, trimming_threshold) from `res` if provided.
- Auto-detects Hájek normalization if available on `res` (normalize_ipw).
- Runs positivity/overlap checks (edge mass, KS, AUC, ESS, tails, ATT identity),
  clipping audit, and calibration (ECE + logistic recalibration).
- Returns a dict with full details and, optionally, a compact summary DataFrame.

**Returns:**

- <code>[dict](#dict)</code> – A dictionary with keys including:
  - n, n_treated, p1
  - edge_mass, edge_mass_by_arm, ks, auc
  - ate_ipw, ate_ess, ate_tails
  - att_weights, att_ess
  - clipping
  - calibration (with reliability_table)
  - flags (GREEN/YELLOW/RED/NA)
  - summary (pd.DataFrame) if return_summary=True
  - meta (use_hajek, thresholds)

####### `causalis.scenarios.unconfoundedness.refutation.overlap.positivity_overlap_checks`

```python
positivity_overlap_checks(m_hat, D, *, m_clipped_from=None, g_clipped_share=None, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
Inputs are cross-fitted m̂ and treatment D (0/1). Returns a structured report with GREEN/YELLOW/RED flags.

####### `causalis.scenarios.unconfoundedness.refutation.overlap.run_overlap_diagnostics`

```python
run_overlap_diagnostics(res=None, *, m_hat=None, D=None, thresholds=DEFAULT_THRESHOLDS, n_bins=10, use_hajek=None, m_clipped_from=None, g_clipped_share=None, return_summary=True, cal_thresholds=None, auc_flip_margin=0.05)
```

Single entry-point for overlap / positivity / calibration diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_overlap_diagnostics(m_hat=..., D=...)
B) With a model/result:
run_overlap_diagnostics(res=\<dml_ate/dml_att result dict or IRM/DoubleML-like model>)

The function:

- Auto-extracts (m_hat, D, trimming_threshold) from `res` if provided.
- Auto-detects Hájek normalization if available on `res` (normalize_ipw).
- Runs positivity/overlap checks (edge mass, KS, AUC, ESS, tails, ATT identity),
  clipping audit, and calibration (ECE + logistic recalibration).
- Returns a dict with full details and, optionally, a compact summary DataFrame.

**Returns:**

- <code>[dict](#dict)</code> – A dictionary with keys including:
  - n, n_treated, p1
  - edge_mass, edge_mass_by_arm, ks, auc
  - ate_ipw, ate_ess, ate_tails
  - att_weights, att_ess
  - clipping
  - calibration (with reliability_table)
  - flags (GREEN/YELLOW/RED/NA)
  - summary (pd.DataFrame) if return_summary=True
  - meta (use_hajek, thresholds)

###### `causalis.scenarios.unconfoundedness.refutation.overlap_diagnostics_atte`

```python
overlap_diagnostics_atte(m, d, eps_list=[0.95, 0.97, 0.98, 0.99])
```

Key overlap metrics for ATTE: availability of suitable controls.
Reports conditional shares: among CONTROLS, fraction with m(X) ≥ threshold; among TREATED, fraction with m(X) ≤ 1 - threshold.

###### `causalis.scenarios.unconfoundedness.refutation.overlap_report_from_result`

```python
overlap_report_from_result(res, *, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

High-level helper that takes dml_ate/dml_att output (or IRM model) and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
dml_ate(normalize_ipw=True) get meaningful ipw_sum\_\* checks without extra arguments.

###### `causalis.scenarios.unconfoundedness.refutation.positivity_overlap_checks`

```python
positivity_overlap_checks(m_hat, D, *, m_clipped_from=None, g_clipped_share=None, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
Inputs are cross-fitted m̂ and treatment D (0/1). Returns a structured report with GREEN/YELLOW/RED flags.

###### `causalis.scenarios.unconfoundedness.refutation.print_sutva_questions`

```python
print_sutva_questions()
```

Print the SUTVA validation questions.

Just prints questions, nothing more.

###### `causalis.scenarios.unconfoundedness.refutation.refute_irm_orthogonality`

```python
refute_irm_orthogonality(inference_fn, data, trim_propensity=(0.02, 0.98), n_basis_funcs=None, n_folds_oos=4, score=None, trimming_threshold=0.01, strict_oos=True, **inference_kwargs)
```

Comprehensive AIPW orthogonality diagnostics for IRM models.

Implements three key diagnostic approaches based on the efficient influence function (EIF):

1. Out-of-sample moment check (non-tautological)
1. Orthogonality (Gateaux derivative) tests
1. Influence diagnostics

**Parameters:**

- **inference_fn** (<code>[Callable](#typing.Callable)</code>) – The inference function (dml_ate or dml_att)
- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causal data object
- **trim_propensity** (<code>[Tuple](#typing.Tuple)\[[float](#float), [float](#float)\]</code>) – Propensity score trimming bounds (min, max) to avoid extreme weights
- **n_basis_funcs** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Number of basis functions for orthogonality derivative tests (constant + covariates).
  If None, defaults to the number of confounders in `data` plus 1 for the constant term.
- **n_folds_oos** (<code>[int](#int)</code>) – Number of folds for out-of-sample moment check
- \*\***inference_kwargs** (<code>[dict](#dict)</code>) – Additional arguments passed to inference_fn

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary containing:
- oos_moment_test: Out-of-sample moment condition results
- orthogonality_derivatives: Gateaux derivative test results
- influence_diagnostics: Influence function diagnostics
- theta: Original treatment effect estimate
- trimmed_diagnostics: Results on trimmed sample
- overall_assessment: Summary diagnostic assessment

**Examples:**

```pycon
>>> from causalis.refutation.orthogonality import refute_irm_orthogonality
>>> from causalis.scenarios.unconfoundedness.ate.dml_ate import dml_ate
>>> 
>>> # Comprehensive orthogonality check
>>> ortho_results = refute_irm_orthogonality(dml_ate, causal_data)
>>> 
>>> # Check key diagnostics
>>> print(f"OOS moment t-stat: {ortho_results['oos_moment_test']['tstat']:.3f}")
>>> print(f"Assessment: {ortho_results['overall_assessment']}")
```

###### `causalis.scenarios.unconfoundedness.refutation.refute_placebo_outcome`

```python
refute_placebo_outcome(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random outcome variables while keeping treatment
and covariates intact. For binary outcomes, generates random binary
variables with the same proportion. For continuous outcomes, generates
random variables from a normal distribution fitted to the original data.
A valid causal design should now yield θ ≈ 0 and a large p-value.

###### `causalis.scenarios.unconfoundedness.refutation.refute_placebo_treatment`

```python
refute_placebo_treatment(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random binary treatment variables while keeping outcome and
covariates intact. Generates random binary treatment with the same
proportion as the original treatment. Breaks the treatment–outcome link.

###### `causalis.scenarios.unconfoundedness.refutation.refute_subset`

```python
refute_subset(inference_fn, data, fraction=0.8, random_state=None, **inference_kwargs)
```

Re-estimate the effect on a random subset (default 80 %)
to check sample-stability of the estimate.

###### `causalis.scenarios.unconfoundedness.refutation.run_overlap_diagnostics`

```python
run_overlap_diagnostics(res=None, *, m_hat=None, D=None, thresholds=DEFAULT_THRESHOLDS, n_bins=10, use_hajek=None, m_clipped_from=None, g_clipped_share=None, return_summary=True, cal_thresholds=None, auc_flip_margin=0.05)
```

Single entry-point for overlap / positivity / calibration diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_overlap_diagnostics(m_hat=..., D=...)
B) With a model/result:
run_overlap_diagnostics(res=\<dml_ate/dml_att result dict or IRM/DoubleML-like model>)

The function:

- Auto-extracts (m_hat, D, trimming_threshold) from `res` if provided.
- Auto-detects Hájek normalization if available on `res` (normalize_ipw).
- Runs positivity/overlap checks (edge mass, KS, AUC, ESS, tails, ATT identity),
  clipping audit, and calibration (ECE + logistic recalibration).
- Returns a dict with full details and, optionally, a compact summary DataFrame.

**Returns:**

- <code>[dict](#dict)</code> – A dictionary with keys including:
  - n, n_treated, p1
  - edge_mass, edge_mass_by_arm, ks, auc
  - ate_ipw, ate_ess, ate_tails
  - att_weights, att_ess
  - clipping
  - calibration (with reliability_table)
  - flags (GREEN/YELLOW/RED/NA)
  - summary (pd.DataFrame) if return_summary=True
  - meta (use_hajek, thresholds)

###### `causalis.scenarios.unconfoundedness.refutation.run_score_diagnostics`

```python
run_score_diagnostics(res=None, *, y=None, d=None, g0=None, g1=None, m=None, theta=None, score=None, trimming_threshold=0.01, n_basis_funcs=None, return_summary=True)
```

Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_score_diagnostics(y=..., d=..., g0=..., g1=..., m=..., theta=...)
B) With a model/result:
run_score_diagnostics(res=\<dml_ate/dml_att result dict or IRM-like model>)

Returns a dictionary with:

- params (score, trimming_threshold)
- oos_moment_test (if fast-path caches available on model; else omitted)
- orthogonality_derivatives (DataFrame)
- influence_diagnostics (full_sample)
- summary (compact DataFrame) if return_summary=True
- meta

###### `causalis.scenarios.unconfoundedness.refutation.run_uncofoundedness_diagnostics`

```python
run_uncofoundedness_diagnostics(*, res=None, X=None, d=None, m_hat=None, names=None, score=None, normalize=None, threshold=0.1, eps_overlap=0.01, return_summary=True)
```

Uncofoundedness diagnostics focused on balance (SMD).

Inputs:

- Either a result/model via `res`, or raw arrays X, d, m_hat (+ optional names, score, normalize).

Returns a dictionary:
{
"params": {"score", "normalize", "smd_threshold"},
"balance": {"smd", "smd_unweighted", "smd_max", "frac_violations", "pass", "worst_features"},
"flags": {"balance_max_smd", "balance_violations"},
"overall_flag": max severity across balance flags,
"summary": pd.DataFrame with balance rows only
}

###### `causalis.scenarios.unconfoundedness.refutation.score`

**Modules:**

- [**score_validation**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation) – AIPW orthogonality diagnostics for IRM-based models.

####### `causalis.scenarios.unconfoundedness.refutation.score.score_validation`

AIPW orthogonality diagnostics for IRM-based models.

This module implements comprehensive orthogonality diagnostics for AIPW/IRM-based
models like dml_ate and dml_att to validate the key assumptions required
for valid causal inference. Based on the efficient influence function (EIF) framework.

Key diagnostics implemented:

- Out-of-sample moment check (non-tautological)
- Orthogonality (Gateaux derivative) tests
- Influence diagnostics

**Functions:**

- [**add_score_flags**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.add_score_flags) – Augment run_score_diagnostics(...) dict with:
- [**aipw_score_ate**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.aipw_score_ate) – Efficient influence function (EIF) for ATE.
- [**aipw_score_atte**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.aipw_score_atte) – Efficient influence function (EIF) for ATTE under IRM/AIPW.
- [**extract_nuisances**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.extract_nuisances) – Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.
- [**influence_summary**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.influence_summary) – Compute influence diagnostics showing where uncertainty comes from.
- [**oos_moment_check**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.oos_moment_check) – Out-of-sample moment check to avoid tautological results (legacy/simple version).
- [**oos_moment_check_from_psi**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.oos_moment_check_from_psi) – OOS moment check using cached ψ_a, ψ_b only.
- [**oos_moment_check_with_fold_nuisances**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.oos_moment_check_with_fold_nuisances) – Out-of-sample moment check using fold-specific nuisances to avoid tautological results.
- [**orthogonality_derivatives**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.orthogonality_derivatives) – Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
- [**orthogonality_derivatives_atte**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.orthogonality_derivatives_atte) – Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.
- [**overlap_diagnostics_atte**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.overlap_diagnostics_atte) – Key overlap metrics for ATTE: availability of suitable controls.
- [**refute_irm_orthogonality**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_irm_orthogonality) – Comprehensive AIPW orthogonality diagnostics for IRM models.
- [**refute_placebo_outcome**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_placebo_outcome) – Generate random outcome variables while keeping treatment
- [**refute_placebo_treatment**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_placebo_treatment) – Generate random binary treatment variables while keeping outcome and
- [**refute_subset**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_subset) – Re-estimate the effect on a random subset (default 80 %)
- [**run_score_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.run_score_diagnostics) – Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.
- [**trim_sensitivity_curve_ate**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.trim_sensitivity_curve_ate) – Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).
- [**trim_sensitivity_curve_atte**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.trim_sensitivity_curve_atte) – Re-estimate θ while progressively trimming CONTROLS with large m(X).

**Attributes:**

- [**ResultLike**](#causalis.scenarios.unconfoundedness.refutation.score.score_validation.ResultLike) –

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.ResultLike`

```python
ResultLike = Dict[str, Any] | Any
```

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.add_score_flags`

```python
add_score_flags(rep_score, thresholds=None, *, effect_size_guard=0.02, oos_gate=True, se_rule=None, se_ref=None)
```

Augment run_score_diagnostics(...) dict with:

- rep['flags'] (per-metric flags)
- rep['thresholds'] (the cutoffs used)
- rep['summary'] with a new 'flag' column
- rep['overall_flag'] (rollup)

Additional logic:

- Practical effect-size guard: if the constant-basis derivative magnitude is tiny
  (\<= effect_size_guard), then downgrade an orthogonality RED to GREEN (if OOS is GREEN)
  or to YELLOW (otherwise). Controlled by `oos_gate`.
- Huge-n relaxation: for very large n (>= 200k), relax tail/kurtosis flags slightly
  under specified value gates.

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.aipw_score_ate`

```python
aipw_score_ate(y, d, g0, g1, m, theta, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATE.
Uses IRM naming: g0,g1 are outcome regressions E[Y|X,D=0/1], m is propensity P(D=1|X).

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.aipw_score_atte`

```python
aipw_score_atte(y, d, g0, g1, m, theta, p_treated=None, trimming_threshold=0.01)
```

Efficient influence function (EIF) for ATTE under IRM/AIPW.

ψ_ATTE(W; θ, η) = \[ D\*(Y - g0(X) - θ) - (1-D)*{ m(X)/(1-m(X)) }*(Y - g0(X)) \] / E[D]

Notes:

- Matches DoubleML's `score='ATTE'` (weights ω=D/E[D], ar{ω}=m(X)/E[D]).
- g1 enters only via θ; ∂ψ/∂g1 = 0.

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.extract_nuisances`

```python
extract_nuisances(model, test_indices=None)
```

Extract cross-fitted nuisance predictions from an IRM-like model or a compatible dummy.

Tries several backends for robustness:

1. IRM attributes: m_hat\_, g0_hat\_, g1_hat\_
1. model.predictions dict with keys: 'ml_m','ml_g0','ml_g1'
1. Direct attributes: ml_m, ml_g0, ml_g1

**Parameters:**

- **model** (<code>[object](#object)</code>) – Fitted internal IRM estimator (causalis.statistics.models.IRM) or a compatible dummy model
- **test_indices** (<code>[ndarray](#numpy.ndarray)</code>) – If provided, extract predictions only for these indices

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]</code> – (m, g0, g1) where:
- m: propensity scores P(D=1|X)
- g0: outcome predictions E[Y|X,D=0]
- g1: outcome predictions E[Y|X,D=1]

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.influence_summary`

```python
influence_summary(y, d, g0, g1, m, theta_hat, k=10, score='ATE', trimming_threshold=0.01)
```

Compute influence diagnostics showing where uncertainty comes from.

**Parameters:**

- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays
- **theta_hat** (<code>[float](#float)</code>) – Estimated treatment effect
- **k** (<code>[int](#int)</code>) – Number of top influential observations to return

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Influence diagnostics including SE, heavy-tail metrics, and top-k cases

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.oos_moment_check`

```python
oos_moment_check(fold_thetas, fold_indices, y, d, g0, g1, m, score_fn=None)
```

Out-of-sample moment check to avoid tautological results (legacy/simple version).

For each fold k, evaluates the AIPW score using θ fitted on other folds,
then tests if the combined moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g0** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **g1** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)
- **m** (<code>[ndarray](#numpy.ndarray)</code>) – Data arrays (outcomes, treatment, predictions)

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.oos_moment_check_from_psi`

```python
oos_moment_check_from_psi(psi_a, psi_b, fold_indices, *, strict=False)
```

OOS moment check using cached ψ_a, ψ_b only.
Returns (fold-wise DF, t_fold_agg, t_strict if requested).

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.oos_moment_check_with_fold_nuisances`

```python
oos_moment_check_with_fold_nuisances(fold_thetas, fold_indices, fold_nuisances, y, d, score_fn=None)
```

Out-of-sample moment check using fold-specific nuisances to avoid tautological results.

For each fold k, evaluates the AIPW score using θ fitted on other folds and
nuisance predictions from the fold-specific model, then tests if the combined
moment condition holds.

**Parameters:**

- **fold_thetas** (<code>[List](#typing.List)\[[float](#float)\]</code>) – Treatment effects estimated excluding each fold
- **fold_indices** (<code>[List](#typing.List)\[[ndarray](#numpy.ndarray)\]</code>) – Indices for each fold
- **fold_nuisances** (<code>[List](#typing.List)\[[Tuple](#typing.Tuple)\[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)\]\]</code>) – Fold-specific nuisance predictions (m, g0, g1) for each fold
- **y** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments
- **d** (<code>[ndarray](#numpy.ndarray)</code>) – Observed outcomes and treatments

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[DataFrame](#pandas.DataFrame), [float](#float)\]</code> – Fold-wise results and combined t-statistic

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.orthogonality_derivatives`

```python
orthogonality_derivatives(X_basis, y, d, g0, g1, m, trimming_threshold=0.01)
```

Compute orthogonality (Gateaux derivative) tests for nuisance functions (ATE case).
Uses IRM naming: g0,g1 outcomes; m propensity.

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.orthogonality_derivatives_atte`

```python
orthogonality_derivatives_atte(X_basis, y, d, g0, m, p_treated, trimming_threshold=0.01)
```

Gateaux derivatives of the ATTE score wrt nuisances (g0, m). g1-derivative is 0.

For ψ_ATTE = \[ D\*(Y - g0 - θ) - (1-D)*(m/(1-m))*(Y - g0) \] / p_treated:

∂\_{g0}[h] : (1/n) Σ h(X_i) * \[ ((1-D_i)*m_i/(1-m_i) - D_i) / p_treated \]
∂\_{m}[s] : (1/n) Σ s(X_i) * \[ -(1-D_i)*(Y_i - g0_i) / ( p_treated * (1-m_i)^2 ) \]

Both have 0 expectation at the truth (Neyman orthogonality).

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.overlap_diagnostics_atte`

```python
overlap_diagnostics_atte(m, d, eps_list=[0.95, 0.97, 0.98, 0.99])
```

Key overlap metrics for ATTE: availability of suitable controls.
Reports conditional shares: among CONTROLS, fraction with m(X) ≥ threshold; among TREATED, fraction with m(X) ≤ 1 - threshold.

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_irm_orthogonality`

```python
refute_irm_orthogonality(inference_fn, data, trim_propensity=(0.02, 0.98), n_basis_funcs=None, n_folds_oos=4, score=None, trimming_threshold=0.01, strict_oos=True, **inference_kwargs)
```

Comprehensive AIPW orthogonality diagnostics for IRM models.

Implements three key diagnostic approaches based on the efficient influence function (EIF):

1. Out-of-sample moment check (non-tautological)
1. Orthogonality (Gateaux derivative) tests
1. Influence diagnostics

**Parameters:**

- **inference_fn** (<code>[Callable](#typing.Callable)</code>) – The inference function (dml_ate or dml_att)
- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causal data object
- **trim_propensity** (<code>[Tuple](#typing.Tuple)\[[float](#float), [float](#float)\]</code>) – Propensity score trimming bounds (min, max) to avoid extreme weights
- **n_basis_funcs** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Number of basis functions for orthogonality derivative tests (constant + covariates).
  If None, defaults to the number of confounders in `data` plus 1 for the constant term.
- **n_folds_oos** (<code>[int](#int)</code>) – Number of folds for out-of-sample moment check
- \*\***inference_kwargs** (<code>[dict](#dict)</code>) – Additional arguments passed to inference_fn

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary containing:
- oos_moment_test: Out-of-sample moment condition results
- orthogonality_derivatives: Gateaux derivative test results
- influence_diagnostics: Influence function diagnostics
- theta: Original treatment effect estimate
- trimmed_diagnostics: Results on trimmed sample
- overall_assessment: Summary diagnostic assessment

**Examples:**

```pycon
>>> from causalis.refutation.orthogonality import refute_irm_orthogonality
>>> from causalis.scenarios.unconfoundedness.ate.dml_ate import dml_ate
>>> 
>>> # Comprehensive orthogonality check
>>> ortho_results = refute_irm_orthogonality(dml_ate, causal_data)
>>> 
>>> # Check key diagnostics
>>> print(f"OOS moment t-stat: {ortho_results['oos_moment_test']['tstat']:.3f}")
>>> print(f"Assessment: {ortho_results['overall_assessment']}")
```

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_placebo_outcome`

```python
refute_placebo_outcome(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random outcome variables while keeping treatment
and covariates intact. For binary outcomes, generates random binary
variables with the same proportion. For continuous outcomes, generates
random variables from a normal distribution fitted to the original data.
A valid causal design should now yield θ ≈ 0 and a large p-value.

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_placebo_treatment`

```python
refute_placebo_treatment(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random binary treatment variables while keeping outcome and
covariates intact. Generates random binary treatment with the same
proportion as the original treatment. Breaks the treatment–outcome link.

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.refute_subset`

```python
refute_subset(inference_fn, data, fraction=0.8, random_state=None, **inference_kwargs)
```

Re-estimate the effect on a random subset (default 80 %)
to check sample-stability of the estimate.

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.run_score_diagnostics`

```python
run_score_diagnostics(res=None, *, y=None, d=None, g0=None, g1=None, m=None, theta=None, score=None, trimming_threshold=0.01, n_basis_funcs=None, return_summary=True)
```

Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_score_diagnostics(y=..., d=..., g0=..., g1=..., m=..., theta=...)
B) With a model/result:
run_score_diagnostics(res=\<dml_ate/dml_att result dict or IRM-like model>)

Returns a dictionary with:

- params (score, trimming_threshold)
- oos_moment_test (if fast-path caches available on model; else omitted)
- orthogonality_derivatives (DataFrame)
- influence_diagnostics (full_sample)
- summary (compact DataFrame) if return_summary=True
- meta

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.trim_sensitivity_curve_ate`

```python
trim_sensitivity_curve_ate(m_hat, D, Y, g0_hat, g1_hat, eps_grid=(0.0, 0.005, 0.01, 0.02, 0.05))
```

Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).

For each epsilon in eps_grid, compute the AIPW/IRM ATE estimate using
m_clipped = clip(m_hat, eps, 1-eps) over the full sample and report
the plug-in standard error from the EIF.

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **D** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **Y** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g0_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g1_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **eps_grid** (<code>[tuple](#tuple)\[[float](#float), ...\]</code>) – Sequence of clipping thresholds ε to evaluate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Columns: ['trim_eps','n','pct_clipped','theta','se'].
  pct_clipped is the percent of observations with m outside [ε,1-ε].

######## `causalis.scenarios.unconfoundedness.refutation.score.score_validation.trim_sensitivity_curve_atte`

```python
trim_sensitivity_curve_atte(inference_fn, data, m, d, thresholds=np.linspace(0.9, 0.995, 12), **inference_kwargs)
```

Re-estimate θ while progressively trimming CONTROLS with large m(X).

###### `causalis.scenarios.unconfoundedness.refutation.sensitivity_analysis`

```python
sensitivity_analysis(effect_estimation, *, cf_y, cf_d, rho=1.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.

Returns a dict with:

- theta, se, alpha, z
- sampling_ci
- theta_bounds_cofounding = (theta - max_bias, theta + max_bias)
- bias_aware_ci = \[theta - (max_bias + z*se), theta + (max_bias + z*se)\]
- max_bias and components (sigma2, nu2)
- params (cf_y, cf_d, rho, use_signed_rr)

###### `causalis.scenarios.unconfoundedness.refutation.sensitivity_benchmark`

```python
sensitivity_benchmark(effect_estimation, benchmarking_set, fit_args=None)
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing cf_y, cf_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Additional keyword arguments for the IRM.fit() method of the short model.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- cf_y, cf_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

###### `causalis.scenarios.unconfoundedness.refutation.sutva`

**Modules:**

- [**sutva_validation**](#causalis.scenarios.unconfoundedness.refutation.sutva.sutva_validation) – SUTVA validation helper.

####### `causalis.scenarios.unconfoundedness.refutation.sutva.sutva_validation`

SUTVA validation helper.

This module provides a simple function to print four SUTVA-related
questions for the user to consider. It has no side effects on import.

**Functions:**

- [**print_sutva_questions**](#causalis.scenarios.unconfoundedness.refutation.sutva.sutva_validation.print_sutva_questions) – Print the SUTVA validation questions.

**Attributes:**

- [**QUESTIONS**](#causalis.scenarios.unconfoundedness.refutation.sutva.sutva_validation.QUESTIONS) (<code>[Iterable](#typing.Iterable)\[[str](#str)\]</code>) –

######## `causalis.scenarios.unconfoundedness.refutation.sutva.sutva_validation.QUESTIONS`

```python
QUESTIONS: Iterable[str] = ('1.) Are your clients independent (i)?', '2.) Do you measure confounders, treatment, and outcome in the same intervals?', '3.) Do you measure confounders before treatment and outcome after?', '4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?')
```

######## `causalis.scenarios.unconfoundedness.refutation.sutva.sutva_validation.print_sutva_questions`

```python
print_sutva_questions()
```

Print the SUTVA validation questions.

Just prints questions, nothing more.

###### `causalis.scenarios.unconfoundedness.refutation.trim_sensitivity_curve_ate`

```python
trim_sensitivity_curve_ate(m_hat, D, Y, g0_hat, g1_hat, eps_grid=(0.0, 0.005, 0.01, 0.02, 0.05))
```

Sensitivity of ATE estimate to propensity clipping epsilon (no re-fit).

For each epsilon in eps_grid, compute the AIPW/IRM ATE estimate using
m_clipped = clip(m_hat, eps, 1-eps) over the full sample and report
the plug-in standard error from the EIF.

**Parameters:**

- **m_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **D** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **Y** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g0_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **g1_hat** (<code>[ndarray](#numpy.ndarray)</code>) – Cross-fitted nuisances and observed arrays.
- **eps_grid** (<code>[tuple](#tuple)\[[float](#float), ...\]</code>) – Sequence of clipping thresholds ε to evaluate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Columns: ['trim_eps','n','pct_clipped','theta','se'].
  pct_clipped is the percent of observations with m outside [ε,1-ε].

###### `causalis.scenarios.unconfoundedness.refutation.trim_sensitivity_curve_atte`

```python
trim_sensitivity_curve_atte(inference_fn, data, m, d, thresholds=np.linspace(0.9, 0.995, 12), **inference_kwargs)
```

Re-estimate θ while progressively trimming CONTROLS with large m(X).

###### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness`

**Modules:**

- [**sensitivity**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity) – Sensitivity functions refactored into a dedicated module.
- [**uncofoundedness_validation**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation) – Uncofoundedness validation module

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity`

Sensitivity functions refactored into a dedicated module.

This module centralizes bias-aware sensitivity helpers and the public
entry points used by refutation utilities for uncofoundedness.

**Functions:**

- [**get_sensitivity_summary**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_analysis) – Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.
- [**sensitivity_benchmark**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.get_sensitivity_summary`

```python
get_sensitivity_summary(effect_estimation, *, label=None)
```

Render a single, unified bias-aware summary string.
If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_analysis`

```python
sensitivity_analysis(effect_estimation, *, cf_y, cf_d, rho=1.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware components and cache them on `effect_estimation["bias_aware"]`.

Returns a dict with:

- theta, se, alpha, z
- sampling_ci
- theta_bounds_cofounding = (theta - max_bias, theta + max_bias)
- bias_aware_ci = \[theta - (max_bias + z*se), theta + (max_bias + z*se)\]
- max_bias and components (sigma2, nu2)
- params (cf_y, cf_d, rho, use_signed_rr)

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_benchmark`

```python
sensitivity_benchmark(effect_estimation, benchmarking_set, fit_args=None)
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing cf_y, cf_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Additional keyword arguments for the IRM.fit() method of the short model.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- cf_y, cf_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation`

Uncofoundedness validation module

**Functions:**

- [**run_uncofoundedness_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation.run_uncofoundedness_diagnostics) – Uncofoundedness diagnostics focused on balance (SMD).
- [**validate_uncofoundedness_balance**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation.validate_uncofoundedness_balance) – Assess covariate balance under the uncofoundedness assumption by computing

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation.run_uncofoundedness_diagnostics`

```python
run_uncofoundedness_diagnostics(*, res=None, X=None, d=None, m_hat=None, names=None, score=None, normalize=None, threshold=0.1, eps_overlap=0.01, return_summary=True)
```

Uncofoundedness diagnostics focused on balance (SMD).

Inputs:

- Either a result/model via `res`, or raw arrays X, d, m_hat (+ optional names, score, normalize).

Returns a dictionary:
{
"params": {"score", "normalize", "smd_threshold"},
"balance": {"smd", "smd_unweighted", "smd_max", "frac_violations", "pass", "worst_features"},
"flags": {"balance_max_smd", "balance_violations"},
"overall_flag": max severity across balance flags,
"summary": pd.DataFrame with balance rows only
}

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation.validate_uncofoundedness_balance`

```python
validate_uncofoundedness_balance(effect_estimation, *, threshold=0.1, normalize=None)
```

Assess covariate balance under the uncofoundedness assumption by computing
standardized mean differences (SMD) both before weighting (raw groups) and
after weighting using the IPW / ATT weights implied by the DML/IRM estimation.

This function expects the result dictionary returned by dml_ate() or dml_att(),
which includes a fitted IRM model and a 'diagnostic_data' entry with the
necessary arrays.

We compute, for each confounder X_j:

- For ATE (weighted): w1 = D/m_hat, w0 = (1-D)/(1-m_hat).
- For ATTE (weighted): Treated weight = 1 for D=1; Control weight w0 = m_hat/(1-m_hat) for D=0.
- If estimation used normalized IPW (normalize_ipw=True), we scale the corresponding
  weights by their sample mean (as done in IRM) before computing balance.

The SMD is defined as |mu1 - mu0| / s_pooled, where mu_g are (weighted) means in the
(pseudo-)populations and s_pooled is the square root of the average of the (weighted)
variances in the two groups.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Output dict from dml_ate() or dml_att(). Must contain 'model' and 'diagnostic_data'.
- **threshold** (<code>[float](#float)</code>) – Threshold for SMD; values below indicate acceptable balance for most use cases.
- **normalize** (<code>[Optional](#typing.Optional)\[[bool](#bool)\]</code>) – Whether to use normalized weights. If None, inferred from effect_estimation['diagnostic_data']['normalize_ipw'].

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary with keys:
- 'smd': pd.Series of weighted SMD values indexed by confounder names
- 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
- 'score': 'ATE' or 'ATTE'
- 'normalized': bool used for weighting
- 'threshold': float
- 'pass': bool indicating whether all weighted SMDs are below threshold

###### `causalis.scenarios.unconfoundedness.refutation.validate_uncofoundedness_balance`

```python
validate_uncofoundedness_balance(effect_estimation, *, threshold=0.1, normalize=None)
```

Assess covariate balance under the uncofoundedness assumption by computing
standardized mean differences (SMD) both before weighting (raw groups) and
after weighting using the IPW / ATT weights implied by the DML/IRM estimation.

This function expects the result dictionary returned by dml_ate() or dml_att(),
which includes a fitted IRM model and a 'diagnostic_data' entry with the
necessary arrays.

We compute, for each confounder X_j:

- For ATE (weighted): w1 = D/m_hat, w0 = (1-D)/(1-m_hat).
- For ATTE (weighted): Treated weight = 1 for D=1; Control weight w0 = m_hat/(1-m_hat) for D=0.
- If estimation used normalized IPW (normalize_ipw=True), we scale the corresponding
  weights by their sample mean (as done in IRM) before computing balance.

The SMD is defined as |mu1 - mu0| / s_pooled, where mu_g are (weighted) means in the
(pseudo-)populations and s_pooled is the square root of the average of the (weighted)
variances in the two groups.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Output dict from dml_ate() or dml_att(). Must contain 'model' and 'diagnostic_data'.
- **threshold** (<code>[float](#float)</code>) – Threshold for SMD; values below indicate acceptable balance for most use cases.
- **normalize** (<code>[Optional](#typing.Optional)\[[bool](#bool)\]</code>) – Whether to use normalized weights. If None, inferred from effect_estimation['diagnostic_data']['normalize_ipw'].

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary with keys:
- 'smd': pd.Series of weighted SMD values indexed by confounder names
- 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
- 'score': 'ATE' or 'ATTE'
- 'normalized': bool used for weighting
- 'threshold': float
- 'pass': bool indicating whether all weighted SMDs are below threshold

### `causalis.statistics`

**Modules:**

- [**functions**](#causalis.statistics.functions) –
- [**models**](#causalis.statistics.models) –

#### `causalis.statistics.functions`

**Modules:**

- [**bootstrap_diff_in_means**](#causalis.statistics.functions.bootstrap_diff_in_means) – Bootstrap difference-in-means inference for CausalData (ATT context).
- [**confounders_balance**](#causalis.statistics.functions.confounders_balance) –
- [**conversion_ztest**](#causalis.statistics.functions.conversion_ztest) – Two-proportion z-test
- [**outcome_stats**](#causalis.statistics.functions.outcome_stats) – Outcome statistics grouped by treatment for CausalData.
- [**ttest**](#causalis.statistics.functions.ttest) – T-test inference for causaldata objects (ATT context).

##### `causalis.statistics.functions.bootstrap_diff_in_means`

Bootstrap difference-in-means inference for CausalData (ATT context).

Computes the ATT-style difference in means (treated - control) and provides:

- Two-sided p-value using a normal approximation with bootstrap standard error
- Percentile confidence interval for the absolute difference
- Relative difference (%) and corresponding CI relative to control mean

Input:

- data: CausalData
- alpha: float in (0, 1), default 0.05
- n_simul: number of bootstrap simulations (int > 0), default 10000

Output: dict with the same keys as ttest:

- p_value
- absolute_difference
- absolute_ci: (low, high)
- relative_difference
- relative_ci: (low, high)

**Functions:**

- [**bootstrap_diff_means**](#causalis.statistics.functions.bootstrap_diff_in_means.bootstrap_diff_means) – Bootstrap inference for difference in means between treated (T=1) and control (T=0).

###### `causalis.statistics.functions.bootstrap_diff_in_means.bootstrap_diff_means`

```python
bootstrap_diff_means(data, alpha=0.05, n_simul=10000)
```

Bootstrap inference for difference in means between treated (T=1) and control (T=0).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – Dictionary with p_value, absolute_difference, absolute_ci, relative_difference, relative_ci
  (matching the structure of inference.atte.ttest).

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

##### `causalis.statistics.functions.confounders_balance`

**Functions:**

- [**confounders_balance**](#causalis.statistics.functions.confounders_balance.confounders_balance) – Compute balance diagnostics for confounders between treatment groups.

###### `causalis.statistics.functions.confounders_balance.confounders_balance`

```python
confounders_balance(data)
```

Compute balance diagnostics for confounders between treatment groups.

Produces a DataFrame indexed by expanded confounder columns (after one-hot
encoding categorical variables if present) with:

- mean_d_0: mean value for control group (t=0)
- mean_d_1: mean value for treated group (t=1)
- abs_diff: abs(mean_d_1 - mean_d_0)
- smd: standardized mean difference (Cohen's d using pooled std)
- ks_pvalue: p-value for the KS test (rounded to 5 decimal places, non-scientific)

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The causal dataset containing the dataframe, treatment, and confounders.
  Accepts CausalData or any object with `df`, `treatment`, and `confounders`
  attributes/properties.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Balance table sorted by |smd| (descending), index named 'confounders'.

##### `causalis.statistics.functions.conversion_ztest`

Two-proportion z-test

Compares conversion rates between treated (D=1) and control (D=0) groups.
Returns p-value, absolute/relative differences, and their confidence intervals

**Functions:**

- [**conversion_z_test**](#causalis.statistics.functions.conversion_ztest.conversion_z_test) – Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

###### `causalis.statistics.functions.conversion_ztest.conversion_z_test`

```python
conversion_z_test(data, alpha=0.05, ci_method='newcombe', se_for_test='pooled')
```

Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **ci_method** (<code>([newcombe](#newcombe), [wald_unpooled](#wald_unpooled), [wald_pooled](#wald_pooled))</code>) – Method for calculating the confidence interval for the absolute difference.
  "newcombe" is the most robust default for conversion rates.
- **se_for_test** (<code>([pooled](#pooled), [unpooled](#unpooled))</code>) – Method for calculating the standard error for the z-test p-value.
  "pooled" (score test) is generally preferred for testing equality of proportions.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value from the z-test
- absolute_difference: Difference in conversion rates (treated - control)
- absolute_ci: Tuple (lower, upper) for the absolute difference CI
- relative_difference: Percentage change relative to control rate
- relative_ci: Tuple (lower, upper) for the relative difference CI

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment/outcome are missing, treatment is not binary, outcome is not binary,
  groups are empty, or alpha is outside (0, 1).

##### `causalis.statistics.functions.outcome_stats`

Outcome statistics grouped by treatment for CausalData.

**Functions:**

- [**outcome_stats**](#causalis.statistics.functions.outcome_stats.outcome_stats) – Comprehensive outcome statistics grouped by treatment.

###### `causalis.statistics.functions.outcome_stats.outcome_stats`

```python
outcome_stats(data)
```

Comprehensive outcome statistics grouped by treatment.

Returns a DataFrame with detailed outcome statistics for each treatment group,
including count, mean, std, min, various percentiles, and max.
This function provides comprehensive outcome analysis and returns
data in a clean DataFrame format suitable for reporting.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with treatment groups as index and the following columns:
- count: number of observations in each group
- mean: average outcome value
- std: standard deviation of outcome
- min: minimum outcome value
- p10: 10th percentile
- p25: 25th percentile (Q1)
- median: 50th percentile (median)
- p75: 75th percentile (Q3)
- p90: 90th percentile
- max: maximum outcome value

**Examples:**

```pycon
>>> stats = outcome_stats(causal_data)
>>> print(stats)
        count      mean       std       min       p10       p25    median       p75       p90       max
treatment                                                                                                
0        3000  5.123456  2.345678  0.123456  2.345678  3.456789  5.123456  6.789012  7.890123  9.876543
1        2000  6.789012  2.456789  0.234567  3.456789  4.567890  6.789012  8.901234  9.012345  10.987654
```

##### `causalis.statistics.functions.ttest`

T-test inference for causaldata objects (ATT context).

**Functions:**

- [**ttest**](#causalis.statistics.functions.ttest.ttest) – Perform a t-test on a CausalData object to compare the outcome variable between

###### `causalis.statistics.functions.ttest.ttest`

```python
ttest(data, alpha=0.05)
```

Perform a t-test on a CausalData object to compare the outcome variable between
treated (T=1) and control (T=0) groups. Returns differences and confidence intervals.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: The p-value from the t-test
- absolute_difference: The absolute difference between treatment and control means
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference confidence interval
- relative_difference: The relative difference (percentage change) between treatment and control means
- relative_ci: Tuple of (lower, upper) bounds for the relative difference confidence interval

**Raises:**

- <code>[ValueError](#ValueError)</code> – If the CausalData object doesn't have both treatment and outcome variables defined,
  or if the treatment variable is not binary.

#### `causalis.statistics.models`

**Modules:**

- [**blp**](#causalis.statistics.models.blp) –
- [**cuped**](#causalis.statistics.models.cuped) –
- [**diff_in_means**](#causalis.statistics.models.diff_in_means) –
- [**irm**](#causalis.statistics.models.irm) – DML IRM estimator consuming CausalData.

**Classes:**

- [**BLP**](#causalis.statistics.models.BLP) – Best linear predictor (BLP) with orthogonal signals.
- [**CUPEDModel**](#causalis.statistics.models.CUPEDModel) – CUPED / ANCOVA estimator for ATE/ITT in randomized experiments.
- [**CUPEDResults**](#causalis.statistics.models.CUPEDResults) – Result container for CUPED / ANCOVA (and optional Lin-interacted) ATE/ITT estimate.
- [**DiffInMeans**](#causalis.statistics.models.DiffInMeans) – Difference-in-means model for CausalData.
- [**IRM**](#causalis.statistics.models.IRM) – Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

##### `causalis.statistics.models.BLP`

```python
BLP(orth_signal, basis, is_gate=False)
```

Best linear predictor (BLP) with orthogonal signals.
Mainly used for CATE and GATE estimation for IRM models.

The Best Linear Predictor (BLP) targets the coefficient vector :math:`\beta_0` that minimizes the mean squared error
between the true treatment effect function :math:`\tau(X)` and a linear combination of basis functions :math:`b(X)`:

.. math::
\\beta_0 = \\arg\\min\_{\\beta \\in \\mathbb{R}^K} \\mathbb{E}\\Big[\\big(\\tau(X) - b(X)^\\top \\beta \\big)^2\\Big].

This is characterized by the moment condition:

.. math::
\\mathbb{E}[b(X)\\psi] = \\mathbb{E}[b(X)b(X)^\\top]\\beta_0,

where :math:`\psi` is the orthogonal signal such that :math:`\mathbb{E}[\psi \mid X] = \tau(X)`.

The estimator is obtained via OLS of the orthogonal signal on the basis:

.. math::
\\hat{\\beta} = (B^\\top B)^{-1}B^\\top\\psi.

**GATE (Group Average Treatment Effect)**

When `is_gate=True`, the basis consists of group indicators (dummy variables).
In this case, the BLP coefficients correspond to the group means of the orthogonal signal,
which approximate the GATEs:

.. math::
\\hat{\\beta}_k = \\frac{1}{n_k}\\sum_{i:G_i=k}\\psi_i \\approx \\text{GATE}\_k.

**Confidence Intervals**

Confidence intervals for any linear combination :math:`\hat{g} = A\hat{\beta}` are computed using the estimated covariance matrix :math:`\widehat{\Omega}`:

.. math::
\\widehat{\\operatorname{Var}}(\\hat{g}) \\approx A\\widehat{\\Omega}A^\\top.

Pointwise and joint confidence intervals (via Gaussian multiplier bootstrap) are supported.

**Parameters:**

- **orth_signal** (<code>:class:`numpy.array`</code>) – The orthogonal signal to be predicted. Has to be of shape `(n_obs,)`,
  where `n_obs` is the number of observations.
- **basis** (<code>:class:`pandas.DataFrame`</code>) – The basis for estimating the best linear predictor. Has to have the shape `(n_obs, d)`,
  where `n_obs` is the number of observations and `d` is the number of predictors.
- **is_gate** (<code>[bool](#bool)</code>) – Indicates whether the basis is constructed for GATEs (dummy-basis).
  Default is `False`.

**Functions:**

- [**confint**](#causalis.statistics.models.BLP.confint) – Confidence intervals for the BLP model.
- [**fit**](#causalis.statistics.models.BLP.fit) – Estimate BLP models.

**Attributes:**

- [**basis**](#causalis.statistics.models.BLP.basis) – Basis.
- [**blp_model**](#causalis.statistics.models.BLP.blp_model) – Best-Linear-Predictor model.
- [**blp_omega**](#causalis.statistics.models.BLP.blp_omega) – Covariance matrix.
- [**orth_signal**](#causalis.statistics.models.BLP.orth_signal) – Orthogonal signal.
- [**summary**](#causalis.statistics.models.BLP.summary) – A summary for the best linear predictor effect after calling :meth:`fit`.

###### `causalis.statistics.models.BLP.basis`

```python
basis
```

Basis.

###### `causalis.statistics.models.BLP.blp_model`

```python
blp_model
```

Best-Linear-Predictor model.

###### `causalis.statistics.models.BLP.blp_omega`

```python
blp_omega
```

Covariance matrix.

###### `causalis.statistics.models.BLP.confint`

```python
confint(basis=None, joint=False, alpha=0.05, n_rep_boot=500)
```

Confidence intervals for the BLP model.

**Parameters:**

- **basis** (<code>:class:`pandas.DataFrame`</code>) – The basis for constructing the confidence interval. Has to have the same form as the basis from
  the construction. If `None` is passed, if the basis is constructed for GATEs, the GATEs are returned.
  Else, the confidence intervals for the basis coefficients are returned (with pointwise cofidence intervals).
  Default is `None`.
- **joint** (<code>[bool](#bool)</code>) – Indicates whether joint confidence intervals are computed.
  Default is `False`.
- **alpha** (<code>[float](#float)</code>) – The significance level.
  Default is `0.05`.
- **n_rep_boot** (<code>[int](#int)</code>) – The number of bootstrap repetitions (only relevant for joint confidence intervals).
  Default is `500`.

**Returns:**

- **df_ci** (<code>[DataFrame](#pandas.DataFrame)</code>) – A data frame with the confidence interval(s).

###### `causalis.statistics.models.BLP.fit`

```python
fit(cov_type='HC0', **kwargs)
```

Estimate BLP models.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – The covariance type to be used in the estimation. Default is `'HC0'`.
  See :meth:`statsmodels.regression.linear_model.OLS.fit` for more information.
- \*\***kwargs** – Additional keyword arguments to be passed to :meth:`statsmodels.regression.linear_model.OLS.fit`.

**Returns:**

- **self** (<code>[object](#object)</code>) –

###### `causalis.statistics.models.BLP.orth_signal`

```python
orth_signal
```

Orthogonal signal.

###### `causalis.statistics.models.BLP.summary`

```python
summary
```

A summary for the best linear predictor effect after calling :meth:`fit`.

##### `causalis.statistics.models.CUPEDModel`

```python
CUPEDModel(cov_type='HC3', alpha=0.05, center_covariates=True, strict_binary_treatment=True, adjustment='ancova', use_t=True)
```

CUPED / ANCOVA estimator for ATE/ITT in randomized experiments.

Fits an outcome regression with pre-treatment covariates (optionally centered):

```
ANCOVA (classic CUPED form):
    Y ~ 1 + D + X^c

Lin (2013) fully interacted adjustment (optional, best-practice default in many RCTs):
    Y ~ 1 + D + X^c + D * X^c
```

The reported effect is the coefficient on D, with robust covariance as requested.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
  Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
- **alpha** (<code>[float](#float)</code>) – Significance level for confidence intervals.
- **center_covariates** (<code>[bool](#bool)</code>) – If True, center covariates at their sample mean (X^c = X - mean(X)).
  This matches the classic CUPED adjusted-outcome form and improves numerical stability.
- **strict_binary_treatment** (<code>[bool](#bool)</code>) – If True, require treatment to be binary {0,1}.
- **adjustment** (<code>('ancova', 'lin')</code>) – - "ancova": Y ~ 1 + D + X^c
- "lin": Y ~ 1 + D + X^c + D\*X^c
- **use_t** (<code>[bool](#bool)</code>) – Passed to statsmodels `.fit(..., use_t=use_t)`. If False, inference is based on
  normal approximation (common asymptotic choice for robust covariances).

<details class="note" open markdown="1">
<summary>Notes</summary>

- Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
- The Lin (2013) specification is often recommended as a more robust regression-adjustment default
  in RCTs (allows different covariate slopes by treatment arm).

</details>

**Functions:**

- [**estimate**](#causalis.statistics.models.CUPEDModel.estimate) – Return the adjusted ATE/ITT estimate and inference.
- [**fit**](#causalis.statistics.models.CUPEDModel.fit) – Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.
- [**summary_dict**](#causalis.statistics.models.CUPEDModel.summary_dict) – Convenience JSON/logging output.

**Attributes:**

- [**adjustment**](#causalis.statistics.models.CUPEDModel.adjustment) (<code>[Literal](#typing.Literal)['ancova', 'lin']</code>) –
- [**alpha**](#causalis.statistics.models.CUPEDModel.alpha) –
- [**center_covariates**](#causalis.statistics.models.CUPEDModel.center_covariates) –
- [**cov_type**](#causalis.statistics.models.CUPEDModel.cov_type) –
- [**strict_binary_treatment**](#causalis.statistics.models.CUPEDModel.strict_binary_treatment) –
- [**use_t**](#causalis.statistics.models.CUPEDModel.use_t) –

###### `causalis.statistics.models.CUPEDModel.adjustment`

```python
adjustment: Literal['ancova', 'lin'] = adjustment
```

###### `causalis.statistics.models.CUPEDModel.alpha`

```python
alpha = float(alpha)
```

###### `causalis.statistics.models.CUPEDModel.center_covariates`

```python
center_covariates = bool(center_covariates)
```

###### `causalis.statistics.models.CUPEDModel.cov_type`

```python
cov_type = str(cov_type)
```

###### `causalis.statistics.models.CUPEDModel.estimate`

```python
estimate(alpha=None)
```

Return the adjusted ATE/ITT estimate and inference.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[CUPEDResults](#causalis.statistics.models.cuped.CUPEDResults)</code> – Effect estimate (coefficient on D), standard error, test statistic, p-value,
  confidence interval, naive comparator, and variance-reduction diagnostic.

###### `causalis.statistics.models.CUPEDModel.fit`

```python
fit(data, covariates=None)
```

Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
- **covariates** (<code>sequence of str</code>) – Subset of `data.confounders_names` to use as CUPED covariates.
  If None, uses all confounders from the object.

**Returns:**

- <code>[CUPEDModel](#causalis.statistics.models.cuped.CUPEDModel)</code> – Fitted estimator.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If requested covariates are missing, not in `data.confounders_names`,
  or treatment is not binary when `strict_binary_treatment=True`.

###### `causalis.statistics.models.CUPEDModel.strict_binary_treatment`

```python
strict_binary_treatment = bool(strict_binary_treatment)
```

###### `causalis.statistics.models.CUPEDModel.summary_dict`

```python
summary_dict(alpha=None)
```

Convenience JSON/logging output.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with estimates, inference, and diagnostics.

###### `causalis.statistics.models.CUPEDModel.use_t`

```python
use_t = bool(use_t)
```

##### `causalis.statistics.models.CUPEDResults`

```python
CUPEDResults(ate, se, t_stat, p_value, ci_low, ci_high, alpha, nobs, cov_type, use_t, adjustment, ate_naive, se_naive, variance_reduction_pct, covariates, beta_covariates, gamma_interactions)
```

Result container for CUPED / ANCOVA (and optional Lin-interacted) ATE/ITT estimate.

**Attributes:**

- [**ate**](#causalis.statistics.models.CUPEDResults.ate) (<code>[float](#float)</code>) – Estimated ATE/ITT (coefficient on treatment indicator D).

- [**se**](#causalis.statistics.models.CUPEDResults.se) (<code>[float](#float)</code>) – Standard error of `ate` under the requested covariance estimator.

- [**t_stat**](#causalis.statistics.models.CUPEDResults.t_stat) (<code>[float](#float)</code>) – Test statistic for H0: ate = 0 (as reported by statsmodels; depends on `use_t`).

- [**p_value**](#causalis.statistics.models.CUPEDResults.p_value) (<code>[float](#float)</code>) – Two-sided p-value (as reported by statsmodels; depends on `use_t`).

- [**ci_low**](#causalis.statistics.models.CUPEDResults.ci_low) (<code>[float](#float)</code>) – Lower bound of (1 - alpha) confidence interval.

- [**ci_high**](#causalis.statistics.models.CUPEDResults.ci_high) (<code>[float](#float)</code>) – Upper bound of (1 - alpha) confidence interval.

- [**alpha**](#causalis.statistics.models.CUPEDResults.alpha) (<code>[float](#float)</code>) – Significance level used for CI.

- [**nobs**](#causalis.statistics.models.CUPEDResults.nobs) (<code>[int](#int)</code>) – Number of observations.

- [**cov_type**](#causalis.statistics.models.CUPEDResults.cov_type) (<code>[str](#str)</code>) – Covariance estimator name (e.g., "HC3").

- [**use_t**](#causalis.statistics.models.CUPEDResults.use_t) (<code>[bool](#bool)</code>) – Whether inference used t-distribution (True) or normal approximation (False),
  as configured in the fitted statsmodels result.

- [**adjustment**](#causalis.statistics.models.CUPEDResults.adjustment) (<code>{'ancova', 'lin'}</code>) – Which adjustment was used: plain ANCOVA or Lin (fully interacted).

- [**ate_naive**](#causalis.statistics.models.CUPEDResults.ate_naive) (<code>[float](#float)</code>) – Unadjusted difference-in-means estimate (via Y ~ 1 + D).

- [**se_naive**](#causalis.statistics.models.CUPEDResults.se_naive) (<code>[float](#float)</code>) – Standard error of `ate_naive` under the same covariance estimator.

- [**variance_reduction_pct**](#causalis.statistics.models.CUPEDResults.variance_reduction_pct) (<code>[float](#float)</code>) – 100 * (1 - se(ate)^2 / se(ate_naive)^2). Can be negative.

- [**covariates**](#causalis.statistics.models.CUPEDResults.covariates) (<code>[list](#list)\[[str](#str)\]</code>) – Names of covariates used for adjustment.

- [**beta_covariates**](#causalis.statistics.models.CUPEDResults.beta_covariates) (<code>[ndarray](#numpy.ndarray)</code>) – Estimated coefficients on centered covariates X^c (empty if no covariates).

- [**gamma_interactions**](#causalis.statistics.models.CUPEDResults.gamma_interactions) (<code>[ndarray](#numpy.ndarray)</code>) – Estimated coefficients on interactions D * X^c (empty unless adjustment="lin").

**Functions:**

- [**summary**](#causalis.statistics.models.CUPEDResults.summary) – Return a summary DataFrame of the results.
- [**to_dict**](#causalis.statistics.models.CUPEDResults.to_dict) – Convert results to a dictionary.

###### `causalis.statistics.models.CUPEDResults.adjustment`

```python
adjustment: Literal['ancova', 'lin']
```

###### `causalis.statistics.models.CUPEDResults.alpha`

```python
alpha: float
```

###### `causalis.statistics.models.CUPEDResults.ate`

```python
ate: float
```

###### `causalis.statistics.models.CUPEDResults.ate_naive`

```python
ate_naive: float
```

###### `causalis.statistics.models.CUPEDResults.beta_covariates`

```python
beta_covariates: np.ndarray
```

###### `causalis.statistics.models.CUPEDResults.ci_high`

```python
ci_high: float
```

###### `causalis.statistics.models.CUPEDResults.ci_low`

```python
ci_low: float
```

###### `causalis.statistics.models.CUPEDResults.cov_type`

```python
cov_type: str
```

###### `causalis.statistics.models.CUPEDResults.covariates`

```python
covariates: List[str]
```

###### `causalis.statistics.models.CUPEDResults.gamma_interactions`

```python
gamma_interactions: np.ndarray
```

###### `causalis.statistics.models.CUPEDResults.nobs`

```python
nobs: int
```

###### `causalis.statistics.models.CUPEDResults.p_value`

```python
p_value: float
```

###### `causalis.statistics.models.CUPEDResults.se`

```python
se: float
```

###### `causalis.statistics.models.CUPEDResults.se_naive`

```python
se_naive: float
```

###### `causalis.statistics.models.CUPEDResults.summary`

```python
summary()
```

Return a summary DataFrame of the results.

###### `causalis.statistics.models.CUPEDResults.t_stat`

```python
t_stat: float
```

###### `causalis.statistics.models.CUPEDResults.to_dict`

```python
to_dict()
```

Convert results to a dictionary.

###### `causalis.statistics.models.CUPEDResults.use_t`

```python
use_t: bool
```

###### `causalis.statistics.models.CUPEDResults.variance_reduction_pct`

```python
variance_reduction_pct: float
```

##### `causalis.statistics.models.DiffInMeans`

```python
DiffInMeans()
```

Difference-in-means model for CausalData.
Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.

**Functions:**

- [**estimate**](#causalis.statistics.models.DiffInMeans.estimate) – Compute the treatment effect using the specified method.
- [**fit**](#causalis.statistics.models.DiffInMeans.fit) – Fit the model by storing the CausalData object.

**Attributes:**

- [**data**](#causalis.statistics.models.DiffInMeans.data) (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.data.causaldata.CausalData)\]</code>) –

###### `causalis.statistics.models.DiffInMeans.data`

```python
data: Optional[CausalData] = None
```

###### `causalis.statistics.models.DiffInMeans.estimate`

```python
estimate(method='ttest', alpha=0.05, **kwargs)
```

Compute the treatment effect using the specified method.

**Parameters:**

- **method** (<code>('ttest', 'bootstrap', 'conversion_ztest')</code>) – The inference method to use.
- "ttest": Standard independent two-sample t-test.
- "bootstrap": Bootstrap-based inference for difference in means.
- "conversion_ztest": Two-proportion z-test for binary outcomes.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the underlying inference function.
- For "bootstrap": can pass `n_simul` (default 10000).

**Returns:**

- <code>[CausalEstimate](#causalis.data.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

###### `causalis.statistics.models.DiffInMeans.fit`

```python
fit(data)
```

Fit the model by storing the CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

**Returns:**

- <code>[DiffInMeans](#causalis.statistics.models.diff_in_means.DiffInMeans)</code> – The fitted model.

##### `causalis.statistics.models.IRM`

```python
IRM(data=None, ml_g=None, ml_m=None, *, n_folds=5, n_rep=1, score='ATE', normalize_ipw=False, trimming_rule='truncate', trimming_threshold=0.01, weights=None, random_state=None)
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – Data container with outcome, binary treatment (0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
- **n_folds** (<code>[int](#int)</code>) – Number of cross-fitting folds.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions of sample splitting. Currently only 1 is supported.
- **score** (<code>('ATE', 'ATTE')</code>) – Target estimand.
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score.
- **trimming_rule** (<code>'truncate'</code>) – Trimming approach for propensity scores.
- **trimming_threshold** (<code>[float](#float)</code>) – Threshold for trimming if rule is "truncate".
- **weights** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray) or [Dict](#typing.Dict)\]</code>) – Optional weights. If array of shape (n,), used as ATE weights. For ATTE, computed internally.
- **random_state** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Random seed for fold creation.

**Functions:**

- [**confint**](#causalis.statistics.models.IRM.confint) –
- [**estimate**](#causalis.statistics.models.IRM.estimate) – Compute treatment effects using stored nuisance predictions.
- [**fit**](#causalis.statistics.models.IRM.fit) – Fit nuisance models via cross-fitting.
- [**gate**](#causalis.statistics.models.IRM.gate) – Estimate Group Average Treatment Effects via BLP on orthogonal signal.
- [**sensitivity_analysis**](#causalis.statistics.models.IRM.sensitivity_analysis) – Compute a simple sensitivity summary and store it as `sensitivity_summary`.

**Attributes:**

- [**coef**](#causalis.statistics.models.IRM.coef) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**coef\_**](#causalis.statistics.models.IRM.coef_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confint\_**](#causalis.statistics.models.IRM.confint_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**data**](#causalis.statistics.models.IRM.data) –
- [**diagnostics\_**](#causalis.statistics.models.IRM.diagnostics_) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Return diagnostic data.
- [**folds\_**](#causalis.statistics.models.IRM.folds_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g0_hat\_**](#causalis.statistics.models.IRM.g0_hat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g1_hat\_**](#causalis.statistics.models.IRM.g1_hat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**m_hat\_**](#causalis.statistics.models.IRM.m_hat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**ml_g**](#causalis.statistics.models.IRM.ml_g) –
- [**ml_m**](#causalis.statistics.models.IRM.ml_m) –
- [**n_folds**](#causalis.statistics.models.IRM.n_folds) –
- [**n_rep**](#causalis.statistics.models.IRM.n_rep) –
- [**normalize_ipw**](#causalis.statistics.models.IRM.normalize_ipw) –
- [**orth_signal**](#causalis.statistics.models.IRM.orth_signal) (<code>[ndarray](#numpy.ndarray)</code>) – Returns the cross-fitted orthogonal signal (psi).
- [**psi\_**](#causalis.statistics.models.IRM.psi_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_a\_**](#causalis.statistics.models.IRM.psi_a_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_b\_**](#causalis.statistics.models.IRM.psi_b_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**pval\_**](#causalis.statistics.models.IRM.pval_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**pvalues**](#causalis.statistics.models.IRM.pvalues) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**random_state**](#causalis.statistics.models.IRM.random_state) –
- [**score**](#causalis.statistics.models.IRM.score) –
- [**se**](#causalis.statistics.models.IRM.se) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**se\_**](#causalis.statistics.models.IRM.se_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**sensitivity_summary**](#causalis.statistics.models.IRM.sensitivity_summary) (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) –
- [**summary**](#causalis.statistics.models.IRM.summary) (<code>[DataFrame](#pandas.DataFrame)</code>) –
- [**summary\_**](#causalis.statistics.models.IRM.summary_) (<code>[Optional](#typing.Optional)\[[DataFrame](#pandas.DataFrame)\]</code>) –
- [**t_stat\_**](#causalis.statistics.models.IRM.t_stat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**trimming_rule**](#causalis.statistics.models.IRM.trimming_rule) –
- [**trimming_threshold**](#causalis.statistics.models.IRM.trimming_threshold) –
- [**weights**](#causalis.statistics.models.IRM.weights) –

###### `causalis.statistics.models.IRM.coef`

```python
coef: np.ndarray
```

###### `causalis.statistics.models.IRM.coef_`

```python
coef_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.confint`

```python
confint(alpha=0.05)
```

###### `causalis.statistics.models.IRM.confint_`

```python
confint_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.data`

```python
data = data
```

###### `causalis.statistics.models.IRM.diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

###### `causalis.statistics.models.IRM.estimate`

```python
estimate(score=None, alpha=0.05)
```

Compute treatment effects using stored nuisance predictions.

**Parameters:**

- **score** (<code>('ATE', 'ATTE', 'CATE')</code>) – Target estimand. Defaults to self.score.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals.

**Returns:**

- <code>[IRMResults](#causalis.statistics.models.irm.IRMResults) or [ndarray](#numpy.ndarray)</code> – For ATE/ATTE, returns an IRMResults container.
  For CATE, returns the per-observation orthogonal signal (np.ndarray).

###### `causalis.statistics.models.IRM.fit`

```python
fit(data=None)
```

Fit nuisance models via cross-fitting.

###### `causalis.statistics.models.IRM.folds_`

```python
folds_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.g0_hat_`

```python
g0_hat_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.g1_hat_`

```python
g1_hat_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.gate`

```python
gate(groups, alpha=0.05)
```

Estimate Group Average Treatment Effects via BLP on orthogonal signal.

**Parameters:**

- **groups** (<code>[DataFrame](#pandas.DataFrame) or [Series](#pandas.Series)</code>) – Group indicators or labels.
- If a single column (Series or 1-col DataFrame) with non-boolean values,
  it is treated as categorical labels and one-hot encoded.
- If multiple columns or boolean/int indicators, it is used as the basis directly.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals (passed to BLP).

**Returns:**

- <code>[BLP](#causalis.statistics.models.blp.BLP)</code> – Fitted Best Linear Predictor model.

###### `causalis.statistics.models.IRM.m_hat_`

```python
m_hat_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.ml_g`

```python
ml_g = ml_g
```

###### `causalis.statistics.models.IRM.ml_m`

```python
ml_m = ml_m
```

###### `causalis.statistics.models.IRM.n_folds`

```python
n_folds = int(n_folds)
```

###### `causalis.statistics.models.IRM.n_rep`

```python
n_rep = int(n_rep)
```

###### `causalis.statistics.models.IRM.normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

###### `causalis.statistics.models.IRM.orth_signal`

```python
orth_signal: np.ndarray
```

Returns the cross-fitted orthogonal signal (psi).

###### `causalis.statistics.models.IRM.psi_`

```python
psi_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.psi_a_`

```python
psi_a_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.psi_b_`

```python
psi_b_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.pval_`

```python
pval_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.pvalues`

```python
pvalues: np.ndarray
```

###### `causalis.statistics.models.IRM.random_state`

```python
random_state = random_state
```

###### `causalis.statistics.models.IRM.score`

```python
score = str(score).upper()
```

###### `causalis.statistics.models.IRM.se`

```python
se: np.ndarray
```

###### `causalis.statistics.models.IRM.se_`

```python
se_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.sensitivity_analysis`

```python
sensitivity_analysis(cf_y, cf_d, rho=1.0, alpha=0.05)
```

Compute a simple sensitivity summary and store it as `sensitivity_summary`.

**Parameters:**

- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome equation.
- **cf_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment equation.
- **rho** (<code>[float](#float)</code>) – Correlation between unobserved components (display only here).
- **alpha** (<code>[float](#float)</code>) – Significance level for CI bounds display.

###### `causalis.statistics.models.IRM.sensitivity_summary`

```python
sensitivity_summary: Optional[str] = None
```

###### `causalis.statistics.models.IRM.summary`

```python
summary: pd.DataFrame
```

###### `causalis.statistics.models.IRM.summary_`

```python
summary_: Optional[pd.DataFrame] = None
```

###### `causalis.statistics.models.IRM.t_stat_`

```python
t_stat_: Optional[np.ndarray] = None
```

###### `causalis.statistics.models.IRM.trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

###### `causalis.statistics.models.IRM.trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

###### `causalis.statistics.models.IRM.weights`

```python
weights = weights
```

##### `causalis.statistics.models.blp`

**Classes:**

- [**BLP**](#causalis.statistics.models.blp.BLP) – Best linear predictor (BLP) with orthogonal signals.

###### `causalis.statistics.models.blp.BLP`

```python
BLP(orth_signal, basis, is_gate=False)
```

Best linear predictor (BLP) with orthogonal signals.
Mainly used for CATE and GATE estimation for IRM models.

The Best Linear Predictor (BLP) targets the coefficient vector :math:`\beta_0` that minimizes the mean squared error
between the true treatment effect function :math:`\tau(X)` and a linear combination of basis functions :math:`b(X)`:

.. math::
\\beta_0 = \\arg\\min\_{\\beta \\in \\mathbb{R}^K} \\mathbb{E}\\Big[\\big(\\tau(X) - b(X)^\\top \\beta \\big)^2\\Big].

This is characterized by the moment condition:

.. math::
\\mathbb{E}[b(X)\\psi] = \\mathbb{E}[b(X)b(X)^\\top]\\beta_0,

where :math:`\psi` is the orthogonal signal such that :math:`\mathbb{E}[\psi \mid X] = \tau(X)`.

The estimator is obtained via OLS of the orthogonal signal on the basis:

.. math::
\\hat{\\beta} = (B^\\top B)^{-1}B^\\top\\psi.

**GATE (Group Average Treatment Effect)**

When `is_gate=True`, the basis consists of group indicators (dummy variables).
In this case, the BLP coefficients correspond to the group means of the orthogonal signal,
which approximate the GATEs:

.. math::
\\hat{\\beta}_k = \\frac{1}{n_k}\\sum_{i:G_i=k}\\psi_i \\approx \\text{GATE}\_k.

**Confidence Intervals**

Confidence intervals for any linear combination :math:`\hat{g} = A\hat{\beta}` are computed using the estimated covariance matrix :math:`\widehat{\Omega}`:

.. math::
\\widehat{\\operatorname{Var}}(\\hat{g}) \\approx A\\widehat{\\Omega}A^\\top.

Pointwise and joint confidence intervals (via Gaussian multiplier bootstrap) are supported.

**Parameters:**

- **orth_signal** (<code>:class:`numpy.array`</code>) – The orthogonal signal to be predicted. Has to be of shape `(n_obs,)`,
  where `n_obs` is the number of observations.
- **basis** (<code>:class:`pandas.DataFrame`</code>) – The basis for estimating the best linear predictor. Has to have the shape `(n_obs, d)`,
  where `n_obs` is the number of observations and `d` is the number of predictors.
- **is_gate** (<code>[bool](#bool)</code>) – Indicates whether the basis is constructed for GATEs (dummy-basis).
  Default is `False`.

**Functions:**

- [**confint**](#causalis.statistics.models.blp.BLP.confint) – Confidence intervals for the BLP model.
- [**fit**](#causalis.statistics.models.blp.BLP.fit) – Estimate BLP models.

**Attributes:**

- [**basis**](#causalis.statistics.models.blp.BLP.basis) – Basis.
- [**blp_model**](#causalis.statistics.models.blp.BLP.blp_model) – Best-Linear-Predictor model.
- [**blp_omega**](#causalis.statistics.models.blp.BLP.blp_omega) – Covariance matrix.
- [**orth_signal**](#causalis.statistics.models.blp.BLP.orth_signal) – Orthogonal signal.
- [**summary**](#causalis.statistics.models.blp.BLP.summary) – A summary for the best linear predictor effect after calling :meth:`fit`.

####### `causalis.statistics.models.blp.BLP.basis`

```python
basis
```

Basis.

####### `causalis.statistics.models.blp.BLP.blp_model`

```python
blp_model
```

Best-Linear-Predictor model.

####### `causalis.statistics.models.blp.BLP.blp_omega`

```python
blp_omega
```

Covariance matrix.

####### `causalis.statistics.models.blp.BLP.confint`

```python
confint(basis=None, joint=False, alpha=0.05, n_rep_boot=500)
```

Confidence intervals for the BLP model.

**Parameters:**

- **basis** (<code>:class:`pandas.DataFrame`</code>) – The basis for constructing the confidence interval. Has to have the same form as the basis from
  the construction. If `None` is passed, if the basis is constructed for GATEs, the GATEs are returned.
  Else, the confidence intervals for the basis coefficients are returned (with pointwise cofidence intervals).
  Default is `None`.
- **joint** (<code>[bool](#bool)</code>) – Indicates whether joint confidence intervals are computed.
  Default is `False`.
- **alpha** (<code>[float](#float)</code>) – The significance level.
  Default is `0.05`.
- **n_rep_boot** (<code>[int](#int)</code>) – The number of bootstrap repetitions (only relevant for joint confidence intervals).
  Default is `500`.

**Returns:**

- **df_ci** (<code>[DataFrame](#pandas.DataFrame)</code>) – A data frame with the confidence interval(s).

####### `causalis.statistics.models.blp.BLP.fit`

```python
fit(cov_type='HC0', **kwargs)
```

Estimate BLP models.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – The covariance type to be used in the estimation. Default is `'HC0'`.
  See :meth:`statsmodels.regression.linear_model.OLS.fit` for more information.
- \*\***kwargs** – Additional keyword arguments to be passed to :meth:`statsmodels.regression.linear_model.OLS.fit`.

**Returns:**

- **self** (<code>[object](#object)</code>) –

####### `causalis.statistics.models.blp.BLP.orth_signal`

```python
orth_signal
```

Orthogonal signal.

####### `causalis.statistics.models.blp.BLP.summary`

```python
summary
```

A summary for the best linear predictor effect after calling :meth:`fit`.

##### `causalis.statistics.models.cuped`

**Classes:**

- [**CUPEDModel**](#causalis.statistics.models.cuped.CUPEDModel) – CUPED / ANCOVA estimator for ATE/ITT in randomized experiments.
- [**CUPEDResults**](#causalis.statistics.models.cuped.CUPEDResults) – Result container for CUPED / ANCOVA (and optional Lin-interacted) ATE/ITT estimate.

###### `causalis.statistics.models.cuped.CUPEDModel`

```python
CUPEDModel(cov_type='HC3', alpha=0.05, center_covariates=True, strict_binary_treatment=True, adjustment='ancova', use_t=True)
```

CUPED / ANCOVA estimator for ATE/ITT in randomized experiments.

Fits an outcome regression with pre-treatment covariates (optionally centered):

```
ANCOVA (classic CUPED form):
    Y ~ 1 + D + X^c

Lin (2013) fully interacted adjustment (optional, best-practice default in many RCTs):
    Y ~ 1 + D + X^c + D * X^c
```

The reported effect is the coefficient on D, with robust covariance as requested.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
  Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
- **alpha** (<code>[float](#float)</code>) – Significance level for confidence intervals.
- **center_covariates** (<code>[bool](#bool)</code>) – If True, center covariates at their sample mean (X^c = X - mean(X)).
  This matches the classic CUPED adjusted-outcome form and improves numerical stability.
- **strict_binary_treatment** (<code>[bool](#bool)</code>) – If True, require treatment to be binary {0,1}.
- **adjustment** (<code>('ancova', 'lin')</code>) – - "ancova": Y ~ 1 + D + X^c
- "lin": Y ~ 1 + D + X^c + D\*X^c
- **use_t** (<code>[bool](#bool)</code>) – Passed to statsmodels `.fit(..., use_t=use_t)`. If False, inference is based on
  normal approximation (common asymptotic choice for robust covariances).

<details class="note" open markdown="1">
<summary>Notes</summary>

- Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
- The Lin (2013) specification is often recommended as a more robust regression-adjustment default
  in RCTs (allows different covariate slopes by treatment arm).

</details>

**Functions:**

- [**estimate**](#causalis.statistics.models.cuped.CUPEDModel.estimate) – Return the adjusted ATE/ITT estimate and inference.
- [**fit**](#causalis.statistics.models.cuped.CUPEDModel.fit) – Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.
- [**summary_dict**](#causalis.statistics.models.cuped.CUPEDModel.summary_dict) – Convenience JSON/logging output.

**Attributes:**

- [**adjustment**](#causalis.statistics.models.cuped.CUPEDModel.adjustment) (<code>[Literal](#typing.Literal)['ancova', 'lin']</code>) –
- [**alpha**](#causalis.statistics.models.cuped.CUPEDModel.alpha) –
- [**center_covariates**](#causalis.statistics.models.cuped.CUPEDModel.center_covariates) –
- [**cov_type**](#causalis.statistics.models.cuped.CUPEDModel.cov_type) –
- [**strict_binary_treatment**](#causalis.statistics.models.cuped.CUPEDModel.strict_binary_treatment) –
- [**use_t**](#causalis.statistics.models.cuped.CUPEDModel.use_t) –

####### `causalis.statistics.models.cuped.CUPEDModel.adjustment`

```python
adjustment: Literal['ancova', 'lin'] = adjustment
```

####### `causalis.statistics.models.cuped.CUPEDModel.alpha`

```python
alpha = float(alpha)
```

####### `causalis.statistics.models.cuped.CUPEDModel.center_covariates`

```python
center_covariates = bool(center_covariates)
```

####### `causalis.statistics.models.cuped.CUPEDModel.cov_type`

```python
cov_type = str(cov_type)
```

####### `causalis.statistics.models.cuped.CUPEDModel.estimate`

```python
estimate(alpha=None)
```

Return the adjusted ATE/ITT estimate and inference.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[CUPEDResults](#causalis.statistics.models.cuped.CUPEDResults)</code> – Effect estimate (coefficient on D), standard error, test statistic, p-value,
  confidence interval, naive comparator, and variance-reduction diagnostic.

####### `causalis.statistics.models.cuped.CUPEDModel.fit`

```python
fit(data, covariates=None)
```

Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
- **covariates** (<code>sequence of str</code>) – Subset of `data.confounders_names` to use as CUPED covariates.
  If None, uses all confounders from the object.

**Returns:**

- <code>[CUPEDModel](#causalis.statistics.models.cuped.CUPEDModel)</code> – Fitted estimator.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If requested covariates are missing, not in `data.confounders_names`,
  or treatment is not binary when `strict_binary_treatment=True`.

####### `causalis.statistics.models.cuped.CUPEDModel.strict_binary_treatment`

```python
strict_binary_treatment = bool(strict_binary_treatment)
```

####### `causalis.statistics.models.cuped.CUPEDModel.summary_dict`

```python
summary_dict(alpha=None)
```

Convenience JSON/logging output.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with estimates, inference, and diagnostics.

####### `causalis.statistics.models.cuped.CUPEDModel.use_t`

```python
use_t = bool(use_t)
```

###### `causalis.statistics.models.cuped.CUPEDResults`

```python
CUPEDResults(ate, se, t_stat, p_value, ci_low, ci_high, alpha, nobs, cov_type, use_t, adjustment, ate_naive, se_naive, variance_reduction_pct, covariates, beta_covariates, gamma_interactions)
```

Result container for CUPED / ANCOVA (and optional Lin-interacted) ATE/ITT estimate.

**Attributes:**

- [**ate**](#causalis.statistics.models.cuped.CUPEDResults.ate) (<code>[float](#float)</code>) – Estimated ATE/ITT (coefficient on treatment indicator D).

- [**se**](#causalis.statistics.models.cuped.CUPEDResults.se) (<code>[float](#float)</code>) – Standard error of `ate` under the requested covariance estimator.

- [**t_stat**](#causalis.statistics.models.cuped.CUPEDResults.t_stat) (<code>[float](#float)</code>) – Test statistic for H0: ate = 0 (as reported by statsmodels; depends on `use_t`).

- [**p_value**](#causalis.statistics.models.cuped.CUPEDResults.p_value) (<code>[float](#float)</code>) – Two-sided p-value (as reported by statsmodels; depends on `use_t`).

- [**ci_low**](#causalis.statistics.models.cuped.CUPEDResults.ci_low) (<code>[float](#float)</code>) – Lower bound of (1 - alpha) confidence interval.

- [**ci_high**](#causalis.statistics.models.cuped.CUPEDResults.ci_high) (<code>[float](#float)</code>) – Upper bound of (1 - alpha) confidence interval.

- [**alpha**](#causalis.statistics.models.cuped.CUPEDResults.alpha) (<code>[float](#float)</code>) – Significance level used for CI.

- [**nobs**](#causalis.statistics.models.cuped.CUPEDResults.nobs) (<code>[int](#int)</code>) – Number of observations.

- [**cov_type**](#causalis.statistics.models.cuped.CUPEDResults.cov_type) (<code>[str](#str)</code>) – Covariance estimator name (e.g., "HC3").

- [**use_t**](#causalis.statistics.models.cuped.CUPEDResults.use_t) (<code>[bool](#bool)</code>) – Whether inference used t-distribution (True) or normal approximation (False),
  as configured in the fitted statsmodels result.

- [**adjustment**](#causalis.statistics.models.cuped.CUPEDResults.adjustment) (<code>{'ancova', 'lin'}</code>) – Which adjustment was used: plain ANCOVA or Lin (fully interacted).

- [**ate_naive**](#causalis.statistics.models.cuped.CUPEDResults.ate_naive) (<code>[float](#float)</code>) – Unadjusted difference-in-means estimate (via Y ~ 1 + D).

- [**se_naive**](#causalis.statistics.models.cuped.CUPEDResults.se_naive) (<code>[float](#float)</code>) – Standard error of `ate_naive` under the same covariance estimator.

- [**variance_reduction_pct**](#causalis.statistics.models.cuped.CUPEDResults.variance_reduction_pct) (<code>[float](#float)</code>) – 100 * (1 - se(ate)^2 / se(ate_naive)^2). Can be negative.

- [**covariates**](#causalis.statistics.models.cuped.CUPEDResults.covariates) (<code>[list](#list)\[[str](#str)\]</code>) – Names of covariates used for adjustment.

- [**beta_covariates**](#causalis.statistics.models.cuped.CUPEDResults.beta_covariates) (<code>[ndarray](#numpy.ndarray)</code>) – Estimated coefficients on centered covariates X^c (empty if no covariates).

- [**gamma_interactions**](#causalis.statistics.models.cuped.CUPEDResults.gamma_interactions) (<code>[ndarray](#numpy.ndarray)</code>) – Estimated coefficients on interactions D * X^c (empty unless adjustment="lin").

**Functions:**

- [**summary**](#causalis.statistics.models.cuped.CUPEDResults.summary) – Return a summary DataFrame of the results.
- [**to_dict**](#causalis.statistics.models.cuped.CUPEDResults.to_dict) – Convert results to a dictionary.

####### `causalis.statistics.models.cuped.CUPEDResults.adjustment`

```python
adjustment: Literal['ancova', 'lin']
```

####### `causalis.statistics.models.cuped.CUPEDResults.alpha`

```python
alpha: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.ate`

```python
ate: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.ate_naive`

```python
ate_naive: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.beta_covariates`

```python
beta_covariates: np.ndarray
```

####### `causalis.statistics.models.cuped.CUPEDResults.ci_high`

```python
ci_high: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.ci_low`

```python
ci_low: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.cov_type`

```python
cov_type: str
```

####### `causalis.statistics.models.cuped.CUPEDResults.covariates`

```python
covariates: List[str]
```

####### `causalis.statistics.models.cuped.CUPEDResults.gamma_interactions`

```python
gamma_interactions: np.ndarray
```

####### `causalis.statistics.models.cuped.CUPEDResults.nobs`

```python
nobs: int
```

####### `causalis.statistics.models.cuped.CUPEDResults.p_value`

```python
p_value: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.se`

```python
se: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.se_naive`

```python
se_naive: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.summary`

```python
summary()
```

Return a summary DataFrame of the results.

####### `causalis.statistics.models.cuped.CUPEDResults.t_stat`

```python
t_stat: float
```

####### `causalis.statistics.models.cuped.CUPEDResults.to_dict`

```python
to_dict()
```

Convert results to a dictionary.

####### `causalis.statistics.models.cuped.CUPEDResults.use_t`

```python
use_t: bool
```

####### `causalis.statistics.models.cuped.CUPEDResults.variance_reduction_pct`

```python
variance_reduction_pct: float
```

##### `causalis.statistics.models.diff_in_means`

**Classes:**

- [**DiffInMeans**](#causalis.statistics.models.diff_in_means.DiffInMeans) – Difference-in-means model for CausalData.

###### `causalis.statistics.models.diff_in_means.DiffInMeans`

```python
DiffInMeans()
```

Difference-in-means model for CausalData.
Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.

**Functions:**

- [**estimate**](#causalis.statistics.models.diff_in_means.DiffInMeans.estimate) – Compute the treatment effect using the specified method.
- [**fit**](#causalis.statistics.models.diff_in_means.DiffInMeans.fit) – Fit the model by storing the CausalData object.

**Attributes:**

- [**data**](#causalis.statistics.models.diff_in_means.DiffInMeans.data) (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.data.causaldata.CausalData)\]</code>) –

####### `causalis.statistics.models.diff_in_means.DiffInMeans.data`

```python
data: Optional[CausalData] = None
```

####### `causalis.statistics.models.diff_in_means.DiffInMeans.estimate`

```python
estimate(method='ttest', alpha=0.05, **kwargs)
```

Compute the treatment effect using the specified method.

**Parameters:**

- **method** (<code>('ttest', 'bootstrap', 'conversion_ztest')</code>) – The inference method to use.
- "ttest": Standard independent two-sample t-test.
- "bootstrap": Bootstrap-based inference for difference in means.
- "conversion_ztest": Two-proportion z-test for binary outcomes.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the underlying inference function.
- For "bootstrap": can pass `n_simul` (default 10000).

**Returns:**

- <code>[CausalEstimate](#causalis.data.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

####### `causalis.statistics.models.diff_in_means.DiffInMeans.fit`

```python
fit(data)
```

Fit the model by storing the CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

**Returns:**

- <code>[DiffInMeans](#causalis.statistics.models.diff_in_means.DiffInMeans)</code> – The fitted model.

##### `causalis.statistics.models.irm`

DML IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE scores.
This is a lightweight clone of DoubleML's IRM tailored for CausalData input.

software{DoubleML,
title = {{DoubleML} -- Double Machine Learning in Python},
author = {Bach, Philipp and Chernozhukov, Victor and Klaassen, Sven and Kurz, Malte S. and Spindler, Martin},
year = {2024},
version = {latest},
url = {https://github.com/DoubleML/doubleml-for-py},
note = {BSD-3-Clause License. Documentation: \\url{https://docs.doubleml.org/stable/index.html}}
}

**Classes:**

- [**IRM**](#causalis.statistics.models.irm.IRM) – Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.
- [**IRMResults**](#causalis.statistics.models.irm.IRMResults) –

**Functions:**

- [**irm_dml**](#causalis.statistics.models.irm.irm_dml) – Functional core for Interactive Regression Model (IRM) Double Machine Learning.

**Attributes:**

- [**HAS_CATBOOST**](#causalis.statistics.models.irm.HAS_CATBOOST) –

###### `causalis.statistics.models.irm.HAS_CATBOOST`

```python
HAS_CATBOOST = True
```

###### `causalis.statistics.models.irm.IRM`

```python
IRM(data=None, ml_g=None, ml_m=None, *, n_folds=5, n_rep=1, score='ATE', normalize_ipw=False, trimming_rule='truncate', trimming_threshold=0.01, weights=None, random_state=None)
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – Data container with outcome, binary treatment (0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
- **n_folds** (<code>[int](#int)</code>) – Number of cross-fitting folds.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions of sample splitting. Currently only 1 is supported.
- **score** (<code>('ATE', 'ATTE')</code>) – Target estimand.
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score.
- **trimming_rule** (<code>'truncate'</code>) – Trimming approach for propensity scores.
- **trimming_threshold** (<code>[float](#float)</code>) – Threshold for trimming if rule is "truncate".
- **weights** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray) or [Dict](#typing.Dict)\]</code>) – Optional weights. If array of shape (n,), used as ATE weights. For ATTE, computed internally.
- **random_state** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Random seed for fold creation.

**Functions:**

- [**confint**](#causalis.statistics.models.irm.IRM.confint) –
- [**estimate**](#causalis.statistics.models.irm.IRM.estimate) – Compute treatment effects using stored nuisance predictions.
- [**fit**](#causalis.statistics.models.irm.IRM.fit) – Fit nuisance models via cross-fitting.
- [**gate**](#causalis.statistics.models.irm.IRM.gate) – Estimate Group Average Treatment Effects via BLP on orthogonal signal.
- [**sensitivity_analysis**](#causalis.statistics.models.irm.IRM.sensitivity_analysis) – Compute a simple sensitivity summary and store it as `sensitivity_summary`.

**Attributes:**

- [**coef**](#causalis.statistics.models.irm.IRM.coef) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**coef\_**](#causalis.statistics.models.irm.IRM.coef_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confint\_**](#causalis.statistics.models.irm.IRM.confint_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**data**](#causalis.statistics.models.irm.IRM.data) –
- [**diagnostics\_**](#causalis.statistics.models.irm.IRM.diagnostics_) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Return diagnostic data.
- [**folds\_**](#causalis.statistics.models.irm.IRM.folds_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g0_hat\_**](#causalis.statistics.models.irm.IRM.g0_hat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g1_hat\_**](#causalis.statistics.models.irm.IRM.g1_hat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**m_hat\_**](#causalis.statistics.models.irm.IRM.m_hat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**ml_g**](#causalis.statistics.models.irm.IRM.ml_g) –
- [**ml_m**](#causalis.statistics.models.irm.IRM.ml_m) –
- [**n_folds**](#causalis.statistics.models.irm.IRM.n_folds) –
- [**n_rep**](#causalis.statistics.models.irm.IRM.n_rep) –
- [**normalize_ipw**](#causalis.statistics.models.irm.IRM.normalize_ipw) –
- [**orth_signal**](#causalis.statistics.models.irm.IRM.orth_signal) (<code>[ndarray](#numpy.ndarray)</code>) – Returns the cross-fitted orthogonal signal (psi).
- [**psi\_**](#causalis.statistics.models.irm.IRM.psi_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_a\_**](#causalis.statistics.models.irm.IRM.psi_a_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_b\_**](#causalis.statistics.models.irm.IRM.psi_b_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**pval\_**](#causalis.statistics.models.irm.IRM.pval_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**pvalues**](#causalis.statistics.models.irm.IRM.pvalues) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**random_state**](#causalis.statistics.models.irm.IRM.random_state) –
- [**score**](#causalis.statistics.models.irm.IRM.score) –
- [**se**](#causalis.statistics.models.irm.IRM.se) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**se\_**](#causalis.statistics.models.irm.IRM.se_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**sensitivity_summary**](#causalis.statistics.models.irm.IRM.sensitivity_summary) (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) –
- [**summary**](#causalis.statistics.models.irm.IRM.summary) (<code>[DataFrame](#pandas.DataFrame)</code>) –
- [**summary\_**](#causalis.statistics.models.irm.IRM.summary_) (<code>[Optional](#typing.Optional)\[[DataFrame](#pandas.DataFrame)\]</code>) –
- [**t_stat\_**](#causalis.statistics.models.irm.IRM.t_stat_) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**trimming_rule**](#causalis.statistics.models.irm.IRM.trimming_rule) –
- [**trimming_threshold**](#causalis.statistics.models.irm.IRM.trimming_threshold) –
- [**weights**](#causalis.statistics.models.irm.IRM.weights) –

####### `causalis.statistics.models.irm.IRM.coef`

```python
coef: np.ndarray
```

####### `causalis.statistics.models.irm.IRM.coef_`

```python
coef_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.confint`

```python
confint(alpha=0.05)
```

####### `causalis.statistics.models.irm.IRM.confint_`

```python
confint_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.data`

```python
data = data
```

####### `causalis.statistics.models.irm.IRM.diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

####### `causalis.statistics.models.irm.IRM.estimate`

```python
estimate(score=None, alpha=0.05)
```

Compute treatment effects using stored nuisance predictions.

**Parameters:**

- **score** (<code>('ATE', 'ATTE', 'CATE')</code>) – Target estimand. Defaults to self.score.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals.

**Returns:**

- <code>[IRMResults](#causalis.statistics.models.irm.IRMResults) or [ndarray](#numpy.ndarray)</code> – For ATE/ATTE, returns an IRMResults container.
  For CATE, returns the per-observation orthogonal signal (np.ndarray).

####### `causalis.statistics.models.irm.IRM.fit`

```python
fit(data=None)
```

Fit nuisance models via cross-fitting.

####### `causalis.statistics.models.irm.IRM.folds_`

```python
folds_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.g0_hat_`

```python
g0_hat_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.g1_hat_`

```python
g1_hat_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.gate`

```python
gate(groups, alpha=0.05)
```

Estimate Group Average Treatment Effects via BLP on orthogonal signal.

**Parameters:**

- **groups** (<code>[DataFrame](#pandas.DataFrame) or [Series](#pandas.Series)</code>) – Group indicators or labels.
- If a single column (Series or 1-col DataFrame) with non-boolean values,
  it is treated as categorical labels and one-hot encoded.
- If multiple columns or boolean/int indicators, it is used as the basis directly.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals (passed to BLP).

**Returns:**

- <code>[BLP](#causalis.statistics.models.blp.BLP)</code> – Fitted Best Linear Predictor model.

####### `causalis.statistics.models.irm.IRM.m_hat_`

```python
m_hat_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.ml_g`

```python
ml_g = ml_g
```

####### `causalis.statistics.models.irm.IRM.ml_m`

```python
ml_m = ml_m
```

####### `causalis.statistics.models.irm.IRM.n_folds`

```python
n_folds = int(n_folds)
```

####### `causalis.statistics.models.irm.IRM.n_rep`

```python
n_rep = int(n_rep)
```

####### `causalis.statistics.models.irm.IRM.normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

####### `causalis.statistics.models.irm.IRM.orth_signal`

```python
orth_signal: np.ndarray
```

Returns the cross-fitted orthogonal signal (psi).

####### `causalis.statistics.models.irm.IRM.psi_`

```python
psi_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.psi_a_`

```python
psi_a_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.psi_b_`

```python
psi_b_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.pval_`

```python
pval_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.pvalues`

```python
pvalues: np.ndarray
```

####### `causalis.statistics.models.irm.IRM.random_state`

```python
random_state = random_state
```

####### `causalis.statistics.models.irm.IRM.score`

```python
score = str(score).upper()
```

####### `causalis.statistics.models.irm.IRM.se`

```python
se: np.ndarray
```

####### `causalis.statistics.models.irm.IRM.se_`

```python
se_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.sensitivity_analysis`

```python
sensitivity_analysis(cf_y, cf_d, rho=1.0, alpha=0.05)
```

Compute a simple sensitivity summary and store it as `sensitivity_summary`.

**Parameters:**

- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome equation.
- **cf_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment equation.
- **rho** (<code>[float](#float)</code>) – Correlation between unobserved components (display only here).
- **alpha** (<code>[float](#float)</code>) – Significance level for CI bounds display.

####### `causalis.statistics.models.irm.IRM.sensitivity_summary`

```python
sensitivity_summary: Optional[str] = None
```

####### `causalis.statistics.models.irm.IRM.summary`

```python
summary: pd.DataFrame
```

####### `causalis.statistics.models.irm.IRM.summary_`

```python
summary_: Optional[pd.DataFrame] = None
```

####### `causalis.statistics.models.irm.IRM.t_stat_`

```python
t_stat_: Optional[np.ndarray] = None
```

####### `causalis.statistics.models.irm.IRM.trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

####### `causalis.statistics.models.irm.IRM.trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

####### `causalis.statistics.models.irm.IRM.weights`

```python
weights = weights
```

###### `causalis.statistics.models.irm.IRMResults`

```python
IRMResults(coef, se, t_stat, pval, confint, summary_table)
```

**Functions:**

- [**summary**](#causalis.statistics.models.irm.IRMResults.summary) – Return the summary DataFrame.
- [**to_dict**](#causalis.statistics.models.irm.IRMResults.to_dict) – Convert results to a dictionary.

**Attributes:**

- [**coef**](#causalis.statistics.models.irm.IRMResults.coef) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**confint**](#causalis.statistics.models.irm.IRMResults.confint) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**pval**](#causalis.statistics.models.irm.IRMResults.pval) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**se**](#causalis.statistics.models.irm.IRMResults.se) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**summary_table**](#causalis.statistics.models.irm.IRMResults.summary_table) (<code>[DataFrame](#pandas.DataFrame)</code>) –
- [**t_stat**](#causalis.statistics.models.irm.IRMResults.t_stat) (<code>[ndarray](#numpy.ndarray)</code>) –

####### `causalis.statistics.models.irm.IRMResults.coef`

```python
coef: np.ndarray
```

####### `causalis.statistics.models.irm.IRMResults.confint`

```python
confint: np.ndarray
```

####### `causalis.statistics.models.irm.IRMResults.pval`

```python
pval: np.ndarray
```

####### `causalis.statistics.models.irm.IRMResults.se`

```python
se: np.ndarray
```

####### `causalis.statistics.models.irm.IRMResults.summary`

```python
summary()
```

Return the summary DataFrame.

####### `causalis.statistics.models.irm.IRMResults.summary_table`

```python
summary_table: pd.DataFrame
```

####### `causalis.statistics.models.irm.IRMResults.t_stat`

```python
t_stat: np.ndarray
```

####### `causalis.statistics.models.irm.IRMResults.to_dict`

```python
to_dict()
```

Convert results to a dictionary.

###### `causalis.statistics.models.irm.irm_dml`

```python
irm_dml(data, **kwargs)
```

Functional core for Interactive Regression Model (IRM) Double Machine Learning.

**Parameters:**

- **data** (<code>[CausalData](#causalis.data.causaldata.CausalData)</code>) – Data container.
- \*\***kwargs** (<code>[dict](#dict)</code>) – Arguments passed to IRM constructor.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary containing 'inference' (results as dict) and 'diagnostics'.
