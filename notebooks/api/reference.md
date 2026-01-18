## `causalis`

Causalis: A Python package for causal inference.

**Modules:**

- [**data_contracts**](#causalis.data_contracts) –
- [**dgp**](#causalis.dgp) –
- [**scenarios**](#causalis.scenarios) –
- [**shared**](#causalis.shared) –

**Attributes:**

- [**design**](#causalis.design) –

### `causalis.data_contracts`

**Modules:**

- [**causal_diagnostic_data**](#causalis.data_contracts.causal_diagnostic_data) –
- [**causal_estimate**](#causalis.data_contracts.causal_estimate) –
- [**causaldata**](#causalis.data_contracts.causaldata) – Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.
- [**causaldata_instrumental**](#causalis.data_contracts.causaldata_instrumental) –

**Classes:**

- [**CausalData**](#causalis.data_contracts.CausalData) – Container for causal inference datasets.
- [**CausalDataInstrumental**](#causalis.data_contracts.CausalDataInstrumental) – Container for causal inference datasets with causaldata_instrumental variables.
- [**CausalDatasetGenerator**](#causalis.data_contracts.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,
- [**CausalEstimate**](#causalis.data_contracts.CausalEstimate) – Result container for causal effect estimates.
- [**DiagnosticData**](#causalis.data_contracts.DiagnosticData) – Base class for all diagnostic data_contracts.
- [**SmokingDGP**](#causalis.data_contracts.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.
- [**UnconfoundednessDiagnosticData**](#causalis.data_contracts.UnconfoundednessDiagnosticData) – Fields common to all models assuming unconfoundedness.

**Functions:**

- [**generate_classic_rct**](#causalis.data_contracts.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.data_contracts.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_rct**](#causalis.data_contracts.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_gold_linear**](#causalis.data_contracts.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.data_contracts.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.data_contracts.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

#### `causalis.data_contracts.CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data_contracts contract.

**Attributes:**

- [**df**](#causalis.data_contracts.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.data_contracts.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.data_contracts.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.data_contracts.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.data_contracts.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.data_contracts.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.data_contracts.CausalData.get_df) – Get a DataFrame with specified columns.

##### `causalis.data_contracts.CausalData.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

##### `causalis.data_contracts.CausalData.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

##### `causalis.data_contracts.CausalData.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

##### `causalis.data_contracts.CausalData.df`

```python
df: pd.DataFrame
```

##### `causalis.data_contracts.CausalData.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, **kwargs)
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code> – A validated CausalData instance.

##### `causalis.data_contracts.CausalData.get_df`

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

##### `causalis.data_contracts.CausalData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

##### `causalis.data_contracts.CausalData.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

##### `causalis.data_contracts.CausalData.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

##### `causalis.data_contracts.CausalData.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

##### `causalis.data_contracts.CausalData.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

##### `causalis.data_contracts.CausalData.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

##### `causalis.data_contracts.CausalData.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data_contracts.CausalDataInstrumental`

Bases: <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>

Container for causal inference datasets with causaldata_instrumental variables.

**Attributes:**

- [**instrument_name**](#causalis.data_contracts.CausalDataInstrumental.instrument_name) (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.

**Functions:**

- [**from_df**](#causalis.data_contracts.CausalDataInstrumental.from_df) – Friendly constructor for CausalDataInstrumental.
- [**get_df**](#causalis.data_contracts.CausalDataInstrumental.get_df) – Get a DataFrame with specified columns including instrument.

##### `causalis.data_contracts.CausalDataInstrumental.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

##### `causalis.data_contracts.CausalDataInstrumental.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

##### `causalis.data_contracts.CausalDataInstrumental.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

##### `causalis.data_contracts.CausalDataInstrumental.df`

```python
df: pd.DataFrame
```

##### `causalis.data_contracts.CausalDataInstrumental.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, instrument=None, **kwargs)
```

Friendly constructor for CausalDataInstrumental.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- **instrument** (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalDataInstrumental](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental)</code> – A validated CausalDataInstrumental instance.

##### `causalis.data_contracts.CausalDataInstrumental.get_df`

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

##### `causalis.data_contracts.CausalDataInstrumental.instrument`

```python
instrument: pd.Series
```

instrument column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The instrument column.

##### `causalis.data_contracts.CausalDataInstrumental.instrument_name`

```python
instrument_name: str = Field(alias='instrument')
```

##### `causalis.data_contracts.CausalDataInstrumental.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

##### `causalis.data_contracts.CausalDataInstrumental.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

##### `causalis.data_contracts.CausalDataInstrumental.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

##### `causalis.data_contracts.CausalDataInstrumental.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

##### `causalis.data_contracts.CausalDataInstrumental.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

##### `causalis.data_contracts.CausalDataInstrumental.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

##### `causalis.data_contracts.CausalDataInstrumental.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data_contracts.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, score_bounding=None, alpha_zi=-1.0, beta_zi=None, g_zi=None, u_strength_zi=0.0, tau_zi=None, pos_dist='gamma', gamma_shape=2.0, lognormal_sigma=1.0, include_oracle=True, seed=None)
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

- [**rng**](#causalis.data_contracts.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.data_contracts.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data_contracts.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data_contracts.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

##### `causalis.data_contracts.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.alpha_zi`

```python
alpha_zi: float = -1.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.gamma_shape`

```python
gamma_shape: float = 2.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

##### `causalis.data_contracts.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

##### `causalis.data_contracts.CausalDatasetGenerator.k`

```python
k: int = 5
```

##### `causalis.data_contracts.CausalDatasetGenerator.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

##### `causalis.data_contracts.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

##### `causalis.data_contracts.CausalDatasetGenerator.pos_dist`

```python
pos_dist: str = 'gamma'
```

##### `causalis.data_contracts.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

##### `causalis.data_contracts.CausalDatasetGenerator.score_bounding`

```python
score_bounding: Optional[float] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

##### `causalis.data_contracts.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

##### `causalis.data_contracts.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

##### `causalis.data_contracts.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

#### `causalis.data_contracts.CausalEstimate`

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
- **time** (<code>[datetime](#datetime.datetime)</code>) – The date and time when the estimate was created.
- **diagnostic_data** (<code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>) – Additional diagnostic data_contracts.
- **sensitivity_analysis** (<code>[dict](#dict)</code>) – Results from sensitivity analysis.

**Functions:**

- [**summary**](#causalis.data_contracts.CausalEstimate.summary) – Return a summary DataFrame of the results.

**Attributes:**

- [**alpha**](#causalis.data_contracts.CausalEstimate.alpha) (<code>[float](#float)</code>) –
- [**ci_lower_absolute**](#causalis.data_contracts.CausalEstimate.ci_lower_absolute) (<code>[float](#float)</code>) –
- [**ci_lower_relative**](#causalis.data_contracts.CausalEstimate.ci_lower_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**ci_upper_absolute**](#causalis.data_contracts.CausalEstimate.ci_upper_absolute) (<code>[float](#float)</code>) –
- [**ci_upper_relative**](#causalis.data_contracts.CausalEstimate.ci_upper_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**confounders**](#causalis.data_contracts.CausalEstimate.confounders) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**diagnostic_data**](#causalis.data_contracts.CausalEstimate.diagnostic_data) (<code>[Optional](#typing.Optional)\[[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)\]</code>) –
- [**estimand**](#causalis.data_contracts.CausalEstimate.estimand) (<code>[str](#str)</code>) –
- [**is_significant**](#causalis.data_contracts.CausalEstimate.is_significant) (<code>[bool](#bool)</code>) –
- [**model**](#causalis.data_contracts.CausalEstimate.model) (<code>[str](#str)</code>) –
- [**model_config**](#causalis.data_contracts.CausalEstimate.model_config) –
- [**model_options**](#causalis.data_contracts.CausalEstimate.model_options) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**n_control**](#causalis.data_contracts.CausalEstimate.n_control) (<code>[int](#int)</code>) –
- [**n_treated**](#causalis.data_contracts.CausalEstimate.n_treated) (<code>[int](#int)</code>) –
- [**outcome**](#causalis.data_contracts.CausalEstimate.outcome) (<code>[str](#str)</code>) –
- [**p_value**](#causalis.data_contracts.CausalEstimate.p_value) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**sensitivity_analysis**](#causalis.data_contracts.CausalEstimate.sensitivity_analysis) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**time**](#causalis.data_contracts.CausalEstimate.time) (<code>[datetime](#datetime.datetime)</code>) –
- [**treatment**](#causalis.data_contracts.CausalEstimate.treatment) (<code>[str](#str)</code>) –
- [**value**](#causalis.data_contracts.CausalEstimate.value) (<code>[float](#float)</code>) –
- [**value_relative**](#causalis.data_contracts.CausalEstimate.value_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –

##### `causalis.data_contracts.CausalEstimate.alpha`

```python
alpha: float
```

##### `causalis.data_contracts.CausalEstimate.ci_lower_absolute`

```python
ci_lower_absolute: float
```

##### `causalis.data_contracts.CausalEstimate.ci_lower_relative`

```python
ci_lower_relative: Optional[float] = None
```

##### `causalis.data_contracts.CausalEstimate.ci_upper_absolute`

```python
ci_upper_absolute: float
```

##### `causalis.data_contracts.CausalEstimate.ci_upper_relative`

```python
ci_upper_relative: Optional[float] = None
```

##### `causalis.data_contracts.CausalEstimate.confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

##### `causalis.data_contracts.CausalEstimate.diagnostic_data`

```python
diagnostic_data: Optional[DiagnosticData] = None
```

##### `causalis.data_contracts.CausalEstimate.estimand`

```python
estimand: str
```

##### `causalis.data_contracts.CausalEstimate.is_significant`

```python
is_significant: bool
```

##### `causalis.data_contracts.CausalEstimate.model`

```python
model: str
```

##### `causalis.data_contracts.CausalEstimate.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `causalis.data_contracts.CausalEstimate.model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

##### `causalis.data_contracts.CausalEstimate.n_control`

```python
n_control: int
```

##### `causalis.data_contracts.CausalEstimate.n_treated`

```python
n_treated: int
```

##### `causalis.data_contracts.CausalEstimate.outcome`

```python
outcome: str
```

##### `causalis.data_contracts.CausalEstimate.p_value`

```python
p_value: Optional[float] = None
```

##### `causalis.data_contracts.CausalEstimate.sensitivity_analysis`

```python
sensitivity_analysis: Dict[str, Any] = Field(default_factory=dict)
```

##### `causalis.data_contracts.CausalEstimate.summary`

```python
summary()
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Summary DataFrame.

##### `causalis.data_contracts.CausalEstimate.time`

```python
time: datetime = Field(default_factory=(datetime.now))
```

##### `causalis.data_contracts.CausalEstimate.treatment`

```python
treatment: str
```

##### `causalis.data_contracts.CausalEstimate.value`

```python
value: float
```

##### `causalis.data_contracts.CausalEstimate.value_relative`

```python
value_relative: Optional[float] = None
```

#### `causalis.data_contracts.DiagnosticData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Base class for all diagnostic data_contracts.

**Attributes:**

- [**model_config**](#causalis.data_contracts.DiagnosticData.model_config) –

##### `causalis.data_contracts.DiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

#### `causalis.data_contracts.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.dgp.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.data_contracts.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.data_contracts.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.data_contracts.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.data_contracts.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.data_contracts.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**alpha_zi**](#causalis.data_contracts.SmokingDGP.alpha_zi) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.data_contracts.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.data_contracts.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_zi**](#causalis.data_contracts.SmokingDGP.beta_zi) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.data_contracts.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.data_contracts.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.data_contracts.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.data_contracts.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_zi**](#causalis.data_contracts.SmokingDGP.g_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**gamma_shape**](#causalis.data_contracts.SmokingDGP.gamma_shape) (<code>[float](#float)</code>) –
- [**include_oracle**](#causalis.data_contracts.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.data_contracts.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**lognormal_sigma**](#causalis.data_contracts.SmokingDGP.lognormal_sigma) (<code>[float](#float)</code>) –
- [**outcome_type**](#causalis.data_contracts.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**pos_dist**](#causalis.data_contracts.SmokingDGP.pos_dist) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.data_contracts.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.data_contracts.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**score_bounding**](#causalis.data_contracts.SmokingDGP.score_bounding) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**seed**](#causalis.data_contracts.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.data_contracts.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.data_contracts.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.data_contracts.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**tau_zi**](#causalis.data_contracts.SmokingDGP.tau_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.data_contracts.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.data_contracts.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.data_contracts.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**u_strength_zi**](#causalis.data_contracts.SmokingDGP.u_strength_zi) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.data_contracts.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.data_contracts.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

##### `causalis.data_contracts.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

##### `causalis.data_contracts.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

##### `causalis.data_contracts.SmokingDGP.alpha_zi`

```python
alpha_zi: float = -1.0
```

##### `causalis.data_contracts.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.SmokingDGP.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

##### `causalis.data_contracts.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.SmokingDGP.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.SmokingDGP.gamma_shape`

```python
gamma_shape: float = 2.0
```

##### `causalis.data_contracts.SmokingDGP.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

##### `causalis.data_contracts.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

##### `causalis.data_contracts.SmokingDGP.k`

```python
k: int = 5
```

##### `causalis.data_contracts.SmokingDGP.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

##### `causalis.data_contracts.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

##### `causalis.data_contracts.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

##### `causalis.data_contracts.SmokingDGP.pos_dist`

```python
pos_dist: str = 'gamma'
```

##### `causalis.data_contracts.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

##### `causalis.data_contracts.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

##### `causalis.data_contracts.SmokingDGP.score_bounding`

```python
score_bounding: Optional[float] = None
```

##### `causalis.data_contracts.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

##### `causalis.data_contracts.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

##### `causalis.data_contracts.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

##### `causalis.data_contracts.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.SmokingDGP.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.data_contracts.SmokingDGP.theta`

```python
theta: float = 1.0
```

##### `causalis.data_contracts.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

##### `causalis.data_contracts.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

##### `causalis.data_contracts.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

##### `causalis.data_contracts.SmokingDGP.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

##### `causalis.data_contracts.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

##### `causalis.data_contracts.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

#### `causalis.data_contracts.UnconfoundednessDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Fields common to all models assuming unconfoundedness.

**Attributes:**

- [**d**](#causalis.data_contracts.UnconfoundednessDiagnosticData.d) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**folds**](#causalis.data_contracts.UnconfoundednessDiagnosticData.folds) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g0_hat**](#causalis.data_contracts.UnconfoundednessDiagnosticData.g0_hat) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g1_hat**](#causalis.data_contracts.UnconfoundednessDiagnosticData.g1_hat) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**m_alpha**](#causalis.data_contracts.UnconfoundednessDiagnosticData.m_alpha) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**m_hat**](#causalis.data_contracts.UnconfoundednessDiagnosticData.m_hat) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**model_config**](#causalis.data_contracts.UnconfoundednessDiagnosticData.model_config) –
- [**nu2**](#causalis.data_contracts.UnconfoundednessDiagnosticData.nu2) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**psi**](#causalis.data_contracts.UnconfoundednessDiagnosticData.psi) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_b**](#causalis.data_contracts.UnconfoundednessDiagnosticData.psi_b) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_nu2**](#causalis.data_contracts.UnconfoundednessDiagnosticData.psi_nu2) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_sigma2**](#causalis.data_contracts.UnconfoundednessDiagnosticData.psi_sigma2) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**riesz_rep**](#causalis.data_contracts.UnconfoundednessDiagnosticData.riesz_rep) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**score**](#causalis.data_contracts.UnconfoundednessDiagnosticData.score) (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) –
- [**sigma2**](#causalis.data_contracts.UnconfoundednessDiagnosticData.sigma2) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**trimming_threshold**](#causalis.data_contracts.UnconfoundednessDiagnosticData.trimming_threshold) (<code>[float](#float)</code>) –
- [**x**](#causalis.data_contracts.UnconfoundednessDiagnosticData.x) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**y**](#causalis.data_contracts.UnconfoundednessDiagnosticData.y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.d`

```python
d: np.ndarray
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.folds`

```python
folds: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.g0_hat`

```python
g0_hat: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.g1_hat`

```python
g1_hat: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.m_alpha`

```python
m_alpha: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.m_hat`

```python
m_hat: np.ndarray
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.nu2`

```python
nu2: Optional[float] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.psi`

```python
psi: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.psi_b`

```python
psi_b: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.psi_nu2`

```python
psi_nu2: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.psi_sigma2`

```python
psi_sigma2: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.riesz_rep`

```python
riesz_rep: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.score`

```python
score: Optional[str] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.sigma2`

```python
sigma2: Optional[float] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.trimming_threshold`

```python
trimming_threshold: float = 0.0
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.x`

```python
x: Optional[np.ndarray] = None
```

##### `causalis.data_contracts.UnconfoundednessDiagnosticData.y`

```python
y: Optional[np.ndarray] = None
```

#### `causalis.data_contracts.causal_diagnostic_data`

**Classes:**

- [**CUPEDDiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData) – Diagnostic data_contracts for CUPED / ANCOVA models.
- [**DiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData) – Base class for all diagnostic data_contracts.
- [**DiffInMeansDiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.DiffInMeansDiagnosticData) – Diagnostic data_contracts for Difference-in-Means model.
- [**UnconfoundednessDiagnosticData**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData) – Fields common to all models assuming unconfoundedness.

##### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Diagnostic data_contracts for CUPED / ANCOVA models.

**Attributes:**

- [**adj_type**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.adj_type) (<code>[str](#str)</code>) –
- [**ate_naive**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.ate_naive) (<code>[float](#float)</code>) –
- [**beta_covariates**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.beta_covariates) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**covariates**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.covariates) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**gamma_interactions**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.gamma_interactions) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**model_config**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.model_config) –
- [**se_naive**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.se_naive) (<code>[float](#float)</code>) –
- [**variance_reduction_pct**](#causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.variance_reduction_pct) (<code>[float](#float)</code>) –

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.adj_type`

```python
adj_type: str
```

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.ate_naive`

```python
ate_naive: float
```

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.beta_covariates`

```python
beta_covariates: np.ndarray
```

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.covariates`

```python
covariates: List[str]
```

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.gamma_interactions`

```python
gamma_interactions: np.ndarray
```

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.se_naive`

```python
se_naive: float
```

###### `causalis.data_contracts.causal_diagnostic_data.CUPEDDiagnosticData.variance_reduction_pct`

```python
variance_reduction_pct: float
```

##### `causalis.data_contracts.causal_diagnostic_data.DiagnosticData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Base class for all diagnostic data_contracts.

**Attributes:**

- [**model_config**](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData.model_config) –

###### `causalis.data_contracts.causal_diagnostic_data.DiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

##### `causalis.data_contracts.causal_diagnostic_data.DiffInMeansDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Diagnostic data_contracts for Difference-in-Means model.

**Attributes:**

- [**model_config**](#causalis.data_contracts.causal_diagnostic_data.DiffInMeansDiagnosticData.model_config) –

##### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData`

Bases: <code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>

Fields common to all models assuming unconfoundedness.

**Attributes:**

- [**d**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.d) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**folds**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.folds) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g0_hat**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.g0_hat) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g1_hat**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.g1_hat) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**m_alpha**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.m_alpha) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**m_hat**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.m_hat) (<code>[ndarray](#numpy.ndarray)</code>) –
- [**model_config**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.model_config) –
- [**nu2**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.nu2) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**psi**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_b**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi_b) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_nu2**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi_nu2) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**psi_sigma2**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi_sigma2) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**riesz_rep**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.riesz_rep) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**score**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.score) (<code>[Optional](#typing.Optional)\[[str](#str)\]</code>) –
- [**sigma2**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.sigma2) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**trimming_threshold**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.trimming_threshold) (<code>[float](#float)</code>) –
- [**x**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.x) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**y**](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.d`

```python
d: np.ndarray
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.folds`

```python
folds: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.g0_hat`

```python
g0_hat: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.g1_hat`

```python
g1_hat: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.m_alpha`

```python
m_alpha: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.m_hat`

```python
m_hat: np.ndarray
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.nu2`

```python
nu2: Optional[float] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi`

```python
psi: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi_b`

```python
psi_b: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi_nu2`

```python
psi_nu2: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.psi_sigma2`

```python
psi_sigma2: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.riesz_rep`

```python
riesz_rep: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.score`

```python
score: Optional[str] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.sigma2`

```python
sigma2: Optional[float] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.trimming_threshold`

```python
trimming_threshold: float = 0.0
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.x`

```python
x: Optional[np.ndarray] = None
```

###### `causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData.y`

```python
y: Optional[np.ndarray] = None
```

#### `causalis.data_contracts.causal_estimate`

**Classes:**

- [**CausalEstimate**](#causalis.data_contracts.causal_estimate.CausalEstimate) – Result container for causal effect estimates.

##### `causalis.data_contracts.causal_estimate.CausalEstimate`

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
- **time** (<code>[datetime](#datetime.datetime)</code>) – The date and time when the estimate was created.
- **diagnostic_data** (<code>[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)</code>) – Additional diagnostic data_contracts.
- **sensitivity_analysis** (<code>[dict](#dict)</code>) – Results from sensitivity analysis.

**Functions:**

- [**summary**](#causalis.data_contracts.causal_estimate.CausalEstimate.summary) – Return a summary DataFrame of the results.

**Attributes:**

- [**alpha**](#causalis.data_contracts.causal_estimate.CausalEstimate.alpha) (<code>[float](#float)</code>) –
- [**ci_lower_absolute**](#causalis.data_contracts.causal_estimate.CausalEstimate.ci_lower_absolute) (<code>[float](#float)</code>) –
- [**ci_lower_relative**](#causalis.data_contracts.causal_estimate.CausalEstimate.ci_lower_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**ci_upper_absolute**](#causalis.data_contracts.causal_estimate.CausalEstimate.ci_upper_absolute) (<code>[float](#float)</code>) –
- [**ci_upper_relative**](#causalis.data_contracts.causal_estimate.CausalEstimate.ci_upper_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**confounders**](#causalis.data_contracts.causal_estimate.CausalEstimate.confounders) (<code>[List](#typing.List)\[[str](#str)\]</code>) –
- [**diagnostic_data**](#causalis.data_contracts.causal_estimate.CausalEstimate.diagnostic_data) (<code>[Optional](#typing.Optional)\[[DiagnosticData](#causalis.data_contracts.causal_diagnostic_data.DiagnosticData)\]</code>) –
- [**estimand**](#causalis.data_contracts.causal_estimate.CausalEstimate.estimand) (<code>[str](#str)</code>) –
- [**is_significant**](#causalis.data_contracts.causal_estimate.CausalEstimate.is_significant) (<code>[bool](#bool)</code>) –
- [**model**](#causalis.data_contracts.causal_estimate.CausalEstimate.model) (<code>[str](#str)</code>) –
- [**model_config**](#causalis.data_contracts.causal_estimate.CausalEstimate.model_config) –
- [**model_options**](#causalis.data_contracts.causal_estimate.CausalEstimate.model_options) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**n_control**](#causalis.data_contracts.causal_estimate.CausalEstimate.n_control) (<code>[int](#int)</code>) –
- [**n_treated**](#causalis.data_contracts.causal_estimate.CausalEstimate.n_treated) (<code>[int](#int)</code>) –
- [**outcome**](#causalis.data_contracts.causal_estimate.CausalEstimate.outcome) (<code>[str](#str)</code>) –
- [**p_value**](#causalis.data_contracts.causal_estimate.CausalEstimate.p_value) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**sensitivity_analysis**](#causalis.data_contracts.causal_estimate.CausalEstimate.sensitivity_analysis) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) –
- [**time**](#causalis.data_contracts.causal_estimate.CausalEstimate.time) (<code>[datetime](#datetime.datetime)</code>) –
- [**treatment**](#causalis.data_contracts.causal_estimate.CausalEstimate.treatment) (<code>[str](#str)</code>) –
- [**value**](#causalis.data_contracts.causal_estimate.CausalEstimate.value) (<code>[float](#float)</code>) –
- [**value_relative**](#causalis.data_contracts.causal_estimate.CausalEstimate.value_relative) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –

###### `causalis.data_contracts.causal_estimate.CausalEstimate.alpha`

```python
alpha: float
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.ci_lower_absolute`

```python
ci_lower_absolute: float
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.ci_lower_relative`

```python
ci_lower_relative: Optional[float] = None
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.ci_upper_absolute`

```python
ci_upper_absolute: float
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.ci_upper_relative`

```python
ci_upper_relative: Optional[float] = None
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.confounders`

```python
confounders: List[str] = Field(default_factory=list)
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.diagnostic_data`

```python
diagnostic_data: Optional[DiagnosticData] = None
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.estimand`

```python
estimand: str
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.is_significant`

```python
is_significant: bool
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.model`

```python
model: str
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.model_options`

```python
model_options: Dict[str, Any] = Field(default_factory=dict)
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.n_control`

```python
n_control: int
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.n_treated`

```python
n_treated: int
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.outcome`

```python
outcome: str
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.p_value`

```python
p_value: Optional[float] = None
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.sensitivity_analysis`

```python
sensitivity_analysis: Dict[str, Any] = Field(default_factory=dict)
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.summary`

```python
summary()
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Summary DataFrame.

###### `causalis.data_contracts.causal_estimate.CausalEstimate.time`

```python
time: datetime = Field(default_factory=(datetime.now))
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.treatment`

```python
treatment: str
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.value`

```python
value: float
```

###### `causalis.data_contracts.causal_estimate.CausalEstimate.value_relative`

```python
value_relative: Optional[float] = None
```

#### `causalis.data_contracts.causaldata`

Causalis Dataclass for storing Cross-sectional DataFrame and column metadata for causal inference.

**Classes:**

- [**CausalData**](#causalis.data_contracts.causaldata.CausalData) – Container for causal inference datasets.

##### `causalis.data_contracts.causaldata.CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data_contracts contract.

**Attributes:**

- [**df**](#causalis.data_contracts.causaldata.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.data_contracts.causaldata.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.data_contracts.causaldata.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.data_contracts.causaldata.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.data_contracts.causaldata.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.data_contracts.causaldata.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.data_contracts.causaldata.CausalData.get_df) – Get a DataFrame with specified columns.

###### `causalis.data_contracts.causaldata.CausalData.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

###### `causalis.data_contracts.causaldata.CausalData.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

###### `causalis.data_contracts.causaldata.CausalData.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

###### `causalis.data_contracts.causaldata.CausalData.df`

```python
df: pd.DataFrame
```

###### `causalis.data_contracts.causaldata.CausalData.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, **kwargs)
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code> – A validated CausalData instance.

###### `causalis.data_contracts.causaldata.CausalData.get_df`

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

###### `causalis.data_contracts.causaldata.CausalData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

###### `causalis.data_contracts.causaldata.CausalData.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

###### `causalis.data_contracts.causaldata.CausalData.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

###### `causalis.data_contracts.causaldata.CausalData.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

###### `causalis.data_contracts.causaldata.CausalData.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

###### `causalis.data_contracts.causaldata.CausalData.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

###### `causalis.data_contracts.causaldata.CausalData.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data_contracts.causaldata_instrumental`

**Classes:**

- [**CausalDataInstrumental**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental) – Container for causal inference datasets with causaldata_instrumental variables.

##### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental`

Bases: <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>

Container for causal inference datasets with causaldata_instrumental variables.

**Attributes:**

- [**instrument_name**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.instrument_name) (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.

**Functions:**

- [**from_df**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.from_df) – Friendly constructor for CausalDataInstrumental.
- [**get_df**](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.get_df) – Get a DataFrame with specified columns including instrument.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.df`

```python
df: pd.DataFrame
```

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, instrument=None, **kwargs)
```

Friendly constructor for CausalDataInstrumental.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- **instrument** (<code>[str](#str)</code>) – Column name representing the causaldata_instrumental variable.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalDataInstrumental](#causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental)</code> – A validated CausalDataInstrumental instance.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.get_df`

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

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.instrument`

```python
instrument: pd.Series
```

instrument column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The instrument column.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.instrument_name`

```python
instrument_name: str = Field(alias='instrument')
```

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

###### `causalis.data_contracts.causaldata_instrumental.CausalDataInstrumental.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

#### `causalis.data_contracts.generate_classic_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

#### `causalis.data_contracts.generate_classic_rct_26`

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

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

#### `causalis.data_contracts.generate_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic RCT dataset.

#### `causalis.data_contracts.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

#### `causalis.data_contracts.obs_linear_26_dataset`

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

#### `causalis.data_contracts.obs_linear_effect`

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

### `causalis.design`

```python
design = None
```

### `causalis.dgp`

**Modules:**

- [**base**](#causalis.dgp.base) –
- [**causaldata**](#causalis.dgp.causaldata) –
- [**causaldata_instrumental**](#causalis.dgp.causaldata_instrumental) –

**Classes:**

- [**CausalDatasetGenerator**](#causalis.dgp.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,
- [**SmokingDGP**](#causalis.dgp.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.

**Functions:**

- [**generate_classic_rct**](#causalis.dgp.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.dgp.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_iv_data**](#causalis.dgp.generate_iv_data) – Generate synthetic dataset with instrumental variables.
- [**generate_rct**](#causalis.dgp.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_cuped_tweedie**](#causalis.dgp.make_cuped_tweedie) – Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_cuped_tweedie_26**](#causalis.dgp.make_cuped_tweedie_26) – Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_gold_linear**](#causalis.dgp.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.dgp.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.dgp.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

#### `causalis.dgp.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, score_bounding=None, alpha_zi=-1.0, beta_zi=None, g_zi=None, u_strength_zi=0.0, tau_zi=None, pos_dist='gamma', gamma_shape=2.0, lognormal_sigma=1.0, include_oracle=True, seed=None)
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

- [**rng**](#causalis.dgp.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.dgp.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

##### `causalis.dgp.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

##### `causalis.dgp.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

##### `causalis.dgp.CausalDatasetGenerator.alpha_zi`

```python
alpha_zi: float = -1.0
```

##### `causalis.dgp.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

##### `causalis.dgp.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

##### `causalis.dgp.CausalDatasetGenerator.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

##### `causalis.dgp.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

##### `causalis.dgp.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

##### `causalis.dgp.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.CausalDatasetGenerator.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.CausalDatasetGenerator.gamma_shape`

```python
gamma_shape: float = 2.0
```

##### `causalis.dgp.CausalDatasetGenerator.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

##### `causalis.dgp.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

##### `causalis.dgp.CausalDatasetGenerator.k`

```python
k: int = 5
```

##### `causalis.dgp.CausalDatasetGenerator.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

##### `causalis.dgp.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

##### `causalis.dgp.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

##### `causalis.dgp.CausalDatasetGenerator.pos_dist`

```python
pos_dist: str = 'gamma'
```

##### `causalis.dgp.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

##### `causalis.dgp.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

##### `causalis.dgp.CausalDatasetGenerator.score_bounding`

```python
score_bounding: Optional[float] = None
```

##### `causalis.dgp.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

##### `causalis.dgp.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

##### `causalis.dgp.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

##### `causalis.dgp.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.CausalDatasetGenerator.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

##### `causalis.dgp.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

##### `causalis.dgp.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

##### `causalis.dgp.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

##### `causalis.dgp.CausalDatasetGenerator.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

##### `causalis.dgp.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

##### `causalis.dgp.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

#### `causalis.dgp.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.dgp.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.dgp.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.dgp.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.dgp.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**alpha_zi**](#causalis.dgp.SmokingDGP.alpha_zi) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.dgp.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.dgp.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_zi**](#causalis.dgp.SmokingDGP.beta_zi) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.dgp.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.dgp.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.dgp.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.dgp.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_zi**](#causalis.dgp.SmokingDGP.g_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**gamma_shape**](#causalis.dgp.SmokingDGP.gamma_shape) (<code>[float](#float)</code>) –
- [**include_oracle**](#causalis.dgp.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.dgp.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**lognormal_sigma**](#causalis.dgp.SmokingDGP.lognormal_sigma) (<code>[float](#float)</code>) –
- [**outcome_type**](#causalis.dgp.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**pos_dist**](#causalis.dgp.SmokingDGP.pos_dist) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.dgp.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.dgp.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**score_bounding**](#causalis.dgp.SmokingDGP.score_bounding) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**seed**](#causalis.dgp.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.dgp.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.dgp.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.dgp.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**tau_zi**](#causalis.dgp.SmokingDGP.tau_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.dgp.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.dgp.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.dgp.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**u_strength_zi**](#causalis.dgp.SmokingDGP.u_strength_zi) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.dgp.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.dgp.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

##### `causalis.dgp.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

##### `causalis.dgp.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

##### `causalis.dgp.SmokingDGP.alpha_zi`

```python
alpha_zi: float = -1.0
```

##### `causalis.dgp.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

##### `causalis.dgp.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

##### `causalis.dgp.SmokingDGP.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

##### `causalis.dgp.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

##### `causalis.dgp.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

##### `causalis.dgp.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.SmokingDGP.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.SmokingDGP.gamma_shape`

```python
gamma_shape: float = 2.0
```

##### `causalis.dgp.SmokingDGP.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

##### `causalis.dgp.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

##### `causalis.dgp.SmokingDGP.k`

```python
k: int = 5
```

##### `causalis.dgp.SmokingDGP.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

##### `causalis.dgp.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

##### `causalis.dgp.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

##### `causalis.dgp.SmokingDGP.pos_dist`

```python
pos_dist: str = 'gamma'
```

##### `causalis.dgp.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

##### `causalis.dgp.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

##### `causalis.dgp.SmokingDGP.score_bounding`

```python
score_bounding: Optional[float] = None
```

##### `causalis.dgp.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

##### `causalis.dgp.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

##### `causalis.dgp.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

##### `causalis.dgp.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.SmokingDGP.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

##### `causalis.dgp.SmokingDGP.theta`

```python
theta: float = 1.0
```

##### `causalis.dgp.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

##### `causalis.dgp.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

##### `causalis.dgp.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

##### `causalis.dgp.SmokingDGP.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

##### `causalis.dgp.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

##### `causalis.dgp.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

#### `causalis.dgp.base`

**Functions:**

- [**estimate_gaussian_copula_corr**](#causalis.dgp.base.estimate_gaussian_copula_corr) – Estimate a Gaussian copula correlation matrix from observational data_contracts.

##### `causalis.dgp.base.estimate_gaussian_copula_corr`

```python
estimate_gaussian_copula_corr(df, cols)
```

Estimate a Gaussian copula correlation matrix from observational data_contracts.
Uses rank -> normal scores -> Pearson correlation approach.

#### `causalis.dgp.causaldata`

**Modules:**

- [**base**](#causalis.dgp.causaldata.base) –
- [**functional**](#causalis.dgp.causaldata.functional) –
- [**preperiod**](#causalis.dgp.causaldata.preperiod) –

**Classes:**

- [**CausalData**](#causalis.dgp.causaldata.CausalData) – Container for causal inference datasets.
- [**CausalDatasetGenerator**](#causalis.dgp.causaldata.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,
- [**SmokingDGP**](#causalis.dgp.causaldata.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.

**Functions:**

- [**generate_classic_rct**](#causalis.dgp.causaldata.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_classic_rct_26**](#causalis.dgp.causaldata.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.
- [**generate_obs_hte_26**](#causalis.dgp.causaldata.generate_obs_hte_26) – Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
- [**generate_rct**](#causalis.dgp.causaldata.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_cuped_tweedie**](#causalis.dgp.causaldata.make_cuped_tweedie) – Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_cuped_tweedie_26**](#causalis.dgp.causaldata.make_cuped_tweedie_26) – Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_gold_linear**](#causalis.dgp.causaldata.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_26_dataset**](#causalis.dgp.causaldata.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.
- [**obs_linear_effect**](#causalis.dgp.causaldata.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

##### `causalis.dgp.causaldata.CausalData`

Bases: <code>[BaseModel](#pydantic.BaseModel)</code>

Container for causal inference datasets.

Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
The stored DataFrame is restricted to only those columns.
Uses Pydantic for validation and as a data_contracts contract.

**Attributes:**

- [**df**](#causalis.dgp.causaldata.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
  NaN values are not allowed in the used columns.
- [**treatment_name**](#causalis.dgp.causaldata.CausalData.treatment_name) (<code>[str](#str)</code>) – Column name representing the treatment variable.
- [**outcome_name**](#causalis.dgp.causaldata.CausalData.outcome_name) (<code>[str](#str)</code>) – Column name representing the outcome variable.
- [**confounders_names**](#causalis.dgp.causaldata.CausalData.confounders_names) (<code>[List](#typing.List)\[[str](#str)\]</code>) – Names of the confounder columns (may be empty).
- [**user_id_name**](#causalis.dgp.causaldata.CausalData.user_id_name) (<code>([str](#str), [optional](#optional))</code>) – Column name representing the unique identifier for each observation/user.

**Functions:**

- [**from_df**](#causalis.dgp.causaldata.CausalData.from_df) – Friendly constructor for CausalData.
- [**get_df**](#causalis.dgp.causaldata.CausalData.get_df) – Get a DataFrame with specified columns.

###### `causalis.dgp.causaldata.CausalData.X`

```python
X: pd.DataFrame
```

Design matrix of confounders.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The DataFrame containing only confounder columns.

###### `causalis.dgp.causaldata.CausalData.confounders`

```python
confounders: List[str]
```

List of confounder column names.

**Returns:**

- <code>[List](#typing.List)\[[str](#str)\]</code> – Names of the confounder columns.

###### `causalis.dgp.causaldata.CausalData.confounders_names`

```python
confounders_names: List[str] = Field(alias='confounders', default_factory=list)
```

###### `causalis.dgp.causaldata.CausalData.df`

```python
df: pd.DataFrame
```

###### `causalis.dgp.causaldata.CausalData.from_df`

```python
from_df(df, treatment, outcome, confounders=None, user_id=None, **kwargs)
```

Friendly constructor for CausalData.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code> – A validated CausalData instance.

###### `causalis.dgp.causaldata.CausalData.get_df`

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

###### `causalis.dgp.causaldata.CausalData.model_config`

```python
model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
```

###### `causalis.dgp.causaldata.CausalData.outcome`

```python
outcome: pd.Series
```

Outcome column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The outcome column.

###### `causalis.dgp.causaldata.CausalData.outcome_name`

```python
outcome_name: str = Field(alias='outcome')
```

###### `causalis.dgp.causaldata.CausalData.treatment`

```python
treatment: pd.Series
```

Treatment column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The treatment column.

###### `causalis.dgp.causaldata.CausalData.treatment_name`

```python
treatment_name: str = Field(alias='treatment')
```

###### `causalis.dgp.causaldata.CausalData.user_id`

```python
user_id: pd.Series
```

user_id column as a Series.

**Returns:**

- <code>[Series](#pandas.Series)</code> – The user_id column.

###### `causalis.dgp.causaldata.CausalData.user_id_name`

```python
user_id_name: Optional[str] = Field(alias='user_id', default=None)
```

##### `causalis.dgp.causaldata.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, score_bounding=None, alpha_zi=-1.0, beta_zi=None, g_zi=None, u_strength_zi=0.0, tau_zi=None, pos_dist='gamma', gamma_shape=2.0, lognormal_sigma=1.0, include_oracle=True, seed=None)
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

- [**rng**](#causalis.dgp.causaldata.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.dgp.causaldata.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.causaldata.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.causaldata.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

###### `causalis.dgp.causaldata.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.alpha_zi`

```python
alpha_zi: float = -1.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.gamma_shape`

```python
gamma_shape: float = 2.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

###### `causalis.dgp.causaldata.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.k`

```python
k: int = 5
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

###### `causalis.dgp.causaldata.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.pos_dist`

```python
pos_dist: str = 'gamma'
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.score_bounding`

```python
score_bounding: Optional[float] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

###### `causalis.dgp.causaldata.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

###### `causalis.dgp.causaldata.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `causalis.dgp.causaldata.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.dgp.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.dgp.causaldata.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.causaldata.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.causaldata.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.dgp.causaldata.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.dgp.causaldata.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**alpha_zi**](#causalis.dgp.causaldata.SmokingDGP.alpha_zi) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.dgp.causaldata.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.dgp.causaldata.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_zi**](#causalis.dgp.causaldata.SmokingDGP.beta_zi) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.dgp.causaldata.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.dgp.causaldata.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.dgp.causaldata.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.dgp.causaldata.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_zi**](#causalis.dgp.causaldata.SmokingDGP.g_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**gamma_shape**](#causalis.dgp.causaldata.SmokingDGP.gamma_shape) (<code>[float](#float)</code>) –
- [**include_oracle**](#causalis.dgp.causaldata.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.dgp.causaldata.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**lognormal_sigma**](#causalis.dgp.causaldata.SmokingDGP.lognormal_sigma) (<code>[float](#float)</code>) –
- [**outcome_type**](#causalis.dgp.causaldata.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**pos_dist**](#causalis.dgp.causaldata.SmokingDGP.pos_dist) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.dgp.causaldata.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.dgp.causaldata.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**score_bounding**](#causalis.dgp.causaldata.SmokingDGP.score_bounding) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**seed**](#causalis.dgp.causaldata.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.dgp.causaldata.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.dgp.causaldata.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.dgp.causaldata.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**tau_zi**](#causalis.dgp.causaldata.SmokingDGP.tau_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.dgp.causaldata.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.dgp.causaldata.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.dgp.causaldata.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**u_strength_zi**](#causalis.dgp.causaldata.SmokingDGP.u_strength_zi) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.dgp.causaldata.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.dgp.causaldata.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

###### `causalis.dgp.causaldata.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

###### `causalis.dgp.causaldata.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

###### `causalis.dgp.causaldata.SmokingDGP.alpha_zi`

```python
alpha_zi: float = -1.0
```

###### `causalis.dgp.causaldata.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.gamma_shape`

```python
gamma_shape: float = 2.0
```

###### `causalis.dgp.causaldata.SmokingDGP.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

###### `causalis.dgp.causaldata.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

###### `causalis.dgp.causaldata.SmokingDGP.k`

```python
k: int = 5
```

###### `causalis.dgp.causaldata.SmokingDGP.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

###### `causalis.dgp.causaldata.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

###### `causalis.dgp.causaldata.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

###### `causalis.dgp.causaldata.SmokingDGP.pos_dist`

```python
pos_dist: str = 'gamma'
```

###### `causalis.dgp.causaldata.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

###### `causalis.dgp.causaldata.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

###### `causalis.dgp.causaldata.SmokingDGP.score_bounding`

```python
score_bounding: Optional[float] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

###### `causalis.dgp.causaldata.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.SmokingDGP.theta`

```python
theta: float = 1.0
```

###### `causalis.dgp.causaldata.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

###### `causalis.dgp.causaldata.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

###### `causalis.dgp.causaldata.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

###### `causalis.dgp.causaldata.SmokingDGP.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

###### `causalis.dgp.causaldata.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

###### `causalis.dgp.causaldata.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `causalis.dgp.causaldata.base`

**Classes:**

- [**CausalDatasetGenerator**](#causalis.dgp.causaldata.base.CausalDatasetGenerator) – Generate synthetic causal inference datasets with controllable confounding,

###### `causalis.dgp.causaldata.base.CausalDatasetGenerator`

```python
CausalDatasetGenerator(theta=1.0, tau=None, beta_y=None, beta_d=None, g_y=None, g_d=None, alpha_y=0.0, alpha_d=0.0, sigma_y=1.0, outcome_type='continuous', confounder_specs=None, k=5, x_sampler=None, use_copula=False, copula_corr=None, target_d_rate=None, u_strength_d=0.0, u_strength_y=0.0, propensity_sharpness=1.0, score_bounding=None, alpha_zi=-1.0, beta_zi=None, g_zi=None, u_strength_zi=0.0, tau_zi=None, pos_dist='gamma', gamma_shape=2.0, lognormal_sigma=1.0, include_oracle=True, seed=None)
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

- [**rng**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.rng) (<code>[Generator](#numpy.random.Generator)</code>) – Internal RNG seeded from `seed`.

**Functions:**

- [**generate**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.causaldata.base.CausalDatasetGenerator.to_causal_data) – Generate a dataset and convert it to a CausalData object.

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.alpha_d`

```python
alpha_d: float = 0.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.alpha_y`

```python
alpha_y: float = 0.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.alpha_zi`

```python
alpha_zi: float = -1.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.gamma_shape`

```python
gamma_shape: float = 2.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.include_oracle`

```python
include_oracle: bool = True
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.k`

```python
k: int = 5
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.outcome_type`

```python
outcome_type: str = 'continuous'
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.pos_dist`

```python
pos_dist: str = 'gamma'
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.score_bounding`

```python
score_bounding: Optional[float] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.seed`

```python
seed: Optional[int] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.sigma_y`

```python
sigma_y: float = 1.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.theta`

```python
theta: float = 1.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.u_strength_d`

```python
u_strength_d: float = 0.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.u_strength_y`

```python
u_strength_y: float = 0.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.use_copula`

```python
use_copula: bool = False
```

####### `causalis.dgp.causaldata.base.CausalDatasetGenerator.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

##### `causalis.dgp.causaldata.functional`

**Classes:**

- [**SmokingDGP**](#causalis.dgp.causaldata.functional.SmokingDGP) – A specialized generating class for smoking-related causal scenarios.

**Functions:**

- [**generate_classic_rct**](#causalis.dgp.causaldata.functional.generate_classic_rct) – Generate a classic RCT dataset with three binary confounders:
- [**generate_rct**](#causalis.dgp.causaldata.functional.generate_rct) – Generate an RCT dataset with randomized treatment assignment.
- [**make_cuped_tweedie**](#causalis.dgp.causaldata.functional.make_cuped_tweedie) – Tweedie-like DGP with mixed marginals and structured HTE.
- [**make_gold_linear**](#causalis.dgp.causaldata.functional.make_gold_linear) – A standard linear benchmark with moderate confounding.
- [**obs_linear_effect**](#causalis.dgp.causaldata.functional.obs_linear_effect) – Generate an observational dataset with linear effects of confounders and a constant treatment effect.

###### `causalis.dgp.causaldata.functional.SmokingDGP`

```python
SmokingDGP(effect_size=2.0, seed=42, **kwargs)
```

Bases: <code>[CausalDatasetGenerator](#causalis.dgp.causaldata.base.CausalDatasetGenerator)</code>

A specialized generating class for smoking-related causal scenarios.
Example of how users can extend CausalDatasetGenerator for specific domains.

**Functions:**

- [**generate**](#causalis.dgp.causaldata.functional.SmokingDGP.generate) – Draw a synthetic dataset of size `n`.
- [**oracle_nuisance**](#causalis.dgp.causaldata.functional.SmokingDGP.oracle_nuisance) – Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.
- [**to_causal_data**](#causalis.dgp.causaldata.functional.SmokingDGP.to_causal_data) – Generate a dataset and convert it to a CausalData object.

**Attributes:**

- [**alpha_d**](#causalis.dgp.causaldata.functional.SmokingDGP.alpha_d) (<code>[float](#float)</code>) –
- [**alpha_y**](#causalis.dgp.causaldata.functional.SmokingDGP.alpha_y) (<code>[float](#float)</code>) –
- [**alpha_zi**](#causalis.dgp.causaldata.functional.SmokingDGP.alpha_zi) (<code>[float](#float)</code>) –
- [**beta_d**](#causalis.dgp.causaldata.functional.SmokingDGP.beta_d) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_y**](#causalis.dgp.causaldata.functional.SmokingDGP.beta_y) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**beta_zi**](#causalis.dgp.causaldata.functional.SmokingDGP.beta_zi) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**confounder_specs**](#causalis.dgp.causaldata.functional.SmokingDGP.confounder_specs) (<code>[Optional](#typing.Optional)\[[List](#typing.List)\[[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]\]\]</code>) –
- [**copula_corr**](#causalis.dgp.causaldata.functional.SmokingDGP.copula_corr) (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray)\]</code>) –
- [**g_d**](#causalis.dgp.causaldata.functional.SmokingDGP.g_d) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_y**](#causalis.dgp.causaldata.functional.SmokingDGP.g_y) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**g_zi**](#causalis.dgp.causaldata.functional.SmokingDGP.g_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**gamma_shape**](#causalis.dgp.causaldata.functional.SmokingDGP.gamma_shape) (<code>[float](#float)</code>) –
- [**include_oracle**](#causalis.dgp.causaldata.functional.SmokingDGP.include_oracle) (<code>[bool](#bool)</code>) –
- [**k**](#causalis.dgp.causaldata.functional.SmokingDGP.k) (<code>[int](#int)</code>) –
- [**lognormal_sigma**](#causalis.dgp.causaldata.functional.SmokingDGP.lognormal_sigma) (<code>[float](#float)</code>) –
- [**outcome_type**](#causalis.dgp.causaldata.functional.SmokingDGP.outcome_type) (<code>[str](#str)</code>) –
- [**pos_dist**](#causalis.dgp.causaldata.functional.SmokingDGP.pos_dist) (<code>[str](#str)</code>) –
- [**propensity_sharpness**](#causalis.dgp.causaldata.functional.SmokingDGP.propensity_sharpness) (<code>[float](#float)</code>) –
- [**rng**](#causalis.dgp.causaldata.functional.SmokingDGP.rng) (<code>[Generator](#numpy.random.Generator)</code>) –
- [**score_bounding**](#causalis.dgp.causaldata.functional.SmokingDGP.score_bounding) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**seed**](#causalis.dgp.causaldata.functional.SmokingDGP.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –
- [**sigma_y**](#causalis.dgp.causaldata.functional.SmokingDGP.sigma_y) (<code>[float](#float)</code>) –
- [**target_d_rate**](#causalis.dgp.causaldata.functional.SmokingDGP.target_d_rate) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –
- [**tau**](#causalis.dgp.causaldata.functional.SmokingDGP.tau) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**tau_zi**](#causalis.dgp.causaldata.functional.SmokingDGP.tau_zi) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[ndarray](#numpy.ndarray)\], [ndarray](#numpy.ndarray)\]\]</code>) –
- [**theta**](#causalis.dgp.causaldata.functional.SmokingDGP.theta) (<code>[float](#float)</code>) –
- [**u_strength_d**](#causalis.dgp.causaldata.functional.SmokingDGP.u_strength_d) (<code>[float](#float)</code>) –
- [**u_strength_y**](#causalis.dgp.causaldata.functional.SmokingDGP.u_strength_y) (<code>[float](#float)</code>) –
- [**u_strength_zi**](#causalis.dgp.causaldata.functional.SmokingDGP.u_strength_zi) (<code>[float](#float)</code>) –
- [**use_copula**](#causalis.dgp.causaldata.functional.SmokingDGP.use_copula) (<code>[bool](#bool)</code>) –
- [**x_sampler**](#causalis.dgp.causaldata.functional.SmokingDGP.x_sampler) (<code>[Optional](#typing.Optional)\[[Callable](#typing.Callable)\[\[[int](#int), [int](#int), [int](#int)\], [ndarray](#numpy.ndarray)\]\]</code>) –

####### `causalis.dgp.causaldata.functional.SmokingDGP.alpha_d`

```python
alpha_d: float = 0.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.alpha_y`

```python
alpha_y: float = 0.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.alpha_zi`

```python
alpha_zi: float = -1.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.beta_d`

```python
beta_d: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.beta_y`

```python
beta_y: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.beta_zi`

```python
beta_zi: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.confounder_specs`

```python
confounder_specs: Optional[List[Dict[str, Any]]] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.copula_corr`

```python
copula_corr: Optional[np.ndarray] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.g_d`

```python
g_d: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.g_y`

```python
g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.g_zi`

```python
g_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.gamma_shape`

```python
gamma_shape: float = 2.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.generate`

```python
generate(n, U=None)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **U** (<code>[ndarray](#numpy.ndarray)</code>) – Unobserved confounder. If None, generated from N(0,1).

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The generated dataset with outcome 'y', treatment 'd', confounders,
  and oracle ground-truth columns.

####### `causalis.dgp.causaldata.functional.SmokingDGP.include_oracle`

```python
include_oracle: bool = True
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.k`

```python
k: int = 5
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.lognormal_sigma`

```python
lognormal_sigma: float = 1.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.oracle_nuisance`

```python
oracle_nuisance(num_quad=21)
```

Return nuisance functions (m(x), g0(x), g1(x)) compatible with IRM.

**Parameters:**

- **num_quad** (<code>[int](#int)</code>) – Number of quadrature points for marginalizing over U.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary of callables mapping X to nuisance values.

####### `causalis.dgp.causaldata.functional.SmokingDGP.outcome_type`

```python
outcome_type: str = 'continuous'
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.pos_dist`

```python
pos_dist: str = 'gamma'
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.propensity_sharpness`

```python
propensity_sharpness: float = 1.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.rng`

```python
rng: np.random.Generator = field(init=False, repr=False)
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.score_bounding`

```python
score_bounding: Optional[float] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.seed`

```python
seed: Optional[int] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.sigma_y`

```python
sigma_y: float = 1.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.target_d_rate`

```python
target_d_rate: Optional[float] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.tau`

```python
tau: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.tau_zi`

```python
tau_zi: Optional[Callable[[np.ndarray], np.ndarray]] = None
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.theta`

```python
theta: float = 1.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.to_causal_data`

```python
to_causal_data(n, confounders=None)
```

Generate a dataset and convert it to a CausalData object.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **confounders** (<code>str or list of str</code>) – List of confounder column names to include. If None, automatically detects numeric confounders.

**Returns:**

- <code>[CausalData](#causalis.dgp.causaldata.CausalData)</code> – A CausalData object containing the generated dataset.

####### `causalis.dgp.causaldata.functional.SmokingDGP.u_strength_d`

```python
u_strength_d: float = 0.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.u_strength_y`

```python
u_strength_y: float = 0.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.u_strength_zi`

```python
u_strength_zi: float = 0.0
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.use_copula`

```python
use_copula: bool = False
```

####### `causalis.dgp.causaldata.functional.SmokingDGP.x_sampler`

```python
x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
```

###### `causalis.dgp.causaldata.functional.generate_classic_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

###### `causalis.dgp.causaldata.functional.generate_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic RCT dataset.

###### `causalis.dgp.causaldata.functional.make_cuped_tweedie`

```python
make_cuped_tweedie(n=10000, seed=42, add_pre=True, pre_name='y_pre', pre_target_corr=0.6, pre_spec=None, include_oracle=False, return_causal_data=True, theta_log=0.1)
```

Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail. Suitable for CUPED benchmarking.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

###### `causalis.dgp.causaldata.functional.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

###### `causalis.dgp.causaldata.functional.obs_linear_effect`

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

##### `causalis.dgp.causaldata.generate_classic_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

##### `causalis.dgp.causaldata.generate_classic_rct_26`

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

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

##### `causalis.dgp.causaldata.generate_obs_hte_26`

```python
generate_obs_hte_26(n=10000, seed=42, include_oracle=True, return_causal_data=True)
```

Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
and a heterogeneous (nonlinear) treatment effect tau(X).
Based on the scenario in notebooks/cases/dml_atte.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

##### `causalis.dgp.causaldata.generate_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic RCT dataset.

##### `causalis.dgp.causaldata.make_cuped_tweedie`

```python
make_cuped_tweedie(n=10000, seed=42, add_pre=True, pre_name='y_pre', pre_target_corr=0.6, pre_spec=None, include_oracle=False, return_causal_data=True, theta_log=0.1)
```

Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail. Suitable for CUPED benchmarking.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

##### `causalis.dgp.causaldata.make_cuped_tweedie_26`

```python
make_cuped_tweedie_26(n=100000, seed=42, add_pre=True, pre_name='y_pre', pre_target_corr=0.6, pre_spec=None, include_oracle=False, return_causal_data=True, theta_log=0.2)
```

Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail.
Includes a pre-period covariate 'y_pre' by default, making it suitable for CUPED benchmarks.
Wrapper for make_tweedie().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

##### `causalis.dgp.causaldata.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

##### `causalis.dgp.causaldata.obs_linear_26_dataset`

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

##### `causalis.dgp.causaldata.obs_linear_effect`

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

##### `causalis.dgp.causaldata.preperiod`

**Classes:**

- [**PreCorrSpec**](#causalis.dgp.causaldata.preperiod.PreCorrSpec) –

**Functions:**

- [**add_preperiod_covariate**](#causalis.dgp.causaldata.preperiod.add_preperiod_covariate) – Standardized utility to add a calibrated pre-period covariate to a DataFrame.
- [**calibrate_sigma_for_target_corr**](#causalis.dgp.causaldata.preperiod.calibrate_sigma_for_target_corr) – Find sigma such that Corr(T(y_pre_base + sigma\*eps), T(y_post)) ~ target_corr.
- [**corr_on_scale**](#causalis.dgp.causaldata.preperiod.corr_on_scale) –

**Attributes:**

- [**CorrMethod**](#causalis.dgp.causaldata.preperiod.CorrMethod) –
- [**Transform**](#causalis.dgp.causaldata.preperiod.Transform) –

###### `causalis.dgp.causaldata.preperiod.CorrMethod`

```python
CorrMethod = Literal['pearson', 'spearman']
```

###### `causalis.dgp.causaldata.preperiod.PreCorrSpec`

```python
PreCorrSpec(target_corr=0.7, transform='log1p', winsor_q=0.999, method='pearson', sigma_lo=0.0, sigma_hi=50.0, sigma_tol=0.001, max_iter=40)
```

**Attributes:**

- [**max_iter**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.max_iter) (<code>[int](#int)</code>) –
- [**method**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.method) (<code>[CorrMethod](#causalis.dgp.causaldata.preperiod.CorrMethod)</code>) –
- [**sigma_hi**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.sigma_hi) (<code>[float](#float)</code>) –
- [**sigma_lo**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.sigma_lo) (<code>[float](#float)</code>) –
- [**sigma_tol**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.sigma_tol) (<code>[float](#float)</code>) –
- [**target_corr**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.target_corr) (<code>[float](#float)</code>) –
- [**transform**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.transform) (<code>[Transform](#causalis.dgp.causaldata.preperiod.Transform)</code>) –
- [**winsor_q**](#causalis.dgp.causaldata.preperiod.PreCorrSpec.winsor_q) (<code>[Optional](#typing.Optional)\[[float](#float)\]</code>) –

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.max_iter`

```python
max_iter: int = 40
```

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.method`

```python
method: CorrMethod = 'pearson'
```

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.sigma_hi`

```python
sigma_hi: float = 50.0
```

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.sigma_lo`

```python
sigma_lo: float = 0.0
```

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.sigma_tol`

```python
sigma_tol: float = 0.001
```

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.target_corr`

```python
target_corr: float = 0.7
```

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.transform`

```python
transform: Transform = 'log1p'
```

####### `causalis.dgp.causaldata.preperiod.PreCorrSpec.winsor_q`

```python
winsor_q: Optional[float] = 0.999
```

###### `causalis.dgp.causaldata.preperiod.Transform`

```python
Transform = Literal['none', 'log1p', 'rank']
```

###### `causalis.dgp.causaldata.preperiod.add_preperiod_covariate`

```python
add_preperiod_covariate(df, y_col, d_col, pre_name, base_builder, spec, rng, mask=None)
```

Standardized utility to add a calibrated pre-period covariate to a DataFrame.

**Parameters:**

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The dataset.
- **y_col** (<code>[str](#str)</code>) – Name of the outcome column.
- **d_col** (<code>[str](#str)</code>) – Name of the treatment column.
- **pre_name** (<code>[str](#str)</code>) – Name of the new pre-period covariate column.
- **base_builder** (<code>[callable](#callable)</code>) – Function df -> y_pre_base (np.ndarray) providing the shared signal.
- **spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Specification for target correlation and scale.
- **rng** (<code>[Generator](#numpy.random.Generator)</code>) – Random number generator.
- **mask** (<code>[ndarray](#numpy.ndarray)</code>) – Boolean mask of rows to use for calibration (e.g. control group).
  If None, use control group (d == 0).

###### `causalis.dgp.causaldata.preperiod.calibrate_sigma_for_target_corr`

```python
calibrate_sigma_for_target_corr(y_pre_base, y_post, rng, spec, *, noise=None)
```

Find sigma such that Corr(T(y_pre_base + sigma\*eps), T(y_post)) ~ target_corr.
Returns (sigma, achieved_corr).

###### `causalis.dgp.causaldata.preperiod.corr_on_scale`

```python
corr_on_scale(y_pre, y_post, *, transform='log1p', winsor_q=0.999, method='pearson')
```

#### `causalis.dgp.causaldata_instrumental`

**Modules:**

- [**base**](#causalis.dgp.causaldata_instrumental.base) –
- [**functional**](#causalis.dgp.causaldata_instrumental.functional) –

**Classes:**

- [**InstrumentalGenerator**](#causalis.dgp.causaldata_instrumental.InstrumentalGenerator) – Generator for synthetic causal inference datasets with instrumental variables.

**Functions:**

- [**generate_iv_data**](#causalis.dgp.causaldata_instrumental.generate_iv_data) – Generate synthetic dataset with instrumental variables.

##### `causalis.dgp.causaldata_instrumental.InstrumentalGenerator`

```python
InstrumentalGenerator(seed=None)
```

Generator for synthetic causal inference datasets with instrumental variables.

Placeholder implementation for future use.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Functions:**

- [**generate**](#causalis.dgp.causaldata_instrumental.InstrumentalGenerator.generate) – Draw a synthetic dataset of size `n`.

**Attributes:**

- [**seed**](#causalis.dgp.causaldata_instrumental.InstrumentalGenerator.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –

###### `causalis.dgp.causaldata_instrumental.InstrumentalGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – An empty DataFrame (placeholder).

###### `causalis.dgp.causaldata_instrumental.InstrumentalGenerator.seed`

```python
seed: Optional[int] = None
```

##### `causalis.dgp.causaldata_instrumental.base`

**Classes:**

- [**InstrumentalGenerator**](#causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator) – Generator for synthetic causal inference datasets with instrumental variables.

###### `causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator`

```python
InstrumentalGenerator(seed=None)
```

Generator for synthetic causal inference datasets with instrumental variables.

Placeholder implementation for future use.

**Parameters:**

- **seed** (<code>[int](#int)</code>) – Random seed for reproducibility.

**Functions:**

- [**generate**](#causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator.generate) – Draw a synthetic dataset of size `n`.

**Attributes:**

- [**seed**](#causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator.seed) (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) –

####### `causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator.generate`

```python
generate(n)
```

Draw a synthetic dataset of size `n`.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – An empty DataFrame (placeholder).

####### `causalis.dgp.causaldata_instrumental.base.InstrumentalGenerator.seed`

```python
seed: Optional[int] = None
```

##### `causalis.dgp.causaldata_instrumental.functional`

**Functions:**

- [**generate_iv_data**](#causalis.dgp.causaldata_instrumental.functional.generate_iv_data) – Generate synthetic dataset with instrumental variables.

###### `causalis.dgp.causaldata_instrumental.functional.generate_iv_data`

```python
generate_iv_data(n=1000)
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

##### `causalis.dgp.causaldata_instrumental.generate_iv_data`

```python
generate_iv_data(n=1000)
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

#### `causalis.dgp.generate_classic_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic classic RCT dataset.

#### `causalis.dgp.generate_classic_rct_26`

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

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

#### `causalis.dgp.generate_iv_data`

```python
generate_iv_data(n=1000)
```

Generate synthetic dataset with instrumental variables.

Placeholder implementation.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Synthetic IV dataset.

#### `causalis.dgp.generate_rct`

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

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> – Synthetic RCT dataset.

#### `causalis.dgp.make_cuped_tweedie`

```python
make_cuped_tweedie(n=10000, seed=42, add_pre=True, pre_name='y_pre', pre_target_corr=0.6, pre_spec=None, include_oracle=False, return_causal_data=True, theta_log=0.1)
```

Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail. Suitable for CUPED benchmarking.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
  If provided, `pre_target_corr` is ignored in favor of `pre_spec.target_corr`.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

#### `causalis.dgp.make_cuped_tweedie_26`

```python
make_cuped_tweedie_26(n=100000, seed=42, add_pre=True, pre_name='y_pre', pre_target_corr=0.6, pre_spec=None, include_oracle=False, return_causal_data=True, theta_log=0.2)
```

Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail.
Includes a pre-period covariate 'y_pre' by default, making it suitable for CUPED benchmarks.
Wrapper for make_tweedie().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

#### `causalis.dgp.make_gold_linear`

```python
make_gold_linear(n=10000, seed=42)
```

A standard linear benchmark with moderate confounding.
Based on the benchmark scenario in docs/research/dgp_benchmarking.ipynb.

#### `causalis.dgp.obs_linear_26_dataset`

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

#### `causalis.dgp.obs_linear_effect`

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

### `causalis.scenarios`

**Modules:**

- [**cate**](#causalis.scenarios.cate) –
- [**classic_rct**](#causalis.scenarios.classic_rct) –
- [**cuped**](#causalis.scenarios.cuped) –
- [**unconfoundedness**](#causalis.scenarios.unconfoundedness) –

#### `causalis.scenarios.cate`

**Modules:**

- [**blp**](#causalis.scenarios.cate.blp) –
- [**cate**](#causalis.scenarios.cate.cate) – Conditional Average Treatment Effect (CATE) inference methods for causalis.
- [**gate**](#causalis.scenarios.cate.gate) – Group Average Treatment Effect (GATE) inference methods for causalis.

**Classes:**

- [**BLP**](#causalis.scenarios.cate.BLP) – Best linear predictor (BLP) with orthogonal signals.

##### `causalis.scenarios.cate.BLP`

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

- [**confint**](#causalis.scenarios.cate.BLP.confint) – Confidence intervals for the BLP model.
- [**fit**](#causalis.scenarios.cate.BLP.fit) – Estimate BLP models.

**Attributes:**

- [**basis**](#causalis.scenarios.cate.BLP.basis) – Basis.
- [**blp_model**](#causalis.scenarios.cate.BLP.blp_model) – Best-Linear-Predictor model.
- [**blp_omega**](#causalis.scenarios.cate.BLP.blp_omega) – Covariance matrix.
- [**orth_signal**](#causalis.scenarios.cate.BLP.orth_signal) – Orthogonal signal.
- [**summary**](#causalis.scenarios.cate.BLP.summary) – A summary for the best linear predictor effect after calling :meth:`fit`.

###### `causalis.scenarios.cate.BLP.basis`

```python
basis
```

Basis.

###### `causalis.scenarios.cate.BLP.blp_model`

```python
blp_model
```

Best-Linear-Predictor model.

###### `causalis.scenarios.cate.BLP.blp_omega`

```python
blp_omega
```

Covariance matrix.

###### `causalis.scenarios.cate.BLP.confint`

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

- **df_ci** (<code>[DataFrame](#pandas.DataFrame)</code>) – A data_contracts frame with the confidence interval(s).

###### `causalis.scenarios.cate.BLP.fit`

```python
fit(cov_type='HC0', diagnostic_data=True, **kwargs)
```

Estimate BLP models.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – The covariance type to be used in the estimation. Default is `'HC0'`.
  See :meth:`statsmodels.regression.linear_model.OLS.fit` for more information.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts. (Currently not used for BLP).
- \*\***kwargs** – Additional keyword arguments to be passed to :meth:`statsmodels.regression.linear_model.OLS.fit`.

**Returns:**

- **self** (<code>[object](#object)</code>) –

###### `causalis.scenarios.cate.BLP.orth_signal`

```python
orth_signal
```

Orthogonal signal.

###### `causalis.scenarios.cate.BLP.summary`

```python
summary
```

A summary for the best linear predictor effect after calling :meth:`fit`.

##### `causalis.scenarios.cate.blp`

**Classes:**

- [**BLP**](#causalis.scenarios.cate.blp.BLP) – Best linear predictor (BLP) with orthogonal signals.

###### `causalis.scenarios.cate.blp.BLP`

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

- [**confint**](#causalis.scenarios.cate.blp.BLP.confint) – Confidence intervals for the BLP model.
- [**fit**](#causalis.scenarios.cate.blp.BLP.fit) – Estimate BLP models.

**Attributes:**

- [**basis**](#causalis.scenarios.cate.blp.BLP.basis) – Basis.
- [**blp_model**](#causalis.scenarios.cate.blp.BLP.blp_model) – Best-Linear-Predictor model.
- [**blp_omega**](#causalis.scenarios.cate.blp.BLP.blp_omega) – Covariance matrix.
- [**orth_signal**](#causalis.scenarios.cate.blp.BLP.orth_signal) – Orthogonal signal.
- [**summary**](#causalis.scenarios.cate.blp.BLP.summary) – A summary for the best linear predictor effect after calling :meth:`fit`.

####### `causalis.scenarios.cate.blp.BLP.basis`

```python
basis
```

Basis.

####### `causalis.scenarios.cate.blp.BLP.blp_model`

```python
blp_model
```

Best-Linear-Predictor model.

####### `causalis.scenarios.cate.blp.BLP.blp_omega`

```python
blp_omega
```

Covariance matrix.

####### `causalis.scenarios.cate.blp.BLP.confint`

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

- **df_ci** (<code>[DataFrame](#pandas.DataFrame)</code>) – A data_contracts frame with the confidence interval(s).

####### `causalis.scenarios.cate.blp.BLP.fit`

```python
fit(cov_type='HC0', diagnostic_data=True, **kwargs)
```

Estimate BLP models.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – The covariance type to be used in the estimation. Default is `'HC0'`.
  See :meth:`statsmodels.regression.linear_model.OLS.fit` for more information.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts. (Currently not used for BLP).
- \*\***kwargs** – Additional keyword arguments to be passed to :meth:`statsmodels.regression.linear_model.OLS.fit`.

**Returns:**

- **self** (<code>[object](#object)</code>) –

####### `causalis.scenarios.cate.blp.BLP.orth_signal`

```python
orth_signal
```

Orthogonal signal.

####### `causalis.scenarios.cate.blp.BLP.summary`

```python
summary
```

A summary for the best linear predictor effect after calling :meth:`fit`.

##### `causalis.scenarios.cate.cate`

Conditional Average Treatment Effect (CATE) inference methods for causalis.

This submodule provides methods for estimating conditional average treatment effects.

**Modules:**

- [**cate_esimand**](#causalis.scenarios.cate.cate.cate_esimand) – DoubleML implementation for estimating CATE (per-observation orthogonal signals).

###### `causalis.scenarios.cate.cate.cate_esimand`

DoubleML implementation for estimating CATE (per-observation orthogonal signals).

This module provides a function that, given a CausalData object, fits a DoubleML IRM
model and augments the data_contracts with a new column 'cate' that contains the orthogonal
signals (an estimate of the conditional average treatment effect for each unit).

**Functions:**

- [**cate_esimand**](#causalis.scenarios.cate.cate.cate_esimand.cate_esimand) – Estimate per-observation CATEs using DoubleML IRM and return a DataFrame with a new 'cate' column.

####### `causalis.scenarios.cate.cate.cate_esimand.cate_esimand`

```python
cate_esimand(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, use_blp=False, X_new=None)
```

Estimate per-observation CATEs using DoubleML IRM and return a DataFrame with a new 'cate' column.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – A CausalData object with defined outcome (outcome), treatment (binary 0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – ML learner for outcome regression g(D, X) = E[Y | D, X] supporting fit/predict.
  Defaults to CatBoostRegressor if None.
- **ml_m** (<code>[classifier](#classifier)</code>) – ML learner for propensity m(X) = P[D=1 | X] supporting fit/predict_proba.
  Defaults to CatBoostClassifier if None.
- **n_folds** (<code>[int](#int)</code>) – Number of folds for cross-fitting.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions for sample splitting.
- **use_blp** (<code>[bool](#bool)</code>) – If True, and X_new is provided, returns cate from obj.blp_predict(X_new) aligned to X_new.
  If False (default), uses obj.\_orthogonal_signals (in-sample estimates) and appends to data_contracts.
- **X_new** (<code>[DataFrame](#pandas.DataFrame)</code>) – New covariate matrix for out-of-sample CATE prediction via best linear predictor.
  Must contain the same feature columns as the confounders in `data_contracts`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – If use_blp is False: returns a copy of data_contracts.df with a new column 'cate'.
  If use_blp is True and X_new is provided: returns a DataFrame with 'cate' column for X_new rows.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment is not binary 0/1 or required metadata is missing.

##### `causalis.scenarios.cate.gate`

Group Average Treatment Effect (GATE) inference methods for causalis.

This submodule provides methods for estimating group average treatment effects.

**Modules:**

- [**gate_esimand**](#causalis.scenarios.cate.gate.gate_esimand) – Group Average Treatment Effect (GATE) estimation using local DML IRM and BLP.

###### `causalis.scenarios.cate.gate.gate_esimand`

Group Average Treatment Effect (GATE) estimation using local DML IRM and BLP.

**Functions:**

- [**gate_esimand**](#causalis.scenarios.cate.gate.gate_esimand.gate_esimand) – Estimate Group Average Treatment Effects (GATEs).

####### `causalis.scenarios.cate.gate.gate_esimand.gate_esimand`

```python
gate_esimand(data, groups=None, n_groups=5, ml_g=None, ml_m=None, n_folds=5, n_rep=1, alpha=0.05)
```

Estimate Group Average Treatment Effects (GATEs).

If `groups` is None, observations are grouped by quantiles of the
plugin CATE proxy (g1_hat - g0_hat).

#### `causalis.scenarios.classic_rct`

**Modules:**

- [**bootstrap_diff_in_means**](#causalis.scenarios.classic_rct.bootstrap_diff_in_means) – Bootstrap difference-in-means inference.
- [**conversion_ztest**](#causalis.scenarios.classic_rct.conversion_ztest) – Two-proportion z-test
- [**dgp**](#causalis.scenarios.classic_rct.dgp) –
- [**diff_in_means**](#causalis.scenarios.classic_rct.diff_in_means) –
- [**rct_design**](#causalis.scenarios.classic_rct.rct_design) – Design module for experimental rct_design utilities.
- [**ttest**](#causalis.scenarios.classic_rct.ttest) – T-test inference for Diff_in_Means model

**Classes:**

- [**DiffInMeans**](#causalis.scenarios.classic_rct.DiffInMeans) – Difference-in-means model for CausalData.
- [**SRMResult**](#causalis.scenarios.classic_rct.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**bootstrap_diff_means**](#causalis.scenarios.classic_rct.bootstrap_diff_means) – Bootstrap inference for difference in means between treated and control groups.
- [**check_srm**](#causalis.scenarios.classic_rct.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.
- [**conversion_z_test**](#causalis.scenarios.classic_rct.conversion_z_test) – Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

##### `causalis.scenarios.classic_rct.DiffInMeans`

```python
DiffInMeans()
```

Difference-in-means model for CausalData.
Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.

**Functions:**

- [**estimate**](#causalis.scenarios.classic_rct.DiffInMeans.estimate) – Compute the treatment effect using the specified method.
- [**fit**](#causalis.scenarios.classic_rct.DiffInMeans.fit) – Fit the model by storing the CausalData object.

**Attributes:**

- [**data**](#causalis.scenarios.classic_rct.DiffInMeans.data) (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.dgp.causaldata.CausalData)\]</code>) –

###### `causalis.scenarios.classic_rct.DiffInMeans.data`

```python
data: Optional[CausalData] = None
```

###### `causalis.scenarios.classic_rct.DiffInMeans.estimate`

```python
estimate(method='ttest', alpha=0.05, diagnostic_data=True, **kwargs)
```

Compute the treatment effect using the specified method.

**Parameters:**

- **method** (<code>('ttest', 'bootstrap', 'conversion_ztest')</code>) – The inference method to use.
- "ttest": Standard independent two-sample t-test.
- "bootstrap": Bootstrap-based inference for difference in means.
- "conversion_ztest": Two-proportion z-test for binary outcomes.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the underlying inference function.
- For "bootstrap": can pass `n_simul` (default 10000).

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

###### `causalis.scenarios.classic_rct.DiffInMeans.fit`

```python
fit(data)
```

Fit the model by storing the CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

**Returns:**

- <code>[DiffInMeans](#causalis.scenarios.classic_rct.diff_in_means.DiffInMeans)</code> – The fitted model.

##### `causalis.scenarios.classic_rct.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.scenarios.classic_rct.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.scenarios.classic_rct.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.scenarios.classic_rct.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.scenarios.classic_rct.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.scenarios.classic_rct.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.scenarios.classic_rct.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.scenarios.classic_rct.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.scenarios.classic_rct.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `causalis.scenarios.classic_rct.SRMResult.alpha`

```python
alpha: float
```

###### `causalis.scenarios.classic_rct.SRMResult.chi2`

```python
chi2: float
```

###### `causalis.scenarios.classic_rct.SRMResult.df`

```python
df: int
```

###### `causalis.scenarios.classic_rct.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

###### `causalis.scenarios.classic_rct.SRMResult.is_srm`

```python
is_srm: bool
```

###### `causalis.scenarios.classic_rct.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

###### `causalis.scenarios.classic_rct.SRMResult.p_value`

```python
p_value: float
```

###### `causalis.scenarios.classic_rct.SRMResult.warning`

```python
warning: str | None = None
```

##### `causalis.scenarios.classic_rct.bootstrap_diff_in_means`

Bootstrap difference-in-means inference.

This module computes the ATE-style difference in means (treated - control) and provides:

- Two-sided p-value using a normal approximation with bootstrap standard error.
- Percentile confidence interval for the absolute difference.
- Relative difference (%) and corresponding CI relative to control mean.

**Functions:**

- [**bootstrap_diff_means**](#causalis.scenarios.classic_rct.bootstrap_diff_in_means.bootstrap_diff_means) – Bootstrap inference for difference in means between treated and control groups.

###### `causalis.scenarios.classic_rct.bootstrap_diff_in_means.bootstrap_diff_means`

```python
bootstrap_diff_means(data, alpha=0.05, n_simul=10000)
```

Bootstrap inference for difference in means between treated and control groups.

This function computes the ATE-style difference in means (treated - control)
and provides a two-sided p-value using a normal approximation with bootstrap
standard error, a percentile confidence interval for the absolute difference,
and relative difference with its corresponding confidence interval.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value using normal approximation.
- absolute_difference: The absolute difference (treated - control).
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
- relative_difference: The relative difference (%) relative to control mean.
- relative_ci: Tuple of (lower, upper) bounds for the relative difference CI.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

##### `causalis.scenarios.classic_rct.bootstrap_diff_means`

```python
bootstrap_diff_means(data, alpha=0.05, n_simul=10000)
```

Bootstrap inference for difference in means between treated and control groups.

This function computes the ATE-style difference in means (treated - control)
and provides a two-sided p-value using a normal approximation with bootstrap
standard error, a percentile confidence interval for the absolute difference,
and relative difference with its corresponding confidence interval.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).
- **n_simul** (<code>[int](#int)</code>) – Number of bootstrap resamples.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: Two-sided p-value using normal approximation.
- absolute_difference: The absolute difference (treated - control).
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
- relative_difference: The relative difference (%) relative to control mean.
- relative_ci: Tuple of (lower, upper) bounds for the relative difference CI.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid, treatment is not binary, or groups are empty.

##### `causalis.scenarios.classic_rct.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
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

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

##### `causalis.scenarios.classic_rct.conversion_z_test`

```python
conversion_z_test(data, alpha=0.05, ci_method='newcombe', se_for_test='pooled')
```

Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
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

##### `causalis.scenarios.classic_rct.conversion_ztest`

Two-proportion z-test

Compares conversion rates between treated (D=1) and control (D=0) groups.
Returns p-value, absolute/relative differences, and their confidence intervals

**Functions:**

- [**conversion_z_test**](#causalis.scenarios.classic_rct.conversion_ztest.conversion_z_test) – Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

###### `causalis.scenarios.classic_rct.conversion_ztest.conversion_z_test`

```python
conversion_z_test(data, alpha=0.05, ci_method='newcombe', se_for_test='pooled')
```

Perform a two-proportion z-test on a CausalData object with a binary outcome (conversion).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
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

##### `causalis.scenarios.classic_rct.dgp`

**Functions:**

- [**generate_classic_rct_26**](#causalis.scenarios.classic_rct.dgp.generate_classic_rct_26) – A pre-configured classic RCT dataset with 3 binary confounders.

###### `causalis.scenarios.classic_rct.dgp.generate_classic_rct_26`

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

- <code>[CausalData](#causalis.dgp.causaldata.CausalData) or [DataFrame](#pandas.DataFrame)</code> –

##### `causalis.scenarios.classic_rct.diff_in_means`

**Classes:**

- [**DiffInMeans**](#causalis.scenarios.classic_rct.diff_in_means.DiffInMeans) – Difference-in-means model for CausalData.

###### `causalis.scenarios.classic_rct.diff_in_means.DiffInMeans`

```python
DiffInMeans()
```

Difference-in-means model for CausalData.
Wraps common RCT inference methods: t-test, bootstrap, and conversion z-test.

**Functions:**

- [**estimate**](#causalis.scenarios.classic_rct.diff_in_means.DiffInMeans.estimate) – Compute the treatment effect using the specified method.
- [**fit**](#causalis.scenarios.classic_rct.diff_in_means.DiffInMeans.fit) – Fit the model by storing the CausalData object.

**Attributes:**

- [**data**](#causalis.scenarios.classic_rct.diff_in_means.DiffInMeans.data) (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.dgp.causaldata.CausalData)\]</code>) –

####### `causalis.scenarios.classic_rct.diff_in_means.DiffInMeans.data`

```python
data: Optional[CausalData] = None
```

####### `causalis.scenarios.classic_rct.diff_in_means.DiffInMeans.estimate`

```python
estimate(method='ttest', alpha=0.05, diagnostic_data=True, **kwargs)
```

Compute the treatment effect using the specified method.

**Parameters:**

- **method** (<code>('ttest', 'bootstrap', 'conversion_ztest')</code>) – The inference method to use.
- "ttest": Standard independent two-sample t-test.
- "bootstrap": Bootstrap-based inference for difference in means.
- "conversion_ztest": Two-proportion z-test for binary outcomes.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the underlying inference function.
- For "bootstrap": can pass `n_simul` (default 10000).

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

####### `causalis.scenarios.classic_rct.diff_in_means.DiffInMeans.fit`

```python
fit(data)
```

Fit the model by storing the CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

**Returns:**

- <code>[DiffInMeans](#causalis.scenarios.classic_rct.diff_in_means.DiffInMeans)</code> – The fitted model.

##### `causalis.scenarios.classic_rct.rct_design`

Design module for experimental rct_design utilities.

**Classes:**

- [**SRMResult**](#causalis.scenarios.classic_rct.rct_design.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**assign_variants_df**](#causalis.scenarios.classic_rct.rct_design.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.
- [**calculate_mde**](#causalis.scenarios.classic_rct.rct_design.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.
- [**check_srm**](#causalis.scenarios.classic_rct.rct_design.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

###### `causalis.scenarios.classic_rct.rct_design.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.scenarios.classic_rct.rct_design.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.scenarios.classic_rct.rct_design.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.scenarios.classic_rct.rct_design.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.scenarios.classic_rct.rct_design.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.scenarios.classic_rct.rct_design.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.scenarios.classic_rct.rct_design.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.scenarios.classic_rct.rct_design.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.scenarios.classic_rct.rct_design.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.alpha`

```python
alpha: float
```

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.chi2`

```python
chi2: float
```

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.df`

```python
df: int
```

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.is_srm`

```python
is_srm: bool
```

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.p_value`

```python
p_value: float
```

####### `causalis.scenarios.classic_rct.rct_design.SRMResult.warning`

```python
warning: str | None = None
```

###### `causalis.scenarios.classic_rct.rct_design.assign_variants_df`

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

###### `causalis.scenarios.classic_rct.rct_design.calculate_mde`

```python
calculate_mde(sample_size, baseline_rate=None, variance=None, alpha=0.05, power=0.8, data_type='conversion', ratio=0.5)
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data_contracts) or baseline mean (for continuous data_contracts).
  Required for conversion data_contracts.
- **variance** (<code>float or tuple of float</code>) – Variance of the data_contracts. For conversion data_contracts, this is calculated from the baseline rate if not provided.
  For continuous data_contracts, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data_contracts. Either 'conversion' for binary/conversion data_contracts or 'continuous' for continuous data_contracts.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data_contracts with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data_contracts with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

###### `causalis.scenarios.classic_rct.rct_design.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
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

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

##### `causalis.scenarios.classic_rct.ttest`

T-test inference for Diff_in_Means model

**Functions:**

- [**ttest**](#causalis.scenarios.classic_rct.ttest.ttest) – Perform a t-test to compare the outcome between treated and control groups.

###### `causalis.scenarios.classic_rct.ttest.ttest`

```python
ttest(data, alpha=0.05)
```

Perform a t-test to compare the outcome between treated and control groups.

This function performs an independent two-sample t-test (Welch's t-test)
on a CausalData object to compare the outcome variable between treated (D=1)
and control (D=0) groups. It returns the p-value, absolute and relative
differences, and their corresponding confidence intervals.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.
- **alpha** (<code>[float](#float)</code>) – The significance level for calculating confidence intervals (between 0 and 1).

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- p_value: The p-value from the t-test.
- absolute_difference: The absolute difference between treatment and control means.
- absolute_ci: Tuple of (lower, upper) bounds for the absolute difference CI.
- relative_difference: The relative difference (percentage change) between means.
- relative_ci: Tuple of (lower, upper) bounds for the relative difference CI.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If the CausalData object doesn't have both treatment and outcome variables
  defined, or if the treatment variable is not binary.

#### `causalis.scenarios.cuped`

**Modules:**

- [**cuped**](#causalis.scenarios.cuped.cuped) –
- [**dgp**](#causalis.scenarios.cuped.dgp) –

**Classes:**

- [**CUPEDModel**](#causalis.scenarios.cuped.CUPEDModel) – CUPED estimator for ATE/ITT in randomized experiments.

##### `causalis.scenarios.cuped.CUPEDModel`

```python
CUPEDModel(cov_type='HC3', alpha=0.05, strict_binary_treatment=True, use_t=True)
```

CUPED estimator for ATE/ITT in randomized experiments.

Fits an outcome regression with pre-treatment covariates (always centered)
using the Lin (2013) fully interacted adjustment:

```
Y ~ 1 + D + X^c + D * X^c
```

The reported effect is the coefficient on D, with robust covariance as requested.
This specification ensures the coefficient on D is the ATE/ITT even if the
treatment effect is heterogeneous with respect to covariates.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
  Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
- **alpha** (<code>[float](#float)</code>) – Significance level for confidence intervals.
- **strict_binary_treatment** (<code>[bool](#bool)</code>) – If True, require treatment to be binary {0,1}.
- **use_t** (<code>[bool](#bool)</code>) – Passed to statsmodels `.fit(..., use_t=use_t)`. If False, inference is based on
  normal approximation (common asymptotic choice for robust covariances).

<details class="note" open markdown="1">
<summary>Notes</summary>

- Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
- The Lin (2013) specification is recommended as a robust regression-adjustment default
  in RCTs.

</details>

**Functions:**

- [**estimate**](#causalis.scenarios.cuped.CUPEDModel.estimate) – Return the adjusted ATE/ITT estimate and inference.
- [**fit**](#causalis.scenarios.cuped.CUPEDModel.fit) – Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.
- [**summary_dict**](#causalis.scenarios.cuped.CUPEDModel.summary_dict) – Convenience JSON/logging output.

**Attributes:**

- [**adjustment**](#causalis.scenarios.cuped.CUPEDModel.adjustment) (<code>[Literal](#typing.Literal)['lin']</code>) –
- [**alpha**](#causalis.scenarios.cuped.CUPEDModel.alpha) –
- [**center_covariates**](#causalis.scenarios.cuped.CUPEDModel.center_covariates) –
- [**cov_type**](#causalis.scenarios.cuped.CUPEDModel.cov_type) –
- [**strict_binary_treatment**](#causalis.scenarios.cuped.CUPEDModel.strict_binary_treatment) –
- [**use_t**](#causalis.scenarios.cuped.CUPEDModel.use_t) –

###### `causalis.scenarios.cuped.CUPEDModel.adjustment`

```python
adjustment: Literal['lin'] = 'lin'
```

###### `causalis.scenarios.cuped.CUPEDModel.alpha`

```python
alpha = float(alpha)
```

###### `causalis.scenarios.cuped.CUPEDModel.center_covariates`

```python
center_covariates = True
```

###### `causalis.scenarios.cuped.CUPEDModel.cov_type`

```python
cov_type = str(cov_type)
```

###### `causalis.scenarios.cuped.CUPEDModel.estimate`

```python
estimate(alpha=None, diagnostic_data=True)
```

Return the adjusted ATE/ITT estimate and inference.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

###### `causalis.scenarios.cuped.CUPEDModel.fit`

```python
fit(data, covariates=None)
```

Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
- **covariates** (<code>sequence of str</code>) – Subset of `data_contracts.confounders_names` to use as CUPED covariates.
  If None, uses all confounders from the object.

**Returns:**

- <code>[CUPEDModel](#causalis.scenarios.cuped.cuped.CUPEDModel)</code> – Fitted estimator.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If requested covariates are missing, not in `data_contracts.confounders_names`,
  or treatment is not binary when `strict_binary_treatment=True`.

###### `causalis.scenarios.cuped.CUPEDModel.strict_binary_treatment`

```python
strict_binary_treatment = bool(strict_binary_treatment)
```

###### `causalis.scenarios.cuped.CUPEDModel.summary_dict`

```python
summary_dict(alpha=None)
```

Convenience JSON/logging output.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with estimates, inference, and diagnostics.

###### `causalis.scenarios.cuped.CUPEDModel.use_t`

```python
use_t = bool(use_t)
```

##### `causalis.scenarios.cuped.cuped`

**Classes:**

- [**CUPEDModel**](#causalis.scenarios.cuped.cuped.CUPEDModel) – CUPED estimator for ATE/ITT in randomized experiments.

###### `causalis.scenarios.cuped.cuped.CUPEDModel`

```python
CUPEDModel(cov_type='HC3', alpha=0.05, strict_binary_treatment=True, use_t=True)
```

CUPED estimator for ATE/ITT in randomized experiments.

Fits an outcome regression with pre-treatment covariates (always centered)
using the Lin (2013) fully interacted adjustment:

```
Y ~ 1 + D + X^c + D * X^c
```

The reported effect is the coefficient on D, with robust covariance as requested.
This specification ensures the coefficient on D is the ATE/ITT even if the
treatment effect is heterogeneous with respect to covariates.

**Parameters:**

- **cov_type** (<code>[str](#str)</code>) – Covariance estimator passed to statsmodels (e.g., "nonrobust", "HC0", "HC1", "HC2", "HC3").
  Note: for cluster-randomized designs, use cluster-robust SEs (not implemented here).
- **alpha** (<code>[float](#float)</code>) – Significance level for confidence intervals.
- **strict_binary_treatment** (<code>[bool](#bool)</code>) – If True, require treatment to be binary {0,1}.
- **use_t** (<code>[bool](#bool)</code>) – Passed to statsmodels `.fit(..., use_t=use_t)`. If False, inference is based on
  normal approximation (common asymptotic choice for robust covariances).

<details class="note" open markdown="1">
<summary>Notes</summary>

- Validity requires covariates be pre-treatment. Post-treatment covariates can bias estimates.
- The Lin (2013) specification is recommended as a robust regression-adjustment default
  in RCTs.

</details>

**Functions:**

- [**estimate**](#causalis.scenarios.cuped.cuped.CUPEDModel.estimate) – Return the adjusted ATE/ITT estimate and inference.
- [**fit**](#causalis.scenarios.cuped.cuped.CUPEDModel.fit) – Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.
- [**summary_dict**](#causalis.scenarios.cuped.cuped.CUPEDModel.summary_dict) – Convenience JSON/logging output.

**Attributes:**

- [**adjustment**](#causalis.scenarios.cuped.cuped.CUPEDModel.adjustment) (<code>[Literal](#typing.Literal)['lin']</code>) –
- [**alpha**](#causalis.scenarios.cuped.cuped.CUPEDModel.alpha) –
- [**center_covariates**](#causalis.scenarios.cuped.cuped.CUPEDModel.center_covariates) –
- [**cov_type**](#causalis.scenarios.cuped.cuped.CUPEDModel.cov_type) –
- [**strict_binary_treatment**](#causalis.scenarios.cuped.cuped.CUPEDModel.strict_binary_treatment) –
- [**use_t**](#causalis.scenarios.cuped.cuped.CUPEDModel.use_t) –

####### `causalis.scenarios.cuped.cuped.CUPEDModel.adjustment`

```python
adjustment: Literal['lin'] = 'lin'
```

####### `causalis.scenarios.cuped.cuped.CUPEDModel.alpha`

```python
alpha = float(alpha)
```

####### `causalis.scenarios.cuped.cuped.CUPEDModel.center_covariates`

```python
center_covariates = True
```

####### `causalis.scenarios.cuped.cuped.CUPEDModel.cov_type`

```python
cov_type = str(cov_type)
```

####### `causalis.scenarios.cuped.cuped.CUPEDModel.estimate`

```python
estimate(alpha=None, diagnostic_data=True)
```

Return the adjusted ATE/ITT estimate and inference.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – A results object containing effect estimates and inference.

####### `causalis.scenarios.cuped.cuped.CUPEDModel.fit`

```python
fit(data, covariates=None)
```

Fit CUPED/ANCOVA (or Lin-interacted) on a CausalData object.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Validated dataset with columns: outcome (post), treatment, and confounders (pre covariates).
- **covariates** (<code>sequence of str</code>) – Subset of `data_contracts.confounders_names` to use as CUPED covariates.
  If None, uses all confounders from the object.

**Returns:**

- <code>[CUPEDModel](#causalis.scenarios.cuped.cuped.CUPEDModel)</code> – Fitted estimator.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If requested covariates are missing, not in `data_contracts.confounders_names`,
  or treatment is not binary when `strict_binary_treatment=True`.

####### `causalis.scenarios.cuped.cuped.CUPEDModel.strict_binary_treatment`

```python
strict_binary_treatment = bool(strict_binary_treatment)
```

####### `causalis.scenarios.cuped.cuped.CUPEDModel.summary_dict`

```python
summary_dict(alpha=None)
```

Convenience JSON/logging output.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Override the instance significance level for confidence intervals.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with estimates, inference, and diagnostics.

####### `causalis.scenarios.cuped.cuped.CUPEDModel.use_t`

```python
use_t = bool(use_t)
```

##### `causalis.scenarios.cuped.dgp`

**Functions:**

- [**make_cuped_tweedie_26**](#causalis.scenarios.cuped.dgp.make_cuped_tweedie_26) – Gold standard Tweedie-like DGP with mixed marginals and structured HTE.

###### `causalis.scenarios.cuped.dgp.make_cuped_tweedie_26`

```python
make_cuped_tweedie_26(n=100000, seed=42, add_pre=True, pre_name='y_pre', pre_target_corr=0.6, pre_spec=None, include_oracle=False, return_causal_data=True, theta_log=0.2)
```

Gold standard Tweedie-like DGP with mixed marginals and structured HTE.
Features many zeros and a heavy right tail.
Includes a pre-period covariate 'y_pre' by default, making it suitable for CUPED benchmarks.
Wrapper for make_tweedie().

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples to generate.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **add_pre** (<code>[bool](#bool)</code>) – Whether to add a pre-period covariate 'y_pre'.
- **pre_name** (<code>[str](#str)</code>) – Name of the pre-period covariate column.
- **pre_target_corr** (<code>[float](#float)</code>) – Target correlation between y_pre and post-outcome y in control group.
- **pre_spec** (<code>[PreCorrSpec](#causalis.dgp.causaldata.preperiod.PreCorrSpec)</code>) – Detailed specification for pre-period calibration (transform, method, etc.).
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – Whether to return a CausalData object.
- **theta_log** (<code>[float](#float)</code>) – The log-uplift theta parameter for the treatment effect.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame) or [CausalData](#causalis.dgp.causaldata.CausalData)</code> –

#### `causalis.scenarios.unconfoundedness`

**Modules:**

- [**cate**](#causalis.scenarios.unconfoundedness.cate) – Conditional Average Treatment Effect (CATE) inference methods for causalis.
- [**dgp**](#causalis.scenarios.unconfoundedness.dgp) –
- [**dml_source**](#causalis.scenarios.unconfoundedness.dml_source) – DoubleML implementation for estimating average treatment effects.
- [**gate**](#causalis.scenarios.unconfoundedness.gate) – Group Average Treatment Effect (GATE) inference methods for causalis.
- [**irm**](#causalis.scenarios.unconfoundedness.irm) – DML IRM estimator consuming CausalData.
- [**refutation**](#causalis.scenarios.unconfoundedness.refutation) – Refutation and robustness utilities for Causalis.

**Classes:**

- [**IRM**](#causalis.scenarios.unconfoundedness.IRM) – Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

**Functions:**

- [**dml_ate_source**](#causalis.scenarios.unconfoundedness.dml_ate_source) – Estimate average treatment effects using DoubleML's interactive regression model (IRM).
- [**dml_atte_source**](#causalis.scenarios.unconfoundedness.dml_atte_source) – Estimate average treatment effects on the treated using DoubleML's interactive regression model (IRM).

##### `causalis.scenarios.unconfoundedness.IRM`

```python
IRM(data=None, ml_g=None, ml_m=None, *, n_folds=5, n_rep=1, normalize_ipw=False, trimming_rule='truncate', trimming_threshold=0.01, weights=None, random_state=None)
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Data container with outcome, binary treatment (0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
- **n_folds** (<code>[int](#int)</code>) – Number of cross-fitting folds.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions of sample splitting. Currently only 1 is supported.
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score.
- **trimming_rule** (<code>'truncate'</code>) – Trimming approach for propensity scores.
- **trimming_threshold** (<code>[float](#float)</code>) – Threshold for trimming if rule is "truncate".
- **weights** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray) or [Dict](#typing.Dict)\]</code>) – Optional weights.
- If array of shape (n,), used as ATE weights (w). Assumed E[w|X] = w.
- If dict, can contain 'weights' (w) and 'weights_bar' (E[w|X]).
- For ATTE, computed internally (w=D/P(D=1), w_bar=m(X)/P(D=1)).
  Note: If weights depend on treatment or outcome, E[w|X] must be provided for correct sensitivity analysis.
- **random_state** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Random seed for fold creation.

**Functions:**

- [**confint**](#causalis.scenarios.unconfoundedness.IRM.confint) – Compute confidence intervals for the estimated coefficient.
- [**estimate**](#causalis.scenarios.unconfoundedness.IRM.estimate) – Compute treatment effects using stored nuisance predictions.
- [**fit**](#causalis.scenarios.unconfoundedness.IRM.fit) – Fit nuisance models via cross-fitting.
- [**gate**](#causalis.scenarios.unconfoundedness.IRM.gate) – Estimate Group Average Treatment Effects via BLP on orthogonal signal.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.IRM.sensitivity_analysis) – Compute a sensitivity analysis following DoubleML (Chernozhukov et al., 2022).

**Attributes:**

- [**coef**](#causalis.scenarios.unconfoundedness.IRM.coef) (<code>[ndarray](#numpy.ndarray)</code>) – Return the estimated coefficient.
- [**data**](#causalis.scenarios.unconfoundedness.IRM.data) –
- [**diagnostics\_**](#causalis.scenarios.unconfoundedness.IRM.diagnostics_) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Return diagnostic data.
- [**ml_g**](#causalis.scenarios.unconfoundedness.IRM.ml_g) –
- [**ml_m**](#causalis.scenarios.unconfoundedness.IRM.ml_m) –
- [**n_folds**](#causalis.scenarios.unconfoundedness.IRM.n_folds) –
- [**n_rep**](#causalis.scenarios.unconfoundedness.IRM.n_rep) –
- [**normalize_ipw**](#causalis.scenarios.unconfoundedness.IRM.normalize_ipw) –
- [**orth_signal**](#causalis.scenarios.unconfoundedness.IRM.orth_signal) (<code>[ndarray](#numpy.ndarray)</code>) – Return the cross-fitted orthogonal signal (psi_b).
- [**pvalues**](#causalis.scenarios.unconfoundedness.IRM.pvalues) (<code>[ndarray](#numpy.ndarray)</code>) – Return the p-values for the estimate.
- [**random_state**](#causalis.scenarios.unconfoundedness.IRM.random_state) –
- [**score**](#causalis.scenarios.unconfoundedness.IRM.score) –
- [**se**](#causalis.scenarios.unconfoundedness.IRM.se) (<code>[ndarray](#numpy.ndarray)</code>) – Return the standard error of the estimate.
- [**summary**](#causalis.scenarios.unconfoundedness.IRM.summary) (<code>[DataFrame](#pandas.DataFrame)</code>) – Return a summary DataFrame of the results.
- [**trimming_rule**](#causalis.scenarios.unconfoundedness.IRM.trimming_rule) –
- [**trimming_threshold**](#causalis.scenarios.unconfoundedness.IRM.trimming_threshold) –
- [**weights**](#causalis.scenarios.unconfoundedness.IRM.weights) –

###### `causalis.scenarios.unconfoundedness.IRM.coef`

```python
coef: np.ndarray
```

Return the estimated coefficient.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The estimated coefficient.

###### `causalis.scenarios.unconfoundedness.IRM.confint`

```python
confint(alpha=0.05)
```

Compute confidence intervals for the estimated coefficient.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Significance level.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with confidence intervals.

###### `causalis.scenarios.unconfoundedness.IRM.data`

```python
data = data
```

###### `causalis.scenarios.unconfoundedness.IRM.diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary containing 'm_hat', 'g0_hat', 'g1_hat', and 'folds'.

###### `causalis.scenarios.unconfoundedness.IRM.estimate`

```python
estimate(score='ATE', alpha=0.05, diagnostic_data=True)
```

Compute treatment effects using stored nuisance predictions.

**Parameters:**

- **score** (<code>('ATE', 'ATTE', 'CATE')</code>) – Target estimand.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – Result container for the estimated effect.

###### `causalis.scenarios.unconfoundedness.IRM.fit`

```python
fit(data=None)
```

Fit nuisance models via cross-fitting.

**Parameters:**

- **data** (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.dgp.causaldata.CausalData)\]</code>) – CausalData container. If None, uses self.data.

**Returns:**

- **self** (<code>[IRM](#causalis.scenarios.unconfoundedness.irm.IRM)</code>) – Fitted estimator.

###### `causalis.scenarios.unconfoundedness.IRM.gate`

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

- <code>[BLP](#causalis.scenarios.cate.blp.BLP)</code> – Fitted Best Linear Predictor model.

###### `causalis.scenarios.unconfoundedness.IRM.ml_g`

```python
ml_g = ml_g
```

###### `causalis.scenarios.unconfoundedness.IRM.ml_m`

```python
ml_m = ml_m
```

###### `causalis.scenarios.unconfoundedness.IRM.n_folds`

```python
n_folds = int(n_folds)
```

###### `causalis.scenarios.unconfoundedness.IRM.n_rep`

```python
n_rep = int(n_rep)
```

###### `causalis.scenarios.unconfoundedness.IRM.normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

###### `causalis.scenarios.unconfoundedness.IRM.orth_signal`

```python
orth_signal: np.ndarray
```

Return the cross-fitted orthogonal signal (psi_b).

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The orthogonal signal.

###### `causalis.scenarios.unconfoundedness.IRM.pvalues`

```python
pvalues: np.ndarray
```

Return the p-values for the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The p-values.

###### `causalis.scenarios.unconfoundedness.IRM.random_state`

```python
random_state = random_state
```

###### `causalis.scenarios.unconfoundedness.IRM.score`

```python
score = 'ATE'
```

###### `causalis.scenarios.unconfoundedness.IRM.se`

```python
se: np.ndarray
```

Return the standard error of the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The standard error.

###### `causalis.scenarios.unconfoundedness.IRM.sensitivity_analysis`

```python
sensitivity_analysis(cf_y, r2_d, rho=1.0, H0=0.0, alpha=0.05)
```

Compute a sensitivity analysis following DoubleML (Chernozhukov et al., 2022).

**Parameters:**

- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome equation (odds form, C_Y^2).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment equation (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation between unobserved components.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level for CI bounds.

###### `causalis.scenarios.unconfoundedness.IRM.summary`

```python
summary: pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The results summary.

###### `causalis.scenarios.unconfoundedness.IRM.trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

###### `causalis.scenarios.unconfoundedness.IRM.trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

###### `causalis.scenarios.unconfoundedness.IRM.weights`

```python
weights = weights
```

##### `causalis.scenarios.unconfoundedness.cate`

Conditional Average Treatment Effect (CATE) inference methods for causalis.

This submodule provides methods for estimating conditional average treatment effects.

**Modules:**

- [**cate_esimand**](#causalis.scenarios.unconfoundedness.cate.cate_esimand) – DoubleML implementation for estimating CATE (per-observation orthogonal signals).

###### `causalis.scenarios.unconfoundedness.cate.cate_esimand`

DoubleML implementation for estimating CATE (per-observation orthogonal signals).

This module provides a function that, given a CausalData object, fits a DoubleML IRM
model and augments the data_contracts with a new column 'cate' that contains the orthogonal
signals (an estimate of the conditional average treatment effect for each unit).

**Functions:**

- [**cate_esimand**](#causalis.scenarios.unconfoundedness.cate.cate_esimand.cate_esimand) – Estimate per-observation CATEs using DoubleML IRM and return a DataFrame with a new 'cate' column.

####### `causalis.scenarios.unconfoundedness.cate.cate_esimand.cate_esimand`

```python
cate_esimand(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, use_blp=False, X_new=None)
```

Estimate per-observation CATEs using DoubleML IRM and return a DataFrame with a new 'cate' column.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – A CausalData object with defined outcome (outcome), treatment (binary 0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – ML learner for outcome regression g(D, X) = E[Y | D, X] supporting fit/predict.
  Defaults to CatBoostRegressor if None.
- **ml_m** (<code>[classifier](#classifier)</code>) – ML learner for propensity m(X) = P[D=1 | X] supporting fit/predict_proba.
  Defaults to CatBoostClassifier if None.
- **n_folds** (<code>[int](#int)</code>) – Number of folds for cross-fitting.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions for sample splitting.
- **use_blp** (<code>[bool](#bool)</code>) – If True, and X_new is provided, returns cate from obj.blp_predict(X_new) aligned to X_new.
  If False (default), uses obj.\_orthogonal_signals (in-sample estimates) and appends to data_contracts.
- **X_new** (<code>[DataFrame](#pandas.DataFrame)</code>) – New covariate matrix for out-of-sample CATE prediction via best linear predictor.
  Must contain the same feature columns as the confounders in `data_contracts`.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – If use_blp is False: returns a copy of data_contracts.df with a new column 'cate'.
  If use_blp is True and X_new is provided: returns a DataFrame with 'cate' column for X_new rows.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If treatment is not binary 0/1 or required metadata is missing.

##### `causalis.scenarios.unconfoundedness.dgp`

**Functions:**

- [**generate_obs_hte_26**](#causalis.scenarios.unconfoundedness.dgp.generate_obs_hte_26) – Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
- [**obs_linear_26_dataset**](#causalis.scenarios.unconfoundedness.dgp.obs_linear_26_dataset) – A pre-configured observational linear dataset with 5 standard confounders.

###### `causalis.scenarios.unconfoundedness.dgp.generate_obs_hte_26`

```python
generate_obs_hte_26(n=10000, seed=42, include_oracle=True, return_causal_data=True)
```

Observational dataset with nonlinear outcome model, nonlinear treatment assignment,
and a heterogeneous (nonlinear) treatment effect tau(X).
Based on the scenario in notebooks/cases/dml_atte.ipynb.

**Parameters:**

- **n** (<code>[int](#int)</code>) – Number of samples.
- **seed** (<code>[int](#int)</code>) – Random seed.
- **include_oracle** (<code>[bool](#bool)</code>) – Whether to include oracle ground-truth columns like 'cate', 'propensity', etc.
- **return_causal_data** (<code>[bool](#bool)</code>) – If True, returns a CausalData object. If False, returns a pandas DataFrame.

###### `causalis.scenarios.unconfoundedness.dgp.obs_linear_26_dataset`

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

##### `causalis.scenarios.unconfoundedness.dml_ate_source`

```python
dml_ate_source(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, score='ATE', alpha=0.05)
```

Estimate average treatment effects using DoubleML's interactive regression model (IRM).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causaldata object containing treatment, target, and confounders variables.
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

##### `causalis.scenarios.unconfoundedness.dml_atte_source`

```python
dml_atte_source(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, alpha=0.05)
```

Estimate average treatment effects on the treated using DoubleML's interactive regression model (IRM).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causaldata object containing treatment, target, and confounders variables.
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

##### `causalis.scenarios.unconfoundedness.dml_source`

DoubleML implementation for estimating average treatment effects.

This module provides functions to estimate average treatment effects (ATE) and
average treatment effects on the treated (ATT) using the DoubleML library.

**Functions:**

- [**dml_ate_source**](#causalis.scenarios.unconfoundedness.dml_source.dml_ate_source) – Estimate average treatment effects using DoubleML's interactive regression model (IRM).
- [**dml_atte_source**](#causalis.scenarios.unconfoundedness.dml_source.dml_atte_source) – Estimate average treatment effects on the treated using DoubleML's interactive regression model (IRM).

###### `causalis.scenarios.unconfoundedness.dml_source.dml_ate_source`

```python
dml_ate_source(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, score='ATE', alpha=0.05)
```

Estimate average treatment effects using DoubleML's interactive regression model (IRM).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causaldata object containing treatment, target, and confounders variables.
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

###### `causalis.scenarios.unconfoundedness.dml_source.dml_atte_source`

```python
dml_atte_source(data, ml_g=None, ml_m=None, n_folds=5, n_rep=1, alpha=0.05)
```

Estimate average treatment effects on the treated using DoubleML's interactive regression model (IRM).

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causaldata object containing treatment, target, and confounders variables.
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

##### `causalis.scenarios.unconfoundedness.irm`

DML IRM estimator consuming CausalData.

Implements cross-fitted nuisance estimation for g0, g1 and m, and supports ATE/ATTE scores.
citation:
software{DoubleML,
title = {{DoubleML} -- Double Machine Learning in Python},
author = {Bach, Philipp and Chernozhukov, Victor and Klaassen, Sven and Kurz, Malte S. and Spindler, Martin},
year = {2024},
version = {latest},
url = {https://github.com/DoubleML/doubleml-for-py},
note = {BSD-3-Clause License. Documentation: \\url{https://docs.doubleml.org/stable/index.html}}
}

**Classes:**

- [**IRM**](#causalis.scenarios.unconfoundedness.irm.IRM) – Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

**Attributes:**

- [**HAS_CATBOOST**](#causalis.scenarios.unconfoundedness.irm.HAS_CATBOOST) –

###### `causalis.scenarios.unconfoundedness.irm.HAS_CATBOOST`

```python
HAS_CATBOOST = True
```

###### `causalis.scenarios.unconfoundedness.irm.IRM`

```python
IRM(data=None, ml_g=None, ml_m=None, *, n_folds=5, n_rep=1, normalize_ipw=False, trimming_rule='truncate', trimming_threshold=0.01, weights=None, random_state=None)
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>

Interactive Regression Model (IRM) with DoubleML-style cross-fitting using CausalData.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Data container with outcome, binary treatment (0/1), and confounders.
- **ml_g** (<code>[estimator](#estimator)</code>) – Learner for E[Y|X,D]. If classifier and Y is binary, predict_proba is used; otherwise predict().
- **ml_m** (<code>[classifier](#classifier)</code>) – Learner for E[D|X] (propensity). Must support predict_proba() or predict() in (0,1).
- **n_folds** (<code>[int](#int)</code>) – Number of cross-fitting folds.
- **n_rep** (<code>[int](#int)</code>) – Number of repetitions of sample splitting. Currently only 1 is supported.
- **normalize_ipw** (<code>[bool](#bool)</code>) – Whether to normalize IPW terms within the score.
- **trimming_rule** (<code>'truncate'</code>) – Trimming approach for propensity scores.
- **trimming_threshold** (<code>[float](#float)</code>) – Threshold for trimming if rule is "truncate".
- **weights** (<code>[Optional](#typing.Optional)\[[ndarray](#numpy.ndarray) or [Dict](#typing.Dict)\]</code>) – Optional weights.
- If array of shape (n,), used as ATE weights (w). Assumed E[w|X] = w.
- If dict, can contain 'weights' (w) and 'weights_bar' (E[w|X]).
- For ATTE, computed internally (w=D/P(D=1), w_bar=m(X)/P(D=1)).
  Note: If weights depend on treatment or outcome, E[w|X] must be provided for correct sensitivity analysis.
- **random_state** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Random seed for fold creation.

**Functions:**

- [**confint**](#causalis.scenarios.unconfoundedness.irm.IRM.confint) – Compute confidence intervals for the estimated coefficient.
- [**estimate**](#causalis.scenarios.unconfoundedness.irm.IRM.estimate) – Compute treatment effects using stored nuisance predictions.
- [**fit**](#causalis.scenarios.unconfoundedness.irm.IRM.fit) – Fit nuisance models via cross-fitting.
- [**gate**](#causalis.scenarios.unconfoundedness.irm.IRM.gate) – Estimate Group Average Treatment Effects via BLP on orthogonal signal.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.irm.IRM.sensitivity_analysis) – Compute a sensitivity analysis following DoubleML (Chernozhukov et al., 2022).

**Attributes:**

- [**coef**](#causalis.scenarios.unconfoundedness.irm.IRM.coef) (<code>[ndarray](#numpy.ndarray)</code>) – Return the estimated coefficient.
- [**data**](#causalis.scenarios.unconfoundedness.irm.IRM.data) –
- [**diagnostics\_**](#causalis.scenarios.unconfoundedness.irm.IRM.diagnostics_) (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code>) – Return diagnostic data.
- [**ml_g**](#causalis.scenarios.unconfoundedness.irm.IRM.ml_g) –
- [**ml_m**](#causalis.scenarios.unconfoundedness.irm.IRM.ml_m) –
- [**n_folds**](#causalis.scenarios.unconfoundedness.irm.IRM.n_folds) –
- [**n_rep**](#causalis.scenarios.unconfoundedness.irm.IRM.n_rep) –
- [**normalize_ipw**](#causalis.scenarios.unconfoundedness.irm.IRM.normalize_ipw) –
- [**orth_signal**](#causalis.scenarios.unconfoundedness.irm.IRM.orth_signal) (<code>[ndarray](#numpy.ndarray)</code>) – Return the cross-fitted orthogonal signal (psi_b).
- [**pvalues**](#causalis.scenarios.unconfoundedness.irm.IRM.pvalues) (<code>[ndarray](#numpy.ndarray)</code>) – Return the p-values for the estimate.
- [**random_state**](#causalis.scenarios.unconfoundedness.irm.IRM.random_state) –
- [**score**](#causalis.scenarios.unconfoundedness.irm.IRM.score) –
- [**se**](#causalis.scenarios.unconfoundedness.irm.IRM.se) (<code>[ndarray](#numpy.ndarray)</code>) – Return the standard error of the estimate.
- [**summary**](#causalis.scenarios.unconfoundedness.irm.IRM.summary) (<code>[DataFrame](#pandas.DataFrame)</code>) – Return a summary DataFrame of the results.
- [**trimming_rule**](#causalis.scenarios.unconfoundedness.irm.IRM.trimming_rule) –
- [**trimming_threshold**](#causalis.scenarios.unconfoundedness.irm.IRM.trimming_threshold) –
- [**weights**](#causalis.scenarios.unconfoundedness.irm.IRM.weights) –

####### `causalis.scenarios.unconfoundedness.irm.IRM.coef`

```python
coef: np.ndarray
```

Return the estimated coefficient.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The estimated coefficient.

####### `causalis.scenarios.unconfoundedness.irm.IRM.confint`

```python
confint(alpha=0.05)
```

Compute confidence intervals for the estimated coefficient.

**Parameters:**

- **alpha** (<code>[float](#float)</code>) – Significance level.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – DataFrame with confidence intervals.

####### `causalis.scenarios.unconfoundedness.irm.IRM.data`

```python
data = data
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.diagnostics_`

```python
diagnostics_: Dict[str, Any]
```

Return diagnostic data.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary containing 'm_hat', 'g0_hat', 'g1_hat', and 'folds'.

####### `causalis.scenarios.unconfoundedness.irm.IRM.estimate`

```python
estimate(score='ATE', alpha=0.05, diagnostic_data=True)
```

Compute treatment effects using stored nuisance predictions.

**Parameters:**

- **score** (<code>('ATE', 'ATTE', 'CATE')</code>) – Target estimand.
- **alpha** (<code>[float](#float)</code>) – Significance level for intervals.
- **diagnostic_data** (<code>[bool](#bool)</code>) – Whether to include diagnostic data_contracts in the result.

**Returns:**

- <code>[CausalEstimate](#causalis.data_contracts.causal_estimate.CausalEstimate)</code> – Result container for the estimated effect.

####### `causalis.scenarios.unconfoundedness.irm.IRM.fit`

```python
fit(data=None)
```

Fit nuisance models via cross-fitting.

**Parameters:**

- **data** (<code>[Optional](#typing.Optional)\[[CausalData](#causalis.dgp.causaldata.CausalData)\]</code>) – CausalData container. If None, uses self.data.

**Returns:**

- **self** (<code>[IRM](#causalis.scenarios.unconfoundedness.irm.IRM)</code>) – Fitted estimator.

####### `causalis.scenarios.unconfoundedness.irm.IRM.gate`

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

- <code>[BLP](#causalis.scenarios.cate.blp.BLP)</code> – Fitted Best Linear Predictor model.

####### `causalis.scenarios.unconfoundedness.irm.IRM.ml_g`

```python
ml_g = ml_g
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.ml_m`

```python
ml_m = ml_m
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.n_folds`

```python
n_folds = int(n_folds)
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.n_rep`

```python
n_rep = int(n_rep)
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.normalize_ipw`

```python
normalize_ipw = bool(normalize_ipw)
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.orth_signal`

```python
orth_signal: np.ndarray
```

Return the cross-fitted orthogonal signal (psi_b).

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The orthogonal signal.

####### `causalis.scenarios.unconfoundedness.irm.IRM.pvalues`

```python
pvalues: np.ndarray
```

Return the p-values for the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The p-values.

####### `causalis.scenarios.unconfoundedness.irm.IRM.random_state`

```python
random_state = random_state
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.score`

```python
score = 'ATE'
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.se`

```python
se: np.ndarray
```

Return the standard error of the estimate.

**Returns:**

- <code>[ndarray](#numpy.ndarray)</code> – The standard error.

####### `causalis.scenarios.unconfoundedness.irm.IRM.sensitivity_analysis`

```python
sensitivity_analysis(cf_y, r2_d, rho=1.0, H0=0.0, alpha=0.05)
```

Compute a sensitivity analysis following DoubleML (Chernozhukov et al., 2022).

**Parameters:**

- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for outcome equation (odds form, C_Y^2).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for treatment equation (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation between unobserved components.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level for CI bounds.

####### `causalis.scenarios.unconfoundedness.irm.IRM.summary`

```python
summary: pd.DataFrame
```

Return a summary DataFrame of the results.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – The results summary.

####### `causalis.scenarios.unconfoundedness.irm.IRM.trimming_rule`

```python
trimming_rule = str(trimming_rule)
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.trimming_threshold`

```python
trimming_threshold = float(trimming_threshold)
```

####### `causalis.scenarios.unconfoundedness.irm.IRM.weights`

```python
weights = weights
```

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
- [**overlap_report_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap_report_from_result) – High-level helper that takes `IRM` result or model and returns a positivity/overlap report as a dict.
- [**plot_m_overlap**](#causalis.scenarios.unconfoundedness.refutation.plot_m_overlap) – Overlap plot for m(x)=P(D=1|X) with high-res rendering.
- [**positivity_overlap_checks**](#causalis.scenarios.unconfoundedness.refutation.positivity_overlap_checks) – Run positivity/overlap diagnostics for DML-IRM (ATE & ATT).
- [**print_sutva_questions**](#causalis.scenarios.unconfoundedness.refutation.print_sutva_questions) – Print the SUTVA validation questions.
- [**refute_irm_orthogonality**](#causalis.scenarios.unconfoundedness.refutation.refute_irm_orthogonality) – Comprehensive AIPW orthogonality diagnostics for IRM models.
- [**refute_placebo_outcome**](#causalis.scenarios.unconfoundedness.refutation.refute_placebo_outcome) – Generate random outcome variables while keeping treatment
- [**refute_placebo_treatment**](#causalis.scenarios.unconfoundedness.refutation.refute_placebo_treatment) – Generate random binary treatment variables while keeping outcome and
- [**refute_subset**](#causalis.scenarios.unconfoundedness.refutation.refute_subset) – Re-estimate the effect on a random subset (default 80 %)
- [**run_overlap_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_overlap_diagnostics) – Single entry-point for overlap / positivity / calibration diagnostics.
- [**run_score_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_score_diagnostics) – Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.
- [**run_uncofoundedness_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.run_uncofoundedness_diagnostics) – Uncofoundedness diagnostics focused on balance (SMD).
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.sensitivity_analysis) – Compute bias-aware components and cache them.
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
Uses Pydantic for validation and as a data_contracts contract.

**Attributes:**

- [**df**](#causalis.scenarios.unconfoundedness.refutation.CausalData.df) (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts restricted to outcome, treatment, and confounder columns.
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

- **df** (<code>[DataFrame](#pandas.DataFrame)</code>) – The DataFrame containing the data_contracts.
- **treatment** (<code>[str](#str)</code>) – Column name representing the treatment variable.
- **outcome** (<code>[str](#str)</code>) – Column name representing the outcome variable.
- **confounders** (<code>[Union](#typing.Union)\[[str](#str), [List](#typing.List)\[[str](#str)\]\]</code>) – Column name(s) representing the confounders/covariates.
- **user_id** (<code>[str](#str)</code>) – Column name representing the unique identifier for each observation/user.
- \*\***kwargs** (<code>[Any](#typing.Any)</code>) – Additional arguments passed to the Pydantic model constructor.

**Returns:**

- <code>[CausalData](#causalis.data_contracts.causaldata.CausalData)</code> – A validated CausalData instance.

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

- **model** (<code>[object](#object)</code>) – Fitted internal IRM estimator (causalis.shared.models.IRM) or a compatible dummy model
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

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **label** (<code>[str](#str)</code>) – The label for the estimand.

**Returns:**

- <code>[Optional](#typing.Optional)\[[str](#str)\]</code> – Formatted summary string or None if extraction fails.

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

- [**overlap_plot**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_plot) –
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
- [**overlap_report_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_report_from_result) – High-level helper that takes `IRM` result or model and returns a positivity/overlap report as a dict.
- [**plot_m_overlap**](#causalis.scenarios.unconfoundedness.refutation.overlap.plot_m_overlap) – Overlap plot for m(x)=P(D=1|X) with high-res rendering.
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

####### `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_plot`

**Functions:**

- [**plot_m_overlap**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_plot.plot_m_overlap) – Overlap plot for m(x)=P(D=1|X) with high-res rendering.

######## `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_plot.plot_m_overlap`

```python
plot_m_overlap(diag, clip=(0.01, 0.99), bins='fd', kde=True, shade_overlap=True, ax=None, figsize=(9, 5.5), dpi=220, font_scale=1.15, save=None, save_dpi=None, transparent=False, color_t=None, color_c=None)
```

Overlap plot for m(x)=P(D=1|X) with high-res rendering.

- x in [0,1]
- Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
- Uses Matplotlib default colors unless color_t/color_c are provided

**Parameters:**

- **diag** (<code>[UnconfoundednessDiagnosticData](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData)</code>) – Diagnostic data containing m_hat and d.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to clip for KDE range.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Histogram bins.
- **kde** (<code>[bool](#bool)</code>) – Whether to show KDE.
- **shade_overlap** (<code>[bool](#bool)</code>) – Whether to shade the overlap area.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.
- **color_t** (<code>[color](#color)</code>) – Color for treated group.
- **color_c** (<code>[color](#color)</code>) – Color for control group.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

####### `causalis.scenarios.unconfoundedness.refutation.overlap.overlap_report_from_result`

```python
overlap_report_from_result(res, *, use_hajek=False, thresholds=DEFAULT_THRESHOLDS, n_bins=10, cal_thresholds=None, auc_flip_margin=0.05)
```

High-level helper that takes `IRM` result or model and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
`IRM(normalize_ipw=True)` get meaningful ipw_sum\_\* checks without extra arguments.

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
- [**extract_diag_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.extract_diag_from_result) – Extract m_hat, D, and trimming epsilon from IRM result or model.
- [**ks_distance**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.ks_distance) – Two-sample Kolmogorov–Smirnov distance between m_hat|D=1 and m_hat|D=0.
- [**overlap_report_from_result**](#causalis.scenarios.unconfoundedness.refutation.overlap.overlap_validation.overlap_report_from_result) – High-level helper that takes `IRM` result or model and returns a positivity/overlap report as a dict.
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

Extract m_hat, D, and trimming epsilon from IRM result or model.
Accepts:

- dict returned by legacy dml_ate/dml_att (prefers key 'diagnostic_data'; otherwise uses 'model'), or
- a fitted IRM/DoubleMLIRM-like model instance with a .data or .data_contracts attribute.
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

High-level helper that takes `IRM` result or model and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
`IRM(normalize_ipw=True)` get meaningful ipw_sum\_\* checks without extra arguments.

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

####### `causalis.scenarios.unconfoundedness.refutation.overlap.plot_m_overlap`

```python
plot_m_overlap(diag, clip=(0.01, 0.99), bins='fd', kde=True, shade_overlap=True, ax=None, figsize=(9, 5.5), dpi=220, font_scale=1.15, save=None, save_dpi=None, transparent=False, color_t=None, color_c=None)
```

Overlap plot for m(x)=P(D=1|X) with high-res rendering.

- x in [0,1]
- Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
- Uses Matplotlib default colors unless color_t/color_c are provided

**Parameters:**

- **diag** (<code>[UnconfoundednessDiagnosticData](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData)</code>) – Diagnostic data containing m_hat and d.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to clip for KDE range.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Histogram bins.
- **kde** (<code>[bool](#bool)</code>) – Whether to show KDE.
- **shade_overlap** (<code>[bool](#bool)</code>) – Whether to shade the overlap area.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.
- **color_t** (<code>[color](#color)</code>) – Color for treated group.
- **color_c** (<code>[color](#color)</code>) – Color for control group.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

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

High-level helper that takes `IRM` result or model and returns a positivity/overlap report as a dict.

If the input result contains a flag indicating normalized IPW (Hájek), this function will
auto-detect it and pass use_hajek=True to the underlying diagnostics, so users of
`IRM(normalize_ipw=True)` get meaningful ipw_sum\_\* checks without extra arguments.

###### `causalis.scenarios.unconfoundedness.refutation.plot_m_overlap`

```python
plot_m_overlap(diag, clip=(0.01, 0.99), bins='fd', kde=True, shade_overlap=True, ax=None, figsize=(9, 5.5), dpi=220, font_scale=1.15, save=None, save_dpi=None, transparent=False, color_t=None, color_c=None)
```

Overlap plot for m(x)=P(D=1|X) with high-res rendering.

- x in [0,1]
- Stable NumPy KDE w/ boundary reflection (no SciPy warnings)
- Uses Matplotlib default colors unless color_t/color_c are provided

**Parameters:**

- **diag** (<code>[UnconfoundednessDiagnosticData](#causalis.data_contracts.causal_diagnostic_data.UnconfoundednessDiagnosticData)</code>) – Diagnostic data containing m_hat and d.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to clip for KDE range.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Histogram bins.
- **kde** (<code>[bool](#bool)</code>) – Whether to show KDE.
- **shade_overlap** (<code>[bool](#bool)</code>) – Whether to shade the overlap area.
- **ax** (<code>[Axes](#matplotlib.axes.Axes)</code>) – Existing axes to plot on.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size.
- **dpi** (<code>[int](#int)</code>) – Dots per inch.
- **font_scale** (<code>[float](#float)</code>) – Font scaling factor.
- **save** (<code>[str](#str)</code>) – Path to save the figure.
- **save_dpi** (<code>[int](#int)</code>) – DPI for saving.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save with transparency.
- **color_t** (<code>[color](#color)</code>) – Color for treated group.
- **color_c** (<code>[color](#color)</code>) – Color for control group.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure.

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
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causal data_contracts object
- **trim_propensity** (<code>[Tuple](#typing.Tuple)\[[float](#float), [float](#float)\]</code>) – Propensity score trimming bounds (min, max) to avoid extreme weights
- **n_basis_funcs** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Number of basis functions for orthogonality derivative tests (constant + covariates).
  If None, defaults to the number of confounders in `data_contracts` plus 1 for the constant term.
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
>>> from causalis.scenarios.unconfoundedness.irm import IRM
>>> 
>>> # Define a wrapper for refutation utilities that expect an inference function
>>> def irm_ate_inference(data, **kwargs):
>>>     return IRM(data, **kwargs).fit().estimate().model_dump()
>>> 
>>> # Comprehensive orthogonality check
>>> ortho_results = refute_irm_orthogonality(irm_ate_inference, causal_data)
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
random variables from a normal distribution fitted to the original data_contracts.
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
run_score_diagnostics(res=<IRM result or IRM-like model>)

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

**Functions:**

- [**influence_summary**](#causalis.scenarios.unconfoundedness.refutation.score.influence_summary) – Compute influence diagnostics showing where uncertainty comes from.
- [**refute_irm_orthogonality**](#causalis.scenarios.unconfoundedness.refutation.score.refute_irm_orthogonality) – Comprehensive AIPW orthogonality diagnostics for IRM models.
- [**refute_placebo_outcome**](#causalis.scenarios.unconfoundedness.refutation.score.refute_placebo_outcome) – Generate random outcome variables while keeping treatment
- [**refute_placebo_treatment**](#causalis.scenarios.unconfoundedness.refutation.score.refute_placebo_treatment) – Generate random binary treatment variables while keeping outcome and
- [**refute_subset**](#causalis.scenarios.unconfoundedness.refutation.score.refute_subset) – Re-estimate the effect on a random subset (default 80 %)
- [**run_score_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.score.run_score_diagnostics) – Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.

####### `causalis.scenarios.unconfoundedness.refutation.score.influence_summary`

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

####### `causalis.scenarios.unconfoundedness.refutation.score.refute_irm_orthogonality`

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
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causal data_contracts object
- **trim_propensity** (<code>[Tuple](#typing.Tuple)\[[float](#float), [float](#float)\]</code>) – Propensity score trimming bounds (min, max) to avoid extreme weights
- **n_basis_funcs** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Number of basis functions for orthogonality derivative tests (constant + covariates).
  If None, defaults to the number of confounders in `data_contracts` plus 1 for the constant term.
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
>>> from causalis.scenarios.unconfoundedness.irm import IRM
>>> 
>>> # Define a wrapper for refutation utilities that expect an inference function
>>> def irm_ate_inference(data, **kwargs):
>>>     return IRM(data, **kwargs).fit().estimate().model_dump()
>>> 
>>> # Comprehensive orthogonality check
>>> ortho_results = refute_irm_orthogonality(irm_ate_inference, causal_data)
>>> 
>>> # Check key diagnostics
>>> print(f"OOS moment t-stat: {ortho_results['oos_moment_test']['tstat']:.3f}")
>>> print(f"Assessment: {ortho_results['overall_assessment']}")
```

####### `causalis.scenarios.unconfoundedness.refutation.score.refute_placebo_outcome`

```python
refute_placebo_outcome(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random outcome variables while keeping treatment
and covariates intact. For binary outcomes, generates random binary
variables with the same proportion. For continuous outcomes, generates
random variables from a normal distribution fitted to the original data_contracts.
A valid causal design should now yield θ ≈ 0 and a large p-value.

####### `causalis.scenarios.unconfoundedness.refutation.score.refute_placebo_treatment`

```python
refute_placebo_treatment(inference_fn, data, random_state=None, **inference_kwargs)
```

Generate random binary treatment variables while keeping outcome and
covariates intact. Generates random binary treatment with the same
proportion as the original treatment. Breaks the treatment–outcome link.

####### `causalis.scenarios.unconfoundedness.refutation.score.refute_subset`

```python
refute_subset(inference_fn, data, fraction=0.8, random_state=None, **inference_kwargs)
```

Re-estimate the effect on a random subset (default 80 %)
to check sample-stability of the estimate.

####### `causalis.scenarios.unconfoundedness.refutation.score.run_score_diagnostics`

```python
run_score_diagnostics(res=None, *, y=None, d=None, g0=None, g1=None, m=None, theta=None, score=None, trimming_threshold=0.01, n_basis_funcs=None, return_summary=True)
```

Single entry-point for score diagnostics (orthogonality) akin to run_overlap_diagnostics.

You can call it in TWO ways:
A) With raw arrays:
run_score_diagnostics(y=..., d=..., g0=..., g1=..., m=..., theta=...)
B) With a model/result:
run_score_diagnostics(res=<IRM result or IRM-like model>)

Returns a dictionary with:

- params (score, trimming_threshold)
- oos_moment_test (if fast-path caches available on model; else omitted)
- orthogonality_derivatives (DataFrame)
- influence_diagnostics (full_sample)
- summary (compact DataFrame) if return_summary=True
- meta

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

- **model** (<code>[object](#object)</code>) – Fitted internal IRM estimator (causalis.shared.models.IRM) or a compatible dummy model
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
- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causal data_contracts object
- **trim_propensity** (<code>[Tuple](#typing.Tuple)\[[float](#float), [float](#float)\]</code>) – Propensity score trimming bounds (min, max) to avoid extreme weights
- **n_basis_funcs** (<code>[Optional](#typing.Optional)\[[int](#int)\]</code>) – Number of basis functions for orthogonality derivative tests (constant + covariates).
  If None, defaults to the number of confounders in `data_contracts` plus 1 for the constant term.
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
>>> from causalis.scenarios.unconfoundedness.irm import IRM
>>> 
>>> # Define a wrapper for refutation utilities that expect an inference function
>>> def irm_ate_inference(data, **kwargs):
>>>     return IRM(data, **kwargs).fit().estimate().model_dump()
>>> 
>>> # Comprehensive orthogonality check
>>> ortho_results = refute_irm_orthogonality(irm_ate_inference, causal_data)
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
random variables from a normal distribution fitted to the original data_contracts.
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
run_score_diagnostics(res=<IRM result or IRM-like model>)

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
sensitivity_analysis(effect_estimation, *, cf_y, r2_d, rho=1.0, H0=0.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware components and cache them.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (odds form, C_Y^2).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results:
  - theta, se, alpha, z
  - sampling_ci
  - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
  - bias_aware_ci = faithful DoubleML CI for the bounds
  - max_bias and components (sigma2, nu2)
  - params (cf_y, r2_d, rho, use_signed_rr)

###### `causalis.scenarios.unconfoundedness.refutation.sensitivity_benchmark`

```python
sensitivity_benchmark(effect_estimation, benchmarking_set, fit_args=None)
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing cf_y, r2_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Additional keyword arguments for the IRM.fit() method of the short model.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- cf_y, r2_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

###### `causalis.scenarios.unconfoundedness.refutation.sutva`

**Modules:**

- [**sutva_validation**](#causalis.scenarios.unconfoundedness.refutation.sutva.sutva_validation) – SUTVA validation helper.

**Functions:**

- [**print_sutva_questions**](#causalis.scenarios.unconfoundedness.refutation.sutva.print_sutva_questions) – Print the SUTVA validation questions.

**Attributes:**

- [**QUESTIONS**](#causalis.scenarios.unconfoundedness.refutation.sutva.QUESTIONS) (<code>[Iterable](#typing.Iterable)\[[str](#str)\]</code>) –

####### `causalis.scenarios.unconfoundedness.refutation.sutva.QUESTIONS`

```python
QUESTIONS: Iterable[str] = ('1.) Are your clients independent (i)?', '2.) Do you measure confounders, treatment, and outcome in the same intervals?', '3.) Do you measure confounders before treatment and outcome after?', '4.) Do you have a consistent label of treatment, such as if a person does not receive a treatment, he has a label 0?')
```

####### `causalis.scenarios.unconfoundedness.refutation.sutva.print_sutva_questions`

```python
print_sutva_questions()
```

Print the SUTVA validation questions.

Just prints questions, nothing more.

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

**Functions:**

- [**compute_bias_aware_ci**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.compute_bias_aware_ci) – Compute bias-aware confidence intervals.
- [**get_sensitivity_summary**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**run_uncofoundedness_diagnostics**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.run_uncofoundedness_diagnostics) – Uncofoundedness diagnostics focused on balance (SMD).
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity_analysis) – Compute bias-aware components and cache them.
- [**sensitivity_benchmark**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model
- [**validate_uncofoundedness_balance**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.validate_uncofoundedness_balance) – Assess covariate balance under the uncofoundedness assumption by computing

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.compute_bias_aware_ci`

```python
compute_bias_aware_ci(effect_estimation, *, cf_y, r2_d, rho=1.0, H0=0.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware confidence intervals.

Returns a dict with:

- theta, se, alpha, z
- sampling_ci
- theta_bounds_cofounding = [theta_lower, theta_upper] = theta ± max_bias
- bias_aware_ci = \[theta - (max_bias + z*se), theta + (max_bias + z*se)\]
- max_bias and components (sigma2, nu2)

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (odds form, C_Y^2).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results.

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.get_sensitivity_summary`

```python
get_sensitivity_summary(effect_estimation, *, label=None)
```

Render a single, unified bias-aware summary string.

If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **label** (<code>[str](#str)</code>) – The label for the estimand.

**Returns:**

- <code>[Optional](#typing.Optional)\[[str](#str)\]</code> – Formatted summary string or None if extraction fails.

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.run_uncofoundedness_diagnostics`

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

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity`

Sensitivity functions refactored into a dedicated module.

This module centralizes bias-aware sensitivity helpers and the public
entry points used by refutation utilities for uncofoundedness.

**Functions:**

- [**get_sensitivity_summary**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.get_sensitivity_summary) – Render a single, unified bias-aware summary string.
- [**sensitivity_analysis**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_analysis) – Compute bias-aware components and cache them.
- [**sensitivity_benchmark**](#causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_benchmark) – Computes a benchmark for a given set of features by refitting a short IRM model

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.get_sensitivity_summary`

```python
get_sensitivity_summary(effect_estimation, *, label=None)
```

Render a single, unified bias-aware summary string.

If bias-aware components are missing, shows a sampling-only variant with max_bias=0
and then formats via `format_bias_aware_summary` for consistency.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **label** (<code>[str](#str)</code>) – The label for the estimand.

**Returns:**

- <code>[Optional](#typing.Optional)\[[str](#str)\]</code> – Formatted summary string or None if extraction fails.

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_analysis`

```python
sensitivity_analysis(effect_estimation, *, cf_y, r2_d, rho=1.0, H0=0.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware components and cache them.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (odds form, C_Y^2).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results:
  - theta, se, alpha, z
  - sampling_ci
  - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
  - bias_aware_ci = faithful DoubleML CI for the bounds
  - max_bias and components (sigma2, nu2)
  - params (cf_y, r2_d, rho, use_signed_rr)

######## `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity.sensitivity_benchmark`

```python
sensitivity_benchmark(effect_estimation, benchmarking_set, fit_args=None)
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing cf_y, r2_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Additional keyword arguments for the IRM.fit() method of the short model.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- cf_y, r2_d, rho: residual-based benchmarking strengths
- theta_long, theta_short, delta: effect estimates and their change (long - short)

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity_analysis`

```python
sensitivity_analysis(effect_estimation, *, cf_y, r2_d, rho=1.0, H0=0.0, alpha=0.05, use_signed_rr=False)
```

Compute bias-aware components and cache them.

**Parameters:**

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] or [Any](#typing.Any)</code>) – The effect estimation object.
- **cf_y** (<code>[float](#float)</code>) – Sensitivity parameter for the outcome (odds form, C_Y^2).
- **r2_d** (<code>[float](#float)</code>) – Sensitivity parameter for the treatment (R^2 form, R_D^2).
- **rho** (<code>[float](#float)</code>) – Correlation parameter.
- **H0** (<code>[float](#float)</code>) – Null hypothesis for robustness values.
- **alpha** (<code>[float](#float)</code>) – Significance level.
- **use_signed_rr** (<code>[bool](#bool)</code>) – Whether to use signed rr.

**Returns:**

- <code>[dict](#dict)</code> – Dictionary with bias-aware results:
  - theta, se, alpha, z
  - sampling_ci
  - theta_bounds_cofounding = (theta - bound_width, theta + bound_width)
  - bias_aware_ci = faithful DoubleML CI for the bounds
  - max_bias and components (sigma2, nu2)
  - params (cf_y, r2_d, rho, use_signed_rr)

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity_benchmark`

```python
sensitivity_benchmark(effect_estimation, benchmarking_set, fit_args=None)
```

Computes a benchmark for a given set of features by refitting a short IRM model
(excluding the provided features) and contrasting it with the original (long) model.
Returns a DataFrame containing cf_y, r2_d, rho and the change in estimates.

**Parameters:**

- **effect_estimation** (<code>[dict](#dict)</code>) – A dictionary containing the fitted IRM model under the key 'model'.
- **benchmarking_set** (<code>[list](#list)\[[str](#str)\]</code>) – List of confounder names to be used for benchmarking (to be removed in the short model).
- **fit_args** (<code>[dict](#dict)</code>) – Additional keyword arguments for the IRM.fit() method of the short model.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – A one-row DataFrame indexed by the treatment name with columns:
- cf_y, r2_d, rho: residual-based benchmarking strengths
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

This function expects the result dictionary or CausalEstimate returned by dml_ate() or dml_att(),
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

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] | [Any](#typing.Any)</code>) – Output from dml_ate() or dml_att(). Must contain 'diagnostic_data'.
- **threshold** (<code>[float](#float)</code>) – Threshold for SMD; values below indicate acceptable balance for most use cases.
- **normalize** (<code>[Optional](#typing.Optional)\[[bool](#bool)\]</code>) – Whether to use normalized weights. If None, inferred from diagnostic_data.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary with keys:
- 'smd': pd.Series of weighted SMD values indexed by confounder names
- 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
- 'score': 'ATE' or 'ATTE'
- 'normalized': bool used for weighting
- 'threshold': float
- 'pass': bool indicating whether all weighted SMDs are below threshold

####### `causalis.scenarios.unconfoundedness.refutation.uncofoundedness.validate_uncofoundedness_balance`

```python
validate_uncofoundedness_balance(effect_estimation, *, threshold=0.1, normalize=None)
```

Assess covariate balance under the uncofoundedness assumption by computing
standardized mean differences (SMD) both before weighting (raw groups) and
after weighting using the IPW / ATT weights implied by the DML/IRM estimation.

This function expects the result dictionary or CausalEstimate returned by dml_ate() or dml_att(),
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

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] | [Any](#typing.Any)</code>) – Output from dml_ate() or dml_att(). Must contain 'diagnostic_data'.
- **threshold** (<code>[float](#float)</code>) – Threshold for SMD; values below indicate acceptable balance for most use cases.
- **normalize** (<code>[Optional](#typing.Optional)\[[bool](#bool)\]</code>) – Whether to use normalized weights. If None, inferred from diagnostic_data.

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

This function expects the result dictionary or CausalEstimate returned by dml_ate() or dml_att(),
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

- **effect_estimation** (<code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\] | [Any](#typing.Any)</code>) – Output from dml_ate() or dml_att(). Must contain 'diagnostic_data'.
- **threshold** (<code>[float](#float)</code>) – Threshold for SMD; values below indicate acceptable balance for most use cases.
- **normalize** (<code>[Optional](#typing.Optional)\[[bool](#bool)\]</code>) – Whether to use normalized weights. If None, inferred from diagnostic_data.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary with keys:
- 'smd': pd.Series of weighted SMD values indexed by confounder names
- 'smd_unweighted': pd.Series of SMD values computed before weighting (raw groups)
- 'score': 'ATE' or 'ATTE'
- 'normalized': bool used for weighting
- 'threshold': float
- 'pass': bool indicating whether all weighted SMDs are below threshold

### `causalis.shared`

**Modules:**

- [**confounders_balance**](#causalis.shared.confounders_balance) –
- [**outcome_plots**](#causalis.shared.outcome_plots) –
- [**outcome_stats**](#causalis.shared.outcome_stats) – Outcome shared grouped by treatment for CausalData.
- [**rct_design**](#causalis.shared.rct_design) – Design module for experimental rct_design utilities.
- [**srm**](#causalis.shared.srm) – Sample Ratio Mismatch (SRM) utilities for randomized experiments.

**Classes:**

- [**SRMResult**](#causalis.shared.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**check_srm**](#causalis.shared.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

#### `causalis.shared.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.shared.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.shared.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.shared.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.shared.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.shared.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.shared.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.shared.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.shared.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

##### `causalis.shared.SRMResult.alpha`

```python
alpha: float
```

##### `causalis.shared.SRMResult.chi2`

```python
chi2: float
```

##### `causalis.shared.SRMResult.df`

```python
df: int
```

##### `causalis.shared.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

##### `causalis.shared.SRMResult.is_srm`

```python
is_srm: bool
```

##### `causalis.shared.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

##### `causalis.shared.SRMResult.p_value`

```python
p_value: float
```

##### `causalis.shared.SRMResult.warning`

```python
warning: str | None = None
```

#### `causalis.shared.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
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

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

#### `causalis.shared.confounders_balance`

**Functions:**

- [**confounders_balance**](#causalis.shared.confounders_balance.confounders_balance) – Compute balance diagnostics for confounders between treatment groups.

##### `causalis.shared.confounders_balance.confounders_balance`

```python
confounders_balance(data)
```

Compute balance diagnostics for confounders between treatment groups.

Produces a DataFrame containing expanded confounder columns (after one-hot
encoding categorical variables if present) with:

- confounders: name of the confounder
- mean_d_0: mean value for control group (t=0)
- mean_d_1: mean value for treated group (t=1)
- abs_diff: abs(mean_d_1 - mean_d_0)
- smd: standardized mean difference (Cohen's d using pooled std)
- ks_pvalue: p-value for the KS test (rounded to 5 decimal places, non-scientific)

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causal dataset containing the dataframe, treatment, and confounders.
  Accepts CausalData or any object with `df`, `treatment`, and `confounders`
  attributes/properties.

**Returns:**

- <code>[DataFrame](#pandas.DataFrame)</code> – Balance table sorted by |smd| (descending).

#### `causalis.shared.outcome_plots`

**Functions:**

- [**outcome_boxplot**](#causalis.shared.outcome_plots.outcome_boxplot) – Prettified boxplot of the outcome by treatment.
- [**outcome_hist**](#causalis.shared.outcome_plots.outcome_hist) – Plot the distribution of the outcome for each treatment on a single, pretty plot.
- [**outcome_plots**](#causalis.shared.outcome_plots.outcome_plots) – Plot the distribution of the outcome for every treatment on one plot,

##### `causalis.shared.outcome_plots.outcome_boxplot`

```python
outcome_boxplot(data, treatment=None, outcome=None, figsize=(9, 5.5), dpi=220, font_scale=1.15, showfliers=True, patch_artist=True, save=None, save_dpi=None, transparent=False)
```

Prettified boxplot of the outcome by treatment.

<details class="features" open markdown="1">
<summary>Features</summary>

- High-DPI figure, scalable fonts
- Soft modern color styling (default Matplotlib palette)
- Optional outliers, gentle transparency
- Optional save to PNG/SVG/PDF

</details>

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. Defaults to the one in `data_contracts`.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches (width, height).
- **dpi** (<code>[int](#int)</code>) – Dots per inch for the figure.
- **font_scale** (<code>[float](#float)</code>) – Scaling factor for all font sizes in the plot.
- **showfliers** (<code>[bool](#bool)</code>) – Whether to show outliers (fliers).
- **patch_artist** (<code>[bool](#bool)</code>) – Whether to fill boxes with color.
- **save** (<code>[str](#str)</code>) – Path to save the figure (e.g., "boxplot.png").
- **save_dpi** (<code>[int](#int)</code>) – DPI for the saved figure. Defaults to 300 for raster formats.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save the figure with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure object.

##### `causalis.shared.outcome_plots.outcome_hist`

```python
outcome_hist(data, treatment=None, outcome=None, bins='fd', density=True, alpha=0.45, sharex=True, kde=True, clip=(0.01, 0.99), figsize=(9, 5.5), dpi=220, font_scale=1.15, save=None, save_dpi=None, transparent=False)
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

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. Defaults to the one in `data_contracts`.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **bins** (<code>[str](#str) or [int](#int)</code>) – Number of bins for histograms (e.g., "fd", "auto", or an integer).
- **density** (<code>[bool](#bool)</code>) – Whether to normalize histograms to form a density.
- **alpha** (<code>[float](#float)</code>) – Transparency for overlaid histograms and bars.
- **sharex** (<code>[bool](#bool)</code>) – If True, use the same x-limits across treatments for numeric outcomes.
- **kde** (<code>[bool](#bool)</code>) – Whether to overlay a smooth density (KDE) for numeric outcomes.
- **clip** (<code>[tuple](#tuple)</code>) – Quantiles to trim tails for nicer view of numeric outcomes.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size in inches (width, height).
- **dpi** (<code>[int](#int)</code>) – Dots per inch for the figure.
- **font_scale** (<code>[float](#float)</code>) – Scaling factor for all font sizes in the plot.
- **save** (<code>[str](#str)</code>) – Path to save the figure (e.g., "outcome.png").
- **save_dpi** (<code>[int](#int)</code>) – DPI for the saved figure. Defaults to 300 for raster formats.
- **transparent** (<code>[bool](#bool)</code>) – Whether to save the figure with a transparent background.

**Returns:**

- <code>[Figure](#matplotlib.figure.Figure)</code> – The generated figure object.

##### `causalis.shared.outcome_plots.outcome_plots`

```python
outcome_plots(data, treatment=None, outcome=None, bins=30, density=True, alpha=0.5, figsize=(7, 4), sharex=True)
```

Plot the distribution of the outcome for every treatment on one plot,
and also produce a boxplot by treatment to visualize outliers.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The causal dataset containing the dataframe and metadata.
- **treatment** (<code>[str](#str)</code>) – Treatment column name. Defaults to the one in `data_contracts`.
- **outcome** (<code>[str](#str)</code>) – Outcome column name. Defaults to the one in `data_contracts`.
- **bins** (<code>[int](#int)</code>) – Number of bins for histograms when the outcome is numeric.
- **density** (<code>[bool](#bool)</code>) – Whether to normalize histograms to form a density.
- **alpha** (<code>[float](#float)</code>) – Transparency for overlaid histograms.
- **figsize** (<code>[tuple](#tuple)</code>) – Figure size for the plots (width, height).
- **sharex** (<code>[bool](#bool)</code>) – If True and the outcome is numeric, use the same x-limits across treatments.

**Returns:**

- <code>[Tuple](#typing.Tuple)\[[Figure](#matplotlib.figure.Figure), [Figure](#matplotlib.figure.Figure)\]</code> – (fig_distribution, fig_boxplot)

#### `causalis.shared.outcome_stats`

Outcome shared grouped by treatment for CausalData.

**Functions:**

- [**outcome_stats**](#causalis.shared.outcome_stats.outcome_stats) – Comprehensive outcome shared grouped by treatment.

##### `causalis.shared.outcome_stats.outcome_stats`

```python
outcome_stats(data)
```

Comprehensive outcome shared grouped by treatment.

Returns a DataFrame with detailed outcome shared for each treatment group,
including count, mean, std, min, various percentiles, and max.
This function provides comprehensive outcome analysis and returns
data_contracts in a clean DataFrame format suitable for reporting.

**Parameters:**

- **data** (<code>[CausalData](#causalis.dgp.causaldata.CausalData)</code>) – The CausalData object containing treatment and outcome variables.

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
   treatment  count      mean       std       min       p10       p25    median       p75       p90       max
0          0   3000  5.123456  2.345678  0.123456  2.345678  3.456789  5.123456  6.789012  7.890123  9.876543
1          1   2000  6.789012  2.456789  0.234567  3.456789  4.567890  6.789012  8.901234  9.012345  10.987654
```

#### `causalis.shared.rct_design`

Design module for experimental rct_design utilities.

**Modules:**

- [**mde**](#causalis.shared.rct_design.mde) – Utility functions for calculating Minimum Detectable Effect (MDE) for experimental rct_design.
- [**split**](#causalis.shared.rct_design.split) – Split (assignment) utilities for randomized controlled experiments.

**Classes:**

- [**SRMResult**](#causalis.shared.rct_design.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**assign_variants_df**](#causalis.shared.rct_design.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.
- [**calculate_mde**](#causalis.shared.rct_design.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.
- [**check_srm**](#causalis.shared.rct_design.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

##### `causalis.shared.rct_design.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.shared.rct_design.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.shared.rct_design.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.shared.rct_design.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.shared.rct_design.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.shared.rct_design.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.shared.rct_design.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.shared.rct_design.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.shared.rct_design.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `causalis.shared.rct_design.SRMResult.alpha`

```python
alpha: float
```

###### `causalis.shared.rct_design.SRMResult.chi2`

```python
chi2: float
```

###### `causalis.shared.rct_design.SRMResult.df`

```python
df: int
```

###### `causalis.shared.rct_design.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

###### `causalis.shared.rct_design.SRMResult.is_srm`

```python
is_srm: bool
```

###### `causalis.shared.rct_design.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

###### `causalis.shared.rct_design.SRMResult.p_value`

```python
p_value: float
```

###### `causalis.shared.rct_design.SRMResult.warning`

```python
warning: str | None = None
```

##### `causalis.shared.rct_design.assign_variants_df`

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

##### `causalis.shared.rct_design.calculate_mde`

```python
calculate_mde(sample_size, baseline_rate=None, variance=None, alpha=0.05, power=0.8, data_type='conversion', ratio=0.5)
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data_contracts) or baseline mean (for continuous data_contracts).
  Required for conversion data_contracts.
- **variance** (<code>float or tuple of float</code>) – Variance of the data_contracts. For conversion data_contracts, this is calculated from the baseline rate if not provided.
  For continuous data_contracts, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data_contracts. Either 'conversion' for binary/conversion data_contracts or 'continuous' for continuous data_contracts.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data_contracts with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data_contracts with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

##### `causalis.shared.rct_design.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
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

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.

##### `causalis.shared.rct_design.mde`

Utility functions for calculating Minimum Detectable Effect (MDE) for experimental rct_design.

**Functions:**

- [**calculate_mde**](#causalis.shared.rct_design.mde.calculate_mde) – Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

###### `causalis.shared.rct_design.mde.calculate_mde`

```python
calculate_mde(sample_size, baseline_rate=None, variance=None, alpha=0.05, power=0.8, data_type='conversion', ratio=0.5)
```

Calculate the Minimum Detectable Effect (MDE) for conversion or continuous data_contracts.

**Parameters:**

- **sample_size** (<code>int or tuple of int</code>) – Total sample size or a tuple of (control_size, treatment_size).
  If a single integer is provided, the sample will be split according to the ratio parameter.
- **baseline_rate** (<code>[float](#float)</code>) – Baseline conversion rate (for conversion data_contracts) or baseline mean (for continuous data_contracts).
  Required for conversion data_contracts.
- **variance** (<code>float or tuple of float</code>) – Variance of the data_contracts. For conversion data_contracts, this is calculated from the baseline rate if not provided.
  For continuous data_contracts, this parameter is required.
  Can be a single float (assumed same for both groups) or a tuple of (control_variance, treatment_variance).
- **alpha** (<code>[float](#float)</code>) – Significance level (Type I error rate).
- **power** (<code>[float](#float)</code>) – Statistical power (1 - Type II error rate).
- **data_type** (<code>[str](#str)</code>) – Type of data_contracts. Either 'conversion' for binary/conversion data_contracts or 'continuous' for continuous data_contracts.
- **ratio** (<code>[float](#float)</code>) – Ratio of the sample allocated to the control group if sample_size is a single integer.

**Returns:**

- <code>[Dict](#typing.Dict)\[[str](#str), [Any](#typing.Any)\]</code> – A dictionary containing:
- 'mde': The minimum detectable effect (absolute)
- 'mde_relative': The minimum detectable effect as a percentage of the baseline (relative)
- 'parameters': The parameters used for the calculation

**Examples:**

```pycon
>>> # Calculate MDE for conversion data_contracts with 1000 total sample size and 10% baseline conversion rate
>>> calculate_mde(1000, baseline_rate=0.1, data_type='conversion')
{'mde': 0.0527..., 'mde_relative': 0.5272..., 'parameters': {...}}
```

```pycon
>>> # Calculate MDE for continuous data_contracts with 500 samples in each group and variance of 4
>>> calculate_mde((500, 500), variance=4, data_type='continuous')
{'mde': 0.3482..., 'mde_relative': None, 'parameters': {...}}
```

<details class="note" open markdown="1">
<summary>Notes</summary>

For conversion data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((p1\*(1-p1)/n1) + (p2\*(1-p2)/n2))

For continuous data_contracts, the MDE is calculated using the formula:
MDE = (z_α/2 + z_β) * sqrt((σ1²/n1) + (σ2²/n2))

where:

- z_α/2 is the critical value for significance level α
- z_β is the critical value for power
- p1 and p2 are the conversion rates in the control and treatment groups
- σ1² and σ2² are the variances in the control and treatment groups
- n1 and n2 are the sample sizes in the control and treatment groups

</details>

##### `causalis.shared.rct_design.split`

Split (assignment) utilities for randomized controlled experiments.

This module provides deterministic assignment of variants to entities based
on hashing a composite key (salt | layer_id | experiment_id | entity_id)
into the unit interval and mapping it to cumulative variant weights.

The implementation mirrors the reference notebook in docs/cases/rct_design.ipynb.

**Functions:**

- [**assign_variants_df**](#causalis.shared.rct_design.split.assign_variants_df) – Deterministically assign variants for each row in df based on id_col.

###### `causalis.shared.rct_design.split.assign_variants_df`

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

#### `causalis.shared.srm`

Sample Ratio Mismatch (SRM) utilities for randomized experiments.

This module implements a chi-square goodness-of-fit SRM check mirroring the
reference implementation demonstrated in docs/cases/rct_design.ipynb.

**Classes:**

- [**SRMResult**](#causalis.shared.srm.SRMResult) – Result of a Sample Ratio Mismatch (SRM) check.

**Functions:**

- [**check_srm**](#causalis.shared.srm.check_srm) – Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

##### `causalis.shared.srm.SRMResult`

```python
SRMResult(chi2, df, p_value, expected, observed, alpha, is_srm, warning=None)
```

Result of a Sample Ratio Mismatch (SRM) check.

**Attributes:**

- [**chi2**](#causalis.shared.srm.SRMResult.chi2) (<code>[float](#float)</code>) – The calculated chi-square statistic.
- [**df**](#causalis.shared.srm.SRMResult.df) (<code>[int](#int)</code>) – Degrees of freedom used in the test.
- [**p_value**](#causalis.shared.srm.SRMResult.p_value) (<code>[float](#float)</code>) – The p-value of the test.
- [**expected**](#causalis.shared.srm.SRMResult.expected) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [float](#float)\]</code>) – Expected counts for each variant.
- [**observed**](#causalis.shared.srm.SRMResult.observed) (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [int](#int)\]</code>) – Observed counts for each variant.
- [**alpha**](#causalis.shared.srm.SRMResult.alpha) (<code>[float](#float)</code>) – Significance level used for the check.
- [**is_srm**](#causalis.shared.srm.SRMResult.is_srm) (<code>[bool](#bool)</code>) – True if an SRM was detected (p_value < alpha), False otherwise.
- [**warning**](#causalis.shared.srm.SRMResult.warning) (<code>([str](#str), [optional](#optional))</code>) – Warning message if the test assumptions might be violated (e.g., small expected counts).

###### `causalis.shared.srm.SRMResult.alpha`

```python
alpha: float
```

###### `causalis.shared.srm.SRMResult.chi2`

```python
chi2: float
```

###### `causalis.shared.srm.SRMResult.df`

```python
df: int
```

###### `causalis.shared.srm.SRMResult.expected`

```python
expected: Dict[Hashable, float]
```

###### `causalis.shared.srm.SRMResult.is_srm`

```python
is_srm: bool
```

###### `causalis.shared.srm.SRMResult.observed`

```python
observed: Dict[Hashable, int]
```

###### `causalis.shared.srm.SRMResult.p_value`

```python
p_value: float
```

###### `causalis.shared.srm.SRMResult.warning`

```python
warning: str | None = None
```

##### `causalis.shared.srm.check_srm`

```python
check_srm(assignments, target_allocation, alpha=0.001, min_expected=5.0, strict_variants=True)
```

Check Sample Ratio Mismatch (SRM) for an RCT via chi-square goodness-of-fit test.

**Parameters:**

- **assignments** (<code>[Iterable](#typing.Iterable)\[[Hashable](#typing.Hashable)\] or [Series](#pandas.Series) or [CausalData](#causalis.dgp.causaldata.CausalData)</code>) – Iterable of assigned variant labels for each unit (user_id, session_id, etc.).
  E.g. Series of ["control", "treatment", ...].
  If CausalData is provided, the treatment column is used.
- **target_allocation** (<code>[Dict](#typing.Dict)\[[Hashable](#typing.Hashable), [Number](#causalis.shared.srm.Number)\]</code>) – Mapping {variant: p} describing intended allocation as PROBABILITIES.
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

- <code>[SRMResult](#causalis.shared.srm.SRMResult)</code> – The result of the SRM check.

**Raises:**

- <code>[ValueError](#ValueError)</code> – If inputs are invalid or empty.
- <code>[ImportError](#ImportError)</code> – If scipy is required but not installed.
