# Inference

The `causalis.scenarios` and `causalis.statistics` modules provide estimators to quantify causal effects from observational or experimental data.  
It offers ready-to-use functions for common targets such as:

- **ATE (Average Treatment Effect):** overall effect of a binary treatment on the outcome  
- **ATT (Average Treatment effect on the Treated)**  
- **CATE / GATE:** heterogeneity of effects across individuals or groups  

Under the hood, ATE/ATT routines rely on **Double Machine Learning (DoubleML / IRM)** with sensible defaults  
(`CatBoost` learners) and **cross-fitting** to reduce bias from flexible nuisance models.

---

## A Very Short ATE Example

Below is the minimal flow using the internal IRM ATE estimator.  
It expects a `CausalData` object with one binary treatment, one outcome, and a list of confounders.

```python
from causalis.scenarios.unconfoundedness.ate import dml_ate

# Assume you already constructed a CausalData object: `causal_data`
# (see the User Guide pages for data preparation and EDA)

results = dml_ate(causal_data, n_folds=5, confidence_level=0.95)

print("ATE (coefficient):", results["coefficient"])      # float
print("Std. error:", results["std_error"])               # float
print("P-value:", results["p_value"])                    # float
print("95% CI:", results["confidence_interval"])         # (lower, upper)
````

### What It Returns

```text
{
  'coefficient': float,                  # estimated average treatment effect
  'std_error': float,                    # standard error
  'p_value': float,                      # p-value for H0: effect == 0
  'confidence_interval': (float, float), # (lower, upper) at the requested level
  'model': IRM,                          # fitted IRM object for advanced inspection
  'diagnostic_data': dict                # comprehensive diagnostic information (optional)
}
```

---

## Advanced Usage

### Custom Machine Learning Models

You can pass your own ML models to replace the default CatBoost learners:

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

results = dml_ate(
    causal_data,
    ml_g=RandomForestRegressor(n_estimators=100, max_depth=5),
    ml_m=RandomForestClassifier(n_estimators=100, max_depth=5),
    n_folds=5,
    confidence_level=0.95
)
```

---

### Additional Parameters

The `dml_ate` function provides several additional parameters for fine-tuning:

```python
results = dml_ate(
    causal_data,
    n_folds=5,                      # number of cross-fitting folds
    n_rep=1,                        # number of repetitions (currently 1 supported)
    score="ATE",                    # "ATE" or "ATTE"
    confidence_level=0.95,          # confidence level for CI
    normalize_ipw=False,            # whether to normalize IPW terms
    trimming_rule="truncate",       # trimming approach for propensity
    trimming_threshold=1e-2,        # trimming threshold
    random_state=42,                # random seed for reproducibility
    store_diagnostic_data=True      # store comprehensive diagnostics
)
```

---

### Diagnostic Data

By default, `dml_ate` can store detailed diagnostic information useful for validation and refutation tests:

```python
results = dml_ate(causal_data, store_diagnostic_data=True)

# Access diagnostic data
diagnostics = results["diagnostic_data"]

# Available diagnostic information:
# - m_hat: estimated propensity scores
# - g0_hat: estimated outcome under control
# - g1_hat: estimated outcome under treatment
# - y: observed outcomes
# - d: observed treatment
# - x: confounders
# - psi: influence function values
# - psi_a, psi_b: score components
# - folds: fold assignments
# - score: score type used
# - normalize_ipw: whether IPW was normalized
# - trimming_threshold: trimming threshold used
# - p1: treatment prevalence
```

This diagnostic data enables comprehensive **refutation tests**, including overlap diagnostics, score validation, and sensitivity analysis.

```
```
