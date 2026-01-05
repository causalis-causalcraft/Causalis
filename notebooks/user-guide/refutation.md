# Refutation: quick checks and sensitivity

Use these tools after you estimate a causal effect (e.g., with DML/IRM) to pressure‑test your result. They cover placebo tests, overlap/positivity, score orthogonality, balance under IPW, and bias‑aware sensitivity.

Prerequisites
- A CausalData object (see the Inference and CausalData guides)
- An inference result dict like res = dml_ate(causal_data, store_diagnostic_data=True, ...)

Tip: For ATE the DR/AIPW score is ψ = (g1 − g0) + (Y − g1)·D/m − (Y − g0)·(1−D)/(1−m). In large samples E[ψ]=0 and θ̂ = E_n[ψ]. Diagnostics below check this and related ingredients.

## 1) Placebo and subset checks
- Goal: detect spurious signal and fragility to sample size.
- Expectation: with placebo outcome/treatment the effect should be near 0 with a large p‑value; under sub‑sampling the estimate should be stable within noise.

Python

```python
from causalis.scenarios.unconfoundedness.refutation import (
    refute_placebo_outcome, refute_placebo_treatment, refute_subset
)
from causalis.scenarios.unconfoundedness.ate import dml_ate

# Placeholder: build CausalData as in the guides
causal_data = ...  # your CausalData

# Placebo: replace Y with noise
placebo_y = refute_placebo_outcome(dml_ate, causal_data, random_state=42)
# Placebo: shuffle D
placebo_d = refute_placebo_treatment(dml_ate, causal_data, random_state=42)
# Subset: re‑estimate on a fraction of data
subset = refute_subset(dml_ate, causal_data, fraction=0.5, random_state=42)
print(placebo_y)  # {'theta': ~0, 'p_value': large}
```

## 2) Overlap (positivity) diagnostics
- Goal: verify usable overlap and stable IPW weights.
- Key metrics (summary keys):
  - edge_mass: share of m̂ near 0/1 (danger ⇒ exploding weights). Rule of thumb: for ε=0.01, yellow ≈ 2%, red ≈ 5% on either side; for ε=0.02, yellow ≈ 5%, red ≈ 10%.
  - ks: two‑sample KS on m̂ for treated vs control (0 best, 1 worst). Red if > 0.35.
  - auc: D predictability from X via m̂ ranks. Good overlap ⇒ ≈ 0.5.
  - ate_ess: effective sample size ratio in treated weights (higher is better; near 1 good).
  - ate_tails: tail indices like q99/median of treated weights; >10 caution, >100 red.
  - att_weights.ATT_identity_relerr: relative error of ATT odds identity (≤5% green; 5–10% yellow; >10% red).
  - clipping: how many propensities were clipped (large values signal poor overlap).
  - calibration: ECE/slope/intercept for m̂ probability calibration (ECE small is good).

Python

```python
from causalis.scenarios.unconfoundedness.refutation import run_overlap_diagnostics

res = ...  # your inference result dict
rep = run_overlap_diagnostics(res=res)  # or run_overlap_diagnostics(m_hat=m, D=D)
print(rep['summary'])
# Also available: rep['edge_mass'], rep['ks'], rep['auc'], rep['ate_ess'],
# rep['ate_tails'], rep['att_weights'], rep['clipping'], rep['calibration']
```

## 3) Score/orthogonality diagnostics (Neyman)
- Goal: confirm E[ψ]=0 holds out‑of‑sample and that ψ is insensitive to small nuisance perturbations.
- Readouts:
  - oos_moment_test: t‑stats oos_tstat_fold and oos_tstat_strict ≈ N(0,1); values near 0 are good.
  - orthogonality_derivatives: max |t| for directions g1, g0, m (ATE) or g0, m (ATT). Rule of thumb: max |t| ≲ 2 is okay.
  - influence_diagnostics: tail metrics of ψ (psi_p99_over_med, psi_kurtosis). Large values flag instability and heavy tails.

Python

```python
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics

res = ...  # your inference result dict
rep_score = run_score_diagnostics(res=res)
print(rep_score['summary'])
# Details: rep_score['oos_moment_test'], ['orthogonality_derivatives'], ['influence_diagnostics']
```

## 4) Uncofoundedness/balance checks
- Goal: see if IPW (for your estimand) balances covariates.
- Metrics:
  - balance_max_smd: max standardized mean difference across covariates (smaller is better; ≤0.1 common rule).
  - balance_frac_violations: share of covariates with SMD ≥ threshold (default 0.10).

Python

```python
from causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation import (
    run_uncofoundedness_diagnostics
)

res = ...  # your inference result dict
rep_uc = run_uncofoundedness_diagnostics(res=res)
print(rep_uc['summary'])
```

## 5) Sensitivity analysis to hidden cofounding
- Goal: widen intervals by a worst‑case additive bias derived from how ψ could move.
- Math: max_bias = sqrt(ν²)·se where ν² aggregates two channels (outcome cf_y and treatment cf_d) and their correlation ρ.
- Two entry points:
  1) sensitivity_analysis: set (cf_y, cf_d, ρ) directly.
  2) sensitivity_benchmark: derive (cf_y, cf_d, ρ) from omitted covariates Z, then plug into sensitivity_analysis.

Python

```python
from causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity import (
    sensitivity_analysis, sensitivity_benchmark
)

res = ...  # your inference result dict
# Direct knobs
print(sensitivity_analysis(res, cf_y=0.01, cf_d=0.01, rho=1.0, level=0.95))

# Calibrate from omitted set Z (e.g., a column name)
bench = sensitivity_benchmark(res, benchmarking_set=['tenure_months'])
print(bench)
print(sensitivity_analysis(res,
                           cf_y=float(bench['cf_y']),
                           cf_d=float(bench['cf_d']),
                           rho=float(bench['rho']),
                           level=0.95))
```

## 6) SUTVA (design prompts)
- These assumptions aren’t testable from data. Use the built‑in checklist to document design decisions.

```python
from causalis.scenarios.unconfoundedness.refutation import print_sutva_questions

print_sutva_questions()
```

## Minimal workflow (ATE)

```python
from causalis.scenarios.unconfoundedness.ate import dml_ate
from causalis.scenarios.unconfoundedness.refutation import (
    refute_placebo_outcome, refute_placebo_treatment, refute_subset
)
from causalis.scenarios.unconfoundedness.refutation import run_overlap_diagnostics
from causalis.scenarios.unconfoundedness.refutation.score.score_validation import run_score_diagnostics
from causalis.scenarios.unconfoundedness.refutation.uncofoundedness.uncofoundedness_validation import (
    run_uncofoundedness_diagnostics
)
from causalis.scenarios.unconfoundedness.refutation.uncofoundedness.sensitivity import sensitivity_analysis

# Build your CausalData as in the guides
causal_data = ...  # your CausalData

# Estimate effect (stores nuisances needed for diagnostics)
res = dml_ate(causal_data, n_folds=4, store_diagnostic_data=True, random_state=123)

# 1) Placebo + subset
_ = refute_placebo_outcome(dml_ate, causal_data, random_state=42)
_ = refute_placebo_treatment(dml_ate, causal_data, random_state=42)
_ = refute_subset(dml_ate, causal_data, fraction=0.5, random_state=42)

# 2) Overlap
ov = run_overlap_diagnostics(res=res);
print(ov['summary'])

# 3) Score/orthogonality
sc = run_score_diagnostics(res=res);
print(sc['summary'])

# 4) Balance
uc = run_uncofoundedness_diagnostics(res=res);
print(uc['summary'])

# 5) Sensitivity (simple example)
print(sensitivity_analysis(res, cf_y=0.02, cf_d=0.02, rho=0.5, level=0.95))
```

Notes
- Always stratify folds by D and enable clipping of extreme propensities; large clipping/edge masses indicate weak overlap.
- For ATT, focus on g0 and m diagnostics and the ATT weight identity in the overlap report.
- Heavy ψ tails often trace back to poor overlap; consider trimming, better learners, or different features.
