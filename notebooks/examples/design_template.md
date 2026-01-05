Data schema & **when to measure** each variable

**Table layout (wide format)**

| column | meaning                                                          | type      | when to measure                          |
| ------ | ---------------------------------------------------------------- | --------- | ---------------------------------------- |
| `id`   | unit identifier                                                  | int/str   | baseline                                 |
| `D`    | treatment indicator (0/1)                                        | int       | **t₁** (decision/exposure time)          |
| `Y`    | outcome (numeric or binary)                                      | float/int | **t₂ > t₁** (follow-up)                  |
| `X_*`  | **confounders** (all causes of both `D` and `Y` you can observe) | float/int | **t₀ < t₁** (strictly **pre-treatment**) |

**Timing rule of thumb (avoid post-treatment bias):**

```
t₀ (baseline): measure X (pre-treatment confounders only)
      ↓
t₁ (assign/observe D)
      ↓
t₂ (observe Y)  — make sure no X measured here leaks into the model
```

* Do **not** include mediators (variables affected by `D`) in `X`. DML/IRM assumes `X` is pre-treatment
* If panel data exist, freeze a **snapshot of X** right before t₁.


1. **Question & estimand**

* State causal question (e.g., “Effect of D on Y for the target population at t₁–t₂”).
* Choose estimand: **ATE** or **ATTE**

2. **Identification assumptions**

* **Uncofoundedness**: $((Y(1),Y(0)) \perp D \mid X)$ (with your chosen, pre-treatment (X)).
* **Overlap (positivity)**: $(0<e(X)=P(D=1\mid X)<1)$ almost surely.
* **Consistency/SUTVA**: well-defined treatment, no interference.
* **Score check**: psi_mean, derivatives, psi_kurtosis


3. **Report**:
* p-value and CI
* Assumptions tests
* Why were such variables chosen
* Quantity of units in research
* Conclusion for the decision-maker and what decisions would be made
