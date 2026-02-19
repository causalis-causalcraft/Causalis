# UnconfoundednessModel (IRM)

# 0) Assumptions

* **SUTVA / consistency:**
  - **No interference:** unit $i$’s outcome does not depend on other units’ treatment assignments.
  - **No hidden versions:** "treatment" and "control" correspond to well-defined interventions.
  - **Consistency:** the observed outcome equals the potential outcome under the realized treatment: $Y_i = Y_i(D_i)$.

* **Unconfoundedness (Conditional Independence Assumption):**
  Potential outcomes are independent of treatment assignment given observed confounders $X$:
  $$ (Y(1), Y(0)) \perp D \mid X $$
  This ensures that after adjusting for $X$, there are no unobserved factors affecting both treatment and outcome.

* **Overlap / positivity:**
  The probability of receiving treatment is strictly between 0 and 1 for all $X$:
  $$ 0 < \Pr(D=1 \mid X) < 1 $$
  In practice, this is enforced via trimming: $\hat{m}(X) \in [\varepsilon, 1-\varepsilon]$, where $\varepsilon$ is the `trimming_threshold`.

# 1) Data + target estimands

You observe i.i.d. units $(i=1, \dots, n)$ with:
* outcome $(Y_i \in \mathbb{R})$,
* treatment $(D_i \in \{0, 1\})$,
* confounders $(X_i \in \mathbb{R}^p)$.

Targets:
* **ATE (Average Treatment Effect):**
  $$ \tau = \mathbb{E}[Y_i(1) - Y_i(0)] $$
* **ATTE (Average Treatment Effect on the Treated):**
  $$ \tau_{att} = \mathbb{E}[Y_i(1) - Y_i(0) \mid D_i=1] $$

---

# 2) Cross-fitting (Sample Splitting)

The data is partitioned into $K$ folds $\{I_k\}_{k=1}^K$ (typically `n_folds=5`). For each fold $k$:
1. Train nuisance models $\hat{g}_{0,k}, \hat{g}_{1,k}, \hat{m}_k$ using all data *except* fold $k$ ($I_k^c$).
2. Generate out-of-sample predictions for units in fold $k$:
   $$ \hat{g}_0(X_i) = \hat{g}_{0,k}(X_i), \quad \hat{g}_1(X_i) = \hat{g}_{1,k}(X_i), \quad \hat{m}(X_i) = \hat{m}_k(X_i) \quad \text{for } i \in I_k $$

---

# 3) Nuisance function estimation

The IRM estimator requires three nuisance components:
* **Outcome models:**
  $$ g_1(X) = \mathbb{E}[Y \mid X, D=1], \quad g_0(X) = \mathbb{E}[Y \mid X, D=0] $$
* **Propensity score:**
  $$ m(X) = \Pr(D=1 \mid X) $$

These are typically estimated using ML models like CatBoost or Random Forest. Propensity scores are clipped to $[\varepsilon, 1-\varepsilon]$ to ensure stability.

---

# 4) Scores and Moment Equations (AIPW/DR)

Define residuals:
$$ u_1 = Y - \hat{g}_1, \quad u_0 = Y - \hat{g}_0 $$

Define (optionally normalized) IPW terms:
$$ h_1 = \frac{D}{\hat{m}}, \quad h_0 = \frac{1-D}{1-\hat{m}} $$

The estimator solves the sample moment equation $\mathbb{E}_n[\psi_a \theta + \psi_b] = 0$.

* **ATE score (Doubly Robust):**
  $$ \psi_a = -1, \quad \psi_b = (\hat{g}_1 - \hat{g}_0) + u_1 h_1 - u_0 h_0 $$
  If `normalize_ipw=True` (Hájek), $h_1$ and $h_0$ are scaled by their empirical means: $\bar{h}_1 = h_1 / \mathbb{E}_n[h_1]$.

* **ATTE score:**
  Let $p_1 = \mathbb{E}[D]$.
  $$ \psi_a = -\frac{D}{p_1}, \quad \psi_b = \frac{D}{p_1}(\hat{g}_1 - \hat{g}_0) + \frac{D}{p_1}u_1 - \frac{1-D}{p_1} \frac{\hat{m}}{1-\hat{m}} u_0 $$

---

# 5) Point estimate and Influence Functions

The treatment effect estimate $\hat{\theta}$ is the solution to the moment equation:
$$ \hat{\theta} = -\frac{\mathbb{E}_n[\psi_b]}{\mathbb{E}_n[\psi_a]} $$

The **Influence Function (IF)** for unit $i$ (which captures the contribution of that unit to the total estimate) is:
$$ \psi_i = -\frac{\psi_{b,i} + \psi_{a,i} \hat{\theta}}{\mathbb{E}_n[\psi_a]} $$

By construction, $\mathbb{E}_n[\psi_i] = 0$.

---

# 6) Variance and Robust Standard Errors

The asymptotic variance is estimated as the sample variance of the influence function:
$$ \widehat{\mathrm{Var}}(\hat{\theta}) = \frac{1}{n^2} \sum_{i=1}^n \psi_i^2 $$
The standard error is:
$$ \widehat{\mathrm{SE}}(\hat{\theta}) = \sqrt{\widehat{\mathrm{Var}}(\hat{\theta})} $$

Confidence Interval (at level $1-\alpha$):
$$ \mathrm{CI}_{abs} = \hat{\theta} \pm z_{1-\alpha/2} \widehat{\mathrm{SE}}(\hat{\theta}) $$

---

# 7) Relative effect (Delta method)

The relative effect (%) is defined as:
$$ \hat{\tau}_{rel} = 100 \cdot \frac{\hat{\theta}}{\hat{\mu}_c} $$
where $\hat{\mu}_c$ is the estimated baseline outcome:
- **ATE:** $\hat{\mu}_c = \mathbb{E}_n[\psi_{\mu_c}]$ where $\psi_{\mu_c} = \hat{g}_0 + u_0 h_0$
- **ATTE:** $\hat{\mu}_c = \mathbb{E}_n[\psi_{\mu_c}]$ where $\psi_{\mu_c} = \frac{D}{p_1}\hat{g}_0 + \frac{1-D}{p_1} \frac{\hat{m}}{1-\hat{m}} u_0$

Let $\hat{\psi}_{\mu, i} = \psi_{\mu_c, i} - \hat{\mu}_c$. The IF for the relative effect (using the delta method) is:
$$ \psi_{rel, i} = 100 \cdot \left( \frac{\psi_i}{\hat{\mu}_c} - \frac{\hat{\theta} \hat{\psi}_{\mu, i}}{\hat{\mu}_c^2} \right) $$

This formulation correctly handles the covariance between the treatment effect and the baseline mean.
$$ \widehat{\mathrm{SE}}(\hat{\tau}_{rel}) = \sqrt{\frac{1}{n^2} \sum \psi_{rel, i}^2} $$

---

# 8) Math pseudocode 

```text
Input: (Y_i, D_i, X_i)_{i=1}^n, score ∈ {ATE, ATTE}, alpha, trimming_threshold=ε

1) Cross-fitting:
   Split {1..n} into K folds I_1, ..., I_K
   For k = 1..K:
     Train g0, g1, m on I_k^c
     Predict g0_hat, g1_hat, m_hat on I_k
   
2) Trimming:
   m_hat ← clip(m_hat, ε, 1-ε)

3) Score ingredients:
   u1 ← Y - g1_hat
   u0 ← Y - g0_hat
   If score == ATE:
     h1 ← D / m_hat, h0 ← (1-D) / (1-m_hat)
     (Optionally normalize h1, h0 by their means)
     psi_a ← -1
     psi_b ← (g1_hat - g0_hat) + u1*h1 - u0*h0
     psi_muc ← g0_hat + u0*h0
   Else if score == ATTE:
     p1 ← mean(D)
     psi_a ← -D / p1
     psi_b ← (D/p1)*(g1_hat - g0_hat) + (D/p1)*u1 - ((1-D)/p1)*(m_hat/(1-m_hat))*u0
     psi_muc ← (D/p1)*g0_hat + ((1-D)/p1)*(m_hat/(1-m_hat))*u0

4) Point estimates:
   theta_hat ← -mean(psi_b) / mean(psi_a)
   mu_c_hat ← mean(psi_muc)

5) Influence functions:
   IF ← -(psi_b + psi_a * theta_hat) / mean(psi_a)
   IF_mu ← psi_muc - mu_c_hat

6) Inference:
   se ← sqrt( mean(IF^2) / n )
   CI ← theta_hat ± z * se
   
7) Relative effect:
   tau_rel ← 100 * theta_hat / mu_c_hat
   IF_rel ← 100 * (IF / mu_c_hat - theta_hat * IF_mu / mu_c_hat^2)
   se_rel ← sqrt( mean(IF_rel^2) / n )
   CI_rel ← tau_rel ± z * se_rel
```


---

# References

## 1) IRM / DML core: cross-fitting + orthogonal scores (your `fit()` + `psi_a/psi_b` moment equation)

* **Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins (2018)** — *Double/debiased machine learning for treatment and structural parameters* (Econometrics Journal). This is the canonical DML reference that (i) formalizes **Neyman-orthogonal scores**, (ii) motivates and analyzes **cross-fitting**, and (iii) includes the **IRM score** for ATE/ATT under unconfoundedness (exactly the structure your `psi_b` is implementing). ([OUP Academic][1])

* **Chernozhukov et al. (2016/2017)** earlier accessible versions / summaries of the same DML program (useful if you want an arXiv cite alongside the journal cite). ([OUP Academic][2])

* **Newey & McFadden (1994)** — large-sample theory for **M/Z-estimators** (solving moments, Jacobian (J), influence function / sandwich variance). This matches your `_solve_moment_equation()` pattern: (\hat\theta) from an estimating equation + IF (= -(\psi(\hat\theta))/J). ([Cambridge Assets][3])

---

## 2) ATE / ATTE efficient influence functions and doubly-robust/AIPW structure (your `psi_b` form)

* **Hahn (1998)** — *On the role of the propensity score in efficient semiparametric estimation of average treatment effects* (Econometrica). Derives efficiency bounds and EIF structure; also highlights that ATE vs ATT behave differently (your ATTE weighting structure aligns with this literature). ([EconPapers][4])

* **Hirano, Imbens & Ridder (2003)** — efficient estimation of ATE with (estimated) propensity score; foundational for modern IPW/AIPW practice under unconfoundedness. ([NBER][5])

* **Bang & Robins (2005)** — classic **doubly-robust** estimators for missing-data/causal models (Biometrics). This is the standard cite for the DR/AIPW form you’re using (outcome regression + propensity correction). ([Wiley Online Library][6])

* **Rosenbaum & Rubin (1983)** — propensity score + ignorability framework (the identifying assumption your IRM scenario relies on). ([OUP Academic][7])

---

## 3) IPW ancestry + Hájek (normalized) vs Horvitz–Thompson (your `normalize_ipw` option and the warning about “ratio-style” inference)

* **Horvitz & Thompson (1952)** — the original **HT** inverse-probability weighting estimator lineage. This is the standard historical/technical cite for IPW. ([CMU Stats & Data Science][8])

* **Hájek ratio/normalized estimator** — “Hájek” normalization is the ratio form (normalize weights / denominators). A clean modern citation that explicitly discusses the Hájek ratio estimator (and uses the term in print) is Aronow et al. (2017, AOAS) while referencing Hájek’s original work. ([Project Euclid][9])
  (If you want a variance-estimation focused reference for Hájek-style ratios, this also exists in applied survey settings.) ([ERIC][10])

---

## 4) Trimming / overlap (your `_clip_propensity` + `trimming_threshold`)

* **Crump, Hotz, Imbens & Mitnik (2009)** — *Dealing with limited overlap…* (Biometrika). This is the go-to cite for why trimming/extreme propensities matter and for principled trimming rules. ([Duke Economics][11])

---

## 5) GATE / BLP on an orthogonal signal (your `gate()` that regresses `orth_signal` on group basis)

* **Chernozhukov, Demirer, Duflo, Fernández-Val (2018/2020)** — *Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments* (widely circulated working paper/PDF). This is the standard cite for **BLP / GATES** style inference built on an **orthogonalized score/pseudo-outcome**. ([Taylor & Francis Online][12])

(Your exact setup—compute an orthogonal signal, then do a linear projection on a basis of group indicators—is directly in this family.)

---

## 6) Sensitivity analysis with (R^2)-style parameters (your `sensitivity_analysis()` + `_sensitivity_element_est()` objects like Riesz representer / (\nu^2))

* **Chernozhukov, Cinelli, Newey, Sharma, Syrgkanis (NBER 2022; arXiv 2021/2024 versions)** — *Omitted Variable Bias in Causal Machine Learning* (includes the “long story short” version; develops sensitivity/OVB theory using (R^2)-type parameterizations and constructs the key components that show up as “sensitivity elements”). ([NBER][13])

---

[1]: https://academic.oup.com/ectj/article/21/1/C1/5056401 "Double/debiased machine learning for treatment and structural parameters | The Econometrics Journal | Oxford Academic"
[2]: https://academic.oup.com/ectj/article/21/1/C1/5056401?utm_source=chatgpt.com "Double/debiased machine learning for treatment and ..."
[3]: https://assets.cambridge.org/97805217/84504/index/9780521784504_index.pdf?utm_source=chatgpt.com "Asymptotic Statistics AW van der Vaart Index"
[4]: https://econpapers.repec.org/RePEc%3Aecm%3Aemetrp%3Av%3A66%3Ay%3A1998%3Ai%3A2%3Ap%3A315-332?utm_source=chatgpt.com "On the Role of the Propensity Score in Efficient ..."
[5]: https://www.nber.org/papers/t0251?utm_source=chatgpt.com "Efficient Estimation of Average Treatment Effects Using the ..."
[6]: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1541-0420.2005.00377.x?utm_source=chatgpt.com "Doubly Robust Estimation in Missing Data and Causal ..."
[7]: https://academic.oup.com/biomet/article-abstract/70/1/41/240879?utm_source=chatgpt.com "Propensity score in observational studies for causal effects"
[8]: https://www.stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf?utm_source=chatgpt.com "Horvitz-Thompson-1952-jasa.pdf"
[9]: https://projecteuclid.org/journals/annals-of-applied-statistics/volume-11/issue-4/Estimating-average-causal-effects-under-general-interference-with-application-to/10.1214/16-AOAS1005.pdf?utm_source=chatgpt.com "Estimating average causal effects under general ..."
[10]: https://files.eric.ed.gov/fulltext/EJ1168725.pdf?utm_source=chatgpt.com "Applying the Hájek Approach in Formula-Based Variance ..."
[11]: https://public.econ.duke.edu/~vjh3/working_papers/overlap.pdf "overlap_08may16.DVI"
[12]: https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476818?utm_source=chatgpt.com "Estimation of Regression Coefficients When Some ..."
[13]: https://www.nber.org/system/files/working_papers/w30302/w30302.pdf?utm_source=chatgpt.com "OMITTED VARIABLE BIAS IN CAUSAL MACHINE ..."
