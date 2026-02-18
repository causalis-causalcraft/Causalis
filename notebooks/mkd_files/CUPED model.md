# CUPEDModel

# 0) Assumptions

-   SUTVA / consistency
  * **No interference:** unit $i$’s outcome does not depend on other units’ treatment assignments.
  * **No hidden versions:** “treatment” and “control” correspond to well-defined interventions.
  * **Consistency:** the observed outcome equals the potential outcome under the realized treatment: if $D_i=1$, then $Y_i=Y_i(1)$; if $D_i=0$, then $Y_i=Y_i(0)$.

* **Random assignment or Unconfoundedness (ATE/ITT identification):** In an RCT, treatment is independent of potential outcomes (and baseline covariates):
  $$
  (Y_i(1),Y_i(0)) \perp D_i.
  $$
  This is the key condition under which the estimand $\tau=\mathbb{E}[Y(1)-Y(0)]$ is identified as an average causal effect.

* **Overlap / positivity:** Both arms occur with nonzero probability:
  $$
  0<\Pr(D_i=1)<1,
  $$
  (and within any randomization strata if stratified), ensuring the ATE/ITT is estimable from the observed design.

* **Regression is a working model (robust inference):** The Lin specification is used as an **adjustment**; it need not be exactly correct. HC2 standard errors target valid large-sample inference under heteroskedasticity and potential misspecification of the conditional mean.

* **Finite second moments:** Outcomes and regressors have finite second moments (so variances exist), which supports HC-type variance estimation and the delta-method approximation used later.

* **Design matrix regularity:** The constructed design matrix
  $$
  Z=[\mathbf{1},, d,, X^c,, d\odot X^c]
  $$
  has full column rank (no perfect multicollinearity), so $(Z^\top Z)^{-1}$ exists. In practice this also means you avoid near-duplicate covariates and handle zero-variance / constant columns.

* **Leverage not degenerate (HC2 well-defined):** For HC2 you require $1-h_{ii}>0$ for all $i$ (true when $Z$ has full rank and no observation is perfectly fit), so the HC2 weights $\omega_i=\hat e_i^2/(1-h_{ii})$ are finite.

* **Relative-effect CI (delta method, “nocov”):** For the relative CI you additionally assume a first-order Taylor approximation is adequate, $\hat\mu_c$ is not near zero, and you **ignore** $\mathrm{Cov}(\hat\tau,\hat\mu_c)$ (your deliberate “nocov” rule).

# 1) Data + target estimand

You observe i.i.d. units $(i=1,\dots,n)$ with:

* outcome $(Y_i \in \mathbb{R})$ (post-period),
* treatment $(D_i \in {0,1})$,
* pre-treatment covariates $(X_i \in \mathbb{R}^p)$ (chosen subset).

Target (ATE/ITT):
$$
\tau = \mathbb{E}!\left[Y_i(1)-Y_i(0)\right].
$$

---

# 2) **Global** centering of covariates (full-sample centering)

Let $p^\star$ be the number of covariates you actually keep (after any variance / quality filtering). For each kept covariate $(j=1,\dots,p^\star)$, center over the **entire sample**:
$$
X^c_{ij} := X_{ij}-\bar X_j,
\qquad
\bar X_j=\frac{1}{n}\sum_{i=1}^n X_{ij}.
$$

In matrix form, with $(\mathbf{1}\in\mathbb{R}^n)$:
$$
X^c = X-\mathbf{1}\bar X^\top,
\qquad
\bar X=\frac{1}{n}X^\top \mathbf{1}.
$$

Key property (by construction):
$$
\frac{1}{n}\sum_{i=1}^n X^c_{ij}=0
\quad \text{for all } j
\qquad\Longleftrightarrow\qquad
\mathbb{E}_n[X^c]=0.
$$

**Why this matters for interpretation (with interactions):**

With global centering, the regression intercepts line up at the *sample mean covariate level*. In particular, the “main” treatment coefficient becomes the **difference in intercepts at $X=\bar X$**, which is exactly what you want when you interpret $\hat\tau$ as the ATE/ITT under Lin’s fully-interacted adjustment.

If you **don’t** center and estimate:
$$
Y_i=\alpha+\tau D_i+X_i^\top\beta + D_i X_i^\top\gamma+\varepsilon_i,
$$
then the interaction term vanishes at $X=0$, so
$$
\tau = \text{treatment effect at } X=0,
$$
which is often not a meaningful reference point (age $=0$, revenue $=0$, etc.). The regression can still be statistically valid, but $\tau$ no longer directly matches the “average-over-the-covariate-distribution” effect unless the covariates happen to have mean zero.

---

# 3) Build the Lin (2013) fully-interacted design matrix $(Z)$

Let
$$
d=(D_1,\dots,D_n)^\top\in\mathbb{R}^n.
$$

Define the elementwise (row-wise) interaction matrix $d\odot X^c\in\mathbb{R}^{n\times p^\star}$ by
$$
(d\odot X^c)_{ij} = D_iX^c_{ij}.
$$

Then the Lin (2013) fully-interacted design matrix is:
$$
Z
=

\begin{bmatrix}
\mathbf{1} & d & X^c & (d\odot X^c)
\end{bmatrix}
\in\mathbb{R}^{n\times k},
\qquad
k=2+2p^\star.
$$

Partition the parameter vector as:
$$
\theta=
\begin{bmatrix}
\alpha\
\tau\
\beta\
\gamma
\end{bmatrix},
\qquad
\beta,\gamma\in\mathbb{R}^{p^\star}.
$$

Row-wise model:
$$
Y_i
=
\alpha
+\tau D_i
+(X_i^c)^\top\beta
+D_i (X_i^c)^\top\gamma
+\varepsilon_i.
$$

---

# 4) Why the coefficient on $(D)$ is the ATE (with global centering)

The regression implies two group-specific conditional mean functions:

* Control $(D=0)$:
  $$
  \mathbb{E}[Y\mid X^c, D=0]=\alpha+(X^c)^\top\beta.
  $$

* Treated $(D=1)$:
  $$
  \mathbb{E}[Y\mid X^c, D=1]=(\alpha+\tau)+(X^c)^\top(\beta+\gamma).
  $$

So the **conditional treatment effect** as a function of covariates is:
$$
\Delta(X^c)
=
\mathbb{E}[Y\mid X^c,1]-\mathbb{E}[Y\mid X^c,0]
\tau+(X^c)^\top\gamma.
$$

Now average this conditional effect over the covariate distribution used by the estimator:
$$
\mathbb{E}[\Delta(X^c)]
=
\tau+\mathbb{E}[X^c]^\top\gamma.
$$

With **global centering**, $\mathbb{E}[X^c]=0$ (in sample: $\mathbb{E}_n[X^c]=0$), hence:
$$
\mathbb{E}[\Delta(X^c)] = \tau.
$$

So, under the globally-centered Lin specification:

* $\tau$ is the **average** treatment effect over the (centered) covariate distribution (ATE/ITT),
* $\gamma$ captures **effect heterogeneity** with respect to $X$.

---

# 5) OLS fit (point estimate)

Fit OLS on $(Y,Z)$:
$$
\hat\theta
=
\arg\min_{\theta},|Y-Z\theta|_2^2
(Z^\top Z)^{-1}Z^\top Y.
$$

The reported ATE/ITT estimate is the coefficient on the treatment column:
$$
\hat\tau = \hat\theta_{\text{(column of }d\text{)}}.
$$

Residuals:
$$
\hat e = Y-Z\hat\theta,
\qquad
\hat e_i = Y_i - Z_i^\top\hat\theta.
$$

---

# 6) Robust covariance / standard error for $(\hat\tau)$ (HC2 only)

Let the OLS “bread” be:
$$
B=(Z^\top Z)^{-1}.
$$

Define the hat matrix and leverages:
$$
H=Z(Z^\top Z)^{-1}Z^\top,
\qquad
h_{ii}=H_{ii}.
$$

**HC2 weights**:
$$
\omega_i=\frac{\hat e_i^2}{1-h_{ii}}.
$$

Define the “meat”:
$$
M
=
\sum_{i=1}^n \omega_i, Z_i Z_i^\top
Z^\top \operatorname{diag}(\omega_1,\dots,\omega_n) Z.
$$

Robust covariance:
$$
\widehat{\mathrm{Var}}(\hat\theta)
=
BMB.
$$

Extract the treatment-component variance and standard error:
$$
\widehat{\mathrm{Var}}(\hat\tau)
=
\left[\widehat{\mathrm{Var}}(\hat\theta)\right]_{\tau\tau},
\qquad
\widehat{\mathrm{SE}}(\hat\tau)=\sqrt{\widehat{\mathrm{Var}}(\hat\tau)}.
$$

A generic test statistic is:
$$
t=\frac{\hat\tau}{\widehat{\mathrm{SE}}(\hat\tau)}
$$

Absolute CI (using some critical value $c_\alpha$):
$$
\mathrm{CI}_{abs}
=
\hat\tau \pm c_\alpha,\widehat{\mathrm{SE}}(\hat\tau)
$$

If your implementation later “recovers” the effective $c_\alpha$ from the computed CI bounds, one way to express it is:
$$
c_\alpha
=
\frac{\max{|\mathrm{CI}_{hi}-\hat\tau|,|\hat\tau-\mathrm{CI}_{lo}|}}
{\widehat{\mathrm{SE}}(\hat\tau)}.
$$

---

# 7) Relative CI  delta method

Define the relative effect (percent) using the control mean:
$$
\hat\mu_c=\frac{1}{n_c}\sum_{i:D_i=0}Y_i,
\qquad
\hat\tau_{rel}=100\cdot\frac{\hat\tau}{\hat\mu_c}.
$$

Let
$$
g(\tau,\mu)=100\cdot\frac{\tau}{\mu}.
$$

Gradient:
$$
\frac{\partial g}{\partial \tau}=\frac{100}{\mu},
\qquad
\frac{\partial g}{\partial \mu}=-100\cdot\frac{\tau}{\mu^2}
$$

You already have $\widehat{\mathrm{Var}}(\hat\tau)$ from HC2:
$$
\widehat{\mathrm{Var}}(\hat\tau)=\left[\widehat{\mathrm{Var}}(\hat\theta)\right]_{\tau\tau}
$$

Estimate the variance of the control mean with the usual sample-mean formula:
$$
\widehat{\mathrm{Var}}(\hat\mu_c)=\frac{s_c^2}{n_c},
\qquad
s_c^2=\frac{1}{n_c-1}\sum_{i:D_i=0}(Y_i-\hat\mu_c)^2.
$$

**Delta method, ignoring covariance** $\mathrm{Cov}(\hat\tau,\hat\mu_c)=0$ (“nocov” rule):
$$
\widehat{\mathrm{Var}}(\hat\tau_{rel})
\approx
\left(\frac{100}{\hat\mu_c}\right)^2\widehat{\mathrm{Var}}(\hat\tau)
+
\left(100\cdot\frac{\hat\tau}{\hat\mu_c^2}\right)^2\widehat{\mathrm{Var}}(\hat\mu_c)
$$

So:
$$
\widehat{\mathrm{SE}}(\hat\tau_{rel})
=
\sqrt{\widehat{\mathrm{Var}}(\hat\tau_{rel})}
$$

Use the **same** critical value $c_\alpha$ as for the absolute CI:
$$
\mathrm{CI}_{rel}
=
\hat\tau_{rel} \pm c_\alpha\widehat{\mathrm{SE}}(\hat\tau_{rel})
$$

---

# 8) Math pseudocode (only math)

```text
Input: (Y_i, D_i, X_i)_{i=1}^n, alpha, variance_min=ε, cov_type=HC2, B(optional)

1) Keep covariates:
   S ← { j : Var_hat(X_{·j}) > ε }
   X ← X_{·S}   (p* = |S|)

2) Global centering:
   X^c ← X − 1 · ( (1/n) · X^T · 1 )^T

3) Design matrix:
   Z ← [ 1 , d , X^c , d ⊙ X^c ]

4) OLS:
   θ_hat ← (Z^T Z)^{-1} Z^T Y
   τ_hat ← component of θ_hat corresponding to d

5) Robust covariance (HC2):
   e_hat ← Y − Z θ_hat
   H ← Z (Z^T Z)^{-1} Z^T
   h_ii ← diag(H)
   ω_i ← e_hat_i^2 / (1 − h_ii)
   V_hat(θ_hat) ← (Z^T Z)^{-1} ( Σ_i ω_i Z_i Z_i^T ) (Z^T Z)^{-1}
   se(τ_hat) ← sqrt( [V_hat]_{ττ} )

6) Absolute CI:
   τ_hat ± c_alpha · se(τ_hat)

7) Relative effect:
   μc_hat ← mean(Y_i : D_i=0)
   τrel_hat ← 100 τ_hat / μc_hat

8) Relative CI (delta_nocov):
   Var_hat(μc_hat) ← s_c^2 / n_c
   Var_hat(τrel_hat) ← (100/μc_hat)^2 Var_hat(τ_hat)
                      + (100 τ_hat / μc_hat^2)^2 Var_hat(μc_hat)
   τrel_hat ± c_alpha · sqrt(Var_hat(τrel_hat))

```

---
# References
## 1) Data + target estimand (potential outcomes, ATE/ITT)

- **Neyman (1923; English translation 1990)** — foundational “Neyman model” for randomized experiments; potential outcomes framing and unbiased difference-in-means for average effects. ([mimuw.edu.pl](https://www.mimuw.edu.pl/~noble/courses/BayesianNetworks/90NeymanTranslation.pdf?utm_source=chatgpt.com "On the Application of Probability Theory to Agricultural ..."))
    
- **Rubin (1974)** — formal potential outcomes / causal effects language for randomized and nonrandomized studies; ATE as a target estimand. (Journal of Educational Psychology; DOI: 10.1037/h0037350.) ([Demographic Research](https://www.demographic-research.org/articles/savereference?format=bibtex&reference=82319&utm_source=chatgpt.com "BibTeX"))
    
- **Holland (1986)** — classic “Statistics and Causal Inference”; clarifies potential outcomes notation and estimands like ATE. ([JSTOR](https://www.jstor.org/stable/2289064?utm_source=chatgpt.com "Statistics and Causal Inference"))
    
- **Imbens (2004)** — explicit discussion of **average treatment effects** as estimands (broader than RCTs, but standard for ATE notation/targets). ([MIT Press Direct](https://direct.mit.edu/rest/article/86/1/4/57476/Nonparametric-Estimation-of-Average-Treatment?utm_source=chatgpt.com "Nonparametric Estimation of Average Treatment Effects"))
    

---

## 2) Global centering of covariates (full-sample centering) + why it matters with interactions

- **Lin (2013)** — uses the fully-interacted regression adjustment (Lin estimator) and discusses centering (often stated “without loss of generality, center covariates”) to interpret the main treatment coefficient as an average effect. ([Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-7/issue-1/Agnostic-notes-on-regression-adjustments-to-experimental-data--Reexamining/10.1214/12-AOAS583.full?utm_source=chatgpt.com "Agnostic notes on regression adjustments to experimental ..."))
    
- **Brambor, Clark & Golder (2006)** — clear explanation of interaction models: main effects are evaluated at the moderator equal to **0**, and mean-centering shifts the reference point to the mean (interpretation changes, fitted values don’t). Great citation for your “$\tau$ is the effect at $X=0$ if uncentered” statement. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/journals/political-analysis/article/understanding-interaction-models-improving-empirical-analyses/9BA57B3720A303C61EBEC6DDFA40744B?utm_source=chatgpt.com "Understanding Interaction Models: Improving Empirical ..."))
    
- **Lei & Ding (2021)** — explicitly notes centering covariates (wlog) in the Lin regression-adjustment setup and studies its properties. ([NSF Public Access Repository](https://par.nsf.gov/servlets/purl/10234039?utm_source=chatgpt.com "Regression adjustment in completely randomized ..."))
    

---

## 3) Build the Lin (2013) fully-interacted design matrix (Z=[1, D, X^c, D\odot X^c])

- **Lin (2013)** — the canonical reference for the “fully interacted” OLS adjustment in experiments (treatment, covariates, and treatment×covariate interactions). ([Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-7/issue-1/Agnostic-notes-on-regression-adjustments-to-experimental-data--Reexamining/10.1214/12-AOAS583.full?utm_source=chatgpt.com "Agnostic notes on regression adjustments to experimental ..."))
    
- **Lei & Ding (2021)** — modern formalization/extension of Lin’s regression adjustment; repeats the same design structure in a randomized-experiment framework. ([IDEAS/RePEc](https://ideas.repec.org/a/oup/biomet/v108y2021i4p815-828..html?utm_source=chatgpt.com "Regression adjustment in completely randomized ..."))
    

---

## 4) Why the coefficient on (D) is the ATE/ITT (with global centering + interactions)

- **Lin (2013)** — main theoretical justification: with treatment×covariate interactions, the OLS coefficient on treatment targets the average treatment effect (and can’t hurt asymptotic precision under the Neyman model). ([Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-7/issue-1/Agnostic-notes-on-regression-adjustments-to-experimental-data--Reexamining/10.1214/12-AOAS583.full?utm_source=chatgpt.com "Agnostic notes on regression adjustments to experimental ..."))
    
- **Freedman (2008)** — motivates why naive regression adjustment can misbehave and why one should be careful; this is exactly the critique Lin reexamines (useful for your “why we do Lin spec” story). ([JSTOR](https://www.jstor.org/stable/30244182?utm_source=chatgpt.com "On Regression Adjustments in Experiments with Several ..."))
    
- **Lei & Ding (2021)** — provides additional asymptotic results/guarantees for the Lin adjustment under regimes with many covariates (supporting your “(\hat\tau) is ATE/ITT under this spec” claim). ([IDEAS/RePEc](https://ideas.repec.org/a/oup/biomet/v108y2021i4p815-828..html?utm_source=chatgpt.com "Regression adjustment in completely randomized ..."))
    

---

## 5) OLS fit (point estimate) for (\hat\tau) under this regression-adjustment estimator

You typically don’t need a separate “OLS paper” citation if you already cite the **experimental regression-adjustment** papers that _define_ the estimator as OLS on that (Z).

- **Lin (2013)** — defines the estimator via OLS on ([1,D,X,D\cdot X]). ([Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-7/issue-1/Agnostic-notes-on-regression-adjustments-to-experimental-data--Reexamining/10.1214/12-AOAS583.full?utm_source=chatgpt.com "Agnostic notes on regression adjustments to experimental ..."))
    
- **Freedman (2008)** — discusses regression adjustment in experiments (OLS adjustment) and what it is / isn’t justified by randomization. ([Department of Statistics](https://www.stat.berkeley.edu/~census/neyregr.pdf?utm_source=chatgpt.com "On regression adjustments to experimental data David A ..."))


---

## 6) Robust covariance / standard error for (\hat\tau) (HC2 only)

- **White (1980)** — original heteroskedasticity-consistent “sandwich” covariance for OLS (foundation for HC estimators). ([JSTOR](https://www.jstor.org/stable/1912934?utm_source=chatgpt.com "A Heteroskedasticity-Consistent Covariance Matrix ..."))
    
- **MacKinnon & White (1985)** — introduces the HC family **including HC2** leverage adjustment (\hat e_i^2/(1-h_{ii})); this is your key HC2 citation. (J. Econometrics; DOI: 10.1016/0304-4076(85)90158-7.) ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/0304407685901587?utm_source=chatgpt.com "Some heteroskedasticity-consistent covariance matrix ..."))
    
- **Zeileis (2004)** — widely cited computational/econometrics reference that summarizes HC estimators and their implementations (nice for “HC2 definition in practice”). ([jstatsoft.org](https://www.jstatsoft.org/article/view/v011i10/19?utm_source=chatgpt.com "Econometric Computing with HC and HAC Covariance ..."))
    
- **Long & Ervin (2000)** — practical discussion of using heteroskedasticity-consistent SEs and finite-sample considerations (optional, but commonly cited). ([JSTOR](https://www.jstor.org/stable/2685594?utm_source=chatgpt.com "Using Heteroscedasticity Consistent Standard Errors in the"))
    

---

## 7) Relative CI via delta method (your “nocov” delta option)

- **Oehlert (1992)** — short classic note reviewing the delta method and when it works well (perfect for citing your Taylor/gradient variance approximation). ([Taylor & Francis Online](https://www.tandfonline.com/doi/abs/10.1080/00031305.1992.10475842?utm_source=chatgpt.com "A Note on the Delta Method: The American Statistician"))
    
- **Deng et al. (2018)** — applied “metric analytics / online experiments” reference that explicitly uses the delta method for **percent change / relative lift** style estimands (very aligned with your (\hat\tau_{rel}=100\hat\tau/\hat\mu_c)). ([Alex Deng](https://alexdeng.github.io/public/files/kdd2018-dm.pdf?utm_source=chatgpt.com "Applying the Delta Method in Metric Analytics: A Practical ..."))