# MultiTreatmentIRM

# 0) Assumptions

* **SUTVA / consistency**
  - No interference across units.
  - No hidden treatment versions.
  - Observed outcome matches the potential outcome under realized arm.

* **Multi-arm unconfoundedness**
  $$
  \left(Y(0), Y(1), \dots, Y(K-1)\right)\perp D \mid X
  $$
  where $D$ is one-hot with $K$ treatment columns (column 0 is baseline/control).

* **Positivity / overlap**
  $$
  \Pr(D_k=1\mid X=x)>0,\quad \forall k\in\{0,\dots,K-1\}
  $$
  and in implementation this is stabilized by multiclass trimming.

* **Fold-level support for cross-fitting**
  Each training fold must contain all treatment arms; otherwise nuisance models for missing arms are not identifiable.

---

# 1) Data and estimand

For i.i.d. units $i=1,\dots,n$, observe:

* outcome: $Y_i\in\mathbb{R}$,
* confounders: $X_i\in\mathbb{R}^p$,
* one-hot treatment vector: $D_i=(D_{i0},\dots,D_{i,K-1})$ with $\sum_k D_{ik}=1$.

Target is a vector of baseline contrasts:
$$
\theta =
\begin{bmatrix}
\theta_1\\
\vdots\\
\theta_{K-1}
\end{bmatrix},
\qquad
\theta_k = \mathbb{E}[Y(k)-Y(0)],\ k=1,\dots,K-1.
$$

---

# 2) Nuisance functions

For each arm $k$:
$$
g_k(x)=\mathbb{E}[Y\mid X=x, D_k=1].
$$

Generalized propensity (multiclass):
$$
m_k(x)=\Pr(D_k=1\mid X=x),\quad
\sum_{k=0}^{K-1}m_k(x)=1.
$$

Cross-fitted predictions are denoted $\hat g_k(X_i)$ and $\hat m_k(X_i)$.

---

# 3) Cross-fitting

Split sample into folds $I_1,\dots,I_F$ (`n_folds=F`). For each fold $f$:

1. Train multiclass propensity model on $I_f^c$ and predict $\hat m_k$ on $I_f$.
2. For each arm $k$, train outcome model only on rows in $I_f^c$ with $D_k=1$, then predict $\hat g_k$ on $I_f$.

Binary-outcome edge case used in implementation:

If within a fold+arm the training outcome is single-class (all 0 or all 1), use a deterministic constant predictor for that arm/fold instead of fitting a classifier.

---

# 4) Trimming in multi-class setup

Implementation uses lower-bound trimming + row renormalization:
$$
\tilde m_{ik}=\max(\hat m_{ik}, \varepsilon),\qquad
\hat m^{trim}_{ik}=\frac{\tilde m_{ik}}{\sum_{j=0}^{K-1}\tilde m_{ij}}.
$$

So each row stays a valid probability simplex after trimming.  
Constraint: $\varepsilon < 1/K$.

---

# 5) Orthogonal score for multi-arm ATE

Define residuals for each arm:
$$
u_{ik}=Y_i-\hat g_k(X_i),\quad k=0,\dots,K-1.
$$

Define IPW representers:
$$
h_{ik}=\frac{D_{ik}}{\hat m^{trim}_{ik}}.
$$

If `normalize_ipw=True`, apply column-wise Hájek normalization:
$$
h_{ik}\leftarrow \frac{h_{ik}}{\mathbb{E}_n[h_{\cdot k}]}.
$$

For each contrast $k=1,\dots,K-1$:
$$
\psi_{b,ik}
=
\big(\hat g_k(X_i)-\hat g_0(X_i)\big)
+u_{ik}h_{ik}
-u_{i0}h_{i0}.
$$

Moment system:
$$
\mathbb{E}_n[\psi_a\theta+\psi_b]=0,\qquad \psi_a=-1.
$$

Thus with $J=\mathbb{E}_n[\psi_a]=-1$:
$$
\hat\theta_k = -\frac{\mathbb{E}_n[\psi_{b,\cdot k}]}{J}
=\mathbb{E}_n[\psi_{b,\cdot k}].
$$

---

# 6) Influence function and inference

Per-contrast influence function:
$$
\widehat{IF}_{ik}
=
-\frac{\psi_{b,ik}+\psi_a\hat\theta_k}{J}
=
\psi_{b,ik}-\hat\theta_k.
$$

Variance/SE:
$$
\widehat{\mathrm{Var}}(\hat\theta_k)=\frac{\mathrm{Var}_n(\widehat{IF}_{\cdot k})}{n},
\qquad
\widehat{SE}(\hat\theta_k)=\sqrt{\widehat{\mathrm{Var}}(\hat\theta_k)}.
$$

Wald interval:
$$
CI_k^{abs}
=
\hat\theta_k
\pm
z_{1-\alpha/2}\widehat{SE}(\hat\theta_k).
$$

P-values are computed per contrast via normal approximation; significance flag uses Bonferroni:
$$
p_k < \frac{\alpha}{K-1}.
$$

---

# 7) Relative effect

Baseline orthogonal signal:
$$
\psi_{\mu_c,i}
=
\hat g_0(X_i)+u_{i0}h_{i0},
\qquad
\hat\mu_c=\mathbb{E}_n[\psi_{\mu_c}].
$$

Relative effect (%):
$$
\hat\tau_k^{rel}=100\cdot \frac{\hat\theta_k}{\hat\mu_c}.
$$

Implementation uses a delta-style plug-in variance:
$$
d_\theta=\frac{100}{\hat\mu_c},\qquad
d_{\mu,k}=-\frac{100\hat\theta_k}{\hat\mu_c^2},
$$
$$
\widehat{\mathrm{Var}}(\hat\tau_k^{rel})
\approx
d_\theta^2\widehat{SE}(\hat\theta_k)^2
+d_{\mu,k}^2\widehat{SE}(\hat\mu_c)^2
+2d_\theta d_{\mu,k}\widehat{\mathrm{Cov}}(\hat\theta_k,\hat\mu_c),
$$
then
$$
CI_k^{rel}=\hat\tau_k^{rel}\pm z_{1-\alpha/2}\widehat{SE}(\hat\tau_k^{rel}).
$$

---

# 8) Math pseudocode

```text
Input:
  (Y_i, X_i, D_i)_{i=1}^n, one-hot D with K arms (0 is baseline)
  ml_g, ml_m, n_folds=F, alpha, trimming_threshold=eps

1) Validate:
   - D is one-hot, K >= 2
   - ml_m is probabilistic multiclass classifier
   - eps in [0, 1/K), n_folds feasible for class counts

2) Cross-fitting:
   Split indices into folds I_1,...,I_F stratified by argmax(D)
   For each fold f:
     Train ml_m on I_f^c and predict m_hat on I_f
     For each arm k in {0,...,K-1}:
       Select rows in I_f^c with D_k=1
       If Y is binary and selected Y is single-class:
         g_hat(I_f, k) <- that constant class
       Else:
         Train ml_g on selected rows
         Predict g_hat(I_f, k)

3) Trim multiclass propensity:
   m_tilde_ik <- max(m_hat_ik, eps)
   m_hat_ik <- m_tilde_ik / sum_j m_tilde_ij

4) Score ingredients:
   u_ik <- Y_i - g_hat_ik
   h_ik <- D_ik / m_hat_ik
   If normalize_ipw:
     h_ik <- h_ik / mean_i(h_ik)  (for each k)

5) Orthogonal score for each contrast k=1..K-1:
   psi_b_ik <- (g_hat_ik - g_hat_i0) + u_ik*h_ik - u_i0*h_i0
   psi_a_i <- -1

6) Point estimate + IF:
   theta_k <- -mean_i(psi_b_ik)/mean_i(psi_a_i)
   IF_ik <- -(psi_b_ik + psi_a_i*theta_k)/mean_i(psi_a_i)

7) Inference:
   se_k <- sqrt( var_i(IF_ik) / n )
   CI_abs_k <- theta_k ± z_{1-alpha/2} * se_k
   p_k from normal test, significance via Bonferroni alpha/(K-1)

8) Relative effect:
   psi_mu_i <- g_hat_i0 + u_i0*h_i0
   mu_c <- mean_i(psi_mu_i)
   tau_rel_k <- 100 * theta_k / mu_c
   delta-method se_rel_k and CI_rel_k

Output:
  MultiCausalEstimate with vectors over contrasts (0 vs k), k=1..K-1
```
# References

## Foundations: propensity score + unconfoundedness

* **Rosenbaum & Rubin (1983)** — introduces the propensity score as a balancing score under ignorability (your whole “(D \perp (Y(d)) \mid X)” setup). ([OUP Academic][1])
* **Hahn (1998)** — semiparametric efficiency / influence-function view of ATE estimators using propensity scores and outcome regression (conceptual basis for IF-based SEs). ([JSTOR][2])

## Doubly-robust / AIPW (your score `psi_b`)

* **Robins, Rotnitzky & Zhao (1994)** — classic IPW estimating equations paper; widely cited as the origin of AIPW-style augmentation logic. ([Taylor & Francis Online][3])
* **Bang & Robins (2005)** — formal “doubly robust” theory: consistency if either outcome model or propensity model is correct (your estimator’s key robustness property). ([PubMed][4])
* **Glynn & Quinn (2010)** — practitioner-friendly AIPW exposition (good to cite in docs as an accessible reference). ([UC Berkeley Law][5])

## Multi-valued / multi-treatment propensity score & effects (your (K\ge2) one-hot setting)

* **Imbens (2000)** — extends propensity score ideas to **multi-valued treatments** / dose-response; canonical reference for generalized propensity score logic. ([JSTOR][6])
* **Lechner (1999/IZA DP 91)** — identification and estimation under CIA with **multiple mutually exclusive treatments** (balancing scores beyond binary). ([Econstor][7])
* **Lopez & Gutman (2017, Stat Sci)** — review + methods for **categorical multiple treatments** (matching/weighting/regression variants; good for positioning your approach). ([arXiv][8])

## Cross-fitting + Neyman-orthogonal / DoubleML framing (your “DoubleML-style cross-fitting” claim)

* **Chernozhukov et al. (2018, Econometrics Journal)** — the modern reference for **Neyman-orthogonal scores + cross-fitting** delivering (\sqrt{n}) inference with ML nuisances. ([OUP Academic][9])
* **Chernozhukov et al. (2016, arXiv:1608.00060)** — earlier/longer technical version focused on treatment/causal parameters (often cited in implementations). ([arXiv][10])
* (Optional efficiency detail) **Hirano, Imbens & Ridder (2003)** — shows efficiency gains/conditions when using an **estimated** propensity score; useful background for the weighting component. ([Wiley Online Library][11])

## Sensitivity analysis (your `cf_y`, `r2_d`, `rho` style)

* **Cinelli & Hazlett (2020, JRSSB)** — modern sensitivity analysis framed via omitted-variable bias with **partial (R^2)** style parameters (matches your `r2_d` / confounding-strength parameterization). ([carloscinelli.com][12])
* **Cinelli, Ferwerda & Hazlett (sensemakr paper)** — practical companion describing the implemented sensitivity summaries/statistics. ([carloscinelli.com][13])
* (Alternative tradition) **Oster (2019, J Business & Econ Stats)** — coefficient-stability approach (different parameterization, but often cited alongside Cinelli–Hazlett in “how sensitive is this?” docs). ([IDEAS/RePEc][14])

[1]: https://academic.oup.com/biomet/article-abstract/70/1/41/240879?utm_source=chatgpt.com "central role of the propensity score in observational studies for ..."
[2]: https://www.jstor.org/stable/2998560?utm_source=chatgpt.com "On the Role of the Propensity Score in Efficient ..."
[3]: https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476818?utm_source=chatgpt.com "Estimation of Regression Coefficients When Some ..."
[4]: https://pubmed.ncbi.nlm.nih.gov/16401269/?utm_source=chatgpt.com "Doubly robust estimation in missing data and causal inference ..."
[5]: https://www.law.berkeley.edu/files/AIPW%281%29.pdf?utm_source=chatgpt.com "An Introduction to the Augmented Inverse Propensity ..."
[6]: https://www.jstor.org/stable/2673642?utm_source=chatgpt.com "Propensity Score - Estimating Dose-Response Functions"
[7]: https://www.econstor.eu/bitstream/10419/20925/1/dp91.pdf?utm_source=chatgpt.com "Identification and Estimation of Causal Effects of Multiple ..."
[8]: https://arxiv.org/abs/1701.05132?utm_source=chatgpt.com "Estimation of causal effects with multiple treatments: a review and new ideas"
[9]: https://academic.oup.com/ectj/article/21/1/C1/5056401?utm_source=chatgpt.com "Double/debiased machine learning for treatment and ..."
[10]: https://arxiv.org/abs/1608.00060?utm_source=chatgpt.com "Double/Debiased Machine Learning for Treatment and Causal Parameters"
[11]: https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00442?utm_source=chatgpt.com "Efficient Estimation of Average Treatment Effects"
[12]: https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20%282020%29%20-%20Making%20Sense%20of%20Sensitivity.pdf?utm_source=chatgpt.com "Making Sense of Sensitivity: Extending Omitted Variable Bias"
[13]: https://carloscinelli.com/files/Cinelli%20et%20al%20%282020%29%20-%20sensemakr.pdf?utm_source=chatgpt.com "sensemakr: Sensitivity Analysis Tools for OLS in R and Stata"
[14]: https://ideas.repec.org/a/taf/jnlbes/v37y2019i2p187-204.html?utm_source=chatgpt.com "Unobservable Selection and Coefficient Stability: Theory and"
