from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Callable, Any

from causalis.dgp.base import _sigmoid, _gaussian_copula
from causalis.data_contracts.multicausaldata import MultiCausalData


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array")
    shift = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - shift)
    denom = exp_scores.sum(axis=1, keepdims=True)
    return exp_scores / np.clip(denom, 1e-12, np.inf)


_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KWARGS)
class MultiCausalDatasetGenerator:
    """
    Generate synthetic causal datasets with multi-class (one-hot) treatments.

    Treatment assignment is modeled via a multinomial logistic (softmax) model:
        P(D=k | X, U) = softmax_k(alpha_d[k] + f_k(X) + u_strength_d[k] * U)

    Outcome depends on confounders and the assigned treatment class:
        outcome_type = "continuous":
            Y = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X)) + eps
        outcome_type = "binary":
            logit P(Y=1|X,D,U) = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))
        outcome_type = "poisson":
            log E[Y|X,D,U] = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))
        outcome_type = "gamma":
            log E[Y|X,D,U] = alpha_y + f_y(X) + u_strength_y * U + sum_k D_k * (theta_k + tau_k(X))

    Parameters
    ----------
    n_treatments : int, default=3
        Number of treatment classes (including control). Column 0 is treated as control.
        Generated treatment columns are a full one-hot encoding that sums to 1.
    d_names : list of str, optional
        Names of treatment columns. If None, uses ["d_0", "d_1", ...].
    theta : float or array-like, optional
        Constant treatment effects on the link scale for each class.
        If scalar, applied to all non-control classes (control effect = 0).
        If length K-1, prepends 0 for control. If length K, uses as provided.
    tau : callable or list of callables, optional
        Heterogeneous effects for each class. If callable, applied to non-control classes.
        Effects are additive with theta on the link scale:
        tau_link_k(X) = theta_k + tau_k(X).
    beta_y : array-like, optional
        Linear coefficients for baseline outcome f_y(X).
    g_y : callable, optional
        Nonlinear baseline outcome function g_y(X).
    alpha_y : float, default=0.0
        Outcome intercept on link scale.
    sigma_y : float, default=1.0
        Std dev for continuous outcomes.
    outcome_type : {"continuous", "binary", "poisson", "gamma"}, default="continuous"
        Outcome family.
    gamma_shape : float, default=2.0
        Shape parameter for gamma outcomes.
    u_strength_y : float, default=0.0
        Strength of unobserved confounder in outcome.
    confounder_specs : list of dict, optional
        Schema for generating confounders (same format as CausalDatasetGenerator).
    k : int, default=5
        Number of confounders if confounder_specs is None.
    x_sampler : callable, optional
        Custom sampler (n, k, seed) -> X ndarray.
    use_copula : bool, default=False
        If True and confounder_specs provided, use Gaussian copula for X.
    copula_corr : array-like, optional
        Correlation matrix for copula.
    beta_d : array-like or list, optional
        Linear coefficients for treatment assignment. If array of shape (k,),
        applies to all non-control classes. If shape (K,k), uses per class.
    g_d : callable or list of callables, optional
        Nonlinear treatment score per class. If callable, applies to non-control classes.
    alpha_d : float or array-like, optional
        Intercepts for treatment scores. If scalar, applies to non-control classes.
    u_strength_d : float or array-like, default=0.0
        Unobserved confounder strength in treatment assignment.
        If scalar, interpreted as [0, c, c, ...] so latent U perturbs non-control
        classes relative to control (and does not cancel in softmax).
    propensity_sharpness : float, default=1.0
        Scales treatment scores to adjust overlap.
    target_d_rate : array-like, optional
        Target marginal class probabilities (length K). Calibrates alpha_d
        using iterative scaling (approximate when u_strength_d != 0).
    include_oracle : bool, default=True
        Whether to include oracle columns for propensities and potential outcomes.
    seed : int, optional
        Random seed.
    """
    n_treatments: int = 3
    d_names: Optional[List[str]] = None

    theta: Optional[Union[float, List[float], np.ndarray]] = 1.0
    tau: Optional[Union[Callable[[np.ndarray], np.ndarray], List[Optional[Callable[[np.ndarray], np.ndarray]]]]] = None

    beta_y: Optional[np.ndarray] = None
    g_y: Optional[Callable[[np.ndarray], np.ndarray]] = None
    alpha_y: float = 0.0
    sigma_y: float = 1.0
    outcome_type: str = "continuous"
    gamma_shape: float = 2.0
    u_strength_y: float = 0.0

    confounder_specs: Optional[List[Dict[str, Any]]] = None
    k: int = 5
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None
    use_copula: bool = False
    copula_corr: Optional[np.ndarray] = None

    beta_d: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None
    g_d: Optional[Union[Callable[[np.ndarray], np.ndarray], List[Optional[Callable[[np.ndarray], np.ndarray]]]]] = None
    alpha_d: Optional[Union[float, List[float], np.ndarray]] = None
    u_strength_d: Union[float, List[float], np.ndarray] = 0.0
    propensity_sharpness: float = 1.0
    target_d_rate: Optional[Union[List[float], np.ndarray]] = None

    include_oracle: bool = True
    seed: Optional[int] = None

    rng: np.random.Generator = field(init=False, repr=False)
    confounder_names_: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        if self.n_treatments < 2:
            raise ValueError("n_treatments must be at least 2")
        if self.confounder_specs is not None:
            self.k = len(self.confounder_specs)
        if self.d_names is None:
            self.d_names = [f"d_{i}" for i in range(self.n_treatments)]
        if len(self.d_names) != self.n_treatments:
            raise ValueError("d_names length must match n_treatments")

    # ---------- confounder sampling ----------

    def _sample_X(self, n: int) -> Tuple[np.ndarray, List[str]]:
        if self.x_sampler is not None:
            X = self.x_sampler(n, self.k, self.seed)
            if self.confounder_specs is not None:
                names = [spec.get("name", f"x{i+1}") for i, spec in enumerate(self.confounder_specs)]
            else:
                names = [f"x{i+1}" for i in range(self.k)]
            return X, names

        if self.confounder_specs is None:
            X = self.rng.normal(size=(n, self.k))
            names = [f"x{i+1}" for i in range(self.k)]
            return X, names

        if self.use_copula:
            X, names = _gaussian_copula(self.rng, n, self.confounder_specs, self.copula_corr)
            self.k = X.shape[1]
            return X, names

        cols = []
        names = []
        for spec in self.confounder_specs:
            name = spec.get("name") or f"x{len(names)+1}"
            dist = spec.get("dist", "normal").lower()
            if dist == "normal":
                mu = spec.get("mu", 0.0)
                sd = spec.get("sd", 1.0)
                col = self.rng.normal(mu, sd, size=n)
            elif dist == "uniform":
                a = spec.get("a", 0.0)
                b = spec.get("b", 1.0)
                col = self.rng.uniform(a, b, size=n)
            elif dist == "bernoulli":
                p = spec.get("p", 0.5)
                col = self.rng.binomial(1, p, size=n).astype(float)
            elif dist == "lognormal":
                mu = spec.get("mu", 0.0)
                sigma = spec.get("sigma", 1.0)
                col = self.rng.lognormal(mean=mu, sigma=sigma, size=n)
            elif dist == "gamma":
                shape = spec.get("shape", 2.0)
                scale = spec.get("scale", None)
                if scale is None:
                    mean = spec.get("mean", 1.0)
                    scale = mean / shape
                col = self.rng.gamma(shape=shape, scale=scale, size=n)
            elif dist == "beta":
                a = spec.get("a", None)
                b = spec.get("b", None)
                if a is None or b is None:
                    mean = spec.get("mean", 0.5)
                    kappa = spec.get("kappa", 10.0)
                    a = mean * kappa
                    b = (1.0 - mean) * kappa
                col = self.rng.beta(a, b, size=n)
            elif dist == "poisson":
                lam = spec.get("lam", 1.0)
                col = self.rng.poisson(lam=lam, size=n).astype(float)
            elif dist == "negbin":
                mu = spec.get("mu", 5.0)
                alpha = spec.get("alpha", 0.5)
                r = 1.0 / max(alpha, 1e-12)
                p = r / (r + mu)
                col = self.rng.negative_binomial(r, p, size=n).astype(float)
            elif dist == "categorical":
                categories = list(spec.get("categories", [0, 1, 2]))
                probs = spec.get("probs", None)
                if probs is not None:
                    p = np.asarray(probs, dtype=float)
                    ps = p / p.sum()
                else:
                    ps = None
                col = self.rng.choice(categories, p=ps, size=n)
                rest = categories[1:]
                if len(rest) == 0:
                    cols.append(np.zeros(n, dtype=float))
                    names.append(f"{name}__onlylevel")
                    continue
                for c in rest:
                    cols.append((col == c).astype(float))
                    names.append(f"{name}_{c}")
                continue
            else:
                raise ValueError(f"Unknown dist: {dist}")

            if "clip_min" in spec or "clip_max" in spec:
                cmin = spec.get("clip_min", -np.inf)
                cmax = spec.get("clip_max", np.inf)
                col = np.clip(col, cmin, cmax)

            cols.append(col.astype(float))
            names.append(name)

        X = np.column_stack(cols) if cols else np.empty((n, 0))
        self.k = X.shape[1]
        return X, names

    # ---------- normalization helpers ----------

    def _normalize_alpha_d(self, K: int) -> np.ndarray:
        if self.alpha_d is None:
            return np.zeros(K, dtype=float)
        if np.isscalar(self.alpha_d):
            val = float(self.alpha_d)
            return np.array([0.0] + [val] * (K - 1), dtype=float)
        arr = np.asarray(self.alpha_d, dtype=float).reshape(-1)
        if arr.size == K - 1:
            arr = np.concatenate([[0.0], arr])
        if arr.size != K:
            raise ValueError("alpha_d must be scalar, length K, or length K-1")
        return arr.astype(float)

    def _normalize_u_strength_d(self, K: int) -> np.ndarray:
        if np.isscalar(self.u_strength_d):
            c = float(self.u_strength_d)
            # Scalar strength is applied to non-control classes only; applying the same
            # value to all classes would cancel out under softmax shift invariance.
            return np.array([0.0] + [c] * (K - 1), dtype=float)
        arr = np.asarray(self.u_strength_d, dtype=float).reshape(-1)
        if arr.size == K - 1:
            arr = np.concatenate([[0.0], arr])
        if arr.size != K:
            raise ValueError("u_strength_d must be scalar, length K, or length K-1")
        return arr.astype(float)

    def _normalize_beta_d(self, K: int, kx: int) -> List[Optional[np.ndarray]]:
        if self.beta_d is None:
            return [None] * K
        if isinstance(self.beta_d, list):
            vals = self.beta_d
            if len(vals) == K - 1:
                vals = [None] + vals
            if len(vals) != K:
                raise ValueError("beta_d list must have length K or K-1")
            out = []
            for v in vals:
                if v is None:
                    out.append(None)
                else:
                    arr = np.asarray(v, dtype=float).reshape(-1)
                    if arr.size != kx:
                        raise ValueError("beta_d element has incompatible size")
                    out.append(arr)
            return out
        arr = np.asarray(self.beta_d, dtype=float)
        if arr.ndim == 1:
            if arr.size != kx:
                raise ValueError("beta_d vector has incompatible size")
            return [np.zeros(kx, dtype=float)] + [arr] * (K - 1)
        if arr.ndim == 2:
            if arr.shape[1] != kx:
                raise ValueError("beta_d matrix has incompatible width")
            if arr.shape[0] == K - 1:
                arr = np.vstack([np.zeros((1, kx), dtype=float), arr])
            if arr.shape[0] != K:
                raise ValueError("beta_d matrix must have shape (K,k) or (K-1,k)")
            return [arr[i] for i in range(K)]
        raise ValueError("beta_d must be array-like or list")

    def _normalize_g_d(self, K: int) -> List[Optional[Callable[[np.ndarray], np.ndarray]]]:
        if self.g_d is None:
            return [None] * K
        if callable(self.g_d):
            return [None] + [self.g_d] * (K - 1)
        if isinstance(self.g_d, list):
            vals = self.g_d
            if len(vals) == K - 1:
                vals = [None] + vals
            if len(vals) != K:
                raise ValueError("g_d list must have length K or K-1")
            return vals
        raise ValueError("g_d must be a callable or list of callables")

    def _normalize_theta(self, K: int) -> np.ndarray:
        if self.theta is None:
            return np.zeros(K, dtype=float)
        if np.isscalar(self.theta):
            return np.array([0.0] + [float(self.theta)] * (K - 1), dtype=float)
        arr = np.asarray(self.theta, dtype=float).reshape(-1)
        if arr.size == K - 1:
            arr = np.concatenate([[0.0], arr])
        if arr.size != K:
            raise ValueError("theta must be scalar, length K, or length K-1")
        return arr.astype(float)

    def _normalize_tau(self, K: int) -> List[Optional[Callable[[np.ndarray], np.ndarray]]]:
        if self.tau is None:
            return [None] * K
        if callable(self.tau):
            return [None] + [self.tau] * (K - 1)
        if isinstance(self.tau, list):
            vals = self.tau
            if len(vals) == K - 1:
                vals = [None] + vals
            if len(vals) != K:
                raise ValueError("tau list must have length K or K-1")
            return vals
        raise ValueError("tau must be a callable or list of callables")

    @staticmethod
    def _normalize_outcome_type(outcome_type: str) -> str:
        # Support the common alias used in some older call sites.
        ttype = str(outcome_type).lower()
        if ttype == "normal":
            return "continuous"
        return ttype

    @staticmethod
    def _exp_link(link: np.ndarray) -> np.ndarray:
        # Clip link values to keep exp(.) numerically stable.
        return np.exp(np.clip(np.asarray(link, dtype=float), -20.0, 20.0))

    @staticmethod
    def _require_supported_outcome_type(ttype: str) -> None:
        if ttype not in {"continuous", "binary", "poisson", "gamma"}:
            raise ValueError("outcome_type must be 'continuous', 'binary', 'poisson', or 'gamma'")

    def _natural_scale_from_link(self, link: np.ndarray, ttype: str) -> np.ndarray:
        self._require_supported_outcome_type(ttype)
        link = np.asarray(link, dtype=float)
        # Oracle potential outcomes are stored on the natural outcome scale.
        if ttype == "continuous":
            return link
        if ttype == "binary":
            return _sigmoid(link)
        return self._exp_link(link)

    def _sample_outcome_from_link(self, link: np.ndarray, ttype: str) -> np.ndarray:
        self._require_supported_outcome_type(ttype)
        link = np.asarray(link, dtype=float).reshape(-1)
        n = link.shape[0]
        # Draw observed outcomes from the family implied by `outcome_type`.
        if ttype == "continuous":
            return link + self.rng.normal(0, float(self.sigma_y), size=n)
        if ttype == "binary":
            p = _sigmoid(link)
            return self.rng.binomial(1, p).astype(float)
        if ttype == "poisson":
            lam = self._exp_link(link)
            return self.rng.poisson(lam).astype(float)

        # Gamma with mean mu=exp(link) and variance mu^2 / shape.
        mu = self._exp_link(link)
        shape = float(self.gamma_shape)
        if shape <= 0:
            raise ValueError("gamma_shape must be > 0 for gamma outcomes")
        scale = mu / shape
        return self.rng.gamma(shape=shape, scale=scale, size=n).astype(float)

    def _calibrate_alpha_d(self, scores_base: np.ndarray, alpha_init: np.ndarray, target: np.ndarray) -> np.ndarray:
        alpha = alpha_init.astype(float).copy()
        target = np.asarray(target, dtype=float).reshape(-1)
        if target.size != scores_base.shape[1]:
            raise ValueError("target_d_rate must have length K")
        if np.any(target <= 0):
            raise ValueError("target_d_rate must be strictly positive")
        target = target / target.sum()

        for _ in range(50):
            probs = _softmax(scores_base + alpha)
            p_bar = probs.mean(axis=0)
            delta = np.log(np.clip(target, 1e-12, 1.0)) - np.log(np.clip(p_bar, 1e-12, 1.0))
            alpha += delta
            alpha -= alpha.mean()
            if np.max(np.abs(delta)) < 1e-6:
                break
        return alpha

    def _draw_multinomial(self, probs: np.ndarray) -> np.ndarray:
        n, K = probs.shape
        if n < K:
            raise ValueError("n must be >= n_treatments to ensure all classes appear")
        classes = None
        counts = None
        for _ in range(10):
            u = self.rng.random(n)
            cdf = np.cumsum(probs, axis=1)
            classes = (u[:, None] < cdf).argmax(axis=1)
            counts = np.bincount(classes, minlength=K)
            if np.all(counts > 0):
                return classes
        # Force at least one example per class
        missing = np.where(counts == 0)[0] if counts is not None else np.arange(K)
        idxs = self.rng.choice(n, size=len(missing), replace=False)
        for k, idx in zip(missing, idxs):
            classes[int(idx)] = int(k)
        return classes

    # ---------- public API ----------

    def generate(self, n: int, U: Optional[np.ndarray] = None) -> pd.DataFrame:
        X, names = self._sample_X(n)
        self.confounder_names_ = names
        if U is None:
            U = self.rng.normal(size=n)
        U = np.asarray(U, dtype=float)

        K = self.n_treatments
        alpha_d = self._normalize_alpha_d(K)
        beta_d_list = self._normalize_beta_d(K, X.shape[1])
        g_d_list = self._normalize_g_d(K)
        u_strength_d = self._normalize_u_strength_d(K)

        scores_base = np.zeros((n, K), dtype=float)
        for k in range(K):
            score_x = np.zeros(n, dtype=float)
            if beta_d_list[k] is not None:
                bt = np.asarray(beta_d_list[k], dtype=float).reshape(-1)
                if bt.size != X.shape[1]:
                    raise ValueError("beta_d incompatible with X")
                score_x += X @ bt
            if g_d_list[k] is not None:
                score_x += np.asarray(g_d_list[k](X), dtype=float)
            scores_base[:, k] = float(self.propensity_sharpness) * score_x

        if self.target_d_rate is not None:
            alpha_d = self._calibrate_alpha_d(scores_base, alpha_d, np.asarray(self.target_d_rate, dtype=float))

        scores_base = scores_base + alpha_d
        scores_obs = scores_base + u_strength_d * U.reshape(-1, 1)

        m_obs = _softmax(scores_obs)
        m = _softmax(scores_base)

        classes = self._draw_multinomial(m_obs)
        D = np.zeros((n, K), dtype=float)
        D[np.arange(n), classes] = 1.0

        theta_vec = self._normalize_theta(K)
        tau_list = self._normalize_tau(K)

        Xf = np.asarray(X, dtype=float)
        loc_base = np.full(n, float(self.alpha_y), dtype=float)
        if self.beta_y is not None:
            by = np.asarray(self.beta_y, dtype=float).reshape(-1)
            if by.size != Xf.shape[1]:
                raise ValueError("beta_y incompatible with X")
            loc_base += Xf @ by
        if self.g_y is not None:
            loc_base += np.asarray(self.g_y(Xf), dtype=float)

        tau_mat = np.tile(theta_vec.reshape(1, -1), (n, 1))
        for k in range(K):
            if tau_list[k] is not None:
                tau_val = np.asarray(tau_list[k](Xf), dtype=float).reshape(-1)
                if tau_val.size != n:
                    raise ValueError("tau function returned wrong shape")
                # Add heterogeneous residual on top of constant treatment effect.
                tau_mat[:, k] += tau_val

        loc = loc_base + (D * tau_mat).sum(axis=1)
        if self.u_strength_y != 0.0:
            loc = loc + float(self.u_strength_y) * U

        ttype = self._normalize_outcome_type(self.outcome_type)
        Y = self._sample_outcome_from_link(loc, ttype)

        df = pd.DataFrame({"y": Y})
        for k, name in enumerate(self.d_names):
            df[name] = D[:, k]
        for j, name in enumerate(names):
            df[name] = X[:, j]

        if self.include_oracle:
            for k, name in enumerate(self.d_names):
                df[f"m_{name}"] = m[:, k]
                df[f"m_obs_{name}"] = m_obs[:, k]
                df[f"tau_link_{name}"] = tau_mat[:, k]

            # Oracle potential outcomes on the natural scale (no U).
            g_vals = [self._natural_scale_from_link(loc_base + tau_mat[:, k], ttype) for k in range(K)]

            for k, name in enumerate(self.d_names):
                df[f"g_{name}"] = g_vals[k]

            g0 = g_vals[0]
            for k in range(1, K):
                df[f"cate_{self.d_names[k]}"] = g_vals[k] - g0

        return df

    def to_multicausal_data(
        self,
        n: int,
        confounders: Optional[Union[str, List[str]]] = None,
    ) -> MultiCausalData:
        df = self.generate(n)

        if confounders is None:
            confounder_cols = list(self.confounder_names_)
        elif isinstance(confounders, str):
            confounder_cols = [confounders]
        else:
            confounder_cols = [c for c in confounders if c in df.columns]

        return MultiCausalData(
            df=df,
            outcome="y",
            treatment_names=self.d_names,
            confounders=confounder_cols,
            control_treatment=self.d_names[0],
        )
