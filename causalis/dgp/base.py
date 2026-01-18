from __future__ import annotations
import numpy as np
import pandas as pd
import uuid
from typing import Dict, Optional, List, Tuple, Any, Callable, Union
from scipy.special import erf, erfinv
from scipy.stats import norm, gamma as st_gamma, beta as st_beta, poisson as st_poisson, nbinom as st_nbinom, lognorm as st_lognorm

def _deterministic_ids(rng: np.random.Generator, n: int) -> List[str]:
    """
    Return deterministic uuid-like hex strings using the provided RNG.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random number generator to use.
    n : int
        Number of IDs to generate.

    Returns
    -------
    list of str
        A list of hex strings.
    """
    return [rng.bytes(16).hex() for _ in range(n)]


def _add_ancillary_info(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    deterministic_ids: bool = False
) -> pd.DataFrame:
    """
    Helper to add standard ancillary columns (age, platform, etc.) to a synthetic DFG.

    The ancillary columns are derived from baseline confounders X only (plus RNG),
    and do not depend on the realized outcome y nor on treatment d. This avoids
    both outcome leakage and post-treatment adjustment/collider issues.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing generated causal data_contracts.
    n : int
        Number of samples in the DataFrame.
    rng : numpy.random.Generator
        The random number generator to use.
    deterministic_ids : bool, default=False
        Whether to generate deterministic hex IDs instead of random UUIDs.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with added ancillary columns: 'user_id', 'age', 'cnt_trans',
        'platform_Android', 'platform_iOS', 'invited_friend'.
    """
    # Build a stable feature matrix from pre-outcome columns only.
    # (This wrapper adds ancillary after gen.generate, so we must be careful to
    # not use y or other post-outcome quantities to avoid outcome leakage.)
    exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate"}
    x_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[x_cols].to_numpy(dtype=float) if x_cols else np.zeros((n, 0), dtype=float)

    # Defensive: ensure X is finite before forming linear combinations.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    kx = int(X.shape[1])
    scale = float(np.sqrt(kx)) if kx > 0 else 1.0

    # Latent scores (bounded) derived from X to induce realistic correlations
    # without leaking y.
    def score() -> np.ndarray:
        if kx == 0:
            return np.zeros(n, dtype=float)
        w = rng.normal(size=kx)
        z = (X @ w) / scale
        z = np.clip(z, -20.0, 20.0)
        return np.tanh(z)

    s_age = score()
    s_tx = score()
    s_pl = score()
    s_inv = score()

    # Age ~ N(35 + 6*s_age, 8), int clipped to [18,90]
    age_mu = 35.0 + 6.0 * s_age
    age = rng.normal(age_mu, 8.0, n).round().clip(18, 90).astype(int)

    # cnt_trans ~ Poisson(exp(a + b*s_tx))
    log_lam_tx = np.clip(0.5 + 0.6 * s_tx, -5.0, 3.0)
    lam_tx = np.exp(log_lam_tx)
    cnt_trans = rng.poisson(lam_tx, n).astype(int)

    # Platform: P(Android) = sigmoid(-0.4 + 1.0*s_pl)
    p_android = _sigmoid(-0.4 + 1.0 * s_pl)
    platform_android = rng.binomial(1, np.clip(p_android, 0.0, 1.0), n).astype(int)
    platform_ios = (1 - platform_android).astype(int)

    # Invited friend: Bernoulli(sigmoid(-2.7 + 1.2*s_inv))
    p_inv = _sigmoid(-2.7 + 1.2 * s_inv)
    invited_friend = rng.binomial(1, np.clip(p_inv, 0.0, 1.0), n).astype(int)
    # User IDs
    if deterministic_ids:
        user_ids = _deterministic_ids(rng, n)
    else:
        user_ids = [str(uuid.uuid4()) for _ in range(n)]
    df.insert(0, "user_id", user_ids)
    df["age"] = age
    df["cnt_trans"] = cnt_trans
    df["platform_Android"] = platform_android
    df["platform_iOS"] = platform_ios
    df["invited_friend"] = invited_friend

    return df


def _sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Numerically stable sigmoid function: 1 / (1 + exp(-z)).

    Handles large positive/negative z without overflow warnings.
    Ensures outputs lie strictly within (0,1) and are strictly monotone in z
    even under floating-point saturation by using nextafter for exact 0/1 cases.

    Parameters
    ----------
    z : float or numpy.ndarray
        Input value or array.

    Returns
    -------
    float or numpy.ndarray
        Sigmoid of z.
    """
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)

    pos_mask = z_arr >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))
    ez = np.exp(z_arr[neg_mask])
    out[neg_mask] = ez / (1.0 + ez)

    if out.ndim == 0:
        val = float(out)
        if not (0.0 < val < 1.0):
            return float(np.nextafter(0.0, 1.0)) if val <= 0.0 else float(np.nextafter(1.0, 0.0))
        return val

    # Order-preserving nudge off exact 0/1 on a flattened view, then reshape back
    flat = out.ravel()
    zflat = z_arr.ravel()

    zero_idx = np.flatnonzero(flat <= 0.0)
    if zero_idx.size:
        order = np.argsort(zflat[zero_idx])  # ascending z
        val = 0.0
        for j in order:
            val = np.nextafter(val, 1.0)
            flat[zero_idx[j]] = val

    one_idx = np.flatnonzero(flat >= 1.0)
    if one_idx.size:
        order = np.argsort(zflat[one_idx])  # ascending z
        k = one_idx.size
        val = 1.0
        steps = []
        for _ in range(k):
            val = np.nextafter(val, 0.0)
            steps.append(val)  # descending sequence < 1
        for rank, j in enumerate(order):
            flat[one_idx[j]] = steps[k - rank - 1]

    return flat.reshape(out.shape)


def _logit(p: float) -> float:
    """
    Clips probability then returns log‑odds (logit).

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        Log-odds of p.
    """
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1 - p)))


def _nearest_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Find the nearest positive semi-definite matrix to a given symmetric matrix
    via eigenvalue clipping.
    """
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    w = np.clip(w, eps, None)
    Cpsd = (V * w) @ V.T
    d = np.sqrt(np.diag(Cpsd))
    # Normalize back to correlation matrix (ones on diagonal)
    return Cpsd / np.outer(d, d)


def estimate_gaussian_copula_corr(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Estimate a Gaussian copula correlation matrix from observational data_contracts.
    Uses rank -> normal scores -> Pearson correlation approach.
    """
    U = df[cols].rank(pct=True).to_numpy()
    U = np.clip(U, 1e-6, 1 - 1e-6)
    # Z = norm.ppf(U)
    Z = np.sqrt(2.0) * erfinv(2.0 * U - 1.0)
    C = np.corrcoef(Z, rowvar=False)
    return _nearest_psd(C)


def _gaussian_copula(
    rng: np.random.Generator,
    n: int,
    specs: List[Dict[str, Any]],
    corr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate mixed-type confounders X using a Gaussian copula.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random number generator to use.
    n : int
        Number of samples to generate.
    specs : list of dict
        Schema for confounder distributions. See `CausalDatasetGenerator` for details.
    corr : numpy.ndarray, optional
        Correlation matrix for the latent Gaussian variables. If None, uses identity.

    Returns
    -------
    X : numpy.ndarray
        Generated confounder matrix.
    names : list of str
        Names of the generated columns.
    """
    d = len(specs)
    if d == 0:
        return np.empty((n, 0)), []

    if corr is None:
        corr = np.eye(d)
    corr = np.asarray(corr, dtype=float)

    # Cholesky (assume PSD; if estimated, pass through nearest_psd)
    L = np.linalg.cholesky(corr + 1e-10 * np.eye(d))
    Z = rng.normal(size=(n, d)) @ L.T
    U = norm.cdf(Z)  # (0,1)

    cols = []
    names = []

    for j, spec in enumerate(specs):
        name = spec.get("name") or f"x{j+1}"
        dist = str(spec.get("dist", "normal")).lower()

        if dist == "categorical":
            cats = list(spec.get("categories", [0, 1, 2]))
            probs = np.asarray(spec.get("probs", None), dtype=float) if spec.get("probs", None) is not None else None
            if probs is None:
                probs = np.ones(len(cats)) / len(cats)
            probs = probs / probs.sum()
            # Use U[:,j] as a categorical draw by CDF thresholds
            thr = np.cumsum(probs)
            draw = np.searchsorted(thr, U[:, j], side="right")
            # Guard boundary: if U[:,j]==1.0
            draw[draw == len(cats)] = len(cats) - 1
            cat_vals = np.asarray(cats)[draw]
            # one-hot encode (except first level)
            rest = cats[1:]
            if len(rest) == 0:
                cols.append(np.zeros(n, dtype=float))
                names.append(f"{name}__onlylevel")
            else:
                for c in rest:
                    cols.append((cat_vals == c).astype(float))
                    names.append(f"{name}_{c}")
            continue

        u = U[:, j]
        if dist == "normal":
            mu = float(spec.get("mu", 0.0)); sd = float(spec.get("sd", 1.0))
            col = mu + sd * norm.ppf(u)
        elif dist == "uniform":
            a = float(spec.get("a", 0.0)); b = float(spec.get("b", 1.0))
            col = a + (b - a) * u
        elif dist == "bernoulli":
            p = float(spec.get("p", 0.5))
            col = (u < p).astype(float)
        elif dist == "lognormal":
            mu = float(spec.get("mu", 0.0))
            sigma = float(spec.get("sigma", 1.0))
            # scipy lognorm: s=sigma, scale=exp(mu)
            col = st_lognorm(s=sigma, scale=np.exp(mu)).ppf(u)
        elif dist == "gamma":
            shape = float(spec.get("shape", 2.0))
            scale = spec.get("scale", None)
            if scale is None:
                mean = float(spec.get("mean", 1.0))
                scale = mean / shape
            col = st_gamma(a=shape, scale=float(scale)).ppf(u)
        elif dist == "beta":
            a = spec.get("a", None); b = spec.get("b", None)
            if a is None or b is None:
                mean = float(spec.get("mean", 0.5))
                kappa = float(spec.get("kappa", 10.0))
                a = mean * kappa
                b = (1.0 - mean) * kappa
            col = st_beta(a=float(a), b=float(b)).ppf(u)
        elif dist == "poisson":
            lam = float(spec.get("lam", 1.0))
            col = st_poisson(mu=lam).ppf(u).astype(float)
        elif dist == "negbin":
            mu = float(spec.get("mu", 5.0))
            alpha = float(spec.get("alpha", 0.5))  # Var = mu + alpha*mu^2
            r = 1.0 / max(alpha, 1e-12)
            p = r / (r + mu)
            col = st_nbinom(n=r, p=p).ppf(u).astype(float)
        else:
            raise ValueError(f"Unsupported dist in copula: {dist}")

        # Apply clipping if specified in spec
        if "clip_min" in spec or "clip_max" in spec:
            cmin = float(spec.get("clip_min", -np.inf))
            cmax = float(spec.get("clip_max", np.inf))
            col = np.clip(col, cmin, cmax)

        cols.append(col.astype(float))
        names.append(name)

    X = np.column_stack(cols) if cols else np.empty((n, 0))
    return X, names
