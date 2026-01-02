from __future__ import annotations
import numpy as np
import pandas as pd
import uuid
from typing import Dict, Optional, Union, List, Tuple, Callable, Any

from .base import CausalDatasetGenerator, _sigmoid, _logit

def _deterministic_ids(rng: np.random.Generator, n: int) -> List[str]:
    """Return deterministic uuid-like hex strings using the provided RNG."""
    return [rng.bytes(16).hex() for _ in range(n)]

def _add_ancillary_info(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    deterministic_ids: bool = False
) -> pd.DataFrame:
    """Helper to add standard ancillary columns (age, platform, etc.) to a synthetic DFG."""
    # Build a stable feature matrix from pre-outcome columns only.
    # (This wrapper adds ancillary after gen.generate, so we must be careful to
    # not use y or other post-outcome quantities to avoid outcome leakage.)
    exclude = {"y", "d", "m", "m_obs", "tau_link", "g0", "g1", "cate", "propensity", "propensity_obs", "mu0", "mu1"}
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

def generate_rct(
    n: int = 20_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_type: str = "binary",              # {"binary","normal","poisson"}; "nonnormal" -> "poisson"
    outcome_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
) -> pd.DataFrame:
    """
    Generate an RCT dataset via CausalDatasetGenerator (thin wrapper), ensuring
    randomized treatment independent of X. Keeps y and d as float for ML compatibility.

    Notes on effect scale (how `outcome_params` maps into the structural effect):
      - outcome_type="normal": treatment shifts the mean by (mean["B"] - mean["A"]) on the outcome scale.
      - outcome_type="binary": treatment shifts the log-odds by (logit(p_B) - logit(p_A)).
        Reported g0/g1 are on the probability scale.
      - outcome_type="poisson": treatment shifts the log-mean by log(lam_B / lam_A) where lam = shape * scale.
        Reported g0/g1 are on the mean (rate) scale.

    Ancillary columns (if add_ancillary=True) are generated from baseline confounders X only
    (plus RNG), and do not depend on the realized outcome y nor on treatment d. This avoids
    both outcome leakage and post-treatment adjustment/collider issues in typical ML pipelines.
    """
    # RNG for ancillary generation
    rng = np.random.default_rng(random_state)

    # Validate split
    split_f = float(split)
    if not (0.0 < split_f < 1.0):
        raise ValueError("split must be in (0,1).")

    # Normalize outcome_type
    ttype = outcome_type.lower()
    if ttype == "nonnormal":
        ttype = "poisson"
    if ttype not in {"binary", "normal", "poisson"}:
        raise ValueError("outcome_type must be 'binary', 'normal', or 'poisson' (or alias 'nonnormal').")

    # Default outcome_params
    if outcome_params is None:
        if ttype == "binary":
            outcome_params = {"p": {"A": 0.10, "B": 0.12}}
        elif ttype == "normal":
            outcome_params = {"mean": {"A": 0.00, "B": 0.20}, "std": 1.0}
        else:
            outcome_params = {"shape": 2.0, "scale": {"A": 1.0, "B": 1.1}}

    # Map to natural-scale means
    if ttype == "binary":
        pA = float(outcome_params["p"]["A"]); pB = float(outcome_params["p"]["B"])
        if not (0.0 < pA < 1.0 and 0.0 < pB < 1.0):
            raise ValueError("For binary outcomes, probabilities must be in (0,1).")
        mu0_nat, mu1_nat = pA, pB
    elif ttype == "normal":
        muA = float(outcome_params["mean"]["A"]); muB = float(outcome_params["mean"]["B"])
        sd  = float(outcome_params.get("std", 1.0))
        if not (sd > 0):
            raise ValueError("For normal outcomes, std must be > 0.")
        mu0_nat, mu1_nat = muA, muB
    else:  # poisson
        shape = float(outcome_params.get("shape", 2.0))
        scaleA = float(outcome_params["scale"]["A"])
        scaleB = float(outcome_params["scale"]["B"])
        lamA = shape * scaleA; lamB = shape * scaleB
        if not (lamA > 0 and lamB > 0):
            raise ValueError("For Poisson outcomes, implied rates must be > 0.")
        mu0_nat, mu1_nat = lamA, lamB

    # Convert to class parameters
    if ttype == "binary":
        alpha_y = _logit(mu0_nat)
        theta = _logit(mu1_nat) - _logit(mu0_nat)
        outcome_type_cls = "binary"
        sigma_y = 1.0
    elif ttype == "normal":
        alpha_y = mu0_nat
        theta = mu1_nat - mu0_nat
        outcome_type_cls = "continuous"
        sigma_y = sd
    else:  # poisson
        alpha_y = float(np.log(mu0_nat))
        theta = float(np.log(mu1_nat / mu0_nat))
        outcome_type_cls = "poisson"
        sigma_y = 1.0

    # Instantiate the unified generator with randomized treatment
    gen = CausalDatasetGenerator(
        theta=theta,
        tau=None,
        beta_y=None,
        beta_d=None,
        g_y=None,
        g_d=None,
        alpha_y=alpha_y,
        alpha_d=_logit(split_f),
        sigma_y=sigma_y,
        outcome_type=outcome_type_cls,
        confounder_specs=confounder_specs,
        k=int(k),
        x_sampler=x_sampler,
        use_copula=False,
        target_d_rate=None,
        u_strength_d=0.0,
        u_strength_y=0.0,
        seed=random_state,
    )

    df = gen.generate(n)

    # Ancillary columns (optional)
    if add_ancillary:
        df = _add_ancillary_info(df, n, rng, deterministic_ids)

    return df

def generate_rct_data(
    n: int = 20_000,
    split: float = 0.5,
    random_state: Optional[int] = 42,
    outcome_type: str = "binary",
    outcome_params: Optional[Dict] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = True,
    deterministic_ids: bool = False,
) -> pd.DataFrame:
    """
    Backward-compatible alias for generate_rct used in documentation.
    Parameters mirror generate_rct and are forwarded directly.
    """
    return generate_rct(
        n=n,
        split=split,
        random_state=random_state,
        outcome_type=outcome_type,
        outcome_params=outcome_params,
        confounder_specs=confounder_specs,
        k=k,
        x_sampler=x_sampler,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
    )

def obs_linear_effect(
    n: int = 10_000,
    theta: float = 1.0,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    target_d_rate: Optional[float] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
) -> pd.DataFrame:
    """
    Generate an observational dataset with linear effects of confounders and a constant treatment effect.

    Parameters
    ----------
    n : int, default=10_000
        Number of samples to generate.
    theta : float, default=1.0
        Constant treatment effect.
    outcome_type : {"continuous", "binary", "poisson"}, default="continuous"
        Family of the outcome distribution.
    sigma_y : float, default=1.0
        Noise level for continuous outcomes.
    target_d_rate : float, optional
        Target treatment prevalence (propensity mean).
    confounder_specs : list of dict, optional
        Schema for confounder distributions.
    beta_y : array-like, optional
        Linear coefficients for confounders in the outcome model.
    beta_d : array-like, optional
        Linear coefficients for confounders in the treatment model.
    random_state : int, optional
        Random seed for reproducibility.
    k : int, default=0
        Number of confounders if specs not provided.
    x_sampler : callable, optional
        Custom sampler for confounders.
    add_ancillary : bool, default=False
        If True, adds standard ancillary columns (age, platform, etc.).
    deterministic_ids : bool, default=False
        If True, generates deterministic user IDs.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset.
    """
    gen = CausalDatasetGenerator(
        theta=theta,
        outcome_type=outcome_type,
        sigma_y=sigma_y,
        target_d_rate=target_d_rate,
        seed=random_state,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        k=int(k),
        x_sampler=x_sampler
    )
    df = gen.generate(n)

    if add_ancillary:
        rng = np.random.default_rng(random_state)
        df = _add_ancillary_info(df, n, rng, deterministic_ids)

    return df

def obs_linear_effect_data(
    n: int = 10_000,
    theta: float = 1.0,
    outcome_type: str = "continuous",
    sigma_y: float = 1.0,
    target_d_rate: Optional[float] = None,
    confounder_specs: Optional[List[Dict[str, Any]]] = None,
    beta_y: Optional[np.ndarray] = None,
    beta_d: Optional[np.ndarray] = None,
    random_state: Optional[int] = 42,
    k: int = 0,
    x_sampler: Optional[Callable[[int, int, int], np.ndarray]] = None,
    add_ancillary: bool = False,
    deterministic_ids: bool = False,
) -> pd.DataFrame:
    """
    Alias for obs_linear_effect for consistency with generate_rct_data.
    Parameters mirror obs_linear_effect and are forwarded directly.
    """
    return obs_linear_effect(
        n=n,
        theta=theta,
        outcome_type=outcome_type,
        sigma_y=sigma_y,
        target_d_rate=target_d_rate,
        confounder_specs=confounder_specs,
        beta_y=beta_y,
        beta_d=beta_d,
        random_state=random_state,
        k=k,
        x_sampler=x_sampler,
        add_ancillary=add_ancillary,
        deterministic_ids=deterministic_ids,
    )
