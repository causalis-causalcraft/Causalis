from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple, Callable, Any

from .base import CausalDatasetGenerator
from ..base import _sigmoid, _logit, _deterministic_ids, _add_ancillary_info

class _PrognosticScore:
    """
    Deterministic (given seed) prognostic signal s(X) used in outcome baseline.

    Works with any X dimension (incl. one-hot expansions) because coef is initialized on first call.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random number generator to use.
    scale : float, default=1.0
        Scale of the generated coefficients.
    """
    def __init__(self, rng: np.random.Generator, scale: float = 1.0):
        self.rng = rng
        self.scale = float(scale)
        self.coef_: Optional[np.ndarray] = None
        self.mean_: Optional[float] = None  # for centering in-sample

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the prognostic score s(X).

        Parameters
        ----------
        X : numpy.ndarray
            Confounder matrix.

        Returns
        -------
        numpy.ndarray
            Calculated prognostic score.
        """
        X = np.asarray(X, dtype=float)
        if X.shape[1] == 0:
            return np.zeros(X.shape[0], dtype=float)

        if self.coef_ is None:
            # Scale ~ 1/sqrt(k) to keep variance stable with dimension
            k = X.shape[1]
            self.coef_ = self.rng.normal(size=k) * (self.scale / np.sqrt(max(1, k)))

        s = X @ self.coef_
        # Center in-sample so alpha_y continues to mean something for continuous outcomes
        if self.mean_ is None:
            self.mean_ = float(s.mean())
        return s - self.mean_

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for __call__.

        Parameters
        ----------
        X : numpy.ndarray
            Confounder matrix.

        Returns
        -------
        numpy.ndarray
            Calculated prognostic score.
        """
        # Same as __call__ but safe to reuse after generator ran
        return self.__call__(X)


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
    # --- CUPED knobs ---
    add_pre: bool = True,
    pre_name: str = "y_pre",
    pre_corr: float = 0.7,           # target Corr(y_pre, y|D=0) for continuous-ish settings
    prognostic_scale: float = 1.0,   # strength of s(X) in outcome baseline
) -> pd.DataFrame:
    """
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

    Parameters
    ----------
    n : int, default=20_000
        Number of samples to generate.
    split : float, default=0.5
        Proportion of samples assigned to the treatment group.
    random_state : int, optional
        Random seed for reproducibility.
    outcome_type : {"binary", "normal", "poisson"}, default="binary"
        Distribution family of the outcome.
    outcome_params : dict, optional
        Parameters defining baseline rates/means and treatment effects.
        e.g., {"p": {"A": 0.1, "B": 0.12}} for binary.
    confounder_specs : list of dict, optional
        Schema for confounder distributions.
    k : int, default=0
        Number of confounders if specs not provided.
    x_sampler : callable, optional
        Custom sampler for confounders.
    add_ancillary : bool, default=True
        Whether to add descriptive columns like 'age', 'platform', etc.
    deterministic_ids : bool, default=False
        Whether to generate deterministic user IDs.
    add_pre : bool, default=True
        Whether to generate a pre-period covariate (`y_pre`).
    pre_name : str, default="y_pre"
        Name of the pre-period covariate column.
    pre_corr : float, default=0.7
        Target correlation between `y_pre` and the outcome Y in the control group.
    prognostic_scale : float, default=1.0
        Scale of the prognostic signal derived from confounders.

    Returns
    -------
    pandas.DataFrame
        Synthetic RCT dataset.
    """
    # RNG for ancillary generation
    rng = np.random.default_rng(random_state)

    # Validate split
    split_f = float(split)
    if not (0.0 < split_f < 1.0):
        raise ValueError("split must be in (0,1).")

    # If you want CUPED to work, you must have some baseline X to carry signal.
    if add_pre and (confounder_specs is None) and (x_sampler is None) and int(k) == 0:
        k = 5

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

    prognostic = _PrognosticScore(rng, scale=prognostic_scale) if add_pre else None

    # Instantiate the unified generator with randomized treatment
    gen = CausalDatasetGenerator(
        theta=theta,
        tau=None,
        beta_y=None,
        beta_d=None,
        g_y=(prognostic if add_pre else None),
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

    if add_pre:
        # Identify the baseline X columns generated by gen (before ancillary columns are added)
        exclude = {"y","d","m","m_obs","tau_link","g0","g1","cate","propensity","propensity_obs","mu0","mu1"}
        x_cols = [c for c in df.columns if c not in exclude]

        X = df[x_cols].to_numpy(dtype=float)
        s = prognostic.score(X)

        # Choose noise so Corr(y_pre, y|D=0) is about pre_corr in continuous-like regimes.
        # For your default "binary" outcome this is heuristic; for CUPED benchmarking prefer outcome_type="normal".
        pre_corr = float(np.clip(pre_corr, 1e-3, 0.999))
        V = float(np.var(s, ddof=0))
        if V <= 1e-12:
            sigma_pre = 1.0
        else:
            # For continuous: y ≈ alpha + s + eps_y  => corr = V / sqrt((V+sig_pre^2)(V+sigma_y^2))
            sigma_y2 = float(sigma_y)**2 if outcome_type_cls == "continuous" else 1.0
            sigma_pre2 = (V * V) / (pre_corr**2 * (V + sigma_y2)) - V
            sigma_pre = float(np.sqrt(max(sigma_pre2, 1e-12)))

        df[pre_name] = s + rng.normal(0.0, sigma_pre, size=n)

    if add_ancillary:
        df = _add_ancillary_info(df, n, rng, deterministic_ids)

    return df



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
    pandas.DataFrame
        Synthetic observational dataset.
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
    Backward-compatible alias for `obs_linear_effect` used in documentation.

    Parameters
    ----------
    n : int, default=10_000
    theta : float, default=1.0
    outcome_type : str, default="continuous"
    sigma_y : float, default=1.0
    target_d_rate : float, optional
    confounder_specs : list of dict, optional
    beta_y : array-like, optional
    beta_d : array-like, optional
    random_state : int, optional
    k : int, default=0
    x_sampler : callable, optional
    add_ancillary : bool, default=False
    deterministic_ids : bool, default=False

    Returns
    -------
    pandas.DataFrame
        Synthetic observational dataset.
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
