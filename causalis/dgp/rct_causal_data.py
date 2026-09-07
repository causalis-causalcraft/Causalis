"""Synthetic randomized experiments for RctCausalData and CUPED benchmarks."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from causalis.data_contracts.rct_causal_data import RctCausalData
from .multicausaldata.base import MultiCausalDatasetGenerator


def generate_rct_causal_data(
    n: int = 10_000,
    *,
    n_treatments: int = 3,
    n_outcomes: int = 3,
    outcome_specs: list[dict[str, Any]] | None = None,
    k: int = 2,
    confounder_specs: list[dict[str, Any]] | None = None,
    target_d_rate: list[float] | np.ndarray | None = None,
    treatment_encoding: Literal["one_hot", "binary"] = "one_hot",
    d_names: list[str] | None = None,
    seed: int | None = 42,
    include_user_id: bool = True,
    include_oracle: bool = False,
    return_causal_data: bool = True,
) -> RctCausalData | pd.DataFrame:
    """Generate multiple outcomes under a shared randomized assignment.

    Reuses ``MultiCausalDatasetGenerator``'s covariate sampling, effect
    normalization, outcome-family sampling, and natural-scale oracle methods.
    Covariates are generated before treatment and are suitable for CUPED.
    Assignment is independent of covariates and outcome noise. Separate RNG
    streams keep assignment/covariates unchanged when outcome settings change.
    Outcomes are conditionally independent given covariates and assignment.

    Parameters
    ----------
    n : int, default 10000
        Number of observations. Must be at least ``n_treatments``.
    n_treatments : int, default 3
        Number of arms including control, at least two.
    n_outcomes : int, default 3
        Number of default continuous outcomes, named y1, y2, ... . Ignored when
        ``outcome_specs`` is provided; its length determines the outcome count.
    outcome_specs : list of dict, optional
        One configuration per outcome. Supported keys (internal DGP meanings):
        ``name`` (default y1, y2, ...), ``outcome_type`` (continuous, binary,
        poisson, gamma; normal aliases continuous), ``alpha_y`` (default 0),
        ``beta_y`` (one coefficient per expanded covariate; defaults to
        0.5 / sqrt(number of covariates)), ``theta`` (default 0.5),
        ``sigma_y`` (default 1), and ``gamma_shape`` (default 2).
        ``theta`` is scalar for all active arms, a length K-1 active-arm vector,
        or a length K vector including control. Effects are additive on the
        identity/logit/log link, not necessarily on the natural outcome scale.
        Covariate distributions use the internal ``confounder_specs`` schema.
    k : int, default 2
        Number of standard normal covariates when no specs are supplied.
    confounder_specs : list of dict, optional
        Internal DGP distribution specs (normal, uniform, bernoulli, gamma,
        categorical, etc.). Categorical covariates expand into dummy columns.
    target_d_rate : array-like, optional
        Strictly positive allocation weights of length K, normalized to sum 1.
        Defaults to equal probabilities. Assignments are IID, not forced to
        exact counts. A missing arm raises an error; increase n or its weight.
    treatment_encoding : {"one_hot", "binary"}, default "one_hot"
        Full one-hot encoding with first column as control, or a single 0/1
        column for a two-arm experiment. Binary requires n_treatments=2.
    d_names : list of str, optional
        Treatment columns, default d_0, d_1, ... (one-hot), or ["d"] (binary).
    seed : int or None, default 42
        Seed for reproducible independent random streams.
    include_user_id : bool, default True
        Include a unique int64 ``user_id`` column.
    include_oracle : bool, default False
        Add m_<arm>, g_<outcome>_<arm>, and cate_<outcome>_<active_arm>
        columns to DataFrame output. g is the conditional potential-outcome
        mean on the natural scale; cate is its difference from control.
        Binary oracle arm labels are 0 and 1. Contract output excludes oracle
        columns, as required by RctCausalData's selected-column policy.
    return_causal_data : bool, default True
        Return RctCausalData; False returns a DataFrame. Both retain small
        ``df.attrs["rct"]`` metadata with allocation probabilities, arm labels,
        outcome families, and link-scale effects relative to control.

    Notes
    -----
    Samples that have constant or identical columns can fail contract validation,
    especially tiny samples with discrete outcomes. No outcomes are redrawn to
    force validation. DataFrame output can be used to inspect such samples.
    Generation uses column arrays and one assignment vector; it does not create
    a rows-by-outcomes-by-arms potential-outcome tensor. Oracle columns are
    optional because they can be much larger than the observed dataset.

    Examples
    --------
    >>> data = generate_rct_causal_data(n=1000, seed=7)
    >>> data.Y.shape, data.D.shape
    ((1000, 3), (1000, 3))
    >>> from causalis.scenarios.cuped import CUPEDModel
    >>> model = CUPEDModel().fit(data, covariates=["x1", "x2"], run_checks=False)
    >>> estimates = model.estimate()
    >>> estimates["y1"]["d_1"].treatment
    'd_1'
    """
    for name, value, minimum in [("n", n, 2), ("n_treatments", n_treatments, 2),
                                  ("k", k, 0), ("n_outcomes", n_outcomes, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if n < n_treatments:
        raise ValueError("n must be >= n_treatments.")
    if treatment_encoding not in {"one_hot", "binary"}:
        raise ValueError("treatment_encoding must be 'one_hot' or 'binary'.")
    if treatment_encoding == "binary" and n_treatments != 2:
        raise ValueError("Binary encoding requires n_treatments=2.")
    expected_d = n_treatments if treatment_encoding == "one_hot" else 1
    treatment_names = d_names if d_names is not None else (
        [f"d_{i}" for i in range(n_treatments)] if expected_d > 1 else ["d"]
    )
    _validate_names(treatment_names, "d_names")
    if len(treatment_names) != expected_d:
        raise ValueError(f"d_names must have length {expected_d}.")
    arms = list(treatment_names) if expected_d > 1 else ["0", "1"]
    weights = np.ones(n_treatments) if target_d_rate is None else np.asarray(target_d_rate, dtype=float)
    if weights.shape != (n_treatments,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("target_d_rate must contain one finite, strictly positive weight per arm.")
    weights = weights / weights.max()  # avoid overflow when summing large weights
    probabilities = weights / weights.sum()
    if np.any(probabilities == 0):
        raise ValueError("target_d_rate weights are too disparate to represent positive probabilities.")

    if outcome_specs is None:
        specs = [{} for _ in range(n_outcomes)]
    elif not isinstance(outcome_specs, list) or not outcome_specs or any(not isinstance(s, dict) for s in outcome_specs):
        raise ValueError("outcome_specs must be a non-empty list of dictionaries.")
    else:
        specs = [dict(s) for s in outcome_specs]
    outcome_names = [s.get("name", f"y{i+1}") for i, s in enumerate(specs)]
    _validate_names(outcome_names, "outcome names")
    allowed = {"name", "outcome_type", "alpha_y", "beta_y", "theta", "sigma_y", "gamma_shape"}
    for spec in specs:
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"Unknown outcome spec keys: {sorted(unknown)}.")

    x_seed, assignment_seed, outcome_seed = np.random.SeedSequence(seed).spawn(3)
    sampler = MultiCausalDatasetGenerator(k=k, confounder_specs=confounder_specs, seed=x_seed)
    X, x_names = sampler._sample_X(n)
    X = np.asarray(X, dtype=float)
    if X.shape != (n, len(x_names)) or not np.isfinite(X).all():
        raise ValueError("Covariate sampler must produce a finite numeric matrix.")
    _validate_names(x_names, "confounder names", allow_empty=True)
    names = outcome_names + list(treatment_names) + x_names + (["user_id"] if include_user_id else [])
    if include_oracle:
        names += [f"m_{arm}" for arm in arms]
        names += [f"g_{name}_{arm}" for name in outcome_names for arm in arms]
        names += [f"cate_{name}_{arm}" for name in outcome_names for arm in arms[1:]]
    if len(set(names)) != len(names):
        raise ValueError("Outcome, treatment, covariate, user_id, and oracle column names must be disjoint.")

    assignment = np.random.default_rng(assignment_seed).choice(n_treatments, size=n, p=probabilities)
    if np.any(np.bincount(assignment, minlength=n_treatments) == 0):
        raise ValueError("Not all treatment arms were observed; increase n or the rare arm's allocation weight.")
    columns = {}
    if include_user_id:
        columns["user_id"] = np.arange(n, dtype=np.int64)
    if expected_d == 1:
        columns[treatment_names[0]] = assignment.astype(np.int8)
    else:
        for i, name in enumerate(treatment_names):
            columns[name] = (assignment == i).astype(np.int8)
    for i, name in enumerate(x_names):
        columns[name] = X[:, i]
    metadata = {"arms": arms, "probabilities": probabilities.tolist(), "outcomes": {}}
    default_beta = np.full(len(x_names), 0.5 / np.sqrt(max(1, len(x_names))))
    for name, spec, child_seed in zip(outcome_names, specs, outcome_seed.spawn(len(specs))):
        gen = MultiCausalDatasetGenerator(
            n_treatments=n_treatments, seed=child_seed, theta=spec.get("theta", 0.5),
            sigma_y=spec.get("sigma_y", 1.0), gamma_shape=spec.get("gamma_shape", 2.0),
        )
        family = gen._normalize_outcome_type(spec.get("outcome_type", "continuous"))
        gen._require_supported_outcome_type(family)
        theta = gen._normalize_theta(n_treatments)
        beta = np.asarray(spec.get("beta_y", default_beta), dtype=float)
        intercept = float(spec.get("alpha_y", 0.0))
        if beta.shape != (len(x_names),) or not np.isfinite(beta).all():
            raise ValueError("beta_y must contain one finite coefficient per expanded covariate.")
        if not np.isfinite(theta).all() or not np.isfinite(intercept):
            raise ValueError("theta and alpha_y must be finite.")
        if not np.isfinite(gen.sigma_y) or gen.sigma_y < 0:
            raise ValueError("sigma_y must be finite and non-negative.")
        if not np.isfinite(gen.gamma_shape) or gen.gamma_shape <= 0:
            raise ValueError("gamma_shape must be finite and positive.")
        baseline = X @ beta + intercept
        link = baseline + theta[assignment]
        if not np.isfinite(link).all():
            raise ValueError("Outcome link overflowed; reduce the covariate/effect magnitudes.")
        columns[name] = gen._sample_outcome_from_link(link, family)
        metadata["outcomes"][name] = {
            "outcome_type": family, "effects_link": (theta - theta[0]).tolist(),
        }
        if include_oracle:
            control_mean = gen._natural_scale_from_link(baseline + theta[0], family)
            columns[f"g_{name}_{arms[0]}"] = control_mean
            for i, arm in enumerate(arms[1:], start=1):
                mean = gen._natural_scale_from_link(baseline + theta[i], family)
                columns[f"g_{name}_{arm}"] = mean
                columns[f"cate_{name}_{arm}"] = mean - control_mean
    if include_oracle:
        for arm, probability in zip(arms, probabilities):
            columns[f"m_{arm}"] = np.full(n, probability)
    frame = pd.DataFrame(columns, copy=False)
    if return_causal_data:
        result = RctCausalData.from_df(
            frame, treatment_names=treatment_names, outcomes=outcome_names,
            confounders=x_names, user_id="user_id" if include_user_id else None,
            control_treatment=treatment_names[0] if expected_d > 1 else None,
        )
        result.df.attrs["rct"] = metadata
        return result
    frame.attrs["rct"] = metadata
    return frame


def _validate_names(names: Any, label: str, *, allow_empty: bool = False) -> None:
    if (not isinstance(names, list) or (not names and not allow_empty)
            or any(not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError(f"{label} must be a list of non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicate names.")
