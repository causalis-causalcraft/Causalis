"""
Split (assignment) utilities for randomized controlled experiments.

This module provides deterministic assignment of variants to entities based
on hashing a composite key (salt | layer_id | experiment_id | entity_id)
into the unit interval and mapping it to cumulative variant weights.

The implementation mirrors the reference notebook in docs/examples/rct_design.ipynb.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional

import pandas as pd


# --- Core helpers (kept internal) ---

def _validate_variants(variants: Dict[str, float]) -> None:
    """Validate variant weights without normalization.

    Rules:
    - Dictionary must be non-empty
    - No negative weights
    - Sum must be > 0 and <= 1.0
    """
    if not variants:
        raise ValueError("Variants dictionary cannot be empty")

    total = 0.0
    for name, weight in variants.items():
        if weight < 0:
            raise ValueError(f"Variant '{name}' has negative weight: {weight}")
        total += float(weight)

    if total <= 0:
        raise ValueError("Sum of variant weights must be > 0")

    if total > 1.0:
        raise ValueError(f"Sum of variant weights ({total}) exceeds 1.0")


def _hash_to_unit_interval(key: str) -> float:
    """Hash a key to a float in [0, 1)."""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    n = int(h[:15], 16)
    return n / (16 ** 15)


# --- DataFrame in -> DataFrame out ---

def assign_variants_df(
    df: pd.DataFrame,
    id_col: str,
    experiment_id: str,
    variants: Dict[str, float],
    *,
    salt: str = "global_ab_salt",
    layer_id: str = "default",
    variant_col: str = "variant",
) -> pd.DataFrame:
    """
    Deterministically assign variants for each row in df based on id_col.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with an identifier column.
    id_col : str
        Column name in df containing entity identifiers (user_id, session_id, etc.).
    experiment_id : str
        Unique identifier for the experiment (versioned for reruns).
    variants : Dict[str, float]
        Mapping from variant name to weight (coverage). Weights must be non-negative
        and their sum must be in (0, 1]. If the sum is < 1, the remaining mass
        corresponds to "not in experiment" and the assignment will be None.
    salt : str, default "global_ab_salt"
        Secret string to de-correlate from other hash uses and make assignments
        non-gameable.
    layer_id : str, default "default"
        Identifier for mutual exclusivity layer or surface. In this case work like
        another random
    variant_col : str, default "variant"
        Name of output column to store assigned variant labels.

    Returns
    -------
    pd.DataFrame
        A copy of df with an extra column `variant_col`.
        Entities outside experiment coverage will have None in the variant column.
    """
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in DataFrame")

    _validate_variants(variants)

    def _row_assign(entity_id: str) -> Optional[str]:
        key = f"{salt}|{layer_id}|{experiment_id}|{entity_id}"
        u = _hash_to_unit_interval(key)

        cumulative = 0.0
        for name, weight in sorted(variants.items()):
            if weight == 0:
                continue
            cumulative += float(weight)
            if u < cumulative:
                return name

        return None  # Not in experiment

    out = df.copy()
    out[variant_col] = df[id_col].astype(str).map(_row_assign)
    return out
