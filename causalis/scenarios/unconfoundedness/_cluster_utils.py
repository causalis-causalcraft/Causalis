"""Small, model-independent reductions for one-way clustered diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compact_cluster_codes(codes: Any, n: int) -> np.ndarray:
    """Validate integer membership and compact it in first-observed order.

    Compact before bincount: externally supplied codes need not start at zero,
    and a sparse large integer must never determine an allocation's size.
    """
    values = np.asarray(codes)
    if values.ndim != 1 or values.size != n:
        raise ValueError("cluster_codes must be one-dimensional and match the estimation sample.")
    if values.dtype.kind not in "iu" or values.dtype.kind == "b" or np.any(values < 0):
        raise ValueError("cluster_codes must contain nonnegative integers.")
    # Membership codes are typically already dense. A row-count-sized hash
    # table wastes tens of MiB for a million rows in only ten thousand groups.
    hint = min(n, int(values.max()) + 1) if n else 0
    dense = pd.factorize(values, sort=False, size_hint=hint)[0]
    dense.setflags(write=False)
    return dense


def inference_metadata(codes: np.ndarray | None) -> dict[str, Any]:
    """Describe the inference convention without claiming few-cluster validity."""
    if codes is None:
        return dict(cov_type="iid", n_clusters=None, inference_distribution="normal",
                    inference_df=None, covariance_correction=None,
                    inferential_available=True, inference_reason=None)
    count = int(np.max(codes)) + 1 if len(codes) else 0
    available = count >= 2
    return dict(cov_type="cluster", n_clusters=count, inference_distribution="t",
                inference_df=count - 1 if available else None,
                covariance_correction=count / (count - 1) if available else None,
                inferential_available=available,
                inference_reason=None if available else "Cluster inference requires at least two retained clusters.")


def mean_covariance(*influence_vectors: np.ndarray,
                    cluster_codes: np.ndarray | None = None) -> np.ndarray:
    """Covariance of row means using IID or centered one-way CR1 influences.

    Reduce one vector at a time; never construct a row-by-cluster indicator or
    stack the full influence vectors. Inputs are centered defensively.
    """
    if not influence_vectors:
        raise ValueError("At least one influence vector is required.")
    vectors = [np.asarray(value, dtype=float) for value in influence_vectors]
    n = vectors[0].size
    if any(value.ndim != 1 or value.size != n or not np.isfinite(value).all()
           for value in vectors):
        raise ValueError("Influence vectors must be finite, one-dimensional and equally sized.")
    dimension = len(vectors)
    if n < 2:
        return np.full((dimension, dimension), np.nan)
    if cluster_codes is None:
        # Pairwise dot products avoid the large temporary made by np.cov.
        means = [float(np.mean(value)) for value in vectors]
        covariance = np.empty((dimension, dimension))
        for j, value in enumerate(vectors):
            centered = value - means[j]
            for k in range(j + 1):
                product = float(np.dot(centered, vectors[k] - means[k]))
                covariance[j, k] = covariance[k, j] = product / (n * (n - 1))
        return covariance
    codes = compact_cluster_codes(cluster_codes, n)
    count = int(codes.max()) + 1
    if count < 2:
        return np.full((dimension, dimension), np.nan)
    sums = [np.bincount(codes, weights=value - float(np.mean(value)), minlength=count)
            for value in vectors]
    covariance = np.empty((dimension, dimension))
    scale = count / (count - 1) / n**2
    for j in range(dimension):
        for k in range(j + 1):
            covariance[j, k] = covariance[k, j] = scale * float(np.dot(sums[j], sums[k]))
    return covariance
