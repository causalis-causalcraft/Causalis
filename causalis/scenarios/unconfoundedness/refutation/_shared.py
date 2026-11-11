"""Private helpers shared across unconfoundedness refutation modules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalis.data_contracts.causal_estimate import CausalEstimate
from causalis.dgp.causaldata import CausalData
from causalis.scenarios.unconfoundedness._score_utils import _normalize_ate_atte_score
from causalis.scenarios.unconfoundedness._cluster_utils import (
    compact_cluster_codes,
    inference_metadata,
)


def _normalize_score(score: Any) -> str:
    """Normalize supported score aliases to ``ATE`` or ``ATTE``."""
    return _normalize_ate_atte_score(score)


def _numeric_digest(frame: pd.DataFrame, columns: Sequence[str], dtype: Any) -> str:
    """Hash numeric rows with bounded temporaries, in the estimator's byte order."""
    digest = hashlib.blake2b(digest_size=16)
    for start in range(0, len(frame), 65_536):
        block = frame.iloc[start:start + 65_536][list(columns)].to_numpy(dtype=dtype)
        digest.update(np.ascontiguousarray(block).view(np.uint8))
    return digest.hexdigest()


def _index_digest(index: pd.Index) -> str:
    hashed = pd.util.hash_pandas_object(index, index=False).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.blake2b(np.ascontiguousarray(hashed).view(np.uint8), digest_size=16).hexdigest()


def _estimation_positions(diagnostic_data: Any, n: int) -> np.ndarray:
    mask = getattr(diagnostic_data, "overlap_mask", None)
    if mask is None:
        return np.arange(n, dtype=int)
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype.kind != "b" or int(np.count_nonzero(mask)) != n:
        raise ValueError("diagnostic_data.overlap_mask must map the original sample to the estimation rows.")
    return np.flatnonzero(mask)


def _aligned_data_array(data: CausalData, diagnostic_data: Any,
                        columns: str | Sequence[str]) -> np.ndarray:
    """Select original-data fallback columns into the saved estimation order."""
    mask = getattr(diagnostic_data, "overlap_mask", None)
    frame = data.df[columns if isinstance(columns, str) else list(columns)]
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim != 1 or mask.dtype.kind != "b" or mask.size != len(frame):
            raise ValueError("Pass the original fitted data in fitted row order for diagnostic fallbacks.")
        frame = frame.iloc[np.flatnonzero(mask)]
    return frame.to_numpy(dtype=float)


@dataclass(frozen=True)
class _ClusterContext:
    codes: np.ndarray | None
    row_positions: np.ndarray
    labels: np.ndarray | None
    cluster_name: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {**inference_metadata(self.codes), "cluster_name": self.cluster_name}

    def subset(self, mask: np.ndarray) -> "_ClusterContext":
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(self.row_positions):
            raise ValueError("Diagnostic finite mask must match the estimation sample.")
        if self.codes is None:
            return _ClusterContext(None, self.row_positions[mask], None, self.cluster_name)
        old = self.codes[mask]
        codes = compact_cluster_codes(old, len(old))
        first = np.flatnonzero(~pd.Series(old).duplicated().to_numpy())
        labels = self.labels[old[first]] if self.labels is not None else None
        return _ClusterContext(codes, self.row_positions[mask], labels, self.cluster_name)


def _resolve_cluster_context(data: CausalData | None, estimate: CausalEstimate,
                             n: int) -> _ClusterContext:
    """Resolve frozen grouping and verified labels without consulting a live model."""
    diagnostic = estimate.diagnostic_data
    if data is not None:
        _validate_estimate_matches_data(data, estimate)
    positions = _estimation_positions(diagnostic, n)
    raw = getattr(diagnostic, "cluster_codes", None)
    options = estimate.model_options
    name = getattr(diagnostic, "cluster_name", None) or options.get("cluster_name")
    clustered = raw is not None or name is not None or options.get("cov_type") == "cluster"
    labels = None
    if raw is not None:
        codes = compact_cluster_codes(raw, n)
        labels = np.arange(int(codes.max()) + 1 if n else 0)
    elif data is not None and data.cluster_name is not None:
        if options.get("cov_type") == "iid":
            raise ValueError("The estimate uses iid inference but supplied data declares clustering.")
        values = data.cluster.iloc[positions]
        if values.isna().any():
            raise ValueError("Cluster labels must not be missing.")
        raw_codes, unique = pd.factorize(values, sort=False)
        codes = compact_cluster_codes(raw_codes, n)
        labels = unique.to_numpy(dtype=object)
        name = data.cluster_name
        clustered = True
    elif clustered:
        raise ValueError("Cluster membership is missing from this estimate. Pass aligned clustered data or refit with diagnostics.")
    else:
        codes = None
    if codes is not None and options.get("cov_type") == "iid":
        raise ValueError("The estimate's iid covariance metadata conflicts with cluster membership.")
    if codes is not None and data is not None and data.cluster_name is not None:
        values = data.cluster.iloc[positions]
        external, unique = pd.factorize(values, sort=False)
        if np.any(external < 0) or not np.array_equal(compact_cluster_codes(external, n), codes):
            raise ValueError("Supplied data cluster partition must match the fitted estimation sample.")
        labels = unique.to_numpy(dtype=object)
    return _ClusterContext(codes, positions, labels, name)


def _validate_estimate_matches_data(
    data: CausalData,
    estimate: CausalEstimate,
    *,
    require_confounders: bool = False,
) -> None:
    """Ensure an estimate is aligned with the supplied causal dataset."""
    if str(estimate.treatment) != str(data.treatment_name):
        raise ValueError(
            "estimate.treatment must match data.treatment_name "
            f"({estimate.treatment!r} != {data.treatment_name!r})."
        )

    if str(estimate.outcome) != str(data.outcome_name):
        raise ValueError(
            "estimate.outcome must match data.outcome_name "
            f"({estimate.outcome!r} != {data.outcome_name!r})."
        )

    df = data.df
    missing_confounders = [name for name in estimate.confounders if name not in df.columns]
    if missing_confounders and require_confounders:
        raise ValueError(
            "estimate.confounders are missing in data.get_df(): "
            + ", ".join(sorted(map(str, missing_confounders)))
        )
    diagnostic = estimate.diagnostic_data
    fingerprint = getattr(diagnostic, "sample_fingerprint", None)
    if fingerprint is not None:
        if len(df) != fingerprint["n_obs"]:
            raise ValueError("Supplied data must match the original fitted sample and row order.")
        expected_columns = fingerprint.get("confounders", estimate.confounders)
        if any(column not in df for column in expected_columns):
            raise ValueError("Supplied data is missing fitted confounders.")
        actual = dict(index_hash=_index_digest(df.index),
                      x_hash=_numeric_digest(df, expected_columns, float),
                      y_hash=_numeric_digest(df, [data.outcome_name], float),
                      d_hash=_numeric_digest(df, [data.treatment_name], int))
        user_id_name = fingerprint.get("user_id_name")
        if user_id_name is not None:
            if data.user_id_name != user_id_name:
                raise ValueError("Supplied data user ID role must match the fitted sample.")
            actual["user_id_hash"] = _index_digest(pd.Index(data.user_id))
        if any(actual[key] != fingerprint[key] for key in actual if key in fingerprint):
            raise ValueError("Supplied data must match the fitted sample contents and row order.")
        if data.cluster_name is not None and "cluster_hash" in fingerprint:
            external, _ = pd.factorize(data.cluster, sort=False)
            digest = hashlib.blake2b(np.ascontiguousarray(external).view(np.uint8), digest_size=16).hexdigest()
            if np.any(external < 0) or digest != fingerprint["cluster_hash"]:
                raise ValueError("Supplied data cluster partition must match the fitted sample.")
    # Frozen codes describe retained rows; labels can change but membership cannot.
    raw_codes = getattr(diagnostic, "cluster_codes", None)
    declared_cluster = (raw_codes is not None or data.cluster_name is not None
                        or estimate.model_options.get("cov_type") == "cluster")
    if declared_cluster:
        saved_d = getattr(diagnostic, "d", None)
        n = len(raw_codes) if raw_codes is not None else len(saved_d) if saved_d is not None else 0
        if n == 0:
            raise ValueError("Clustered diagnostics need saved estimation rows; refit with diagnostics.")
        codes = compact_cluster_codes(raw_codes, n) if raw_codes is not None else None
        positions = _estimation_positions(diagnostic, n)
        overlap = getattr(diagnostic, "overlap_mask", None)
        input_n = len(overlap) if overlap is not None else n
        if len(df) != input_n:
            raise ValueError("Pass the original fitted sample in fitted row order.")
        if len(positions) and positions[-1] >= len(df):
            raise ValueError("Supplied data is shorter than the original fitted sample.")
        if fingerprint is None:
            for key, columns in (("y", data.outcome_name), ("d", data.treatment_name),
                                 ("x", list(estimate.confounders))):
                saved = getattr(diagnostic, key, None)
                if saved is None and raw_codes is None:
                    raise ValueError("Cannot verify legacy clustered sample alignment; refit with diagnostics.")
                if saved is not None:
                    saved = np.asarray(saved)
                    expected_shape = (n,) if isinstance(columns, str) else (n, len(columns))
                    if saved.shape != expected_shape:
                        raise ValueError("Supplied data must match the saved estimation rows and order.")
                    # Legacy snapshots have no digest. Bound comparison copies
                    # by row blocks rather than materializing an entire X matrix.
                    for start in range(0, n, 65_536):
                        stop = min(start + 65_536, n)
                        actual = df.iloc[positions[start:stop]][columns].to_numpy(dtype=float)
                        saved_block = np.asarray(saved[start:stop], dtype=float)
                        finite = np.isfinite(saved_block)
                        if not np.array_equal(actual[finite], saved_block[finite]):
                            raise ValueError("Supplied data must match the saved estimation rows and order.")
        if data.cluster_name is not None and codes is not None:
            external, _ = pd.factorize(data.cluster.iloc[positions], sort=False)
            if np.any(external < 0) or not np.array_equal(compact_cluster_codes(external, n), codes):
                raise ValueError("Supplied data cluster partition must match the fitted estimation sample.")
