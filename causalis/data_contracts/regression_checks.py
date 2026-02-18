from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class RegressionChecks(BaseModel):
    """Lightweight OLS/regression health checks for CUPED diagnostics."""

    ate_naive: float
    ate_adj: float
    ate_gap: float
    ate_gap_over_se_naive: Optional[float] = None

    k: int
    rank: int
    full_rank: bool
    condition_number: float
    p_main_covariates: int
    near_duplicate_pairs: List[Tuple[str, str, float]] = Field(default_factory=list)
    vif: Optional[Dict[str, float]] = None

    resid_scale_mad: float
    n_std_resid_gt_3: int
    n_std_resid_gt_4: int
    max_abs_std_resid: float

    max_leverage: float
    leverage_cutoff: float
    n_high_leverage: int

    max_cooks: float
    cooks_cutoff: float
    n_high_cooks: int

    min_one_minus_h: float
    n_tiny_one_minus_h: int

    winsor_q: Optional[float] = None
    ate_adj_winsor: Optional[float] = None
    ate_adj_winsor_gap: Optional[float] = None
