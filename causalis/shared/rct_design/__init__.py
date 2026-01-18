"""
Design module for experimental rct_design utilities.
"""

from causalis.shared.rct_design.mde import calculate_mde
from causalis.shared.rct_design.split import assign_variants_df
from causalis.scenarios.classic_rct import SRMResult, check_srm

__all__ = ["calculate_mde", "assign_variants_df", "SRMResult", "check_srm"]