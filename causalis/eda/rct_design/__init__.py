"""
Design module for experimental rct_design utilities.
"""

from causalis.eda.rct_design.mde import calculate_mde
from causalis.eda.rct_design.split import assign_variants_df
from causalis.eda.rct_design.srm import SRMResult, check_srm

__all__ = ["calculate_mde", "assign_variants_df", "SRMResult", "check_srm"]