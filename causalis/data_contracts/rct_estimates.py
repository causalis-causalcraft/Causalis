"""Mapping of RCT outcome/arm estimates with a combined summary table."""

from __future__ import annotations

import pandas as pd

from .causal_estimate import CausalEstimate


class RctEstimates(dict[str, dict[str, CausalEstimate]]):
    """Batch estimates, indexed by outcome and then active treatment arm.

    Retains ordinary dictionary access, iteration, and items(). ``summary``
    combines the existing CausalEstimate summaries into a notebook-friendly
    table with one column per treatment-versus-control comparison. Values and
    confidence intervals use exactly the single-estimate formatting.
    """

    def summary(
        self, outcome: str | None = None, treatment: str | None = None,
    ) -> pd.DataFrame:
        """Summarize one outcome or all outcomes, optionally selecting an arm.

        Rows are the fields from CausalEstimate.summary(). When ``outcome`` is
        specified, columns are comparison labels, e.g. ``variant_a vs control``.
        Otherwise columns have two levels: outcome and comparison. Binary
        treatment comparisons use ``<treatment>=1 vs <treatment>=0``.
        Unknown selectors raise ValueError. The returned table is independent
        of the stored estimates; no model fitting or inference is repeated.
        """
        if outcome is not None and outcome not in self:
            raise ValueError(f"Unknown outcome: {outcome!r}.")
        names = list(self) if outcome is None else [outcome]
        columns = []
        keys = []
        for name in names:
            for arm, estimate in self[name].items():
                if treatment is not None and arm != treatment:
                    continue
                control = estimate.model_options.get("control_treatment")
                comparison = f"{arm} vs {control}" if control is not None else f"{arm}=1 vs {arm}=0"
                columns.append(estimate.summary()["value"])
                keys.append((name, comparison))
        if not columns:
            if treatment is not None:
                raise ValueError(f"Unknown treatment for the selected outcomes: {treatment!r}.")
            return pd.DataFrame(index=pd.Index([], name="field"))
        result = pd.concat(columns, axis=1)
        if outcome is None:
            result.columns = pd.MultiIndex.from_tuples(keys, names=["outcome", "comparison"])
        else:
            result.columns = pd.Index([key[1] for key in keys], name="comparison")
        return result
