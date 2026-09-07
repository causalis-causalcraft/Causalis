"""Compare joint construction with independent single-outcome contracts.

Run from the repository root with:
    .venv/bin/python benchmarks/rct_causal_data.py --rows 100000 --outcomes 32

Reports construction time and retained DataFrame bytes (not peak process RSS).
Generation and imports are excluded from timing. Each mode runs separately so
its objects can be released before the next mode. Timings are observational,
not a fixed CI threshold. Increase --rows to exercise larger datasets.
"""
import argparse
import gc
from time import perf_counter

import numpy as np
import pandas as pd

from causalis.data_contracts import CausalData, MultiCausalData, RctCausalData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--outcomes", type=int, default=32)
    parser.add_argument("--confounders", type=int, default=8)
    parser.add_argument("--arms", type=int, default=1, help="1 for binary encoding; 2–15 for explicit one-hot arms")
    args = parser.parse_args()
    if args.rows < 2 or args.outcomes < 1 or args.confounders < 0:
        parser.error("require rows >= 2, outcomes >= 1, confounders >= 0")
    if not 1 <= args.arms <= min(15, args.rows):
        parser.error("require 1 <= arms <= min(15, rows) for the MultiCausalData comparison")
    rng = np.random.default_rng(42)
    outcomes = [f"y{i}" for i in range(args.outcomes)]
    confounders = [f"x{i}" for i in range(args.confounders)]
    names = outcomes + confounders
    df = pd.DataFrame({name: rng.normal(size=args.rows) for name in names})
    if args.arms == 1:
        df["d"] = np.arange(args.rows) % 2
        treatments = ["d"]
        options = {}
    else:
        treatments = [f"arm{i}" for i in range(args.arms)]
        assignment = np.arange(args.rows) % args.arms
        for i, name in enumerate(treatments):
            df[name] = (assignment == i).astype(np.int8)
        options = {"control_treatment": treatments[0]}

    for mode in ("joint", "separate"):
        gc.collect()
        start = perf_counter()
        if mode == "joint":
            objects = [RctCausalData.from_df(df, treatments, outcomes, confounders, **options)]
        elif args.arms > 1:
            objects = [
                MultiCausalData.from_df(
                    df, outcome=name, treatment_names=treatments,
                    confounders=confounders, **options,
                )
                for name in outcomes
            ]
        else:
            objects = [CausalData.from_df(df, "d", name, confounders) for name in outcomes]
        elapsed = perf_counter() - start
        size = sum(obj.df.memory_usage(index=True, deep=True).sum() for obj in objects)
        print(f"{mode}: {elapsed:.3f} s, retained data {size / 1024**2:.1f} MiB")
        del objects


if __name__ == "__main__":
    main()
