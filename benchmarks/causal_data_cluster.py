"""Compare CausalData initialization with optional cluster identifier dtypes.

Run from the repository root with:
    .venv/bin/python benchmarks/causal_data_cluster.py --rows 100000 --groups 1000

Reports median initialization time and retained DataFrame MiB, not peak RSS.
Imports, data generation, garbage collection, and memory measurement are excluded
from timing. All modes use the same numeric data. Timings are observational only;
there are no CI thresholds. Increase --rows to exercise larger datasets.
"""

import argparse
import gc
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

from causalis.data_contracts import CausalData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--confounders", type=int, default=8)
    parser.add_argument("--groups", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.rows < 2 or args.confounders < 0 or args.repeats < 1:
        parser.error("require rows >= 2, confounders >= 0, repeats >= 1")
    if not 2 <= args.groups <= args.rows:
        parser.error("require 2 <= groups <= rows")

    rng = np.random.default_rng(42)
    confounders = [f"x{i}" for i in range(args.confounders)]
    df = pd.DataFrame({name: rng.normal(size=args.rows) for name in ["y"] + confounders})
    df["d"] = (np.arange(args.rows) % 2).astype(np.int8)
    labels = np.arange(args.rows) % args.groups
    names = np.array([f"g{i}" for i in range(args.groups)], dtype=object)
    df["cluster_int"] = labels
    df["cluster_string"] = names[labels]
    df["cluster_categorical"] = pd.Categorical.from_codes(labels, categories=names)

    for mode, column in (
        ("without cluster", None),
        ("integer", "cluster_int"),
        ("string", "cluster_string"),
        ("categorical", "cluster_categorical"),
    ):
        elapsed = []
        for _ in range(args.repeats):
            gc.collect()
            start = perf_counter()
            data = CausalData.from_df(df, "d", "y", confounders, cluster=column)
            elapsed.append(perf_counter() - start)
            retained_mib = data.df.memory_usage(index=True, deep=True).sum() / 1024**2
            del data
        print(f"{mode}: median {median(elapsed):.4f} s, retained data {retained_mib:.2f} MiB")


if __name__ == "__main__":
    main()
