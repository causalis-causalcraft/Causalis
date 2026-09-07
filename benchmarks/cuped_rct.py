"""Compare batch RCT CUPED with separate binary fits (regression checks off).

Run from the repository root with .venv/bin/python benchmarks/cuped_rct.py.
Dataset generation is excluded; independent contract conversion is included.
The batch path reuses designs and decompositions while each fit still computes
its own residuals, robust covariance and inference. Times are observational.
"""
import argparse
from time import perf_counter

import numpy as np
import pandas as pd

from causalis.data_contracts import CausalData, RctCausalData
from causalis.scenarios.cuped import CUPEDModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--outcomes", type=int, default=8)
    args = parser.parse_args()
    if args.rows < 30 or args.outcomes < 2:
        parser.error("require rows >= 30 and outcomes >= 2")
    rng = np.random.default_rng(42)
    assignment = np.arange(args.rows) % 3
    x = rng.normal(size=args.rows)
    names = [f"y{i}" for i in range(args.outcomes)]
    df = pd.DataFrame({name: 10 + i + assignment + x + rng.normal(size=args.rows)
                       for i, name in enumerate(names)})
    df["x"] = x
    for i, arm in enumerate(["control", "a", "b"]):
        df[arm] = (assignment == i).astype(np.int8)
    data = RctCausalData.from_df(df, ["control", "a", "b"], names, "x", control_treatment="control")
    start = perf_counter()
    model = CUPEDModel().fit(data, covariates=["x"], run_checks=False)
    batch = model.estimate(diagnostic_data=False)
    print(f"batch: {perf_counter() - start:.3f} s", flush=True)
    del model
    start = perf_counter()
    for arm in ["a", "b"]:
        pair = df.loc[(df.control == 1) | (df[arm] == 1)]
        for name in names:
            single = CausalData.from_df(pair, arm, name, "x")
            estimate = CUPEDModel().fit(single, covariates=["x"], run_checks=False).estimate(diagnostic_data=False)
            np.testing.assert_allclose(estimate.value, batch[name][arm].value, rtol=1e-10, atol=1e-10)
    print(f"separate: {perf_counter() - start:.3f} s (effects verified equal)")


if __name__ == "__main__":
    main()
