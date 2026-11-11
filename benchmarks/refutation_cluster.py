"""Reproducible clustered refutation timings and memory, in isolated workers.

Run with the repo-local Python, e.g.:
    .venv/bin/python benchmarks/refutation_cluster.py --rows 100000 1000000

Each operation/size/layout gets a fresh subprocess. Timings exclude tracing;
allocation peaks are measured separately after warmup. Worker peak RSS includes
imports, DGP generation and fitting; pre-operation peak RSS is reported separately.
Cheap real cross-fitted learners isolate diagnostic costs from learner choice.
No machine-dependent performance thresholds are imposed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
from statistics import median
import subprocess
import sys
from time import perf_counter
import tracemalloc


OPERATIONS = ("overlap", "balance", "score", "sensitivity", "influence_plot", "residual_plot")


def _rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def _worker(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from causalis.dgp import generate_obs_clustered
    from causalis.scenarios.unconfoundedness import IRM
    from causalis.scenarios.unconfoundedness.refutation import (
        run_overlap_diagnostics, run_score_diagnostics, run_unconfoundedness_diagnostics,
        sensitivity_analysis, plot_influence_instability, plot_residual_diagnostics,
    )

    n = args.worker_rows
    groups = max(30, n // 100)
    sizes = None
    if args.layout == "unequal":
        rng = np.random.default_rng(42)
        probabilities = rng.lognormal(sigma=1.0, size=groups)
        sizes = 1 + rng.multinomial(n - groups, probabilities / probabilities.sum())
    start = perf_counter()
    data = generate_obs_clustered(n=n, n_clusters=groups, cluster_sizes=sizes, seed=19)
    generation_time = perf_counter() - start
    model = IRM(data, ml_g=DummyRegressor(), ml_m=DummyClassifier(strategy="prior"),
                n_folds=3, random_state=19)
    start = perf_counter()
    model.fit()
    fit_time = perf_counter() - start
    start = perf_counter()
    estimate = model.estimate(score=args.score)
    estimate_time = perf_counter() - start
    calls = dict(
        overlap=lambda: run_overlap_diagnostics(data, estimate),
        balance=lambda: run_unconfoundedness_diagnostics(data, estimate),
        score=lambda: run_score_diagnostics(data, estimate),
        sensitivity=lambda: sensitivity_analysis(estimate, r2_y=0.03, r2_d=0.03),
        influence_plot=lambda: plot_influence_instability(estimate, data=data),
        residual_plot=lambda: plot_residual_diagnostics(estimate, data=data),
    )
    operation = calls[args.operation]

    def invoke():
        result = operation()
        if args.operation.endswith("plot"):
            result.canvas.draw()  # Include renderer work, not only figure setup.
            plt.close(result)
        del result

    invoke()
    elapsed = []
    for _ in range(args.repeats):
        gc.collect()
        start = perf_counter()
        invoke()
        elapsed.append(perf_counter() - start)
    gc.collect()
    baseline_rss = _rss_mib()
    tracemalloc.start()
    invoke()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(json.dumps(dict(rows=n, clusters=groups, layout=args.layout, score=args.score,
                          operation=args.operation, repeats=args.repeats,
                          median_seconds=median(elapsed), peak_allocated_mib=peak / 1024**2,
                          pre_operation_peak_rss_mib=baseline_rss, worker_peak_rss_mib=_rss_mib(),
                          generation_seconds=generation_time, fit_seconds=fit_time,
                          estimate_seconds=estimate_time)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[100_000, 1_000_000])
    parser.add_argument("--layouts", nargs="+", choices=["balanced", "unequal"],
                        default=["balanced", "unequal"])
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--score", choices=["ATE", "ATTE"], default="ATE")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-rows", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--layout", choices=["balanced", "unequal"], help=argparse.SUPPRESS)
    parser.add_argument("--operation", choices=OPERATIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 3000 for n in args.rows):
        parser.error("require repeats >= 1 and rows >= 3000")
    if args.worker_rows is not None:
        _worker(args)
        return
    environment = os.environ.copy()
    import tempfile
    plot_cache = tempfile.TemporaryDirectory(prefix="ckit-refutation-mpl-")
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=plot_cache.name)
    for n in args.rows:
        for layout in args.layouts:
            for operation in args.operations:
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--worker-rows", str(n), "--layout", layout,
                           "--operation", operation, "--repeats", str(args.repeats),
                           "--score", args.score]
                result = subprocess.run(command, env=environment, capture_output=True, text=True)
                if result.returncode:
                    sys.stderr.write(result.stderr)
                    raise SystemExit(result.returncode)
                print(result.stdout.strip(), flush=True)
    plot_cache.cleanup()


if __name__ == "__main__":
    main()
