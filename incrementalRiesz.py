#!/usr/bin/env python
"""
Incremental vs brute-force Riesz s-energy subset selection experiments.

- Reports CPU / OS / Python info.
- Runs both incremental and brute-force enumeration with a 10 s
  per-(n,k) time limit.
- Plots runtime vs n for each (method, k) and saves as PNG
  without a title and with integer x-axis ticks.
"""

import time
import math
import random
from math import comb
import sys
import platform
import multiprocessing
import os
import itertools

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Environment reporting
# ----------------------------------------------------------------------
def print_environment_info():
    print("=== Environment info ===")
    print(f"Python        : {sys.version.split()[0]}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"Machine       : {platform.machine()}")
    print(f"Logical cores : {multiprocessing.cpu_count()}")

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    if conda_env:
        print(f"Conda env     : {conda_env}")

    # Optional detailed CPU info (if py-cpuinfo is installed)
    try:
        import cpuinfo  # pip install py-cpuinfo

        info = cpuinfo.get_cpu_info()
        print(f"CPU           : {info.get('brand_raw', 'N/A')}")
    except Exception:
        print("CPU           : (install 'py-cpuinfo' for detailed model)")

    print(f"pandas        : {pd.__version__}")
    print(f"matplotlib    : {plt.matplotlib.__version__}")
    print("========================\n")


# ----------------------------------------------------------------------
# Riesz energy utilities
# ----------------------------------------------------------------------
def precompute_riesz_matrix(points, s=1.0):
    """Return pairwise Riesz contributions 1 / ||x_i - x_j||^s."""
    n = len(points)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = points[i]
        for j in range(i + 1, n):
            xj = points[j]
            d = math.dist(xi, xj)
            val = float("inf") if d == 0 else 1.0 / (d ** s)
            mat[i][j] = mat[j][i] = val
    return mat


def energy_of_subset(mat, subset):
    """Compute Riesz energy of a subset given the pairwise matrix."""
    e = 0.0
    m = len(subset)
    for a in range(m):
        i = subset[a]
        row_i = mat[i]
        for b in range(a + 1, m):
            j = subset[b]
            e += row_i[j]
    return e


# ----------------------------------------------------------------------
# Incremental enumeration
# ----------------------------------------------------------------------
def incremental_best_subset(mat, k, time_limit=10.0):
    """
    Incremental enumeration:
    maintain partial energy and only add contributions of the new point.
    """
    n = len(mat)
    best_E = float("inf")
    best_subset = None
    visited = 0
    start = time.perf_counter()
    timeout = False

    def backtrack(start_idx, depth, chosen, current_E):
        nonlocal best_E, best_subset, visited, timeout

        if time.perf_counter() - start > time_limit:
            timeout = True
            return

        if depth == k:
            visited += 1
            if current_E < best_E:
                best_E = current_E
                best_subset = tuple(chosen)
            return

        max_start = n - (k - depth) + 1
        for i in range(start_idx, max_start):
            inc = 0.0
            row_i = mat[i]
            for j in chosen:
                inc += row_i[j]

            chosen.append(i)
            backtrack(i + 1, depth + 1, chosen, current_E + inc)
            chosen.pop()

            if timeout:
                return

    backtrack(0, 0, [], 0.0)
    total = comb(n, k)
    elapsed = time.perf_counter() - start

    return {
        "method": "incremental",
        "n": n,
        "k": k,
        "best_energy": best_E,
        "best_subset": best_subset,
        "visited_subsets": visited,
        "total_subsets": total,
        "fraction_visited": visited / total if total > 0 else 1.0,
        "time_sec": elapsed,
        "timeout_10s": timeout,
    }


# ----------------------------------------------------------------------
# Brute-force enumeration (no incremental updates)
# ----------------------------------------------------------------------
def bruteforce_best_subset(mat, k, time_limit=10.0):
    """
    Brute-force enumeration WITHOUT incremental updates:
    for each k-subset, recompute energy from scratch.
    """
    n = len(mat)
    best_E = float("inf")
    best_subset = None
    visited = 0
    start = time.perf_counter()
    timeout = False

    total = comb(n, k)

    for subset in itertools.combinations(range(n), k):
        if time.perf_counter() - start > time_limit:
            timeout = True
            break

        e = energy_of_subset(mat, subset)
        visited += 1
        if e < best_E:
            best_E = e
            best_subset = subset

    elapsed = time.perf_counter() - start

    return {
        "method": "bruteforce",
        "n": n,
        "k": k,
        "best_energy": best_E,
        "best_subset": best_subset,
        "visited_subsets": visited,
        "total_subsets": total,
        "fraction_visited": visited / total if total > 0 else 1.0,
        "time_sec": elapsed,
        "timeout_10s": timeout,
    }


# ----------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------
def run_experiments(n_values, k_values, s=1.0, dim=2, time_limit=10.0, seed=0):
    random.seed(seed)
    rows = []

    for n in n_values:
        points = [tuple(random.random() for _ in range(dim)) for _ in range(n)]
        mat = precompute_riesz_matrix(points, s=s)

        for k in k_values:
            rows.append(incremental_best_subset(mat, k, time_limit=time_limit))
            rows.append(bruteforce_best_subset(mat, k, time_limit=time_limit))

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print_environment_info()

    # Experiment grid
    n_values = [20, 26, 32, 38]
    k_values = [3, 4, 5]
    s = 1.0
    dim = 2
    time_limit = 10.0  # seconds per (n, k)

    # Measure total wall-clock and CPU time
    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    results_df = run_experiments(
        n_values=n_values,
        k_values=k_values,
        s=s,
        dim=dim,
        time_limit=time_limit,
        seed=42,
    )

    t_wall = time.perf_counter() - t_wall_start
    t_cpu = time.process_time() - t_cpu_start

    print("=== Timing summary for all experiments ===")
    print(f"Total wall-clock time : {t_wall:.4f} seconds")
    print(f"Total CPU time        : {t_cpu:.4f} seconds")
    print("Per-instance timings:")
    print(results_df.to_string(index=False))
    print("==========================================\n")

    # --- Plot: time vs n, incremental vs brute-force ---
    plt.figure()
    markers = {"incremental": "o", "bruteforce": "s"}

    for method in sorted(results_df["method"].unique()):
        for k in sorted(results_df["k"].unique()):
            sub = results_df[(results_df["method"] == method) & (results_df["k"] == k)]
            if sub.empty:
                continue
            label = f"{method}, k={k}"
            plt.plot(
                sub["n"],
                sub["time_sec"],
                marker=markers.get(method, "o"),
                label=label,
            )

    plt.xlabel("n (number of points)")
    plt.ylabel("time (seconds)")
    # no title
    plt.legend()
    plt.tight_layout()

    # integer x-axis ticks
    xticks = sorted(results_df["n"].unique())
    plt.xticks(xticks)

    png_filename = "incremental_vs_bruteforce_riesz_runtime.png"
    plt.savefig(png_filename, dpi=300)
    print(f"Saved runtime plot to: {png_filename}")
